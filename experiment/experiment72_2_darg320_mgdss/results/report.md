# Experiment 72.2 — DARG-320 Disease Semantic Supervision

The hypothesis is that supervising DARG fused F before CAM may deliver a disease-class gradient earlier to relation-conditioned propagation than supervising the CAM144 endpoint. It is a position-mechanism hypothesis, with no assumed improvement; fine heads require a simultaneous dimension/schema adaptation, so this is not a pure single-variable position ablation.

## Status and scope

Status: **COMPLETE_AUDITED**; formal tasks 60/60; independent checkpoint reads 60/60; failures 0.
This is a mechanism diagnostic under the inherited E69.3-G2 protocol, not a leakage-free generalization estimate. Fold 2 adds labeled rows 377 (ASD) and 407 (TD) to effective training and alpha calibration while retaining them in the test set. Checkpoint selection uses test ACC; preprocessing precedes outer CV; the graph is transductive over all 871 patients.
Initial model, all three optimizer states, E61 epoch-10 warmup, fold split, effective train set, relation-pair state and E69.3 gate initialization parity: 10/10 folds. Common E69.3 RNG checkpoints matched 21209/21209 recorded epochs; exact RNG status is reported in `rng_parity.csv`.
The experiment executes the locked E61-E2 `forward_e2_state` / `e2e_adaptive_forward`, preserving original GCN plus CADR messages and learnable edge/adaptive gate logits. E69.3 fold-addition code regenerates fold-2 train labels, weights and patient pairs; rows 377/407 remain in the test set. No 53.3 fallback forward is used.

## Source and method audit

SPromptGL MICCAI 2025 GFR/CAM/PLN and four-modality classifier; BrainPrompt MICCAI 2025 ABIDE disease descriptions and normalized-cosine CE; and SABER ICME 2026 equations (11)–(13) were checked against the locked E72.1 code review and local source copies. This uses only the disease-level class supervision equation, not full BrainPrompt or SABER architectures. Shared frozen CLIP RN50 text encoding differs from BrainPrompt's original Llama encoder; τ=0.1 is this study's SABER-style adaptation, not asserted as SABER's paper temperature.
E72.2 differs from E72.1 only in the semantic source location and required input projection schema: E72.1 supervised the 144-dimensional CAM flatten and its four 36-dimensional CAM positions; E72.2 supervises E61-E2 DARG fused F [871,320], globally with Linear(320,10), and in four F-coordinate segments of widths [48,6,10,256] with separate Linear(dim,10) heads. The F segments already include cross-modality propagation and are not independent raw modality evidence. The original CAM and 144-dimensional OutputLayer path, labels, base losses, optimizers, warmup, gates, training schedule, Prompt banks and six group definitions are retained.
The original classifier still receives the same CAM output and supplies all inference predictions. Each training pass calls E2E DARG and CAM once. Auxiliary projections are outside the base optimizers, initialized under private RNG, and omitted from inference. The joint loss remains `0.5 L_global + 0.5 L_fine`; fine weights are `.10/.10/.10/.70`; global-only uses `L_global`; total loss is `L_base + alpha L_sem`.
Frozen RN50 cache SHA `c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`; shared SVD basis SHA `3e79ee238447ff47ad1029408b99db94c3b4c4fbc657aa5a5251754563dc3868`; BrainPrompt bank SHA `f7105b19f5a81fb078c51636136b1e18d56fa1b204bb4a5d82a488b6f48b8176`; SPromptGL bank SHA `ee741144039d60c5de8bb36576476bdc9fa67ed2ff3f7cb97a13bba08c78a55a`. Prototype cosine: BrainPrompt-2 0.9847826; SPromptGL-8 0.9070364; maximum projection cosine error 1.110e-16. The τ=1 BrainPrompt CE and SABER-form log-softmax CE had loss/gradient equivalence `PASS`.
Alpha was calibrated once per fold/group using F-level `0.10*||grad_F L_cls||/||grad_F L_sem||` and frozen. Preflight initial weighted ratios span 0.100000–0.100000; alpha spans 0.010355–0.753085. Fold-2 alpha calibration includes both exposed labels. The normalized BrainPrompt prototype cosine also limits τ=1 logit gap; see `preflight.json` for geometry-derived bounds.

## Results

All values are ten-fold mean ± population SD (%). E69.3-G2 is read-only historical reference. E72.2 values retain the fold-2 overlap and are diagnostic only.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E69.3-G2 | 92.0742 +/- 2.7482 | 91.8995 +/- 2.7186 | 93.3759 +/- 2.9908 | 89.5854 +/- 3.4710 | 94.2137 +/- 3.9815 | 91.8995 +/- 2.7186 | 91.2984 +/- 2.9222 |
| G1 | 91.8443 +/- 2.7963 | 91.6279 +/- 2.8080 | 93.7126 +/- 2.7331 | 88.8293 +/- 4.0698 | 94.4265 +/- 3.7676 | 91.6279 +/- 2.8080 | 90.9778 +/- 3.0409 |
| G2 | 91.8443 +/- 2.7486 | 91.6363 +/- 2.7353 | 93.6234 +/- 2.7964 | 88.8415 +/- 3.3605 | 94.4311 +/- 3.4953 | 91.6363 +/- 2.7353 | 90.9911 +/- 2.9676 |
| G3 | 91.7294 +/- 2.8130 | 91.4740 +/- 2.8405 | 93.5581 +/- 2.7797 | 88.0915 +/- 4.2858 | 94.8566 +/- 3.6268 | 91.4740 +/- 2.8405 | 90.7845 +/- 3.1154 |
| G4 | 92.0742 +/- 2.6997 | 91.9625 +/- 2.6387 | 92.7122 +/- 2.9101 | 90.5671 +/- 2.9215 | 93.3580 +/- 4.2605 | 91.9625 +/- 2.6387 | 91.3946 +/- 2.8144 |
| G5 | 92.0742 +/- 2.6997 | 91.9549 +/- 2.6200 | 93.9113 +/- 3.0327 | 90.5610 +/- 3.5186 | 93.3488 +/- 4.9804 | 91.9549 +/- 2.6200 | 91.4017 +/- 2.7759 |
| G6 | 91.7307 +/- 2.6174 | 91.5773 +/- 2.6101 | 93.9196 +/- 2.7875 | 89.5793 +/- 3.0937 | 93.5754 +/- 3.0556 | 91.5773 +/- 2.6101 | 90.9415 +/- 2.8219 |

Paired deltas are left minus right, in percentage points; W/T/L are strict ACC wins/ties/losses.

| Comparison (left − right) | ΔACC pp | W/T/L | ΔHard-AUC | ΔProb-AUC | ΔSEN | ΔSPE | 9 non-overlap folds: mean ΔACC / W-T-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1_vs_E69.3-G2 | -0.2299 | 0/8/2 | -0.2717 | +0.3367 | -0.7561 | +0.2128 | -0.2554 / 0-7-2 |
| G2_vs_E69.3-G2 | -0.2299 | 0/8/2 | -0.2633 | +0.2474 | -0.7439 | +0.2174 | -0.2554 / 0-7-2 |
| G3_vs_E69.3-G2 | -0.3448 | 0/7/3 | -0.4255 | +0.1822 | -1.4939 | +0.6429 | -0.3831 / 0-6-3 |
| G4_vs_E69.3-G2 | +0.0000 | 1/7/2 | +0.0630 | -0.6637 | +0.9817 | -0.8557 | -0.2554 / 0-7-2 |
| G5_vs_E69.3-G2 | +0.0000 | 1/8/1 | +0.0553 | +0.5354 | +0.9756 | -0.8649 | -0.1277 / 0-8-1 |
| G6_vs_E69.3-G2 | -0.3435 | 0/7/3 | -0.3222 | +0.5437 | -0.0061 | -0.6383 | -0.3817 / 0-6-3 |
| G1_vs_E72.1-G1 | +0.2299 | 3/5/2 | +0.2829 | +0.1330 | +1.2134 | -0.6475 | +0.0000 / 2-5-2 |
| G2_vs_E72.1-G2 | -0.0013 | 2/6/2 | -0.0186 | -0.0787 | -0.2500 | +0.2128 | -0.0015 / 2-5-2 |
| G3_vs_E72.1-G3 | -0.2299 | 1/7/2 | -0.3512 | -0.0031 | -2.2195 | +1.5171 | -0.3831 / 0-7-2 |
| G4_vs_E72.1-G4 | +0.1149 | 3/5/2 | +0.2818 | -1.2014 | +2.4878 | -1.9241 | -0.0000 / 2-5-2 |
| G5_vs_E72.1-G5 | +0.2299 | 4/4/2 | +0.2974 | +0.0513 | +1.2378 | -0.6429 | +0.1277 / 3-4-2 |
| G6_vs_E72.1-G6 | -0.3448 | 0/8/2 | -0.3859 | +0.4858 | -0.9939 | +0.2220 | -0.2554 / 0-8-1 |
| G1_vs_G3 | +0.1149 | 1/9/0 | +0.1538 | +0.1545 | +0.7378 | -0.4302 | +0.1277 / 1-8-0 |
| G2_vs_G4 | -0.2299 | 1/7/2 | -0.3263 | +0.9112 | -1.7256 | +1.0731 | +0.0000 / 1-7-1 |
| G1_vs_G2 | +0.0000 | 1/8/1 | -0.0084 | +0.0893 | -0.0122 | -0.0046 | +0.0000 / 1-7-1 |
| G3_vs_G4 | -0.3448 | 1/6/3 | -0.4885 | +0.8459 | -2.4756 | +1.4986 | -0.1277 / 1-6-2 |
| G5_vs_G6 | +0.3435 | 3/7/0 | +0.3775 | -0.0083 | +0.9817 | -0.2266 | +0.2540 / 2-7-0 |
| G1_vs_G5 | -0.2299 | 0/8/2 | -0.3270 | -0.1987 | -1.7317 | +1.0777 | -0.1277 / 0-8-1 |
| G2_vs_G6 | +0.1136 | 3/5/2 | +0.0589 | -0.2963 | -0.7378 | +0.8557 | +0.1263 / 3-4-2 |

### Comparison with E72.1

E72.1 and E72.2 have different supervised representations and projection input dimensions, so these cross-experiment differences are descriptive; the fine-grain differences are not a strict one-variable ablation. Global-only groups are the closest position comparison, though their projection input dimensions necessarily differ.

| Group | E72.1 ACC | E72.2 ACC | ΔACC pp | E72.1 Prob-AUC | E72.2 Prob-AUC | ΔProb-AUC pp |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 91.6144 | 91.8443 | +0.2299 | 93.5796 | 93.7126 | +0.1330 |
| G2 | 91.8456 | 91.8443 | -0.0013 | 93.7021 | 93.6234 | -0.0787 |
| G3 | 91.9592 | 91.7294 | -0.2299 | 93.5612 | 93.5581 | -0.0031 |
| G4 | 91.9592 | 92.0742 | +0.1149 | 93.9136 | 92.7122 | -1.2014 |
| G5 | 91.8443 | 92.0742 | +0.2299 | 93.8600 | 93.9113 | +0.0513 |
| G6 | 92.0755 | 91.7307 | -0.3448 | 93.4338 | 93.9196 | +0.4858 |

### Fold-level metrics

`fold_metrics.csv` provides every fold's ACC, Hard-AUC, Prob-AUC, SEN, SPE, confusion counts, best epoch, actual epoch count, overlap flags and checkpoint SHA. `paired_comparisons.csv` contains ten-fold deltas and separate nine-fold summaries excluding fold 2; exclusion does not undo the training or checkpoint-selection label exposure.

## Fold 2 label exposure

Rows 377 and 407 remain both training and test patients. E69.3 effective class weights: `[1.0826445817947388, 0.9290780425071716]`; regenerated relation-pair SHA `c5ceb5c322490afe4c411751db8579ce4c960d5d7062d3c5c12640f56fbaf43c`, 4096 pairs, same/different distribution `{'1': 2048, '0': 2048}`. Labels are `{'377': 0, '407': 1}` in ASD=0, TD=1 coding.
The following table gives each E72.2 group's selected-checkpoint predictions and global semantic margin change from epoch-10 initialization. The baseline row is recorded in `fold2_exposure.csv`.

| Model | Row | True class | Prediction | P(ASD) | P(TD) | True-class logit margin | Semantic margin change |
|---|---:|---|---|---:|---:|---:|---:|
| E69.3-G2 | 377 | ASD | ASD | 0.5771 | 0.4229 | +0.3107 | N/A |
| E69.3-G2 | 407 | TD | TD | 0.2274 | 0.7726 | +1.2228 | N/A |
| G1 | 377 | ASD | ASD | 0.5116 | 0.4884 | +0.0463 | -0.0605 |
| G1 | 407 | TD | TD | 0.3829 | 0.6171 | +0.4771 | -0.0336 |
| G2 | 377 | ASD | ASD | 0.5542 | 0.4458 | +0.2175 | +0.1082 |
| G2 | 407 | TD | TD | 0.2764 | 0.7236 | +0.9624 | -0.0322 |
| G3 | 377 | ASD | ASD | 0.7369 | 0.2631 | +1.0299 | +0.0711 |
| G3 | 407 | TD | TD | 0.3155 | 0.6845 | +0.7747 | -0.2721 |
| G4 | 377 | ASD | ASD | 0.8258 | 0.1742 | +1.5559 | -0.1913 |
| G4 | 407 | TD | TD | 0.1187 | 0.8813 | +2.0048 | -0.1786 |
| G5 | 377 | ASD | ASD | 0.6387 | 0.3613 | +0.5696 | +0.1413 |
| G5 | 407 | TD | TD | 0.4344 | 0.5656 | +0.2638 | -0.0717 |
| G6 | 377 | ASD | ASD | 0.6997 | 0.3003 | +0.8460 | +0.0511 |
| G6 | 407 | TD | TD | 0.2611 | 0.7389 | +1.0404 | -0.0299 |
| E69.3-G2 full | — | — | — | — | — | ACC 90.8046; AUC 92.3648 | — |
| E69.3-G2 excluding 377/407 | — | — | — | — | — | ACC 90.5882; AUC 92.3333 | — |
| G1 full | — | — | — | — | — | ACC 90.8046; AUC 92.0997 | — |
| G1 excluding 377/407 | — | — | — | — | — | ACC 90.5882; AUC 92.1667 | — |
| G2 full | — | — | — | — | — | ACC 90.8046; AUC 91.8346 | — |
| G2 excluding 377/407 | — | — | — | — | — | ACC 90.5882; AUC 91.8889 | — |
| G3 full | — | — | — | — | — | ACC 90.8046; AUC 93.0011 | — |
| G3 excluding 377/407 | — | — | — | — | — | ACC 90.5882; AUC 93.0000 | — |
| G4 full | — | — | — | — | — | ACC 93.1034; AUC 92.6829 | — |
| G4 excluding 377/407 | — | — | — | — | — | ACC 92.9412; AUC 92.6667 | — |
| G5 full | — | — | — | — | — | ACC 91.9540; AUC 93.0011 | — |
| G5 excluding 377/407 | — | — | — | — | — | ACC 91.7647; AUC 93.0000 | — |
| G6 full | — | — | — | — | — | ACC 90.8046; AUC 92.3118 | — |
| G6 excluding 377/407 | — | — | — | — | — | ACC 90.5882; AUC 92.2778 | — |

Removing these two rows from the fold-2 metric calculation is post-hoc only. Both patients' labels were used in training, alpha calibration and historical test-ACC checkpoint selection; neither reduced nor full metrics are independent validation.

## Mechanism diagnostics

| Group | Mean α | Initial weighted grad ratio | Best-epoch grad cosine < 0 | Train global similarity acc | PHENO / ANAT / FUNC / CORR similarity acc | Adaptive gates P/A/F/C |
|---|---:|---:|---:|---:|---|---|
| G1 | 0.47725 | 0.1000 | 0.00% | 89.56% | 57.23% / 53.63% / 57.25% / 90.84% | 0.0010 / 0.0010 / 0.0010 / 0.9916 |
| G2 | 0.04672 | 0.1000 | 0.00% | 94.53% | 56.68% / 52.93% / 57.26% / 93.39% | 0.0010 / 0.0010 / 0.0009 / 0.9917 |
| G3 | 0.19479 | 0.1000 | 0.00% | 94.25% | 58.36% / 54.43% / 56.87% / 94.23% | 0.0010 / 0.0010 / 0.0009 / 0.9917 |
| G4 | 0.01772 | 0.1000 | 0.00% | 93.25% | 56.65% / 52.70% / 55.89% / 94.30% | 0.0010 / 0.0010 / 0.0009 / 0.9919 |
| G5 | 0.31154 | 0.1000 | 0.00% | 93.46% | N/A | 0.0010 / 0.0010 / 0.0009 / 0.9920 |
| G6 | 0.03048 | 0.1000 | 0.00% | 93.53% | N/A | 0.0010 / 0.0010 / 0.0009 / 0.9916 |

`mechanism.csv` records global/fine and modality losses, semantic accuracy/margins on effective training patients, F-level classification and semantic gradient norms/cosine, per-coordinate-segment gradient cosine, initial and selected-epoch weighted gradient ratios, effective ranks, gates and original-GCN/CADR message norms. Negative F-gradient cosine is a local conflict measure. Nonzero CADR message magnitude or a large adaptive gate establishes numerical contribution only, not diagnostic benefit.
Fine segments are slices of fused DARG F, so modality-specific scores cannot be interpreted as evidence from isolated raw modalities. CORR receives 70% of fine loss and starts with adaptive gate near 0.99; any strengthening may reinforce an already dominant direction rather than add complementary information. No anonymous-prompt group exists, so semantic specificity cannot be isolated from ordinary extra label supervision.
The OOF associations with PHENO site/age/sex and ANAT/FUNC quality are descriptive only. Quality features are not brain lesions. Associations are in audit_status.json under site_age_sex_quality_margin_associations; representation ranks are included in mechanism.csv and audit_status.json.

## Verdict and limits

Diagnostic verdict: **DIAGNOSTIC_NEGATIVE**. Multi-grain contrasts: G1−G5 -0.2299 pp (0/8/2), G2−G6 +0.1136 pp (3/5/2). These support only the protocol-specific diagnostic conclusion, not a paper-level innovation claim.
The experiment can test whether changing the supervised representation location affects optimization and predictions under E69.3-G2. It cannot establish unbiased generalization, independent clinical validity or uniquely medical-text-driven benefit. A positive mechanism hypothesis requires replication under no-overlap training/test, split-contained preprocessing and validation-only checkpoint selection.
Protected source/data/history SHA audit: `PASS`. Independent readback: 60/60. Passing audit rows with no fold overlap: 54/54 group-fold checkpoints; fold-2 overlap checkpoints: 6/6.

Artifacts: `summary.csv`, `fold_metrics.csv`, `paired_comparisons.csv`, `mechanism.csv`, `fold2_exposure.csv`, `checkpoint_audit.csv`, `rng_parity.csv`, `audit_status.json`.


## Quantitative interpretation

No E72.2 group has a strictly higher ten-fold mean ACC than E69.3-G2. G4 and G5 tie the baseline mean, but their only winning fold is fold 2, the label-exposed fold: G4 gains two correct cases there and loses one each in folds 4 and 7; G5 gains one there and loses one in fold 10. Across the nine no-index-overlap folds, every group's mean ACC change versus the baseline is negative. Both exposed patients themselves remain correctly classified in the baseline and all six new groups, so the fold-2 gain is not correction of their two hard predictions.
G5 gives a descriptive ranking/sensitivity tradeoff: Prob-AUC rises by 0.5354 pp and SEN by 0.9756 pp, while SPE falls by 0.8649 pp and mean ACC does not improve. G4's tied ACC comes with a 0.6637 pp Prob-AUC decrease. These are partial metric changes, not evidence of a stable overall performance gain.
Moving global-only supervision to F is not consistently better than CAM144: G5 minus E72.1-G5 is +0.2299 pp ACC (4/4/2 W/T/L), but G6 minus E72.1-G6 is -0.3448 pp (0/8/2). G5's nine-fold ACC difference is +0.1277 pp, whereas G6's is -0.2554 pp. These small opposite-direction temperature-dependent differences do not support a general benefit from supervising F.
Fine-grain necessity is unsupported: G1-G5 is -0.2299 pp (0/8/2); G2-G6 is +0.1136 pp (3/5/2), with Prob-AUC -0.2963 pp and SEN -0.7378 pp. No fine-vs-global contrast wins in at least six folds, and the gain at tau=0.1 is not accompanied by consistent multi-metric improvement. This is budget redistribution, not a pure added-loss experiment.
Prompt and temperature effects are also inconsistent: BrainPrompt beats SPromptGL at tau=1 by 0.1149 pp with only one strict winning fold, while at tau=0.1 BrainPrompt is lower by 0.2299 pp and the nine non-overlap-fold mean difference is effectively zero. BrainPrompt fine+global tau=1 and 0.1 tie in ACC; SPromptGL tau=0.1 increases ACC but lowers Prob-AUC; global-only BrainPrompt tau=1 outperforms tau=0.1 in ACC. No prompt/temperature setting establishes a stable new classification gain.
The likely limitation is not pervasive opposition to the classification gradient: all 60 selected-checkpoint eval total F-gradient cosines are positive. CORR fine similarity accuracy is approximately 90.8–94.3% on training patients, whereas PHENO/ANAT/FUNC are approximately 52.7–58.4%. CORR already dominates disease-message blending and receives 70% of fine loss, while the other heads contribute little auxiliary discrimination. This pattern is compatible with redundant supervision and weak complementary signal; it is not causal proof because no additional modality-weight or anonymous-prompt control was run.
The initial gradient budget is equalized, but does not remain equal throughout training: mean selected-checkpoint eval weighted ratios range approximately 0.0117–0.0263 across groups, below the common initial 0.1. Alpha remains frozen as required. Different late-stage ratios are observations, not evidence that calibration was secretly changed.
The resulting research decision is diagnostic NO-GO for claiming a second innovation module or a general advantage of DARG320 supervision. G5 is only a descriptive partial-ranking observation, not a validated candidate. Do not search more temperatures or weights on these same folds. Any follow-up must begin with a new hypothesis and a no-overlap, split-contained preprocessing, validation-selected protocol.

## Selected-checkpoint F gradients, CADR intervention and prediction changes

| Group | Eval grad cosine | Eval weighted ratio | Eval total negative cosines | Train global effective rank | CADR-off hard flips | Full−off ACC pp | Corrected / harmed vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 | 0.5044 | 0.0249 | 0/10 | 2.593 | 0 | +0.0000 | 4 / 6 |
| G2 | 0.5910 | 0.0170 | 0/10 | 2.640 | 2 | +0.2299 | 2 / 4 |
| G3 | 0.6061 | 0.0255 | 0/10 | 1.925 | 3 | +0.3435 | 4 / 7 |
| G4 | 0.6187 | 0.0117 | 0/10 | 4.553 | 0 | +0.0000 | 8 / 8 |
| G5 | 0.4305 | 0.0263 | 0/10 | 1.953 | 3 | +0.3448 | 4 / 4 |
| G6 | 0.5243 | 0.0152 | 0/10 | 3.026 | 0 | +0.0000 | 7 / 10 |

F-gradient fields named best_checkpoint_eval_* are independently recomputed after loading the selected checkpoint in deterministic eval mode; unprefixed gradient fields in mechanism.csv are the recorded train-mode pre-update gradient diagnostics at the selected epoch. These differ by mode and parameter update timing. The fixed-checkpoint disease_off intervention sets the adaptive CADR contribution to zero using the inherited E61 implementation, performs no training, and does not select a perturbation from test labels. It demonstrates small local prediction sensitivity for some groups, not superiority over a separately trained no-CADR model.
Across all groups only three FUNC coordinate-segment eval gradient cosines are negative (one each in G2, G5 and G6); PHENO/ANAT/CORR eval segment cosines are nonnegative. CORR adaptive gates remain about 0.9916–0.9920, versus about 0.9918 in the historical selected checkpoints. Weighted CADR CORR norms are about 1.54–1.65 and substantially exceed the corresponding original-message blend norm; other modality gates remain about 0.001. Gate values and nonzero messages therefore confirm numerical participation, while the intervention and paired metrics set the boundary on diagnostic claims.

## E72.1 historical complete metrics

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E72.1-G1 | 91.6144 +/- 3.0928 | 91.3449 +/- 3.1326 | 93.5796 +/- 3.1493 | 87.6159 +/- 4.4998 | 95.0740 +/- 3.0461 | 91.3449 +/- 3.1326 | 90.6150 +/- 3.4682 |
| E72.1-G2 | 91.8456 +/- 2.8422 | 91.6549 +/- 2.8624 | 93.7021 +/- 2.7258 | 89.0915 +/- 4.0005 | 94.2183 +/- 3.2023 | 91.6549 +/- 2.8624 | 91.0000 +/- 3.1090 |
| E72.1-G3 | 91.9592 +/- 3.0034 | 91.8252 +/- 2.9378 | 93.5612 +/- 2.9438 | 90.3110 +/- 3.9653 | 93.3395 +/- 5.2721 | 91.8252 +/- 2.9378 | 91.2679 +/- 3.1143 |
| E72.1-G4 | 91.9592 +/- 2.8219 | 91.6807 +/- 2.8693 | 93.9136 +/- 2.8305 | 88.0793 +/- 4.0074 | 95.2821 +/- 2.8803 | 91.6807 +/- 2.8693 | 91.0072 +/- 3.1472 |
| E72.1-G5 | 91.8443 +/- 2.5491 | 91.6574 +/- 2.5272 | 93.8600 +/- 2.5634 | 89.3232 +/- 3.7249 | 93.9917 +/- 4.0891 | 91.6574 +/- 2.5272 | 91.0367 +/- 2.7219 |
| E72.1-G6 | 92.0755 +/- 2.5988 | 91.9633 +/- 2.5482 | 93.4338 +/- 3.3954 | 90.5732 +/- 3.1031 | 93.3534 +/- 4.1622 | 91.9633 +/- 2.5482 | 91.3944 +/- 2.7034 |

## Per-fold independent metrics

| Group | Fold | ACC | Hard-AUC | Prob-AUC | SEN | SPE | ASD-F1 | TP/FP/FN/TN | Best epoch (1-based) | Total epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| G1 | 1 | 95.4545 | 95.2776 | 96.1079 | 92.6829 | 97.8723 | 95.0000 | 38/1/3/46 | 116 | 317 |
| G1 | 2 | 90.8046 | 90.6416 | 92.0997 | 87.8049 | 93.4783 | 90.0000 | 36/3/5/43 | 104 | 305 |
| G1 | 3 | 87.3563 | 87.3807 | 88.8123 | 87.8049 | 86.9565 | 86.7470 | 36/6/5/40 | 186 | 387 |
| G1 | 4 | 94.2529 | 93.9362 | 96.3830 | 90.0000 | 97.8723 | 93.5065 | 36/1/4/46 | 261 | 450 |
| G1 | 5 | 91.9540 | 92.1809 | 95.8511 | 95.0000 | 89.3617 | 91.5663 | 38/5/2/42 | 200 | 401 |
| G1 | 6 | 96.5517 | 96.4362 | 96.5957 | 95.0000 | 97.8723 | 96.2025 | 38/1/2/46 | 153 | 354 |
| G1 | 7 | 91.9540 | 91.4362 | 95.6383 | 85.0000 | 97.8723 | 90.6667 | 34/1/6/46 | 162 | 363 |
| G1 | 8 | 88.5057 | 88.2447 | 90.1064 | 85.0000 | 91.4894 | 87.1795 | 34/4/6/43 | 320 | 450 |
| G1 | 9 | 89.6552 | 89.1223 | 91.3830 | 82.5000 | 95.7447 | 88.0000 | 33/2/7/45 | 141 | 342 |
| G1 | 10 | 91.9540 | 91.6223 | 94.1489 | 87.5000 | 95.7447 | 90.9091 | 35/2/5/45 | 133 | 334 |
| G2 | 1 | 95.4545 | 95.1220 | 95.5371 | 90.2439 | 100.0000 | 94.8718 | 37/0/4/47 | 151 | 352 |
| G2 | 2 | 90.8046 | 90.6416 | 91.8346 | 87.8049 | 93.4783 | 90.0000 | 36/3/5/43 | 147 | 348 |
| G2 | 3 | 87.3563 | 87.2481 | 89.9258 | 85.3659 | 89.1304 | 86.4198 | 35/5/6/41 | 217 | 418 |
| G2 | 4 | 93.1034 | 92.6862 | 95.4787 | 87.5000 | 97.8723 | 92.1053 | 35/1/5/46 | 113 | 314 |
| G2 | 5 | 91.9540 | 91.9947 | 96.2766 | 92.5000 | 91.4894 | 91.3580 | 37/4/3/43 | 214 | 415 |
| G2 | 6 | 96.5517 | 96.4362 | 97.3936 | 95.0000 | 97.8723 | 96.2025 | 38/1/2/46 | 265 | 450 |
| G2 | 7 | 91.9540 | 91.4362 | 94.8936 | 85.0000 | 97.8723 | 90.6667 | 34/1/6/46 | 205 | 406 |
| G2 | 8 | 88.5057 | 88.2447 | 88.4043 | 85.0000 | 91.4894 | 87.1795 | 34/4/6/43 | 161 | 362 |
| G2 | 9 | 89.6552 | 89.4947 | 91.8617 | 87.5000 | 91.4894 | 88.6076 | 35/4/5/43 | 236 | 437 |
| G2 | 10 | 93.1034 | 93.0585 | 94.6277 | 92.5000 | 93.6170 | 92.5000 | 37/3/3/44 | 194 | 395 |
| G3 | 1 | 95.4545 | 95.1220 | 96.1598 | 90.2439 | 100.0000 | 94.8718 | 37/0/4/47 | 127 | 328 |
| G3 | 2 | 90.8046 | 90.6416 | 93.0011 | 87.8049 | 93.4783 | 90.0000 | 36/3/5/43 | 227 | 428 |
| G3 | 3 | 87.3563 | 87.2481 | 88.2821 | 85.3659 | 89.1304 | 86.4198 | 35/5/6/41 | 218 | 419 |
| G3 | 4 | 94.2529 | 93.9362 | 95.2128 | 90.0000 | 97.8723 | 93.5065 | 36/1/4/46 | 178 | 379 |
| G3 | 5 | 91.9540 | 92.1809 | 95.3191 | 95.0000 | 89.3617 | 91.5663 | 38/5/2/42 | 218 | 419 |
| G3 | 6 | 96.5517 | 96.4362 | 97.3936 | 95.0000 | 97.8723 | 96.2025 | 38/1/2/46 | 197 | 398 |
| G3 | 7 | 90.8046 | 90.1862 | 95.1064 | 82.5000 | 97.8723 | 89.1892 | 33/1/7/46 | 168 | 369 |
| G3 | 8 | 88.5057 | 88.2447 | 89.5213 | 85.0000 | 91.4894 | 87.1795 | 34/4/6/43 | 222 | 423 |
| G3 | 9 | 89.6552 | 89.1223 | 91.8085 | 82.5000 | 95.7447 | 88.0000 | 33/2/7/45 | 179 | 380 |
| G3 | 10 | 91.9540 | 91.6223 | 93.7766 | 87.5000 | 95.7447 | 90.9091 | 35/2/5/45 | 129 | 330 |
| G4 | 1 | 95.4545 | 95.1220 | 96.0042 | 90.2439 | 100.0000 | 94.8718 | 37/0/4/47 | 217 | 418 |
| G4 | 2 | 93.1034 | 93.0806 | 92.6829 | 92.6829 | 93.4783 | 92.6829 | 38/3/3/43 | 176 | 377 |
| G4 | 3 | 87.3563 | 87.5133 | 90.4030 | 90.2439 | 84.7826 | 87.0588 | 37/7/4/39 | 246 | 447 |
| G4 | 4 | 93.1034 | 92.6862 | 94.4149 | 87.5000 | 97.8723 | 92.1053 | 35/1/5/46 | 167 | 368 |
| G4 | 5 | 91.9540 | 91.9947 | 95.5851 | 92.5000 | 91.4894 | 91.3580 | 37/4/3/43 | 300 | 450 |
| G4 | 6 | 96.5517 | 96.4362 | 97.0213 | 95.0000 | 97.8723 | 96.2025 | 38/1/2/46 | 126 | 327 |
| G4 | 7 | 90.8046 | 90.9309 | 91.2766 | 92.5000 | 89.3617 | 90.2439 | 37/5/3/42 | 92 | 293 |
| G4 | 8 | 88.5057 | 88.2447 | 86.9681 | 85.0000 | 91.4894 | 87.1795 | 34/4/6/43 | 301 | 450 |
| G4 | 9 | 90.8046 | 90.5585 | 91.0638 | 87.5000 | 93.6170 | 89.7436 | 35/3/5/44 | 203 | 404 |
| G4 | 10 | 93.1034 | 93.0585 | 91.7021 | 92.5000 | 93.6170 | 92.5000 | 37/3/3/44 | 307 | 450 |
| G5 | 1 | 95.4545 | 95.1220 | 97.4053 | 90.2439 | 100.0000 | 94.8718 | 37/0/4/47 | 249 | 450 |
| G5 | 2 | 91.9540 | 91.9936 | 93.0011 | 92.6829 | 91.3043 | 91.5663 | 38/4/3/42 | 183 | 384 |
| G5 | 3 | 87.3563 | 87.6458 | 88.6002 | 92.6829 | 82.6087 | 87.3563 | 38/8/3/38 | 210 | 411 |
| G5 | 4 | 94.2529 | 93.9362 | 96.3830 | 90.0000 | 97.8723 | 93.5065 | 36/1/4/46 | 302 | 450 |
| G5 | 5 | 91.9540 | 92.1809 | 95.3191 | 95.0000 | 89.3617 | 91.5663 | 38/5/2/42 | 227 | 428 |
| G5 | 6 | 96.5517 | 96.4362 | 97.4468 | 95.0000 | 97.8723 | 96.2025 | 38/1/2/46 | 188 | 389 |
| G5 | 7 | 91.9540 | 91.4362 | 95.0000 | 85.0000 | 97.8723 | 90.6667 | 34/1/6/46 | 133 | 334 |
| G5 | 8 | 88.5057 | 88.2447 | 88.7766 | 85.0000 | 91.4894 | 87.1795 | 34/4/6/43 | 197 | 398 |
| G5 | 9 | 90.8046 | 90.5585 | 92.5000 | 87.5000 | 93.6170 | 89.7436 | 35/3/5/44 | 414 | 450 |
| G5 | 10 | 91.9540 | 91.9947 | 94.6809 | 92.5000 | 91.4894 | 91.3580 | 37/4/3/43 | 141 | 342 |
| G6 | 1 | 94.3182 | 94.2138 | 96.0042 | 92.6829 | 95.7447 | 93.8272 | 38/2/3/45 | 140 | 341 |
| G6 | 2 | 90.8046 | 90.7741 | 92.3118 | 90.2439 | 91.3043 | 90.2439 | 37/4/4/42 | 136 | 337 |
| G6 | 3 | 87.3563 | 87.2481 | 90.8271 | 85.3659 | 89.1304 | 86.4198 | 35/5/6/41 | 213 | 414 |
| G6 | 4 | 94.2529 | 93.9362 | 97.4468 | 90.0000 | 97.8723 | 93.5065 | 36/1/4/46 | 325 | 450 |
| G6 | 5 | 90.8046 | 90.9309 | 95.3191 | 92.5000 | 89.3617 | 90.2439 | 37/5/3/42 | 179 | 380 |
| G6 | 6 | 96.5517 | 96.4362 | 97.2872 | 95.0000 | 97.8723 | 96.2025 | 38/1/2/46 | 145 | 346 |
| G6 | 7 | 91.9540 | 91.8085 | 95.2660 | 90.0000 | 93.6170 | 91.1392 | 36/3/4/44 | 173 | 374 |
| G6 | 8 | 88.5057 | 88.2447 | 88.1915 | 85.0000 | 91.4894 | 87.1795 | 34/4/6/43 | 212 | 413 |
| G6 | 9 | 90.8046 | 90.5585 | 93.2979 | 87.5000 | 93.6170 | 89.7436 | 35/3/5/44 | 129 | 330 |
| G6 | 10 | 91.9540 | 91.6223 | 93.2447 | 87.5000 | 95.7447 | 90.9091 | 35/2/5/45 | 112 | 313 |

## Exact frozen equations, projection schema and implementation audit

u_g=Normalize(P_g F); u_m=Normalize(P_m F_m); s_ic=u_i dot T_c; DiseaseCE=CE(s/tau,y). BP is tau=1; SABER-style adaptation is tau=0.1. L_fine=.1 L_PHENO+.1 L_ANAT+.1 L_FUNC+.7 L_CORR. Joint L_sem=.5 L_global+.5 L_fine; global-only L_sem=L_global. L_base=L_cls+L_graph+.05 L_pair+.01 L_adv+.01 L_preserve; L_total=L_base+alpha L_sem. Alpha=.1||grad_F L_cls||/(||grad_F L_sem||+1e-12), checked finite in (0,10], computed in deterministic eval once at epoch-10 and frozen.
G1/G2: BrainPrompt-2 joint tau=1/0.1; G3/G4: SPromptGL-8 joint tau=1/0.1; G5/G6: BrainPrompt-2 global-only tau=1/0.1. All six instantiate the same F heads: global 3210 parameters and four fine heads 3240 parameters, 6450 total. Global-only fine parameters receive no training gradient. E72.1 used 1820 total auxiliary parameters (1450 global, 370 shared fine), an additional necessary adaptation and a limit on single-variable causal attribution.
Original optimizers stay GFR Adam lr=.0038 WD=.11, CAM Adam lr=.0038 WD=.11, CADR AdamW lr=.001 WD=.01; new head AdamW lr=.001 WD=.001 updates after those three, with Xavier weights/zero biases, no dropout, private seed 720000+fold. Maximum 450 epochs, warmup 10, wait>200, strict test-ACC improvement checkpoint rule. Five workers were used, GPU0 PPU-ZW810E 98 GiB, with roughly 5.5 GiB aggregate observed GPU use during parallel training. No additional formal group, seed reset per epoch, hyperparameter search or historical retraining was performed.
The full-text local PDF audit and official raw source checks are recorded in .runtime/official_source_audit/manifest.json with exact source/PDF SHAs. Official BrainPrompt compute_label_loss in both graph/CNN variants normalizes fused patient and projected label vectors then passes their dot products directly to F.cross_entropy; the original label_transform is trainable, whereas E72/E72.1/E72.2 deliberately use frozen SVD-space prototypes. Official ABIDE text order is TC then ASD; the cache is verified in reversed ASD/TD order. SPromptGL network.py includes all eight original prompts including Nromal and its PEN aggregation/injection, which this experiment does not reintroduce. SABER Module III contains patient semantic retrieval and gated residual injection (8)–(10), classifier CE (11), class contrastive softmax (12) and combined loss (13); only the class-loss form is adapted here. SABER official code was not verified.
The locked E69.3 commit's run.py, config.json and manifest.json were fetched independently from the public fixed-commit URL and exactly matched the local SHA-locked files. The inherited source audit was used at preflight; fresh online source and full-text revalidation completed during training/post-training without changing the frozen design. The initial audit reader was archived exactly, then corrected only after training for no_grad inference and expanded mechanism/reporting diagnostics; training run.py, mgdss.py, config.json and all checkpoints were unchanged.
The preferred /data experiment directory returned real Errno 5 Input/output error; .runtime/directory_access_audit.json records this and the /model fallback. No protected root, data or historical experiment file was changed.

## Fold 2 full and row-excluded complete metrics

| Model | Evaluation | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | n |
|---|---|---:|---:|---:|---:|---:|---:|
| E69.3-G2 | full | 90.8046 | 90.6416 | 92.3648 | 87.8049 | 93.4783 | 87 |
| E69.3-G2 | without | 90.5882 | 90.4167 | 92.3333 | 87.5000 | 93.3333 | 85 |
| G1 | full | 90.8046 | 90.6416 | 92.0997 | 87.8049 | 93.4783 | 87 |
| G1 | without | 90.5882 | 90.4167 | 92.1667 | 87.5000 | 93.3333 | 85 |
| G2 | full | 90.8046 | 90.6416 | 91.8346 | 87.8049 | 93.4783 | 87 |
| G2 | without | 90.5882 | 90.4167 | 91.8889 | 87.5000 | 93.3333 | 85 |
| G3 | full | 90.8046 | 90.6416 | 93.0011 | 87.8049 | 93.4783 | 87 |
| G3 | without | 90.5882 | 90.4167 | 93.0000 | 87.5000 | 93.3333 | 85 |
| G4 | full | 93.1034 | 93.0806 | 92.6829 | 92.6829 | 93.4783 | 87 |
| G4 | without | 92.9412 | 92.9167 | 92.6667 | 92.5000 | 93.3333 | 85 |
| G5 | full | 91.9540 | 91.9936 | 93.0011 | 92.6829 | 91.3043 | 87 |
| G5 | without | 91.7647 | 91.8056 | 93.0000 | 92.5000 | 91.1111 | 85 |
| G6 | full | 90.8046 | 90.7741 | 92.3118 | 90.2439 | 91.3043 | 87 |
| G6 | without | 90.5882 | 90.5556 | 92.2778 | 90.0000 | 91.1111 | 85 |

Fold-2 training class counts before/after addition: {'0': 362, '1': 422} / {'0': 363, '1': 423}; pre-addition weights (class order [0, 1]) [1.0828729281767955, 0.9289099526066351]; post-addition weights [1.0826445817947388, 0.9290780425071716]. These row-excluded values remain label-exposed, validation=test-selected diagnostics.

## Descriptive covariate associations

| Group | Site mean abs rho | Age-bin mean abs rho | Sex mean abs rho | ANAT quality mean abs rho | FUNC motion/quality mean abs rho |
|---|---:|---:|---:|---:|---:|
| G1 | 0.0363 | 0.0296 | 0.0501 | 0.0264 | 0.0383 |
| G2 | 0.0359 | 0.0278 | 0.0370 | 0.0246 | 0.0512 |
| G3 | 0.0294 | 0.0283 | 0.0325 | 0.0202 | 0.0311 |
| G4 | 0.0284 | 0.0225 | 0.0378 | 0.0209 | 0.0373 |
| G5 | 0.0318 | 0.0257 | 0.0295 | 0.0396 | 0.0373 |
| G6 | 0.0384 | 0.0292 | 0.0534 | 0.0275 | 0.0436 |

These are pooled diagnostic OOF margin associations after training; folds use differently learned projection heads. They are descriptive summaries rather than causal or independently validated confounding estimates. Site and age are encoded features, not raw continuous covariates; the JSON retains the individual strongest feature for each category.

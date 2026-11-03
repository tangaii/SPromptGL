# Experiment 61 — Post-alignment OOF Reliability versus End-to-end Adaptive CADR Graph Propagation

## Scope and protocol

This experiment contains exactly two new groups and 20 outer-fold classification tasks. The sole base is Experiment 53.3 third group `CADR_WD_0p01_GFR_WD_0p11`; its historical fold metrics and checkpoints remain read-only. E1 runs one shared 10-epoch warmup, then a patient-disjoint three-fold, ten-epoch joint four-modality relation OOF fit, then fixes fold-level reliability. E2 branches from the same warmup and learns four scalar gates from classification loss without OOF.

Locked source commit: `b7e477c9657422a119adc7872ccf8c8188871e9e`. The original SPromptGL GFR description is in [the MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf); this run follows the checked 53.3 source path. E61 retains PHENO48, ANAT6, FUNC10 and CORR256, the original 871-node transductive graph, CAM, GFR, GraphConstructLoss, relation BCE, conditional adversary and preservation loss.

All groups use the uncalibrated relation score `sigmoid(logit)` and the fixed 0.55/0.45 thresholds. E53.3 used old OOF Platt parameters; E61 intentionally does not, because E2 has no OOF. Thus comparisons with the historical base are exploratory and include this shared score-calibration change. The historical `val=test` best-checkpoint rule and pre-CV RFE/standardization remain unchanged; these results are not unbiased external generalization estimates.

The first ten epochs use the original 53.3 training forward and optimizer calls. Exact ten-epoch twin replay passed before training. Same-structure GFR/CAM and relation-core initial parameters, fold split, relation-pair cache and construction RNG matched SHA references. The only initialization exception is the old OOF-derived reliability/calibration state, replaced by neutral buffers and an E2 gate initialized at 0.1. After branching, matched epochs use the same CAM dropout call count/order; E1 OOF uses isolated torch RNG and local NumPy generators.

Runtime used PPU-ZW810E with maximum 4 concurrent workers; 10 shared warmups, 20/20 formal tasks, and 20/20 independent checkpoint audits completed. Post-fork RNG traces matched exactly for 3584/3584 compared epochs; mismatches: 0.

## Classification results

Values are 10-fold mean ± population SD, in percent. The historical row is read-only.

|Group|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|CADR_WD_0p01_GFR_WD_0p11|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|89.0915 ± 3.5263|94.6392 ± 3.5168|91.8653 ± 2.8348|91.2470 ± 3.0658|
|POSTALIGN_OOF|91.9592 ± 2.8684|91.7434 ± 2.8950|93.1856 ± 3.0236|88.8476 ± 4.1651|94.6392 ± 3.3856|91.7434 ± 2.8950|91.0928 ± 3.1408|
|E2E_ADAPTIVE_GATE|91.4995 ± 2.4842|91.2480 ± 2.5179|93.8126 ± 2.6938|87.8476 ± 3.5514|94.6485 ± 2.4105|91.2480 ± 2.5179|90.5253 ± 2.7692|

## Paired fold differences

Differences are percentage points; intervals are descriptive paired-t intervals over the ten correlated folds, not confirmatory inference.

|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E1 − historical 53.3|-0.115 pp (1/7/2, descriptive CI -0.582 to +0.352)|-0.122 pp (2/6/2, descriptive CI -0.603 to +0.359)|-0.483 pp (2/1/7, descriptive CI -1.177 to +0.211)|-0.244 pp (1/7/2, descriptive CI -1.251 to +0.764)|+0.000 pp (1/8/1, descriptive CI -0.717 to +0.717)|-0.154 pp (2/6/2, descriptive CI -0.698 to +0.389)|
|E2 − historical 53.3|-0.575 pp (1/3/6, descriptive CI -1.156 to +0.007)|-0.617 pp (2/1/7, descriptive CI -1.181 to -0.053)|+0.144 pp (4/0/6, descriptive CI -0.764 to +1.053)|-1.244 pp (1/3/6, descriptive CI -2.491 to +0.004)|+0.009 pp (2/5/3, descriptive CI -1.441 to +1.460)|-0.722 pp (2/1/7, descriptive CI -1.292 to -0.151)|
|E2 − E1|-0.460 pp (1/4/5, descriptive CI -1.035 to +0.115)|-0.495 pp (3/1/6, descriptive CI -1.080 to +0.089)|+0.627 pp (7/1/2, descriptive CI +0.020 to +1.234)|-1.000 pp (2/3/5, descriptive CI -2.718 to +0.718)|+0.009 pp (3/3/4, descriptive CI -1.609 to +1.628)|-0.568 pp (3/1/6, descriptive CI -1.166 to +0.031)|

## Mechanism evidence

### E1 post-alignment OOF

|Modality|OOF-AUC mean ± SD|Mean reliability|Nonzero-reliability folds|
|---|---:|---:|---:|
|PHENO|0.5012 ± 0.0073|0.0000|0/10|
|ANAT|0.5016 ± 0.0044|0.0000|0/10|
|FUNC|0.5032 ± 0.0060|0.0000|0/10|
|CORR|0.5081 ± 0.0068|0.0000|0/10|

OOF reliability was computed only from the fixed tenth-epoch endpoint of each inner model. The inner models jointly trained all four relation encoders, scorers and the conditional modality discriminator; outer-test patients were excluded. Each outer-training patient appeared once as an inner validation patient. OOF estimates reliability only; the main fold model generated every propagated edge score.

### E2 learned gates

|Modality|Final gate mean ± SD|
|---|---:|
|PHENO|0.0999 ± 0.0074|
|ANAT|0.0992 ± 0.0014|
|FUNC|0.0971 ± 0.0023|
|CORR|0.1200 ± 0.0066|

### E2 edge and message contribution

Counts and contribution norms are averaged across ten independently audited checkpoints. Message ratios compare each weighted 871-node modality message with the Frobenius norm of the original 871×320 input; component norms are descriptive and are not causal effects.

|Modality|Gate|Positive edges|Negative edges|Uncertain edges %|CADR message / input %|Original GCN / input %|Positive GCN / input %|Negative GCN / input %|Direct difference / input %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PHENO|0.0999|793.6|1500.3|35.33|0.02177|0.00119|0.0001403|0.0000069|0.02177|
|ANAT|0.0992|314.2|664.1|73.43|0.00949|0.00248|0.0002865|0.0000062|0.00946|
|FUNC|0.0971|723.9|922.5|53.85|0.01400|0.00171|0.0002015|0.0000070|0.01399|
|CORR|0.1200|1747.6|1527.6|1.54|0.05110|0.02135|0.0034144|0.0001293|0.05009|

Classification-only gradient replay at the best checkpoint used the outer-training classification loss in deterministic evaluation mode, without stepping an optimizer. These are mean absolute scalar gate gradients and mean relation-module gradient norms; per-fold values are in `mechanism_metrics.csv`.

|Modality|Adaptive gate gradient|Edge gate gradient|Scorer gradient norm|Encoder gradient norm|
|---|---:|---:|---:|---:|
|PHENO|1.086e-05|3.955e-06|8.424e-05|1.200e-04|
|ANAT|3.292e-06|1.313e-06|4.237e-05|1.774e-05|
|FUNC|6.848e-06|4.102e-06|8.614e-05|5.526e-05|
|CORR|1.443e-04|9.285e-06|1.460e-05|8.310e-04|

### Fixed-checkpoint interventions

Each cell gives full-model minus intervention ACC in percentage points, mean number of changed test hard predictions, and mean absolute probability change. Interventions do not use test labels to choose the perturbation.

|Group|Intervention|ΔACC pp|Changed predictions / fold|Mean absolute Δprobability|
|---|---|---:|---:|---:|
|E61-E1|all CADR off|+0.000|0.00|0.000000|
|E61-E1|score permutation|+0.000|0.00|0.000000|
|E61-E1|positive GCN off|+0.000|0.00|0.000000|
|E61-E1|negative GCN off|+0.000|0.00|0.000000|
|E61-E1|direct difference off|+0.000|0.00|0.000000|
|E61-E1|PHENO CADR off|+0.000|0.00|0.000000|
|E61-E1|ANAT CADR off|+0.000|0.00|0.000000|
|E61-E1|FUNC CADR off|+0.000|0.00|0.000000|
|E61-E1|CORR CADR off|+0.000|0.00|0.000000|
|E61-E2|all CADR off|+0.000|0.00|0.000242|
|E61-E2|score permutation|+0.000|0.00|0.000303|
|E61-E2|positive GCN off|+0.000|0.00|0.000185|
|E61-E2|negative GCN off|+0.000|0.00|0.000088|
|E61-E2|direct difference off|+0.000|0.00|0.000225|
|E61-E2|PHENO CADR off|+0.000|0.00|0.000039|
|E61-E2|ANAT CADR off|+0.000|0.00|0.000013|
|E61-E2|FUNC CADR off|+0.000|0.00|0.000031|
|E61-E2|CORR CADR off|+0.000|0.00|0.000240|

Gate values describe the learned blend under this training objective. They are not reliability estimates or calibrated probabilities. Per-fold gates, gradient norms, edge counts, score distributions, GCN components, direct-difference messages and fixed-checkpoint intervention metrics are in `mechanism_metrics.csv` and `checkpoint_audit.csv`.

The post-alignment OOF AUCs describe relation quality after the aligned inner training procedure. No separate pre-alignment OOF fit was run, so this experiment does not estimate a pre/post alignment AUC change. GCN-positive, GCN-negative and direct-neighbor contributions are separately reported; a nonzero message norm alone is not evidence that a component improved diagnosis.

## Scientific judgment

Overall assessment: **NO-GO** under a conservative descriptive rule: a group receives GO only when ACC, Hard-AUC and Probability-AUC all improve over the historical third group, SEN falls by no more than 1 pp, and active relation evidence is supported by a fixed-checkpoint intervention. PARTIAL requires some metric gain and intervention evidence; otherwise the result is NO-GO. These labels are exploratory, not significance claims.

|Method|Judgment|ACC Δ pp|Hard-AUC Δ pp|Prob-AUC Δ pp|SEN Δ pp|Active modality-folds|Disease-off ACC drop pp|Shuffled-score ACC drop pp|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|E61-E1 POSTALIGN_OOF|NO-GO|-0.115|-0.122|-0.483|-0.244|0|+0.000|+0.000|
|E61-E2 E2E_ADAPTIVE_GATE|NO-GO|-0.575|-0.617|+0.144|-1.244|40|+0.000|+0.000|

## Scientific reading

Category supervision or a learned gate can improve a metric without establishing that disease-relation alignment is useful. Evaluate the paired metrics alongside the no-CADR, single-modality, score-shuffle and positive/negative/direct component interventions. If E1 has zero reliability, the method correctly falls back to the same original GraphLearn/GCN path. If E2 learns small gates or has no active scored edges, it did not establish useful adaptive disease propagation.

The current comparison is single-seed and uses overlapping cross-validation training sets. It is exploratory. A mathematical relation embedding or a nonzero gate does not establish a biological ASD mechanism.

The per-fold `acc`, `hard_auc`, `prob_auc`, `sen`, and `spe` columns in `fold_metrics.csv` are stored as 0–1 proportions; `summary.csv` and this report display percent.

## Files

- Experiment root: `/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e`
- Formal checkpoints: `/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/.runtime/checkpoints`
- Shared warmups and OOF cache: `/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/.runtime`
- Fold metrics: `results/fold_metrics.csv`
- Paired comparisons: `results/paired_comparisons.csv`
- Independent audits and fixed-checkpoint interventions: `results/checkpoint_audit.csv`
- Mechanism details: `results/mechanism_metrics.csv`
- Task manifest: `manifest.json`

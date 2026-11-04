# Experiment 72.1 — E69.3-G2-MGDSS

## Audit Status

Status: `COMPLETE_AUDITED`; formal folds completed: 60/60; independent checkpoints: 60/60; failures: 0.
This is an intentional label-exposure mechanism diagnostic. Fold 2 adds 1-based patient rows 377 and 407 to the effective training set, leaves both in its test set, and uses both labels for classification, disease semantic loss, alpha calibration, and test-ACC checkpoint selection. Aggregate and row-excluded metrics are not leakage-free generalization estimates.
Initial E69.3-G2 model/warmup state, optimizer state, split, relation-pair SHA and gate initialization parity: 10/10 folds. Exact epoch RNG comparison against the E69.3-G2 reference: PASS.
All 24 frozen preflight checks passed; repeat/resume parameter, optimizer, RNG and logits parity passed on folds 1 and 2. Reference and current device/backend: PPU-ZW810E with CUDA 12.9. The other nine folds have no train/test index overlap.

## Source Audit

The formal base is the immutable E69.3-G2 commit `41dbcb45a0273673857051d8e821b8c3315eef00`. Its runner calls E61 `forward_e2_state` with the E2 adaptive blend, original GCN message plus disease-conditioned CADR message, learnable sigmoid edge/adaptive gates, and unchanged GFR/CAM/Outputlayer. Gates initialize to edge `[0.5,0.5,0.5,0.5]` and adaptive `[0.001,0.001,0.001,0.99]` using trainable logit parameters. E69.3's `apply_fold_specific_training_additions` is used directly after the E61 epoch-10 warmup; fold 2's 4096 relation pairs (2048 same-class, 2048 different-class) and class weights are regenerated with that function's own rule and local RNG. No E53 fallback propagation is substituted.
The SPromptGL paper/code audit covers its four-modality features, GFR, GraphLearn/GCN, CAM/PLN and original classifier. The BrainPrompt source audit covers the two official ABIDE descriptions and `compute_label_loss()`: normalized cosine scores are passed directly to cross-entropy, hence temperature 1.0. The SABER paper equations (11)-(13) are implemented only as class-level softmax cross-entropy from equation (12); this is not a full SABER reproduction.
SPromptGL and BrainPrompt official sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [SPromptGL code](https://github.com/wanxixi11/SPromptGL), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt code](https://github.com/AngusMonroe/BrainPrompt), [BrainPrompt graph network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptg_net.py), [BrainPrompt label network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptc_net.py), and [SABER paper](https://arxiv.org/abs/2607.01901).
E72 was previously tested on the 53.3 base; its global BrainPrompt/SABER style groups are historical revalidation evidence, not novel supervision principles. E72's six ACCs were G1 91.6144%, G2 91.2709%, G3 91.3845%, G4 91.8443%, G5 91.9579%, G6 91.9592%. Those values are descriptive only because the E72 and E72.1 bases, CADR forward paths, initial states and effective training labels differ.

## Frozen Design

All six groups keep the E69.3-G2 original classification path and classifier. The single final CAM hook captures `[871,4,36]`; E69.3 names the classifier module `Outputlayer`. The captured flatten is elementwise identical to its `[871,144]` input, CAM is called once per forward, and the auxiliary heads do not feed inference.
Text encoder: `frozen OpenAI CLIP RN50`, dimension 1024, RN50 weights SHA `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`; this shared CLIP encoder controls the text-encoder variable and does not reproduce BrainPrompt's original Llama encoder.
Official BrainPrompt text strings, preserved verbatim and mapped to `[ASD,TD]`: 1. `A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.` ; 2. `A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.`. Raw text SHA `dbbfa807a88834a35d097aea130337c884e5264920980dd57e584c6f083f828d`.
Official SPromptGL text strings, preserved verbatim (including `Nromal`): 1. `Nromal brain volume` ; 2. `Uniform cortical thickness` ; 3. `Symmetrical brain structure` ; 4. `Normal ventricular morphology` ; 5. `Altered brain volume` ; 6. `Abnormal functional connectivity` ; 7. `White matter integrity issues` ; 8. `Different activation patterns`. Raw text SHA `23791903292668e54164e4d3db1b0556776f29f5fa37cc46fe7fe13c48cefa6e`.
The ten frozen embeddings span a shared rank-10 CPU float64 SVD basis; maximum pre/post projection cosine error is `1.110e-16`.
Frozen RN50 CLIP embeddings and the shared rank-10 CPU-float64 SVD span are copied from E72. Text cache SHA: `c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`; shared basis SHA: `3e79ee238447ff47ad1029408b99db94c3b4c4fbc657aa5a5251754563dc3868`; BrainPrompt bank SHA: `f7105b19f5a81fb078c51636136b1e18d56fa1b204bb4a5d82a488b6f48b8176`; SPromptGL bank SHA: `ee741144039d60c5de8bb36576476bdc9fa67ed2ff3f7cb97a13bba08c78a55a`. Prototype cosine values: BrainPrompt-2 0.9847826; SPromptGL-8 0.9070364. The geometry-preserving projection is an Experiment 72 adaptation, not a step specified by the papers.

| Group | Prompt source | Temperature | Supervision grain |
|---|---|---:|---|
| G1 | BrainPrompt-2 | 1 | fine + global |
| G2 | BrainPrompt-2 | 0.1 | fine + global |
| G3 | SPromptGL-8 | 1 | fine + global |
| G4 | SPromptGL-8 | 0.1 | fine + global |
| G5 | BrainPrompt-2 | 1 | global only |
| G6 | BrainPrompt-2 | 0.1 | global only |
For normalized patient vectors and frozen class prototypes, both losses are the same cosine-softmax CE at the same temperature. BrainPrompt uses `tau=1`; the SABER-style adaptation fixes `tau=0.1`. Thus the comparison tests temperature, not two mathematically distinct class-loss theories. The global/fine joint loss is `0.5 L_global + 0.5 L_fine`; `L_fine=.10 L_PHENO+.10 L_ANAT+.10 L_FUNC+.70 L_CORR`. The global-only comparison redistributes the same auxiliary budget; it does not keep global loss fixed and add fine loss.
The unchanged base objective is `L_base=L_cls+L_graph+0.05 L_pair+0.01 L_adv+0.01 L_preserve`; total training loss is `L_total=L_base+alpha L_sem`. The six group definitions above are exhaustive; no additional formal group was run.
The BrainPrompt prototype cosine bounds the tau-1 cosine-logit gap by `0.174456`; its theoretical maximum correct-class probability is `54.3504%` (minimum `45.6496%`), which limits auxiliary confidence. Alpha was calibrated per fold/group to initial `0.10 * ||grad_CAM(L_cls)|| / ||grad_CAM(L_sem)||`, constrained to `(0,10]`, and then frozen. Fold-2 calibration uses both exposed labels.

## Results

Means and population SD are in percent. E69.3-G2 is the read-only historical reference; the six experiment groups are diagnostic measurements with the fold-2 overlap retained.
| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E69.3-G2 | 92.0742 +/- 2.7482 | 91.8995 +/- 2.7186 | 93.3759 +/- 2.9908 | 89.5854 +/- 3.4710 | 94.2137 +/- 3.9815 | 91.8995 +/- 2.7186 | 91.2984 +/- 2.9222 |
| G1 | 91.6144 +/- 3.0928 | 91.3449 +/- 3.1326 | 93.5796 +/- 3.1493 | 87.6159 +/- 4.4998 | 95.0740 +/- 3.0461 | 91.3449 +/- 3.1326 | 90.6150 +/- 3.4682 |
| G2 | 91.8456 +/- 2.8422 | 91.6549 +/- 2.8624 | 93.7021 +/- 2.7258 | 89.0915 +/- 4.0005 | 94.2183 +/- 3.2023 | 91.6549 +/- 2.8624 | 91.0000 +/- 3.1090 |
| G3 | 91.9592 +/- 3.0034 | 91.8252 +/- 2.9378 | 93.5612 +/- 2.9438 | 90.3110 +/- 3.9653 | 93.3395 +/- 5.2721 | 91.8252 +/- 2.9378 | 91.2679 +/- 3.1143 |
| G4 | 91.9592 +/- 2.8219 | 91.6807 +/- 2.8693 | 93.9136 +/- 2.8305 | 88.0793 +/- 4.0074 | 95.2821 +/- 2.8803 | 91.6807 +/- 2.8693 | 91.0072 +/- 3.1472 |
| G5 | 91.8443 +/- 2.5491 | 91.6574 +/- 2.5272 | 93.8600 +/- 2.5634 | 89.3232 +/- 3.7249 | 93.9917 +/- 4.0891 | 91.6574 +/- 2.5272 | 91.0367 +/- 2.7219 |
| G6 | 92.0755 +/- 2.5988 | 91.9633 +/- 2.5482 | 93.4338 +/- 3.3954 | 90.5732 +/- 3.1031 | 93.3534 +/- 4.1622 | 91.9633 +/- 2.5482 | 91.3944 +/- 2.7034 |

Paired deltas use left minus right in percentage points. W/T/L counts strict ACC wins, ties and losses across ten folds.

| Comparison | ΔACC pp | W/T/L | 9-fold ΔACC / wins | ΔHard-AUC | ΔProb-AUC | ΔSEN | ΔSPE | Fold ΔACC pp |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1_vs_E69.3-G2 | -0.4598 | 1/5/4 | -0.2554 / 1 | -0.5546 | +0.2037 | -1.9695 | +0.8603 | `[0.0, -2.29885057, -1.14942529, 0.0, 0.0, 0.0, -1.14942529, 1.14942529, -1.14942529, 0.0]` |
| G2_vs_E69.3-G2 | -0.2286 | 0/8/2 | -0.2540 / 0 | -0.2446 | +0.3261 | -0.4939 | +0.0046 | `[-1.13636364, 0.0, -1.14942529, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| G3_vs_E69.3-G2 | -0.1149 | 2/5/3 | +0.0000 / 2 | -0.0743 | +0.1853 | +0.7256 | -0.8742 | `[0.0, -1.14942529, 0.0, 0.0, 1.14942529, 0.0, -1.14942529, 0.0, -1.14942529, 1.14942529]` |
| G4_vs_E69.3-G2 | -0.1149 | 2/5/3 | -0.2554 / 1 | -0.2188 | +0.5377 | -1.5061 | +1.0685 | `[0.0, 1.14942529, 0.0, 0.0, 1.14942529, 0.0, -1.14942529, 0.0, -1.14942529, -1.14942529]` |
| G5_vs_E69.3-G2 | -0.2299 | 1/6/3 | -0.2554 / 1 | -0.2421 | +0.4841 | -0.2622 | -0.2220 | `[0.0, 0.0, 0.0, -1.14942529, 1.14942529, -1.14942529, -1.14942529, 0.0, 0.0, 0.0]` |
| G6_vs_E69.3-G2 | +0.0013 | 2/6/2 | -0.1263 / 1 | +0.0637 | +0.0578 | +0.9878 | -0.8603 | `[-1.13636364, 1.14942529, 0.0, 0.0, 1.14942529, 0.0, 0.0, 0.0, 0.0, -1.14942529]` |
| G1_vs_G3 | -0.3448 | 1/5/4 | -0.2554 / 1 | -0.4803 | +0.0184 | -2.6951 | +1.7345 | `[0.0, -1.14942529, -1.14942529, 0.0, -1.14942529, 0.0, 0.0, 1.14942529, 0.0, -1.14942529]` |
| G2_vs_G4 | -0.1136 | 3/3/4 | +0.0015 / 3 | -0.0258 | -0.2115 | +1.0122 | -1.0638 | `[-1.13636364, -1.14942529, -1.14942529, 0.0, -1.14942529, 0.0, 1.14942529, 0.0, 1.14942529, 1.14942529]` |
| G1_vs_G2 | -0.2312 | 2/5/3 | -0.0015 / 2 | -0.3100 | -0.1225 | -1.4756 | +0.8557 | `[1.13636364, -2.29885057, 0.0, 0.0, 0.0, 0.0, -1.14942529, 1.14942529, -1.14942529, 0.0]` |
| G3_vs_G4 | +0.0000 | 1/8/1 | +0.2554 / 1 | +0.1445 | -0.3524 | +2.2317 | -1.9426 | `[0.0, -2.29885057, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.29885057]` |
| G5_vs_G6 | -0.2312 | 2/4/4 | -0.1292 / 2 | -0.3059 | +0.4263 | -1.2500 | +0.6383 | `[1.13636364, -1.14942529, 0.0, -1.14942529, 0.0, -1.14942529, -1.14942529, 0.0, 0.0, 1.14942529]` |
| G1_vs_G5 | -0.2299 | 3/3/4 | +0.0000 / 3 | -0.3125 | -0.2804 | -1.7073 | +1.0823 | `[0.0, -2.29885057, -1.14942529, 1.14942529, -1.14942529, 1.14942529, 0.0, 1.14942529, -1.14942529, 0.0]` |
| G2_vs_G6 | -0.2299 | 1/6/3 | -0.1277 / 1 | -0.3084 | +0.2683 | -1.4817 | +0.8649 | `[0.0, -1.14942529, -1.14942529, 0.0, -1.14942529, 0.0, 0.0, 0.0, 0.0, 1.14942529]` |

Across the nine folds without index overlap, paired mean ΔACC and win counts are included in the comparison table. Fold 2 full-test and row-excluded metrics are listed below for E69.3-G2 and all six groups; row-excluded values remain post-hoc because training and checkpoint selection saw those labels.

## Fold 2 Label Exposure

True labels: row 377 = 0, row 407 = 1 (`ASD=0`, `TD=1`). Effective training grew from 784 to 786 while test size remained 87; regenerated relation pair SHA is `c5ceb5c322490afe4c411751db8579ce4c960d5d7062d3c5c12640f56fbaf43c`. Effective class weights are [1.0826445817947388, 0.9290780425071716]; counts before/after addition are {'0': 362, '1': 422} and {'0': 363, '1': 423}.
| Model | Row | True class | Prediction | P(ASD) | P(TD) | True-class logit margin | Semantic correct-class margin change |
|---|---:|---|---|---:|---:|---:|---:|
| E69.3-G2 | 377 | ASD | ASD | 0.5771 | 0.4229 | +0.3107 | N/A |
| E69.3-G2 | 407 | TD | TD | 0.2274 | 0.7726 | +1.2228 | N/A |
| G1 | 377 | ASD | ASD | 0.5302 | 0.4698 | +0.1209 | +0.0797 |
| G1 | 407 | TD | TD | 0.3029 | 0.6971 | +0.8334 | +0.0757 |
| G2 | 377 | ASD | ASD | 0.7295 | 0.2705 | +0.9920 | +0.0931 |
| G2 | 407 | TD | TD | 0.2614 | 0.7386 | +1.0386 | +0.1040 |
| G3 | 377 | ASD | ASD | 0.6094 | 0.3906 | +0.4449 | -0.4803 |
| G3 | 407 | TD | TD | 0.3821 | 0.6179 | +0.4808 | +0.4261 |
| G4 | 377 | ASD | ASD | 0.7497 | 0.2503 | +1.0970 | -0.0645 |
| G4 | 407 | TD | TD | 0.3956 | 0.6044 | +0.4239 | +0.1219 |
| G5 | 377 | ASD | ASD | 0.7264 | 0.2736 | +0.9764 | +0.1531 |
| G5 | 407 | TD | TD | 0.2317 | 0.7683 | +1.1990 | +0.1108 |
| G6 | 377 | ASD | ASD | 0.5997 | 0.4003 | +0.4041 | +0.1381 |
| G6 | 407 | TD | TD | 0.2224 | 0.7776 | +1.2520 | +0.0278 |

| Model | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | n |
|---|---:|---:|---:|---:|---:|---:|
| E69.3-G2 full | 90.8046 | 90.6416 | 92.3648 | 87.8049 | 93.4783 | 87 |
| E69.3-G2 without 377/407 | 90.5882 | 90.4167 | 92.3333 | 87.5000 | 93.3333 | 85 |
| G1 full | 88.5057 | 88.2025 | 92.1527 | 82.9268 | 93.4783 | 87 |
| G1 without 377/407 | 88.2353 | 87.9167 | 92.1111 | 82.5000 | 93.3333 | 85 |
| G2 full | 90.8046 | 90.6416 | 93.1071 | 87.8049 | 93.4783 | 87 |
| G2 without 377/407 | 90.5882 | 90.4167 | 93.2222 | 87.5000 | 93.3333 | 85 |
| G3 full | 89.6552 | 89.8197 | 92.4178 | 92.6829 | 86.9565 | 87 |
| G3 without 377/407 | 89.4118 | 89.5833 | 92.3889 | 92.5000 | 86.6667 | 85 |
| G4 full | 91.9540 | 91.8611 | 93.4252 | 90.2439 | 93.4783 | 87 |
| G4 without 377/407 | 91.7647 | 91.6667 | 93.4444 | 90.0000 | 93.3333 | 85 |
| G5 full | 90.8046 | 90.9067 | 93.2131 | 92.6829 | 89.1304 | 87 |
| G5 without 377/407 | 90.5882 | 90.6944 | 93.2222 | 92.5000 | 88.8889 | 85 |
| G6 full | 91.9540 | 91.9936 | 94.0085 | 92.6829 | 91.3043 | 87 |
| G6 without 377/407 | 91.7647 | 91.8056 | 94.1111 | 92.5000 | 91.1111 | 85 |
Semantic margin change compares the epoch-10 global projection with the selected checkpoint. For row 377 it increased in G1, G2, G5 and G6 but decreased in G3 and G4; for row 407 it increased in all six groups. Both patients' labels were exposed during training and alpha calibration, so this is an exposure-associated mechanism observation, not independent semantic evidence.

## Mechanism

`mechanism.csv` contains per-fold best-epoch losses, calibrated and observed gradient ratios, total/per-position classification-semantic gradient cosines, training similarity accuracy/margins, projected representation norms/effective ranks, gate values, and GCN/CADR message norms. Negative cosines are local gradient conflict diagnostics; nonzero message norms only establish numerical contribution, not diagnostic benefit. Fold-level correlations of semantic margin with PHENO site/age/sex, ANAT quality and FUNC motion/quality are descriptive; quality features are not lesion evidence.

| Group | alpha mean | Initial ratio mean | Best ratio mean | Best grad cosine | Negative folds | Global / CORR train acc | PHENO / ANAT / FUNC / CORR gradient cosine |
|---|---:|---:|---:|---:|---:|---:|---|
| G1 | 1.01419 | 0.100000 | 0.1714 | +0.3847 | 0/10 | 91.7% / 91.0% | +0.309/+0.267/+0.212/+0.453 |
| G2 | 0.09778 | 0.100000 | 0.1067 | +0.4535 | 0/10 | 94.2% / 94.2% | +0.351/+0.303/+0.244/+0.544 |
| G3 | 0.40590 | 0.100000 | 0.2187 | +0.3364 | 0/10 | 93.3% / 93.1% | +0.256/+0.221/+0.160/+0.406 |
| G4 | 0.03602 | 0.100000 | 0.0723 | +0.4636 | 0/10 | 93.9% / 92.3% | +0.300/+0.197/+0.232/+0.568 |
| G5 | 0.59421 | 0.100000 | 0.1752 | +0.2903 | 0/10 | 94.6% / N/A | +0.255/+0.210/+0.140/+0.386 |
| G6 | 0.05760 | 0.100000 | 0.1077 | +0.3178 | 0/10 | 94.4% / N/A | +0.225/+0.174/+0.128/+0.428 |

The first gradient-ratio column is the pre-registered calibration target: its mean remains 0.100000 across groups. The best-epoch ratio is a later observation and is not recalibrated. Fine-grain accuracy is N/A for G5/G6 because those groups do not compute any per-position semantic loss; it is not zero.

| Group | PHENO / ANAT / FUNC / CORR training similarity accuracy | Corresponding mean loss |
|---|---|---|
| G1 | 75.3% / 75.5% / 72.0% / 91.0% | 0.6594 / 0.6611 / 0.6639 / 0.6329 |
| G2 | 79.3% / 78.2% / 74.1% / 94.2% | 0.4900 / 0.4977 / 0.5350 / 0.3018 |
| G3 | 75.7% / 74.5% / 67.7% / 93.1% | 0.6162 / 0.6252 / 0.6442 / 0.5371 |
| G4 | 77.5% / 72.6% / 70.7% / 92.3% | 0.4772 / 0.5767 / 0.5885 / 0.2047 |
| G5 | N/A / N/A / N/A / N/A | N/A / N/A / N/A / N/A |
| G6 | N/A / N/A / N/A / N/A | N/A / N/A / N/A / N/A |

| Group | Best adaptive gates P / A / F / C | Median weighted GCN message norm P / A / F / C | Median weighted CADR message norm P / A / F / C |
|---|---|---|---|
| G1 | 0.0010 / 0.0010 / 0.0009 / 0.9917 | 0.0051 / 0.0051 / 0.0016 / 0.0000 | 0.0012 / 0.0005 / 0.0008 / 1.5971 |
| G2 | 0.0010 / 0.0010 / 0.0009 / 0.9916 | 0.0028 / 0.0030 / 0.0007 / 0.0000 | 0.0010 / 0.0004 / 0.0007 / 1.5954 |
| G3 | 0.0010 / 0.0010 / 0.0009 / 0.9920 | 0.0039 / 0.0025 / 0.0031 / 0.0000 | 0.0011 / 0.0005 / 0.0008 / 1.6037 |
| G4 | 0.0010 / 0.0010 / 0.0009 / 0.9917 | 0.0023 / 0.0015 / 0.0004 / 0.0000 | 0.0011 / 0.0005 / 0.0008 / 1.5929 |
| G5 | 0.0010 / 0.0010 / 0.0009 / 0.9917 | 0.0055 / 0.0010 / 0.0008 / 0.0000 | 0.0011 / 0.0005 / 0.0008 / 1.6032 |
| G6 | 0.0009 / 0.0010 / 0.0009 / 0.9920 | 0.0089 / 0.0062 / 0.0009 / 0.0000 | 0.0011 / 0.0005 / 0.0008 / 1.6364 |

| Group | Global effective rank | Global correct margin ASD / TD | OOF similarity accuracy P / A / F / C |
|---|---:|---:|---|
| G1 | 1.612 | +0.099 / +0.140 | 71.3% / 76.8% / 70.0% / 86.2% |
| G2 | 1.815 | +0.100 / +0.134 | 77.0% / 79.4% / 71.9% / 88.3% |
| G3 | 1.377 | +0.275 / +0.357 | 69.3% / 72.7% / 66.0% / 90.6% |
| G4 | 3.002 | +0.207 / +0.261 | 72.9% / 74.9% / 68.2% / 89.0% |
| G5 | 1.499 | +0.110 / +0.138 | N/A / N/A / N/A / N/A |
| G6 | 1.716 | +0.116 / +0.125 | N/A / N/A / N/A / N/A |

The OOF representation summaries include the two fold-2 exposed patients and are explicitly diagnostic. There is no anonymous-prompt control, so these results cannot isolate medical-text semantics from ordinary auxiliary label supervision. CORR receives 70% of fine loss while E69.3 initializes its adaptive gate to 0.99; changes in CORR similarity therefore do not establish complementary modality evidence. The E69.3 historical intervention audit also found that turning CADR off did not alter hard predictions, so gate magnitude alone is not evidence of diagnostic utility.

Site/age/sex/quality correlations use OOF diagnostic margins and are descriptive only. Across groups, mean absolute Spearman correlations range 0.020–0.036 for site, 0.022–0.033 for age bins, 0.026–0.053 for sex, 0.016–0.028 for ANAT quality, and 0.062–0.079 for FUNC motion/quality. The strongest single association is `func_mean_fd` (G4, r=-0.184); this is not causal evidence or evidence of a lesion signal.

## Verdict and Limits

Diagnostic verdict: **DIAGNOSTIC_NEGATIVE**. G1 vs G5 is -0.2299 pp (3/3/4); G2 vs G6 is -0.2299 pp (1/6/3). Both multi-grain contrasts are negative, so the frozen decision rule classifies this as a diagnostic NO-GO for multi-grain supervision. Any future positive claim requires a new no-overlap protocol.
The inherited protocol uses test ACC for checkpoint selection, preprocessing before the outer split, and a transductive 871-node patient graph. Fold 2 explicitly trains and calibrates on two patients retained in the test set. Neither full-test metrics nor metrics with those two rows removed are unbiased independent-test estimates.
Protected source/data/history SHA audit: `PASS`. Exact RNG trace comparison to E69.3 across common epochs: PASS. Independent checkpoint audit: 60/60.

Artifacts: `fold_metrics.csv`, `summary.csv`, `paired_comparisons.csv`, `mechanism.csv`, `fold2_exposure.csv`, `checkpoint_audit.csv`, `rng_parity.csv`, and `audit_status.json`.

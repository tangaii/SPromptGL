# Experiment 72: MGDSS Results

- Audit status: `COMPLETE_AUDITED`; independently audited checkpoints: 60/60.
- Frozen base: Experiment 53.3 / CADR_WD_0p01_GFR_WD_0p11; historical ACC 92.0742%.
- Cohort: ABIDE A, 871 participants (403 ASD=0, 468 TD=1), four input modalities with 320 features.
- All six groups preserve the original CADR-GFR -> CAM -> 144-d OutputLayer path. The semantic heads are training-only.

## Source Audit

SPromptGL's audited path uses four modality feature blocks, GFR, CAM's four 36-dimensional positions, and the original OutputLayer. The eight prompts are in the source `network.py`; the original spelling `Nromal` is preserved. GFR, CAM/PLN and PEN are part of the frozen source audit; Experiment 72 changes none of them.
BrainPrompt's official `compute_label_loss()` normalizes patient and disease vectors, forms cosine similarities, and applies cross entropy directly, equivalent to temperature 1.0. This run uses BrainPrompt's exact ABIDE text with frozen CLIP RN50 embeddings, so it tests the BrainPrompt text source and supervision style, not its Llama text encoder or complete architecture.
SABER equations (8)-(10) describe semantic retrieval and gated residual injection; (11)-(13) define classification, disease-class contrastive supervision and the total objective. Only equation (12)'s class loss is adapted here. There is no claim of full SABER reproduction, and no residual injection or new graph/hypergraph path.
Experiment 59 already tested BrainPrompt-style and SABER-style global disease supervision: both reached 91.8443% ACC, below 53.3's 92.0742%. G5/G6 are a controlled revalidation with RN50, a frozen shared text span, and frozen-gradient-budget calibration, not a previously untried supervision idea.
Primary sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [SPromptGL official repository](https://github.com/wanxixi11/SPromptGL), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt official repository](https://github.com/AngusMonroe/BrainPrompt), [graph network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptg_net.py), [label network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptc_net.py), and [SABER paper](https://arxiv.org/abs/2607.01901).

## Frozen Design

For patient projection `u = normalize(P(H))` and frozen class prototypes `t_ASD,t_TD`, similarities are `s_c = u dot t_c`. BrainPrompt-style loss is `CE(s, y)` at tau=1. SABER-style adaptation is `-log(exp(s_y/tau)/sum_c exp(s_c/tau))` at the frozen tau=0.1. The losses are mathematically identical at tau=1; this experiment compares the temperature, not different classes of supervision theory.
The joint objective is `L_sem = 0.5 L_global + 0.5 L_fine`, where `L_fine = .10 L_PHENO + .10 L_ANAT + .10 L_FUNC + .70 L_CORR`. Global-only groups use `L_sem = L_global`; this is a fixed auxiliary-budget grain allocation comparison, not an unchanged global term plus an added fine term.
Fine features are the single captured CAM tensor `[871,4,36]`, in `[PHENO, ANAT, FUNC, CORR]` order, after cross-modal interaction. Global features flatten that same tensor to 144 values. The shared fine projection is `Linear(36,r)`; the global projection is `Linear(144,r)`, with `r=10`. No CAM repeat forward is used.
Disease text uses the verified Experiment 60 frozen CLIP RN50 1024-d cache (`c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`; RN50 weights SHA `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`). BrainPrompt raw-text SHA: `dbbfa807a88834a35d097aea130337c884e5264920980dd57e584c6f083f828d`; class-bank SHA: `f7105b19f5a81fb078c51636136b1e18d56fa1b204bb4a5d82a488b6f48b8176`. SPromptGL eight-text SHA: `23791903292668e54164e4d3db1b0556776f29f5fa37cc46fe7fe13c48cefa6e`; class-bank SHA: `ee741144039d60c5de8bb36576476bdc9fa67ed2ff3f7cb97a13bba08c78a55a`. CPU float64 SVD shared span rank: 10; basis SHA: `3e79ee238447ff47ad1029408b99db94c3b4c4fbc657aa5a5251754563dc3868`.
Prototype cosines are BrainPrompt-2 0.9847826 and SPromptGL-8 0.9070364; projected geometry error is within 1e-10. The class-logit gap is bounded by the prototype chord length. The possible probability range for either class under BrainPrompt-2 is 0.4565-0.5435 at tau=1 and 0.1487-0.8513 at tau=0.1. For SPromptGL-8 these ranges are 0.3938-0.6062 and 0.0132-0.9868, respectively. These are geometric bounds, not observed calibration intervals; tau was frozen before training.
At tau=1, CE and SABER-form log-sum-exp loss differed by 0, with maximum gradient difference 1.73e-18. The six groups, prompt source, temperature and grain are fixed in `config.json`.

## Preflight and Training

All P1-P20 checks: `{"P1_53_3_initialization_10_of_10": "PASS", "P2_E59_warmup_10_of_10": "PASS", "P3_auxiliary_projection_initialization_6_groups_x_10_folds": "PASS", "P4_prompt_source_and_class_order": "PASS", "P5_frozen_text_span_geometry": "PASS", "P6_brainprompt_tau1_cosine_CE": "PASS", "P7_saber_equation12_loss": "PASS", "P8_value_and_gradient_equivalence_tau1": "PASS", "P9_outer_train_labels_only": "PASS", "P10_original_logits_unchanged": "PASS", "P11_single_CAM_forward": "PASS", "P12_auxiliary_parameters_and_optimizer_separate": "PASS", "P13_gradient_budget_calibration_60_of_60": "PASS", "P14_finite_losses_and_CAM_gradients": "PASS", "P15_global_only_Pf_unused": "PASS", "P16_main_RNG_unchanged_by_auxiliary_setup": "PASS", "P17_fold1_repeat_and_resume": "PASS", "P18_original_sources_data_and_history_unchanged": "PASS", "P19_generic_prompts_no_patient_label_routing": "PASS", "P20_frozen_config_manifest": "PASS"}`. 53.3 initialization parity: 10/10 folds; E59 shared warmup parity: 10/10 folds. Fold1 two-step repeat/resume: `PASS`.
The initial unscaled semantic CAM-gradient norm was calibrated with outer-training labels only using `alpha = .10 * g_cls / g_sem`; alpha was frozen per fold/group, finite and <=10. The observed initial weighted ratios span 0.10000000 to 0.10000000. Later-epoch ratios are diagnostic only. The fitted gradient cosine can reveal conflict; see `mechanism.csv`.
Formal task count: 60/60 complete. Failed tasks: 0.

## Results

All entries below are 10-fold mean +/- population SD in percent. Fold-level rows, including TP/FP/FN/TN, best epoch and executed epochs, are in `fold_metrics.csv`.
| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 53.3 | 92.0742 +/- 2.8427 | 91.8653 +/- 2.8348 | 93.6684 +/- 3.2619 | 89.0915 +/- 3.5263 | 94.6392 +/- 3.5168 | 91.8653 +/- 2.8348 | 91.2470 +/- 3.0658 |
| G1 | 91.6144 +/- 3.0928 | 91.5115 +/- 3.0295 | 93.8176 +/- 2.9709 | 90.3171 +/- 2.8515 | 92.7058 +/- 4.4823 | 91.5115 +/- 3.0295 | 90.9301 +/- 3.1861 |
| G2 | 91.2709 +/- 2.9237 | 91.0439 +/- 2.9745 | 93.1593 +/- 3.2785 | 88.0915 +/- 4.7030 | 93.9963 +/- 3.5967 | 91.0439 +/- 2.9745 | 90.3154 +/- 3.1980 |
| G3 | 91.3845 +/- 2.8809 | 91.1449 +/- 2.8447 | 93.4602 +/- 3.2230 | 88.0854 +/- 3.8476 | 94.2044 +/- 4.4368 | 91.1449 +/- 2.8447 | 90.4675 +/- 3.0603 |
| G4 | 91.8443 +/- 3.0667 | 91.7219 +/- 2.9656 | 93.5581 +/- 2.9999 | 90.3171 +/- 2.8515 | 93.1267 +/- 5.1258 | 91.7219 +/- 2.9656 | 91.1757 +/- 3.1199 |
| G5 | 91.9579 +/- 3.0034 | 91.8010 +/- 2.9768 | 93.3402 +/- 2.9864 | 89.8232 +/- 3.9435 | 93.7789 +/- 4.4814 | 91.8010 +/- 2.9768 | 91.2038 +/- 3.1940 |
| G6 | 91.9592 +/- 2.8684 | 91.7403 +/- 2.8413 | 93.4802 +/- 2.8486 | 88.8415 +/- 3.5416 | 94.6392 +/- 3.8838 | 91.7403 +/- 2.8413 | 91.1157 +/- 3.0790 |

Paired ACC differences use left minus right in percentage points. W/T/L counts strict fold wins, ties and losses. Every fold delta is listed for reproducibility.

| Comparison | Mean delta ACC (pp) | W/T/L | Per-fold delta ACC (pp) |
|---|---:|---:|---|
| G1_vs_53.3 | -0.4598 | 0/6/4 | `[0.0,0.0,-1.14942529,0.0,-1.14942529,0.0,-1.14942529,0.0,-1.14942529,0.0]` |
| G2_vs_53.3 | -0.8033 | 0/3/7 | `[-1.13636364,-1.14942529,0.0,-1.14942529,0.0,0.0,-1.14942529,-1.14942529,-1.14942529,-1.14942529]` |
| G3_vs_53.3 | -0.6897 | 0/4/6 | `[0.0,-1.14942529,0.0,-1.14942529,-1.14942529,0.0,-1.14942529,0.0,-1.14942529,-1.14942529]` |
| G4_vs_53.3 | -0.2299 | 1/6/3 | `[0.0,-1.14942529,0.0,0.0,1.14942529,0.0,0.0,0.0,-1.14942529,-1.14942529]` |
| G5_vs_53.3 | -0.1162 | 1/7/2 | `[1.13636364,0.0,0.0,0.0,-1.14942529,0.0,-1.14942529,0.0,0.0,0.0]` |
| G6_vs_53.3 | -0.1149 | 1/7/2 | `[0.0,-1.14942529,1.14942529,0.0,0.0,0.0,0.0,0.0,-1.14942529,0.0]` |
| G1_vs_G3 | +0.2299 | 3/6/1 | `[0.0,1.14942529,-1.14942529,1.14942529,0.0,0.0,0.0,0.0,0.0,1.14942529]` |
| G2_vs_G4 | -0.5734 | 0/5/5 | `[-1.13636364,0.0,0.0,-1.14942529,-1.14942529,0.0,-1.14942529,-1.14942529,0.0,0.0]` |
| G1_vs_G2 | +0.3435 | 5/3/2 | `[1.13636364,1.14942529,-1.14942529,1.14942529,-1.14942529,0.0,0.0,1.14942529,0.0,1.14942529]` |
| G3_vs_G4 | -0.4598 | 0/7/3 | `[0.0,0.0,0.0,-1.14942529,-2.29885057,0.0,-1.14942529,0.0,0.0,0.0]` |
| G5_vs_G6 | -0.0013 | 3/4/3 | `[1.13636364,1.14942529,-1.14942529,0.0,-1.14942529,0.0,-1.14942529,0.0,1.14942529,0.0]` |
| G1_vs_G5 | -0.3435 | 0/7/3 | `[-1.13636364,0.0,-1.14942529,0.0,0.0,0.0,0.0,0.0,-1.14942529,0.0]` |
| G2_vs_G6 | -0.6883 | 0/4/6 | `[-1.13636364,0.0,-1.14942529,-1.14942529,0.0,0.0,-1.14942529,-1.14942529,0.0,-1.14942529]` |

## Mechanism

`mechanism.csv` records each group's per-fold losses, CAM gradient norms and cosines, per-position training disease-similarity accuracy/margin, and OOF held-out similarity diagnostics. A negative CAM gradient cosine means the unweighted semantic gradient points against the original classification gradient locally; frequency and per-position values are retained per fold.

| Group | Mean alpha | Initial weighted ratio | Mean CAM grad cosine | Folds with negative cosine | Mean train position result |
|---|---:|---:|---:|---:|---|
| G1 | 1.02675 | 0.1000 | +0.3861 | 0/10 | global train acc/margin 95.8%/+0.149; PHENO 79.1%/+0.084/0.653; ANAT 78.4%/+0.078/0.656; FUNC 71.0%/+0.062/0.665; CORR 95.8%/+0.147/0.623; OOF global 89.8%/+0.131; OOF CORR 88.7%/+0.124 |
| G2 | 0.09905 | 0.1000 | +0.4270 | 0/10 | global train acc/margin 95.6%/+0.133; PHENO 79.4%/+0.066/0.492; ANAT 77.5%/+0.059/0.509; FUNC 71.1%/+0.047/0.572; CORR 93.7%/+0.120/0.303; OOF global 88.5%/+0.113; OOF CORR 88.5%/+0.100 |
| G3 | 0.41151 | 0.1000 | +0.4043 | 0/10 | global train acc/margin 94.8%/+0.357; PHENO 75.6%/+0.182/0.616; ANAT 74.4%/+0.156/0.626; FUNC 71.5%/+0.146/0.632; CORR 94.0%/+0.341/0.541; OOF global 90.1%/+0.312; OOF CORR 88.9%/+0.292 |
| G4 | 0.03650 | 0.1000 | +0.4470 | 0/10 | global train acc/margin 91.5%/+0.235; PHENO 72.1%/+0.101/0.548; ANAT 72.6%/+0.091/0.561; FUNC 69.7%/+0.078/0.634; CORR 88.9%/+0.224/0.274; OOF global 89.9%/+0.209; OOF CORR 86.2%/+0.191 |
| G5 | 0.60358 | 0.1000 | +0.2991 | 0/10 | global train acc/margin 94.8%/+0.143; OOF global 86.9%/+0.120 |
| G6 | 0.05855 | 0.1000 | +0.3345 | 0/10 | global train acc/margin 95.5%/+0.136; OOF global 89.7%/+0.118 |

Training mechanism numbers are in `mechanism.csv`: each fold includes global and per-position training similarity accuracy, correct-class margin and loss, plus per-position and total CAM-gradient cosine. Test-set patient embeddings/margins are out-of-fold diagnostic measurements only and were not used for alpha, modality weights, temperature, or checkpoint selection.
The following cosine pairs are `[similarity to ASD prototype / similarity to TD prototype]`; epoch 10 is the shared warmup state and best OOF is the independently read best checkpoint. Margins are correct-class minus other-class, split by true class. Patient projection norms remain near one by construction.
| Group | E10 ASD | E10 TD | Best OOF ASD | Best OOF TD | Best margin ASD/TD | Norm/effective rank | Between/within |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 | -0.045/-0.038 | -0.051/-0.046 | +0.051/-0.073 | -0.074/+0.062 | +0.124/+0.136 | 1.000/1.36 | 2.95 |
| G2 | -0.045/-0.038 | -0.051/-0.046 | +0.008/-0.086 | -0.071/+0.059 | +0.095/+0.130 | 1.000/1.91 | 2.10 |
| G3 | +0.128/+0.088 | +0.124/+0.080 | +0.132/-0.137 | -0.207/+0.142 | +0.269/+0.349 | 1.000/1.48 | 2.70 |
| G4 | +0.128/+0.088 | +0.124/+0.080 | +0.099/-0.091 | -0.189/+0.036 | +0.191/+0.226 | 1.000/3.95 | 1.28 |
| G5 | -0.045/-0.038 | -0.051/-0.046 | +0.036/-0.068 | -0.076/+0.057 | +0.105/+0.134 | 1.000/1.48 | 2.37 |
| G6 | -0.045/-0.038 | -0.051/-0.046 | +0.034/-0.071 | -0.095/+0.035 | +0.105/+0.130 | 1.000/1.78 | 2.23 |
Across these OOF folds, mean absolute Spearman correlations of global correct margin span site=0.028-0.034, age_bins=0.022-0.029, sex=0.017-0.058, ANAT_quality=0.011-0.024, FUNC_quality_motion=0.055-0.090; the largest feature association is G4 func_mean_fd (-0.202). These are descriptive checks only.
PHENO columns encode age bins, site and sex; FUNC includes mean FD, number/percentage of FD outliers and other quality features. These associations describe possible site/quality dependence, not anatomical lesions or causal confounding. There is no anonymous-prompt control, so this experiment cannot establish that medical text semantics are indispensable; the auxiliary labels may act as ordinary additional supervision.

## Decision and Limitations

Pre-registered verdict: **NO-GO**. Performance-gate candidates: none. Multi-grain gate: FAIL (G1 vs G5: -0.3435 pp, 0/10 wins; G2 vs G6: -0.6883 pp, 0/10 wins).
Even a GO candidate supports only follow-up with multiple seeds, a strict independent validation split, and component ablations; it does not establish a MICCAI-level innovation. G3/G4 have no SPromptGL global-only counterparts, so their performance cannot identify a causal SPromptGL fine-grain contribution.
The original protocol retains historical val=test checkpoint selection, preprocessing before outer CV, and the transductive 871-node patient graph. Consequently these results are exploratory internal CV measurements, not an unbiased external generalization estimate.
Independent audit: 60/60 checkpoints passed. Protected root code, data, and historical experiment SHA comparison: `PASS` (1739 files). Historical Experiment 53.3, E59, E60, E70, E70.1 and E71 sources were read-only.
The independent reader's post-training schema adapters were corrected after its first pass exposed a pair-SHA field-path mismatch and a separate task confusion-count schema. Training code, configuration and checkpoint bytes retain their frozen hashes; both audit-script hashes and the first failed adapter attempt are recorded in `audit_status.json`.

## Artifacts

- `config.json`: frozen six-group protocol.
- `manifest.json`: immutable source/config hashes and per-task status.
- `results/fold_metrics.csv`: all ten fold metrics for each group.
- `results/summary.csv`: mean and population SD, including 53.3.
- `results/paired_comparisons.csv`: all registered paired comparisons.
- `results/mechanism.csv`: per-fold loss, gradient, modality and OOF similarity analysis.
- `results/audit_status.json`: independent checkpoint/source audit and representation/confound summaries.

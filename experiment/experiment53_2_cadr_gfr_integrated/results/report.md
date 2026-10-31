# Experiment 53.2 — CADR-GFR Results

## Execution and independent audit

- Formal group: `E1_CADR_GFR_FULL`; seed 0; folds completed: 10/10.
- Independently reloaded and audited checkpoints: 10/10.
- Each checkpoint was reconstructed from the fold's 43.1 untrained anchor and 53.2 relation cache; model/CAM weights were loaded from disk and all seven classification metrics were recomputed.
- Held-out label permutation left predictions and all disease adjacency matrices unchanged; relation-pair endpoints were outer-train only.
- All base GFR parameters used Adam `lr=0.0038, weight_decay=0.11`; CAM used the same values; added parameters used AdamW `lr=0.001`, matrix WD `1e-4`, bias/normalization WD `0`.

## Results

| 指标 | CADR-GFR mean ± population SD | 实验51 E1只读参考 | 均值差 (pp) |
|---|---:|---:|---:|
| ACC | 91.8443 ± 2.8892% | 91.9592% | -0.1149 |
| Hard-AUC | 91.6526 ± 2.8765% | 91.7434% | -0.0908 |
| Probability-AUC | 93.4205 ± 3.1701% | 93.1856% | +0.2349 |
| SPE-TD | 94.2137 ± 3.4858% | 94.6392% | -0.4255 |
| SEN-ASD | 89.0915 ± 3.3444% | 88.8476% | +0.2439 |
| BA | 91.6526 ± 2.8765% | 91.7434% | -0.0908 |
| ASD-F1 | 91.0178 ± 3.1135% | 91.0928% | -0.0750 |

### Per-fold results (%)

| Fold | Best epoch | Epochs | ACC | Hard-AUC | Probability-AUC | SPE | SEN | BA | ASD-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 114 | 316 | 95.4545 | 95.1220 | 96.0042 | 100.0000 | 90.2439 | 95.1220 | 94.8718 |
| 2 | 124 | 326 | 89.6552 | 89.5546 | 91.1983 | 91.3043 | 87.8049 | 89.5546 | 88.8889 |
| 3 | 175 | 377 | 87.3563 | 87.2481 | 89.1304 | 89.1304 | 85.3659 | 87.2481 | 86.4198 |
| 4 | 235 | 437 | 94.2529 | 93.9362 | 96.1170 | 97.8723 | 90.0000 | 93.9362 | 93.5065 |
| 5 | 229 | 431 | 93.1034 | 93.2447 | 96.8617 | 91.4894 | 95.0000 | 93.2447 | 92.6829 |
| 6 | 150 | 352 | 96.5517 | 96.4362 | 97.0745 | 97.8723 | 95.0000 | 96.4362 | 96.2025 |
| 7 | 324 | 450 | 91.9540 | 91.6223 | 94.2021 | 95.7447 | 87.5000 | 91.6223 | 90.9091 |
| 8 | 179 | 381 | 88.5057 | 88.2447 | 87.3404 | 91.4894 | 85.0000 | 88.2447 | 87.1795 |
| 9 | 202 | 404 | 89.6552 | 89.4947 | 92.7660 | 91.4894 | 87.5000 | 89.4947 | 88.6076 |
| 10 | 135 | 337 | 91.9540 | 91.6223 | 93.5106 | 95.7447 | 87.5000 | 91.6223 | 90.9091 |

Historical Experiment 51 E1 reference (read-only): ACC 91.9592%, Hard-AUC 91.7434%, Probability-AUC 93.1856%, SPE 94.6392%, SEN 88.8476%. These are paired historical fold references, not a newly trained control.

| 指标 | CADR-GFR − 实验51 E1逐折均值差 | 胜/平/负折 |
|---|---:|---:|
| ACC | -0.1149 pp | 2/5/3 |
| Hard-AUC | -0.0908 pp | 2/5/3 |
| Probability-AUC | +0.2349 pp | 5/1/4 |
| SPE-TD | -0.4255 pp | 1/6/3 |
| SEN-ASD | +0.2439 pp | 2/7/1 |

## Disease relation evidence

| 模态 | OOF AUC | OOF PR-AUC | OOF separation | Brier raw → calibrated | ECE raw → calibrated | OOF可靠度 |
|---|---:|---:|---:|---:|---:|---:|
| PHENO | 0.5055 ± 0.0033 | 0.5055 | 0.0022 | 0.2507 → 0.2500 | 0.0161 → 0.0033 | 0.000 |
| ANAT | 0.5038 ± 0.0040 | 0.5044 | 0.0013 | 0.2505 → 0.2500 | 0.0156 → 0.0053 | 0.000 |
| FUNC | 0.5141 ± 0.0071 | 0.5146 | 0.0093 | 0.2536 → 0.2499 | 0.0328 → 0.0033 | 0.000 |
| CORR | 0.6660 ± 0.0279 | 0.6391 | 0.6136 | 0.2312 → 0.2274 | 0.0519 → 0.0120 | 0.664 |

CORR mean patient-disjoint OOF AUC was 0.6660 (10/10 folds passed AUC≥0.55 and positive separation); mean count of modalities passing the frozen reliability rule was 1.00/4 per fold. Reliability used only outer-train inner-OOF data. OOF metrics are internal relation-model selection evidence, not an independent clinical cohort estimate.

The adversary was conditioned on outer-train same/different pair labels. Mean discriminator loss/accuracy were 1.0717/0.5632 (four-way chance is 0.25), and mean discriminator gradient norm was 1.020e-01. Mean adversarial encoder gradient norms were PHENO 7.888e-03, ANAT 5.430e-03, FUNC 6.325e-03, CORR 0.000e+00; CORR was detached from the adversarial path while all four scorers retained pair BCE and preservation supervision. The nonzero student gradients confirm execution, but the discriminator remains above chance and this audit has no post-joint-training patient-disjoint OOF estimate. Alignment benefit and negative transfer are therefore inconclusive, not established.

## Integrated graph propagation

| 模态 | OOF AUC | 候选边 | 有效正边 | 有效负边 | 实际正边训练精度 | 实际负边训练精度 | 原GraphLearn边 | GraphLearn梯度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PHENO | 0.5055 | 3527.5 | 0.0 | 0.0 | n/a | n/a | 82343.1 | 1.753e-01 |
| ANAT | 0.5038 | 3672.2 | 0.0 | 0.0 | n/a | n/a | 136105.5 | 2.368e-01 |
| FUNC | 0.5141 | 3566.9 | 0.0 | 0.0 | n/a | n/a | 70218.4 | 1.758e-01 |
| CORR | 0.6660 | 3331.3 | 1709.4 | 1526.2 | 0.9437 | 0.9499 | 106957.5 | 2.063e+00 |

| 模态 | 正向GCN/输入范数 | 负向GCN/输入范数 | 直接差分/输入范数 | 融合可靠度门 | 分类→评分器梯度 |
|---|---:|---:|---:|---:|---:|
| PHENO | 9.865e-07 | 0.000e+00 | 0.000e+00 | 0.0202 | 0.000e+00 |
| ANAT | 9.977e-07 | 0.000e+00 | 0.000e+00 | 0.0202 | 0.000e+00 |
| FUNC | 1.059e-06 | 0.000e+00 | 0.000e+00 | 0.0202 | 0.000e+00 |
| CORR | 6.526e-07 | 2.970e-07 | 3.002e-01 | 0.6754 | 1.417e-04 |

The four disease graphs were built separately on each modality's own GraphLearn support (with only the frozen feature-only fallback when support was sparse). Effective edge precision uses only nonzero edges actually sent to propagation. In these folds, OOF reliability admitted CORR; PHENO, ANAT, and FUNC had zero reliability and therefore zero disease-positive/negative edges. Their original GraphLearn and parameterized GCN paths still executed, while their disease-conditioned graph paths contributed no edges. Positive and negative supports were normalized separately. The original GFR's four 320-D GCN paths and bounded within-GFR neighbor-difference channels feed one 320-D representation sent to the original PEN-free CAM.
Across modality-fold diagnostics, mean positive/negative GCN message-to-input ratio was 4.991e-07; the GCN message path is therefore near zero under WD=0.11. The unscaled bounded difference channel ratio averaged 7.506e-02, while the final representation differed from raw input by 2.466e-03; the internal direct channel, especially CORR, carries most measurable propagation. Disabling all disease propagation changed held-out continuous outputs, but caused 0/871 hard-label flips. This is measurable path influence, not a diagnostic gain.

| 固定checkpoint干预 | 测试表示相对变化 | 测试log-prob最大差 | 测试预测翻转 | ACC差 (pp) | Probability-AUC差 (pp) |
|---|---:|---:|---:|---:|---:|
| direct_off | 2.433e-03 | 3.798e-02 | 0/871 (0.0000) | +0.0000 | -0.0106 |
| disease_off | 2.433e-03 | 3.798e-02 | 0/871 (0.0000) | +0.0000 | -0.0106 |
| modality_off:ANAT | 1.284e-08 | 4.411e-07 | 0/871 (0.0000) | +0.0000 | +0.0000 |
| modality_off:CORR | 2.433e-03 | 3.799e-02 | 0/871 (0.0000) | +0.0000 | -0.0106 |
| modality_off:FUNC | 1.365e-08 | 3.695e-07 | 0/871 (0.0000) | +0.0000 | +0.0000 |
| modality_off:PHENO | 1.263e-08 | 4.053e-07 | 0/871 (0.0000) | +0.0000 | +0.0000 |
| negative_off | 6.928e-04 | 1.291e-02 | 0/871 (0.0000) | +0.0000 | +0.0053 |

Interventions use the same trained checkpoint and are not additional training groups. Flip counts and rates are restricted to each outer test fold; representation and log-probability changes in this table are also test-fold quantities. A change in graph or message values does not by itself establish diagnostic value; paired metrics show whether the classifier's held-out predictions used the path.

## Source audit and interpretation

SPromptGL's local implementation has four modality-specific GraphLearn projections, projected-cosine thresholding at 0.9, symmetric `normalize_adj(A + I)`, and two parameterized GraphConv layers with ReLU after layer one. The public GFR's PHENO overwrite was not reused: Experiment 51 `LegacyFixedGFR` correctly accumulates all four full-320-D messages. The executed E1 graph objective is the mean GraphConstructLoss plus classification; the paper's label-derived adjacency objective is not silently added. The original CAM is reused and PEN is physically absent.

DeepASD adversarially aligns modality feature representations before constructing modality-specific graphs and global graph fusion. CADR-GFR instead aligns disease-conditioned patient-pair relation representations asymmetrically around CORR; it does not copy DeepASD's feature encoder, global graph fusion, or classifier. MGDR/MM-GTUNets/SMGD and the alignment caution paper motivate retaining modality-specific evidence and gated alignment, but do not validate this experiment.

Prior evidence: 43.1 established the repaired GFR initialization; 47.1/52 showed strong GCN decay can shrink learned messages; 47.3 did not establish true patient identity over strong rewiring; 48.2 reported near-chance relation learning; 53/53.1 exposed weak relation reliability and control/branch issues. CADR-GFR keeps the historical GCN WD and tests an integrated direct neighbor-difference path. Results here must be interpreted against those unresolved risks.

Initialization reconstruction used seed `510000 + fold*100 + 1`, loaded the matching 43.1 anchor tensors, captured the post-construction RNG, and initialized CADR additions inside an isolated Torch RNG scope using `532000 + fold`. The preflight compared all ten shared/CAM/RNG/split hashes with the 53.1 initialization audit and tested two shared GFR/CAM optimization steps against the disabled-CADR E1 path.

The historical protocol uses the test fold as validation for checkpoint selection, preprocessing/RFE occurred before CV, and all 871 feature nodes participate in transductive graph construction. These choices can make performance optimistic; one seed and reused patients do not establish unbiased generalization or clinical utility.

## Decision

- ENGINEERING: GO (10/10 independent reads required).
- RELATION: GO under the frozen internal OOF criterion (CORR AUC threshold 0.55); this is not external relation validation.
- GRAPH: NO-GO for disease-conditioned edges active in every modality; all four original GraphLearn branches and their finite gradients were separately audited.
- PROPAGATION: GO for a measurable test-fold continuous-output intervention and nonzero classification-to-scorer gradient; hard-label changes are reported separately.
- DIAGNOSTIC: NO-GO; requires positive mean and at least 6/10 wins for ACC or Probability-AUC, with mean SEN decline no worse than 2 pp.
- MICCAI CANDIDATE: NO-GO under this exploratory screen; external validation and independent confirmation remain necessary.

## Artifacts

- `results/fold_metrics.csv`, `results/summary.csv`, `results/paired_comparisons.csv`.
- `results/diagnostics.csv`, `results/interventions.csv`, `results/checkpoint_audit.csv`, `results/preflight.csv`.
- `manifest.json` records source, data, anchor, split, relation-cache and checkpoint hashes.

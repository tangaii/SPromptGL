# Experiment 81.6 — PRCL-RES

Execution: COMPLETE_AUDITED; 30/30 formal tasks, 30/30 best and latest checks; public RNG mismatches 0; missing scalar epochs 0.
Research: `NO_GO_PRCL_RES`; `NO_GO_PRCL_RES_BASELINE`; `PAIRWISE_INCREMENT_PRELIMINARY`; `MEDICAL_INPUT_INCREMENT_PRELIMINARY`.

Metrics are ten-fold mean ± population SD, %. ASD=0, TD=1; Hard-AUC is hard-prediction balanced accuracy; Prob-AUC is TD-probability ROC-AUC. Test ACC selects checkpoints under the retained historical protocol.

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.6-G1-Medical-Pairwise-PRCL-RES | 91.8443 ± 2.8892 | 91.7615 ± 2.8483 | 93.7207 ± 2.8486 | 90.8171 ± 4.1910 | 92.7058 ± 4.9537 | 91.1854 ± 3.0432 |
| E81.6-G2-Medical-Unary-PRCL-RES | 91.7307 ± 2.7165 | 91.5485 ± 2.7363 | 93.7382 ± 3.1883 | 89.0915 ± 4.0230 | 94.0056 ± 3.4385 | 90.8842 ± 2.9748 |
| E81.6-G3-Anonymous-Pairwise-PRCL-RES | 91.7307 ± 2.7165 | 91.5541 ± 2.6768 | 93.5426 ± 2.9733 | 89.3293 ± 4.1813 | 93.7789 ± 4.8851 | 90.9318 ± 2.8554 |
| E74.4-G2-Positive-DCR | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |

## Excluding Fold 2

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.6-G1-Medical-Pairwise-PRCL-RES | 92.0875 ± 2.9468 | 91.9772 ± 2.9238 | 93.7595 ± 3.0002 | 90.6098 ± 4.3687 | 93.3446 ± 4.8150 | 91.3824 ± 3.1467 |
| E81.6-G2-Medical-Unary-PRCL-RES | 91.8336 ± 2.8449 | 91.6493 ± 2.8667 | 93.8554 ± 3.3403 | 89.2344 ± 4.2165 | 94.0641 ± 3.6197 | 90.9825 ± 3.1203 |
| E81.6-G3-Anonymous-Pairwise-PRCL-RES | 91.8336 ± 2.8449 | 91.6555 ± 2.8033 | 93.6381 ± 3.1195 | 89.4986 ± 4.3749 | 93.8123 ± 5.1482 | 91.0354 ± 2.9920 |
| E74.4-G2-Positive-DCR | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 | 91.4703 ± 3.0580 |

## Paired comparisons

| Comparison | Scope | ACC Δ pp [descriptive 95% interval] | W/T/L | Hard-AUC Δ pp | Prob-AUC Δ pp |
|---|---|---:|---|---:|---:|
| G1_vs_DARG | all10 | -0.3448 [-1.0218, +0.3321] | 1/6/3 | -0.2679 | +0.2261 |
| G1_vs_DARG | without_fold2 | -0.1277 [-0.6586, +0.4032] | 1/6/2 | -0.0709 | +0.1452 |
| G1_vs_G2 | all10 | +0.1136 [-0.3515, +0.5787] | 2/7/1 | +0.2129 | -0.0174 |
| G1_vs_G2 | without_fold2 | +0.2540 [-0.1334, +0.6414] | 2/7/0 | +0.3279 | -0.0959 |
| G1_vs_G3 | all10 | +0.1136 [-0.3515, +0.5787] | 2/7/1 | +0.2074 | +0.1782 |
| G1_vs_G3 | without_fold2 | +0.2540 [-0.1334, +0.6414] | 2/7/0 | +0.3217 | +0.1214 |

All six per-fold differences are in paired_comparisons.csv. Intervals are descriptive across folds with overlapping training populations, not independent-patient confidence inference.

## Backbone and residual decomposition

| Group/scope | DARG → current off corrected/harmed/net | Current off → full corrected/harmed/net | DARG → full corrected/harmed/net |
|---|---|---|---|
| E81.6-G1-Medical-Pairwise-PRCL-RES/all10 | 8/13/-5 | 2/0/2 | 9/12/-3 |
| E81.6-G1-Medical-Pairwise-PRCL-RES/without_fold2 | 7/10/-3 | 2/0/2 | 8/9/-1 |
| E81.6-G2-Medical-Unary-PRCL-RES/all10 | 3/12/-9 | 5/0/5 | 4/8/-4 |
| E81.6-G2-Medical-Unary-PRCL-RES/without_fold2 | 3/10/-7 | 4/0/4 | 4/7/-3 |
| E81.6-G3-Anonymous-Pairwise-PRCL-RES/all10 | 1/13/-12 | 9/1/8 | 1/5/-4 |
| E81.6-G3-Anonymous-Pairwise-PRCL-RES/without_fold2 | 1/12/-11 | 9/1/8 | 1/4/-3 |

Residual-off performance is evaluated at checkpoints selected using the full model. It is a frozen counterfactual with changed classifier inputs, not an independently trained or optimally selected DARG baseline.

| Current residual-off path | ACC | Hard-AUC | Prob-AUC | SEN | SPE | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.6-G1-Medical-Pairwise-PRCL-RES | 91.6144 ± 2.8249 | 91.5487 ± 2.8037 | 93.6888 ± 2.8615 | 90.8171 ± 4.1910 | 92.2803 ± 4.6660 | 90.9493 ± 3.0025 |
| E81.6-G2-Medical-Unary-PRCL-RES | 91.1560 ± 3.1773 | 90.9910 ± 3.1648 | 93.7754 ± 3.1787 | 88.8415 ± 3.8785 | 93.1406 ± 4.0882 | 90.3099 ± 3.3798 |
| E81.6-G3-Anonymous-Pairwise-PRCL-RES | 90.8111 ± 3.1340 | 90.6954 ± 3.0760 | 93.6008 ± 2.9407 | 89.3232 ± 4.4860 | 92.0675 ± 5.6610 | 90.0323 ± 3.2160 |

## Prompt, competition and residual diagnostics

| Group | Gamma | Prompt attention pair MAE | Normal weight | Competition entropy | Residual mean/P50/P95/max % | Near 9.9% cap |
|---|---:|---:|---:|---:|---|---:|
| E81.6-G1-Medical-Pairwise-PRCL-RES | 0.506224 | 0.0015892 | 0.459927 | 0.699110 | 9.0872/9.1526/9.4565/9.9450 | 1.26% |
| E81.6-G2-Medical-Unary-PRCL-RES | 0.498371 | 0.0013873 | 0.474036 | 0.565403 | 8.7854/8.8628/9.2470/9.9470 | 1.61% |
| E81.6-G3-Anonymous-Pairwise-PRCL-RES | 0.501328 | 0.0017333 | 0.461008 | 0.693928 | 8.8170/8.8982/9.3159/9.8242 | 0.00% |

Residual means are fold patient averages; P50/P95 columns average the fold quantiles, and max is the largest patient value. Per-patient raw norms, token statistics and all gradient summaries are reduced to scalars in mechanism.csv.

| Frozen G1 intervention | Probability MAE | Prediction flips | ACC Δ pp | BA Δ pp | Corrected/harmed relative to full | Attention MAE |
|---|---:|---:|---:|---:|---|---:|
| null | 0.00836833 | 2 | -0.2299 | -0.2128 | 0/2 | 0.11899139 |
| global_off | 0.00156521 | 1 | -0.1149 | -0.1220 | 0/1 | 0.00000000 |
| prompt_off | 0.00016596 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00407382 |
| anonymous | 0.00015411 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00376528 |
| swap_groups | 0.00004931 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00176933 |
| unary | 0.00882953 | 3 | -0.3448 | -0.3191 | 0/3 | 0.04809156 |
| delete_prompt_0 | 0.00002915 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00054196 |
| delete_prompt_1 | 0.00003239 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00056874 |
| delete_prompt_2 | 0.00002956 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00053684 |
| delete_prompt_3 | 0.00003183 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00057748 |
| delete_prompt_4 | 0.00002994 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00050382 |
| delete_prompt_5 | 0.00002855 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00049386 |
| delete_prompt_6 | 0.00002619 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00044073 |
| delete_prompt_7 | 0.00002377 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00041040 |
| swap_prompt_0 | 0.00000792 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00027463 |
| swap_prompt_1 | 0.00001341 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00044583 |
| swap_prompt_2 | 0.00001183 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00038885 |
| swap_prompt_3 | 0.00001971 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00066003 |
| swap_prompt_4 | 0.00000792 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00027463 |
| swap_prompt_5 | 0.00001341 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00044583 |
| swap_prompt_6 | 0.00001183 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00038885 |
| swap_prompt_7 | 0.00001971 | 0 | +0.0000 | +0.0000 | 0/0 | 0.00066003 |

These use frozen weights without training or reselection. Global-off substitutes uniform 1/9 coordinate matching while keeping relation messages and d identical. Text replacement also changes input geometry; inference sensitivity alone does not establish correct medical semantics.

| G1 prompt | Attention entropy | Null weight | Patient attention SD |
|---|---:|---:|---:|
| 1. Nromal brain volume | 0.834069 | 0.047735 | 0.0450046 |
| 2. Uniform cortical thickness | 0.833676 | 0.049282 | 0.0450090 |
| 3. Symmetrical brain structure | 0.833545 | 0.048212 | 0.0450944 |
| 4. Normal ventricular morphology | 0.834425 | 0.048486 | 0.0449934 |
| 5. Altered brain volume | 0.834923 | 0.047461 | 0.0448797 |
| 6. Abnormal functional connectivity | 0.834726 | 0.048917 | 0.0449042 |
| 7. White matter integrity issues | 0.834988 | 0.046820 | 0.0449459 |
| 8. Different activation patterns | 0.835504 | 0.047638 | 0.0448175 |

Top latent relation positions and the complete per-prompt difference matrix are retained in mechanism.csv. These latent positions have no validated anatomical interpretation. Normal/abnormal text groups are not patient truth labels.

## Fixed input geometry

medical: off-diagonal cosine mean 0.769207, SD 0.065018, range [0.637009, 0.909524].

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00000 | 0.76974 | 0.84018 | 0.70010 | 0.90952 | 0.81368 | 0.80995 | 0.77561 |
| 2 | 0.76974 | 1.00000 | 0.79874 | 0.73086 | 0.73167 | 0.77186 | 0.73345 | 0.72054 |
| 3 | 0.84018 | 0.79874 | 1.00000 | 0.72256 | 0.81841 | 0.82894 | 0.81799 | 0.80777 |
| 4 | 0.70010 | 0.73086 | 0.72256 | 1.00000 | 0.63987 | 0.68064 | 0.63701 | 0.64870 |
| 5 | 0.90952 | 0.73167 | 0.81841 | 0.63987 | 1.00000 | 0.78275 | 0.82545 | 0.81534 |
| 6 | 0.81368 | 0.77186 | 0.82894 | 0.68064 | 0.78275 | 1.00000 | 0.78804 | 0.81774 |
| 7 | 0.80995 | 0.73345 | 0.81799 | 0.63701 | 0.82545 | 0.78804 | 1.00000 | 0.80069 |
| 8 | 0.77561 | 0.72054 | 0.80777 | 0.64870 | 0.81534 | 0.81774 | 0.80069 | 1.00000 |

anonymous: off-diagonal cosine mean 0.007216, SD 0.042060, range [-0.076038, 0.110780].

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00000 | 0.04181 | 0.11078 | 0.02620 | 0.01756 | 0.05064 | -0.01213 | -0.00409 |
| 2 | 0.04181 | 1.00000 | -0.06834 | -0.05444 | 0.00420 | 0.03162 | 0.01184 | -0.02349 |
| 3 | 0.11078 | -0.06834 | 1.00000 | 0.02946 | 0.04300 | 0.00239 | -0.07604 | -0.07356 |
| 4 | 0.02620 | -0.05444 | 0.02946 | 1.00000 | 0.05628 | -0.01096 | -0.00947 | 0.04625 |
| 5 | 0.01756 | 0.00420 | 0.04300 | 0.05628 | 1.00000 | -0.02576 | 0.03588 | 0.00032 |
| 6 | 0.05064 | 0.03162 | 0.00239 | -0.01096 | -0.02576 | 1.00000 | 0.04335 | -0.01300 |
| 7 | -0.01213 | 0.01184 | -0.07604 | -0.00947 | 0.03588 | 0.04335 | 1.00000 | 0.02174 |
| 8 | -0.00409 | -0.02349 | -0.07356 | 0.04625 | 0.00032 | -0.01300 | 0.02174 | 1.00000 |

The anonymous Gaussian vectors match raw per-item L2 norm, not the medical Gram matrix. Their different geometry, a single anonymous draw and lack of per-prompt patient annotations limit semantic attribution.

## Training, numerical stability and audit

E81.6-G1-Medical-Pairwise-PRCL-RES: mean routed/new-to-total CAM gradient ratio 0.0248726; weighted residual regularizer mean 6.40606e-06; mean residual-reg/classification ratio 2.99102e-05; classification-only trained text gradient norms are in mechanism.csv.
E81.6-G2-Medical-Unary-PRCL-RES: mean routed/new-to-total CAM gradient ratio 0.0201514; weighted residual regularizer mean 5.90785e-06; mean residual-reg/classification ratio 2.93845e-05; classification-only trained text gradient norms are in mechanism.csv.
E81.6-G3-Anonymous-Pairwise-PRCL-RES: mean routed/new-to-total CAM gradient ratio 0.0215574; weighted residual regularizer mean 6.13808e-06; mean residual-reg/classification ratio 2.91889e-05; classification-only trained text gradient norms are in mechanism.csv.
Scheduler: 30/30 tasks, wall 328.4s, maximum 10 workers; failures 0. One measured worker is completed before increasing concurrency. Original three optimizer steps precede the PRCL optimizer.
Thirty independent best reloads recompute six metrics and saved predictions; all latest states retain four optimizer moments, full RNG, next epoch, early stopping and complete scalar journals. Ten-fold post-training zero-residual replay and serialized two-step fresh-model recovery pass. No historical source/result mutation is accepted; source_map is checked again at audit end.

## Direct answers

Q1: Prompt participation is measured by independent Query/message differences and frozen prompt interventions above; numerical prompt collapse flag=False. Computational participation does not itself imply diagnostic value.
Q2: Pairwise vs Unary: PAIRWISE_INCREMENT_PRELIMINARY; use the independent G1−G2 ten-/nine-fold comparisons.
Q3: Actual medical input: MEDICAL_INPUT_INCREMENT_PRELIMINARY; G1−G3 and input geometry both matter. This experiment cannot alone establish clinical semantic understanding.
Q4: RES frozen inference effect=True; its probability/accuracy changes are in the global-off table. This is not an independently retrained RES ablation.
Q5: Joint ACC and BA baseline criterion: NO_GO_PRCL_RES_BASELINE; nine-fold results are explicitly shown above.
Q6: Full G1 relative to historical DARG: net -3, excluding Fold 2 -1; distinguish this from correction of current residual-off errors.

## Provenance and limitations

[SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) and [official code](https://github.com/wanxixi11/SPromptGL) were reviewed before implementation. KD-Brain borrows only semantic Query conditioning from [Attention_AggregateLayer](https://github.com/LiuSy0731/KDBrain/blob/main/KDBrain/models/model.py); ItemizedCLIP borrows independent text conditioning from its [model code](https://github.com/MLNeurosurg/ItemizedCLIP/blob/main/Radiology_Tasks/src/open_mri/my_model.py), without patient-item supervision losses. The requested CVF page returned 403; the official repository-linked [ItemizedCLIP arXiv paper](https://arxiv.org/abs/2512.11141) supplied the primary paper fallback. Exact source identities and differences are in runtime/source_review.json and reviewed_sources.json.
Historical E81.4-G2 is context, not a matched no-Prompt ablation. E81.5 unbounded fusion was NO_GO and is not reused. This study keeps seed0, test=val checkpoint selection, pre-CV preprocessing/RFE, transductive patient graphs and the Fold2 overlap. Blocking only new training residuals does not repair the original leakage or indirect graph effects. Single-seed overlapping CV is preliminary and does not establish independent medical generalization or a MICCAI-level validated innovation.
Frozen source/config/dependency SHA: `a7af6fe23308852acc67c730d45b23dcd13ebc135f23dbc50b178c297500e79c`. Prompt cache SHA: `594615814fc337941cf38467f27e4ff9b56baac49ef2842e3f8f1555aebd0721`. The manifest retains every file identity; no post-result hyperparameter adjustment is permitted.

## Execution history retained

Before the accepted source freeze, one worker stopped after one unsaved optimizer update because metric logging passed CUDA indices to a NumPy-index helper. No checkpoint or task ledger was written. The logger was corrected to use the historical train_np indices; the model, loss and hyperparameters were unchanged. The original source, preflight, manifest and failure log remain in runtime/failures/attempt01, and all ten-fold preflight checks passed after the correction.
The preferred OSSFS directory then returned errno=5 during atomic log/checkpoint fsync or replace after 25 task checkpoint pairs had completed. The five interrupted tasks and their error logs were retained under runtime/failures/io_relocation_eio; all 25 completed best/latest pairs were SHA-verified during migration to /model/spromptgl/experiment/experiment81_6_prcl_res. Two fold-9 tasks and G3 fold 10 resumed from complete latest states. G1 and G2 fold 10 had no usable latest state and restarted from their original E61 epoch-10 warmups; G1's orphan best and scalar journal were archived. No completed task was retrained, and the accepted training source/configuration SHA remained a7af6fe23308852acc67c730d45b23dcd13ebc135f23dbc50b178c297500e79c.
The final five tasks completed in 328.4s with at most two recovery workers. Summed task worker elapsed time across all 30 ledgers is 30447.4s; this sum includes overlapping workers and is not wall-clock duration. The initial 25-task wall time is not reconstructed as a single uninterrupted scheduler interval.

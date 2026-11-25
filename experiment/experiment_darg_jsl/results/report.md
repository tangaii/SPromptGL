# DARG + JSL Formal Audit

Status: `COMPLETE_AUDITED`.

## PERFORMANCE

The primary results are fold-wise mean +/- population SD in percent. Hard-AUC is balanced accuracy computed from hard predictions. Prob-AUC uses continuous ASD probabilities.

| Fold | Correct | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84/88 | 95.45 | 95.12 | 96.63 | 90.24 | 100.00 | 94.87 |
| 2 | 83/87 | 95.40 | 95.25 | 97.24 | 92.68 | 97.83 | 95.00 |
| 3 | 78/87 | 89.66 | 89.55 | 91.36 | 87.80 | 91.30 | 88.89 |
| 4 | 83/87 | 95.40 | 95.19 | 99.15 | 92.50 | 97.87 | 94.87 |
| 5 | 80/87 | 91.95 | 92.18 | 95.90 | 95.00 | 89.36 | 91.57 |
| 6 | 84/87 | 96.55 | 96.44 | 97.77 | 95.00 | 97.87 | 96.20 |
| 7 | 78/87 | 89.66 | 89.31 | 92.77 | 85.00 | 93.62 | 88.31 |
| 8 | 79/87 | 90.80 | 90.74 | 89.68 | 90.00 | 91.49 | 90.00 |
| 9 | 78/87 | 89.66 | 89.12 | 92.29 | 82.50 | 95.74 | 88.00 |
| 10 | 81/87 | 93.10 | 93.06 | 94.73 | 92.50 | 93.62 | 92.50 |
| Mean +/- population SD | 808/871 | 92.76 +/- 2.63 | 92.60 +/- 2.66 | 94.75 +/- 2.94 | 90.32 +/- 3.93 | 94.87 +/- 3.34 | 92.02 +/- 2.95 |

Against the supplied historical ten-fold summaries, JSL minus DARG is +0.5747 pp ACC, +0.5674 pp BA, +1.2568 pp Prob-AUC, +0.2561 pp SEN, +0.8788 pp SPE, and +0.5623 pp ASD-F1. JSL minus DPSL is -0.3449 pp ACC, -0.4380 pp BA, -0.1114 pp Prob-AUC, -1.7317 pp SEN, +0.8557 pp SPE, and -0.4883 pp ASD-F1. The DARG comparison is not a causal JSL estimate because DARG has two historical train-test overlaps and this run has 19.

`paired_comparisons.csv` also reports metrics recalculated from matched patient predictions. Across all 871 patients, JSL has 808 correct, DPSL 811, and DARG 803. JSL versus DPSL changes 9 patients from wrong to right and 12 from right to wrong. On the 19 overlap patients, JSL is correct on 13, DPSL on 16, and DARG on 4. On the other 852 patients, JSL and DPSL each have 795 correct, with 8 corrections and 8 harms in opposite directions; DARG has 799 correct. Excluding folds 2, 3, 4, and 8 leaves 523 patients, with 485 JSL, 487 DPSL, and 488 DARG correct. Pooled patient AUCs in that file are descriptive across folds; the fold-wise macro metrics above are primary.

## MECHANISM

Residual-off to Full at identical Best weights corrects 1 patient and harms 0, increasing fold-macro ACC by 0.1149 pp, Hard-AUC/BA by 0.1087 pp, Prob-AUC by 0.0212 pp, SEN by 0.0000 pp, SPE by 0.2174 pp, and ASD-F1 by 0.1084 pp. The direct residual changes 1 of 871 hard predictions. Spectral-in-only and spectral-out-only each produce zero hard prediction flips. Their fold-macro Prob-AUC changes relative to Residual-off are -0.0053 pp and +0.0053 pp respectively.

Across test patients, mean CAM residual fraction is 1.7788%; mean per-fold maximum is 3.0812%, and the largest observed value is 7.4855%, below the 8% cap. All 19 overlap patients have exactly zero parallel, complement, and joint residual. Both Jacobian branches have nonzero gradient norms in every fold. The added shared-parameter gradient reaches the 0.1% budget within floating-point precision. From Residual-off, the mean true-class margin change is -0.0386 for 64 originally incorrect patients and +0.0516 for 807 originally correct patients.

The co-trained Residual-off DARG is correct on 807/871 patients compared with 803/871 for the historical DARG reference. This is descriptive shared-parameter adaptation under joint DARG/JSL training, not an isolated causal estimate. The one-patient Residual-off-to-Full change measures the direct residual effect at fixed trained weights.

## ABLATION_VALIDITY

The model retains GFR, DARG, CAM144, the historical classifier, and the frozen Original O basis U in R^(144x16), with a 16-dimensional spectral space and its 128-dimensional orthogonal complement. JSL uses the current classifier margin, its CAM Jacobian, spectral geometry, and a compact spectral controller. DSE prompts, CLIP queries, cross-attention, Gaussian/EMA evidence, and DSE losses are absent from the active path.

The 19 historical overlap patients remain in DARG training and relation eligibility but are excluded from JSL labels and class statistics. Their added residual is zero. The primary objective tests JSL without DSE. The protocol keeps validation-equals-test checkpoint selection, pre-CV feature processing, a transductive patient graph, and the intentional train-test overlap, so these results are not independent generalization evidence.

The first completed attempt was marked non-authoritative because its source manifest omitted `code/jsl_model.py`. Its checkpoints and outputs remain in `archive/attempt_unsealed`. The active module hash was added before the authoritative re-preflight and retraining. All 20 final Best/Latest state payload hashes match the preserved attempt exactly, confirming that the seal correction did not change the trained trajectory or weights.

## CONTROLLER_AND_OBJECTIVES

For CAM vector `c`, the fixed basis `U`, center `mu`, and eigenvalues `lambda` define whitened coordinates `z=(c-mu)U` and `w=z/sqrt(lambda+1e-8)`. Each of the 16 tokens encodes `[w_j, |w_j|, log(1+w_j^2)]`. Softmax token scores produce a weighted spectral summary. The controller also receives the current margin `m=logit_ASD-logit_TD`, `tanh(m)`, `log(1+|m|)`, log-scaled in/out Jacobian norms, and mean/absolute whitened coordinates.

The frozen classifier Jacobian is `J=partial m / partial c`. Its two components are `J_in=UU^T J` and `J_out=J-J_in`. The controller receives `[tanh(m), log(1+|m|), log(1+0.08*||J_in||*||c||), log(1+0.08*||J_out||*||c||), mean(w), mean(|w|)]` alongside the spectral summary. It outputs `shift=0.75*tanh(s0)`, `trust=sigmoid(s1)`, and `allocation=sigmoid(s2)`. With `desired=trust*shift`, the raw corrections are `raw_in=desired*allocation*J_in/(||J_in||^2+1e-4)` and `raw_out=desired*(1-allocation)*J_out/(||J_out||^2+1e-4)`. Their sum is scaled by `tanh(||raw||/(0.08||c||))/(||raw||/(0.08||c||))`, with the zero-ratio limit set to one. The resulting `delta` satisfies `||delta|| <= 0.08||c||`, and `CAM_enhanced=c+delta`. Overlap patients are masked before the norm bound.

On legal JSL training members only, let `q_i=1-2y_i`, `m0_i` be the native margin, `m1_i` the full margin, and `g_i=q_i(m1_i-m0_i)`. `CE1` and `CE0` are class-balanced mean cross-entropies on full and native CAM. The JSL objective is

`L_JSL = 1.0 CE1 + 0.5 CE0 + 0.25 (CE1-CE0) + 0.002 L_gain + 0.001 L_protect + 0.00025 L_in + 0.00025 L_out + 0.0005 R2`.

`L_gain` is the class-balanced mean of `softplus(-g_i/0.5)` over natively incorrect patients (`q_i*m0_i <= 0`). `L_protect` is the class-balanced mean of `relu(q_i*m0_i-q_i*m1_i)` over natively correct patients. For candidates formed by adding each branch separately to native CAM, branch gains are `g_in=q_i(m_in_i-m0_i)` and `g_out=q_i(m_out_i-m0_i)`. `L_in` and `L_out` are the respective legal-member means of `softplus(-g_in/0.5)` and `softplus(-g_out/0.5)`. `R2` is the legal-member mean of `(||delta_i||/||c_i||)^2`. The `0.5` values are the frozen margin temperature.

The historical DARG loss and its gradients remain active. For shared parameters, the applied gradient is `grad_DARG + alpha*grad_JSL_shared`, where `alpha=min(1, 0.001||grad_DARG||/max(||grad_JSL_shared||,1e-12))`. JSL-specific parameters receive the unscaled JSL gradient and use a separate AdamW optimizer with learning rate `0.0005` and weight decay `0.01`.

## AUDIT_STATUS

Preflight and post-training audit both pass. The audit verifies 10/10 Best forwards, 10/10 Latest reloads, 10/10 actual next-step training and evaluation replays, all comparable archived RNG stages, six historical Dropout calls per training forward, 10/10 Jacobian orthogonality/reconstruction, both branch gradients, the 8% residual cap, the 0.1% shared-gradient cap, zero overlap residuals, and DSE independence.

The original DPSL frozen set remained at 3,949 files with identical before/after SHA manifests. The author-format `result_ABIDE.txt` was emitted from this run's per-epoch journal and Best predictions. Its format check passes and its SHA-256 is `83cda6dd5dfd0f869b19af51e825718539ed4ff60c8e722ea6084ab54b040ce0`. Its duplicated `trn-loss-MF` fields represent classifier CE twice and the actual GFR alignment loss once.

Detailed paired predictions, per-fold mechanism metrics, RNG records, recovery witnesses, and audit booleans are in the adjacent CSV and JSON files. The status is `COMPLETE_AUDITED` for this ablation under the stated historical protocol, not a claim of generalization improvement.

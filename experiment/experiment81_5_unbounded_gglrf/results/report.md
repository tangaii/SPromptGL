# E81.5 — Unbounded Global-Guided Local Relational Feature Fusion

Independent audit: **PASS**. Research decision: **NO_GO_UNBOUNDED**.

## Source, single-variable contrast and training identity

Reviewed the [original paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [official source](https://github.com/wanxixi11/SPromptGL) and [published E81.4 commit](https://github.com/tangaii/SPromptGL/commit/3e599ea9a2cdd3c90b6d4e5ce557e8031922d274). All 13 local E81.4 published files match that commit byte for byte. GFR learns cosine patient graphs and DGCN propagation; CAM combines cross-modal messages. PEN SPGS uses frozen CLIP prompt similarities; RES equation6 adds weighted locals to global features. The official implementation uses fixed local reshaping and feature-coordinate softmax, which differs from the written RES abstraction. Neither the paper nor its RES defines the E81.4-specific 10% cap. PEN/CLIP/prompts remain excluded.

The sole formal architectural change is delta=raw_delta, replacing the smooth 10% norm mapping. No tanh, gate, parameter, new classifier or new token layout is introduced. The same 16,384 parameters, private CPU salt814000+fold and exact same-fold initial state_dict are verified. Global/local encoders, 8 heads, 15 nonself real neighbors, zero-value Null, Pairwise products and signed differences, global beta, selected=16*beta*m_local and shared Linear(32,9) remain identical. All ten historical trained checkpoints reproduce exactly in local capped mode, including attention/beta/raw/delta/CAM/logp/hidden/loss.

Each formal task starts independently from E61 epoch10 through audited E74.4-G2 prepare(fold), restoring full model, three optimizers, RNG, split and 4096 pairs. The fourth AdamW(.0005,.01) steps last. Classification uses the original LayerNorm144→Linear144,36→GELU→Dropout.35→Linear36,2→log_softmax once. Loss is historical_out.total + .001*L_res; L_res=mean_legal_train((norm(delta)/(norm(CAM.detach)+eps))²). There is no coefficient search or adjustment.

Eta=.1 scales only the new relation-input path; the direct identity path remains unscaled and the norm denominator is detached. Zero-head gradients/updates match DARG exactly. After learning a residual, classifier inputs and parameter trajectories change; common RNG does not imply common optimization. Regularizer-gradient telemetry uses autograd.grad without parameter.grad accumulation or RNG consumption, followed by one total backward.

The E81.4 training aggregate SHA is 7dd23d91341597d6eae1f7ae648c718279aed9f59b0cdd5021982dd4d5ae5b68; its corrected audit aggregate is 0fa19058f70a75838e62b43599820d1ac763827a001fd33754259887496ee86c. The historical audit-only dict/Logp fix is preserved separately and no historical files or checkpoints are changed.

## Six metrics

Percent, ten-fold mean ± population SD; Hard-AUC=balanced accuracy of hard labels; Prob-AUC=probability ROC-AUC; ASD=0, TD=1.

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2-Positive-DCR | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | 92.1891±2.6675 | 92.0054±2.6198 | 93.5688±2.5794 | 89.8110±3.8125 | 94.1998±4.7671 | 91.4385±2.7824 |
| E81.5-G1-Pairwise-Unbounded-GGLRF | 90.8085±2.7763 | 90.6151±2.8087 | 93.0611±2.8992 | 87.8537±5.5514 | 93.3765±4.5119 | 89.8166±3.1014 |
| E81.3-G2-PairwiseRelation-E2E | 91.7294±2.9505 | 91.5401±2.9690 | 93.4772±2.8732 | 89.0793±4.0349 | 94.0009±3.4622 | 90.8841±3.1806 |
| E81.5-G1-Pairwise-Unbounded-GGLRF_ResidualOff | 74.8472±13.3416 | 75.4717±12.3987 | 86.7624±11.2310 | 84.3476±11.4936 | 66.5957±28.2593 | 76.6528±9.2796 |

## Paired fold changes

All differences are percentage points. The paired t95% intervals are descriptive only: cross-validation training sets overlap. E81.3 is background context, not the single-variable comparator.

| Comparison | Metric | Mean Δ | Descriptive95% interval | W/T/L |
|---|---|---:|---:|---:|
| E815_vs_E814_capped | acc | -1.3806 | [-2.6530, -0.1083] | 1/3/6 |
| E815_vs_E814_capped | hard_auc | -1.3903 | [-2.7233, -0.0573] | 3/0/7 |
| E815_vs_E814_capped | prob_auc | -0.5078 | [-1.6920, +0.6765] | 5/0/5 |
| E815_vs_E814_capped | sen | -1.9573 | [-6.5634, +2.6488] | 3/3/4 |
| E815_vs_E814_capped | spe | -0.8233 | [-4.5989, +2.9523] | 3/1/6 |
| E815_vs_E814_capped | asd_f1 | -1.6219 | [-3.0721, -0.1718] | 3/0/7 |
| E815_vs_DARG | acc | -1.3806 | [-2.7107, -0.0505] | 2/1/7 |
| E815_vs_DARG | hard_auc | -1.4143 | [-2.7991, -0.0295] | 2/1/7 |
| E815_vs_DARG | prob_auc | -0.4335 | [-1.4713, +0.6043] | 4/0/6 |
| E815_vs_DARG | sen | -2.2134 | [-6.4844, +2.0576] | 3/3/4 |
| E815_vs_DARG | spe | -0.6152 | [-4.0901, +2.8597] | 3/2/5 |
| E815_vs_DARG | asd_f1 | -1.6424 | [-3.1683, -0.1166] | 2/1/7 |
| E815_vs_E813_context | acc | -0.9208 | [-2.1935, +0.3518] | 1/5/4 |
| E815_vs_E813_context | hard_auc | -0.9250 | [-2.1884, +0.3384] | 3/1/6 |
| E815_vs_E813_context | prob_auc | -0.4161 | [-1.4594, +0.6271] | 5/1/4 |
| E815_vs_E813_context | sen | -1.2256 | [-4.7124, +2.2612] | 3/3/4 |
| E815_vs_E813_context | spe | -0.6244 | [-4.0035, +2.7546] | 4/2/4 |
| E815_vs_E813_context | asd_f1 | -1.0675 | [-2.3966, +0.2616] | 3/1/6 |

Per-fold differences for all metrics and both scopes are stored in paired_comparisons.csv.

## Excluding Fold2

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2-Positive-DCR | 92.2153±2.9131 | 92.0481±2.8723 | 93.6142±3.0541 | 90.0474±4.0959 | 94.0487±4.9847 | 91.4703±3.0580 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | 92.2153±2.8105 | 92.0067±2.7615 | 93.6202±2.7141 | 89.4919±3.8899 | 94.5215±4.9209 | 91.4244±2.9326 |
| E81.5-G1-Pairwise-Unbounded-GGLRF | 90.9367±2.8983 | 90.7624±2.9237 | 93.0324±3.0546 | 88.4011±5.5897 | 93.1237±4.6882 | 89.9838±3.2262 |
| E81.5-G1-Pairwise-Unbounded-GGLRF_ResidualOff | 77.2887±11.7545 | 77.6980±11.0111 | 87.4949±11.6096 | 82.6084±10.7951 | 72.7875±22.4485 | 77.7624±9.1306 |

## Residual size and classification effect

No theoretical10% bound remains; actual learned size is measured rather than assumed. Ratio statistics below average fold-specific test-patient summaries; the reported overall maximum is the maximum over folds.

| Statistic | E81.5 |
|---|---:|
| delta_ratio_mean | 1.618398 |
| delta_ratio_p50 | 1.4259072 |
| delta_ratio_p95 | 2.9368496 |
| above10_fraction | 1 |
| above20_fraction | 1 |
| above50_fraction | 0.97126437 |
| above100_fraction | 0.76206896 |
| raw_delta_norm_mean | 7.0604743 |
| delta_norm_mean | 7.0604743 |
| cam_norm_mean | 4.5260289 |
| enhanced_norm_mean | 8.5461709 |
| residual_hidden_mae | 0.19288886 |
| residual_probability_mae | 0.25055365 |
| beta_entropy | 0.90531693 |
| beta_max_mean | 0.2025771 |
| beta_patient_sd_mean | 0.027551991 |
| attention_entropy | 0.83766551 |
| null_weight | 0.044056808 |
| message_norm_mean | 4.7003446 |
| global_query_norm | 5.8681775 |
| global_off_beta_mae | 0.03047492 |
| global_off_attention_mae | 0.02897081 |
| global_off_delta_mae | 0.23274167 |
| delta_ratio_overall_max | 6.4792962 |

Historical capped E81.4-G2: mean9.8224%, median9.8814%, P95 9.9699%, and61.37% at or above9.9%. The ratio distribution is not an anatomical or medical-confidence measure.

## Historical backbone → current off → full

| Transition | Corrected | Harmed | Net | Net without Fold2 |
|---|---:|---:|---:|---:|
| DARG → current residual-off | 19 | 170 | -151 | -117 |
| current residual-off → full | 163 | 24 | +139 | +107 |
| DARG → full | 13 | 25 | -12 | -10 |

Counts are held-out fold predictions. Current-off uses the checkpoint selected for full-model ACC; it is not a separately trained or independently selected best baseline. E81.4-G2 previously showed −15/+15/0 net corrections across these three stages.

## Frozen historical E81.4 weights: remove cap at inference

These interventions never train, select checkpoints, or adjust formal hyperparameters. Raw_delta, attention and beta are identical between capped/unbounded at the same frozen weights. This differs from the formal independent re-training contrast.

| Frozen mode | ACC | BA | Prob-AUC | SEN | SPE | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| capped | 92.1891±2.6675 | 92.0054±2.6198 | 93.5688±2.5794 | 89.8110±3.8125 | 94.1998±4.7671 | 91.4385±2.7824 |
| unbounded | 91.2696±3.4238 | 91.0650±3.4060 | 92.9196±2.8686 | 88.5732±4.6618 | 93.5569±5.1333 | 90.3974±3.6759 |

Frozen removal: mean ratio 156.2786%; probability MAE 0.056517; flips 8; corrected/harmed 0/8; ΔACC -0.9195pp; ΔBA -0.9404pp. All frozen outputs finite: True.

## Regularizer, gradients and stability

| Scalar (fold mean) | Value |
|---|---:|
| train_res_mean | 2.8193579 |
| train_res_max | 18.156322 |
| train_weighted_res_mean | 0.0028193579 |
| train_weighted_res_max | 0.018156322 |
| train_weighted_res_to_cls_mean | 0.011676414 |
| train_weighted_res_to_cls_max | 0.079445793 |
| train_residual_regularizer_weighted_grad_mean | 0.033846722 |
| train_regularizer_to_gglrf_grad_mean | 0.092875652 |
| train_global_encoder_grad_mean | 0.6293009 |
| train_local_encoder_grad_mean | 0.29911618 |
| train_key_grad_mean | 0.1216859 |
| train_value_grad_mean | 0.14805364 |
| train_output_projection_grad_mean | 0.2720121 |
| train_gglrf_grad_mean | 0.90394936 |
| train_darg_grad_mean | 0.30102287 |
| train_cam_grad_mean | 2.2406861 |
| train_routed_to_total_cam144_ratio_mean | 0.38940662 |
| train_routed_to_total_cam144_ratio_first10_mean | 0.00082889442 |
| train_routed_to_total_cam144_ratio_last10_mean | 0.36674127 |
| train_delta_ratio_mean_first10_mean | 0.012059405 |
| train_delta_ratio_mean_last10_mean | 1.6072644 |
| train_delta_ratio_max_max | 9.0153119 |

All formal losses, measured gradients, checkpoints and outputs passed finite-value checks. A larger residual is not automatically better discriminative evidence. Regularizer/total-gradient norm ratios are descriptive, can be affected by cancellation, and do not bound parameter drift. Per-fold size, gradient, beta, null, global-off and Pairwise→Unary intervention diagnostics are in mechanism.csv. The Unary replacement is a frozen distribution substitution, not a second formal training group.

## Audit, resources and limitations

10/10 formal tasks, 10/10 independent best metrics, 10/10 latest/four optimizers, 10/10 exact initial identities and Base-off checks; 10/10 exact E81.4 capped checkpoint replays; common-epoch public RNG mismatches=0 against DARG and E81.4. A fresh audit process repeated all preflight checks and serialized two-step recovery. Strict earliest maximal ACC selection is verified.

Frozen training source/config/history aggregate SHA256: `b9918d251ef8c8cac39399db0445a5ac0a3e82b2fe34c5b2d613ca939ff16ebc`; current audited-source SHA256: `4b2411b46d41395540a3c15459200e688e2e69fecb45c352468ae6f73cd97873`. Full dependency hashes, historical versions and source review are retained. Historical dependency hashes are checked again after the complete audit.

Raw JSONL logs lack three scalar records (fold1 epoch75; fold2 epochs137/301); the storage cause is not established. The first independent audit stopped rather than waiving the check. Both affected folds were deterministically replayed from the original warmup solely for audit: all 756 retained scalar rows, both best/latest models, all four optimizers and RNG reproduced exactly. Three recovered records are stored in separate immutable recovery JSON files; original logs, tasks and checkpoints remain byte-identical. The original audit is snapshotted; only audit.py adds verified-record reconciliation and dual revision SHA checks. No formal training code, configuration, checkpoint or research threshold changed. A preliminary diagnostic replay comparator was corrected to compare CPU-normalized tensors after a CPU/CUDA metadata mismatch; no formal artifacts were changed.

Inherited limitations: test=validation ACC checkpoint selection is biased rather than independent testing; pre-CV RFE/standardization; transductive patient graph; Fold2 patients377/407 in original train/test; overlapping CV training sets. Training-only new-residual blocking does not repair the original overlap or indirect graph effects. All eval patients receive the full model without label-dependent fallback. The16 latent tokens are not validated brain regions. No medical prompt was tested.

Research decision: **NO_GO_UNBOUNDED**. No further experiment is started automatically. Causal attribution is limited to this fixed residual-mapping contrast within the inherited protocol.

## Requested answers

Q1. Actual mean residual ratio=161.8398%, P50=142.5907%, P95=293.6850%; fraction above10%=100.00%, overall maximum=647.9296%.

Q2. Versus capped E81.4: ACC -1.3806pp, BA -1.3903pp.

Q3. Versus DARG: ACC -1.3806pp, BA -1.4143pp; both exceed baseline=False.

Q4. DARG→current-off net -151; residual contribution +139; DARG→full net -12.

Q5. Without Fold2, DARG→full net -10; ACC/BA differences -1.2786/-1.2857pp. No recorded nonfinite loss/gradient/output; large finite residuals and fold heterogeneity must still be considered.

## Resource record

Single GPU; configured limit10, actual maximum9 concurrent workers after one isolated measurement fold; one CPU thread each. Dispatch wall time 644.52s; failures 0; free GPU bytes at dispatch 103097708528; observed preflight peak 1358318080 bytes.

## Scientific interpretation after independent audit

The predeclared verdict is **NO_GO_UNBOUNDED**. Removing the cap greatly increased representation changes while reducing both primary metrics. This result does not support the hypothesis that the fixed 10% cap prevented useful diagnostic fusion under this protocol. A protective or stabilizing effect is consistent with the data; no claim that 10% is optimal follows from this one contrast.

The residual-off current path falls to74.8472% ACC, compared with historical DARG92.1891% and capped E81.4's current-off90.4650%. The three-stage counts change from E81.4's −15/+15/0 to E81.5's −151/+139/−12. Thus the new residual is strongly used and compensates for much poorer current residual-off performance, but does not establish diagnostic gain beyond historical DARG. Turning off a branch trained jointly with the classifier induces an input-distribution change; these figures show strong joint adaptation/dependence, not the optimum of a separately trained or selected backbone.

Eta0.1 did not prevent this dependence. The added-input/total CAM gradient norm ratio averages38.94% over fold means; eta scales one derivative path, not the whole classifier input effect or the eventual parameter drift. All parameter paths remain trainable. The residual norm averages161.84%, with test-fold maxima reaching647.93%; this is finite but substantial perturbation rather than a modest extension past10%.

Against capped E81.4, ACC W/T/L is1/3/6 and BA3/0/7; against DARG both are2/1/7. Fold5 has the largest ACC loss at−4.5977pp, while fold1 improves1.1364pp. The negative result is not limited to Fold2: nine-fold ACC90.9367%, BA90.7624%, net−10 vs DARG. ASD sensitivity falls2.2134pp vs DARG, TD specificity0.6152pp, probability AUC0.4335pp and ASD-F1 1.6424pp. This is not a trade where only an unimportant secondary metric declined.

At fixed historical E81.4 weights, simply removing the cap causes0 corrections and8 harms (net−8); re-training E81.5 causes net−12 vs DARG. The frozen intervention and re-training contrast have distinct parameter trajectories and cannot substitute for each other. Both are inconsistent with an automatic benefit from larger feature modifications.

The fixed regularizer did not dominate the scalar classification loss on average. Across all executed epochs, weighted L_res averages0.0027963 and its mean ratio to classification loss is1.1724%, maximum10.3716%. Its gradient norm averages0.03354, maximum0.38209; the regularizer/total-new-module norm ratio averages9.516%, with some cancellation-sensitive ratios above1. These ratios are not additive contribution fractions. No coefficient was changed in response. All recorded losses, gradients and outputs are finite; absence of NaN/Inf does not imply desirable representation stability.

No extra formal group, hyperparameter search, medical prompt, classifier, frozen-backbone training, historical-result fallback or subsequent experiment was introduced.

## Per-fold learned residual magnitude

Ratios are test-patient summaries at each selected full-model checkpoint, in percent.

| Fold | Mean | P50 | P95 | Max | Above10% | Above100% |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 196.1405 | 161.4931 | 386.3588 | 499.6100 | 100.0000 | 100.0000 |
| 2 | 131.0178 | 98.0491 | 304.6726 | 355.3395 | 100.0000 | 49.4253 |
| 3 | 115.1226 | 108.5784 | 164.1842 | 207.9236 | 100.0000 | 59.7701 |
| 4 | 181.4164 | 178.5729 | 237.7181 | 310.1080 | 100.0000 | 100.0000 |
| 5 | 193.2100 | 189.7925 | 289.7649 | 322.8543 | 100.0000 | 95.4023 |
| 6 | 192.3394 | 148.6895 | 396.0374 | 463.9783 | 100.0000 | 90.8046 |
| 7 | 145.8539 | 99.2800 | 356.5021 | 368.8281 | 100.0000 | 49.4253 |
| 8 | 198.3748 | 185.5388 | 364.4794 | 647.9296 | 100.0000 | 78.1609 |
| 9 | 108.7001 | 104.1971 | 189.1809 | 237.6531 | 100.0000 | 54.0230 |
| 10 | 156.2224 | 151.7158 | 247.9513 | 284.7178 | 100.0000 | 85.0575 |

## Complete resource and engineering record

Formal training elapsed 828.8s (13.81min), including initialization/dispatch gaps. Isolated first-fold loop 136.7s; remaining scheduler 644.5s; combined loop/scheduler 781.2s. Actual maximum9 concurrent workers, limit10; maximum measured per-worker allocated GPU memory 1.336GiB, observed whole-device use17,747MiB. Existing E82 audit was observed and left untouched.

All 7299 common-epoch RNG triplets match DARG/E81.4 exactly (mismatch0). Formal records:10 tasks,20 best/latest checkpoints. A separate audit-only replay verified759 affected-fold epochs, including756 retained rows and3 recovered rows. Original artifacts are unchanged; raw-log storage loss cause remains unestablished. The first diagnostic comparator encountered CPU/CUDA tensor placement and was corrected to exact CPU normalization before successful replay.

Source versions remain explicit: frozen training `b9918d251ef8c8cac39399db0445a5ac0a3e82b2fe34c5b2d613ca939ff16ebc`; audited current source `4b2411b46d41395540a3c15459200e688e2e69fecb45c352468ae6f73cd97873`. Only audit.py differs. The original frozen audit and exact recovery proofs are in runtime/audit_training_frozen.py, runtime/audit_repair.json, runtime/log_recovery_f01.json and runtime/log_recovery_f02.json. No training configuration or checkpoint was replaced.

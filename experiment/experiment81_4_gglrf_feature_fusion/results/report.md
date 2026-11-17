# E81.4 — Global-Guided Local Relational Fusion (GGLRF)

Independent audit: **PASS**. Research decision: **NO_GO_E81_4**.

## Source and implementation review

The [SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) and [official code](https://github.com/wanxixi11/SPromptGL) were checked against the [project implementation](https://github.com/tangaii/SPromptGL). RES motivates global-conditioned selection; this implementation retains 16 latent locals and writes a bounded residual into CAM144. The inherited PEN-free CAM and original 144→36→2 OutputLayer with dropout 0.35 remain active. GFR/DARG/CAM/OutputLayer/GGLRF all train end to end. No PEN, CLIP, text prompt or new classifier was added.

G1/G2 each add 16,384 trainable parameters, identical same-fold initialization, 15 real distinct neighbors and one zero-value Null. Only Unary versus product/signed-difference Key/Value construction differs. G1 query-key attention itself models implicit interactions; G2 is not compared against a relation-free model.

The pre-hook runs the original classifier once. Training reuses historical_out.total plus 0.001 residual regularization. Eta=0.1 applies only to the new z_rel input path; the norm bound uses detached CAM and rho=0.10. Zero residual restores baseline gradients exactly. Nonzero fusion changes the classifier input and optimization trajectory, so this is not full-time historical-gradient preservation.

## Six metrics

Percent, ten-fold mean ± population SD. Hard-AUC is hard-prediction balanced accuracy; Prob-AUC is probability ROC-AUC. ASD=0, TD=1.

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2-Positive-DCR | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E81.4-G1-Global-Unary-Feature-Fusion | 91.9592±2.6778 | 91.7901±2.6595 | 93.6327±2.9503 | 89.5793±3.3078 | 94.0009±3.7145 | 91.1758±2.8672 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | 92.1891±2.6675 | 92.0054±2.6198 | 93.5688±2.5794 | 89.8110±3.8125 | 94.1998±4.7671 | 91.4385±2.7824 |
| E81.1-G1-Visual-PRCL-v2 | 92.0742±2.9789 | 91.9051±2.8926 | 93.4621±3.0770 | 89.8232±3.2483 | 93.9870±5.1280 | 91.3509±3.0094 |
| E81.3-G2-PairwiseRelation-E2E | 91.7294±2.9505 | 91.5401±2.9690 | 93.4772±2.8732 | 89.0793±4.0349 | 94.0009±3.4622 | 90.8841±3.1806 |

## Paired changes

Percentage points; W/T/L uses same-fold differences. The 95% paired t intervals are descriptive: overlapping training folds violate an independent patient-sampling interpretation.

| Comparison | Metric | Mean Δ (pp) | Descriptive 95% interval | W/T/L |
|---|---|---:|---:|---:|
| G1_vs_DARG | acc | -0.2299 | [-0.5766, +0.1168] | 0/8/2 |
| G1_vs_DARG | hard_auc | -0.2393 | [-0.5619, +0.0833] | 1/5/4 |
| G1_vs_DARG | prob_auc | +0.1381 | [-0.1973, +0.4736] | 6/0/4 |
| G1_vs_DARG | sen | -0.4878 | [-1.8765, +0.9009] | 1/7/2 |
| G1_vs_DARG | spe | +0.0093 | [-1.4414, +1.4599] | 2/5/3 |
| G1_vs_DARG | asd_f1 | -0.2833 | [-0.6504, +0.0839] | 1/5/4 |
| G2_vs_DARG | acc | +0.0000 | [-0.5482, +0.5482] | 2/6/2 |
| G2_vs_DARG | hard_auc | -0.0240 | [-0.6240, +0.5760] | 3/5/2 |
| G2_vs_DARG | prob_auc | +0.0742 | [-0.5583, +0.7068] | 4/3/3 |
| G2_vs_DARG | sen | -0.2561 | [-1.8160, +1.3038] | 2/6/2 |
| G2_vs_DARG | spe | +0.2081 | [-0.6630, +1.0793] | 2/7/1 |
| G2_vs_DARG | asd_f1 | -0.0205 | [-0.6985, +0.6575] | 3/5/2 |
| G2_vs_G1 | acc | +0.2299 | [-0.4187, +0.8785] | 4/4/2 |
| G2_vs_G1 | hard_auc | +0.2153 | [-0.4279, +0.8585] | 6/2/2 |
| G2_vs_G1 | prob_auc | -0.0639 | [-0.5716, +0.4438] | 4/0/6 |
| G2_vs_G1 | sen | +0.2317 | [-1.3082, +1.7717] | 2/6/2 |
| G2_vs_G1 | spe | +0.1989 | [-1.3343, +1.7321] | 4/4/2 |
| G2_vs_G1 | asd_f1 | +0.2628 | [-0.4573, +0.9829] | 6/2/2 |
| G2_vs_E81.1_G1_context | acc | +0.1149 | [-0.3518, +0.5817] | 2/7/1 |
| G2_vs_E81.1_G1_context | hard_auc | +0.1003 | [-0.3831, +0.5837] | 2/5/3 |
| G2_vs_E81.1_G1_context | prob_auc | +0.1067 | [-0.4081, +0.6216] | 4/0/6 |
| G2_vs_E81.1_G1_context | sen | -0.0122 | [-1.4487, +1.4243] | 1/7/2 |
| G2_vs_E81.1_G1_context | spe | +0.2128 | [-0.9203, +1.3459] | 3/5/2 |
| G2_vs_E81.1_G1_context | asd_f1 | +0.0876 | [-0.4357, +0.6110] | 2/5/3 |
| G2_vs_E81.3_G2_context | acc | +0.4598 | [+0.0352, +0.8844] | 4/6/0 |
| G2_vs_E81.3_G2_context | hard_auc | +0.4653 | [-0.0513, +0.9819] | 4/4/2 |
| G2_vs_E81.3_G2_context | prob_auc | +0.0917 | [-0.3797, +0.5630] | 6/0/4 |
| G2_vs_E81.3_G2_context | sen | +0.7317 | [-1.7495, +3.2129] | 3/5/2 |
| G2_vs_E81.3_G2_context | spe | +0.1989 | [-1.4939, +1.8917] | 3/5/2 |
| G2_vs_E81.3_G2_context | asd_f1 | +0.5544 | [-0.0645, +1.1733] | 4/4/2 |

Per-fold differences for all six metrics are preserved in paired_comparisons.csv.

## Excluding Fold 2

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2-Positive-DCR | 92.2153±2.9131 | 92.0481±2.8723 | 93.6142±3.0541 | 90.0474±4.0959 | 94.0487±4.9847 | 91.4703±3.0580 |
| E81.4-G1-Global-Unary-Feature-Fusion | 91.9598±2.8227 | 91.7822±2.8032 | 93.7324±3.0939 | 89.5054±3.4789 | 94.0590±3.9111 | 91.1555±3.0216 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | 92.2153±2.8105 | 92.0067±2.7615 | 93.6202±2.7141 | 89.4919±3.8899 | 94.5215±4.9209 | 91.4244±2.9326 |

## Historical backbone / current backbone / feature contribution

The historical model is its original best checkpoint, never a training initializer. The current original branch is the trained E81.4 checkpoint with residual disabled. Counts below are test-fold patient predictions.

| Group | Transition | Corrected | Harmed | Net | Net excluding Fold 2 |
|---|---|---:|---:|---:|---:|
| E81.4-G1-Global-Unary-Feature-Fusion | historical → current residual-off | 6 | 13 | -7 | -7 |
| E81.4-G1-Global-Unary-Feature-Fusion | current residual-off → full | 6 | 1 | +5 | +5 |
| E81.4-G1-Global-Unary-Feature-Fusion | historical → full | 5 | 7 | -2 | -2 |
| E81.4-G1-Global-Unary-Feature-Fusion | global-guidance-off → full | 7 | 1 | +6 | +6 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | historical → current residual-off | 2 | 17 | -15 | -15 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | current residual-off → full | 16 | 1 | +15 | +15 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | historical → full | 4 | 4 | +0 | +0 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | global-guidance-off → full | 14 | 0 | +14 | +14 |

Current trained residual-off metrics (ten-fold mean ± population SD):

| Group | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.4-G1-Global-Unary-Feature-Fusion | 91.3845±2.7877 | 91.2209±2.7748 | 93.6113±2.8660 | 89.0793±3.7445 | 93.3626±4.0179 | 90.5500±3.0212 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | 90.4650±4.9903 | 90.3250±4.8691 | 93.6005±2.6502 | 88.5732±4.9226 | 92.0768±7.4174 | 89.6743±5.0980 |

## Frozen inference interventions

Each intervention uses the selected trained checkpoint and unchanged learned parameters. Global-off zeros global inputs to both relation and readout queries while keeping local input fixed. Pairwise→Unary uses G2 projections on Unary inputs and entails distribution replacement; it does not replace independent G1/G2 training.

| Group | Intervention | Prob MAE | ΔACC pp | ΔBA pp | ΔF1 pp | Flips | Corrected / harmed vs full |
|---|---|---:|---:|---:|---:|---:|---:|
| E81.4-G1-Global-Unary-Feature-Fusion | residual_off | 0.016536 | -0.5747 | -0.5691 | -0.6258 | 7 | 1 / 6 |
| E81.4-G1-Global-Unary-Feature-Fusion | null_only | 0.016536 | -0.5747 | -0.5691 | -0.6258 | 7 | 1 / 6 |
| E81.4-G1-Global-Unary-Feature-Fusion | global_off | 0.011424 | -0.6897 | -0.6592 | -0.6890 | 8 | 1 / 7 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | residual_off | 0.029073 | -1.7241 | -1.6804 | -1.7643 | 17 | 1 / 16 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | null_only | 0.029073 | -1.7241 | -1.6804 | -1.7643 | 17 | 1 / 16 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | global_off | 0.014509 | -1.6092 | -1.5794 | -1.7185 | 14 | 0 / 14 |
| E81.4-G2-Global-Pairwise-Feature-Fusion | pairwise_to_unary | 0.028454 | -1.2618 | -1.2656 | -1.4287 | 13 | 1 / 12 |

Null-only and residual-off logp/hidden are exactly equal for all 20 checkpoints.

## Global readout, residual and gradient diagnostics

| Diagnostic (fold means) | G1 | G2 |
|---|---:|---:|
| beta_entropy | 0.8290974 | 0.7921873 |
| beta_max_mean | 0.2651322 | 0.3294209 |
| beta_patient_sd_mean | 0.03740487 | 0.02932333 |
| beta_near_uniform_fraction | 0 | 0 |
| beta_single_local_collapse_fraction | 0.02643678 | 0.04252874 |
| attention_entropy | 0.7542355 | 0.8487445 |
| null_weight | 0.06051352 | 0.05037105 |
| message_norm_mean | 3.895656 | 3.574049 |
| head_message_cancellation_ratio | 0.9989298 | 0.9992762 |
| local_direction_cancellation_ratio | 0.9002308 | 0.6722018 |
| global_query_norm | 5.864473 | 5.850023 |
| delta_ratio_mean | 0.09838785 | 0.09822387 |
| delta_ratio_p50 | 0.09874963 | 0.0988139 |
| delta_ratio_p95 | 0.09966715 | 0.09969928 |
| near_bound_fraction | 0.5846787 | 0.6137409 |
| near_zero_residual_fraction | 0 | 0 |
| residual_hidden_mae | 0.01555303 | 0.02642947 |
| global_off_beta_mae | 0.04205673 | 0.04705429 |
| global_off_attention_mae | 0.03456321 | 0.02853474 |
| global_off_delta_mae | 0.01643246 | 0.01524338 |
| train_global_encoder_grad_mean | 0.04748379 | 0.04837106 |
| train_global_readout_grad_mean | 0.0112115 | 0.01480149 |
| train_delta144_mean | 0.01089731 | 0.01140886 |
| train_routed_to_total_cam144_ratio_mean | 0.05094124 | 0.04967586 |
| train_routed_to_total_cam144_ratio_first10_mean | 0.0004400078 | 0.0008088132 |
| train_routed_to_total_cam144_ratio_last10_mean | 0.1007091 | 0.07926012 |

Normalized entropies are in [0,1]. The routed/total gradient-norm ratio is a descriptive norm ratio, not a bound on parameter drift. Eta is exactly 0.1 on the new input path; a new path can still be small, large, or partly cancel the identity path. Per-local beta and residual vectors, confidence/error partitions and global-off changes are in mechanism.csv. Head/local cancellation ratios diagnose averaging directions; 16 locals are not averaged by the model. Latent tokens have no validated anatomical meaning.

## Training, reproducibility and audit

Formal tasks: 20/20; independent best reload: 20/20; latest/four optimizers: 20/20; historical best reload: 10/10. Base-off exact checks: 10/10; shared-epoch RNG mismatches: 0.

Scheduler: 10 workers maximum, 922.2s dispatch wall time; failures: 0. One CPU thread per worker; private CPU initialization; no public reseeding or new stochastic layer.

Serialized fresh-model two-step replay: PASS_EXACT_SERIALIZED_FRESH_MODELS. Preflight checked the first zero-head gradient and second-update global/local/readout gradients, 15 unique neighbors, patient isolation, eta 0/.1/1 and detached norm boundaries.

Frozen training source/config/history SHA256: `7dd23d91341597d6eae1f7ae648c718279aed9f59b0cdd5021982dd4d5ae5b68`; corrected audit/source SHA256: `0fa19058f70a75838e62b43599820d1ac763827a001fd33754259887496ee86c`. All task/checkpoint SHA values, source dependencies and independent records are available in manifest.json and audit_status.json.

The first independent audit stopped on an audit-only dict-versus-Logp argument error. The corrected audit selects out[logp]; no training logic/configuration or completed checkpoint changed. The original frozen audit is preserved in runtime, and every other dependency is checked against the original source map. A fresh post-training process repeated all ten exact Base-off checks and serialized recovery without replacing the original frozen preflight.

## Scientific interpretation and limits

Decision: **NO_GO_E81_4**; secondary: `PAIRWISE_GAIN_BELOW_BASELINE`.

Baseline floor met: False; G2 ACC and BA improve over G1: True; nine-fold support: False; positive residual/global/final net support: False.

The comparison with E81.1/E81.3 is historical context, not a controlled attribution of the integration point alone: their complete architectures and objectives differ. Classification changes from current-off to full isolate the selected checkpoint residual effect; changes from historical to current-off also include changed training and selected epoch. An active attention or a nonzero residual alone does not establish diagnostic gain.

The inherited protocol selects checkpoints using the test fold (val=test), uses preprocessing/RFE before CV and a transductive graph, and preserves Fold 2 training/test overlap IDs 377/407. New training residuals and regularization exclude those two rows, but the historical and indirect effects remain. Therefore these are same-protocol exploratory results, not unbiased independent test estimates. Fold intervals are descriptive and no MICCAI-level validation or medical prompt effectiveness is claimed. No E81.5 is launched.

## Artifacts

Code/config/design/manifest are in this experiment directory. results/ contains preflight.json, fold_metrics.csv, paired_comparisons.csv, mechanism.csv, audit_status.json and this report. runtime/ holds only necessary best/latest checkpoints, task ledgers, compact logs and scheduling provenance.

## 最终四问与科研判断

1. **相对E81.3，最终性能的点估计改善，但不能归因于特征写回这一单独因素。** E81.4-G2相对E81.3-G2：ACC +0.4598 pp，Hard-AUC +0.4653 pp，Prob-AUC +0.0917 pp；架构、聚合方式和损失同时改变，因此只是历史改版比较。当前关闭残差分支相对DARG净损失15例，比E81.3的净损失11例更多；没有证据表明主干退化得到更好控制。

2. **G2相对G1有有限的配对改善迹象，尚未建立稳定的显式关系优势。** ACC +0.2299 pp，W/T/L=4/4/2，描述性95%区间[−0.4187,+0.8785] pp；Hard-AUC +0.2153 pp，W/T/L=6/2/2，区间[−0.4279,+0.8585] pp。Prob-AUC −0.0639 pp。冻结G2的Pairwise→Unary替换使ACC下降1.2618 pp、产生13次翻转（相对完整模型纠正1、误伤12），说明已训练模型确实使用Pairwise输入，但替换存在分布变化，不能代替正式G1/G2对照。

3. **G2没有超过DARG，也没有同时达到两个性能门槛。** ACC 92.1891327%与历史DARG完全持平；Hard-AUC 92.0053953%低于历史92.0293738%（−0.0240 pp），也低于规定92.0294%门槛。不能将近似相等写成达标。去Fold 2后，G2/DARG ACC均为92.2153%，Hard-AUC分别为92.0067%/92.0481%；仍无净增益。

4. **主要表现为新增残差补偿当前训练分支的损失，未形成超过历史DARG的净纠错。** G2历史→当前关闭残差：纠正2、误伤17（−15）；当前关闭残差→完整：纠正16、误伤1（+15）；历史→完整：纠正4、误伤4（0）。去Fold 2后上述净值仍为−15/+15/0。关闭残差后的指标还包含联合训练适配和所选epoch的影响，不能把所有差异因果归为梯度干扰。Fold 8/9贡献新增残差净纠错12/15例，收益较集中。

全局引导确实参与模型：关闭两个Query的全局输入后，G2概率MAE=0.014509，ACC下降1.6092 pp，完整模型相对此干预纠正14、误伤0，且全部发生在Fold 2之外。但这证明选定模型的功能依赖，仍未证明相对历史DARG存在额外诊断增益。

读出并非均匀平均：G2平均归一化beta熵0.7922、最大权重0.3294、患者间权重SD均值0.0293。另一方面，每折最常出现的top-token平均覆盖83.49%的测试患者；4.25%的患者最大权重超过0.8，存在固定潜在坐标偏好和部分尖锐读出，不能把所有权重变化解释为充分个体化或脑区重要性。

残差并未趋于零：G2各折相对范数统计量的均值为：均值9.8224%、P50 9.8814%、P95 9.9699%；61.37%的患者达到9.9%以上，频繁接近10%边界。平均概率变化MAE=0.029073，hidden MAE=0.026429。按预先固定的|margin|≥1分区，48个当前关闭残差分支的高置信错误中没有被残差纠正的患者。平滑上限、Null-only和关闭残差的精确回退均通过，不能据此声称诊断创新成功。

最终判定：**NO_GO_E81_4**；相对对照的次级信号归类为 **PAIRWISE_GAIN_BELOW_BASELINE**。不自动启动E81.5，不将本实验用于医学Prompt有效性或MICCAI级验证的主张。

## 补充源码核查与执行记录

[官方GFR/CAM/PEN实现](https://github.com/wanxixi11/SPromptGL/blob/main/network.py)中，GraphLearn对投影特征计算cosine相似度并作阈值掩码，GCN在加自环、归一化邻接上对拼接特征传播。论文Eq.(1)还描述标签监督邻接项；[发布训练代码simple-2路径](https://github.com/wanxixi11/SPromptGL/blob/main/model.py)虽然获得ALoss，但实际反传为加权分类损失与四模态平均graph loss，未把该返回项加入总损失。论文公式、发布实现与历史DARG继承链据此分别核查，本次没有重装发布PEN或更改历史目标。

首个单worker训练循环耗时67.5s，随后19任务调度耗时922.2s，二者合计989.6s。依据首次派发日志创建时间至调度结束时间，正式阶段实际经过约1047.2s（约17.5分钟，包含首任务载入及阶段衔接，不含预检、来源核查和审计；时间戳精度约1s）。最大10并发、每worker一CPU线程、单卡PPU-ZW810E 98,304MiB；首个进程观察显存约1.9GiB，并发过程观察约17.5GiB。20任务无失败，无重训或覆盖完成任务。

审计首次因字典/张量参数错误停止；修复只涉及audit.py的调用和版本核对，训练冻结SHA与原始Preflight完整保留。训练后新进程重复10/10精确Base-off和两步恢复通过。最终全部20个最佳Checkpoint、20个latest及四优化器通过，历史DARG最佳Checkpoint也逐折重算通过；合计11,147个共有epoch三元组比较（包括组间比较），RNG mismatch=0。修复记录与原始审计脚本分别保存在runtime/audit_repair.json和runtime/audit_training_frozen.py。

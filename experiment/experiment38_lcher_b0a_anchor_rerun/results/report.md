# Experiment 38 Rerun: LCHER on the B0-A Anchor

**Status:** PARTIAL  
**Anchor replay:** ANCHOR_REPLAY_PASS (10/10)  
**Scientific decision:** NO-GO  
**Delivery:** root experiment sync `FAILED_OSSFS_EIO`; root handoff `FAILED_PENDING_APPEND`. OSSFS error: `cp -a could not create target directory: Input/output error; no target directory was created`.

This is the rerun of Experiment 38 with the exact Experiment 32 B0-A implementation as the shared anchor. It replaces the previous experiment38 report's invalid physically PEN-deleted baseline framing. No training result from the earlier experiment38 directory is counted here.

## Audit and Base

Local Git HEAD: `7747a1521629f95413faca133ba8c82ed7f1faf2`; audited GitHub main revision: `66d6a3b6e8ddc0332fe1f50cbf31caa1ae929afa`. Root source hashes and input, mapping, prompt, CLIP cache and fold-split hashes are in `manifest.json`. The paper checked was [SPromptGL, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf). Read-only audit covered the root README, main/model/network/layers/utils and ABIDE shell, Experiment 11's verified FC/ROI map, Experiment 32 B0 integration/run/metrics/report, local Experiments 13/15/16/17/18/19/28/33, and available `/tmp` reports/methods for Experiments 34–37. The old Experiment 38 files were read-only background; their physically deleted B1 was rejected as an anchor because removing unused PLN construction changes initialization/RNG trajectory.

The B0-A scaffold follows Experiment 32: instantiate the complete original PLN (including unused PEN MLP/avg-pool and eight legacy prompt buffers), retain the original RNG consumption, then bypass SPGS+RES after CAM and pass flattened CAM to the original OutputLayer. PEN is not executed in this forward path. Original GFR behavior, hard threshold, normalization, four graph losses, simple-2 objective, and checkpoint rule remain unchanged.

Replay was a single seed-0 process with a continuous RNG stream across ten folds. All ten folds matched E32 B0 on confusion counts, ACC/Hard-AUC/SPE/SEN, best epoch, executed epochs, shared trunk initialization SHA and fold-entry RNG SHA. Immutable per-fold anchors include model, both Adam states and Python/NumPy/Torch CPU/CUDA RNG. E1–E9 restore the corresponding B0 anchor; this pairing does not claim naturally continuous E-group RNG streams.

GFR graphs are 871-patient graphs, not ROI graphs; each branch propagates the full concatenated 320-dimensional input. The source's i==1 overwrite of i==0 is preserved. In simple-2, ALoss is computed but not optimized; the optimized objective is classification loss plus the mean of four GraphConstruct losses. The 4×36 CAM representation is already mixed and is not interpreted as ROIs. The historical full-cohort RFE/preprocessing, transductive graph, and `val_idx=test_idx` model selection remain, so these are internal historical-protocol comparisons, not unbiased external generalization results.

## Frozen LCHER

The fixed candidate library contains H01–H13, 26 ASD/TD text prompts and 39 unique FC indices; H14–H16 remain archival. Text is the supplied literature synthesis, not a new GPT-4 call or clinician certification, and network-level literature does not prove an effect for each HO edge. Experiment 11's ROI mapping verifies the 39 candidate indices and both endpoint names; unknown indices 195/225/232/236 are excluded. Only raw input columns `64 + corr_index` carry these FC identities; mixed GFR coordinates are not used as ROI indices.

The frozen RN50 CLIP bank is hashed in the manifest and cached once. Per fold, an alpha=10 Ridge reference uses outer-train TD only, with five-fold OOF references for outer-train participants; z scores are signed and clipped to ±3. The same patient receives both rival prompts. A straight-through Top-8 selector operates over 39 eligible raw FC edges. Selected/removed input-level counterfactuals replace only the target participant's selected raw FC values with the fold TD reference and rerun full GFR→CAM→OutputLayer. The post-CAM signed evidence residual is zero-initialized and individually norm-limited to 5%; new parameters are added to the existing PLN Adam with zero weight decay.

## Results and Paired Analysis

Formal metrics are ACC, original one-hot hard-label ROC-AUC, SPE-TD, and SEN-ASD, shown as ten-fold mean ± population SD (%). ASD is label 0; TD is label 1.

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|B0A_ANCHOR|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E1_ANON_FULL|91.3845 ± 2.7399|91.2140 ± 2.6918|93.3488 ± 4.2833|89.0793 ± 3.3765|
|E2_REAL_NO_CF|91.3858 ± 2.9257|91.1773 ± 2.9597|94.0009 ± 3.0316|88.3537 ± 4.1166|
|E3_REAL_KEEP_ONLY|91.3858 ± 3.0581|91.1617 ± 3.0686|94.2137 ± 3.4858|88.1098 ± 4.2058|
|E4_LCHER_FULL|91.3845 ± 2.8347|91.1798 ± 2.7726|93.7743 ± 4.3931|88.5854 ± 3.1872|
|E5_WRONG_EDGE_FULL|91.7307 ± 2.7165|91.5332 ± 2.6542|93.9870 ± 4.5327|89.0793 ± 3.3854|
|E6_NEUTRAL_FULL|91.6157 ± 2.6800|91.4936 ± 2.5795|92.9140 ± 5.1451|90.0732 ± 3.3470|
|E7_NO_AUX_FULL|91.7294 ± 3.0388|91.5066 ± 3.0224|94.4218 ± 3.7740|88.5915 ± 3.5393|
|E8_NO_RIVAL_FULL|91.9592 ± 3.1745|91.7651 ± 3.2173|94.4265 ± 4.0206|89.1037 ± 5.3624|
|E9_SWAP_CLASS_FULL|91.7294 ± 2.7656|91.5992 ± 2.7011|93.1314 ± 4.4386|90.0671 ± 2.9753|

各创新组相对共同B0-A anchor的逐折配对差异（百分点；W/T/L按ACC与Hard-AUC）：

|组别|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔSPE|ΔSEN|
|---|---:|---:|---:|---:|---:|---:|
|E1_ANON_FULL|-0.6897|0/4/6|-0.6406|1/2/7|-1.2812|+0.0000|
|E2_REAL_NO_CF|-0.6883|2/0/8|-0.6773|2/0/8|-0.6290|-0.7256|
|E3_REAL_KEEP_ONLY|-0.6883|2/2/6|-0.6929|2/2/6|-0.4163|-0.9695|
|E4_LCHER_FULL|-0.6897|0/5/5|-0.6748|0/4/6|-0.8557|-0.4939|
|E5_WRONG_EDGE_FULL|-0.3435|2/4/4|-0.3215|3/3/4|-0.6429|+0.0000|
|E6_NEUTRAL_FULL|-0.4585|2/3/5|-0.3611|4/1/5|-1.7160|+0.9939|
|E7_NO_AUX_FULL|-0.3448|1/5/4|-0.3480|2/4/4|-0.2081|-0.4878|
|E8_NO_RIVAL_FULL|-0.1149|2/6/2|-0.0896|3/5/2|-0.2035|+0.0244|
|E9_SWAP_CLASS_FULL|-0.3448|2/4/4|-0.2554|3/3/4|-1.4986|+0.9878|

E4相对全部预注册创新对照的逐折配对差异（百分点；W/T/L按ACC与Hard-AUC）：

|E4−对照|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔSPE|ΔSEN|
|---|---:|---:|---:|---:|---:|---:|
|B0A_ANCHOR|-0.6897|0/5/5|-0.6748|0/4/6|-0.8557|-0.4939|
|E1_ANON_FULL|+0.0000|2/6/2|-0.0342|4/3/3|+0.4255|-0.4939|
|E2_REAL_NO_CF|-0.0013|3/4/3|+0.0025|4/3/3|-0.2266|+0.2317|
|E3_REAL_KEEP_ONLY|-0.0013|2/6/2|+0.0181|4/3/3|-0.4394|+0.4756|
|E5_WRONG_EDGE_FULL|-0.3461|2/3/5|-0.3533|2/3/5|-0.2128|-0.4939|
|E6_NEUTRAL_FULL|-0.2312|2/4/4|-0.3137|3/0/7|+0.8603|-1.4878|
|E7_NO_AUX_FULL|-0.3448|1/5/4|-0.3268|1/5/4|-0.6475|-0.0061|
|E8_NO_RIVAL_FULL|-0.5747|2/2/6|-0.5852|2/2/6|-0.6522|-0.5183|
|E9_SWAP_CLASS_FULL|-0.3448|1/5/4|-0.4194|1/5/4|+0.6429|-1.4817|

机制诊断：B0-A+zero-alpha的forward、主损失、四图损失、主干梯度和两个Adam首步通过BITWISE_EXACT预检；测试标签置换不影响先验/推理。预检中的CF梯度在alpha、query/key、context、selector及signed residual路径均有限且非零，原始FC干预经完整GFR产生非零输出响应。

E4固定checkpoint审计：NULL概率均值 0.3126；selected-delete减random-delete的正确类logp下降均值 0.0737，正值 8/10 folds；文本交换平均绝对logp变化 0.001258，类别翻转比例 0.001149；禁用路由的平均绝对logp变化 0.000743。CAM相对残差折均值 0.1746%，跨所有训练组及折记录的最大值 3.7226%，低于5%上限。

判定：**NO-GO**。E4 ACC/AUC均低于B0-A约0.6897/0.6748个百分点，且低于匿名同预算E1约0.0000/0.0342个百分点；E5错误边映射和E6中性文本的ACC/AUC都高于E4。selected-delete优于random-delete虽为正向8/10折，但文本交换对logit的作用极小，平均类别翻转约0.1%；这不足以证明正确医学文本或连接映射具有独立任务贡献，更不能作生物医学因果解释。

预注册门槛逐项结果：`{"E4_beats_B0A_acc_auc_0_23pp": false, "E4_beats_E1_acc_auc_0_23pp": false, "E4_beats_E5_E6_E9": false, "E4_prompt_swap_changes_output": true, "E4_selected_deletion_beats_random": true, "E4_sen_drop_vs_B0A_no_more_than_0_5pp": true}`。十折训练集相互重叠，W/T/L是描述性配对结果，不是独立队列统计显著性。val=test checkpoint选择、CV前RFE/标准化和transductive图仍带来乐观偏差风险；不构成无偏独立泛化证据。

产物阶段路径：`/tmp/SPromptGL-E38-B0A-stage/experiment/experiment38_lcher_b0a_anchor_rerun`。正式目标目录：`/data/adodas/SPromptGL-main/experiment/experiment38_lcher_b0a_anchor_rerun`。`fold_metrics.csv`含100折混淆计数、最佳epoch、入口及初始化哈希；`diagnostics.csv`含机制量。E2 fold 1因先前诊断schema错误已作为不完整artifact从原anchor恢复一次，best epoch与指标不变，并补齐最佳checkpoint；唯一fold键仍为100。无效临时checkpoint保存在`.runtime/failed_attempts/`。根实验目录同步状态：FAILED_OSSFS_EIO（cp -a could not create target directory: Input/output error; no target directory was created）；handoff追加状态：FAILED_PENDING_APPEND（OSError: [Errno 5] Input/output error），原handoff前缀SHA保持不变。完整产物仅在`/tmp`，这是临时存储且尚未同步。

## Mechanism and Decision

Preflight verified bitwise equivalence between B0-A and zero-alpha LCHER for forward output, classification/GFR losses, common-parameter gradients and the first two Adam updates with CF disabled. The held-out-label invariance and raw-FC full-GFR-response checks passed. Counterfactual gradient paths were finite and nonzero at preflight. Best-checkpoint residual audit recomputed all 90 E1–E9 checkpoints; maximum participant residual/CAM ratio was 3.7226% (5% cap passed: True). Evidence ranking/ablation effects are model-dependence diagnostics, not biological causal evidence.

The frozen screening outcome is **NO-GO**. LCHER-FULL did not clear the required margin over B0-A or the matched anonymous control, and it did not outperform wrong-edge and neutral-text controls on the primary metrics. Selected deletion exceeded equal-budget random deletion in 8/10 folds, but exact prompt swapping produced only a small log-probability response. This does not establish independent contribution from the correct medical wording or edge mapping.

The ten folds share participants across training sets. Fold-paired W/T/L counts are descriptive and are not ten independent cohorts or a significance test. No SOTA, clinical biomarker, causal, or acceptance claim is made.

## Runtime and Recovery

Formal training completed 100/100 unique folds across 10/10 groups, GPU0 50 folds and GPU1 50 folds. Formal wall-clock interval through recovery was 7341.5 seconds. E2 fold 1 had a committed metric row but no best checkpoint after an earlier diagnostics-schema failure; it was therefore treated as artifact-incomplete and replayed once from the exact paired anchor (95.9 s), with the same best epoch and metrics. This is one recovery attempt, not an additional unique fold. Other invalid partial checkpoints remain under `.runtime/failed_attempts/`. Both metrics and diagnostics now have 100 unique `(group, fold)` keys.

Staged directory: `/tmp/SPromptGL-E38-B0A-stage/experiment/experiment38_lcher_b0a_anchor_rerun`. Formal target: `/data/adodas/SPromptGL-main/experiment/experiment38_lcher_b0a_anchor_rerun`. `/tmp` is temporary storage; delivery remains `COMPLETE_WITH_DELIVERY_BLOCKED` until verified. Full pending handoff text is `handoff_append_pending.md`.

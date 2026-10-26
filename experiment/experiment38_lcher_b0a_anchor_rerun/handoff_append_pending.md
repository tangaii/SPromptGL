
## Experiment 38 Rerun — LCHER on the Experiment 32 B0-A Anchor

执行状态：PARTIAL（科学任务 COMPLETE）；筛选结论：NO-GO。100/100 folds、10/10组；GPU0 50 folds、GPU1 50 folds；正式阶段耗时 7341.5 秒。

本轮严格以实验32 B0-A为共同起点。其PLN按历史构造顺序完整初始化并注册未使用的PEN参数和原8条Prompt缓存以保持RNG/参数轨迹；隔离版forward在CAM后跳过SPGS+RES，直接送入原OutputLayer。该历史兼容脚手架不执行PEN。旧实验38的物理删PEN B1（ACC 91.3845%）不是本轮基线，也未作为正式任务或checkpoint使用。

Anchor replay: **ANCHOR_REPLAY_PASS (10/10)**。B0-A以seed=0单进程连续十折重放；逐折混淆计数、四指标、best epoch、epochs executed、主干初始化SHA与入口RNG SHA均与实验32 B0一致。B0-A结果为ACC 92.0742±2.6000%、Hard-AUC 91.8546±2.5815%、SPE-TD 94.6300±4.0457%、SEN-ASD 89.0793±3.7283%。E1–E9从对应fold不可变anchor恢复，新增分支独立RNG初始化；这是配对入口设计，不代表E组自然连续跨折RNG。

源码审计：MICCAI 2025论文 `https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf` 与本地README/main/model/network/layers/utils及ABIDE shell逐项核对；Git HEAD和相关SHA见manifest。GFR四图为受试者关系图，四支传播全320维拼接输入；GraphLearn硬阈值、归一化与原GCN保留，i==1覆盖i==0。simple-2优化分类损失加四图GraphConstructLoss均值；ALoss虽计算但不参与反传。保留871人transductive图、val=test选模和CV前全局RFE/预处理。实验11核验39条候选FC的数值唯一身份及ROI双端名称，195/225/232/236排除；原始FC位置为输入X的64+index。CAM的4×36为混合表示，不解释为ROI。

固定先验由13组、26条ASD/TD竞争文本和39条候选FC边构成；H14–H16只存档。文本是用户提供的文献综合候选，不是GPT-4在线生成、专家认证或精确HO单边临床证据。真实Prompt冻结CLIP RN50缓存；匿名、中性、边错配和类别交换组按预注册产生。每fold仅用outer-train TD标签拟合Ridge参考，train TD用5折OOF；候选真实FC计算signed z并截断±3。固定Top-8通过原始FC干预并完整重跑GFR，CAM后signed残差逐患者限制为CAM范数的5%。

历史审计路径包括本地experiment11/13/15/16/17/18/19/28/32/33，以及此前/tmp中的实验34–37。旧实验38的旧版代码/结果仅只读审阅，不并入本轮折数；其B1基线因物理删除PEN改变初始化/RNG轨迹，不可与E32 B0-A作配对因果比较。

正式指标为原仓库one-hot硬预测ROC-AUC（不报告probability-AUC），按ACC、Hard-AUC、SPE-TD、SEN-ASD顺序：

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

# 实验七：DTSF 双主题语义特征融合

## 审计结论

本实验先核对远程 `main`（commit `36ad0ff`）与本地源码，并阅读实验一至六报告和补丁。实验一 P1 是当前固定基线（ACC 91.80±0.11%、Hard-AUC 91.62±0.12%）；实验二正确模态状态 Prompt 仅有小幅描述性收益，错配版本较低；实验三提示收益不能归因于简单 Prompt 替换，原始 P1 路径仍需保留；实验四疾病特定 Correlation 文本没有稳定收益；实验五显示 4096→128→9 投影存在参数收缩，E0 仍是最佳基线；实验六两种标量方向门控均未改善整体指标。因此本实验不改 Prompt 文本、不使用实验五 E1/E2/E3 的 MLP 调整、不使用实验六 B1/B2 标量门控，不增加 ASD 特定文本、额外损失、正交约束或新的 GCN/分类器。

审计确认原始形状为冻结 CLIP 文本 `[32,1024]`、P1 分组文本 `[4,2,4096]`、P1 原型 `[4,2,9]`、局部特征 `[871,4,4,9]`、Softmax-Max 权重 `[871,4,4]`；原始 `part_x = similarity_max.unsqueeze(-1) * part_x` 与后续 global/local 融合保持不变。四个局部 Token 是潜在模态特征，不解释为具体 ROI。

## 配置与数学机制

F0 冻结实验五 E0/P1 的 30 折结果。F1 保留原始 P1 并加入正确模态的双主题分支；F2 仅将新增分支按 PHENO←FUNC、ANAT←Correlation、FUNC←PHENO、Correlation←ANAT 错配；F3 将八个主题平均后复制到所有模态和主题；F4 使用真实主题但固定两个注意力为 0.5。F1-F4 使用完全相同的 9378 个新增参数。

每个模态的 4 条描述先做 L2 归一化并按主题平均、再次归一化，得到 `T∈R^(4×2×1024)`。新增共享投影 `theme_proj: 1024→9`，得到归一化主题特征 `V`。令 `q=normalize(local_x)`，计算 `scores[n,m,t,c]=q·V[m,c]`；F1/F2/F3 用 `softmax(scores)`，F4 使用 `[0.5,0.5]`。两个响应为 `u_c=a_c*(local_x⊙V_c)`，拼接后计算 `delta=0.1*semantic_fuse([u1,u2])`，最终 `local_new=local_base+delta`。`semantic_fuse` 在原始 `ModalFusion.apply(my_weight_init)` 完成后置零；新增参数 Adam 学习率为 0.0038、weight decay 为 0，原有参数仍为 0.11。

## 运行与协议

每个 config×seed 在一个进程内顺序完成 10 折；seed 为 0/1/2，450 epoch 上限、200 patience、ACC 最佳 checkpoint、原始损失、Adam、GCN、分类器和 hard-label AUC 均不变。GPU0 承担 F1 seed0/2、F2 seed1、F3 seed0/2、F4 seed1；GPU1 承担其余六个任务。

```bash
python experiment/experiment7_dual_theme_semantic_fusion/run.py --prepare
python experiment/experiment7_dual_theme_semantic_fusion/run.py --all
python experiment/experiment7_dual_theme_semantic_fusion/run.py --refresh
```

F1 每个最佳 checkpoint 另做 REAL、SWAPPED、POOLED、UNIFORM、OFF 五种反事实推理，不重新训练。`manifest.json` 记录源码、输入、Prompt、补丁和参考结果哈希；`fusion_patch.diff` 只包含本实验的局部修改。原始源码、实验一至六和 Git 历史不修改，本实验不执行 git commit 或 git push。

## 交付与验收

本次正式运行完成 12/12 任务、120/120 折、480/480 机制诊断和 150/150 反事实记录。`results/report.md` 给出主结果、每 seed 均值、配对胜/平/负、四模态诊断和反事实分析；F0 只使用冻结的实验五 E0/P1 30 折参考，不重复训练。



<!-- MASPE EXPERIMENT 1 REPLAY START -->
## 实验一重新执行：MASPE（原始 SPromptGL 协议）

- 状态：`COMPLETE`。已结束 12/12 个 config×seed 任务，记录 120/120 折。
- 原因：上一版独立验证/概率 AUC 实验不符合复现要求；完整归档于 `experiment/_archive/experiment1_maspe_strict_20261009`，历史结果不与本轮混合。
- 新实验目录：`experiment/experiment1_modality_aware_semantic_prompt`。
- 代码与协议：P0 使用根目录原码字节级副本；P1–P3 仅改隔离副本 `network.py` 的提示加载/原型分组/模态匹配。其余源码哈希与根目录一致。固定 10 折 random_state=10，原始 `val_idx == tst_idx`、ACC checkpoint 选择、hard-AUC、transductive 传播、450 epoch 与 200 patience。
- 数据：根目录 ABIDE 两个指定文件；不重做预处理。P0–P3 使用相同输入。
- 源码哈希：`source_hashes.json`；提示文本：`prompts.json`；最小 patch：`prompt_patch.diff`。
- 运行命令：`python experiment/experiment1_modality_aware_semantic_prompt/run.py --all`；调度器启动 12 个独立 worker，前 6 个绑定物理 GPU 0、后 6 个绑定物理 GPU 1，每个任务在隔离临时目录中按原顺序执行 10 折。

任务状态：
- P0 seed=0: `COMPLETE`，10/10 folds
- P0 seed=1: `COMPLETE`，10/10 folds
- P0 seed=2: `COMPLETE`，10/10 folds
- P1 seed=0: `COMPLETE`，10/10 folds
- P1 seed=1: `COMPLETE`，10/10 folds
- P1 seed=2: `COMPLETE`，10/10 folds
- P2 seed=0: `COMPLETE`，10/10 folds
- P2 seed=1: `COMPLETE`，10/10 folds
- P2 seed=2: `COMPLETE`，10/10 folds
- P3 seed=0: `COMPLETE`，10/10 folds
- P3 seed=1: `COMPLETE`，10/10 folds
- P3 seed=2: `COMPLETE`，10/10 folds

- P0 seed=0 当前逐折均值：ACC 91.73%，Hard-AUC 91.56%，SEN-ASD 89.34%，SPE-ASD 93.78%。论文参考 ACC 92.30±2.58%，AUC 92.22±2.57%。

结果文件：`results/fold_results.csv`、`results/summary.csv`、`results/report.md`。测试折仍参与 checkpoint 选择，报告会明确说明该原始评价方式。
<!-- MASPE EXPERIMENT 1 REPLAY END -->

<!-- CSMP EXPERIMENT 2 START -->
## 实验二：对比状态驱动的模态语义提示优化（CSMP）

- 状态：`COMPLETE`；已结束 6/6 个 config×seed 任务，记录 60/60 折。
- 实验目录：`experiment/experiment2_contrastive_state_prompts`。V2 为固定状态对比 Prompt，V2S 复用相同文本并固定错配模态。
- 代码与协议：根目录原始文件隔离复制，仅对 `network.py` 应用最小 Prompt patch；固定 10 折、val=test、ACC checkpoint、hard-AUC、450/200 参数均不变。
- 哈希、参考快照和报告：`source_hashes.json`、`results/reference_fold_results.csv`、`results/report.md`。
- 运行命令：`python experiment/experiment2_contrastive_state_prompts/run.py --prepare` 后执行 `python experiment/experiment2_contrastive_state_prompts/run.py --all`；GPU0 运行 V2 三个 seed，GPU1 运行 V2S 三个 seed。

任务状态：
- V2 seed=0: `COMPLETE`，10/10 folds
- V2 seed=1: `COMPLETE`，10/10 folds
- V2 seed=2: `COMPLETE`，10/10 folds
- V2S seed=0: `COMPLETE`，10/10 folds
- V2S seed=1: `COMPLETE`，10/10 folds
- V2S seed=2: `COMPLETE`，10/10 folds

实验一基线参考：P0 ACC 91.69±0.05%、Hard-AUC 91.54±0.08%；P1 ACC 91.80±0.11%、Hard-AUC 91.62±0.12%。
- 结论：V2 轻微高于 P0，且正确模态映射优于 V2S；V2 未超过 P1，建议保留为候选但暂不替换 P1。
<!-- CSMP EXPERIMENT 2 END -->

<!-- PEPCA EXPERIMENT 3 START -->
## 实验三：PEPCA（Prompt Effectiveness & Prototype Collapse Analysis）

- 状态：COMPLETE；完成 15/15 个 config×seed 任务，记录 150/150 折。
- 研究问题：原始医学 Prompt 是否有效、固定 0.5/1.0 缩放是否足够、无关文本是否改变结果，以及 MLP 后两原型是否塌缩。
- 配置：D0 原始 8 条医学 Prompt；D1 实验一 P1 的 32 条模态 Prompt；D2 固定 0.5；D3 固定 1.0；D4 固定 8 条无关自然语言 Prompt。
- 协议：根目录原始 shell、10 折 random_state=10、val=test、ACC checkpoint、450/200、hard-label AUC、原始损失/Adam/初始化/transductive 图传播；未修改根目录源码。
- 审计：GitHub main 79b3a62 与本地核心源码 SHA-256 一致；冻结 Prompt、参考折结果、patch 和数据哈希见 experiment/experiment3_prompt_effectiveness_diagnostics/provenance.json。
- 结果：experiment/experiment3_prompt_effectiveness_diagnostics/results/fold_metrics.csv；诊断：prompt_diagnostics.csv；分析：report.md。

任务状态：
- D0 seed=0: COMPLETE，10/10 folds
- D0 seed=1: COMPLETE，10/10 folds
- D0 seed=2: COMPLETE，10/10 folds
- D1 seed=0: COMPLETE，10/10 folds
- D1 seed=1: COMPLETE，10/10 folds
- D1 seed=2: COMPLETE，10/10 folds
- D2 seed=0: COMPLETE，10/10 folds
- D2 seed=1: COMPLETE，10/10 folds
- D2 seed=2: COMPLETE，10/10 folds
- D3 seed=0: COMPLETE，10/10 folds
- D3 seed=1: COMPLETE，10/10 folds
- D3 seed=2: COMPLETE，10/10 folds
- D4 seed=0: COMPLETE，10/10 folds
- D4 seed=1: COMPLETE，10/10 folds
- D4 seed=2: COMPLETE，10/10 folds
- D0 完整 seeds=3/3：ACC 91.69%、Hard-AUC 91.54%。
- D1 完整 seeds=3/3：ACC 91.80%、Hard-AUC 91.62%。
- D2 完整 seeds=3/3：ACC 91.46%、Hard-AUC 91.31%。
- D3 完整 seeds=3/3：ACC 91.54%、Hard-AUC 91.35%。
- D4 完整 seeds=3/3：ACC 91.58%、Hard-AUC 91.37%。

诊断摘要：

- D0：原型|cos| 0.8808，权重均值 0.5137，w≤0.51 76.22%，collapse(|cos|>0.98) 80.00%。
- D1：原型|cos| 0.8474，权重均值 0.5153，w≤0.51 77.43%，collapse(|cos|>0.98) 73.33%。
- D2：语义诊断 N/A 或尚未完成。
- D3：语义诊断 N/A 或尚未完成。
- D4：原型|cos| 0.8395，权重均值 0.5236，w≤0.51 65.25%，collapse(|cos|>0.98) 66.67%。

发现与结论：
- D1 的 seed 均值 ACC 91.80%、Hard-AUC 91.62%，相对 D0 为 +0.11 和 +0.08 个百分点。
- D0 相对 D2、D3、D4 的总体均值均略高，但配对 seed/逐折包含平局和反向结果，只作描述性证据。
- D0、D1、D4 的 MLP 在 3/3 个 seed 检查中有非零梯度；D2、D3 为 0/3，固定最终缩放切断语义 MLP 的有效分类梯度。
- D3 seed=0 曾留下一个被中断的重复尝试，随后重启任务完整完成；结果汇总只读取最后一次完整 10 折尝试，原始日志保留该审计记录。

后续实验建议：
- 在不使用 val=test 的独立外部测试或嵌套交叉验证协议上复核 D0/D1/D4 的小幅差异。
- 若继续机制拆解，优先分别固定 CLIP 文本特征和 MLP 投影，再比较原型相似度与门控权重，避免把两者作用混为一谈。

<!-- PEPCA EXPERIMENT 3 END -->

<!-- DAFCP EXPERIMENT 4 START -->
## 实验四：DAFCP（Disease-Aware Functional Connectivity Prompting）

- 状态：COMPLETE；C1/C2 完成 6/6 个任务，记录 60/60 新折。
- 目标：只替换 Correlation 的 8 条文本，比较 ASD 疾病相关语义与匹配的 Population-level 通用语义；PHENO/ANAT/FUNC 的 P1 文本和 PLN 全部冻结。
- 配置：C0 为实验一 P1 冻结基线；C1 为 ASD_DISEASE_FC；C2 为 MATCHED_GENERIC_FC。
- 依据：Lin et al. 2025 Autism Research DOI 10.1002/aur.70128；Nature Mental Health 2025 DOI 10.1038/s44220-025-00431-5。Prompt 只表达候选网络层面语义，不主张固定 ROI 或普遍 ASD 异常。
- 审计与哈希：见 experiment/experiment4_disease_aware_fc_prompt/manifest.json；原始源码未修改。
- 结果：experiment/experiment4_disease_aware_fc_prompt/results/report.md、fold_results.csv、diagnostics.csv。

任务状态：
- C1 seed=0: COMPLETE，10/10 folds，diagnostics=40/40
- C1 seed=1: COMPLETE，10/10 folds，diagnostics=40/40
- C1 seed=2: COMPLETE，10/10 folds，diagnostics=40/40
- C2 seed=0: COMPLETE，10/10 folds，diagnostics=40/40
- C2 seed=1: COMPLETE，10/10 folds，diagnostics=40/40
- C2 seed=2: COMPLETE，10/10 folds，diagnostics=40/40
- C0：ACC 91.80%、Hard-AUC 91.62%（完整 seeds 3/3）。
- C1：ACC 91.46%、Hard-AUC 91.29%（完整 seeds 3/3）。
- C2：ACC 91.50%、Hard-AUC 91.28%（完整 seeds 3/3）。

Correlation 诊断：
- C0：|cos| 0.7105（非零），w 均值 0.5364，w≤0.51 62.28%，非零原型塌缩率 55.56%，全零比例 10.00%。
- C1：|cos| 0.8566（非零），w 均值 0.5251，w≤0.51 62.25%，非零原型塌缩率 63.33%，全零比例 0.00%。
- C2：|cos| 0.8785（非零），w 均值 0.5265，w≤0.51 54.47%，非零原型塌缩率 50.00%，全零比例 0.00%。

关键对照结论见 report.md；仅在 C1 同时超过 C0/C2 且 seed/逐折方向稳定时，才支持 ASD 疾病语义的额外价值。
后续建议：使用独立测试或嵌套交叉验证复核；再分别冻结 CLIP 和 MLP 研究共享疾病语义与模态专属语义的融合。
<!-- DAFCP EXPERIMENT 4 END -->

<!-- SPCR EXPERIMENT 5 START -->
## 实验五：SPCR（Semantic Projection Collapse and Repair）

- 状态：COMPLETE；完成 12/12 个任务，记录 120/120 折，诊断 960/960 行。
- 设计：固定实验一 P1 的 32 条 Prompt；E0 原始 ReLU/weight_decay，E1 去最后 ReLU，E2 去 MLP weight_decay，E3 同时修复。
- 审计：4096→128→9 MLP，共 525577 参数；E2/E3 仅将 t1/t2 四个参数置于 weight_decay=0，其余 ModalFusion 与 MessagePassing0 保持原始协议。
- 分层诊断：INIT/BEST 五阶段投影、全零和反向高相似统计、参数范数及 Softmax-Max 权重；详见 manifest.json 与 report.md。
- 结果：experiment/experiment5_semantic_projection_collapse/results/fold_metrics.csv、projection_diagnostics.csv、report.md。
- GPU 分配：GPU0 承担 E0 seed0/2、E1 seed1、E2 seed0/2、E3 seed1；GPU1 承担其余六个任务。
- 原始源码与实验一至四未修改；本次不自动 git commit 或 git push。
- E0：ACC 91.80%、Hard-AUC 91.62%（完整 seeds 3/3）。
- E1：ACC 91.65%、Hard-AUC 91.48%（完整 seeds 3/3）。
- E2：ACC 91.54%、Hard-AUC 91.36%（完整 seeds 3/3）。
- E3：ACC 91.69%、Hard-AUC 91.50%（完整 seeds 3/3）。
- 推荐配置与根因判断以 results/report.md 的分层诊断和 2×2 因子效应为准。
<!-- SPCR EXPERIMENT 5 END -->

<!-- SDPG EXPERIMENT 6 START -->

实验六 SDPG（已完成 6/6 任务、60 折与 240 条诊断）：B0 冻结实验五 E0 并与实验一 P1 逐折核对；B1 signed、B2 magnitude-only，各 seed 0/1/2 十折，按两张 GPU 各三任务并行。结果见 `/data/adodas/SPromptGL-main/experiment/experiment6_semantic_direction_preserving_gate/results/report.md`，BEST 门控诊断见 `gate_diagnostics.csv`。不执行 git commit 或 git push。

<!-- SDPG EXPERIMENT 6 END -->

<!-- SDPA EXPERIMENT 8 START -->

## 实验八（方案一）：SDPA 共享疾病原型对齐

状态：完成15/15任务、150折、150条诊断和150条反事实。目录与完整中文报告：/data/adodas/SPromptGL-main/experiment/experiment8_scheme1_shared_disease_alignment/results/report.md。
动机：P1仍是已有基线；普通文本改写、ASD相关FC措辞、MLP去ReLU/去衰减、方向标量门控和双主题特征残差均没有稳定额外收益。实验七F1约91.69%/91.49%低于P1，支持改为检验保留类别方向的共享疾病语义。
设计：S0冻结E0/P1；S1正确ASD/TD对齐与决策融合；S2 ADHD/TD错配；S3中性技术两类；S4仅对齐；S5仅融合。新增Linear(36,1024,bias=False)+alpha共36,865参数，gamma=.2*tanh(alpha)，对齐lambda=.1，仅训练索引有监督；P1原路径与其32条文本保持原样。
论文依据：BrainPrompt疾病类别交叉熵、SABER受控决策语义注入、BrainSCK知识注入/激活区分。核查使用官方公开论文，不假装读取项目里不存在的上传PDF；没有构造不存在的个体认知量表。
代码版本：GitHub main 0925d5011397b0feb28d04de761b81f163f99914；具体源码/CLIP/Prompt/Patch哈希和初始化验证见manifest.json。
数据SHA256：data/ABIDE/processed_standard_data.csv = b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1。
数据SHA256：data/ABIDE/modal_feat_dict.npy = 8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e。
S0：ACC 91.80% (seed SD 0.11 pp)，Hard-AUC 91.62% (seed SD 0.12 pp)，Probability-AUC N/A，BA 91.62%。
S1：ACC 91.73% (seed SD 0.09 pp)，Hard-AUC 91.51% (seed SD 0.10 pp)，Probability-AUC 93.39%，BA 91.51%。
S2：ACC 91.73% (seed SD 0.19 pp)，Hard-AUC 91.55% (seed SD 0.16 pp)，Probability-AUC 93.70%，BA 91.55%。
S3：ACC 91.77% (seed SD 0.14 pp)，Hard-AUC 91.56% (seed SD 0.16 pp)，Probability-AUC 93.46%，BA 91.56%。
S4：ACC 91.81% (seed SD 0.20 pp)，Hard-AUC 91.63% (seed SD 0.20 pp)，Probability-AUC 93.25%，BA 91.63%。
S5：ACC 91.61% (seed SD 0.09 pp)，Hard-AUC 91.50% (seed SD 0.07 pp)，Probability-AUC 93.54%，BA 91.50%。
S1机制：ASD/TD原型cos=.946976；gamma均值+0.106731，范围+0.056056到+0.158459；负gamma折数0；P1/疾病概率相关均值0.9577。
S1反事实ADHD：翻转率0.000%，ASD概率绝对变化0.006359，ACC变化+0.000pp。
S1反事实NEUTRAL：翻转率0.230%，ASD概率绝对变化0.015893，ACC变化-0.153pp。
S1反事实SWAPPED：翻转率3.025%，ASD概率绝对变化0.034884，ACC变化-2.565pp。
S1反事实OFF：翻转率0.268%，ASD概率绝对变化0.016561，ACC变化-0.192pp。
- 完整方案是否超过P1：S1−S0 ACC -0.075 pp，seed0/1/2方向为正/负/负，按实际配对结果作描述性判断。
- 正确疾病是否超过错误疾病：S1−S2 ACC -0.000 pp，seed0/1/2方向为负/负/正，按实际配对结果作描述性判断。
- 疾病语义是否超过技术文本：S1−S3 ACC -0.038 pp，seed0/1/2方向为正/负/零，按实际配对结果作描述性判断。
- 直接决策融合是否提供收益：S1−S4 ACC -0.077 pp，seed0/1/2方向为负/负/正，按实际配对结果作描述性判断。
- 额外类别对齐监督是否提供收益：S1−S5 ACC +0.115 pp，seed0/1/2方向为正/负/正，按实际配对结果作描述性判断。
- S1的平均ACC未超过P1，当前方案不能替换P1；即使疾病分支单独分类良好，也须检查其是否只重复原始分类信息。
- S1真实ASD/TD平均margin为+1.2269/-1.4098；30/30折同时表现为ASD组正margin与TD组负margin。这说明学会类别方向，不自动证明医学知识贡献。
- S1疾病概率与P1概率平均相关系数0.9577；相关高且融合增益小支持信息重复的解释，不能据相关性单独认定因果。
- S1有0/30折gamma不为正，0/30折平均绝对融合响应小于1e-6；按实测符号和响应描述分支使用情况。
- 保留本次严格对照结果，不自动追加Prompt、损失或参数搜索。若继续研究，应先在独立验证协议下确认疾病分支互补性，再使用真实受试者脑连接构造可验证的个体化语义；没有认知量表时不伪造个体化临床文本。
- 当前四个局部Token和共享疾病分支均不能定位具体脑区异常；模型预测敏感性、疾病类别监督和医学语义正确性是不同证据。

下一步：依真实S1/S2/S3/S4/S5对照判断知识、辅助监督和容量贡献；保持当前P1结果，不自动调Prompt或超参数。若进一步研究，优先独立验证及真实脑连接驱动的个体化语义，不能伪造临床输入。
根目录源码及实验一至七未修改；临时权重仅训练期间保留，完成后清理；未执行git commit或git push。

<!-- SDPA EXPERIMENT 8 END -->

## 实验九：PDAI Prompt-Disease-Adversarial Alignment Interaction

- 代码与数据审计：根目录核心源码SHA、ABIDE输入和RN50哈希写入 `experiment/experiment9_prompt_disease_adversarial_interaction/manifest.json`；根目录源码与实验一至八保持未修改。
- 设计：P0/P1/P3 × D0/D1 × A0/A1 的12组矩阵；E0/E4/E8冻结实验一参考，E5冻结实验八S1；新训练E1/E2/E3/E6/E7/E9/E10/E11共240折。D1新增36,865参数；A1为LayerNorm(36)-Linear(36,32)-ReLU-Linear(32,4)，GRL scale=0.03。
- 运行：0:27:33；24/24任务、240/240折、240/240诊断成功，全部为attempt 1；GPU0/GPU1各分配12个任务，物理映射和哈希见manifest。
- 全组ACC/Hard-AUC（seed内十折均值，再对三个seed求均值）：

| Group | ACC | Hard-AUC |
|---|---:|---:|
| E0 | 91.690% | 91.540% |
| E1 | 91.653% | 91.470% |
| E2 | 91.499% | 91.331% |
| E3 | 91.844% | 91.684% |
| E4 | 91.805% | 91.616% |
| E5 | 91.729% | 91.514% |
| E6 | 91.729% | 91.598% |
| E7 | 91.691% | 91.529% |
| E8 | 91.537% | 91.310% |
| E9 | 91.883% | 91.692% |
| E10 | 91.652% | 91.497% |
| E11 | 91.730% | 91.542% |

- 结论：E7相对E4/E5/E6的ACC配对差分别为-0.114/-0.038/-0.038pp；没有显示P1疾病与对抗联合分支的稳定增益。D1/A1交互取决于Prompt：P0为正、P1近零、P3为负。A1判别器接近四分类机会水平，但E3−E1和E11−E9的平均两两MMD分别变化+0.0343/-0.0383，不一致；E7疾病/P1概率相关0.9599、gamma均值+0.10372，没有降低历史预测冗余。
- 限制：逐折配对只是描述性分析，不把30个共享fold seed记录视为独立样本；潜在Token不解释为ROI；疾病分支表现不证明医学语义正确。完整解释见 `results/report.md`。

## 实验十：DSSC Disease Semantic Specificity Control

- 设计：P3固定；R0/R1冻结实验九E8/E9；C1为ADHD/TD；C2为NEUTRAL_CLASS0/NEUTRAL_CLASS1。疾病投影、对齐CE、gamma和训练协议完全复用E9；对抗分支关闭。
- 运行：6个任务、60折、60条疾病诊断、360条反事实；GPU0承担C1 seed0、C1 seed2、C2 seed1，GPU1承担C1 seed1、C2 seed0、C2 seed2。

| Config | ACC | Hard-AUC |
|---|---:|---:|
| R0 | 91.537% | 91.310% |
| R1 | 91.883% | 91.692% |
| C1 | 91.691% | 91.530% |
| C2 | 91.806% | 91.613% |

- 结论以 `results/report.md` 为准；正确ASD文本与ADHD/中性文本的差异必须结合seed方向、原型几何、疾病分支/P3相关性和反事实翻转率解读。固定类别原型的可学习性不等于医学知识理解。


## 实验十一：Correlation 特征到 ROI 连接索引恢复

- 状态：`COMPLETE`。按 Nilearn 0.7.1 的相关矩阵默认收缩协方差、上三角展开和 Fisher-z 流程，从 ABIDE 原始 `.1D` 时间序列重建 871 名受试者的 256 维入选特征；256/256 唯一匹配，最大绝对误差 `3.997e-15`，整体 RMSE `7.189e-17`。受试者表型、QC 与 CSV 行映射均通过逐值核对。
- 标签恢复：111 列 ROI 中 110 个标准 Harvard–Oxford 名称和列序通过全部 871 个文件头、ABIDE 掩膜及 FSL 标签交叉核验。额外编码 3455 位于第 83 列，保留为未知；涉及它的 4 条边（特征 195/225/232/236）不进入解剖语义。其余 252 条连接可用已核实的两端名称。
- 结论边界：名称和相关系数只支持解剖标记及数值描述，不代表 ASD 特异性、因果作用或临床功能。详见 `experiment/experiment11_roi_mapping_recovery/results/report.md` 与 `results/roi_label_report.md`。

## 实验十二：AGESA 真实解剖连接语义对齐可行性

- 状态：工程 `PASS`；单折 pilot `COMPLETE`；医学语义增益 `NOT_SUPPORTED`。原始 GFR、PLN、P0 路径和分类头保持不变，只在传播后的 Correlation 特征片段加入逐连接语义残差；4 条未知端点连接残差严格为零。
- 这是 seed 0 / outer fold 1 的单折机制试验：外层训练再分 665/118 人训练/验证，88 人测试；内部验证集选择 checkpoint，与历史 `val=test` 结果不可直接比较。未扩展为 30 折，也未据测试集调参。
- F0/F1(真实解剖文本)/F2(置换解剖文本)/F3(匿名特征编号) 的测试 ACC 依次为 94.318/88.636/90.909/93.182%；BA 为 94.369/89.050/91.178/92.683%；Probability-AUC 为 96.783/96.783/96.938/97.250%。F1 相对 F0/F2/F3 的 BA 差为 -5.319/-2.128/-3.633 pp，未显示真实解剖语义的分类收益。
- F1 平均相对残差强度 1.855%，文本替换会改变残差和概率，但 REAL 与 SHUFFLED 的概率均值绝对差仅 `1.548e-3`；模型对分支有数值响应不等于医学语义有效。完整设置、机制诊断和限制见 `experiment/experiment12_roi_semantic_alignment_feasibility/results/report.md`。

## 实验十三：原协议复现与患者个体化功能连接 Prompt

- 状态：`COMPLETE`；8/8 配置、24/24 任务、240/240 folds。训练在 GPU0/GPU1 各完成 120 折，正式训练壁钟时间 109.61 分钟；最终 240 条诊断及 150 条 P4 同 checkpoint 反事实记录完整。源码、数据、ROI 映射和历史实验哈希校验见 `experiment/experiment13_patient_specific_fc_prompt/manifest.json`。
- 恢复原始 10 折 `random_state=10`、3 seeds、`val=test`、ACC 选择 checkpoint、原始损失与超参数。该协议让测试折参与 checkpoint 选择，且继承全队列标准化/RFE 的信息共享风险；结果用于本协议内的配对比较，不作为无偏的独立泛化估计。四个局部 token 是潜在表示，不解释为 ROI。
- F0 为 P0 原始路径；F1/F2/F3 分别为真实 ROI 语义、跨边错配语义和匿名特征编号；P4/P5/P6/P7 分别测试患者自身状态、患者状态错配、匿名编号和 ROI 名称错配。除 F0 外各组均为同结构的 32,865 参数残差分支；每折用训练受试者估计连接状态，3455 相关的四条边保持零残差。

| Config | ACC | BA / Hard-AUC | Probability-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|---:|
| F0 | 91.691% | 91.539% | 93.335% | 89.661% | 93.418% |
| F1 | 91.499% | 91.340% | 93.104% | 89.333% | 93.347% |
| F2 | 91.538% | 91.365% | 93.382% | 89.169% | 93.562% |
| F3 | 91.346% | 91.177% | 93.346% | 88.852% | 93.501% |
| P4 | 91.231% | 91.066% | 93.320% | 88.923% | 93.210% |
| P5 | 91.423% | 91.242% | 93.376% | 88.921% | 93.563% |
| P6 | 91.122% | 90.910% | 93.103% | 88.033% | 93.787% |
| P7 | 91.538% | 91.363% | 93.528% | 89.091% | 93.634% |

- 核心配对差（百分点）：F1−F0 的 ACC/BA/Probability-AUC 为 -0.192/-0.199/-0.232；F1−F2 为 -0.039/-0.025/-0.279；F1−F3 为 +0.152/+0.164/-0.242。P4−F0 为 -0.460/-0.473/-0.015；P4−F1 为 -0.267/-0.274/+0.217；P4−P5 为 -0.192/-0.176/-0.055；P4−P6 为 +0.110/+0.157/+0.217；P4−P7 为 -0.307/-0.296/-0.208。逐 seed 和逐折配对见实验报告；三个 seeds 共用折划分，不作为 30 个独立样本进行显著性推断。
- P4 的平均相对残差为 2.168%，有效梯度记录 210/210，分支并非近零。REAL 与状态/名称反事实会改变预测，但影响弱于关闭整个分支：患者状态错配、匿名编号、ROI 名称错配、OFF 的 ASD 概率平均绝对变化分别为 0.002494、0.002391、0.003756、0.012355；P4 未稳定优于真实解剖静态组或错配患者状态组。
- 结论：工程有效性 `PASS`；患者自身状态效应、ROI 身份效应及独立语义诊断价值均为 `NOT_SUPPORTED`。现有结果不能将数值重加权/特征编码响应解释为临床语义贡献，也不支持据此添加全局疾病分支。详细结果见 `experiment/experiment13_patient_specific_fc_prompt/results/report.md`。
- 发布记录：实验十三目录已上传至 [SPromptGL GitHub](https://github.com/tangaii/SPromptGL/tree/main/experiment/experiment13_patient_specific_fc_prompt)，提交 `a1da09b40f95356ad2c995e11cc324c2a004489d`；共 40 个文件，整个 `cache` 目录排除。实验报告与代码保留在本地路径及仓库实验目录。


## 实验十四：Hierarchical Semantic Prompt Design and Cross-Level Interaction

日期2026-10-13T18:09:38Z；目录 `/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt`；COMPLETE，14组42任务420折，双GPU各21任务210折，耗时74.62分钟；420诊断、150反事实。未commit/push。

本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；远端main `a1da09b40f95356ad2c995e11cc324c2a004489d`；源/数据SHA256：`{"main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67", "ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763", "data/ABIDE/processed_standard_data.csv": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1", "data/ABIDE/modal_feat_dict.npy": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e", "CLIP/clip/clip.py": "091f5f4a590efe5892763780967af11997a3831407bd300cd416eb241e63d639", "CLIP/clip/model.py": "bd74aaecab16cb7809ae69bf2ce41adbac64e13fe4e9af6c196cc22af41c5c91", "CLIP/clip/simple_tokenizer.py": "d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad"}`。Prompt `7cda11ceff8ae1a72c075318e9e189cd415953353653aaa23a4dabcda2550d87`，CLIP `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`，patch `a0a54b91637e6bff23225ed2857fba40df2ce4e055f6067ca98a04790bf4d4d9`。

|组|Local|Disease|ACC|BA|Hard-AUC|Prob-AUC|SEN|SPE|
|---|---|---|---:|---:|---:|---:|---:|---:|
|E00|P0|OFF|91.691±0.054|91.539±0.084|91.539±0.084|93.335±0.415|89.661±0.646|93.418±0.510|
|E01|OLD_P1|OFF|91.499±0.163|91.295±0.231|91.295±0.231|93.597±0.087|88.593±1.225|93.996±0.764|
|E02|NEW_P1_SHARED|OFF|91.423±0.143|91.237±0.166|91.237±0.166|93.571±0.206|88.762±0.573|93.713±0.268|
|E03|NEW_P3_SHARED|OFF|91.538±0.217|91.350±0.218|91.350±0.218|93.419±0.174|88.996±0.467|93.703±0.408|
|E04|NEW_P3_SPECIFIC|OFF|91.576±0.195|91.365±0.212|91.365±0.212|93.393±0.197|88.591±0.532|94.138±0.201|
|E05|NEW_P3_SPECIFIC|HISTORICAL|91.614±0.281|91.470±0.286|91.470±0.286|93.520±0.053|89.587±0.351|93.353±0.307|
|E06|NEW_P3_SPECIFIC|FC_CENTRIC|91.614±0.094|91.399±0.095|91.399±0.095|93.724±0.292|88.518±0.582|94.280±0.538|
|E07|NEW_P3_SPECIFIC|BALANCED|91.729±0.094|91.552±0.109|91.552±0.109|93.594±0.211|89.252±0.506|93.853±0.370|
|E08|NEW_P3_SPECIFIC|SELECTIVE|91.691±0.217|91.482±0.190|91.482±0.190|93.545±0.035|88.756±0.301|94.208±0.632|
|E09|NEW_P3_SPECIFIC|CLINICAL_CORE|91.614±0.094|91.450±0.138|91.450±0.138|93.615±0.189|89.258±0.630|93.642±0.358|
|E10|NEW_P3_SPECIFIC|NEUTRAL|91.652±0.108|91.456±0.133|91.456±0.133|93.550±0.149|88.846±0.531|94.067±0.268|
|E11|P0|SELECTIVE|91.729±0.163|91.534±0.168|91.534±0.168|93.418±0.174|89.083±0.352|93.986±0.300|
|E12|NEW_P1_SHARED|SELECTIVE|91.385±0.163|91.183±0.151|91.183±0.151|93.432±0.225|88.512±0.233|93.853±0.261|
|E13|NEW_P3_MISMATCHED|SELECTIVE|91.730±0.248|91.480±0.230|91.480±0.230|93.596±0.233|88.256±0.305|94.704±0.534|

主要配对与机制：
按BA描述性排序，局部组最高为E00（P0），新疾病方案最高为E07（BALANCED），全部组合最高为E07。排名本身不证明稳定最优，完整ACC/概率AUC及seed方向均须同时判断。
新P1对P0：ACC/BA/Prob-AUC差-0.2682pp/-0.3019pp/+0.2361pp，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。
新P3对P0：ACC/BA/Prob-AUC差-0.1149pp/-0.1745pp/+0.0583pp，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。
模态专属对共享：ACC/BA/Prob-AUC差+0.0379pp/+0.0152pp/-0.0260pp，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。
正确模态对应：ACC/BA/Prob-AUC差-0.0387pp/+0.0018pp/-0.0509pp，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。
正确疾病文本对中性：ACC/BA/Prob-AUC差+0.0387pp/+0.0254pp/-0.0042pp，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。
疾病层对新P3的互补：ACC/BA/Prob-AUC差+0.1149pp/+0.1170pp/+0.1519pp，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。
BA最高组合E07相对E00：+0.0131pp，seed0/1/2 +0.0242/+0.2383/-0.2232pp。
BA最高组合E07相对E01：+0.2576pp，seed0/1/2 +0.6191/+0.2041/-0.0505pp。
E08疾病/原分类器概率相关平均0.96094，融合贡献绝对均值0.079819；提示明显冗余，类别监督可将静态文本学成类别代码。
同checkpoint局部错配/中性疾病/疾病类交换的平均概率变化为0.000241/0.014388/0.032112，平均翻转率为0.000000/0.001149/0.017607。敏感性与重新训练语义必要性是不同问题。
疾病ON的270折中，融合贡献近零(<1e-8)为0折；投影/alpha均有有效有限梯度。局部MLP即使获得早期梯度，后期也可能出现范数收缩或两原型塌缩；分模态表为实际证据。
四模态只有原型和门控诊断，没有逐模态消融，不能断言某模态主导性能。特别是Correlation最大门控或最低cos都不等于该模态的因果收益。局部两类max门控无方向，不可解释为识别了患者特定脑区异常。
正确疾病内容与模态对应未同时给出三seed一致的BA收益；本轮不足以支持继续通过措辞搜索获取语义创新。建议保留P0/旧P1作为参照，下一步优先独立验证与局部—疾病对齐机制检验，避免继续自动改写Prompt。
新P3专属相对新P1共享的ACC/BA/Prob-AUC差为+0.1533pp/+0.1274pp/-0.1778pp；ACC与BA三个seed均提高，概率AUC方向不一致且均值降低，因此没有一致的全指标赢家。
四个新疾病方案中BA最高为E07/BALANCED，Probability-AUC最高为E06/FC_CENTRIC；Balanced相对原P0的ACC/BA/Prob-AUC差为+0.0383pp/+0.0131pp/+0.2592pp。BA及连续AUC的seed方向不一致，不能宣布最优疾病文本。
历史P1参考ACC/Hard-AUC为91.8047/91.6163%，本轮E01为91.4995/91.2949%。E01逐字复用旧P1文本和投影数学，但fold2–10使用E00配对入口；历史P1使用自身连续十折随机流。两者不是相同初始化轨迹的重复。本轮正式对照以重训E01为准，历史数值单独保留，不混入420折。E07相对历史P1参考ACC仍为-0.0753pp；该历史差异只作描述，非同初始化配对。
E08排除近零向量对后，四模态诊断为：PHENO: 有效非零原型cos=0.97389，gate=0.50548，近均匀=88.22%；ANAT: 有效非零原型cos=0.99802，gate=0.50177，近均匀=96.72%；FUNC: 有效非零原型cos=0.99808，gate=0.50217，近均匀=94.44%；Correlation: 有效非零原型cos=0.88957，gate=0.52087，近均匀=64.76%。ANAT/FUNC的两类方向高度接近，Correlation门控偏离.5最大；这只说明机制响应较强，没有模态消融支持它主导性能。包含零向量的cos均值会被拉低，不能把低cos直接当作语义分离。
E08最终疾病融合REAL−OFF的ACC/BA/Prob-AUC变化为+0.1533pp/+0.1426pp/-0.0018pp；OFF与独立E04重训模型的差为-0.0383pp/-0.0256pp/+0.1536pp。OFF保留辅助监督及融合联合训练改变的全部参数，后者不是纯辅助CE的因果估计。最终融合主要改变少量分类阈值结果，没有可见的额外连续AUC增益。
跨层增益差P3相对E11：BA +0.1219pp、三seed均正；Prob-AUC +0.0690pp，seed0/1/2 +0.6702/-0.3904/-0.0729pp。这提供BA层面的描述性线索，但连续排序收益不一致，且不能替代E08相对E04的绝对改善检验，暂不支持稳定跨层协同。
跨层增益差P3相对E12：BA +0.1719pp、三seed均正；Prob-AUC +0.2916pp，seed0/1/2 -0.1433/+1.0279/-0.0098pp。这提供BA层面的描述性线索，但连续排序收益不一致，且不能替代E08相对E04的绝对改善检验，暂不支持稳定跨层协同。
270个疾病ON折的gamma符号分布为正270/零0/负0，实际贡献均非近零；所有420折局部MLP均有非零有限梯度。有效梯度、非零响应和文本替换敏感性支持工程连通性，不能替代正确/中性与正确/错配的语义价值对照。
修复记录：准备期发现的一句NEW_P3_SHARED转录差异已在该组训练前改回任务原文，并核对80句新增文本；E00已保存的model/Adam/RNG未变。OSS挂载的短暂rename可见性及checkpoint阻塞通过有界读取和本机临时state修正；12个已完成E00折不重跑，迁移26个文件逐字节SHA256一致。结束后的维护只处理完成任务跳过、报表哈希和冻结文件复用，不改变训练数学，也未增加训练。

全部预设配对/跨层增益差及三seed方向、胜平负见report；公共初始化、入口RNG、梯度、文本长度、E00历史parity与checkpoint恢复均通过。固定14组，无个体化模块，无超参数搜索；原GCN/RES/损失和val=test保留。val=test、全局标准化及RFE共享信息限制独立泛化推断，专属/共享存在文本细化与数量混杂。实验一至十三与根源码不变，结束清理临时训练状态。

结果：`/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/results/fold_metrics.csv`、`/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/results/diagnostics.csv`、`/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/results/counterfactual.csv`、`/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/results/report.md`、`/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/manifest.json`。重跑 `python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py all`；恢复 `python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py resume`；刷新报告 `python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py report`。


## 实验十五：AARSP 解剖锚定ROI局部语义提示学习

状态COMPLETE；8组24任务240折，240诊断、150个A5同checkpoint反事实。GPU0/GPU1各12任务120折，4 workers/card、2线程/worker；正式耗时20.54分钟，完成时间2026-10-14T08:41:49.496481+00:00。

目的：把原无真实ROI身份的局部SPGS替换为HO患者内图语义；GFR输出后、PLN输入前只增加Correlation段残差。110节点、252可信连接，未知特征195/225/232/236直接增量为0。数值统计→32维节点→冻结1024维CLIP与可学习查询匹配→一层连接权重消息→对称边解码；beta=.3*tanh(alpha)、alpha0。A2–A7各新增38,242参数，Adam .0038/.11，原参数与CAM/RES/分类器保持原计算。

区别：实验十二/十三为独立逐边静态/LOW-MID-HIGH残差；十四为潜在Token与全局疾病头。本轮用真实HO节点聚合并跨邻居传播，未复制AAL、超图、全局疾病/额外损失或措辞搜索。八组A0原P0、A1关闭SPGS保留RES、A2常量数值图、A3匿名身份、A4真名称、A5真名称+功能、A6固定无固定点错配、A7自身消息。

协议：原871人/48-6-10-256、固定RFE/输入、10折seed0/1/2、val=test、450/200、ACC严格提高选checkpoint及原Hard-AUC。每config×seed起始播种一次并连续十折，新增初始化fork_rng；相同入口下主干一致，后续各组因早停自然分叉，不恢复A0折入口。根源码与实验一至十四494个保护文件哈希均保持不变。

本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`，GitHub核验main `9de3b1561a6ac107b0f4dad8ce01df74638c0a4c`；代码、三论文、数据、映射、CLIP、Prompt、补丁与split SHA256见`/data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/manifest.json`。

|组|设计|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN-ASD %|SPE-TD %|
|---|---|---:|---:|---:|---:|---:|---:|
|A0|原始P0|91.6910±0.0542|91.5394±0.0841|91.5394±0.0841|93.3352±0.4154|89.6606±0.6456|93.4181±0.5102|
|A1|无SPGS保留RES|91.5378±0.1954|91.3490±0.2062|91.3490±0.2062|93.5704±0.3043|88.9207±0.7145|93.7774±0.6062|
|A2|数值图/常量文本|91.4995±0.0939|91.3293±0.1115|91.3293±0.1115|93.4008±0.2196|89.0894±0.8823|93.5692±0.7594|
|A3|匿名ROI身份|91.5761±0.1434|91.3889±0.1272|91.3889±0.1272|93.5230±0.0746|88.9248±0.4238|93.8529±0.5316|
|A4|真实HO名称|91.8056±0.1439|91.6322±0.1591|91.6322±0.1591|93.4817±0.1330|89.4146±0.7074|93.8498±0.5655|
|A5|真实HO名称+功能|91.8060±0.1954|91.6172±0.1921|91.6172±0.1921|93.5741±0.1418|89.1687±0.2988|94.0657±0.1996|
|A6|固定语义错配|91.6527±0.1434|91.4706±0.1313|91.4706±0.1313|93.5142±0.2560|89.0915±0.0050|93.8498±0.2642|
|A7|真实文本/仅自身消息|91.6149±0.1622|91.4013±0.1756|91.4013±0.1756|93.4194±0.3410|88.5935±0.7000|94.2091±0.5231|

关键比较（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE，百分点）：
A5−A0：+0.11494/+0.07784/+0.07784/+0.23894/-0.49187/+0.64755pp；三seed及胜平负见report。
A5−A2：+0.30651/+0.28786/+0.28786/+0.17337/+0.07927/+0.49645pp；三seed及胜平负见report。
A5−A3：+0.22989/+0.22833/+0.22833/+0.05119/+0.24390/+0.21277pp；三seed及胜平负见report。
A5−A4：+0.00044/-0.01504/-0.01504/+0.09240/-0.24593/+0.21585pp；三seed及胜平负见report。
A5−A6：+0.15326/+0.14654/+0.14654/+0.05998/+0.07724/+0.21585pp；三seed及胜平负见report。
A5−A7：+0.19114/+0.21591/+0.21591/+0.15471/+0.57520/-0.14339pp；三seed及胜平负见report。

反事实：
SHUFFLED：翻转0.00000%，ASD概率绝对变化0.000000e+00；ACC/BA/Prob-AUC相对REAL差+0.00000/+0.00000/+0.00000pp。
CONSTANT：翻转0.00000%，ASD概率绝对变化0.000000e+00；ACC/BA/Prob-AUC相对REAL差+0.00000/+0.00000/+0.00000pp。
OFF：翻转0.19157%，ASD概率绝对变化1.404159e-03；ACC/BA/Prob-AUC相对REAL差-0.19157/-0.20732/+0.00533pp。
SELF_ONLY：翻转0.00000%，ASD概率绝对变化0.000000e+00；ACC/BA/Prob-AUC相对REAL差+0.00000/+0.00000/+0.00000pp。

科学判断：三分类指标均值同时提高=True，三个seed方向一致=False；医学语义必要性=NOT_SUPPORTED；结构必要性=NOT_SUPPORTED。非零梯度、原型几何和替换敏感性不能代替正确/错配、匿名和常量的重训对照。
暂不建议进入ROI条件化全局ASD/TD疾病对齐；保留完整结果，不自动扩展或调参。
完工复核：A5的三指标均值小幅提高，但seed0均下降；SEN-ASD三个seed均下降，均值-0.49187pp。A4/A5各正确分类2399/2613人次，ACC差+0.00044pp来自折大小及等权平均，功能描述未形成一致额外收益。30/30个A5 checkpoint中，错配/常量文本改变内部ROI响应（平均绝对变化0.029734/0.041551），回写残差、连续概率和类别却完全不变；SELF_ONLY同样无变化。OFF改变全部30折的连续概率、4折类别，折均翻转率0.19157%，支持残差的数值作用，不能归因于正确医学知识。
机制：冻结原型未发生训练诱导的趋同，各模块训练过程中有非零有限梯度；A5最佳checkpoint的查询/消息权重范数均值0.007235/0.001285，实际邻居消息绝对均值5.70067e-9。A5残差相对L2均值0.61387%，28/30折近似共有边偏移（共有比例≥0.99），没有近零残差折；全部新增分支共4/180折残差近零。内部响应有区别，但语义和传播差异没有传递到最终预测。保留原weight_decay=0.11，没有按结果调参，也未单独验证其因果作用。
GPU0/GPU1累计训练worker时间分别77.78006/75.54906分钟；由于任务并行，正式墙钟仍为20.54分钟。后续先验证所选checkpoint对文本与邻居的实际依赖，再考虑单独立项全局疾病语义对齐。
失败与恢复：{}；已完成折不重训，model/两Adam/完整RNG恢复预检通过，无失败attempt/重复指标混入；完成后清理临时源码、文本缓存和checkpoint，无残留本实验训练进程。
限制：val=test、共享上游标准化/RFE及三seed复用fold保留；A2文本几何、A7传播规则不同；文本功能描述保守且非临床专家认证，ROI响应不是ASD因果脑区。
目录 `/data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt`；报告 `/data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/results/report.md`；manifest `/data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/manifest.json`。运行 `python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py all`，恢复 `python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py resume`。未执行git commit/push。


## 实验十六：ROI语义通路修复与医学语义必要性验证

COMPLETE：10组30任务300fold，300诊断、B5六条件180条反事实；GPU0/1各15任务150fold，4worker/卡、2线程/worker，正式77.61分钟，完成2026-10-14T11:17:12.161544+00:00。

前十五轮及既有handoff原字节完整保留。十五A5文本/邻居干预delta与概率逐值不变，query收缩、消息小、FC直通及decoder bias可能绕过文本；本轮不再改Prompt措辞，而检验路径修复。
B0原P0、B1精确十五A5、B2新组WD0、B3关闭FC直通、B4两项；B5新耦合FUNCTION、B6CONSTANT、B7ANONYMOUS、B8SHUFFLED、B9SELF_ONLY。旧分支38,242，新耦合39,201参数；原组WD.11不变，B1/B3新组.11，其余新组0。
新耦合：中心化cosine→tanh(score/.2)×数值节点→正负边权分开归一化→三bias-free映射→tanh→对称64维解码（无FC）→未detach零参考→.3tanh(alpha)残差，alpha0，110HO/252边，四未知delta0。没有新增疾病Prompt、损失、global head或超参数搜索。
版本：本地1167dd392837a02eb4f83481ae49f02607a9dae5，GitHub核验52f9ae9cbe0a3b49ded4a78e51a358dda9632604；六核心、三论文、输入/RFE、映射、CLIP、十五Prompt、源码/patch/split哈希见manifest。保护实验一至十五521文件。
协议：871人，48/6/10/256；450epoch、wait>200、严格ACC首最佳；val=test、ASD0、Hard-AUC原口径，六指标。每config×seed连续十fold，仅初始播种，fork_rng隔离新模块；早停后自然RNG分叉，不伪称全部fold同初始化。

|组|设计|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN-ASD %|SPE-TD %|
|---|---|---:|---:|---:|---:|---:|---:|
|B0|原始P0|91.6910±0.0542|91.5394±0.0841|91.5394±0.0841|93.3352±0.4154|89.6606±0.6456|93.4181±0.5102|
|B1|旧ROI/WD.11/FC直通|91.8060±0.1954|91.6172±0.1921|91.6172±0.1921|93.5741±0.1418|89.1687±0.2988|94.0657±0.1996|
|B2|旧ROI/WD0/FC直通|91.6144±0.0939|91.4559±0.0662|91.4559±0.0662|93.2676±0.2096|89.4167±0.3124|93.4952±0.4409|
|B3|旧ROI/WD.11/关闭FC直通|91.5374±0.1081|91.4040±0.1171|91.4040±0.1171|93.4302±0.1723|89.7459±0.4238|93.0620±0.3629|
|B4|旧ROI/WD0/关闭FC直通|91.4995±0.2816|91.3302±0.2128|91.3302±0.2128|93.4464±0.1538|89.1667±0.7121|93.4937±1.1360|
|B5|新耦合/FUNCTION|90.8499±0.2712|90.6942±0.2358|90.6942±0.2358|92.9061±0.3921|88.6748±0.3103|92.7135±0.7651|
|B6|新耦合/CONSTANT|91.2700±0.4307|91.1237±0.4083|91.1237±0.4083|93.2565±0.1643|89.1037±0.1991|93.1437±0.6217|
|B7|新耦合/ANONYMOUS|90.6953±0.5857|90.5182±0.6138|90.5182±0.6138|92.8077±0.6236|88.2581±0.9385|92.7783±0.3677|
|B8|新耦合/SHUFFLED|90.3148±0.3009|90.1643±0.2932|90.1643±0.2932|92.8211±0.1502|88.1809±0.1269|92.1477±0.5048|
|B9|新耦合/SELF_ONLY|90.6971±0.5615|90.5842±0.5753|90.5842±0.5753|92.9397±0.4212|89.1687±0.8164|91.9997±0.3568|

配对主结果（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE，pp）：
B2−B1：-0.19157/-0.16125/-0.16125/-0.30651/+0.24797/-0.57046；各seed及胜平负见report。
B3−B1：-0.26863/-0.21323/-0.21323/-0.14391/+0.57724/-1.00370；各seed及胜平负见report。
B4−B2：-0.11494/-0.12577/-0.12577/+0.17880/-0.25000/-0.00154；各seed及胜平负见report。
B4−B3：-0.03788/-0.07378/-0.07378/+0.01620/-0.57927/+0.43170；各seed及胜平负见report。
B5−B4：-0.64960/-0.63601/-0.63601/-0.54033/-0.49187/-0.78014；各seed及胜平负见report。
B5−B6：-0.42015/-0.42951/-0.42951/-0.35040/-0.42886/-0.43016；各seed及胜平负见report。
B5−B7：+0.15456/+0.17596/+0.17596/+0.09843/+0.41667/-0.06475；各seed及胜平负见report。
B5−B8：+0.53509/+0.52987/+0.52987/+0.08503/+0.49390/+0.56583；各seed及胜平负见report。
B5−B9：+0.15282/+0.10997/+0.10997/-0.03364/-0.49390/+0.71385；各seed及胜平负见report。
B5−B0：-0.84117/-0.84518/-0.84518/-0.42910/-0.98577/-0.70459；各seed及胜平负见report。
B5−B1：-0.95611/-0.92302/-0.92302/-0.66804/-0.49390/-1.35214；各seed及胜平负见report。

2×2主效应与交互（相同六指标，pp）：
WD主效应(0−.11)：-0.11472/-0.11752/-0.11752/-0.14516/-0.16565/-0.06938。
FC直通关闭主效应：-0.19179/-0.16950/-0.16950/+0.01745/+0.16362/-0.50262。
交互(B4−B3)−(B2−B1)：+0.15369/+0.08746/+0.08746/+0.32271/-0.82724/+1.00216。

B5反事实：
SHUFFLED：翻转5.85510%；Δ概率均值5.540049e-02/最大9.597066e-01；Δdelta均值1.716494e-02/最大3.936933e-01；ACC/BA/Prob-AUC干预−REAL-3.48398/-3.48531/-0.83086pp。
ANONYMOUS：翻转4.82236%；Δ概率均值5.021094e-02/最大9.714450e-01；Δdelta均值1.967197e-02/最大3.936933e-01；ACC/BA/Prob-AUC干预−REAL-3.36904/-3.37770/-0.60384pp。
CONSTANT：翻转4.51367%；Δ概率均值4.703803e-02/最大9.614444e-01；Δdelta均值1.921442e-02/最大3.936933e-01；ACC/BA/Prob-AUC干预−REAL-2.98372/-2.95613/-0.44049pp。
SELF_ONLY：翻转6.92746%；Δ概率均值6.689315e-02/最大9.602090e-01；Δdelta均值2.409882e-02/最大3.936933e-01；ACC/BA/Prob-AUC干预−REAL-4.78622/-4.90346/-1.15144pp。
OFF：翻转7.88488%；Δ概率均值7.793347e-02/最大9.598432e-01；Δdelta均值1.221209e-01/最大1.968466e-01；ACC/BA/Prob-AUC干预−REAL-5.20724/-5.76907/-0.25077pp。

数值通路：B5在30/30个最佳checkpoint上对错配文本产生非零delta及预测变化；其中概率平均变化>1e-8有30/30折。邻居替换引起非零预测变化有30/30折。有效delta均值<1e-8的B5折数为0/30。实验十五A5正确/错配/常量/SELF_ONLY的delta和预测变化曾逐值为0；据此区分数值连接恢复与医学必要性。
旧分支2×2：取消WD的B2−B1 ACC为-0.19157pp；仅关FC直通B3−B1为-0.26863pp；两项同时改变B4−B1为-0.30651pp。必须结合参数范数、消息幅度、文本替换影响及交互项，不将单一分类变化等同于因果机制。
分类：B5−B0的ACC/BA/Prob-AUC为-0.84117/-0.84518/-0.42910pp；三指标均值同时提高=False，各指标三个seed均为正=False。
医学语义必要性：B5稳定超过B6常量、B7匿名和B8错配（三指标、三seed全正）=False；同checkpoint REAL−SHUFFLED满足同一方向条件=True。合并判断=NOT_SUPPORTED。任何非零梯度或文本敏感性本身不证明正确医学语义有效。
邻居必要性：B5稳定超过B9=False；即使SELF_ONLY干预显著改变预测，也可能说明分布变化敏感，不能代替B9重训对照。
下一步：当前证据不足以进入全局疾病语义对齐。保留这次通路修复及负结果，不自动增加Prompt、损失、模型或超参数版本。

逐层因果证据（均为30折平均，最大变化取跨fold最大）：
B1：query范数/初始=3.922880e-04，数值编码范数/初始=3.146995e-02；消息幅度=5.700674e-09；edge signal绝对均值=7.275621e-02；零输入decoder输出绝对均值=7.275621e-02；共同边delta能量=1.000000；文本替换Δedge signal均值=0.000000e+00，Δdelta均值/最大=0.000000e+00/0.000000e+00，Δ概率均值/最大=0.000000e+00/0.000000e+00，有实质概率变化（均值>1e−8）=0/30折。
B2：query范数/初始=1.052907e+00，数值编码范数/初始=1.201142e+00；消息幅度=3.599296e-01；edge signal绝对均值=2.241044e+01；零输入decoder输出绝对均值=4.058255e-01；共同边delta能量=1.000000；文本替换Δedge signal均值=5.097669e-02，Δdelta均值/最大=2.062915e-12/4.470348e-08，Δ概率均值/最大=9.455957e-11/1.192093e-07，有实质概率变化（均值>1e−8）=0/30折。
B3：query范数/初始=3.749028e-04，数值编码范数/初始=3.270408e-02；消息幅度=3.485323e-09；edge signal绝对均值=7.301696e-02；零输入decoder输出绝对均值=7.301696e-02；共同边delta能量=1.000000；文本替换Δedge signal均值=0.000000e+00，Δdelta均值/最大=0.000000e+00/0.000000e+00，Δ概率均值/最大=0.000000e+00/0.000000e+00，有实质概率变化（均值>1e−8）=0/30折。
B4：query范数/初始=1.077772e+00，数值编码范数/初始=1.218468e+00；消息幅度=3.988871e-01；edge signal绝对均值=2.519943e+01；零输入decoder输出绝对均值=4.189735e-01；共同边delta能量=0.997908；文本替换Δedge signal均值=1.797640e-01，Δdelta均值/最大=3.547456e-05/3.065921e-01，Δ概率均值/最大=2.252893e-04/9.503576e-02，有实质概率变化（均值>1e−8）=1/30折。
以上证实WD与参数收缩有关：B1/B3远小于初始范数，B2/B4保留范数。然而取消WD没有普遍恢复最终文本依赖；中间edge signal仍可变化，而delta变化被显著压低。B2/B4较大的edge signal与末端tanh饱和一致，共同边能量说明大量空间差异已消失。关闭FC直通也未普遍消除退化，因此不能把FC直通唯一认定为根因。独立偏置效应与(1+r)绕行效应没有单独重训因子，仍未唯一证实。
B5修复状态：query范数/初始=1.141254，原始正/负邻居消息幅度=1.026197e-01/4.423040e-02，映射后正/负消息=3.070069e-01/1.549830e-01，抵消比例=0.102186，共同边delta能量=0.842484，delta相对GFR Correlation特征L2=0.323259。零节点残差严格为0，并未强制所有受试者残差为0。
正确对应关系的有限正证据：B5−B8 ACC/BA/Prob-AUC=+0.53509/+0.52987/+0.08503pp；ACC三个seed=+0.57210/+0.57471/+0.45846pp，BA三个seed=+0.60481/+0.57609/+0.40871pp；ACC/BA三seed方向一致，但Prob-AUC三seed=-0.20948/+0.07805/+0.38651pp，仍不一致。同checkpoint REAL−SHUFFLED三个seed的三项指标均为正，说明正确已学习对应关系对当前模型有作用。
因此不能概括为完全没有任何对应关系证据；但B5低于原始P0和常量控制，真实文本对匿名控制及邻居传播优势也不稳定。预先规定的全部控制×三指标×三seed联合标准未通过。可支持“对应关系有局部作用”，尚不能支持“完整医学知识产生稳定独立诊断增益”。
直接回答：优化收缩是否缓解与数值通路是否恢复，按上述参数范数及干预链分别判断；正确医学语义是否增加诊断信息，必须用B5−B6/B7/B8及同checkpoint REAL−SHUFFLED联合判定。没有把“文本进入分类器”包装成“临床知识正确”。
OFF导致已训练模型性能下降，只说明已训练主干/分类器与残差共同适应；它不能解释为相对重新训练P0的净增益。B5的相对L2为每fold受试者均值，不能据其小于1保证每个受试者的残差均不大于原特征。新方案仍有部分共同边偏移和末端tanh饱和，恢复文本敏感性不等于所有数值退化均已消除。

B0与十三F0/十四E00/十五A0六指标逐fold完全一致；B1与十五A5六指标、初始化、入口RNG、best/总epoch完全一致。九E9历史六指标均值见report，不能当作同初始化严格对照。
预检PASS：历史两步Adam、初始化、梯度、Gram/零参考值与梯度、完整恢复、零输入、未知维、对称边与合成语义/邻居预测链；正式近零或负结果原样保留。三seed共用fold，val=test及原上游处理限制仍存在，不能外推独立泛化或ASD因果病灶。
任务恢复记录{}；仅最新原子提交，完成后临时模型/源码/cache清除，无本实验训练进程。目录/data/adodas/SPromptGL-main/experiment/experiment16_roi_semantic_pathway_repair；报告/data/adodas/SPromptGL-main/experiment/experiment16_roi_semantic_pathway_repair/results/report.md。未执行git commit/push。

历史E9描述性六指标（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE）：91.88262/91.69151/91.69151/93.82272/89.24797/94.13506%；B5−E9描述性差值-1.03274/-0.99735/-0.99735/-0.91661/-0.57317/-1.42152pp。旧E9随机流/源码机制不同，此差值不作为严格配对因果比较。
工程修复：预检中单行零参考广播与批量解码的浮点舍入出现约1e−8差异，改为同形状零输入且不detach，严格零节点零残差校验通过，数学与参数量不变。完成后补齐runner在BETWEEN_FOLDS checkpoint已提交而JSON尚未提交时的恢复同步；模拟最终fold的中断窗口验证通过。没有改动模型、优化器、RNG、指标或任何正式训练fold。


<!-- experiment17_disease_semantic_selective_alignment -->

## 实验十七（DS3A-v2）正式完成

MICCAI 2027第一创新模块候选：疾病语义与质量条件化的选择性Shared/Private对齐。下文保留完整正式统计和负结果，未执行额外Prompt或超参数搜索。

代码版本：本地HEAD 1167dd392837a02eb4f83481ae49f02607a9dae5；项目远端审计 74dbdc0e4f4815ef7ee2d665464239ec942c894d。源/数据哈希：`/data/adodas/SPromptGL-main/experiment/experiment17_disease_semantic_selective_alignment/manifest.json`。

正式实验目录：`/data/adodas/SPromptGL-main/experiment/experiment17_disease_semantic_selective_alignment`；报告：`/data/adodas/SPromptGL-main/experiment/experiment17_disease_semantic_selective_alignment/results/report.md`；运行器：`/data/adodas/SPromptGL-main/experiment/experiment17_disease_semantic_selective_alignment/run.py`；源码扩展：`/data/adodas/SPromptGL-main/experiment/experiment17_disease_semantic_selective_alignment/ds3a.py`。

### 实验十七验收记录

完成时间：2026-10-14T14:14:34.370589+00:00；正式训练墙钟 35.77 分钟。36/36连续十折任务、360/360 folds、360条诊断、240条D5反事实。GPU0/GPU1各18任务、180 folds；每卡最多4进程。

## 结果与评价口径

先计算每个seed的10折均值，再报告三个seed均值±population SD，单位为%。ASD=0；概率取log-softmax输出exp()的第0列。Hard-AUC由硬标签one-hot计算，二分类时等于BA，不能作为独立的排序性能证据。

|组别|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|---:|
|D0 原始P0|91.6910±0.0542|91.5394±0.0841|91.5394±0.0841|93.3352±0.4154|89.6606±0.6456|93.4181±0.5102|
|D1 容量控制|91.6527±0.2362|91.4887±0.2694|91.4887±0.2694|93.4717±0.0240|89.4898±0.6269|93.4875±0.0984|
|D2 Shared+Private强制对齐|91.4228±0.2362|91.2246±0.1981|91.2246±0.1981|93.1752±0.0860|88.7459±0.5068|93.7034±0.8122|
|D3 仅Shared统一对齐|91.4616±0.1428|91.2527±0.1576|91.2527±0.1576|93.5012±0.0603|88.5122±0.6961|93.9932±0.4690|
|D4 特征条件Align/Skip|91.5378±0.1434|91.3394±0.1210|91.3394±0.1210|93.3976±0.0371|88.6809±0.3053|93.9978±0.4603|
|D5 完整DS3A-v2|91.2700±0.0939|91.1406±0.0819|91.1406±0.0819|93.3024±0.0826|89.4228±0.2213|92.8585±0.3683|
|D6 中性文本|91.6144±0.0939|91.4394±0.0733|91.4394±0.0733|93.2374±0.0563|89.1707±0.9131|93.7080±0.8915|
|D7 无关疾病文本|91.7289±0.0939|91.5598±0.0787|91.5598±0.0787|93.3887±0.0721|89.4146±0.1208|93.7049±0.2633|
|D8 可学习类别向量|91.4612±0.0542|91.2803±0.1008|91.2803±0.1008|93.3289±0.1291|88.8496±0.7042|93.7111±0.5026|
|D9 关闭QC条件|91.4995±0.0939|91.3597±0.1170|91.3597±0.1170|93.4829±0.0410|89.5833±0.6049|93.1360±0.4654|
|D10 不融合新增Private|91.3462±0.1084|91.1533±0.1501|91.1533±0.1501|93.4674±0.0827|88.5955±1.0506|93.7111±0.8260|
|D11 四模态统一Shared对齐|91.4999±0.2485|91.3394±0.2594|91.3394±0.2594|93.2426±0.2344|89.1789±0.7051|93.4998±0.4380|

三项最佳：acc：D7 (91.7289%)；ba：D7 (91.5598%)；probability_auc：D3 (93.5012%)。

## 每个seed的十折均值

|组别|seed|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|---:|---:|
|D0|0|91.7294|91.5625|91.5625|93.8906|89.3415|93.7835|
|D0|1|91.6144|91.4268|91.4268|92.8916|89.0793|93.7743|
|D0|2|91.7294|91.6288|91.6288|93.2234|90.5610|92.6966|
|D1|0|91.6144|91.4508|91.4508|93.5045|89.3354|93.5661|
|D1|1|91.9592|91.8360|91.8360|93.4476|90.3232|93.3488|
|D1|2|91.3845|91.1793|91.1793|93.4631|88.8110|93.5476|
|D2|0|91.3845|91.2219|91.2219|93.1021|89.3171|93.1267|
|D2|1|91.1546|90.9834|90.9834|93.2960|88.8354|93.1314|
|D2|2|91.7294|91.4687|91.4687|93.1276|88.0854|94.8520|
|D3|0|91.6144|91.3895|91.3895|93.5802|88.5793|94.1998|
|D3|1|91.4995|91.3367|91.3367|93.4339|89.3293|93.3441|
|D3|2|91.2709|91.0319|91.0319|93.4895|87.6280|94.4357|
|D4|0|91.3845|91.2247|91.2247|93.4500|89.0915|93.3580|
|D4|1|91.4995|91.2867|91.2867|93.3685|88.3598|94.2137|
|D4|2|91.7294|91.5066|91.5066|93.3742|88.5915|94.4218|
|D5|0|91.2683|91.1495|91.1495|93.2171|89.5793|92.7197|
|D5|1|91.3858|91.2362|91.2362|93.4141|89.1098|93.3626|
|D5|2|91.1560|91.0362|91.0362|93.2759|89.5793|92.4931|
|D6|0|91.6144|91.3638|91.3638|93.2613|88.0976|94.6300|
|D6|1|91.4995|91.4158|91.4158|93.2911|90.3293|92.5023|
|D6|2|91.7294|91.5385|91.5385|93.1597|89.0854|93.9917|
|D7|0|91.6144|91.4694|91.4694|93.3327|89.5854|93.3534|
|D7|1|91.7281|91.5487|91.5487|93.3429|89.3232|93.7743|
|D7|2|91.8443|91.6612|91.6612|93.4904|89.3354|93.9870|
|D8|0|91.4995|91.3528|91.3528|93.4402|89.3476|93.3580|
|D8|1|91.3845|91.1377|91.1377|93.1478|87.8537|94.4218|
|D8|2|91.4995|91.3505|91.3505|93.3986|89.3476|93.3534|
|D9|0|91.4995|91.3155|91.3155|93.5408|88.8476|93.7835|
|D9|1|91.3845|91.2436|91.2436|93.4564|89.5732|92.9140|
|D9|2|91.6144|91.5199|91.5199|93.4513|90.3293|92.7105|
|D10|0|91.4995|91.3467|91.3467|93.5561|89.3354|93.3580|
|D10|1|91.2696|90.9809|90.9809|93.4890|87.1098|94.8520|
|D10|2|91.2696|91.1323|91.1323|93.3570|89.3415|92.9232|
|D11|0|91.7294|91.5385|91.5385|93.5255|89.0854|93.9917|
|D11|1|91.6157|91.5066|91.5066|93.2505|90.0854|92.9278|
|D11|2|91.1546|90.9729|90.9729|92.9516|88.3659|93.5800|

## 预设配对比较与D5全部对照

差值为百分点；W/T/L为同seed、同fold的胜/平/负。连续随机流因训练epoch不同而在后续fold自然分叉，配对数据划分一致，并非全部fold初始化相同。30次训练共享10套划分，不视为30个独立数据集，不做独立样本显著性宣称。

|比较|指标|均值差(pp)|seed0/1/2差(pp)|W/T/L|
|---|---|---:|---|---|
|D2−D1|acc|-0.2299|-0.2299/-0.8046/+0.3448|6/11/13|
|D2−D1|ba|-0.2640|-0.2289/-0.8526/+0.2894|8/5/17|
|D2−D1|hard_auc|-0.2640|-0.2289/-0.8526/+0.2894|8/5/17|
|D2−D1|probability_auc|-0.2965|-0.4024/-0.1516/-0.3354|11/1/18|
|D2−D1|sensitivity_asd|-0.7439|-0.0183/-1.4878/-0.7256|7/12/11|
|D2−D1|specificity_td|+0.2158|-0.4394/-0.2174/+1.3043|10/10/10|
|D3−D2|acc|+0.0387|+0.2299/+0.3448/-0.4585|8/17/5|
|D3−D2|ba|+0.0281|+0.1676/+0.3533/-0.4368|13/8/9|
|D3−D2|hard_auc|+0.0281|+0.1676/+0.3533/-0.4368|13/8/9|
|D3−D2|probability_auc|+0.3260|+0.4782/+0.1380/+0.3618|18/1/11|
|D3−D2|sensitivity_asd|-0.2337|-0.7378/+0.4939/-0.4573|9/11/10|
|D3−D2|specificity_td|+0.2899|+1.0731/+0.2128/-0.4163|10/14/6|
|D3−D1|acc|-0.1911|+0.0000/-0.4598/-0.1136|8/10/12|
|D3−D1|ba|-0.2360|-0.0612/-0.4993/-0.1474|9/6/15|
|D3−D1|hard_auc|-0.2360|-0.0612/-0.4993/-0.1474|9/6/15|
|D3−D1|probability_auc|+0.0295|+0.0757/-0.0137/+0.0264|15/1/14|
|D3−D1|sensitivity_asd|-0.9776|-0.7561/-0.9939/-1.1829|7/11/12|
|D3−D1|specificity_td|+0.5057|+0.6337/-0.0046/+0.8881|9/15/6|
|D11−D3|acc|+0.0383|+0.1149/+0.1162/-0.1162|8/15/7|
|D11−D3|ba|+0.0866|+0.1490/+0.1699/-0.0589|14/5/11|
|D11−D3|hard_auc|+0.0866|+0.1490/+0.1699/-0.0589|14/5/11|
|D11−D3|probability_auc|-0.2587|-0.0547/-0.1834/-0.5379|13/0/17|
|D11−D3|sensitivity_asd|+0.6667|+0.5061/+0.7561/+0.7378|12/12/6|
|D11−D3|specificity_td|-0.4934|-0.2081/-0.4163/-0.8557|6/12/12|
|D4−D3|acc|+0.0762|-0.2299/+0.0000/+0.4585|8/16/6|
|D4−D3|ba|+0.0867|-0.1648/-0.0500/+0.4748|12/9/9|
|D4−D3|hard_auc|+0.0867|-0.1648/-0.0500/+0.4748|12/9/9|
|D4−D3|probability_auc|-0.1036|-0.1303/-0.0654/-0.1152|12/0/18|
|D4−D3|sensitivity_asd|+0.1687|+0.5122/-0.9695/+0.9634|9/12/9|
|D4−D3|specificity_td|+0.0046|-0.8418/+0.8696/-0.0139|9/12/9|
|D5−D4|acc|-0.2678|-0.1162/-0.1136/-0.5734|6/11/13|
|D5−D4|ba|-0.1988|-0.0752/-0.0505/-0.4705|9/8/13|
|D5−D4|hard_auc|-0.1988|-0.0752/-0.0505/-0.4705|9/8/13|
|D5−D4|probability_auc|-0.0952|-0.2328/+0.0456/-0.0983|14/0/16|
|D5−D4|sensitivity_asd|+0.7419|+0.4878/+0.7500/+0.9878|8/18/4|
|D5−D4|specificity_td|-1.1394|-0.6383/-0.8511/-1.9288|3/13/14|
|D5−D9|acc|-0.2294|-0.2312/+0.0013/-0.4585|8/8/14|
|D5−D9|ba|-0.2190|-0.1661/-0.0074/-0.4837|11/4/15|
|D5−D9|hard_auc|-0.2190|-0.1661/-0.0074/-0.4837|11/4/15|
|D5−D9|probability_auc|-0.1805|-0.3237/-0.0423/-0.1754|11/0/19|
|D5−D9|sensitivity_asd|-0.1606|+0.7317/-0.4634/-0.7500|7/16/7|
|D5−D9|specificity_td|-0.2775|-1.0638/+0.4487/-0.2174|8/9/13|
|D5−D6|acc|-0.3444|-0.3461/-0.1136/-0.5734|5/11/14|
|D5−D6|ba|-0.2987|-0.2143/-0.1796/-0.5024|7/7/16|
|D5−D6|hard_auc|-0.2987|-0.2143/-0.1796/-0.5024|7/7/16|
|D5−D6|probability_auc|+0.0650|-0.0442/+0.1230/+0.1162|15/0/15|
|D5−D6|sensitivity_asd|+0.2520|+1.4817/-1.2195/+0.4939|6/17/7|
|D5−D6|specificity_td|-0.8495|-1.9103/+0.8603/-1.4986|6/11/13|
|D5−D7|acc|-0.4589|-0.3461/-0.3422/-0.6883|5/11/14|
|D5−D7|ba|-0.4192|-0.3199/-0.3125/-0.6250|9/5/16|
|D5−D7|hard_auc|-0.4192|-0.3199/-0.3125/-0.6250|9/5/16|
|D5−D7|probability_auc|-0.0863|-0.1156/+0.0713/-0.2145|16/0/14|
|D5−D7|sensitivity_asd|+0.0081|-0.0061/-0.2134/+0.2439|9/14/7|
|D5−D7|specificity_td|-0.8464|-0.6337/-0.4117/-1.4940|7/7/16|
|D5−D8|acc|-0.1911|-0.2312/+0.0013/-0.3435|7/12/11|
|D5−D8|ba|-0.1397|-0.2033/+0.0984/-0.3143|11/6/13|
|D5−D8|hard_auc|-0.1397|-0.2033/+0.0984/-0.3143|11/6/13|
|D5−D8|probability_auc|-0.0265|-0.2231/+0.2663/-0.1227|14/1/15|
|D5−D8|sensitivity_asd|+0.5732|+0.2317/+1.2561/+0.2317|10/15/5|
|D5−D8|specificity_td|-0.8526|-0.6383/-1.0592/-0.8603|6/10/14|
|D5−D10|acc|-0.0762|-0.2312/+0.1162/-0.1136|7/15/8|
|D5−D10|ba|-0.0127|-0.1972/+0.2553/-0.0962|11/5/14|
|D5−D10|hard_auc|-0.0127|-0.1972/+0.2553/-0.0962|11/5/14|
|D5−D10|probability_auc|-0.1650|-0.3389/-0.0749/-0.0811|13/1/16|
|D5−D10|sensitivity_asd|+0.8272|+0.2439/+2.0000/+0.2378|10/13/7|
|D5−D10|specificity_td|-0.8526|-0.6383/-1.4894/-0.4302|8/7/15|
|D5−D0|acc|-0.4210|-0.4611/-0.2286/-0.5734|4/11/15|
|D5−D0|ba|-0.3987|-0.4130/-0.1906/-0.5926|7/5/18|
|D5−D0|hard_auc|-0.3987|-0.4130/-0.1906/-0.5926|7/5/18|
|D5−D0|probability_auc|-0.0328|-0.6734/+0.5225/+0.0525|14/0/16|
|D5−D0|sensitivity_asd|-0.2378|+0.2378/+0.0305/-0.9817|8/13/9|
|D5−D0|specificity_td|-0.5597|-1.0638/-0.4117/-0.2035|7/11/12|
|D5−D1|acc|-0.3827|-0.3461/-0.5734/-0.2286|6/8/16|
|D5−D1|ba|-0.3481|-0.3013/-0.5998/-0.1431|10/3/17|
|D5−D1|hard_auc|-0.3481|-0.3013/-0.5998/-0.1431|10/3/17|
|D5−D1|probability_auc|-0.1693|-0.2873/-0.0335/-0.1872|11/0/19|
|D5−D1|sensitivity_asd|-0.0671|+0.2439/-1.2134/+0.7683|8/13/9|
|D5−D1|specificity_td|-0.6290|-0.8464/+0.0139/-1.0546|7/8/15|
|D5−D2|acc|-0.1528|-0.1162/+0.2312/-0.5734|7/14/9|
|D5−D2|ba|-0.0840|-0.0724/+0.2528/-0.4325|12/6/12|
|D5−D2|hard_auc|-0.0840|-0.0724/+0.2528/-0.4325|12/6/12|
|D5−D2|probability_auc|+0.1272|+0.1151/+0.1181/+0.1483|17/0/13|
|D5−D2|sensitivity_asd|+0.6768|+0.2622/+0.2744/+1.4939|11/14/5|
|D5−D2|specificity_td|-0.8449|-0.4070/+0.2313/-2.3589|7/10/13|
|D5−D3|acc|-0.1916|-0.3461/-0.1136/-0.1149|7/11/12|
|D5−D3|ba|-0.1121|-0.2401/-0.1005/+0.0043|11/3/16|
|D5−D3|hard_auc|-0.1121|-0.2401/-0.1005/+0.0043|11/3/16|
|D5−D3|probability_auc|-0.1988|-0.3631/-0.0198/-0.2136|13/0/17|
|D5−D3|sensitivity_asd|+0.9106|+1.0000/-0.2195/+1.9512|14/8/8|
|D5−D3|specificity_td|-1.1348|-1.4801/+0.0185/-1.9426|7/7/16|
|D5−D11|acc|-0.2299|-0.4611/-0.2299/+0.0013|7/12/11|
|D5−D11|ba|-0.1987|-0.3890/-0.2704/+0.0632|10/7/13|
|D5−D11|hard_auc|-0.1987|-0.3890/-0.2704/+0.0632|10/7/13|
|D5−D11|probability_auc|+0.0598|-0.3084/+0.1636/+0.3243|16/0/14|
|D5−D11|sensitivity_asd|+0.2439|+0.4939/-0.9756/+1.2134|5/20/5|
|D5−D11|specificity_td|-0.6414|-1.2720/+0.4348/-1.0870|8/10/12|

## Shared/Private、Align/Skip与残差

|组别|训练Shared原型距离|测试Shared原型距离|Private逐样本距离|Shared/Private绝对cos CORR/PHENO|Align均值/受试者SD|Skip<0.5|PHENO/CORR残差%|
|---|---:|---:|---:|---|---|---:|---|
|D1|1.95969|1.97125|1.95037|0.1284/0.1361|1.0000/0.0000|0.00%|3.9423/4.4701|
|D2|0.00021|0.00624|0.24622|0.1395/0.1361|1.0000/0.0000|0.00%|0.8454/0.5685|
|D3|0.00025|0.00621|1.97476|0.1106/0.1272|1.0000/0.0000|0.00%|3.3732/3.9557|
|D4|0.00017|0.00614|1.98729|0.1519/0.1223|0.0321/0.0001|96.67%|4.1069/4.6144|
|D5|0.00057|0.02260|2.09545|0.1692/0.1288|0.1696/0.2439|86.68%|4.5146/4.9427|
|D6|0.00065|0.02185|1.95826|0.1394/0.1278|0.1906/0.2519|84.15%|4.6470/5.3203|
|D7|0.00049|0.02080|1.97944|0.1418/0.1297|0.2124/0.2650|81.88%|4.7547/5.4206|
|D8|0.00065|0.02668|2.00182|0.1418/0.1280|0.1804/0.2418|85.08%|4.3178/4.8603|
|D9|0.00016|0.00616|2.03834|0.1337/0.1262|0.0029/0.0001|100.00%|3.4285/3.9724|
|D10|0.00041|0.00996|2.19880|0.1280/0.1405|0.5166/0.3602|46.39%|3.6437/4.1502|
|D11|0.00072|0.00582|2.03794|0.1502/0.1720|1.0000/0.0000|0.00%|3.6284/4.1649|

D5 Align 5%/50%/95%分位数的折均值：0.0015/0.0657/0.7143；<0.1比例65.67%，>0.9比例4.10%。ASD/TD事后均值0.1825/0.1584；QC proxy均值0.5004，它没有真实质量标签校准。

D5训练CORR/PHENO类别原型cosine折均值0.9997/0.9986。原型对齐距离下降可以由类别坍塌导致；不能单独证明共享诊断信息增加。Shared的ASD线性probe BA（CORR/PHENO）60.14%/49.36%，Private为78.87%/49.39%。这些固定Ridge(alpha=1)训练折拟合、测试折评估的probe仅用于描述表示，不参与选模。

D5站点probe ACC（Shared CORR/PHENO）为24.49%/22.39%。新增分支未直接读取站点，但年龄、性别、FC与站点仍可能相关；原主干仍含站点，不能宣称消除站点混杂。

D5查询向量中心化方向方差（CORR/PHENO）0.00000223/0.00016953；平均两受试者查询cosine0.9977/0.8262。冻结真实文本原型cosine为0.946976；中性/无关原型为0.968670/0.742523。文本几何与医学内容同时变化，阴性文本对照不能完全隔离语言几何因素。

全部辅助模型最大单受试者、单目标Token相对L2残差为8.9442%，低于预注册10%上限。D5目标Token近零残差(<1e-5)比例为0.00%；PHENO/CORR位置分别为4.5146%/4.9427%，ANAT/FUNC残差恒为0。该位置已经经过CAM交互，不能把位置贡献当作纯模态因果贡献，更不能解释为ROI。

## D5同一最佳checkpoint反事实

不重训、不重新选epoch、不重新初始化；所有条件使用同一受试者和checkpoint，原主干参数及注入前CAM Token逐值不变，训练后均恢复REAL。SKIP同时令共享融合输入为0，Private仍可响应；它不能回撤训练期对齐损失。Shared/Private编码本身在全部干预中逐值不变，改变的是gate及融合输入。

|条件|ΔACC(pp)|ΔBA(pp)|ΔProb-AUC(pp)|翻转率%|ASD概率绝对变化|Align绝对变化|delta绝对变化|logit绝对变化|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|REAL|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|
|NEUTRAL|0.00000000|0.00000000|0.00000000|0.00000000|0.00002099|0.00133814|0.00001566|0.00009089|
|UNRELATED|0.00000000|0.00000000|0.00000000|0.00000000|0.00002186|0.00142572|0.00001617|0.00009395|
|SWAPPED|0.00000000|0.00000000|0.00000000|0.00000000|0.00000667|0.00065184|0.00000707|0.00002926|
|SKIP|-0.11450714|-0.10536672|0.02498627|0.34352142|0.00335586|0.16957467|0.00172805|0.01438092|
|NO_QC|-0.19113549|-0.18224901|0.01293593|0.26776385|0.00331886|0.16849146|0.00191389|0.01470170|
|NO_PRIVATE|-0.87948450|-0.88126866|0.01757256|0.87948450|0.00967728|0.00000000|0.00704652|0.04450252|
|RESIDUAL_OFF|-0.99355625|-0.97625655|0.04623655|1.22257054|0.01175727|0.00000000|0.00563711|0.05265755|

文本干预在30/30个checkpoint产生>1e-7的ASD概率变化。平均文本干预概率变化0.000016508。计算图敏感性与正确临床语义的独立诊断价值是两项不同证据；后者还须D5同时优于D6/D7/D8。

## 五项科学判断

**A 对齐价值**：D3相对D1在ACC/BA/Prob-AUC均为正且三个seed均为正：False；Shared-only相对Shared+Private满足同标准：False。详见配对差值，不以原型距离作为分类收益替代。

**B 选择性价值**：D4相对D3满足三指标、三seed同向标准：False。D5权重均值0.1696，受试者SD0.2439；Skip<0.5比例86.68%。低均值或低方差可能是总体收缩，而非有意义的患者特异选择；该阈值只作描述。

**C 疾病语义价值**：D5同时相对中性、无关文本、可学习类别原型达到三指标及三seed同向标准：False。若False，本轮不能证明ASD医学语言具有超过任意向量或新增容量的独立贡献。

**D 数据角色价值**：D5相对D9质量条件的稳定收益：False；相对D10 Private融合的稳定收益：False。D11−D3用于检验质量指标强制对齐，但不能单由D5−D11分离QC、语义、选择性等多个因素。年龄/性别不是临床症状；Shared与Private probe和原型坍塌诊断仅提供有限的共享信息证据。

**E 分类价值**：D5相对原P0、容量控制的三指标三seed同向改善分别为False/False。SEN/SPE也完整列出；不能只挑ACC。

## 与历史E9及既有方法的关系

历史实验九E9：ACC 91.8826%、BA/Hard-AUC 91.6915%、Prob-AUC 93.8227%、SEN 89.2480%、SPE 94.1351%。它更换P3且连续随机轨迹不同，仅作描述性比较。

D5−E9描述性差值(pp)：acc -0.6126；ba -0.5509；probability_auc -0.5203；sensitivity_asd +0.1748；specificity_td -1.2766。

本轮三项工程改进：辅助编码器读取真实原始模态并排除直接站点输入；ANAT/FUNC主要作QC条件而非疾病共享目标；零初始化且相对L2≤10%的残差保留原判别路径。Common/Private、类原型对齐及语义融合均有既有工作，不可称作通用组件首创。文献审计和具体方法差异见README。

## 复现、限制与下一阶段

D0全部30fold与实验13F0、14E00、15A0、16B0六指标、best/executed epochs逐值一致；初始化采用各历史版本对应序列化格式核对，14–16入口RNG逐值一致。原始参数1462957；新增参数98637，D8再多2048。所有注册参数均进入优化器，关闭组件允许无梯度；实际参与情况见diagnostics的sampled_gradient_max。完整model/双optimizer/全部RNG恢复已通过两步后第三步逐值复演。

正式运行资源峰值：{"rss_gib": 23.387893676757812, "gpu0_mib": 2931, "gpu1_mib": 2947, "cpu_cores_used": 8.79086434257409}。失败/恢复记录：{}。不同组入口初始化与D0相同fold数：{"D0": 30, "D1": 4, "D2": 3, "D3": 3, "D4": 3, "D5": 4, "D6": 5, "D7": 3, "D8": 3, "D9": 3, "D10": 3, "D11": 3}。

限制：val=test，并以测试折ACC选checkpoint；全体871人transductive图；沿用预先标准化与RFE，而非本轮训练折内重算。结果是历史协议内比较，不能声称独立无泄漏泛化。固定12组没有无类别对齐训练对照，无法证明类别条件对齐的增量价值；D2另对Private加同系数对齐，合计对齐预算更大，D11激活704个QC共享编码参数，D8多2048，均不隐瞒有效容量差异。停止梯度避免gate直接利用小权重规避损失，但不证明gate等于真实互信息或可靠性。

下一阶段应先检验真实临床症状与FC是否存在共享诊断信号，并在训练内选模、独立站点验证的协议中复核；如要隔离类条件对齐或QC交互，另预注册单因素对照。当前不自动修改Prompt、不继续超参数搜索，也不因小幅均值提升宣称达到MICCAI 2027标准。

根源码、实验1–16全部哈希验收通过；handoff仅追加；无自动commit/push。完整参数/数据/文本/源码哈希和配置位于manifest.json，temporary sources由integration_patch.diff重建，正式完成后清理临时checkpoint。



### 实验十七综合解释补充

本轮不支持将完整DS³A-v2作为已经验证有效的MICCAI 2027第一创新模块。D5相对P0的ACC、BA在三个seed均下降，均值差为−0.4210/−0.3987个百分点；Prob-AUC均值差−0.0328个百分点。D7不相关疾病文本取得最佳ACC/BA，但相对P0仅+0.0379/+0.0204个百分点，且不是三seed一致改善；不能解释成无关疾病具有ASD医学意义。D3最高概率AUC相对P0提高0.1660个百分点，ACC/BA同时下降，表明排序与硬决策指标不一致。

对齐确实降低了训练类别原型跨模态距离，但CORR与PHENO的ASD/TD原型各自非常接近，共享方向趋同是重要解释。FC Shared probe BA约60.14%，Private约78.87%，PHENO两条路径均约49.4%。这些结果没有证实年龄/性别提供独立的疾病共享证据。Shared/Private绝对cos较低说明方向有差别，不能等同于诊断信息已经得到完整分解。

D4没有文本/QC时的Align均值约0.0321、受试者SD约0.0001，更接近统一抑制。D5的Align有较大患者间变化，但NO_QC干预产生的权重变化约0.1685，远大于中性/不相关文本约0.0013–0.0014的变化；这种动态主要来自QC条件。D9关闭QC重训反而改善三个主要均值，因此有动态变化不等于更有效的选择。

中性、不相关和ASD/TD顺序交换三个文本反事实的预测翻转率、ACC/BA/Prob-AUC变化全部为0，平均概率变化仅约1.65e-5。语言通路没有严格数值归零，但缺少实质的分类贡献，FC查询方向cos约0.9977也提示仍有方向趋同。新增残差整体非零且满足10%上限，解决了无界增强问题，未解决疾病语言的独立价值问题。

D5最佳checkpoint关闭Private使ACC下降约0.8795个百分点，说明已训练模型依赖该通路；从头训练D10却略优于D5，说明这个反事实不能证明加入Private提高了算法总体性能。同样，NO_QC对已训练D5有影响，不等于QC条件优于不使用它的重训模型。保留P0较稳妥；下一阶段应先获得并验证有独立信息的临床表型或更直接的疾病证据，再用训练内选模和独立站点评估检验机制，而不是继续换背景性疾病句子。

对齐损失和模型推理的类别原型仅来自训练索引。diagnostics中test类别原型距离是训练结束后的描述性统计，没有用于模型参数、对齐损失、gate、checkpoint选择或反事实推理；它不能当作推理时可用的类别参考。Ridge probe字段中的probe_prob_auc使用连续decision score，不是校准概率；正式Probability-AUC始终使用原输出exp()。alpha的初始范数为0，其带eps的范数比仅是数值记账，不解释为收缩比。


工程说明：36个训练任务均一次成功。最终汇总曾因25行Torch/numpy均值约1.11e-16的差异触发严格相等断言，随后改为已有逐fold校验相同的1e-12容差；没有重训或改动指标。manifest分别保留execution_code_sha256与当前code_sha256，数学模块及训练协议未改变。从manifest准备至完成44.77分钟，其中正式训练35.77分钟；此前文献/源码审计未计入这段时间。


<!-- experiment18_discriminability_preserving_selective_alignment -->

## 实验十八：DS³A-v3正式完成

本地源码版本HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；实验目录 `/data/adodas/SPromptGL-main/experiment/experiment18_discriminability_preserving_selective_alignment`；完整源/数据/证据签名 `/data/adodas/SPromptGL-main/experiment/experiment18_discriminability_preserving_selective_alignment/manifest.json`。30任务、300真实fold、300诊断、150反事实已验收；双GPU各150fold。

研究动机、数学控制、OOF可靠度、六指标和种子稳定性、判别保护、同强度/错配权重/错配患者比较、Private蒸馏、反事实、负结果及MICCAI2027判断如下，旧内容完整保留。

# 实验十八：DS³A-v3 判别性保持与可靠度驱动的选择性多模态对齐

训练完成：2026-10-14T17:50:16.794813+00:00；30/30任务、300/300真实fold、300条诊断、150条V5反事实。训练墙钟28.53分钟。

## 1. 审计、研究动机与历史失败

GFR四图传播拼接320维；i=0先赋值，i=1覆盖前次combined_feat，i=2/3累加。保留这一实际顺序。输出已混合。PLN InputLayer→Encoder→FeedForward生成871×4×36潜在特征；注入在最后FeedForward后、原SPGS前。P0原8句CLIP RN50→2×4096→4096/128/9 ReLU MLP→cosine/softmax/max标量；随后原global平均池化、局部channel-softmax和RES加法、log-softmax分类器完整保留。

simple-2为加权NLL+四原输入图正则均值；ALoss计算但不反传。两个原Adam，PLN原my_weight_init执行后fork_rng创建辅助模块。450epoch，wait>200停止，严格ACC提升保存首次最佳；val=test，871人transductive，全队列既有标准化/RFE保留。不同组执行epoch不同后连续折随机流自然分叉，不能假称后续折初始化相同。

已实际阅读本地SPromptGL PDF Fig.1、Section2.1–2.2、Eq1–7和消融。Fig.2 ABIDE从CAM86.48到GFR91.73、RES92.08、SPGS92.30；主要增量来自GFR。论文Eq1的类别图项与源码未反传ALoss、论文图加权求和与源码覆盖行为、论文实现超参与本地shell有差异；本实验复现历史源码协议，不修补这些差异。

实验13患者FC文本未稳定超过自身状态/身份错配；14层级投影趋同且无稳定净收益；15文本内部响应未传递至最终预测、强衰减伴随消息接近零；16修复了文本敏感性但B5较P0 ACC下降0.841pp、残差约32.33%；17D5较P0 ACC/BA下降0.4210/0.3987pp，FC/PHENO类别原型cos0.9997/0.9986，Shared probes约60.14/49.36%BA，文本反事实零翻转。不能把距离降低、低Shared/Private cosine或非零梯度当作临床信息有效。E9真实30折六指标作为历史描述性对照。

PHENO48=26年龄分箱+20站点+2性别；辅助分支仅28年龄/性别，主干仍含20站点。ANAT6/FUNC10是QC；func_gsr指Ghost-to-Signal Ratio。CORR256为既有FC/RFE；不重算。当前无ADOS/ADI-R/SRS真实症状；CAM位置和局部Token不能解释为ROI。

实验十七的类别原型级对齐既不保证患者一一对应，也不保护Shared的判别信息；其learned Gate易整体缩小，不能从低对齐距离或患者变化推断选择有效。DS³A-v3改为训练OOF固定代理、患者级距离、FC Shared类别监督和严格同权重预算控制。零初始化≤10%残差保留，但取消新疾病语义、QC神经Gate、Shared/Private余弦正则及r条件融合。Shared/Private独立编码器不自动构成信息解耦。

## 2. 固定结构与精确控制变量

四个32维编码器读取FC256/年龄性别28；四表示拼接128→64→72，两36维方向以0.1*tanh(alpha)*detach(CAM范数)归一化残差注入PHENO/CORR位置，随后完整执行原SPGS/RES/分类器。r不送入融合，无推理期可靠度Gate。

Ltotal=Loriginal+0.1*I_disc*Ldisc+0.1*I_pair*mean_train(w_i*d_i)+0.05*I_private*Lprivate。d_i=||normalize(sFC_i)−normalize(sPH_pair(i))||²；分母仅训练人数。Private软BCE目标是FC教师OOF ASD概率，准确名称是FC教师诊断信息蒸馏。

|组|新增注册参数|判别监督|权重|蒸馏|配对|
|---|---:|---|---|---|---|
|V0 原始P0|0|False|none|False|本人|
|V1 容量控制|31597|False|none|False|本人|
|V2 FC Shared判别性保护|31597|True|none|False|本人|
|V3 判别保护+统一患者对齐|31597|True|ones|False|本人|
|V4 同强度常量对齐|31597|True|mean|False|本人|
|V5 完整选择性对齐|31597|True|real|False|本人|
|V6 可靠度错配控制|31597|True|permuted|False|本人|
|V7 选择性对齐+FC教师Private蒸馏|31597|True|real|True|本人|
|V8 无判别保护的统一对齐|31597|False|ones|False|本人|
|V9 同类别跨患者配对错配|31597|True|real|False|train同类无自身置乱|

V4/V5只改样本权重分配，V5/V6保留权重数值集合，V5/V9只改训练对齐PHENO对象，V3/V8只改判别监督，V5/V7只改蒸馏。新参数init在原my_weight_init后fork_rng，原两个Adam及正则保持；原参数1462957。关闭头保留注册但无有效梯度，后文列实际结构容量。

## 3. 训练折内可对齐性与站点诊断

各项均为10套高度重叠outer-train内部五折OOF，不是outer-test正式指标，不作独立数据集统计推断。inner随机种子18000+outer_fold，Scaler+LR(C1/lbfgs/1000)逐inner train拟合。测试标签未进入教师/可靠度拟合。

|输入|OOF ACC%|OOF BA%|OOF Prob-AUC%|预测ASD比例%|
|---|---:|---:|---:|---:|
|FC|84.3603|84.2284|90.4486|45.6818|
|PHENO|51.1674|49.7787|50.7015|31.4194|
|FC_PHENO|83.1866|83.0499|89.7023|45.7073|
|SITE|53.8460|52.2469|50.8693|28.3842|
|FC_SITE|83.7097|83.5677|90.4128|45.5926|
|FC_PHENO_SITE|82.8422|82.7108|89.5253|45.7968|

增量(pp)：FC+PHENO−FC oof_acc -1.1737 / oof_ba -1.1784 / oof_probability_auc -0.7463；控制SITE后 FC+PHENO+SITE−FC+SITE oof_acc -0.8674 / oof_ba -0.8569 / oof_probability_auc -0.8875。

FC+PHENO连续AUC增量为正的outer-train划分为0/10；加入SITE后增量为正0/10。PHENO单模态BA=49.7787%，AUC=50.7015%；FC/PHENO类别一致率=50.9887%，ASD概率相关=0.012683。

年龄/性别可包含采样与人口学关联，却不是症状量表；若其OOF增量弱，联合正确性选择可能主要挑选FC容易病例或性别/年龄常见群体，无法凭重加权创造原输入不存在的疾病信息。SITE增量说明应关注混杂，不能断定因果站点影响；辅助路径不直接读取SITE也不能宣称消除了站点相关性。全部十组按预注册继续训练，未由这些诊断选择超参。

## 4. OOF可靠度及冻结证据

q=两单模态OOF均正确；六证据的Scaler+LR仅outer train拟合r。train概率为严格OOF，test概率来自全outer train拟合教师；test q缺失。r训练拟合本身未再进行cross-fitting，r−q关系是拟合数据上的诊断，不能称独立校准质量或医学可靠度。

|fold|q正例%|r均值|r SD|r p05/50/95|r>0.5%|Kish有效占比%|r−q相关|r预测q训练AUC|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|1|44.1890|0.441890|0.434273|0.001412/0.501488/0.921395|50.0639|50.8693|0.883205|0.967044|
|2|44.0051|0.440051|0.437089|0.001090/0.467589/0.940747|49.4898|50.3377|0.888861|0.971298|
|3|43.6224|0.436224|0.428806|0.001072/0.447726/0.913939|49.2347|50.8574|0.872393|0.966083|
|4|42.6020|0.426021|0.429123|0.001031/0.086398/0.913688|48.7245|49.6372|0.877263|0.965549|
|5|43.3673|0.433673|0.424603|0.001403/0.482100/0.912583|49.7449|51.0568|0.863928|0.966753|
|6|40.9439|0.409439|0.416754|0.001057/0.009726/0.898072|47.1939|49.1147|0.856015|0.958533|
|7|44.2602|0.442603|0.428414|0.001098/0.435016/0.922402|49.1071|51.6286|0.872361|0.969619|
|8|42.0918|0.420918|0.429991|0.001224/0.007734/0.933995|47.9592|48.9339|0.880404|0.965292|
|9|43.8776|0.438776|0.422680|0.001359/0.540684/0.913939|50.7653|51.8677|0.861347|0.951909|
|10|43.6224|0.436224|0.434585|0.001178/0.451957/0.921174|49.1071|50.1882|0.884779|0.971700|

冻结证据SHA256：`117361e911818624f9aa8910acee3d6d48392a545153cf1b4d03922ac5f08a0c`；q单类别常数退化fold=[]。V4/V5总量64位误差最大1.14e-13（预设容差1e−12），V5/V6排序后的权重逐值相同，V5/V9权重逐值相同且train同类置乱无固定点。r存在患者差异不证明选择有益，必须结合V4/V6/V9对照。

## 5. 全部正式六指标

先各seed独立取十折均值，再计算3seed mean±population SD，单位%。ASD=0，连续AUC用exp(log_softmax)第0类；Hard-AUC按原硬one-hot口径，在此二分类等于BA。

|组|ACC|BA|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|---:|
|V0 原始P0|91.6910±0.0542|91.5394±0.0841|91.5394±0.0841|93.3352±0.4154|89.6606±0.6456|93.4181±0.5102|
|V1 容量控制|91.6140±0.0928|91.4640±0.1226|91.4640±0.1226|93.7157±0.1372|89.5732±0.7306|93.3549±0.5231|
|V2 FC Shared判别性保护|91.2700±0.0006|91.0957±0.0415|91.0957±0.0415|93.3430±0.2953|88.9167±0.7022|93.2747±0.6194|
|V3 判别保护+统一患者对齐|91.6910±0.3017|91.5411±0.2610|91.5411±0.2610|93.5690±0.1076|89.5854±0.4008|93.4968±0.9030|
|V4 同强度常量对齐|91.6532±0.2361|91.5285±0.2046|91.5285±0.2046|93.3276±0.1126|90.0691±0.3478|92.9880±0.7079|
|V5 完整选择性对齐|91.6149±0.0006|91.4763±0.0170|91.4763±0.0170|93.4271±0.1446|89.6687±0.2358|93.2840±0.2028|
|V6 可靠度错配控制|91.5765±0.2863|91.3843±0.2719|91.3843±0.2719|93.5272±0.3072|88.9187±0.7643|93.8498±0.8254|
|V7 选择性对齐+FC教师Private蒸馏|91.4995±0.3251|91.3511±0.3230|91.3511±0.3230|93.5394±0.3412|89.4228±0.5878|93.2794±0.5592|
|V8 无判别保护的统一对齐|91.4612±0.1434|91.3029±0.0965|91.3029±0.0965|93.2145±0.0754|89.2541±0.8178|93.3518±0.9264|
|V9 同类别跨患者配对错配|91.4233±0.0548|91.2133±0.0734|91.2133±0.0734|93.4268±0.1283|88.5813±0.4033|93.8452±0.2691|

三项最佳：acc=V0；ba=V3；probability_auc=V1。仅报告排序，不据此选新设计。

## 6. 每seed十折均值

|组|seed|ACC|BA|Hard-AUC|Prob-AUC|SEN|SPE|
|---|---:|---:|---:|---:|---:|---:|---:|
|V0|0|91.72936|91.56250|91.56250|93.89057|89.34146|93.78353|
|V0|1|91.61442|91.42678|91.42678|92.89162|89.07927|93.77428|
|V0|2|91.72936|91.62878|91.62878|93.22340|90.56098|92.69658|
|V1|0|91.72806|91.57450|91.57450|93.72013|89.80488|93.34413|
|V1|1|91.50078|91.29315|91.29315|93.54546|88.58537|94.00093|
|V1|2|91.61311|91.52449|91.52449|93.88144|90.32927|92.71970|
|V2|0|91.27090|91.15044|91.15044|92.92568|89.81707|92.48381|
|V2|1|91.26959|91.04998|91.04998|93.53734|88.10366|93.99630|
|V2|2|91.26959|91.08670|91.08670|93.56586|88.82927|93.34413|
|V3|0|91.84431|91.68750|91.68750|93.72122|89.59146|93.78353|
|V3|1|91.26959|91.17442|91.17442|93.49314|90.07317|92.27567|
|V3|2|91.95925|91.76127|91.76127|93.49269|89.09146|94.43108|
|V4|0|91.38454|91.30963|91.30963|93.19184|90.56098|92.05828|
|V4|1|91.95925|91.80178|91.80178|93.32342|89.82927|93.77428|
|V4|2|91.61573|91.47422|91.47422|93.46748|89.81707|93.13136|
|V5|0|91.61573|91.48263|91.48263|93.32443|89.82927|93.13599|
|V5|1|91.61442|91.45307|91.45307|93.32533|89.33537|93.57077|
|V5|2|91.61442|91.49335|91.49335|93.63157|89.84146|93.14524|
|V6|0|91.27090|91.14486|91.14486|93.25294|89.57927|92.71045|
|V6|1|91.95925|91.76454|91.76454|93.37246|89.32927|94.19981|
|V6|2|91.49948|91.24339|91.24339|93.95616|87.84756|94.63922|
|V7|0|91.26959|91.16958|91.16958|93.48755|89.84146|92.49769|
|V7|1|91.95925|91.80482|91.80482|93.98080|89.83537|93.77428|
|V7|2|91.26959|91.07880|91.07880|93.14997|88.59146|93.56614|
|V8|0|91.61442|91.36839|91.36839|93.17932|88.09756|94.63922|
|V8|1|91.49948|91.37393|91.37393|93.31930|89.82927|92.91859|
|V8|2|91.26959|91.16653|91.16653|93.14482|89.83537|92.49769|
|V9|0|91.50078|91.31503|91.31503|93.57369|89.07317|93.55689|
|V9|1|91.38454|91.17982|91.17982|93.26116|88.58537|93.77428|
|V9|2|91.38454|91.14490|91.14490|93.44570|88.08537|94.20444|

## 7. 预设配对比较

同seed/fold差，单位pp；W/T/L为逐折胜/平/负，1e−12仅用于浮点平局识别。三seed共用10套划分且后续折因epoch不同RNG自然分叉，不把30折视为30独立数据集，不宣称统计显著。

|比较|指标|均值差pp|seed0/1/2差pp|W/T/L|
|---|---|---:|---|---|
|V2−V1|acc|-0.34396|-0.45716/-0.23119/-0.34352|6/12/12|
|V2−V1|ba|-0.36834|-0.42406/-0.24317/-0.43779|9/6/15|
|V2−V1|hard_auc|-0.36834|-0.42406/-0.24317/-0.43779|9/6/15|
|V2−V1|probability_auc|-0.37272|-0.79445/-0.00812/-0.31558|9/1/20|
|V2−V1|sensitivity_asd|-0.65650|+0.01220/-0.48171/-1.50000|5/15/10|
|V2−V1|specificity_td|-0.08017|-0.86031/-0.00463/+0.62442|8/11/11|
|V3−V2|acc|+0.42102|+0.57341/+0.00000/+0.68966|13/13/4|
|V3−V2|ba|+0.44536|+0.53706/+0.12444/+0.67458|15/10/5|
|V3−V2|hard_auc|+0.44536|+0.53706/+0.12444/+0.67458|15/10/5|
|V3−V2|probability_auc|+0.22606|+0.79554/-0.04420/-0.07317|18/1/11|
|V3−V2|sensitivity_asd|+0.66870|-0.22561/+1.96951/+0.26220|11/13/6|
|V3−V2|specificity_td|+0.22202|+1.29972/-1.72063/+1.08696|8/15/7|
|V3−V8|acc|+0.22989|+0.22989/-0.22989/+0.68966|11/12/7|
|V3−V8|ba|+0.23811|+0.31911/-0.19951/+0.59475|14/7/9|
|V3−V8|hard_auc|+0.23811|+0.31911/-0.19951/+0.59475|14/7/9|
|V3−V8|probability_auc|+0.35453|+0.54190/+0.17384/+0.34787|19/0/11|
|V3−V8|sensitivity_asd|+0.33130|+1.49390/+0.24390/-0.74390|11/12/7|
|V3−V8|specificity_td|+0.14493|-0.85569/-0.64292/+1.93340|8/13/9|
|V5−V4|acc|-0.03831|+0.23119/-0.34483/-0.00131|6/17/7|
|V5−V4|ba|-0.05219|+0.17300/-0.34871/+0.01913|7/14/9|
|V5−V4|hard_auc|-0.05219|+0.17300/-0.34871/+0.01913|7/14/9|
|V5−V4|probability_auc|+0.09953|+0.13259/+0.00190/+0.16410|13/2/15|
|V5−V4|sensitivity_asd|-0.40041|-0.73171/-0.49390/+0.02439|2/20/8|
|V5−V4|specificity_td|+0.29602|+1.07771/-0.20352/+0.01388|8/17/5|
|V5−V6|acc|+0.03831|+0.34483/-0.34483/+0.11494|8/16/6|
|V5−V6|ba|+0.09208|+0.33777/-0.31147/+0.24996|12/10/8|
|V5−V6|hard_auc|+0.09208|+0.33777/-0.31147/+0.24996|12/10/8|
|V5−V6|probability_auc|-0.10008|+0.07148/-0.04713/-0.32458|14/1/15|
|V5−V6|sensitivity_asd|+0.75000|+0.25000/+0.00610/+1.99390|9/17/4|
|V5−V6|specificity_td|-0.56583|+0.42553/-0.62905/-1.49399|6/15/9|
|V5−V9|acc|+0.19157|+0.11494/+0.22989/+0.22989|9/17/4|
|V5−V9|ba|+0.26309|+0.16760/+0.27324/+0.34845|15/10/5|
|V5−V9|hard_auc|+0.26309|+0.16760/+0.27324/+0.34845|15/10/5|
|V5−V9|probability_auc|+0.00026|-0.24926/+0.06416/+0.18588|15/2/13|
|V5−V9|sensitivity_asd|+1.08740|+0.75610/+0.75000/+1.75610|11/16/3|
|V5−V9|specificity_td|-0.56121|-0.42091/-0.20352/-1.05920|5/15/10|
|V7−V5|acc|-0.11538|-0.34613/+0.34483/-0.34483|6/12/12|
|V7−V5|ba|-0.12528|-0.31305/+0.35176/-0.41455|7/9/14|
|V7−V5|hard_auc|-0.12528|-0.31305/+0.35176/-0.41455|7/9/14|
|V7−V5|probability_auc|+0.11233|+0.16312/+0.65547/-0.48161|17/0/13|
|V7−V5|sensitivity_asd|-0.24593|+0.01220/+0.50000/-1.25000|7/13/10|
|V7−V5|specificity_td|-0.00463|-0.63830/+0.20352/+0.42091|8/17/5|
|V5−V0|acc|-0.07619|-0.11364/+0.00000/-0.11494|6/16/8|
|V5−V0|ba|-0.06300|-0.07987/+0.02629/-0.13543|10/8/12|
|V5−V0|hard_auc|-0.06300|-0.07987/+0.02629/-0.13543|10/8/12|
|V5−V0|probability_auc|+0.09191|-0.56614/+0.43371/+0.40817|17/1/12|
|V5−V0|sensitivity_asd|+0.00813|+0.48780/+0.25610/-0.71951|7/17/6|
|V5−V0|specificity_td|-0.13414|-0.64755/-0.20352/+0.44866|7/13/10|
|V7−V0|acc|-0.19157|-0.45977/+0.34483/-0.45977|4/15/11|
|V7−V0|ba|-0.18828|-0.39292/+0.37805/-0.54997|5/11/14|
|V7−V0|hard_auc|-0.18828|-0.39292/+0.37805/-0.54997|5/11/14|
|V7−V0|probability_auc|+0.20424|-0.40302/+1.08918/-0.07344|15/1/14|
|V7−V0|sensitivity_asd|-0.23780|+0.50000/+0.75610/-1.96951|5/15/10|
|V7−V0|specificity_td|-0.13876|-1.28585/+0.00000/+0.86957|9/15/6|
|V5−V3|acc|-0.07619|-0.22858/+0.34483/-0.34483|7/15/8|
|V5−V3|ba|-0.06472|-0.20487/+0.27865/-0.26792|8/10/12|
|V5−V3|hard_auc|-0.06472|-0.20487/+0.27865/-0.26792|8/10/12|
|V5−V3|probability_auc|-0.14191|-0.39679/-0.16782/+0.13888|15/0/15|
|V5−V3|sensitivity_asd|+0.08333|+0.23780/-0.73780/+0.75000|6/17/7|
|V5−V3|specificity_td|-0.21277|-0.64755/+1.29510/-1.28585|7/14/9|
|V1−V0|acc|-0.07706|-0.00131/-0.11364/-0.11625|7/16/7|
|V1−V0|ba|-0.07531|+0.01200/-0.13363/-0.10429|10/7/13|
|V1−V0|hard_auc|-0.07531|+0.01200/-0.13363/-0.10429|10/7/13|
|V1−V0|probability_auc|+0.38048|-0.17044/+0.65384/+0.65804|17/0/13|
|V1−V0|sensitivity_asd|-0.08740|+0.46341/-0.49390/-0.23171|10/10/10|
|V1−V0|specificity_td|-0.06321|-0.43941/+0.22664/+0.02313|8/12/10|
|V2−V0|acc|-0.42102|-0.45846/-0.34483/-0.45977|2/17/11|
|V2−V0|ba|-0.44364|-0.41206/-0.37680/-0.54208|4/13/13|
|V2−V0|hard_auc|-0.44364|-0.41206/-0.37680/-0.54208|4/13/13|
|V2−V0|probability_auc|+0.00776|-0.96489/+0.64572/+0.34246|16/1/13|
|V2−V0|sensitivity_asd|-0.74390|+0.47561/-0.97561/-1.73171|4/18/8|
|V2−V0|specificity_td|-0.14339|-1.29972/+0.22202/+0.64755|7/16/7|
|V3−V0|acc|-0.00000|+0.11494/-0.34483/+0.22989|9/13/8|
|V3−V0|ba|+0.00171|+0.12500/-0.25235/+0.13250|11/9/10|
|V3−V0|hard_auc|+0.00171|+0.12500/-0.25235/+0.13250|11/9/10|
|V3−V0|probability_auc|+0.23382|-0.16935/+0.60152/+0.26929|16/0/14|
|V3−V0|sensitivity_asd|-0.07520|+0.25000/+0.99390/-1.46951|7/16/7|
|V3−V0|specificity_td|+0.07863|+0.00000/-1.49861/+1.73451|8/12/10|
|V4−V0|acc|-0.03788|-0.34483/+0.34483/-0.11364|6/17/7|
|V4−V0|ba|-0.01081|-0.25287/+0.37500/-0.15456|10/8/12|
|V4−V0|hard_auc|-0.01081|-0.25287/+0.37500/-0.15456|10/8/12|
|V4−V0|probability_auc|-0.00762|-0.69873/+0.43180/+0.24407|15/0/15|
|V4−V0|sensitivity_asd|+0.40854|+1.21951/+0.75000/-0.74390|10/13/7|
|V4−V0|specificity_td|-0.43016|-1.72525/+0.00000/+0.43478|7/13/10|
|V6−V0|acc|-0.11451|-0.45846/+0.34483/-0.22989|4/19/7|
|V6−V0|ba|-0.15509|-0.41764/+0.33777/-0.38538|7/9/14|
|V6−V0|hard_auc|-0.15509|-0.41764/+0.33777/-0.38538|7/9/14|
|V6−V0|probability_auc|+0.19199|-0.63763/+0.48084/+0.73275|17/0/13|
|V6−V0|sensitivity_asd|-0.74187|+0.23780/+0.25000/-2.71341|5/15/10|
|V6−V0|specificity_td|+0.43170|-1.07308/+0.42553/+1.94265|11/11/8|
|V8−V0|acc|-0.22989|-0.11494/-0.11494/-0.45977|3/18/9|
|V8−V0|ba|-0.23640|-0.19411/-0.05284/-0.46225|6/10/14|
|V8−V0|hard_auc|-0.23640|-0.19411/-0.05284/-0.46225|6/10/14|
|V8−V0|probability_auc|-0.12072|-0.71125/+0.42768/-0.07858|13/2/15|
|V8−V0|sensitivity_asd|-0.40650|-1.24390/+0.75000/-0.72561|6/14/10|
|V8−V0|specificity_td|-0.06630|+0.85569/-0.85569/-0.19889|8/14/8|
|V9−V0|acc|-0.26776|-0.22858/-0.22989/-0.34483|3/18/9|
|V9−V0|ba|-0.32610|-0.24747/-0.24695/-0.48387|7/8/15|
|V9−V0|hard_auc|-0.32610|-0.24747/-0.24695/-0.48387|7/8/15|
|V9−V0|probability_auc|+0.09165|-0.31688/+0.36954/+0.22229|18/1/11|
|V9−V0|sensitivity_asd|-1.07927|-0.26829/-0.49390/-2.47561|5/12/13|
|V9−V0|specificity_td|+0.42707|-0.22664/+0.00000/+1.50786|11/12/7|

## 8. 判别性、Shared/Private信息与坍塌

最佳checkpoint之后记录；类别原型为训练未加权类均值再归一化，cos趋近1可与信息弱化并存，但单个cos不充分证明所有信息消失。probe固定Scaler+LR训练折拟合、测试折诊断，不参与选模；实验17用Ridge且原型按Align加权，因此历史probe/原型数值不是完全相同口径。当前V3/V8及V2/V1比较才是主要保护证据。

|组|FC/PH Shared类原型cos(train)|FC Shared BA/AUC%|FC Private BA/AUC%|PH Shared BA/AUC%|PH Private BA/AUC%|FC/PH Shared-Private abs cos|
|---|---|---|---|---|---|---|
|V1|0.733351/0.982237|78.5678/85.6542|78.0034/85.8571|48.8075/51.0912|49.7578/51.3224|0.169364/0.145375|
|V2|-0.268629/0.992858|85.9978/92.0082|75.0456/82.6598|48.5539/51.2107|48.7388/51.1344|0.104528/0.141361|
|V3|0.950586/0.999940|86.5718/92.7832|76.8358/83.9795|48.8792/51.8151|49.3261/51.5079|0.136883/0.171498|
|V4|0.920418/0.999909|86.9229/92.8123|76.1995/84.1154|48.5320/52.0337|49.4989/51.2413|0.136815/0.181956|
|V5|0.874152/0.998994|86.7417/92.9502|76.3631/83.8359|48.5913/52.5944|48.9873/51.3496|0.122367/0.183108|
|V6|0.912955/0.999887|86.5675/92.5574|74.8073/82.8337|48.1920/51.8531|50.0921/51.7528|0.146715/0.194225|
|V7|0.866589/0.998942|86.8944/92.6100|83.7558/90.7431|48.3457/52.2765|49.7385/51.8568|0.223657/0.183086|
|V8|0.999967/0.999998|62.1523/68.6126|77.0861/84.6934|48.6718/51.8414|50.3067/51.9433|0.128062/0.144496|
|V9|0.896252/0.999948|86.7264/92.8683|76.4885/83.6037|48.6936/51.9212|49.0275/51.2635|0.160616/0.211753|

V2−V1机制均值差：FC Shared类原型cos -1.001981；FC Shared probe BA +7.4299pp；FC Private probe BA -2.9578pp。

V3−V8机制均值差：FC Shared类原型cos -0.049381；FC Shared probe BA +24.4194pp；FC Private probe BA -0.2503pp。

V7−V5机制均值差：FC Shared类原型cos -0.007563；FC Shared probe BA +0.1528pp；FC Private probe BA +7.3927pp。

|组|FC Shared类cos>0.99折数|PH Shared类cos>0.99折数|跨模态Shared平方距离|FC Shared同/异类距离|FC Shared/Private秩|PH Shared/Private秩|
|---|---:|---:|---:|---|---|---|
|V1|5/30|11/30|2.006986|0.498013/0.819752|15.4918/15.1809|16.5050/16.3565|
|V2|0/30|24/30|1.978510|1.018689/2.134465|19.0995/16.5522|17.0087/16.9360|
|V3|0/30|30/30|0.042643|0.035127/0.132654|16.1351/16.2584|14.4071/16.6954|
|V4|1/30|30/30|0.078013|0.068223/0.224218|18.4939/17.3902|15.1646/16.6363|
|V5|0/30|30/30|0.138992|0.104484/0.348283|18.3932/16.9945|10.3158/16.6708|
|V6|0/30|30/30|0.087752|0.079991/0.250014|18.9002/17.7452|15.0610/16.5696|
|V7|0/30|30/30|0.148222|0.114424/0.371959|18.7854/18.7094|10.5137/16.5989|
|V8|30/30|30/30|0.001945|0.003598/0.003669|22.6822/14.4535|17.6203/15.4263|
|V9|0/30|30/30|0.110184|0.091205/0.294557|18.6492/17.2053|15.9514/16.6083|

Shared/Private低cos只说明方向不同；高Private probe只说明该表示可线性读出诊断关联，不证明融合净收益。V7必须同时结合V7−V5正式性能和Private probe增量解释，不能以蒸馏BCE下降代替保护成功。

## 9. 对齐强度、残差与梯度

loss及其梯度是最佳checkpoint eval模式的固定几何，未执行optimizer更新；训练期component梯度每25epoch及前两步采样，记录最大值而非每epoch重复统计。

|组|原分类loss|原图loss|disc原始|pair原始/加权|Private原始|权重均值/SD|PH/FC残差%|最大单Token残差%|
|---|---:|---:|---:|---|---:|---|---|---:|
|V0|0.137306|-1.802015|0.000000|0.000000/0.000000|0.000000|0.000000/0.000000|0.0000/0.0000|0.0000|
|V1|0.140289|-1.907603|0.766285|2.006986/0.000000|0.741064|0.000000/0.000000|3.4581/4.0405|7.8293|
|V2|0.141190|-1.872727|0.112011|1.978510/0.000000|0.743930|0.000000/0.000000|5.4251/5.5773|7.8261|
|V3|0.163243|-1.824557|0.140684|0.042643/0.042643|0.746040|1.000000/0.000000|5.7206/5.9768|8.1449|
|V4|0.153715|-1.835597|0.139046|0.078013/0.033714|0.741870|0.432582/0.000000|5.1648/5.5259|8.1786|
|V5|0.128025|-1.920928|0.127373|0.138992/0.028266|0.739435|0.432582/0.428632|5.3374/5.5363|7.3639|
|V6|0.150676|-1.773314|0.139374|0.087752/0.032134|0.720409|0.432582/0.428632|5.5021/5.8365|8.4060|
|V7|0.139955|-1.907850|0.122893|0.148222/0.030091|0.287227|0.432582/0.428632|5.7448/5.7845|7.8661|
|V8|0.115518|-1.832440|0.891336|0.001945/0.001945|0.754991|1.000000/0.000000|3.5640/4.1903|8.4812|
|V9|0.134250|-1.851111|0.120241|0.107575/0.033197|0.739380|0.432582/0.428632|5.2320/5.4453|8.2821|

|组|分类梯度|原图梯度|0.1disc梯度|0.1pair梯度|0.05private梯度|有效新增结构参数|近零目标Token%|alpha PH/FC均值|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|V0|5.003649|0.382460|0.000000|0.000000|0.000000|0|0.0000|+0.000000/+0.000000|
|V1|5.251659|0.321590|0.000000|0.000000|0.000000|31498|0.0000|-0.040734/+0.021470|
|V2|5.726591|0.304866|0.143713|0.000000|0.000000|31564|0.0000|-0.119785/+0.046030|
|V3|4.363817|0.315680|0.098610|0.023849|0.000000|31564|0.0000|-0.090152/-0.092602|
|V4|4.501138|0.353843|0.139268|0.023269|0.000000|31564|0.0000|+0.118514/+0.219503|
|V5|4.078380|0.276525|0.112946|0.022203|0.000000|31564|0.0000|+0.192515/+0.112513|
|V6|5.071393|0.394003|0.175105|0.024124|0.000000|31564|0.0000|+0.048803/+0.170764|
|V7|4.673539|0.242006|0.122352|0.023140|0.045440|31597|0.0000|-0.070964/-0.105146|
|V8|4.254308|0.314825|0.000000|0.001016|0.000000|31498|0.0000|+0.073633/+0.061300|
|V9|5.422844|0.285648|0.120375|0.024904|0.000000|31564|0.0000|-0.091920/+0.080430|

新增alpha有符号；其初始范数为0，norm-ratio仅数值记账，不解释为权重收缩率。ANAT/FUNC新残差严格0，注入PHENO/CORR位置已经混入CAM上下文，其大小不能作为纯模态因果贡献或真实ROI证据。全部预测/loss/梯度有限。训练期实测component梯度与头启用状态吻合，disabled_head_components不计有效容量。参数范数与方差的完整数字见单条fold诊断。

## 10. V5同checkpoint五条件反事实

不重训、不重新选epoch、主干参数及注入前Token逐值不变；置乱仅新增年龄/性别输入且不用测试标签，结束恢复REAL逐值一致。r不进入推理，未做推理交换r。

|条件|ΔACC pp|ΔBA pp|ΔProb-AUC pp|翻转%|ASD概率绝对变化|logit绝对变化|残差绝对变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|REAL|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|
|SHARED_OFF|-0.87904911|-0.90956695|-0.06669149|0.95567746|0.01158492|0.05785531|0.00764891|
|PRIVATE_OFF|-0.30651341|-0.30205960|0.00893106|0.38314176|0.00283614|0.00900496|0.00158164|
|RESIDUAL_OFF|-1.26219087|-1.26340599|-0.05649316|1.33881923|0.01386775|0.06722283|0.00586959|
|PHENO_SHUFFLED|-0.26776385|-0.28139595|-0.00702639|0.34439220|0.00281849|0.00887127|0.00111548|

这些只检验训练后分支依赖；OFF下降可能因为主干/分类器共同适应，不能代替V0重训净增益。Shared/Private输入清零带来分布变化，PHENO错配敏感也不自动说明真实对应提高算法性能。患者对应是否有净价值主要看V5−V9重训对照。

## 11. 预设科学判断

BA和Prob-AUC三个seed均正向，且采用更保守的三个seed ACC均不下降条件才记稳定支持；这是描述性规则，无统计显著性含义。所有SEN/SPE差已列，提升整体ACC不能掩盖ASD漏检。

- A 判别性保护（V2−V1）：稳定支持=False；ACC/BA/Prob-AUC均值差 -0.34396pp/-0.36834pp/-0.37272pp。
- B 统一对齐增量（V3−V2）：稳定支持=False；ACC/BA/Prob-AUC均值差 +0.42102pp/+0.44536pp/+0.22606pp。
- C 对齐时判别保护（V3−V8）：稳定支持=False；ACC/BA/Prob-AUC均值差 +0.22989pp/+0.23811pp/+0.35453pp。
- D 同强度患者选择（V5−V4）：稳定支持=False；ACC/BA/Prob-AUC均值差 -0.03831pp/-0.05219pp/+0.09953pp。
- E 正确可靠度分配（V5−V6）：稳定支持=False；ACC/BA/Prob-AUC均值差 +0.03831pp/+0.09208pp/-0.10008pp。
- F 真实患者配对（V5−V9）：稳定支持=False；ACC/BA/Prob-AUC均值差 +0.19157pp/+0.26309pp/+0.00026pp。
- G Private教师蒸馏（V7−V5）：稳定支持=False；ACC/BA/Prob-AUC均值差 -0.11538pp/-0.12528pp/+0.11233pp。
- H1 V5超过P0（V5−V0）：稳定支持=False；ACC/BA/Prob-AUC均值差 -0.07619pp/-0.06300pp/+0.09191pp。
- H2 V7超过P0（V7−V0）：稳定支持=False；ACC/BA/Prob-AUC均值差 -0.19157pp/-0.18828pp/+0.20424pp。

如果V5仅超过V3却不能超过V4，应优先考虑整体弱化对齐而非患者选择；若V5未稳定超过V6/V9，本轮分别不能支持正确可靠度或真实患者对应的独立价值。这并不证明总体效应为零，但不足以支持预设机制。

## 12. 与P0及历史E9的比较

历史E9真实30折均值：acc 91.88262%/ba 91.69151%/hard_auc 91.69151%/probability_auc 93.82272%/sensitivity_asd 89.24797%/specificity_td 94.13506%。其P3及随机轨迹不同，只作同数据划分上的历史描述性对照。

|组−E9|指标|均值差pp|seed0/1/2差pp|W/T/L|
|---|---|---:|---|---|
|V0−E9|acc|-0.19157|-0.34483/+0.00000/-0.22989|2/21/7|
|V0−E9|ba|-0.15216|-0.31841/+0.00000/-0.13808|9/9/12|
|V0−E9|hard_auc|-0.15216|-0.31841/+0.00000/-0.13808|9/9/12|
|V0−E9|probability_auc|-0.48752|-0.26264/-0.81554/-0.38438|15/0/15|
|V0−E9|sensitivity_asd|+0.41260|+0.00610/-0.00000/+1.23171|9/14/7|
|V0−E9|specificity_td|-0.71693|-0.64292/+0.00000/-1.50786|6/11/13|
|V1−E9|acc|-0.26863|-0.34613/-0.11364/-0.34613|5/15/10|
|V1−E9|ba|-0.22747|-0.30641/-0.13363/-0.24237|11/7/12|
|V1−E9|hard_auc|-0.22747|-0.30641/-0.13363/-0.24237|11/7/12|
|V1−E9|probability_auc|-0.10704|-0.43307/-0.16170/+0.27366|14/1/15|
|V1−E9|sensitivity_asd|+0.32520|+0.46951/-0.49390/+1.00000|12/12/6|
|V1−E9|specificity_td|-0.78014|-1.08233/+0.22664/-1.48474|6/11/13|
|V2−E9|acc|-0.61259|-0.80329/-0.34483/-0.68966|1/15/14|
|V2−E9|ba|-0.59581|-0.73047/-0.37680/-0.68016|3/11/16|
|V2−E9|hard_auc|-0.59581|-0.73047/-0.37680/-0.68016|3/11/16|
|V2−E9|probability_auc|-0.47975|-1.22752/-0.16982/-0.04192|11/0/19|
|V2−E9|sensitivity_asd|-0.33130|+0.48171/-0.97561/-0.50000|6/16/8|
|V2−E9|specificity_td|-0.86031|-1.94265/+0.22202/-0.86031|6/14/10|
|V3−E9|acc|-0.19157|-0.22989/-0.34483/+0.00000|5/17/8|
|V3−E9|ba|-0.15045|-0.19341/-0.25235/-0.00558|9/10/11|
|V3−E9|hard_auc|-0.15045|-0.19341/-0.25235/-0.00558|9/10/11|
|V3−E9|probability_auc|-0.25370|-0.43198/-0.21402/-0.11509|11/2/17|
|V3−E9|sensitivity_asd|+0.33740|+0.25610/+0.99390/-0.23780|8/15/7|
|V3−E9|specificity_td|-0.63830|-0.64292/-1.49861/+0.22664|6/13/11|
|V4−E9|acc|-0.22945|-0.68966/+0.34483/-0.34352|5/15/10|
|V4−E9|ba|-0.16297|-0.57128/+0.37500/-0.29264|12/7/11|
|V4−E9|hard_auc|-0.16297|-0.57128/+0.37500/-0.29264|12/7/11|
|V4−E9|probability_auc|-0.49514|-0.96137/-0.38373/-0.14031|8/1/21|
|V4−E9|sensitivity_asd|+0.82114|+1.22561/+0.75000/+0.48780|12/11/7|
|V4−E9|specificity_td|-1.14709|-2.36818/+0.00000/-1.07308|4/15/11|
|V5−E9|acc|-0.26776|-0.45846/+0.00000/-0.34483|5/13/12|
|V5−E9|ba|-0.21517|-0.39828/+0.02629/-0.27350|8/7/15|
|V5−E9|hard_auc|-0.21517|-0.39828/+0.02629/-0.27350|8/7/15|
|V5−E9|probability_auc|-0.39561|-0.82878/-0.38183/+0.02379|14/0/16|
|V5−E9|sensitivity_asd|+0.42073|+0.49390/+0.25610/+0.51220|9/13/8|
|V5−E9|specificity_td|-0.85106|-1.29047/-0.20352/-1.05920|6/13/11|
|V6−E9|acc|-0.30608|-0.80329/+0.34483/-0.45977|2/18/10|
|V6−E9|ba|-0.30725|-0.73605/+0.33777/-0.52346|9/6/15|
|V6−E9|hard_auc|-0.30725|-0.73605/+0.33777/-0.52346|9/6/15|
|V6−E9|probability_auc|-0.29553|-0.90026/-0.33470/+0.34837|11/0/19|
|V6−E9|sensitivity_asd|-0.32927|+0.24390/+0.25000/-1.48171|10/9/11|
|V6−E9|specificity_td|-0.28523|-1.71600/+0.42553/+0.43478|9/10/11|
|V7−E9|acc|-0.38314|-0.80460/+0.34483/-0.68966|2/14/14|
|V7−E9|ba|-0.34045|-0.71134/+0.37805/-0.68805|6/8/16|
|V7−E9|hard_auc|-0.34045|-0.71134/+0.37805/-0.68805|6/8/16|
|V7−E9|probability_auc|-0.28328|-0.66566/+0.27364/-0.45781|12/0/18|
|V7−E9|sensitivity_asd|+0.17480|+0.50610/+0.75610/-0.73780|8/15/7|
|V7−E9|specificity_td|-0.85569|-1.92877/+0.00000/-0.63830|8/9/13|
|V8−E9|acc|-0.42146|-0.45977/-0.11494/-0.68966|1/19/10|
|V8−E9|ba|-0.38856|-0.51252/-0.05284/-0.60033|6/9/15|
|V8−E9|hard_auc|-0.38856|-0.51252/-0.05284/-0.60033|6/9/15|
|V8−E9|probability_auc|-0.60823|-0.97388/-0.38786/-0.46296|10/0/20|
|V8−E9|sensitivity_asd|+0.00610|-1.23780/+0.75000/+0.50610|7/13/10|
|V8−E9|specificity_td|-0.78323|+0.21277/-0.85569/-1.70675|6/13/11|
|V9−E9|acc|-0.45933|-0.57341/-0.22989/-0.57471|2/16/12|
|V9−E9|ba|-0.47826|-0.56588/-0.24695/-0.62195|8/7/15|
|V9−E9|hard_auc|-0.47826|-0.56588/-0.24695/-0.62195|8/7/15|
|V9−E9|probability_auc|-0.39587|-0.57952/-0.44600/-0.16208|13/1/16|
|V9−E9|sensitivity_asd|-0.66667|-0.26220/-0.49390/-1.24390|8/12/10|
|V9−E9|specificity_td|-0.28986|-0.86957/+0.00000/-0.00000|8/11/11|

## 13. 运行、恢复与完整验收

GPU0/GPU1分别15任务、150fold；每卡最多4worker、每worker2CPU线程；正式墙钟28.53分钟。任务如下：

GPU0：V0_seed0, V0_seed2, V1_seed1, V2_seed0, V2_seed2, V3_seed1, V4_seed0, V4_seed2, V5_seed1, V6_seed0, V6_seed2, V7_seed1, V8_seed0, V8_seed2, V9_seed1。

GPU1：V0_seed1, V1_seed0, V1_seed2, V2_seed1, V3_seed0, V3_seed2, V4_seed1, V5_seed0, V5_seed2, V6_seed1, V7_seed0, V7_seed2, V8_seed1, V9_seed0, V9_seed2。

资源峰值：{"worker_rss_sum_gib": 25.102310180664062, "gpu0_mib": 3059, "gpu1_mib": 3059}。GPU峰值是整卡含其他项目；worker RSS为本实验进程驻留内存加总。失败/恢复：{}；调度限流：[]。

P0逐折六指标及best/executed epoch与实验13F0/14E00/15A0/16B0/17D0全部30折一致；历史对应初始化序列化、14–17RNG也核对。原前向、原两步Adam及RNG、新零alpha初始化、完整第三步恢复、权重0.03比例、训练标签边界和所有五反事实均预检通过。

不同组与V0相同初始主干hash的fold数：{"V0": 30, "V1": 5, "V2": 3, "V3": 3, "V4": 3, "V5": 4, "V6": 3, "V7": 5, "V8": 3, "V9": 4}。由于组间epoch差异导致下一fold随机流分叉，不能把所有fold解释为严格相同初始化配对。

完整源、数据、CLIP、证据及旧实验保护hash见manifest；当前300唯一折、300诊断、150反事实、两卡各150折和同强度冻结证据验收通过。完成控制器随后安全追加handoff、清理本实验临时源码/权重，最后标记COMPLETE。未自动commit/push。

## 14. 文献边界、MICCAI2027与下一阶段

Shared/Private、辅助类别监督、蒸馏、加权对齐和残差融合均为已有组件。候选研究点为outer-train跨模态联合正确性代理驱动的可拒绝对齐，且同权重预算、错配权重和同类错配患者分别验证选择与对应。r仅控制训练损失，不进入推理融合；不是推理期患者Gate，也不是真实医学可靠度或实际对齐收益标签。

- [MGDR](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)：模态关系图、SVD共性表示、私有表示对比约束及融合已有先例；本轮不使用SVD或私有对比损失。 阅读范围：官方可解析全文方法及消融；shell下载返回空文本二页包装，未把包装当全文。
- [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)：原GFR/CAM/SPGS/RES作为固定主路径，保留实际源码与论文差异。 阅读范围：根目录11页原论文全文及Fig.1图像。
- [Class-Aware Semantic Decoupling](https://papers.miccai.org/miccai-2026/1073-Paper1452.html)：CSSM/SDL/CSSF在类别语义层实现共享/特异、对齐和融合；本轮无类别文本查询或缺失模态任务。 阅读范围：官方摘要；PDF入口不可可靠解析，不声称已读全文。
- [To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/download/39248/43209)：显式对齐的价值随模态冗余和特异信息变化；本轮冻结系数，用同强度常量控制区分选择与整体强度。 阅读范围：官方AAAI26论文入口与14页arXiv全文。

本历史协议内第一模块支持标准（患者选择+正确权重+真实对应+P0净收益同时成立）=False。完整MICCAI2027科学证据要求仍为未达到：当前test选checkpoint、全队列标准化/历史RFE、transductive评估及无独立站点/队列验证限制外部泛化结论；组件组合本身也不是独立创新。

若以上机制未获支持，接受负结果、保留P0，不事后改Prompt或搜索损失系数。若局部指标改善而核心配对不稳定，同样不包装成患者级选择有效。下一步最值得研究的问题是：具有真实疾病症状信息的临床表型，是否在训练内选模和独立站点评估中提供超过FC的增量，并能预测哪些患者确实从跨模态对齐获益。当前年龄/性别联合正确性不等于该收益目标。


<!-- experiment18_post_training_interpretation -->

## 15. 实测结果综合解读及代理选择的局限

本节在全部正式训练与验收完成后，基于冻结CSV、evidence.npz和原输入补充综合解读。没有修改模型、损失、配置、初始化或执行源码，也没有增加训练、重新选Checkpoint或搜索超参数。以下可靠度按10个outer-train分别统计后取均值，仅使用训练标签。

**判别性保护在表征层有效，但不是净分类收益的充分条件。** V2相对V1，FC Shared probe BA提高 7.42994pp，训练类原型cos从 0.733351 降至 -0.268629；主模型ACC/BA/Prob-AUC却分别变化 -0.34396/-0.36834/-0.37272pp。统一对齐下，V8的FC Shared原型cos=0.999967、probe BA=62.15234%；V3相应为 0.950586、86.57177%，probe BA增量 24.41944pp。因此本轮确实缓解FC Shared的判别信息弱化。V3−V8的Prob-AUC三个seed均提高，但ACC/BA在seed1下降，仍不足以声称全指标稳定收益。

**PHENO没有随之获得可泛化的判别信息。** V5年龄/性别Shared类原型cos=0.998994，30/30折大于0.99；其probe BA=48.59130%，而FC Shared为 86.74166%。结合训练OOF中PHENO接近机会水平、加入FC后无增量，不能把配对距离减小理解为学到了新的疾病共同信息。类别cos趋同与低probe共同提示PHENO的疾病区分弱，但一个cos本身不是完全坍塌的充分证据。

**真正的患者选择没有超过同强度控制。** V5−V4的ACC/BA/Prob-AUC为 -0.03831/-0.05219/+0.09953pp。连续AUC三个seed虽均正向，但ACC与BA未稳定提高，SEN-ASD下降 0.40041pp。V5−V6的ACC/BA均值略正，Prob-AUC却为 -0.10008pp且seed方向混合；没有支持正确可靠度分配具有稳定、独立的净价值。

**真实患者配对有局部阈值证据，但不是完整方法成功。** V5−V9的ACC/BA/SEN-ASD为 +0.19157/+0.26309/+1.08740pp，这三项在三个seed均为正；SPE-TD却下降 0.56121pp，连续AUC增量仅 +0.00026pp（seed0为负）。可以报告真实配对改善了阈值分类的局部现象，不能把它说成连续排序或全指标稳定改善。V9按患者索引无固定点，同类标签保持；由于年龄/性别分箱重复，平均 7.64125% 的置乱配对仍具有完全相同的28维PHENO（fold范围 6.50510–8.67347%）。控制实际改变了约 92.36% 的PHENO特征配对，不能把索引置乱等同于所有年龄/性别内容均不同。

**联合正确性代理带有类别预算偏差。**

|训练折类别|q均值|V5正确r均值|V6置乱r均值|
|---|---:|---:|---:|
|ASD=0|0.254194|0.307508|0.435155|
|TD=1|0.586174|0.540269|0.430374|

V5偏向给TD分配更强的对齐约束，V6全体置乱后类别均值更接近。V5/V6严格保持总体权重集合和预算，却不保持每类预算，因此其差异即使为正，也不能完全区分个体选择与类别强度分配。这是预定控制的识别边界，不事后添加训练组。q描述两教师都预测正确，没有直接测量某患者对齐后的诊断增益；较弱PHENO教师可能使r偏向容易病例、人口学分组或预测类别。训练OOF与测试标签隔离保证证据来源合规，不能保证这个代理目标本身正确。

**Private蒸馏保护了FC可读出信息，但没有提高最终诊断。** V7相对V5，FC Private probe BA由 76.36312% 提高至 83.75582%（+7.39271pp），Prob-AUC提高 +6.90715pp；最终ACC/BA反而变化 -0.11538/-0.12528pp，最终Prob-AUC +0.11233pp且seed不一致。教师信息可以进入Private，不意味着融合增加了相对原P0的新诊断信息，也不证明Shared/Private已解耦。

**辅助分支被使用，失败不是因为残差接近零。** V5在PHENO/CORR位置的平均相对残差为 5.33744%/5.53629%，近零目标Token比例为 0.00000%，原ANAT/FUNC位置新增残差严格为0。关闭全部残差，预测翻转 1.33882%、ACC变化 -1.26219pp；患者PHENO错配置换时翻转 0.34439%、平均ASD概率变化 0.00281849、ACC变化 -0.26776pp。翻转计数分别为 35 和 9，分母为三seed合计2613次测试记录（同871人重复评估，不能称2613名独立患者）。Shared/Private关闭分别翻转 25/10 条记录。这说明训练后存在分支依赖和少量患者配对敏感性，但OFF干预有共同适应及分布变化，且REAL模型以test ACC选Checkpoint，因此不能把OFF下降当成相对V0的净收益。

**正式结论及研究优先级。** 本轮最佳平均ACC为V0/V3并列；V3的最佳BA仅比V0高 +0.00171pp，不能作为实质提升。最高Prob-AUC为容量控制V1，仍低于历史E9。完整V5相对P0的ACC/BA为 -0.07619/-0.06300pp，相对E9的ACC/Prob-AUC为 -0.26776/-0.39561pp；V7相对E9相应为 -0.38314/-0.28328pp。已验证的局部机制是FC判别保护、Private蒸馏可读出信息以及部分真实配对阈值敏感性；患者选择的独立价值与超过P0/E9的稳定净收益未获支持，未达到支撑MICCAI2027第一创新模块的证据要求。保留P0，不再为当前年龄/性别代理调Prompt或搜索损失系数。下一步唯一优先问题是：真实症状表型在训练内选模和独立站点评估中能否提供超过FC的增量，并据此定义患者实际对齐收益；当前输入不含这些症状，不得伪造。


<!-- experiment19_contrastive_disease_evidence_alignment -->

## 实验十九：AG-CDEA解剖锚定疾病对比证据对齐

正式完成：14组、42任务、420折、420诊断、330同checkpoint反事实；两卡各21任务210折，最多4worker/卡、2线程/worker。正式墙钟0.894小时。原源码与实验1–18共654历史文件哈希未变，未commit/push/pull/reset/checkout，临时权重/源码已清理且无本轮worker。

研究动机：实验9/10疾病文字必要性未确证；12/13患者FC文字无稳定收益；14投影趋同；15数值旁路使文本效应不传到预测；16修复敏感性但残差32.33%、净性能下降；17类原型坍塌；18 FC判别保护有效但弱年龄/性别互补性与可靠度代理没有稳定最终收益。停止方案一年龄/性别对齐，方案二检验可信FC的解剖候选+疾病差异查询能否带来额外证据。

结构：252个可信Fisher-z连接→[z,abs(z),tanh(z)]→共享32维编码+边ID+LN；冻结RN50 ASD−TD四域单位差异查询，经自由Key检索真实解剖候选；score/0.2与固定log(n) NULL做softmax，NULL Value=0。读取4×32 Value、4个u、4个NULL→136/64/36，以alpha=0的≤10%相对L2残差只增强CAM后CORR位置3、SPGS前。原8句P0、GFR/CAM/SPGS/RES及输出头不变；原损失另可启用0.1 BCE(u_global[train],ytrain==0)，仅用outer train标签。

固定消融：C0 P0；C1零查询数值容量；C2 REAL无证据监督；C3 REAL全252候选；C4完整；C5中性；C6无关医学；C7仅类别；C8类别方向反转；C9域文字错配[2,3,0,1]；C10同列置换错配解剖；C11同Gram随机正交子空间；C12随机可学习查询；C13强制匹配。全部450epoch/200 patience、原连续10fold/seed0,1,2；原双Adam .0038/WD.11、新参数独立WD0；fork_rng不改变原初始化。

实验11映射全871/871数值唯一匹配；ROI名称联合111标签表，不用旧UNKNOWN名称列。110可信ROI，排除FC195/225/232/236未知端点，原主干320维保留。四域候选72/56/45/32、并集165，可重叠；四域是弱粗先验，不是已验证功能网络或病灶。CLIP文本32句一次冻结，REAL八候选未改字；缓存静态主题/差异查询和mask，不含患者数据。C11 Gram误差2.08e−7，方向近正交。

新增注册C1–C11/C13=53221、C12=57317；扣除恒失活项，C1=17413、C2/C4–C11=50437、C3=53221、C12=54533、C13=50181。Key功能自由度rank(Q)×32；自由投影能吸收固定查询旋转/类别方向，注册及非零梯度量不等于独立医学信息。

训练前另已推导NULL导致u≥−0.055692909、sigmoid≥0.486080371；C13无此下界，NULL对照含证据值域和恒零融合列的差异。NULL不是校准医学可靠度。

六指标单位%，先每seed平均十折，再三seed均值±population SD：

|组|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|---:|
|C0|91.6910±0.0542|91.5394±0.0841|91.5394±0.0841|93.3352±0.4154|89.6606±0.6456|93.4181±0.5102|
|C1|91.4228±0.6249|91.2442±0.6111|91.2442±0.6111|92.9442±0.5416|88.9207±0.4257|93.5677±0.7977|
|C2|91.4999±0.1872|91.3091±0.1776|91.3091±0.1776|93.2580±0.1771|88.7622±0.1078|93.8560±0.2662|
|C3|91.6527±0.1954|91.4442±0.2067|91.4442±0.2067|93.3212±0.1188|88.6809±0.4989|94.2075±0.1776|
|C4|91.5378±0.1434|91.3788±0.1822|91.3788±0.1822|93.4030±0.0323|89.3333±1.1213|93.4243±0.8638|
|C5|91.8447±0.1629|91.6395±0.1824|91.6395±0.1824|93.2297±0.0982|89.0793±0.5316|94.1998±0.1813|
|C6|91.6527±0.2709|91.4713±0.2796|91.4713±0.2796|93.6272±0.2159|89.2439±0.3178|93.6987±0.2728|
|C7|91.6140±0.0939|91.4249±0.1053|91.4249±0.1053|93.7667±0.2005|89.0000±0.3064|93.8498±0.1036|
|C8|91.4612±0.2709|91.2435±0.2669|91.2435±0.2669|93.5155±0.0924|88.4228±0.3026|94.0641±0.2639|
|C9|91.4228±0.0542|91.2623±0.0509|91.2623±0.0509|93.4965±0.0701|89.1728±0.8178|93.3518±0.7659|
|C10|91.6910±0.1434|91.5556±0.1416|91.5556±0.1416|93.4308±0.1421|89.8333±0.2041|93.2778±0.2679|
|C11|91.6906±0.2861|91.4671±0.2646|91.4671±0.2646|93.2045±0.1789|88.5894±0.1992|94.3447±0.6196|
|C12|91.4616±0.2170|91.3031±0.2229|91.3031±0.2229|93.4904±0.2345|89.1789±0.3140|93.4274±0.2683|
|C13|91.4612±0.1434|91.2615±0.1715|91.2615±0.1715|93.2626±0.1312|88.6748±0.6477|93.8483±0.3637|

核心比较均值差pp，全部seed差与逐折胜/平/负见报告；同划分30折不是30独立外部样本：

|配对|ΔACC|ΔBA|ΔProb-AUC|三seed三指标均正向|
|---|---:|---:|---:|---|
|C1−C0|-0.2682|-0.2951|-0.3910|False|
|C2−C1|+0.0771|+0.0649|+0.3138|False|
|C4−C2|+0.0379|+0.0697|+0.1450|False|
|C4−C3|-0.1149|-0.0654|+0.0818|False|
|C4−C5|-0.3069|-0.2607|+0.1733|False|
|C4−C6|-0.1149|-0.0925|-0.2242|False|
|C4−C7|-0.0762|-0.0461|-0.3637|False|
|C4−C8|+0.0766|+0.1354|-0.1125|False|
|C4−C9|+0.1149|+0.1165|-0.0935|False|
|C4−C10|-0.1533|-0.1768|-0.0278|False|
|C4−C11|-0.1528|-0.0883|+0.1985|False|
|C4−C12|+0.0762|+0.0757|-0.0874|False|
|C4−C13|+0.0766|+0.1173|+0.1404|False|
|C4−C0|-0.1533|-0.1605|+0.0678|False|
|C4−C1|+0.1149|+0.1346|+0.4588|False|

科学判断：正确疾病文本=False，正确解剖对应=False，NULL稳定净增益=False，C4对P0稳定净增益=False；完整第一创新模块候选支持=False。最佳均值组：{'acc': 'C5', 'ba': 'C5', 'probability_auc': 'C7'}。历史E9仅描述性，C4−E9：acc=-0.3448pp；ba=-0.3127pp；probability_auc=-0.4197pp。

C4的11项反事实同最佳权重、主干输入及注入前CAM不变，不重训、不读test标签置乱，恢复REAL逐值一致。替换文本的概率/注意力变化说明依赖查询坐标，不自动说明医学内容正确；FC_SHUFFLED/OFF有分布变化，必须结合重训对照。

|干预|翻转%|ASD概率绝对差|ΔACC pp|ΔBA pp|ΔProb-AUC pp|
|---|---:|---:|---:|---:|---:|
|REAL|0.0000|0|+0.0000|+0.0000|+0.0000|
|NEUTRAL|0.3065|0.004426235|-0.2299|-0.2552|-0.0388|
|UNRELATED|0.3065|0.003128021|-0.2299|-0.2552|-0.0017|
|CLASS_ONLY|0.2682|0.002643311|-0.1916|-0.2135|+0.0301|
|SWAPPED|0.3065|0.005138844|-0.2299|-0.2552|-0.0353|
|DOMAIN_PERMUTED|0.1916|0.00258942|-0.1149|-0.1312|+0.0213|
|ANATOMY_PERMUTED|0.2294|0.003984277|-0.2294|-0.2259|-0.0383|
|RANDOM_ROTATED|0.3448|0.004232061|-0.2682|-0.2969|-0.0424|
|NULL_OFF|0.1145|0.003071573|-0.1145|-0.1126|+0.0195|
|RESIDUAL_OFF|0.4593|0.006983531|-0.3827|-0.3970|-0.0281|
|FC_SHUFFLED|0.2678|0.005848443|-0.2678|-0.2730|+0.0213|

机制：C4平均相对残差3.8804%，最大6.6734%；非CORR残差严格0；近零折(<1e−6相对L2) 0/30。各域NULL均值 [0.8365833501021067, 0.8722666561603546, 0.9127297858397166, 0.814840966463089]。域Top5只是数据关联，未逐域删除，不能声称某域驱动全部改善。

运行资源峰值 {'worker_rss_sum_gib': 25.16901397705078, 'gpu0_mib': 50039, 'gpu1_mib': 54245}；正式失败/恢复 {'C9_seed0': 1, 'C11_seed0': 1}；更早工程writer错误尝试已整体舍弃7折，未混入CSV。C0全部30折与实验13–18六指标/epoch/初始化/RNG parity通过；模型/两Adam/完整RNG第三步复演通过。

限制：保留历史全局RFE/标准化、871人transductive、val=test ACC选checkpoint，不能称独立无泄漏泛化。静态CLIP词句没有患者级临床真值；自由Key的重参数化、NULL负分值域、不同早停导致后续连续fold随机流分叉均公开。

建议停止这一固定静态Prompt+自由Key路线的继续调词/调参。唯一优先问题：有独立真值、与FC定位对应的患者级临床证据，在限制自由语义映射后能否超越同几何匿名查询？本轮未添加配置。

完整报告：`/data/adodas/SPromptGL-main/experiment/experiment19_contrastive_disease_evidence_alignment/results/report.md`；SHA256 `9de20fe5c10e0f559c0cc33e8b4a81c312e9e9df7dc30ec8a3de72261b564ecf`。目录：`/data/adodas/SPromptGL-main/experiment/experiment19_contrastive_disease_evidence_alignment`；manifest保存数据、根源码、14组配置、旧实验保护、CLIP、split及所有签名。run.py SHA256 `c8468eb7702bd6ca16da680df52b62bfcabf0835ed1f250828bb6380a9108355`；cdea.py `7c6b69927dcc32976a3221ad4ba90c2896d86066fbd0c420e0a1306c1a121219`；prompts.json `18fd8677adb08ed1ce79c335c17f31b643ccdbdab852d08bbd3553c0a6abaeb5`；semantic_cache.npz `9d07a138a962767eec51aa4cd62a360366277628b9c95036e428eade9892f57d`；domain_mapping.csv `cd34776253b961f37112d52d588f8e086139446326b93df5b237972821e0c9bf`。代码HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`，未执行Git提交或发布。


## 实验二十：固定疾病概念锚定与患者级证据图联合验证

时间：2026-10-16T07:00:12.388403+00:00；状态：BLOCKED。实验目录：`/tmp/experiment20_grounded_concept_graph_alignment`。

实验10–19表明类别码容量、近零或共有语义响应、自由Key重参数化都是替代解释。本轮冻结REAL/NEUTRAL、真实252连接、4域候选[72,56,45,32]，从Encoder→FeedForward后以零初始化≤10% CORR残差加入局部证据及共享ROI患者线图，并原样保留P0/P3/E9/E3。

20配置：{"T00": {"prompt": "P0", "disease": false, "adv": false, "condition": null, "use_graph": false}, "T01": {"prompt": "P3", "disease": false, "adv": false, "condition": null, "use_graph": false}, "T02": {"prompt": "P3", "disease": true, "adv": false, "condition": null, "use_graph": false}, "T03": {"prompt": "P0", "disease": true, "adv": true, "condition": null, "use_graph": false}, "T04": {"prompt": "P0", "disease": false, "adv": false, "condition": "NUMERIC", "use_graph": false}, "T05": {"prompt": "P0", "disease": false, "adv": false, "condition": "REAL", "use_graph": false}, "T06": {"prompt": "P0", "disease": false, "adv": false, "condition": "NEUTRAL", "use_graph": false}, "T07": {"prompt": "P0", "disease": false, "adv": false, "condition": "WRONG_ANATOMY", "use_graph": false}, "T08": {"prompt": "P0", "disease": false, "adv": false, "condition": "ISOMETRIC", "use_graph": false}, "T09": {"prompt": "P0", "disease": false, "adv": false, "condition": "ANONYMOUS", "use_graph": false}, "T10": {"prompt": "P0", "disease": false, "adv": false, "condition": "NUMERIC", "use_graph": true}, "T11": {"prompt": "P0", "disease": false, "adv": false, "condition": "REAL", "use_graph": true}, "T12": {"prompt": "P0", "disease": false, "adv": false, "condition": "NEUTRAL", "use_graph": true}, "T13": {"prompt": "P0", "disease": false, "adv": false, "condition": "WRONG_ANATOMY", "use_graph": true}, "T14": {"prompt": "P3", "disease": true, "adv": false, "condition": "NUMERIC", "use_graph": false}, "T15": {"prompt": "P3", "disease": true, "adv": false, "condition": "REAL", "use_graph": false}, "T16": {"prompt": "P3", "disease": true, "adv": false, "condition": "NEUTRAL", "use_graph": false}, "T17": {"prompt": "P3", "disease": true, "adv": false, "condition": "NUMERIC", "use_graph": true}, "T18": {"prompt": "P3", "disease": true, "adv": false, "condition": "REAL", "use_graph": true}, "T19": {"prompt": "P3", "disease": true, "adv": false, "condition": "NEUTRAL", "use_graph": true}}

工程：0/60任务，0/600fold，0/600诊断，0/1080反事实；preflight=PASS。Git HEAD 1167dd392837a02eb4f83481ae49f02607a9dae5，参照 f36ed2e9f06b39e65d9ae70c549256dd39cf8e80；代码/数据/文本/权重/先验哈希见manifest。

阻塞：Task failed and formal run stopped: T01_seed2; old=1 epoch=299 best_acc=0.965909090909 wait=165
EPOCH fold=1 epoch=324 best_acc=0.965909090909 wait=190
EPOCH fold=1 epoch=335 best_acc=0.965909090909 wait=201
Traceback (most recent call last):
  File "/tmp/experiment20_grounded_concept_graph_alignment/run.py", line 1224, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/tmp/experiment20_grounded_concept_graph_alignment/run.py", line 1215, in main
    i=sys.argv.index('--worker');worker(sys.argv[i+1],int(sys.argv[i+2]));return
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/experiment20_grounded_concept_graph_alignment/run.py", line 885, in worker
    assert_historical_fold(row)
  File "/tmp/experiment20_grounded_concept_graph_alignment/run.py", line 729, in assert_historical_fold
    else:assert abs(row[key]-value)<(5.1e-9 if row['config']=='T01' and key in METRICS else 1e-12),('HISTORICAL_PARITY_FAILED',row['config'],row['seed'],row['fold'],key,row[key],value)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: ('HISTORICAL_PARITY_FAILED', 'T01', 2, 1, 'acc', 0.9659090909090909, 0.9659)
；未完成部分不作科学成败判定。


运行壁钟：未启动正式CV；GPU各30任务/300fold，最多4worker×2CPU线程。反事实为同checkpoint九条件；NA不算有效评估。全部结果和唯一后续建议见 `/tmp/experiment20_grounded_concept_graph_alignment/results/report.md`。
原test=val选模、全局监督RFE/标准化与transductive协议存在乐观风险，不称无泄漏泛化或医学因果。固定语义未优于阴性控制时停止追分。没有修改原源码或实验1–19，未commit/push。


## 实验二十：固定疾病概念锚定与患者级证据图联合验证

时间：2026-10-16T08:00:14.547333+00:00；状态：COMPLETE。实验目录：`/tmp/experiment20_grounded_concept_graph_alignment`。

实验10–19表明类别码容量、近零或共有语义响应、自由Key重参数化都是替代解释。本轮冻结REAL/NEUTRAL、真实252连接、4域候选[72,56,45,32]，从Encoder→FeedForward后以零初始化≤10% CORR残差加入局部证据及共享ROI患者线图，并原样保留P0/P3/E9/E3。

20配置：{"T00": {"prompt": "P0", "disease": false, "adv": false, "condition": null, "use_graph": false}, "T01": {"prompt": "P3", "disease": false, "adv": false, "condition": null, "use_graph": false}, "T02": {"prompt": "P3", "disease": true, "adv": false, "condition": null, "use_graph": false}, "T03": {"prompt": "P0", "disease": true, "adv": true, "condition": null, "use_graph": false}, "T04": {"prompt": "P0", "disease": false, "adv": false, "condition": "NUMERIC", "use_graph": false}, "T05": {"prompt": "P0", "disease": false, "adv": false, "condition": "REAL", "use_graph": false}, "T06": {"prompt": "P0", "disease": false, "adv": false, "condition": "NEUTRAL", "use_graph": false}, "T07": {"prompt": "P0", "disease": false, "adv": false, "condition": "WRONG_ANATOMY", "use_graph": false}, "T08": {"prompt": "P0", "disease": false, "adv": false, "condition": "ISOMETRIC", "use_graph": false}, "T09": {"prompt": "P0", "disease": false, "adv": false, "condition": "ANONYMOUS", "use_graph": false}, "T10": {"prompt": "P0", "disease": false, "adv": false, "condition": "NUMERIC", "use_graph": true}, "T11": {"prompt": "P0", "disease": false, "adv": false, "condition": "REAL", "use_graph": true}, "T12": {"prompt": "P0", "disease": false, "adv": false, "condition": "NEUTRAL", "use_graph": true}, "T13": {"prompt": "P0", "disease": false, "adv": false, "condition": "WRONG_ANATOMY", "use_graph": true}, "T14": {"prompt": "P3", "disease": true, "adv": false, "condition": "NUMERIC", "use_graph": false}, "T15": {"prompt": "P3", "disease": true, "adv": false, "condition": "REAL", "use_graph": false}, "T16": {"prompt": "P3", "disease": true, "adv": false, "condition": "NEUTRAL", "use_graph": false}, "T17": {"prompt": "P3", "disease": true, "adv": false, "condition": "NUMERIC", "use_graph": true}, "T18": {"prompt": "P3", "disease": true, "adv": false, "condition": "REAL", "use_graph": true}, "T19": {"prompt": "P3", "disease": true, "adv": false, "condition": "NEUTRAL", "use_graph": true}}

工程：60/60任务，600/600fold，600/600诊断，1080/1080反事实；preflight=PASS。Git HEAD 1167dd392837a02eb4f83481ae49f02607a9dae5，参照 f36ed2e9f06b39e65d9ae70c549256dd39cf8e80；代码/数据/文本/权重/先验哈希见manifest。

主结果，ACC/BA/Hard-AUC/Prob-AUC/SEN-ASD/SPE-TD（%，三seed mean±population SD）：

- T00: 91.69105±0.05418 / 91.53935±0.08408 / 91.53935±0.08408 / 93.33520±0.41541 / 89.66057±0.64562 / 93.41813±0.51023
- T01: 91.53779±0.21674 / 91.31013±0.21555 / 91.31013±0.21555 / 93.35305±0.20779 / 88.26626±0.51022 / 94.35399±0.56412
- T02: 91.88262±0.19536 / 91.69151±0.19290 / 91.69151±0.19290 / 93.82272±0.23719 / 89.24797±0.11931 / 94.13506±0.27073
- T03: 91.84431±0.09385 / 91.68351±0.12182 / 91.68351±0.12182 / 93.52231±0.22794 / 89.66057±0.51106 / 93.70644±0.26868
- T04: 91.53779±0.10837 / 91.36332±0.12316 / 91.36332±0.12316 / 93.31927±0.13938 / 89.16667±0.46856 / 93.55998±0.35123
- T05: 91.65273±0.10837 / 91.49140±0.11760 / 91.49140±0.11760 / 93.26474±0.24911 / 89.41667±0.46856 / 93.56614±0.34557
- T06: 91.53866±0.14348 / 91.33555±0.15725 / 91.33555±0.15725 / 93.42084±0.09869 / 88.67480±0.61872 / 93.99630±0.45893
- T07: 91.49948±0.16255 / 91.32158±0.19052 / 91.32158±0.19052 / 93.55169±0.27287 / 89.00610±0.64584 / 93.63706±0.27156
- T08: 91.42285±0.05418 / 91.26935±0.05303 / 91.26935±0.05303 / 93.22838±0.03026 / 89.33333±0.20164 / 93.20537±0.20278
- T09: 91.42285±0.10837 / 91.25795±0.12150 / 91.25795±0.12150 / 93.44358±0.07581 / 89.16870±0.65312 / 93.34721±0.52306
- T10: 91.84387±0.18823 / 91.65545±0.16364 / 91.65545±0.16364 / 93.54332±0.14766 / 89.17276±0.11952 / 94.13814±0.43719
- T11: 91.80643±0.14382 / 91.59976±0.14871 / 91.59976±0.14871 / 93.50250±0.20284 / 88.84553±0.34209 / 94.35399±0.20499
- T12: 91.50035±0.16225 / 91.31930±0.15412 / 91.31930±0.15412 / 93.67694±0.07811 / 89.00000±0.23288 / 93.63861±0.26507
- T13: 91.57611±0.10837 / 91.37900±0.12199 / 91.37900±0.12199 / 93.52154±0.01334 / 88.76016±0.50223 / 93.99784±0.30640
- T14: 91.88262±0.05418 / 91.69355±0.10099 / 91.69355±0.10099 / 93.29073±0.26170 / 89.25203±0.76425 / 94.13506±0.56275
- T15: 91.84474±0.16317 / 91.71198±0.14412 / 91.71198±0.14412 / 93.47719±0.22388 / 90.07520±0.35212 / 93.34875±0.52499
- T16: 91.65273±0.14336 / 91.44705±0.16783 / 91.44705±0.16783 / 93.55644±0.06122 / 88.67886±0.50910 / 94.21523±0.17561
- T17: 91.69105±0.05418 / 91.55253±0.06819 / 91.55253±0.06819 / 93.72639±0.08944 / 89.82724±0.35071 / 93.27783±0.27406
- T18: 91.65273±0.05418 / 91.45101±0.05266 / 91.45101±0.05266 / 93.41204±0.12765 / 88.83943±0.53724 / 94.06260±0.50149
- T19: 91.49991±0.16194 / 91.33995±0.18481 / 91.33995±0.18481 / 93.38260±0.25851 / 89.25407±0.50717 / 93.42584±0.20499
- T05−T06: acc +0.11407pp / ba +0.15586pp / probability_auc -0.15610pp
- T05−T08: acc +0.22989pp / ba +0.22206pp / probability_auc +0.03637pp
- T05−T07: acc +0.15326pp / ba +0.16982pp / probability_auc -0.28694pp
- T11−T10: acc -0.03744pp / ba -0.05569pp / probability_auc -0.04082pp
- T11−T12: acc +0.30608pp / ba +0.28046pp / probability_auc -0.17444pp
- T18−T19: acc +0.15282pp / ba +0.11106pp / probability_auc +0.02944pp
- T18−T17: acc -0.03831pp / ba -0.10152pp / probability_auc -0.31435pp
- T18−T02: acc -0.22989pp / ba -0.24050pp / probability_auc -0.41068pp

严格三seed三主要指标门槛（不等于显著性）：局部=False；图=False；组合=False。

运行壁钟：3608.4653470516205；GPU各30任务/300fold，最多4worker×2CPU线程。反事实为同checkpoint九条件；NA不算有效评估。全部结果和唯一后续建议见 `/tmp/experiment20_grounded_concept_graph_alignment/results/report.md`。
原test=val选模、全局监督RFE/标准化与transductive协议存在乐观风险，不称无泄漏泛化或医学因果。固定语义未优于阴性控制时停止追分。没有修改原源码或实验1–19，未commit/push。


### 实验二十最终科学复核补充


验收和机制结论

正式调度从 2026-10-16T06:59:25.537118+00:00 至 2026-10-16T07:59:34.002424+00:00，耗时 60.141 分钟（包含历史复刻门槛、一次旧表精度修复后的精确断点恢复）。GPU0/GPU1各30任务、300折；默认每卡4worker、每worker2线程。资源监控曾因容器内存余量短暂下调后续补位上限，余量恢复后自动恢复。峰值worker总RSS为31.394GiB，GPU0/1峰值分别为4883/5229MiB。

工程验收：20/20配置、60/60任务、600/600逐折指标、600/600独立诊断、1080/1080反事实记录，其中1020条有效评估、60条Local的GRAPH_NUMERIC为NA。原始6个核心文件及实验1–19的710个文件哈希未变，根handoff未变；临时训练状态目录已删除，无残留训练进程，未commit或push。完整更新的handoff位于本实验目录。

历史parity：T00对实验19 C0及实验13 F0的逐折六指标、epoch、可用初始化/RNG记录匹配；T01对历史P3的五个已有指标及原日志epoch匹配，历史未保存的Prob-AUC为本轮重新计算而非伪造历史值；T02/T03对实验9 E9/E3六指标和epoch逐折匹配。P3旧CSV将ACC/AUC截到四位，已使用原日志完整SEN及已知类计数恢复唯一整数混淆计数，最终精度仍为1e-12。预检还通过四路径初值/预测/三步Adam严格相等、20路径零残差等价、完整状态三步恢复和held-out标签置乱不影响训练更新。

**方向一：本轮固定疾病概念局部对齐未获验证。** T05−P0的ACC/BA/Prob-AUC分别为-0.03831/-0.04795/-0.07046pp。T05相对中性T06的ACC为+0.11407pp，但Prob-AUC为-0.15610pp；相对错解剖T07和匿名T09的Prob-AUC分别为-0.28694/-0.17883pp。真实文本在部分离散指标上优于随机/匿名控制，但不能据此声称稳定增加疾病诊断信息。

**方向二：医学概念建图未获验证，数值传播仍是替代解释。** T11−T10数值图的ACC/BA/Prob-AUC为-0.03744/-0.05569/-0.04082pp。T11−中性图T12虽然ACC为+0.30608pp，Prob-AUC却为-0.17444pp。数值图T10相对P0的ACC/BA/Prob-AUC为+0.15282/+0.11610/+0.20813pp；这是内部描述性线索，不能排除新增数值输入/容量贡献。Graph与Local参数不同，单独T11−T05的改善不构成语义图归因。

**方向三：未显示对成熟E9的独立互补。** T15−E9的ACC/BA/Prob-AUC为-0.03788/+0.02046/-0.34552pp。T15的BA均值最高（91.71198%），但相对E9的SEN增加0.82724pp、SPE下降0.78631pp，不能包装成整体分类能力提升。T18−E9的ACC/BA/Prob-AUC为-0.22989/-0.24050/-0.41068pp；T18−数值图T17的Prob-AUC为-0.31435pp，三个seed均为负。最高ACC由T02(E9)与T14(数值Local)并列，最高Prob-AUC为T02；真实语义组合没有超过成熟基线。

核心组残差与梯度复核：

|组|alpha有符号均值|mean(abs(alpha))|相对残差均值%|全折最大残差%|非零响应比例|
|---|---:|---:|---:|---:|---:|
|T05|-0.000437|0.170202|1.63283|5.78997|1.000|
|T11|-0.032607|0.192689|1.78939|7.30128|1.000|
|T15|0.044260|0.144420|1.39583|5.63583|1.000|
|T18|-0.034314|0.204206|1.93737|5.85021|1.000|

全部新组最大残差低于10%，非CORR三个位置的delta严格为0；核心组所有测试样本的残差均非零。alpha、编码器、edge ID、readout及图组graph_proj均有有限非零采样梯度，冻结prior不反传。不能用跨折有符号alpha均值接近0判断分支塌缩。局部/图组分别有165/246个edge ID可进入候选读出或一跳传播；其余嵌入无梯度符合结构定义。

REAL四域归一化注意力熵约0.9616–0.9717，权重较分散。这些权重由固定prior及患者FC gate直接确定，Graph/E9不会额外训练该权重；多个REAL组一致属预期。图组消息前后特征相对变化约0.45–0.51、患者邻接方差非零，证明传播实际发生，不能据此证明医学语义正确。edge ID对节点特征的影响也非零，保留额外容量解释。

核心反事实：下表为30折等权均值，翻转单位为%，概率绝对变化使用0–1概率单位；不是30个独立外部样本。

|组|换中性文本翻转%|中性文本概率变化|打乱患者FC翻转%|关闭分支翻转%|仅图改数值边权的概率变化|
|---|---:|---:|---:|---:|---:|
|T05|0.000000|0.00014145|0.114943|0.191571|NA|
|T11|0.000000|0.00027012|0.344828|0.344828|0.00001193|
|T15|0.076628|0.00038705|0.344828|0.268199|NA|
|T18|0.076628|0.00027221|0.152821|0.229450|0.00001036|

T05在中性、同Gram随机、错误掩码、匿名和prior归零干预下类别预测均无翻转，但概率存在微小变化；T11换中性文本也无类别翻转。E9组合组对文本有少量类别敏感性，但不足以结合训练对照证明正确医学内容必要。患者FC打乱及分支关闭通常引起更大的概率变化。T11/T18仅把概念图边权换为数值图时零类别翻转，平均概率变化约1e-5；这比关闭分支或打乱患者数值小得多。这里的解释是基于已完成对照的推断，不是因果病理结论。

最终判断：三方向的疾病语义假设均未通过本轮严格内部验真；不存在支持MICCAI2027第一疾病语义创新点的充分证据。保留P0/E9基线，停止本组静态文本及组合的自动追分。唯一后续建议是在训练折内预处理、独立验证集选模及独立站点评估中，冻结检验已有数值建图线索，不据这套测试折改词或继续选超参。

已核实的创新边界参照：[SPromptGL原文](https://papers.miccai.org/miccai-2025/0861-Paper5129.html)、[MGDR原文](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)、[BrainPrompt原文](https://arxiv.org/abs/2504.16096)。本轮没有患者症状量表监督；注意力及静态Top连接不是症状真值、病灶或因果脑区。

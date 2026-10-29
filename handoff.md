

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


## 实验二十一：三线验证与有界自适应疾病证据对齐

时间 2026-10-16T09:14:21.680283+00:00；状态 **BLOCKED**。目录 `/data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment`。根HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`，GitHub参照 `f11bf7966fcf5756c96f0b074e162827056d8f3d`；代码/模型/数据/文本/先验SHA见manifest。

A：原实验20 checkpoint已清理，仅复核聚合反事实非加性响应和outer-train先验排序；不能精确分解历史T11/T18向量。B：NOT_JOINABLE，验证对应0/871，CLINICAL_ALIGNMENT_NOT_ESTABLISHED；未将临床字段或缺失模式用于训练。

C：20固定组，U00 P0/U01 E9；U02–05严格复现T05/T11/T10/T04；U06–13检验有界患者FC修正、数值/中性/错ROI/同Gram及患者线图；U14–19比较E9局部/图的真实/数值/中性。eta=0.35sigmoid，初值.1；温度.5、train KL=.02、图profile .7、零alpha且CORR残差≤10%。无自由Key/BCE证据头，原主干与E9不变。

工程 0/60任务、0/600fold、0/600诊断、0/1200反事实（local图条件120 NA）；parity=待核验。


科学门槛：{"engineering": "PENDING", "medical_semantics": "PENDING", "adaptation": "PENDING", "graph_semantics": "PENDING", "e9_synergy": "PENDING"}。不把分支敏感性或最高ACC等同正确医学语义。三seed共享fold，val=test/test ACC选模与CV前全局RFE/标准化存在乐观偏差；不是独立无泄漏泛化。
工程阻塞：AssertionError()；未完成部分不做科学结论。
运行墙钟 未完成秒，两卡各30任务300fold，资源 {}。完整A/B/C、机制/反事实、三seed差值与W/T/L见 `/data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/results/report.md`。
后续唯一问题：在可信患者身份链及训练内选模、独立站点评估中，医学证据能否提供超过纯数值FC的可迁移增量。原源码与实验1–20未改；本条保留此前handoff全部字节，未commit/push。


## 实验二十一：三线验证与有界自适应疾病证据对齐

时间 2026-10-16T11:14:19.341517+00:00；状态 **COMPLETE**。目录 `/data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment`。根HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`，GitHub参照 `f11bf7966fcf5756c96f0b074e162827056d8f3d`；代码/模型/数据/文本/先验SHA见manifest。

A：原实验20 checkpoint已清理，仅复核聚合反事实非加性响应和outer-train先验排序；不能精确分解历史T11/T18向量。B：NOT_JOINABLE，验证对应0/871，CLINICAL_ALIGNMENT_NOT_ESTABLISHED；未将临床字段或缺失模式用于训练。

C：20固定组，U00 P0/U01 E9；U02–05严格复现T05/T11/T10/T04；U06–13检验有界患者FC修正、数值/中性/错ROI/同Gram及患者线图；U14–19比较E9局部/图的真实/数值/中性。eta=0.35sigmoid，初值.1；温度.5、train KL=.02、图profile .7、零alpha且CORR残差≤10%。无自由Key/BCE证据头，原主干与E9不变。

工程 60/60任务、600/600fold、600/600诊断、1200/1200反事实（local图条件120 NA）；parity={'status': 'PASS', 'folds': 180, 'groups': ['U00', 'U01', 'U02', 'U03', 'U04', 'U05'], 'metrics': 'All 6 metrics + best/executed epochs + initial trunk SHA + entry RNG SHA; 60 P0/E9 + 120 static folds', 'utc': '2026-10-16T10:19:59.920890+00:00'}。

六指标ACC/BA/Hard-AUC/ProbAUC/SEN-ASD/SPE-TD，%，三seed mean±population SD：

- U00: 91.69105±0.05418 / 91.53935±0.08408 / 91.53935±0.08408 / 93.33520±0.41541 / 89.66057±0.64562 / 93.41813±0.51023
- U01: 91.88262±0.19536 / 91.69151±0.19290 / 91.69151±0.19290 / 93.82272±0.23719 / 89.24797±0.11931 / 94.13506±0.27073
- U02: 91.65273±0.10837 / 91.49140±0.11760 / 91.49140±0.11760 / 93.26474±0.24911 / 89.41667±0.46856 / 93.56614±0.34557
- U03: 91.80643±0.14382 / 91.59976±0.14871 / 91.59976±0.14871 / 93.50250±0.20284 / 88.84553±0.34209 / 94.35399±0.20499
- U04: 91.84387±0.18823 / 91.65545±0.16364 / 91.65545±0.16364 / 93.54332±0.14766 / 89.17276±0.11952 / 94.13814±0.43719
- U05: 91.53779±0.10837 / 91.36332±0.12316 / 91.36332±0.12316 / 93.31927±0.13938 / 89.16667±0.46856 / 93.55998±0.35123
- U06: 91.76768±0.05418 / 91.57483±0.06226 / 91.57483±0.06226 / 93.52428±0.15638 / 89.16260±0.23575 / 93.98705±0.17750
- U07: 91.46160±0.14348 / 91.32311±0.15205 / 91.32311±0.15205 / 93.38681±0.15954 / 89.50407±0.41816 / 93.14215±0.30308
- U08: 91.57611±0.10837 / 91.38605±0.10802 / 91.38605±0.10802 / 93.20538±0.16567 / 88.92073±0.22853 / 93.85137±0.20831
- U09: 91.65273±0.05418 / 91.45673±0.06007 / 91.45673±0.06007 / 93.45306±0.22030 / 88.91870±0.12219 / 93.99476±0.00436
- U10: 91.49991±0.18770 / 91.29514±0.16590 / 91.29514±0.16590 / 93.23405±0.10038 / 88.59553±0.41080 / 93.99476±0.60291
- U11: 91.72936±0.09385 / 91.52993±0.09301 / 91.52993±0.09301 / 93.28563±0.31276 / 88.92480±0.30692 / 94.13506±0.26830
- U12: 91.76768±0.19536 / 91.57857±0.21555 / 91.57857±0.21555 / 93.33397±0.34859 / 89.09146±0.69848 / 94.06568±0.44270
- U13: 91.57611±0.21674 / 91.37908±0.24588 / 91.37908±0.24588 / 93.37350±0.32574 / 88.83740±0.69563 / 93.92075±0.20388
- U14: 91.88262±0.14336 / 91.68664±0.13672 / 91.68664±0.13672 / 93.42229±0.21599 / 89.24593±0.11785 / 94.12735±0.26868
- U15: 91.57611±0.14336 / 91.40849±0.14849 / 91.40849±0.14849 / 93.49144±0.21586 / 89.17683±0.50151 / 93.64015±0.36563
- U16: 91.72936±0.18770 / 91.55458±0.16698 / 91.55458±0.16698 / 93.48439±0.18033 / 89.33333±0.19666 / 93.77582±0.53061
- U17: 91.49948±0.09385 / 91.31622±0.06456 / 91.31622±0.06456 / 93.50854±0.06691 / 89.00000±0.41983 / 93.63244±0.53569
- U18: 91.84474±0.18717 / 91.68249±0.18300 / 91.68249±0.18300 / 93.37343±0.07536 / 89.65854±0.30638 / 93.70644±0.36678
- U19: 91.80556±0.05388 / 91.60604±0.06414 / 91.60604±0.06414 / 93.54551±0.15733 / 89.00610±0.41735 / 94.20598±0.34934
- U06−U02: acc +0.11494pp / ba +0.08342pp / probability_auc +0.25954pp
- U06−U07: acc +0.30608pp / ba +0.25172pp / probability_auc +0.13747pp
- U06−U08: acc +0.19157pp / ba +0.18877pp / probability_auc +0.31890pp
- U06−U09: acc +0.11494pp / ba +0.11810pp / probability_auc +0.07122pp
- U06−U10: acc +0.26776pp / ba +0.27968pp / probability_auc +0.29023pp
- U06−U00: acc +0.07663pp / ba +0.03548pp / probability_auc +0.18908pp
- U11−U03: acc -0.07706pp / ba -0.06983pp / probability_auc -0.21688pp
- U11−U04: acc -0.11451pp / ba -0.12553pp / probability_auc -0.25770pp
- U11−U12: acc -0.03831pp / ba -0.04864pp / probability_auc -0.04834pp
- U11−U13: acc +0.15326pp / ba +0.15085pp / probability_auc -0.08788pp
- U11−U00: acc +0.03831pp / ba -0.00942pp / probability_auc -0.04957pp
- U14−U01: acc -0.00000pp / ba -0.00487pp / probability_auc -0.40043pp
- U14−U15: acc +0.30651pp / ba +0.27815pp / probability_auc -0.06915pp
- U14−U18: acc +0.03788pp / ba +0.00415pp / probability_auc +0.04886pp
- U16−U01: acc -0.15326pp / ba -0.13693pp / probability_auc -0.33833pp
- U16−U14: acc -0.15326pp / ba -0.13206pp / probability_auc +0.06210pp
- U16−U17: acc +0.22989pp / ba +0.23836pp / probability_auc -0.02415pp
- U16−U19: acc -0.07619pp / ba -0.05146pp / probability_auc -0.06112pp

科学门槛：{"engineering": "ENGINEERING_PASS", "medical_semantics": "MEDICAL_SEMANTICS_FAIL", "adaptation": "ADAPTATION_FAIL", "graph_semantics": "GRAPH_SEMANTICS_FAIL", "e9_synergy": "E9_SYNERGY_FAIL"}。不把分支敏感性或最高ACC等同正确医学语义。三seed共享fold，val=test/test ACC选模与CV前全局RFE/标准化存在乐观偏差；不是独立无泄漏泛化。
运行墙钟 6740.07884144783秒，两卡各30任务300fold，资源 {'worker_rss_gib': 27.784751892089844, 'gpu0_mib': 38173, 'gpu1_mib': 36393}。完整A/B/C、机制/反事实、三seed差值与W/T/L见 `/data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/results/report.md`。
后续唯一问题：在可信患者身份链及训练内选模、独立站点评估中，医学证据能否提供超过纯数值FC的可迁移增量。原源码与实验1–20未改；本条保留此前handoff全部字节，未commit/push。


<!-- SABER21_FINAL_REVIEW_32ec5228dff4c5a3f496ae85844b8639ffe62029e83bf2e39c5f8e4c9c9d20c6 -->

实验二十一最终复核：ENGINEERING_PASS；MEDICAL_SEMANTICS/ADAPTATION/GRAPH_SEMANTICS/E9_SYNERGY均FAIL。ACC并列最高U01/E9与U14为91.88262%；BA/Prob-AUC最高均为E9（91.69151%/93.82272%）。U14相对E9 ACC持平、Prob-AUC−0.40043pp；U16相对E9 ACC/BA/Prob-AUC−0.15326/−0.13693/−0.33833pp。U06局部有有限平均优势，但没有跨seed通过所有数值/中性/同Gram/错ROI及概率AUC门槛；其相对P0的SEN-ASD还下降0.49797pp。四核心组ADAPT_OFF均零类别翻转；U11/U16去图语义因子也零翻转。分支平均相对残差约1.15%–1.82%，并非整体失活，仍没有稳定的正确医学语义必要性。新旧static/adaptive同时改变attention温度/公式、KL、图权重，不能独立归因eta。B包0/871身份可核验，A包仅非加性聚合复核，均未伪造临床监督或旧checkpoint向量分解。完整报告已补充所有对照的科学止损及解释边界；不具备本轮MICCAI2027第一创新点证据。

60/60任务、600/600折、600诊断、1200反事实（120 NA）、GPU各300折、180历史parity，原位resume已验证跳过完成任务；根核心和785历史文件保持原SHA；临时训练状态/权重及本实验worker已清理；未commit/push。后续唯一问题：在可追溯患者身份、训练内选模与独立站点中，真实患者医学证据是否能提供超过数值FC的可迁移增量。


## 实验二十二：概念条件化有符号功能连接证据

<!-- SABER22:a7ddf0b8b025a5f68f2c40470f1be15669ec705732120e55edb77a13ed048982 -->

时间2026-10-16T15:30:31.690952+00:00，状态**BLOCKED**；目录 `/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity`。HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`，公开审计 `c2f07d274a281861a79dd35b979449b6b25e992e`；六核心SHA一致，全部源代码/文本/数据/先验/CLIP签名见manifest。

历史动机：19自由Key重参数化、20固定先验依赖弱、21绝对值患者修正缺乏signed/domain差别且图profile丢幅度、临床身份0/871不可核验。22仅outer-train训练标签构建FC有符号弱疾病关联，冻结文本、同容量数值/中性/同Gram/错ROI、unsigned/shared、幅度图、E9正交残差。不是确证临床症状/病理。

完成0/54任务、0/540新fold、0/540诊断、0/2340反事实（完整时270 NA）；额外180历史fold没有重训。两GPU固定各27任务/270fold。耗时未完成秒；资源{'worker_rss_gib': 24.84347152709961, 'gpu0_mib': 7915, 'gpu1_mib': 8479}。

- M00: P0 + disease-concept signed local; prior=REAL, graph=False, variant=SIGNED, complement=False
- M01: P0 + numerical signed local; prior=NUMERIC, graph=False, variant=SIGNED, complement=False
- M02: P0 + neutral signed local; prior=NEUTRAL, graph=False, variant=SIGNED, complement=False
- M03: P0 + wrong-ROI signed local; prior=WRONG_ANATOMY, graph=False, variant=SIGNED, complement=False
- M04: P0 + isometric random signed local; prior=ISOMETRIC, graph=False, variant=SIGNED, complement=False
- M05: P0 + real text; omit FC direction; prior=REAL, graph=False, variant=UNSIGNED, complement=False
- M06: P0 + real text; one shared domain adaptation; prior=REAL, graph=False, variant=SHARED, complement=False
- M07: P0 + true prior + signed evidence + magnitude-preserving graph; prior=REAL, graph=True, variant=SIGNED, complement=False
- M08: P0 + numerical signed graph, no semantic factor; prior=NUMERIC, graph=True, variant=SIGNED, complement=False
- M09: P0 + neutral signed magnitude graph; prior=NEUTRAL, graph=True, variant=SIGNED, complement=False
- M10: E9 + true signed local; prior=REAL, graph=False, variant=SIGNED, complement=False
- M11: E9 + numerical signed local; prior=NUMERIC, graph=False, variant=SIGNED, complement=False
- M12: E9 + neutral signed local; prior=NEUTRAL, graph=False, variant=SIGNED, complement=False
- M13: E9 + true signed local with orthogonal CORR residual; prior=REAL, graph=False, variant=SIGNED, complement=True
- M14: E9 + true signed magnitude graph; prior=REAL, graph=True, variant=SIGNED, complement=False
- M15: E9 + numerical signed graph; prior=NUMERIC, graph=True, variant=SIGNED, complement=False
- M16: E9 + neutral signed graph; prior=NEUTRAL, graph=True, variant=SIGNED, complement=False
- M17: E9 + true signed graph with orthogonal CORR residual; prior=REAL, graph=True, variant=SIGNED, complement=True

工程预检：{"status": "PASS", "started_utc": "2026-10-16T15:27:28.831330+00:00", "checks": {"synthetic_30_cases_plus_complement": "PASS: parameters19437/20525, exact zero-alpha, candidate weights, train-KL, 8 coefficient gradients, 2-step all gradients,10% bound", "signed_and_concept_specificity": "PASS: opposite d gives different SIGNED and identical UNSIGNED attention; swapped train labels reverses contrast; SHARED coefficient algebra exact", "deterministic_sparse_dense": {"status": "PASS", "max_value_error": 7.152557373046875e-07, "max_gradient_error": 6.984919309616089e-10, "tolerance": 1e-05, "deterministic": true}, "graph_absolute_magnitude": "PASS: equal normalized profiles but redistributed domain mass changes message; cosine-only/semantic-off remain identical; self-loop+nonuniform neighbours prevent row-normalization cancellation", "all18_zero_alpha": "PASS: 18 original-trunk initial parameters/RNG/prediction exact; graph/local same-family learned init hashes equal", "three_step_atomic_recovery": "PASS: full model/buffers/original2Adam+E9/branch groups + Python/NumPy/CPU/CUDA RNG exact", "heldout_label_invariance": "PASS: modified heldout labels passed into EvalHelper -> identical signed buffers,graph,full first train update/optimizer/RNG", "counterfactual_restoration": "PASS: all13 actions restore every registered buffer,weights,rawFC,RNG and bitwise REAL predictions; no cached adjacency exists in CodeA"}, "baseline_parity": {"P0": "PASS: exact Exp9 initial original params/Adam/RNG/prediction + first original gradients/Adam update; later branch alpha intentionally diverges", "E9": "PASS: exact Exp9 initial original params/Adam/RNG/prediction + first original gradients/Adam update; later branch alpha intentionally diverges"}, "completed_utc": "2026-10-16T15:28:48.208569+00:00", "formal_fold_training": false, "precision": "float32; exact parity/recovery; dense tolerance1e-5; no pilot scores"}

科学门槛：{"medical_text": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}, "correct_anatomy": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}, "signed_fc": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}, "domain_specific": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}, "graph_magnitude": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}, "E9_complement": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}, "E9_synergy": {"status": "INCONCLUSIVE", "reason": "正式540fold未完整"}}

五条主要认识：1. signed contrast来源仅训练诊断标签，不能提升为临床真值；2. 正确文本必须超过数值/中性/随机/错解剖，最高ACC不够；3. 图幅度需固定checkpoint消融与数值图共同审查，非Shapley；4. E9正交潜在方向不保证独立增量；5. val=test、全局RFE/标准化、transductive和共享fold限制泛化及显著性。详细实测量与反事实见report。

工程阻塞：RuntimeError('Task failed; formal run stopped: M07_seed1; best_acc=0.954545454545 wait=201\nTraceback (most recent call last):\n  File "/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity/run.py", line 1292, in <module>\n    if __name__==\'__main__\':main()\n                            ^^^^^^\n  File "/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity/run.py", line 1277, in main\n    i=sys.argv.index(\'--worker\');worker(sys.argv[i+1],int(sys.argv[i+2]));return\n                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity/run.py", line 405, in worker\n    diagnostic=diagnostics(a,config,seed,fold,out,tracker);diagnostic[\'task_signature\']=sig\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity/run.py", line 824, in diagnostics\n    strength_histogram=json.dumps(torch.histc(strength[te].float(),bins=10,min=0,max=.2).cpu().tolist()),\n                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: _histc_cuda with floating point input does not have a deterministic implementation, but you set \'torch.use_deterministic_algorithms(True)\'. You can turn off determinism just for this operation, or you can use the \'warn_only=True\' option, if that\'s acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.\n')；没有完成的正式fold不作结论。

后续唯一问题：训练内选模与独立站点下真实医学先验能否超过同容量数值/中性；不自动调词、调参或建立实验23。root CORE与实验1–21只读，根handoff完整原前缀逐字保留；未commit/push。


## 实验二十二：概念条件化有符号功能连接证据

<!-- SABER22:bed97788e1227db7af8ea47ba5ade5ddc812224eb3c1f998cb0f6e83ff7d1f96 -->

时间2026-10-16T16:39:45.585540+00:00，状态**COMPLETE**；目录 `/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity`。HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`，公开审计 `c2f07d274a281861a79dd35b979449b6b25e992e`；六核心SHA一致，全部源代码/文本/数据/先验/CLIP签名见manifest。

历史动机：19自由Key重参数化、20固定先验依赖弱、21绝对值患者修正缺乏signed/domain差别且图profile丢幅度、临床身份0/871不可核验。22仅outer-train训练标签构建FC有符号弱疾病关联，冻结文本、同容量数值/中性/同Gram/错ROI、unsigned/shared、幅度图、E9正交残差。不是确证临床症状/病理。

完成54/54任务、540/540新fold、540/540诊断、2340/2340反事实（完整时270 NA）；额外180历史fold没有重训。两GPU固定各27任务/270fold。耗时4188.169287919998秒；资源{'worker_rss_gib': 26.17574691772461, 'gpu0_mib': 8801, 'gpu1_mib': 8825}。

- M00: P0 + disease-concept signed local; prior=REAL, graph=False, variant=SIGNED, complement=False
- M01: P0 + numerical signed local; prior=NUMERIC, graph=False, variant=SIGNED, complement=False
- M02: P0 + neutral signed local; prior=NEUTRAL, graph=False, variant=SIGNED, complement=False
- M03: P0 + wrong-ROI signed local; prior=WRONG_ANATOMY, graph=False, variant=SIGNED, complement=False
- M04: P0 + isometric random signed local; prior=ISOMETRIC, graph=False, variant=SIGNED, complement=False
- M05: P0 + real text; omit FC direction; prior=REAL, graph=False, variant=UNSIGNED, complement=False
- M06: P0 + real text; one shared domain adaptation; prior=REAL, graph=False, variant=SHARED, complement=False
- M07: P0 + true prior + signed evidence + magnitude-preserving graph; prior=REAL, graph=True, variant=SIGNED, complement=False
- M08: P0 + numerical signed graph, no semantic factor; prior=NUMERIC, graph=True, variant=SIGNED, complement=False
- M09: P0 + neutral signed magnitude graph; prior=NEUTRAL, graph=True, variant=SIGNED, complement=False
- M10: E9 + true signed local; prior=REAL, graph=False, variant=SIGNED, complement=False
- M11: E9 + numerical signed local; prior=NUMERIC, graph=False, variant=SIGNED, complement=False
- M12: E9 + neutral signed local; prior=NEUTRAL, graph=False, variant=SIGNED, complement=False
- M13: E9 + true signed local with orthogonal CORR residual; prior=REAL, graph=False, variant=SIGNED, complement=True
- M14: E9 + true signed magnitude graph; prior=REAL, graph=True, variant=SIGNED, complement=False
- M15: E9 + numerical signed graph; prior=NUMERIC, graph=True, variant=SIGNED, complement=False
- M16: E9 + neutral signed graph; prior=NEUTRAL, graph=True, variant=SIGNED, complement=False
- M17: E9 + true signed graph with orthogonal CORR residual; prior=REAL, graph=True, variant=SIGNED, complement=True

全部组ACC/BA/HardAUC/ProbAUC/SEN-ASD/SPE-TD/F1，三seed mean±population SD，%：

- M00: 91.57654±0.05449 / 91.37381±0.04225 / 91.37381±0.04225 / 93.36373±0.15602 / 88.67886±0.57920 / 94.06876±0.56002 / 90.69559±0.06914
- M01: 91.53779±0.05418 / 91.36486±0.04714 / 91.36486±0.04714 / 93.55388±0.07843 / 89.16667±0.30748 / 93.56306±0.30528 / 90.72767±0.06591
- M02: 91.69061±0.21735 / 91.55025±0.20548 / 91.55025±0.20548 / 93.47664±0.26124 / 89.59756±0.00498 / 93.50293±0.40665 / 90.91096±0.20879
- M03: 91.61486±0.16194 / 91.43842±0.21773 / 91.43842±0.21773 / 93.34955±0.22741 / 89.24593±1.03586 / 93.63090±0.61404 / 90.81798±0.27429
- M04: 91.49948±0.16255 / 91.31274±0.19950 / 91.31274±0.19950 / 93.08982±0.19422 / 88.84350±0.70423 / 93.78199±0.30528 / 90.63937±0.24291
- M05: 91.53779±0.14336 / 91.37310±0.11906 / 91.37310±0.11906 / 93.45577±0.10530 / 89.25407±0.30096 / 93.49214±0.53898 / 90.74046±0.10490
- M06: 91.61486±0.09332 / 91.47176±0.06094 / 91.47176±0.06094 / 93.54342±0.12953 / 89.66260±0.83668 / 93.28091±0.86257 / 90.85275±0.07392
- M07: 91.84431±0.28155 / 91.66717±0.28101 / 91.66717±0.28101 / 93.31136±0.08871 / 89.41667±0.23858 / 93.91767±0.36831 / 91.06105±0.27963
- M08: 91.61442±0.16255 / 91.42151±0.18102 / 91.42151±0.18102 / 93.51538±0.24557 / 88.92073±0.64663 / 93.92229±0.44495 / 90.76307±0.21750
- M09: 91.72936±0.24730 / 91.54192±0.26744 / 91.54192±0.26744 / 93.32183±0.33283 / 89.16463±0.58207 / 93.91921±0.10577 / 90.90836±0.32622
- M10: 91.80643±0.14348 / 91.62364±0.15191 / 91.62364±0.15191 / 93.53705±0.11534 / 89.25407±0.30587 / 93.99322±0.00218 / 91.00578±0.17695
- M11: 91.46160±0.05449 / 91.24573±0.08127 / 91.24573±0.08127 / 93.34067±0.26136 / 88.42886±0.83868 / 94.06260±0.71191 / 90.56462±0.13299
- M12: 91.72980±0.16194 / 91.56513±0.16752 / 91.56513±0.16752 / 93.59319±0.27815 / 89.42073±0.30638 / 93.70953±0.20278 / 90.93950±0.18384
- M13: 91.49948±0.09385 / 91.31103±0.08337 / 91.31103±0.08337 / 93.48596±0.07868 / 88.91870±0.22289 / 93.70336±0.36890 / 90.65862±0.05928
- M14: 91.80599±0.10837 / 91.63767±0.08233 / 91.63767±0.08233 / 93.40937±0.20668 / 89.49797±0.30909 / 93.77737±0.46391 / 91.03189±0.06771
- M15: 91.72980±0.16286 / 91.54108±0.15226 / 91.54108±0.15226 / 93.62351±0.12813 / 89.08740±0.00287 / 93.99476±0.30308 / 90.89587±0.16528
- M16: 91.80599±0.05418 / 91.65993±0.08970 / 91.65993±0.08970 / 93.38649±0.17983 / 89.82927±0.53914 / 93.49060±0.36256 / 91.07049±0.11204
- M17: 91.69148±0.14348 / 91.50060±0.17033 / 91.50060±0.17033 / 93.58221±0.14164 / 88.93089±0.51075 / 94.07031±0.19624 / 90.84589±0.19517
- M00−M01: acc +0.03875pp / ba +0.00895pp / probability_auc -0.19015pp
- M00−M02: acc -0.11407pp / ba -0.17643pp / probability_auc -0.11291pp
- M00−M03: acc -0.03831pp / ba -0.06460pp / probability_auc +0.01418pp
- M00−M04: acc +0.07706pp / ba +0.06107pp / probability_auc +0.27391pp
- M00−M05: acc +0.03875pp / ba +0.00071pp / probability_auc -0.09204pp
- M00−M06: acc -0.03831pp / ba -0.09794pp / probability_auc -0.17969pp
- M00−P0: acc -0.11451pp / ba -0.16554pp / probability_auc +0.02853pp
- M00−old_real_local: acc -0.19114pp / ba -0.20101pp / probability_auc -0.16055pp
- M07−M08: acc +0.22989pp / ba +0.24565pp / probability_auc -0.20402pp
- M07−M09: acc +0.11494pp / ba +0.12525pp / probability_auc -0.01047pp
- M07−M00: acc +0.26776pp / ba +0.29336pp / probability_auc -0.05237pp
- M07−old_real_graph: acc +0.11494pp / ba +0.13724pp / probability_auc +0.02573pp
- M10−M11: acc +0.34483pp / ba +0.37791pp / probability_auc +0.19638pp
- M10−M12: acc +0.07663pp / ba +0.05851pp / probability_auc -0.05614pp
- M10−E9: acc -0.07619pp / ba -0.06787pp / probability_auc -0.28567pp
- M10−old_e9_local: acc -0.07619pp / ba -0.06300pp / probability_auc +0.11476pp
- M13−M10: acc -0.30695pp / ba -0.31261pp / probability_auc -0.05109pp
- M13−E9: acc -0.38314pp / ba -0.38048pp / probability_auc -0.33676pp
- M14−M15: acc +0.07619pp / ba +0.09659pp / probability_auc -0.21414pp
- M14−M16: acc +0.00000pp / ba -0.02226pp / probability_auc +0.02287pp
- M14−E9: acc -0.07663pp / ba -0.05385pp / probability_auc -0.41335pp
- M14−old_e9_graph: acc +0.07663pp / ba +0.08309pp / probability_auc -0.07502pp
- M17−M14: acc -0.11451pp / ba -0.13707pp / probability_auc +0.17284pp
- M17−E9: acc -0.19114pp / ba -0.19091pp / probability_auc -0.24051pp

工程预检：{"status": "PASS", "started_utc": "2026-10-16T15:32:19.566716+00:00", "checks": {"synthetic_30_cases_plus_complement": "PASS: parameters19437/20525, exact zero-alpha, candidate weights, train-KL, 8 coefficient gradients, 2-step all gradients,10% bound", "signed_and_concept_specificity": "PASS: opposite d gives different SIGNED and identical UNSIGNED attention; swapped train labels reverses contrast; SHARED coefficient algebra exact", "deterministic_sparse_dense": {"status": "PASS", "max_value_error": 7.152557373046875e-07, "max_gradient_error": 6.984919309616089e-10, "tolerance": 1e-05, "deterministic": true}, "graph_absolute_magnitude": "PASS: equal normalized profiles but redistributed domain mass changes message; cosine-only/semantic-off remain identical; self-loop+nonuniform neighbours prevent row-normalization cancellation", "all18_zero_alpha": "PASS: 18 original-trunk initial parameters/RNG/prediction exact; graph/local same-family learned init hashes equal", "three_step_atomic_recovery": "PASS: full model/buffers/original2Adam+E9/branch groups + Python/NumPy/CPU/CUDA RNG exact", "full_fold_diagnostic_path": "PASS: complete graph diagnostic incl deterministic CPU histograms,branch/graph/strength interventions and exact REAL restoration", "heldout_label_invariance": "PASS: modified heldout labels passed into EvalHelper -> identical signed buffers,graph,full first train update/optimizer/RNG", "counterfactual_restoration": "PASS: all13 actions restore every registered buffer,weights,rawFC,RNG and bitwise REAL predictions; no cached adjacency exists in CodeA"}, "baseline_parity": {"P0": "PASS: exact Exp9 initial original params/Adam/RNG/prediction + first original gradients/Adam update; later branch alpha intentionally diverges", "E9": "PASS: exact Exp9 initial original params/Adam/RNG/prediction + first original gradients/Adam update; later branch alpha intentionally diverges"}, "completed_utc": "2026-10-16T15:33:39.040300+00:00", "formal_fold_training": false, "precision": "float32; exact parity/recovery; dense tolerance1e-5; no pilot scores"}

科学门槛：{"medical_text": {"status": "FAIL", "pairs": [["M00", "M01"], ["M00", "M02"], ["M00", "M04"], ["M00", "P0"]], "same_checkpoint_sensitivity": true, "criterion": "每个对照ACC/BA/ProbAUC均须三个seed差值全正；仅内部描述标准"}, "correct_anatomy": {"status": "FAIL", "pairs": [["M00", "M03"]], "same_checkpoint_sensitivity": true, "criterion": "每个对照ACC/BA/ProbAUC均须三个seed差值全正；仅内部描述标准"}, "signed_fc": {"status": "FAIL", "pairs": [["M00", "M05"]], "same_checkpoint_sensitivity": true, "criterion": "每个对照ACC/BA/ProbAUC均须三个seed差值全正；仅内部描述标准"}, "domain_specific": {"status": "FAIL", "pairs": [["M00", "M06"]], "same_checkpoint_sensitivity": true, "criterion": "每个对照ACC/BA/ProbAUC均须三个seed差值全正；仅内部描述标准"}, "E9_complement": {"status": "FAIL", "pairs": [["M13", "M10"], ["M13", "E9"]], "same_checkpoint_sensitivity": true, "criterion": "每个对照ACC/BA/ProbAUC均须三个seed差值全正；仅内部描述标准", "graph_pairs": [["M17", "M14"], ["M17", "E9"]], "graph_all_positive": false}, "E9_synergy": {"status": "FAIL", "local_all_controls": false, "graph_all_controls": false, "criterion": "plain正确语义组同时超过E9、数值及中性，三个seed三指标均正"}, "graph_magnitude": {"status": "FAIL", "trained_graph_controls": false, "same_checkpoint_magnitude_sensitive": true, "consistent_selected_checkpoint_benefit": false, "scope": "固定checkpoint消融不能代替无magnitude重训；不是Shapley或外部医学证据"}}

五条主要认识：1. signed contrast来源仅训练诊断标签，不能提升为临床真值；2. 正确文本必须超过数值/中性/随机/错解剖，最高ACC不够；3. 图幅度需固定checkpoint消融与数值图共同审查，非Shapley；4. E9正交潜在方向不保证独立增量；5. val=test、全局RFE/标准化、transductive和共享fold限制泛化及显著性。详细实测量与反事实见report。

后续唯一问题：训练内选模与独立站点下真实医学先验能否超过同容量数值/中性；不自动调词、调参或建立实验23。root CORE与实验1–21只读，根handoff完整原前缀逐字保留；未commit/push。


### 实验二十二最终实测复核（五项结论与工程修订）




<!-- SABER22-FINAL-REVIEW:91dedf200ccd94e9ee4971b7733afcf6c532afae961e3846f32fd5b34316f34b -->

工程 **PASS**：54/54逻辑任务、540/540新fold，GPU0/1各270fold；540诊断与2340反事实（2070有效、270 NA）全部完成。180历史参考只读导入，不记为新训练。P0/E9零alpha初始模型、原Adam、原参数首次梯度/更新精确等价，18组初始化与三步精确恢复通过。789个历史文件与六核心SHA保护，原始handoff前134834字节保持。

1. **当前正确医学文本没有提供稳定增量，保留E9。** 本轮最高ACC/BA为M07：91.84431±0.28155% / 91.66717±0.28101%；最高Prob-AUC为数值图M15：93.62351±0.12813%。三项均低于历史E9的91.88262±0.19536%、91.69151±0.19290%、93.82272±0.23719%。M00相对P0的ACC/BA为−0.11451/−0.16554pp，相对NEUTRAL的ACC/BA/ProbAUC为−0.11407/−0.17643/−0.11291pp，相对错ROI的ACC/BA也更低。M00相对同Gram随机的ProbAUC三个seed都正，但ACC/BA不一致，不能据单项包装医学有效性。

2. **数值图和容量解释仍然不能排除，真实图没有取得三指标共同优势。** M07−M08的ACC三个seed都正（+0.45977/+0.11494/+0.11494pp），平均+0.22989pp，但ProbAUC平均−0.20402pp；M07−M09的ACC为+0.80329/−0.11364/−0.34483pp，不稳定。M14与中性图M16的平均ACC完全相同，BA低0.02226pp。数值图M15拥有本轮最高ProbAUC。数据不支持“医学语义图必须保留”，同时也不能把数值图当已经超过E9的创新。graph比local多1088参数，M07−M00、M14−M10不是等容量比较，医学图结论应依靠M07−M08/M09和M14−M15/M16。

3. **有符号、分域和图幅度在数学上工作，但实际决策价值极弱。** 所有540fold新参数都观测到有限非零梯度；六核心组平均残差约0.51%–1.42%，全18组最大残差6.72075%，不是分支整体失活。ADAPT_OFF、SIGN_OFF、SHARED_DOMAIN在六组共540次有效checkpoint条件评估均没有任何类别翻转。均值|Δp|：ADAPT_OFF约2.37e−6至1.10e−5，SHARED_DOMAIN约3.16e−8至6.57e−7。三图核心组GRAPH_STRENGTH_OFF也均零翻转，|Δp|仅约1.97e−6至1.26e−5。代码响应与梯度连通不等于临床/分类收益；不能因设计修复了旧数学缺陷就宣称新机制有效。

4. **文本敏感性存在，但正确医学含义未被验证。** NEUTRAL替换平均|Δp|约3.82e−5至2.77e−4，多数核心组零类别翻转；M00、M14仅出现极少翻转。M07置换NEUTRAL/ISOMETRIC/WRONG_ANATOMY均零翻转，M10/M13/M17中性替换也均零翻转。全分支关闭及分支FC置乱的概率效应更大，说明分支主要读到了患者数值信息；这种非加性响应大小不是Shapley因果贡献率。图语义因子与幅度因子不是解释全部性能变化的证据。

5. **E9协同与正交互补均止损。** M10/M14相对E9的ACC为−0.07619/−0.07663pp、ProbAUC为−0.28567/−0.41335pp。M13−M10的ACC三个seed全负（−0.46108/−0.34483/−0.11494pp），平均−0.30695pp；M17−M14的BA三个seed全负，平均−0.13707pp。几何正交没有产生稳定诊断互补。medical_text、correct_anatomy、signed_fc、domain_specific、graph_magnitude、E9_complement、E9_synergy七门槛全部FAIL，不能作为MICCAI2027第一疾病语义创新点的已验证证据。

唯一后续问题：在**训练内选模与独立站点验证**下，冻结真实医学语义先验能否稳定超过同容量数值/中性对照；若没有，不再以微小测试ACC波动追加Prompt/超参搜索。本轮无新实验23，无自动commit/push。训练contrast仅来自outer-train ASD/TD关联，0/871患者临床身份可核验，不能称症状真值。历史test=val、全局监督RFE/标准化和全871 transductive的乐观偏差限制本结论的外部泛化。

**工程修订记录**

首批任务在逐折诊断的CUDA float histc处被确定性模式拒绝，未聚合任何正式fold；按规则停止。仅将两个直方图输入转移到CPU，补充完整逐折诊断preflight。三个权威模型/registry/builder文件、模型数学、损失、Adam、Prompt与超参完全未动。八个原子checkpoint只迁移签名，逐值核验model/optimizer/RNG/best-model/best-optimizer均不变，再从原状态精确继续。旧失败日志保留；没有失败attempt指标混入540fold。manifest.engineering_revisions保留前后代码SHA、阻塞原因、旧preflight和迁移证明。


## 实验二十三：增益校准的选择性患者关系对齐（GC-SRA） — COMPLETE — 2026-10-17T09:22:37.589621+00:00

实验目录：`/tmp/experiment23_gain_calibrated_selective_relational_alignment`（/data创建返回I/O错误，按用户指令在/tmp执行）。
本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；远端 `b00a66acb2a492d957e1241a0599a8551c61767c`。根六核心文件与远端相同；不修改实验1–22。
G0原始P0；G1统一；G2同预算常量；G3同预算FC不确定度；G4训练内OOF条件增益；G5集合内置乱增益；G6同容量特征差分；G7仅PHENO关系核错配。
增益为FC与FC+年龄/性别Logistic教师OOF对数损失差；Ridge(alpha=150)预测；不是临床因果增益。新增关系通道只更新GFR输出CORR256，FC锚保留；零alpha、差分Decoder、有界残差，无新损失/疾病Prompt。
动机：17/18表型共享与联合正确性未带来可靠增益；19中性优于真实且Key可重参数化；20/21固定语义图响应接近零；22η不敏感、正交残差未优于E9。
阶段状态：COMPLETE；完成24/24任务、240/240折。
快筛：mean_delta=-0.033275383，top20=0.022055858，Spearman=-0.093193745，gate SD=0.049090342；判停=False。
证据SHA `62daeaa780a89af0d142db9b6f7a7d6608c23d83928bd1c73723bf70f5fa5667`；核心/数据/参考代码SHA详见manifest。
风险：保留test=val选模、全队列既有标准化/RFE、871人transductive与22轮探索；meta OOF一阶教师训练池重叠，只作探索诊断。Exp11已有数值身份恢复链，与Exp21仅因未找到subject_IDs.txt判0/871的有界搜索结论相矛盾；本轮不恢复身份、不使用临床量表。
G0 acc=91.69105±0.05418%, ba=91.53935±0.08408%, hard_auc=91.53935±0.08408%, probability_auc=93.33520±0.41541%, sensitivity_asd=89.66057±0.64562%, specificity_td=93.41813±0.51023%, f1_asd=90.93289±0.12038%
G1 acc=83.54319±0.10745%, ba=83.49082±0.20551%, hard_auc=83.49082±0.20551%, probability_auc=88.24634±0.88564%, sensitivity_asd=82.79878±2.72398%, specificity_td=84.18286±2.37033%, f1_asd=82.30066±0.54297%
G2 acc=87.71378±0.33868%, ba=87.66148±0.26794%, hard_auc=87.66148±0.26794%, probability_auc=91.04256±0.35053%, sensitivity_asd=87.09756±0.61524%, specificity_td=88.22541±1.13198%, f1_asd=86.79885±0.22708%
G3 acc=86.87261±0.38022%, ba=86.69050±0.39426%, hard_auc=86.69050±0.39426%, probability_auc=90.32799±0.20928%, sensitivity_asd=84.38008±0.73671%, specificity_td=89.00093±0.44270%, f1_asd=85.58450±0.46214%
G4 acc=87.25488±0.52283%, ba=87.18469±0.50280%, hard_auc=87.18469±0.50280%, probability_auc=90.75143±0.25537%, sensitivity_asd=86.35366±1.41040%, specificity_td=88.01573±1.54518%, f1_asd=86.26987±0.53385%
G5 acc=87.06287±0.66613%, ba=87.08774±0.62249%, hard_auc=87.08774±0.62249%, probability_auc=90.42794±0.44081%, sensitivity_asd=87.35772±0.68411%, specificity_td=86.81776±1.28115%, f1_asd=86.18318±0.66114%
G6 acc=86.64359±0.05484%, ba=86.62204±0.05584%, hard_auc=86.62204±0.05584%, probability_auc=90.40863±0.37323%, sensitivity_asd=86.29065±0.49623%, specificity_td=86.95344±0.46463%, f1_asd=85.66850±0.09830%
G7 acc=87.67372±0.38936%, ba=87.55467±0.34445%, hard_auc=87.55467±0.34445%, probability_auc=91.32636±0.28379%, sensitivity_asd=86.09146±0.21199%, specificity_td=89.01788±0.85965%, f1_asd=86.60599±0.32901%
科学判据：{"contrasts": {"G4-G0": {"stable_ba_probability_auc": false, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}, "G4-G1": {"stable_ba_probability_auc": true, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}, "G4-G2": {"stable_ba_probability_auc": false, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}, "G4-G3": {"stable_ba_probability_auc": false, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}, "G4-G5": {"stable_ba_probability_auc": false, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}, "G4-G6": {"stable_ba_probability_auc": false, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}, "G4-G7": {"stable_ba_probability_auc": false, "acc_sensitivity_nonworse_every_seed": false, "pass_all": false}}, "all_controls_supported": false, "interpretation": "本轮不能确认增益门控患者关系对齐的稳定独立创新价值；不继续调整本轮参数"}
G4同checkpoint文本无干预；REAL/GATE_OFF/GATE_CONSTANT/RELATION_OFF共120行，结果见report。
训练墙钟秒：1791.533306837082; GPU任务按配置×seed交替，各120折。
后续唯一建议：控制不成立则停止这套路线，优先取得具有真实互补信息的影像/临床模态或TADPOLE，不继续调本轮超参数。


### 实验二十三历史阶段补记：审计与Stage0（原写入于2026-10-17T08:43受I/O阻塞）

实际补写：2026-10-17T09:25:52.122252+00:00。本地/远端核心源码及实验1–22保护SHA记录于manifest；六份参考Python逐字保存，三个CPU自测通过。mean_delta=-0.033275383，top20=+0.022055858，Spearman=-0.093193745，gate SD=0.049090342；四项判停不同时成立，故按预注册进入240折，不根据测试结果改设计。完整最终结果见前述COMPLETE章节。


### 实验二十三历史阶段补记（实际补写 2026-10-17T09:25:52.122285+00:00）

下文保留原阶段时间与当时状态；当前最终状态为COMPLETE。


## 实验二十三：增益校准的选择性患者关系对齐（GC-SRA） — PREFLIGHT_BLOCKED — 2026-10-17T08:48:36.510612+00:00

实验目录：`/tmp/experiment23_gain_calibrated_selective_relational_alignment`（/data创建返回I/O错误，按用户指令在/tmp执行）。
本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；远端 `b00a66acb2a492d957e1241a0599a8551c61767c`。根六核心文件与远端相同；不修改实验1–22。
G0原始P0；G1统一；G2同预算常量；G3同预算FC不确定度；G4训练内OOF条件增益；G5集合内置乱增益；G6同容量特征差分；G7仅PHENO关系核错配。
增益为FC与FC+年龄/性别Logistic教师OOF对数损失差；Ridge(alpha=150)预测；不是临床因果增益。新增关系通道只更新GFR输出CORR256，FC锚保留；零alpha、差分Decoder、有界残差，无新损失/疾病Prompt。
动机：17/18表型共享与联合正确性未带来可靠增益；19中性优于真实且Key可重参数化；20/21固定语义图响应接近零；22η不敏感、正交残差未优于E9。
阶段状态：BLOCKED_PREFLIGHT；完成0/24任务、0/240折。
快筛：mean_delta=-0.033275383，top20=0.022055858，Spearman=-0.093193745，gate SD=0.049090342；判停=False。
证据SHA `62daeaa780a89af0d142db9b6f7a7d6608c23d83928bd1c73723bf70f5fa5667`；核心/数据/参考代码SHA详见manifest。
风险：保留test=val选模、全队列既有标准化/RFE、871人transductive与22轮探索；meta OOF一阶教师训练池重叠，只作探索诊断。Exp11已有数值身份恢复链，与Exp21仅因未找到subject_IDs.txt判0/871的有界搜索结论相矛盾；本轮不恢复身份、不使用临床量表。
后续：仅在严格预检通过后执行冻结240折；不依据测试结果改设计。


### 实验二十三历史阶段补记（实际补写 2026-10-17T09:25:52.122293+00:00）

下文保留原阶段时间与当时状态；当前最终状态为COMPLETE。


## 实验二十三：增益校准的选择性患者关系对齐（GC-SRA） — PREFLIGHT — 2026-10-17T08:51:54.967616+00:00

实验目录：`/tmp/experiment23_gain_calibrated_selective_relational_alignment`（/data创建返回I/O错误，按用户指令在/tmp执行）。
本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；远端 `b00a66acb2a492d957e1241a0599a8551c61767c`。根六核心文件与远端相同；不修改实验1–22。
G0原始P0；G1统一；G2同预算常量；G3同预算FC不确定度；G4训练内OOF条件增益；G5集合内置乱增益；G6同容量特征差分；G7仅PHENO关系核错配。
增益为FC与FC+年龄/性别Logistic教师OOF对数损失差；Ridge(alpha=150)预测；不是临床因果增益。新增关系通道只更新GFR输出CORR256，FC锚保留；零alpha、差分Decoder、有界残差，无新损失/疾病Prompt。
动机：17/18表型共享与联合正确性未带来可靠增益；19中性优于真实且Key可重参数化；20/21固定语义图响应接近零；22η不敏感、正交残差未优于E9。
阶段状态：PREFLIGHT_PASS；完成0/24任务、0/240折。
快筛：mean_delta=-0.033275383，top20=0.022055858，Spearman=-0.093193745，gate SD=0.049090342；判停=False。
证据SHA `62daeaa780a89af0d142db9b6f7a7d6608c23d83928bd1c73723bf70f5fa5667`；核心/数据/参考代码SHA详见manifest。
风险：保留test=val选模、全队列既有标准化/RFE、871人transductive与22轮探索；meta OOF一阶教师训练池重叠，只作探索诊断。Exp11已有数值身份恢复链，与Exp21仅因未找到subject_IDs.txt判0/871的有界搜索结论相矛盾；本轮不恢复身份、不使用临床量表。
后续：仅在严格预检通过后执行冻结240折；不依据测试结果改设计。


### 实验二十三最终科学复核 — 2026-10-17T09:25:52.122300+00:00

本节只对冻结CSV进行分析，不修改任何训练源文件、证据、模型、超参数或checkpoint。

|Config|seed0 ASD漏诊|seed1 ASD漏诊|seed2 ASD漏诊|三seed平均ASD漏诊|三seed平均TD误报|
|---|---|---|---|---|---|
|G0|43|44|38|41.667|30.667|
|G1|83|69|56|69.333|74.000|
|G2|55|49|52|52.000|55.000|
|G3|64|66|59|63.000|51.333|
|G4|52|50|63|55.000|56.000|
|G5|49|55|49|51.000|61.667|
|G6|55|53|58|55.333|61.000|
|G7|55|57|56|56.000|51.333|
每个seed十个外层测试折覆盖同一871人（403 ASD、468 TD）；三个seed不是三个独立队列。原表按每fold指标等权平均，本表仅将每seed逐fold混淆计数相加，两者加权口径需区分。

G4相对G0每seed新增ASD漏诊：[9, 6, 25]；平均多13.333人/403 ASD。

1. **整体方案止损。** G1–G7全部低于G0；G4相对P0的ACC/BA/概率AUC为−4.436/−4.355/−2.584pp，相对历史E9为−4.628/−4.507/−3.071pp。新组中ACC最高的同预算常量G2仍明显低于P0；最高ACC、BA、概率AUC均来自原P0。
2. **统一强桥接更有害，但不能由此证明增益预测有效。** G4相对G1的BA和概率AUC三个seed全正，平均+3.694/+2.505pp；ASD SEN有一个seed下降。该比较支持降低统一交换强度可能更稳，不建立正确个体增益分配的必要性。
3. **同预算常量否定了核心增益选择假设。** G4−G2的BA三个seed为−0.024/−0.558/−0.848pp，概率AUC为−0.391/−0.141/−0.341pp，全部为负。平均ACC/BA/概率AUC分别−0.459/−0.477/−0.291pp。
4. **普通置信度、置乱增益与关系对应均未被稳定胜过。** G4−G3的BA/概率AUC均有负seed；G4−G5平均BA仅+0.097pp，seed为+1.444/−0.162/−0.992pp；G4−G7平均BA/概率AUC为−0.370/−0.575pp，错配关系在两个seed更好。因此没有稳定证据证明正确患者gain赋值或正确PHENO邻域对应是必要的。
5. **关系相对特征的局部信号不足以构成性能创新。** G4−G6平均BA+0.563pp，三个seed均正；概率AUC平均+0.343pp但seed1为−0.441pp，且两者都明显低于P0。不能将这项局部比较包装为患者关系路由已验证。G6仍有AP幅值标尺及共享并集支持的边界不变。
6. **不是新增分支未接通或近零输出。** G4相对CORR范数的平均残差1.123%，最大2.070%；非零响应比例100%，组件梯度均有效。真实AF/AP top12邻域交集平均约2.015%（按k=12归一化），提示两类邻域大幅不同；这本身不是诊断互补证据。
7. **预测确实依赖新增分支，但依赖不等于净收益。** 同checkpoint关闭gate/关系均使8.687%预测翻转，ASD概率平均绝对变化0.090518，ACC下降1.877pp；改成常量门控翻转0.918%，平均|Δp|0.006953，ACC下降0.535pp。然而重新训练常量G2优于G4，故不能把同checkpoint依赖当作选择性策略优于常量的证据。
8. **概率排序与硬分类出现不同方向。** 关闭分支后概率AUC从90.751%升至92.099%，但ACC从87.255%降至85.377%。这与分类边界/概率校准和分支共适应敏感性相容，是机制线索而非已证实原因。OFF属于分布改变，不能替代重训练P0。
9. **没有证据支持当前输入下稳定的非冗余表型收益。** Stage0 FC+年龄/性别的OOF BA/概率AUC低于FC-only，meta OOF Spearman为负；top20平均正增益与门控非恒定使预注册四条件判停未成立，因此完整训练是正确执行规则，并不代表可行性证据强。负结果只适用于这套输入、教师、核与协议，不是所有表型无信息的定理。
10. **后续唯一建议。** 保留P0；E9保持历史强参照。停止本轮超参数/Prompt扩展，转向具备真实互补信息的模态或TADPOLE，并另行预注册真正独立选模/外部验证；不能以本轮结果主张已验证的医学对齐创新或临床因果获益。

根handoff此前阶段追加曾返回I/O错误，最终追加及本次历史阶段补记成功；原159214字节内容与原SHA保持不变。


## 实验二十四 CARA：STAGE0_AUDIT

2026-10-18T10:29:04.378803+00:00
指定目录实际写入通过；根HEAD 1167dd392837a02eb4f83481ae49f02607a9dae5，远端 8cad1af1e79edc03b8db00514e8de939f4f70a11。六核心与本地HEAD/远端一致；871×320，403 ASD/468 TD，SITE20为精确one-hot。GFR覆盖、ALoss不反传、4图正则平均、val=test均按历史保留。五份冻结参考源码逐字落盘且CPU合同/标签污染/梯度单测通过。计划48折pilot；按冻结门槛决定是否续跑到240折；尚无正式成绩。


## Experiment24 PREFLIGHT_GPU_PASS
CPU合同、GPU原P0首折实跑、历史逐值parity=3通过。


## Experiment24 PREFLIGHT_EXTENDED_PASS
8组原初始化/RNG与零新增参数、原优化器覆盖、测试标签污染不变、epoch及fold边界精确恢复均通过。五份参考代码未修改；A0 seed0 fold1计入48折pilot。


## Experiment24 PILOT_RUNNING
冻结8组×3seed×前2折=48折，GPU0/1各12任务×2折=24折，2worker/卡。已完成A0首折且计入pilot；不使用/tmp；五份参考代码无修订。只在预注册门槛达标时续跑192折。


## Experiment24 PILOT_COMPLETE
{"continue_full": false, "pilot_candidates": [], "means": {"A0": {"acc": 0.919801462904911, "ba": 0.9181689642983386, "probability_auc": 0.9388326075675186}, "A1": {"acc": 0.9216954022988505, "ba": 0.919317780134323, "probability_auc": 0.9390601144679347}, "A2": {"acc": 0.9216954022988505, "ba": 0.9206433368681513, "probability_auc": 0.938971744019013}, "A3": {"acc": 0.919801462904911, "ba": 0.9178042011687463, "probability_auc": 0.9390319111331724}, "A4": {"acc": 0.9140978753047718, "ba": 0.9116295510781195, "probability_auc": 0.9403649887562704}, "A5": {"acc": 0.91977969348659, "ba": 0.9177271120537291, "probability_auc": 0.9387686800087242}, "A6": {"acc": 0.9159918146987112, "ba": 0.9141424682054407, "probability_auc": 0.9391278024713642}, "A7": {"acc": 0.9178857540926506, "ba": 0.914810887239307, "probability_auc": 0.9377326775117889}}, "note": "Fast futility screen only, not a statistical test; reuse test-set selection."}


## Experiment24 PILOT_STOP_FINAL_ACCEPTANCE
UTC 2026-10-18T11:04:11.368138+00:00
目录：/data/adodas/SPromptGL-main/experiment/experiment24_conflict_aware_relational_alignment
本地HEAD 1167dd392837a02eb4f83481ae49f02607a9dae5；远端审计 8cad1af1e79edc03b8db00514e8de939f4f70a11。五冻结代码逐字一致，0新增参数、原GFR/PLN路径/双Adam不变。
最终PILOT_STOP：24/24 pilot任务，48/48折，GPU0/1各24折，预注册门槛无候选通过，192折不执行。Pilot调度17分44秒；从preflight至pilot结束22分04秒。
六指标顺序 ACC/BA/Prob-AUC/SEN-ASD/SPE-TD/F1（%，三seed均值±population SD）：
A0 91.980±0.000, 91.817±0.083, 93.883±0.464, 89.431±1.521, 94.203±1.356, 91.382±0.181
A1 92.170±0.268, 91.932±0.287, 93.906±0.160, 88.211±0.575, 95.652±0.000, 91.408±0.314
A2 92.170±0.272, 92.064±0.254, 93.897±0.190, 90.650±1.521, 93.478±1.537, 91.693±0.290
A3 91.980±0.469, 91.780±0.466, 93.903±0.118, 88.618±0.575, 94.943±0.501, 91.223±0.548
A4 91.410±0.467, 91.163±0.499, 94.036±0.086, 87.398±1.150, 94.928±0.512, 90.602±0.566
A5 91.978±0.003, 91.773±0.031, 93.877±0.173, 88.618±0.575, 94.928±0.512, 91.272±0.094
A6 91.599±0.269, 91.414±0.270, 93.913±0.297, 88.618±0.575, 94.211±0.518, 90.911±0.316
A7 91.789±0.271, 91.481±0.350, 93.773±0.068, 86.585±1.725, 96.377±1.025, 90.825±0.511
A3未达到相对纯监督A1的至少+0.15pp门槛。A4−A0 ACC/BA/AUC=−0.570359/−0.653941/+0.153238pp；BA越−0.30pp下限。A4−A1=−0.759753/−0.768823/+0.130487pp；A4−A5=−0.568182/−0.609756/+0.159631pp。A4三个seed BA均低于A0，ASD漏诊[9,10,7]变为[11,11,9]。
A4选中checkpoint共同支持均值0.217467，平均接受1.5/3图；PHENO/ANAT/FUNC拒绝100%/16.67%/33.33%。比例仅在最佳checkpoint重算，非训练全程累计；PHENO CE不可达来自保留的i=1覆盖。无新残差分支，不能按零输出解释。
A0 6fold对13/17/18共18次ACC/BA/ProbAUC/best-epoch parity；三seed fold2边界RNG匹配历史fold3入口。原模型首步/两Adam逐值等价、epoch/fold恢复通过。根六核心、数据和805历史文件内容SHA/mtime保持；handoff原字节前缀保持。
48条主要机制诊断完整；只读周期best-state图构成47/48，首折已清理权重，未补训练。每task边界只保留RNG/结果，无模型权重。
/data实际读写均成功，无EIO/OOM/失败训练/折重训。仅临时报告脚本修复两处名称引用，无五源码改动；无git commit/push。报告results/report.md，manifest标记PILOT_STOP。
科学判断：当前没有同时胜纯监督、错配和P0的证据，不具备已验证MICCAI2027创新价值；两个fold的资源判停不证明普遍无效。停止当前CARA追分，优先获得身份可追溯的互补模态并用训练内选模/独立站点验证。历史val=test、CV前监督RFE/标准化及871人transductive/多轮探索偏差已披露。


## Experiment25 PREFLIGHT_PASS
已审核原始源码、历史P0完整30折、site/训练折QC预处理；GPU forward、gradient、原模型关闭门控等价性通过。尚未训练。


## Experiment25 COMPLETE
6组新增方法×3seeds×10folds=180折完成；历史P0为完整30折ACC 91.691%。 两卡各90折。逐折结果、负对照、研究结论见实验25/results/report.md。 若结果为阴性不可包装MICCAI2027创新点。


## 实验二十五：EPAGC 最终结果与复核
2026-10-18T18:13:42.946597+00:00

B3未在BA或连续概率AUC上超过历史P0；按冻结止损原则，终止当前EPAGC路线，不搜索rho、阈值或追加同类实验。

|组|ACC|BA|ASD Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|
|---|---:|---:|---:|---:|---:|---:|
|P0|91.69105±0.05418|91.53935±0.08408|93.33520±0.41541|89.66057±0.64562|93.41813±0.51023|90.93289±0.12038|
|B0|91.42328±0.10775|91.29137±0.09347|93.36284±0.15810|89.66260±0.23570|92.92014±0.34745|90.67249±0.08752|
|B1|91.38497±0.09332|91.20160±0.08602|93.26793±0.25670|88.91260±0.50983|93.49060±0.53321|90.53733±0.09273|
|B2|91.65273±0.05418|91.47202±0.04166|93.36368±0.16193|89.16667±0.11508|93.77737±0.17200|90.84381±0.06329|
|B3|91.53866±0.21705|91.34098±0.20175|93.31457±0.07812|88.75813±0.10923|93.92384±0.36045|90.66784±0.23480|
|B4|91.38541±0.28102|91.21162±0.32447|93.52375±0.17527|89.00203±0.99785|93.42121±0.43919|90.55911±0.37801|
|B5|91.46116±0.14336|91.29268±0.11789|93.30813±0.43133|89.16260±0.30203|93.42276±0.53775|90.64265±0.10936|

B3-P0: ΔACC/BA/Prob-AUC=-0.152386/-0.198367/-0.020626 pp。
B3-B0: ΔACC/BA/Prob-AUC=+0.115378/+0.049614/-0.048263 pp。
B3-B4: ΔACC/BA/Prob-AUC=+0.153257/+0.129359/-0.209182 pp。
B3-B5: ΔACC/BA/Prob-AUC=+0.077499/+0.048304/+0.006447 pp。

18/18任务、180/180折；历史P0只读30折；双卡各90折；墙钟47.39分钟；GPU smoke与完整下一epoch恢复通过；五份参考源码逐字一致，新增参数0；847个历史/核心/数据SHA与mtime不变。根源码SHA见实验25manifest；数据SHA b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1。
当前研究仍受test选模、预CV RFE、全队列transductive和多轮探索限制。完整六指标、逐seed配对、阴性控制、边支持/权重与站点图统计见 experiment/experiment25_evidence_preserving_asymmetric_graph_calibration/results/report.md。未commit/push。


## Experiment26 COMPLETE
6组类别Prompt对齐方法×3seeds×10folds=180折完成；历史P0完整30折ACC 91.691%。 两卡各90折。逐折结果、证据读出/无关与同几何随机控制见实验26/results/report.md。 若结果为阴性不可包装MICCAI2027创新点。


## Experiment26 PREFLIGHT_PASS
实验位于/tmp/experiment26_signed_disease_evidence_routing。源码与aa5cf9e核心逐字一致；原始五文件SHA已通过。冻结真实/无关/同几何随机文本，GPU正向、类别交换、梯度和双Adam/RNG恢复验收通过；尚未正式训练。
（本条为预检阶段补记：原GPU验收时间2026-10-19T06:08:51.380836+00:00；当时/data写入EIO，本次于2026-10-19T06:56:51.935208+00:00重试，无重新训练。）


## 实验二十六：SDER最终审计与阴性结论（2026-10-19T06:58:30.275896+00:00）

- 正式训练COMPLETE：6组×3seed×10fold=180唯一新增折；18/18任务成功，无失败重跑；GPU0/1各90折、每卡最多2worker。UTC 2026-10-19T06:10:11.048650+00:00 至 2026-10-19T06:54:42.117848+00:00，墙钟44.52分钟。
- 预检真实GPU通过：ASD=0、异常−正常；类别交换反转新增证据，原SPGS类别max不变；原主干初始化/前向保持、辅助loss梯度有效、两Adam与CPU/CUDA RNG精确回放；总参数1,462,957，新增0。
- 五份参考源码逐字SHA核验后，仅run.py追加只读反事实AUC/分模态统计和根handoff EIO容错，完整diff在manifest；不改数学、seed、超参或Prompt。
- 科学状态SCIENTIFIC_NEGATIVE：D2未稳定超过P0/无关文本/同几何随机及D3/D6，也低于历史E9。

|组|ACC %|BA/HardAUC %|ASD ProbAUC %|SEN-ASD %|SPE-TD %|ASD F1 %|
|---|---:|---:|---:|---:|---:|---:|
|P0|91.69105±0.05418|91.53935±0.08408|93.33520±0.41541|89.66057±0.64562|93.41813±0.51023|90.93289±0.12038|
|D1|91.61529±0.16225|91.44925±0.22140|93.54872±0.17167|89.32927±1.06790|93.56923±0.63056|90.80644±0.27422|
|D2|91.80556±0.19554|91.61379±0.23546|93.42005±0.09775|89.08943±0.87777|94.13814±0.44320|90.96543±0.29605|
|D3|91.61486±0.16225|91.44361±0.18347|93.46140±0.06766|89.32724±0.35930|93.55998±0.00786|90.83009±0.18644|
|D4|91.65273±0.19536|91.45697±0.19613|93.43397±0.17144|88.92073±0.30317|93.99322±0.17000|90.82173±0.23822|
|D5|91.38454±0.09385|91.22575±0.10155|93.28605±0.17175|89.09350±0.40829|93.35800±0.30092|90.56147±0.12634|
|D6|91.80599±0.14336|91.64889±0.14625|93.50284±0.20305|89.73780±0.23283|93.55998±0.17561|91.04993±0.15155|

- D2−P0：ACC/BA/ProbAUC +0.11451/+0.07444/+0.08485pp，BA三个seed +0.37985/−0.02420/−0.13234pp；ASD漏诊125→132（三seed重复队列）。
- D2−D1/D3/D4/D5/D6的概率AUC分别−0.12867/−0.04136/−0.01392/+0.13400/−0.08280pp；真实文本未同时稳定胜过两种负对照。历史E9 91.88262/91.69151/93.82272，D2三指标均低。
- 分支非近零：D2 score SD≈0.35493；关闭读出mean|Δp_ASD|=.0121338、翻转率.38227%；类交换.0264095/.76411%。文本方向进入决策，但不证明医学必要性。FUNC绝对证据最大，CORR门控均值最高；四个token不是ROI，ANAT/FUNC多为质量指标。
- 原始test=val、CV前标准化/监督RFE、全871人transductive传播与25轮探索的乐观风险全部保留；不称独立无泄漏临床泛化，不报告伪独立P值。
- 代码版本：本地HEAD 1167dd392837a02eb4f83481ae49f02607a9dae5；六核心与远端aa5cf9ea39c74eaa57fb7a2cd1a3eca3e15a0974逐字相同。数据SHA b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1；RN50 SHA afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762。
- 文本cache SHA 7cbf8c428c455ff09bff9d938ee6cd04b9ca194a0b7382262031e3605a7e91e4；完整源码/历史基线/报告签名见manifest。保护清单1258文件及25轮报告逐项未变，原handoff前179932字节及原hash保留。
- 实验目录 /tmp/experiment26_signed_disease_evidence_routing；报告 results/report.md；预检根handoff曾遇/data Errno5，已补记，最终写入成功并fsync；临时状态/权重/源码副本已清理。未执行git commit/push。
- 后续唯一建议：不再在本历史测试参与选模协议上改写静态Prompt追分；先在训练内选模/LOSO或独立队列中检验真实互补模态，当前不足以升级为MICCAI2027第一创新点。


## 实验二十六：GitHub发布与I/O恢复后交接更新（2026-10-19T07:08:54+00:00）

- 当前阶段：训练工程 `COMPLETE`，科学结论 `SCIENTIFIC_NEGATIVE`。此前“SDER最终审计与阴性结论”中的六指标表、逐seed比较及机制结论继续有效。18/18任务、180/180唯一新增折，GPU0/1各90折；未追加训练或修改冻结设计。
- I/O状态：用户已确认 `/data` I/O问题修复。本次使用锁、同目录临时文件、fsync及原子替换更新项目根 `handoff.md`，并核验原有字节前缀完整保留。实验期间EIO及阶段补记属于历史执行记录，予以保留。
- 实际交付目录：`/tmp/experiment26_signed_disease_evidence_routing`；完整中文报告为该目录下 `results/report.md`，180条逐折分类/机制/反事实字段为 `results/fold_metrics.csv`，运行状态与签名为 `manifest.json`。交付共9个文件，临时训练状态与权重已清理。
- GitHub发布：用户授权上传后，已通过独立临时Git副本将全部9个文件发布至 [experiment/experiment26_signed_disease_evidence_routing](https://github.com/tangaii/SPromptGL/tree/bfd5f160314704c98106fd3d8abd9402d4f60f61/experiment/experiment26_signed_disease_evidence_routing)。提交 [`bfd5f160314704c98106fd3d8abd9402d4f60f61`](https://github.com/tangaii/SPromptGL/commit/bfd5f160314704c98106fd3d8abd9402d4f60f61)，提交说明 `Add Experiment 26 signed disease evidence routing`；本次已再次核实远端main指向该提交，提交中的9个文件逐个SHA256与本地交付一致。
- 版本含义：此前实验报告与handoff中的“未commit/push”记录对应训练交付阶段；随后按用户上传指令完成了上述发布。`/data/adodas/SPromptGL-main`本地HEAD仍为 `1167dd392837a02eb4f83481ae49f02607a9dae5`，原六核心源码SHA与实验26审计签名一致，原工作区既有状态保留。
- 冻结产物SHA256：manifest为 `1ee30468918fb6e7e1dda215c1ee52b5383339ac31f763a14f98a92abe87a2f3`；fold_metrics为 `e01c0c407cc0739359bb833c6bf0bfd40f81779b259250cc14c5708e90f18029`；report为 `ca3f3a1ca2c643c392dcf779931dfb8e895e4fa6fd5ffc61c4d58b08ca10627a`；文本原型缓存为 `7cbf8c428c455ff09bff9d938ee6cd04b9ca194a0b7382262031e3605a7e91e4`。实验manifest中的handoff签名指向训练完成时快照；本条是其后的授权交接追加。
- 核心判断：D2的ACC/BA/ASD Prob-AUC为91.80556/91.61379/93.42005%，相对历史P0均值仅+0.11451/+0.07444/+0.08485pp，ASD灵敏度下降0.57114pp、三seed累计漏诊125→132。D2概率AUC低于无关文本D4、仅辅助监督D3及FC-only D6，且三指标均低于历史E9；未证明真实医学Prompt或四组证据的独立增益。
- 机制结论：有符号读出确实改变预测，非近零失效；同checkpoint关闭读出/类别交换的响应只支持类别方向进入决策。真实文本没有稳定胜过完整训练的负对照，不能据此主张医学语义必要性或MICCAI2027创新成立。
- 后续建议保持：保留P0/E9参照，停止在同一历史测试参与选模协议上改写静态Prompt追分；先在训练内选模、LOSO或独立队列下验证真实互补模态。继续披露test=val、预CV监督RFE/标准化、全队列transductive与多轮探索的局限。
- 本次追加前handoff为184280字节，SHA256为 `414136beddce33a6eb032141cc5a1e941ee6253613d84902eda0050889764c82`；旧内容逐字节保留。本次操作仅更新根handoff。


## Experiment 27 — EGRP-CAM | UTC 2026-10-19T08:11:55+00:00
<!-- EGRP27-GROUNDING-STOP:ba674d08c28362d0c3e4d6d1927ee66520c6e0b3fa024dd31d2899682265a963 -->

- STATUS: STOP_UNGROUNDED_INPUTS；STAGE: GROUNDING_ONLY；SCIENCE: NOT_EVALUABLE。Gate 0完整审计结束；这是预注册科学可识别性判停，不是训练失败，也不是模型性能阴性试验。
- REPO: 本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；远端main `bfd5f160314704c98106fd3d8abd9402d4f60f61`；六核心文件SHA逐字相同。config.py在当前根目录不存在，参数来自main.py与历史shell。源/数据/论文/历史文件的完整SHA及代码锚点见实验manifest与feasibility.json。
- DATA SHA256: `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`；input 871×320，403 ASD/468 TD，0=ASD；PHENO48=年龄26独热+站点20独热+性别2，ANAT6/FUNC10全为QA，CORR256是RFE后的LedoitWolf相关Fisher-z。质量指标不能称脑体积、皮层厚度、白质完整性或激活图，CAM后的4×4×9是潜在Token，不是ROI。
- 论文/源码差异：MMGPL已有疾病概念Token加权、概念响应Token图及GCN Prompt；SPromptGL是患者GFR→CAM→SPGS/RES。当前Attention还做全局min缩放和行L2归一化；若未来有真实关系，偏置须在归一化后/softmax前，只接首层CAM。本轮未创建补丁。原i=1覆盖及ALoss不加入simple-2反传如实记录，未修复。
- Gate0: 10条候选逐项登记，符合两种不同疾病/临床测量端点且方向可独立检验的关系为0，要求至少2。体积/厚度/白质/侧化/脑室/激活端点缺失；症状量表不在固定输入；年龄/性别—FC属于发育，头动/QC—FC属于采集关系，均不计入疾病grounding。REJECT只意味着新增修正为0，不删除原CAM。
- 身份链复核纠正：利用实验11旧映射重建871×64的PHENO/QC指纹，871人唯一匹配，最大误差7.105427357601e-15；没有使用标签选择匹配。实验21仅按subject_IDs.txt搜索得NOT_JOINABLE，未消费实验11已有指纹映射，故其0/871结论不能代表当前完整证据。历史文件不改。
- 旁路量表只作可用性审计：ADOS_TOTAL非负观测ASD282/403、TD19/468；ADI_R社会分276/403与0/468；SRS_RAW_TOTAL为163/403与137/468。非负观测不等于临床有效/版本可比；原始量表未加入320维，也未用其缺失模式监督。引入它们属于另一个输入实验，需新基线和独立诊断/量表泄漏审计。
- Gate1: NOT_RUN_G0_FAILED，未计算关联、置乱p值或OOF增益；不可把未运行写成效应0。Stage2实际0/180fold，GPU0=0、GPU1=0，未启动GPU、CLIP、训练worker或模型副本；R0–R5仅为冻结的条件性计划。
- 历史P0实验13 F0与E9实验9均核实完整30折，ACC/BA/ProbAUC分别91.69105/91.53935/93.33520%和91.88262/91.69151/93.82272%，仅作来源核验，无新模型分数。没有使用实验24两折pilot作为完整基线。
- 历史教训：17/18年龄性别选择对齐无稳定收益；19自由Key与中性控制否定简单文本归因；20–22医学概念图及E9组合未胜严格控制；23增益路由失败；24按pilot门槛止损；25质量相似无可靠增益；26方向敏感不等于真实文案必要。此次不重做这些模块。
- 协议及偏差：未来条件性训练仍应披露test=val选模、外层CV前RFE/全队列QC标准化、全871 transductive及此前26轮探索；本轮未训练，不能把既往指标写成本轮独立外部泛化。医学关系方法未可评估，不能声称MICCAI2027创新成立。
- 实验文件：`/data/adodas/SPromptGL-main/experiment/experiment27_evidence_grounded_relational_prompt_cam`；报告`results/report.md`，数据/候选账本`results/feasibility.json`，状态`manifest.json`；保留三份CPU审计源码，共6个文件。根源码、数据、历史1–26文件保护SHA保持；不自动git add/commit/push。
- 后续最小建议：先在已有身份链上独立核验量表版本/覆盖/可靠度与诊断依赖，或获取同患者定量MRI/DTI/PET等真实互补测量；预先固定至少两项可检验关系，再在训练内选模与独立站点/外队列上另立协议。不能继续把这四组QA/人口学上下文更名为疾病模态追分。


<!-- EXPERIMENT28_IA_FDSA_FINAL -->
## 实验二十八：IA-FDSA影像锚定的细粒度疾病语义对齐

更新时间：2026-10-19T16:06:43.239949+00:00。STATUS=COMPLETE，科学结论=NO-GO。

本轮保留原GFR/CAM/RES及历史协议，以原始FC及训练折TD参考/站点分层差异加权252真实连接的504条方向Prompt，在共享文本64维空间监督影像锚；年龄/性别与QA仅作受约束条件修正，替换SPGS。

十组：E0_P0, E1_NO_SPGS, E2_OLD_FC, E3_FINE_FC, E4_FINE_FULL, E5_NUMERIC_FULL, E6_WRONG_ROI, E7_WRONG_DIRECTION, E8_FINE_NO_ALIGN, E9_WRONG_AUX_PAIR。

实际完成300/300折、30/30任务，GPU分配/完成{'0': 150, '1': 150}。P0历史parity：{"passed": true, "rows": 30, "fields": ["acc", "ba", "probability_auc", "best_epoch", "initial_trunk_sha256"], "checked_utc": "2026-10-19T14:46:34.585349+00:00", "maximum_acc_ba_auc_error": 0.0, "initial_state_and_rng_preflight": true}。

E0_P0：acc=91.69105±0.05418%, ba=91.53935±0.08408%, probability_auc=93.33520±0.41541%, sensitivity_asd=89.66057±0.64562%, specificity_td=93.41813±0.51023%, f1_asd=90.93289±0.12038%。
E1_NO_SPGS：acc=91.53779±0.19536%, ba=91.34905±0.20625%, probability_auc=93.57039±0.30428%, sensitivity_asd=88.92073±0.71450%, specificity_td=93.77737±0.60617%, f1_asd=90.70334±0.21945%。
E2_OLD_FC：acc=91.53779±0.05418%, ba=91.34309±0.06147%, probability_auc=93.55219±0.26848%, sensitivity_asd=88.83943±0.19917%, specificity_td=93.84675±0.10248%, f1_asd=90.70297±0.07397%。
E3_FINE_FC：acc=91.69105±0.05418%, ba=91.47256±0.07651%, probability_auc=93.38635±0.15774%, sensitivity_asd=88.66667±0.65496%, specificity_td=94.27845±0.53157%, f1_asd=90.81486±0.10378%。
E4_FINE_FULL：acc=91.61442±0.16255%, ba=91.42719±0.17620%, probability_auc=93.19882±0.50570%, sensitivity_asd=89.00610±0.45706%, specificity_td=93.84829±0.10466%, f1_asd=90.77521±0.23647%。
E5_NUMERIC_FULL：acc=91.46160±0.30108%, ba=91.27532±0.34054%, probability_auc=93.35904±0.15532%, sensitivity_asd=88.91667±0.92844%, specificity_td=93.63398±0.26743%, f1_asd=90.62720±0.39481%。
E6_WRONG_ROI：acc=91.42285±0.19536%, ba=91.23025±0.19801%, probability_auc=93.25822±0.16141%, sensitivity_asd=88.75407±0.42892%, specificity_td=93.70644±0.35952%, f1_asd=90.57917±0.22862%。
E7_WRONG_DIRECTION：acc=91.46160±0.21612%, ba=91.29334±0.26263%, probability_auc=93.42765±0.15626%, sensitivity_asd=89.09146±0.88464%, specificity_td=93.49522±0.36620%, f1_asd=90.63693±0.32649%。
E8_FINE_NO_ALIGN：acc=91.30834±0.14289%, ba=91.11195±0.14899%, probability_auc=93.37431±0.19609%, sensitivity_asd=88.59146±0.52689%, specificity_td=93.63244±0.44470%, f1_asd=90.43667±0.18806%。
E9_WRONG_AUX_PAIR：acc=91.57654±0.21612%, ba=91.38426±0.22948%, probability_auc=93.49009±0.14858%, sensitivity_asd=88.91870±0.64688%, specificity_td=93.84983±0.44094%, f1_asd=90.74542±0.25832%。

预设配对：[{"left": "E3_FINE_FC", "right": "E2_OLD_FC", "metrics": {"acc": {"delta_pp": 0.15325670498084273, "per_seed_delta_pp": [0.2298850574712652, 0.11494252873563206, 0.11494252873563092], "wins": 8, "ties": 16, "losses": 6}, "ba": {"delta_pp": 0.12946740822634992, "per_seed_delta_pp": [0.1633367929424001, 0.08218451749734901, 0.1428809142393006], "wins": 10, "ties": 12, "losses": 8}, "hard_auc": {"delta_pp": 0.12946740822634992, "per_seed_delta_pp": [0.1633367929424001, 0.08218451749734901, 0.1428809142393006], "wins": 10, "ties": 12, "losses": 8}, "probability_auc": {"delta_pp": -0.16583936884697317, "per_seed_delta_pp": [-0.2776900340696309, 0.14457029399156163, -0.3643983664628503], "wins": 12, "ties": 0, "losses": 18}, "sensitivity_asd": {"delta_pp": -0.17276422764227584, "per_seed_delta_pp": [-0.5243902439024373, -0.4878048780487809, 0.4939024390243907], "wins": 6, "ties": 17, "losses": 7}, "specificity_td": {"delta_pp": 0.4316990440949745, "per_seed_delta_pp": [0.8510638297872343, 0.6521739130434789, -0.20814061054578947], "wins": 7, "ties": 18, "losses": 5}, "f1_asd": {"delta_pp": 0.1118866421206203, "per_seed_delta_pp": [0.14987206885022863, 0.0414365579121867, 0.1443512995994456], "wins": 10, "ties": 12, "losses": 8}}}, {"left": "E4_FINE_FULL", "right": "E3_FINE_FC", "metrics": {"acc": {"delta_pp": -0.07662835249042137, "per_seed_delta_pp": [0.0, -0.2298850574712652, 1.1102230246251565e-15], "wins": 9, "ties": 12, "losses": 9}, "ba": {"delta_pp": -0.045363183742845134, "per_seed_delta_pp": [0.09077243293247106, -0.2089810699217065, -0.01788091423929994], "wins": 11, "ties": 9, "losses": 10}, "hard_auc": {"delta_pp": -0.045363183742845134, "per_seed_delta_pp": [0.09077243293247106, -0.2089810699217065, -0.01788091423929994], "wins": 11, "ties": 9, "losses": 10}, "probability_auc": {"delta_pp": -0.18752961350150107, "per_seed_delta_pp": [0.2130423501274814, -0.9220403420500456, 0.146409151418061], "wins": 14, "ties": 1, "losses": 15}, "sensitivity_asd": {"delta_pp": 0.339430894308943, "per_seed_delta_pp": [1.249999999999999, 0.012195121951220633, -0.24390243902439046], "wins": 7, "ties": 19, "losses": 4}, "specificity_td": {"delta_pp": -0.43015726179463476, "per_seed_delta_pp": [-1.0684551341350579, -0.4301572617946359, 0.20814061054578947], "wins": 6, "ties": 14, "losses": 10}, "f1_asd": {"delta_pp": -0.03964593692112069, "per_seed_delta_pp": [0.13878553971553576, -0.25918635813284907, 0.0014630076539512427], "wins": 11, "ties": 9, "losses": 10}}}, {"left": "E4_FINE_FULL", "right": "E5_NUMERIC_FULL", "metrics": {"acc": {"delta_pp": 0.15282131661442025, "per_seed_delta_pp": [0.688349007314526, -0.34482758620689724, 0.11494252873563206], "wins": 13, "ties": 9, "losses": 8}, "ba": {"delta_pp": 0.1518693170280446, "per_seed_delta_pp": [0.751060445387064, -0.389316012725347, 0.0938635184224168], "wins": 15, "ties": 5, "losses": 10}, "hard_auc": {"delta_pp": 0.1518693170280446, "per_seed_delta_pp": [0.751060445387064, -0.389316012725347, 0.0938635184224168], "wins": 15, "ties": 5, "losses": 10}, "probability_auc": {"delta_pp": -0.16021750411768934, "per_seed_delta_pp": [0.23933913945984875, -0.9011021863225156, 0.1811105345095987], "wins": 15, "ties": 1, "losses": 14}, "sensitivity_asd": {"delta_pp": 0.08943089430894242, "per_seed_delta_pp": [1.7195121951219516, -1.2134146341463425, -0.23780487804878184], "wins": 14, "ties": 8, "losses": 8}, "specificity_td": {"delta_pp": 0.21430773974714748, "per_seed_delta_pp": [-0.21739130434782483, 0.4347826086956508, 0.4255319148936165], "wins": 8, "ties": 16, "losses": 6}, "f1_asd": {"delta_pp": 0.14800944810250258, "per_seed_delta_pp": [0.8585091500497377, -0.5228565447317057, 0.10837573898947572], "wins": 15, "ties": 5, "losses": 10}}}, {"left": "E4_FINE_FULL", "right": "E6_WRONG_ROI", "metrics": {"acc": {"delta_pp": 0.19157088122605304, "per_seed_delta_pp": [0.22988505747126411, 0.22988505747126298, 0.11494252873563206], "wins": 10, "ties": 15, "losses": 5}, "ba": {"delta_pp": 0.19693824597820728, "per_seed_delta_pp": [0.28418526206539063, 0.2056079510841391, 0.10102152478509209], "wins": 14, "ties": 10, "losses": 6}, "hard_auc": {"delta_pp": 0.19693824597820728, "per_seed_delta_pp": [0.28418526206539063, 0.2056079510841391, 0.10102152478509209], "wins": 14, "ties": 10, "losses": 6}, "probability_auc": {"delta_pp": -0.05939998345404336, "per_seed_delta_pp": [-0.052204372645018315, -0.5413968547641101, 0.4154012770469983], "wins": 13, "ties": 0, "losses": 17}, "sensitivity_asd": {"delta_pp": 0.25203252032520385, "per_seed_delta_pp": [0.9939024390243911, -0.2317073170731698, -0.0060975609756097615], "wins": 10, "ties": 15, "losses": 5}, "specificity_td": {"delta_pp": 0.14184397163120588, "per_seed_delta_pp": [-0.42553191489361536, 0.6429232192414425, 0.20814061054579058], "wins": 8, "ties": 15, "losses": 7}, "f1_asd": {"delta_pp": 0.19604619654674788, "per_seed_delta_pp": [0.32145945580199675, 0.15606692214816364, 0.11061221169008316], "wins": 14, "ties": 10, "losses": 6}}}, {"left": "E4_FINE_FULL", "right": "E7_WRONG_DIRECTION", "metrics": {"acc": {"delta_pp": 0.15282131661442025, "per_seed_delta_pp": [0.11494252873563315, 0.2285788923719956, 0.11494252873563206], "wins": 12, "ties": 11, "losses": 7}, "ba": {"delta_pp": 0.13385114655957042, "per_seed_delta_pp": [0.06841282913292002, 0.25609756097561, 0.07704304957018127], "wins": 13, "ties": 8, "losses": 9}, "hard_auc": {"delta_pp": 0.13385114655957042, "per_seed_delta_pp": [0.06841282913292002, 0.25609756097561, 0.07704304957018127], "wins": 13, "ties": 8, "losses": 9}, "probability_auc": {"delta_pp": -0.2288343373720529, "per_seed_delta_pp": [-0.22201665124884526, -0.9368865774689216, 0.47240021660160836], "wins": 15, "ties": 0, "losses": 15}, "sensitivity_asd": {"delta_pp": -0.08536585365853704, "per_seed_delta_pp": [-0.5060975609756113, 0.51219512195122, -0.26219512195121975], "wins": 7, "ties": 16, "losses": 7}, "specificity_td": {"delta_pp": 0.3530681467776753, "per_seed_delta_pp": [0.6429232192414447, -1.1102230246251565e-15, 0.41628122109158233], "wins": 10, "ties": 13, "losses": 7}, "f1_asd": {"delta_pp": 0.1382863532986034, "per_seed_delta_pp": [0.03923726391422466, 0.26491282410976535, 0.11070897187182016], "wins": 13, "ties": 8, "losses": 9}}}, {"left": "E4_FINE_FULL", "right": "E8_FINE_NO_ALIGN", "metrics": {"acc": {"delta_pp": 0.306078021595263, "per_seed_delta_pp": [0.45977011494252923, 0.2285788923719956, 0.22988505747126411], "wins": 10, "ties": 15, "losses": 5}, "ba": {"delta_pp": 0.31524183419447627, "per_seed_delta_pp": [0.5143160127253465, 0.1969777306468723, 0.23443175921120996], "wins": 11, "ties": 12, "losses": 7}, "hard_auc": {"delta_pp": 0.31524183419447627, "per_seed_delta_pp": [0.5143160127253465, 0.1969777306468723, 0.23443175921120996], "wins": 11, "ties": 12, "losses": 7}, "probability_auc": {"delta_pp": -0.17548866978031633, "per_seed_delta_pp": [0.26986642900656665, -0.9023769770537694, 0.10604453870625362], "wins": 17, "ties": 0, "losses": 13}, "sensitivity_asd": {"delta_pp": 0.41463414634146345, "per_seed_delta_pp": [1.4634146341463405, -0.4756097560975603, 0.25609756097561], "wins": 7, "ties": 18, "losses": 5}, "specificity_td": {"delta_pp": 0.21584952204748725, "per_seed_delta_pp": [-0.43478260869564966, 0.8695652173913027, 0.21276595744680882], "wins": 9, "ties": 14, "losses": 7}, "f1_asd": {"delta_pp": 0.3385412133864518, "per_seed_delta_pp": [0.6603025480038649, 0.10207729456121138, 0.2532437975942792], "wins": 11, "ties": 12, "losses": 7}}}, {"left": "E4_FINE_FULL", "right": "E9_WRONG_AUX_PAIR", "metrics": {"acc": {"delta_pp": 0.03787878787878821, "per_seed_delta_pp": [0.4584639498432607, -0.34482758620689724, 1.1102230246251565e-15], "wins": 10, "ties": 13, "losses": 7}, "ba": {"delta_pp": 0.04292829584170062, "per_seed_delta_pp": [0.4854922045982734, -0.4079330340019405, 0.05122571692876887], "wins": 12, "ties": 10, "losses": 8}, "hard_auc": {"delta_pp": 0.04292829584170062, "per_seed_delta_pp": [0.4854922045982734, -0.4079330340019405, 0.05122571692876887], "wins": 12, "ties": 10, "losses": 8}, "probability_auc": {"delta_pp": -0.2912671194242008, "per_seed_delta_pp": [-0.14425441664222216, -1.140131089099976, 0.41058414746959593], "wins": 12, "ties": 0, "losses": 18}, "sensitivity_asd": {"delta_pp": 0.08739837398373992, "per_seed_delta_pp": [0.9756097560975618, -1.4634146341463417, 0.7499999999999996], "wins": 9, "ties": 15, "losses": 6}, "specificity_td": {"delta_pp": -0.0015417823003390414, "per_seed_delta_pp": [-0.004625346901016014, 0.6475485661424596, -0.6475485661424607], "wins": 7, "ties": 16, "losses": 7}, "f1_asd": {"delta_pp": 0.029789630097583435, "per_seed_delta_pp": [0.5412111731446456, -0.5565838294630066, 0.10474154661111122], "wins": 12, "ties": 10, "losses": 8}}}, {"left": "E4_FINE_FULL", "right": "E1_NO_SPGS", "metrics": {"acc": {"delta_pp": 0.07662835249042137, "per_seed_delta_pp": [0.45977011494252923, -0.34482758620689724, 0.11494252873563206], "wins": 11, "ties": 11, "losses": 8}, "ba": {"delta_pp": 0.07814391973707109, "per_seed_delta_pp": [0.45898106992171117, -0.4163432684280588, 0.19179395771756091], "wins": 12, "ties": 10, "losses": 8}, "hard_auc": {"delta_pp": 0.07814391973707109, "per_seed_delta_pp": [0.45898106992171117, -0.4163432684280588, 0.19179395771756091], "wins": 12, "ties": 10, "losses": 8}, "probability_auc": {"delta_pp": -0.3715732948263817, "per_seed_delta_pp": [0.18379549197897305, -1.4428374811037692, 0.1443221046456511], "wins": 11, "ties": 2, "losses": 17}, "sensitivity_asd": {"delta_pp": 0.08536585365853704, "per_seed_delta_pp": [0.4878048780487809, -1.4756097560975612, 1.2439024390243913], "wins": 9, "ties": 15, "losses": 6}, "specificity_td": {"delta_pp": 0.07092198581560294, "per_seed_delta_pp": [0.4301572617946359, 0.6429232192414425, -0.8603145235892695], "wins": 6, "ties": 18, "losses": 6}, "f1_asd": {"delta_pp": 0.0718729966963183, "per_seed_delta_pp": [0.4838331725393896, -0.5426179928760277, 0.27440381042559303], "wins": 12, "ties": 10, "losses": 8}}}, {"left": "E4_FINE_FULL", "right": "E0_P0", "metrics": {"acc": {"delta_pp": -0.07662835249042173, "per_seed_delta_pp": [0.0, -0.2298850574712652, 0.0], "wins": 8, "ties": 15, "losses": 7}, "ba": {"delta_pp": -0.11215714146040501, "per_seed_delta_pp": [-0.013771688364427881, -0.24874777193655387, -0.0739519640802333], "wins": 9, "ties": 13, "losses": 8}, "hard_auc": {"delta_pp": -0.11215714146040501, "per_seed_delta_pp": [-0.013771688364427881, -0.24874777193655387, -0.0739519640802333], "wins": 9, "ties": 13, "losses": 8}, "probability_auc": {"delta_pp": -0.1363800455765892, "per_seed_delta_pp": [-0.502087046772407, -0.38481194016380615, 0.4777588502064456], "wins": 15, "ties": 0, "losses": 15}, "sensitivity_asd": {"delta_pp": -0.6544715447154473, "per_seed_delta_pp": [-0.018292682926829285, -0.7195121951219507, -1.225609756097562], "wins": 5, "ties": 19, "losses": 6}, "specificity_td": {"delta_pp": 0.43015726179463476, "per_seed_delta_pp": [-0.009250693802033139, 0.22201665124884307, 1.0777058279370944], "wins": 8, "ties": 17, "losses": 5}, "f1_asd": {"delta_pp": -0.15768146289309412, "per_seed_delta_pp": [0.007298255290024701, -0.34890583417176213, -0.1314368097975449], "wins": 9, "ties": 13, "losses": 8}}}]。

历史P0 ACC91.69105/BA91.53935/ProbAUC93.33520%；历史Exp9 E9 ACC91.88262/BA91.69151/ProbAUC93.82272%，不同于本轮E9。

细粒度语义、正确ROI/方向、语义监督及辅助配对的必要性以完整重训负对照判断；teacher-off预期不改变推理，不能据此宣称语义无用；其他反事实和梯度/教师坍塌详见实验28/results/report.md。

执行时间 2026-10-19T14:31:51.551269+00:00 至 2026-10-19T16:06:43.239949+00:00；任务耗时 {"E0_P0_seed0": 758.4221819378436, "E0_P0_seed1": 675.4982658289373, "E0_P0_seed2": 715.5966301560402, "E1_NO_SPGS_seed0": 724.2013568766415, "E1_NO_SPGS_seed1": 730.7728206887841, "E1_NO_SPGS_seed2": 1598.8827958218753, "E2_OLD_FC_seed0": 1356.6969047486782, "E2_OLD_FC_seed1": 1293.9028919674456, "E2_OLD_FC_seed2": 971.613663174212, "E3_FINE_FC_seed0": 1623.1214179322124, "E3_FINE_FC_seed1": 1313.170266482979, "E3_FINE_FC_seed2": 1255.7978079989552, "E4_FINE_FULL_seed0": 834.8987557217479, "E4_FINE_FULL_seed1": 1486.7043199650943, "E4_FINE_FULL_seed2": 1159.8286404497921, "E5_NUMERIC_FULL_seed0": 1732.7575345486403, "E5_NUMERIC_FULL_seed1": 1018.8799349367619, "E5_NUMERIC_FULL_seed2": 1701.5514034181833, "E6_WRONG_ROI_seed0": 675.9066250547767, "E6_WRONG_ROI_seed1": 729.298168964684, "E6_WRONG_ROI_seed2": 745.6419029682875, "E7_WRONG_DIRECTION_seed0": 753.6403955854475, "E7_WRONG_DIRECTION_seed1": 733.8398070782423, "E7_WRONG_DIRECTION_seed2": 777.7721496336162, "E8_FINE_NO_ALIGN_seed0": 901.7212639078498, "E8_FINE_NO_ALIGN_seed1": 951.894286762923, "E8_FINE_NO_ALIGN_seed2": 804.2143075279891, "E9_WRONG_AUX_PAIR_seed0": 939.9437686949968, "E9_WRONG_AUX_PAIR_seed1": 819.9777272865176, "E9_WRONG_AUX_PAIR_seed2": 752.5718774758279}。

源码SHA {"main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67", "ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763"}；数据SHA {"/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1", "/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv": "b383308bb413a94d1646511a9bea5bc16f54c975b938ac4d947d676e357c5eb2", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv": "4f9cd719d27c1e8b9a6cf8933534c0b796101cacf248e509fcfd3c5868fa3ebf"}；Prompt缓存SHA 2a3cf9332ad3d1dbdba70731a46acee43f3f7e52c92fe3ab2366fa90b0d54420；历史保护 {"status": "UNCHANGED_SHA256", "files": 985}。

偏差与局限：test=val选模、外层前RFE/预处理、全871人transductive、多轮探索；不称独立无泄漏泛化、医学可靠度或个体病灶验证。若NO-GO，停止静态FC文本追分，优先真实互补模态和独立训练内选模验证。无自动Git操作。

实测机制与全部对照的最终判读：E4相对完整P0的ΔACC/ΔBA/ΔProb-AUC分别为−0.07663/−0.11216/−0.13638pp；相对历史实验9 E9分别为−0.26820/−0.26432/−0.62390pp（描述性）。细粒度E3相对粗粒度E2的BA为+0.12947pp，但Prob-AUC为−0.16584pp，未获得指标一致的增量。E4相对数值码/错配ROI/反向语义/关闭对齐损失/错误辅助患者配对的Prob-AUC差分别为−0.16022/−0.05940/−0.22883/−0.17549/−0.29127pp。正确辅助融合E4相对CORR主锚E3也未改善BA和Prob-AUC，因此医学文本、方向、正确ROI及正确患者辅助信息的必要性未通过预设联合门槛；NO-GO，不具备当前证据下MICCAI 2027第一创新点价值。

反事实与响应：关闭新增残差时ASD概率平均绝对变化0.01399688、翻转率1.52995%；关闭辅助修正变化0.00130648；关闭CAM上下文变化0.00664742。teacher-off严格不改变推理，为训练专用教师的预期行为。E4残差平均L2=0.32283577、非零比例100%，不是未接入或整体近零；文本增强/减弱向量cos=0.99670637，教师两类中心cos=0.99998766、student-teacher train cos=0.05732746，未提供强医学语义区分证据。ASD漏诊事件累计133次，P0为125次（三seed重复评估）。

最终完整中文报告：experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/report.md；包括三个seed、全部预设配对、300条机制诊断及270条反事实。训练完成后仅扩充报告生成和诊断字段解释，全部训练/恢复函数AST、模型、Prompt及协议保持不变；变更逐行diff和SHA记录于manifest。实验28只保留一个唯一标记的总结条目，历史handoff前缀保持原字节。


<!-- EXPERIMENT28_SUPPLEMENT_E10_COARSE_FULL_FINAL -->
## 实验28补充对照E10：粗粒度Prompt＋完整辅助条件修正

更新时间：2026-10-20T05:48:00.853249+00:00。STATUS=COMPLETE。

在/tmp/experiment28_supplement_e10_coarse_full实际新增3任务/30折；历史E0–E9的300折只读。GPU0 seed0/2共20折，GPU1 seed1共10折。E10完整复用E4结构/损失/辅助机制，仅将fine教师换为原E2冻结coarse；28517新参数、原anchor语义监督保持。

E10指标：{"acc": {"mean": 0.9180599094392198, "sd_population": 0.001083688553542569, "per_seed": [0.9172936259143156, 0.9172936259143156, 0.9195924764890282]}, "ba": {"mean": 0.9160162131570436, "sd_population": 0.0009888149715177185, "per_seed": [0.9152295187382957, 0.9154083278806885, 0.9174107928521469]}, "hard_auc": {"mean": 0.9160162131570436, "sd_population": 0.0009888149715177185, "per_seed": [0.9152295187382957, 0.9154083278806885, 0.9174107928521469]}, "probability_auc": {"mean": 0.9322242841993638, "sd_population": 0.0025061213947540745, "per_seed": [0.9287223889352678, 0.9344480043320322, 0.9335024593307912]}, "sensitivity_asd": {"mean": 0.8892479674796748, "sd_population": 0.0011356678898129398, "per_seed": [0.8884146341463415, 0.8908536585365854, 0.8884756097560975]}, "specificity_td": {"mean": 0.9427844588344128, "sd_population": 0.002657864551074746, "per_seed": [0.9420444033302499, 0.9399629972247918, 0.9463459759481962]}, "f1_asd": {"mean": 0.9096702499513171, "sd_population": 0.0009443882936076199, "per_seed": [0.908943173719545, 0.90906356954102, 0.9110040065933861]}}。

全部配对：[{"left": "E10_COARSE_FULL", "right": "E2_OLD_FC", "metrics": {"acc": {"delta_pp": 0.26819923371647436, "per_seed_delta_pp": [0.22988505747126411, 0.22988505747126411, 0.344827586206895], "wins": 10, "ties": 17, "losses": 3}, "ba": {"delta_pp": 0.2585324488767567, "per_seed_delta_pp": [0.22833419823560241, 0.23600832562442167, 0.3112548227702461], "wins": 11, "ties": 14, "losses": 5}, "hard_auc": {"delta_pp": 0.2585324488767567, "per_seed_delta_pp": [0.22833419823560241, 0.23600832562442167, 0.3112548227702461], "wins": 11, "ties": 14, "losses": 5}, "probability_auc": {"delta_pp": -0.3297590307077906, "per_seed_delta_pp": [-0.5808928047652351, 0.1605221001331203, -0.568906387491257], "wins": 9, "ties": 1, "losses": 20}, "sensitivity_asd": {"delta_pp": 0.0853658536585374, "per_seed_delta_pp": [0.2439024390243916, 0.2500000000000002, -0.23780487804877956], "wins": 6, "ties": 18, "losses": 6}, "specificity_td": {"delta_pp": 0.43169904409497417, "per_seed_delta_pp": [0.21276595744680882, 0.22201665124884418, 0.8603145235892695], "wins": 7, "ties": 21, "losses": 2}, "f1_asd": {"delta_pp": 0.26405356551443976, "per_seed_delta_pp": [0.25134416799143633, 0.2475926856558197, 0.2932238428960632], "wins": 11, "ties": 14, "losses": 5}}}, {"left": "E4_FINE_FULL", "right": "E3_FINE_FC", "metrics": {"acc": {"delta_pp": -0.07662835249042137, "per_seed_delta_pp": [0.0, -0.2298850574712652, 1.1102230246251565e-15], "wins": 9, "ties": 12, "losses": 9}, "ba": {"delta_pp": -0.045363183742845134, "per_seed_delta_pp": [0.09077243293247106, -0.2089810699217065, -0.01788091423929994], "wins": 11, "ties": 9, "losses": 10}, "hard_auc": {"delta_pp": -0.045363183742845134, "per_seed_delta_pp": [0.09077243293247106, -0.2089810699217065, -0.01788091423929994], "wins": 11, "ties": 9, "losses": 10}, "probability_auc": {"delta_pp": -0.18752961350150107, "per_seed_delta_pp": [0.2130423501274814, -0.9220403420500456, 0.146409151418061], "wins": 14, "ties": 1, "losses": 15}, "sensitivity_asd": {"delta_pp": 0.339430894308943, "per_seed_delta_pp": [1.249999999999999, 0.012195121951220633, -0.24390243902439046], "wins": 7, "ties": 19, "losses": 4}, "specificity_td": {"delta_pp": -0.43015726179463476, "per_seed_delta_pp": [-1.0684551341350579, -0.4301572617946359, 0.20814061054578947], "wins": 6, "ties": 14, "losses": 10}, "f1_asd": {"delta_pp": -0.03964593692112069, "per_seed_delta_pp": [0.13878553971553576, -0.25918635813284907, 0.0014630076539512427], "wins": 11, "ties": 9, "losses": 10}}}, {"left": "E3_FINE_FC", "right": "E2_OLD_FC", "metrics": {"acc": {"delta_pp": 0.15325670498084273, "per_seed_delta_pp": [0.2298850574712652, 0.11494252873563206, 0.11494252873563092], "wins": 8, "ties": 16, "losses": 6}, "ba": {"delta_pp": 0.12946740822634992, "per_seed_delta_pp": [0.1633367929424001, 0.08218451749734901, 0.1428809142393006], "wins": 10, "ties": 12, "losses": 8}, "hard_auc": {"delta_pp": 0.12946740822634992, "per_seed_delta_pp": [0.1633367929424001, 0.08218451749734901, 0.1428809142393006], "wins": 10, "ties": 12, "losses": 8}, "probability_auc": {"delta_pp": -0.16583936884697317, "per_seed_delta_pp": [-0.2776900340696309, 0.14457029399156163, -0.3643983664628503], "wins": 12, "ties": 0, "losses": 18}, "sensitivity_asd": {"delta_pp": -0.17276422764227584, "per_seed_delta_pp": [-0.5243902439024373, -0.4878048780487809, 0.4939024390243907], "wins": 6, "ties": 17, "losses": 7}, "specificity_td": {"delta_pp": 0.4316990440949745, "per_seed_delta_pp": [0.8510638297872343, 0.6521739130434789, -0.20814061054578947], "wins": 7, "ties": 18, "losses": 5}, "f1_asd": {"delta_pp": 0.1118866421206203, "per_seed_delta_pp": [0.14987206885022863, 0.0414365579121867, 0.1443512995994456], "wins": 10, "ties": 12, "losses": 8}}}, {"left": "E4_FINE_FULL", "right": "E10_COARSE_FULL", "metrics": {"acc": {"delta_pp": -0.19157088122605304, "per_seed_delta_pp": [1.1102230246251565e-15, -0.34482758620689724, -0.22988505747126298], "wins": 9, "ties": 11, "losses": 10}, "ba": {"delta_pp": -0.17442822439325192, "per_seed_delta_pp": [0.025775027639268755, -0.36280487804877914, -0.1862548227702454], "wins": 9, "ties": 11, "losses": 10}, "hard_auc": {"delta_pp": -0.17442822439325192, "per_seed_delta_pp": [0.025775027639268755, -0.36280487804877914, -0.1862548227702454], "wins": 9, "ties": 11, "losses": 10}, "probability_auc": {"delta_pp": -0.023609951640683653, "per_seed_delta_pp": [0.5162451208230856, -0.9379921481916043, 0.3509171724464677], "wins": 16, "ties": 2, "losses": 12}, "sensitivity_asd": {"delta_pp": 0.08130081300812979, "per_seed_delta_pp": [0.48170731707317005, -0.7256097560975605, 0.4878048780487798], "wins": 10, "ties": 16, "losses": 4}, "specificity_td": {"delta_pp": -0.4301572617946344, "per_seed_delta_pp": [-0.43015726179463254, -1.1102230246251565e-15, -0.8603145235892695], "wins": 4, "ties": 19, "losses": 7}, "f1_asd": {"delta_pp": -0.1918128603149401, "per_seed_delta_pp": [0.03731344057432806, -0.46534248587648214, -0.14740953564266635], "wins": 9, "ties": 11, "losses": 10}}}, {"left": "E10_COARSE_FULL", "right": "E3_FINE_FC", "metrics": {"acc": {"delta_pp": 0.11494252873563168, "per_seed_delta_pp": [-1.1102230246251565e-15, 0.11494252873563206, 0.22988505747126411], "wins": 9, "ties": 16, "losses": 5}, "ba": {"delta_pp": 0.1290650406504068, "per_seed_delta_pp": [0.0649974052932023, 0.15382380812707264, 0.16837390853094547], "wins": 12, "ties": 12, "losses": 6}, "hard_auc": {"delta_pp": 0.1290650406504068, "per_seed_delta_pp": [0.0649974052932023, 0.15382380812707264, 0.16837390853094547], "wins": 12, "ties": 12, "losses": 6}, "probability_auc": {"delta_pp": -0.1639196618608174, "per_seed_delta_pp": [-0.3032027706956042, 0.015951806141558666, -0.2045080210284067], "wins": 14, "ties": 0, "losses": 16}, "sensitivity_asd": {"delta_pp": 0.2581300813008133, "per_seed_delta_pp": [0.7682926829268288, 0.7378048780487811, -0.7317073170731703], "wins": 7, "ties": 18, "losses": 5}, "specificity_td": {"delta_pp": -3.7007434154171886e-16, "per_seed_delta_pp": [-0.6382978723404253, -0.43015726179463476, 1.068455134135059], "wins": 6, "ties": 19, "losses": 5}, "f1_asd": {"delta_pp": 0.15216692339381943, "per_seed_delta_pp": [0.1014720991412077, 0.20615612774363298, 0.1488725432966176], "wins": 12, "ties": 12, "losses": 6}}}, {"left": "E10_COARSE_FULL", "right": "E4_FINE_FULL", "metrics": {"acc": {"delta_pp": 0.19157088122605304, "per_seed_delta_pp": [-1.1102230246251565e-15, 0.34482758620689724, 0.22988505747126298], "wins": 10, "ties": 11, "losses": 9}, "ba": {"delta_pp": 0.17442822439325192, "per_seed_delta_pp": [-0.025775027639268755, 0.36280487804877914, 0.1862548227702454], "wins": 10, "ties": 11, "losses": 9}, "hard_auc": {"delta_pp": 0.17442822439325192, "per_seed_delta_pp": [-0.025775027639268755, 0.36280487804877914, 0.1862548227702454], "wins": 10, "ties": 11, "losses": 9}, "probability_auc": {"delta_pp": 0.023609951640683653, "per_seed_delta_pp": [-0.5162451208230856, 0.9379921481916043, -0.3509171724464677], "wins": 12, "ties": 2, "losses": 16}, "sensitivity_asd": {"delta_pp": -0.08130081300812979, "per_seed_delta_pp": [-0.48170731707317005, 0.7256097560975605, -0.4878048780487798], "wins": 4, "ties": 16, "losses": 10}, "specificity_td": {"delta_pp": 0.4301572617946344, "per_seed_delta_pp": [0.43015726179463254, 1.1102230246251565e-15, 0.8603145235892695], "wins": 7, "ties": 19, "losses": 4}, "f1_asd": {"delta_pp": 0.1918128603149401, "per_seed_delta_pp": [-0.03731344057432806, 0.46534248587648214, 0.14740953564266635], "wins": 10, "ties": 11, "losses": 9}}}]。

2×2交互：{"acc": {"delta_pp": -0.34482758620689574, "per_seed_delta_pp": [-0.22988505747126411, -0.45977011494252923, -0.3448275862068939], "positive": 6, "zero": 13, "negative": 11}, "ba": {"delta_pp": -0.30389563261960184, "per_seed_delta_pp": [-0.13756176530313136, -0.4449893955461281, -0.329135737009546], "positive": 10, "zero": 7, "negative": 13}, "hard_auc": {"delta_pp": -0.30389563261960184, "per_seed_delta_pp": [-0.13756176530313136, -0.4449893955461281, -0.329135737009546], "positive": 10, "zero": 7, "negative": 13}, "probability_auc": {"delta_pp": 0.14222941720628954, "per_seed_delta_pp": [0.7939351548927165, -1.0825624421831659, 0.715315538909318], "positive": 18, "zero": 0, "negative": 12}, "sensitivity_asd": {"delta_pp": 0.25406504065040564, "per_seed_delta_pp": [1.0060975609756073, -0.23780487804877956, -0.006097560975610872], "positive": 9, "zero": 14, "negative": 7}, "specificity_td": {"delta_pp": -0.8618563058896085, "per_seed_delta_pp": [-1.2812210915818667, -0.6521739130434789, -0.65217391304348], "positive": 6, "zero": 12, "negative": 12}, "f1_asd": {"delta_pp": -0.30369950243556043, "per_seed_delta_pp": [-0.11255862827590057, -0.5067790437886688, -0.29176083524211194], "positive": 11, "zero": 6, "negative": 13}}。

结论：{"aux_coarse_supported": false, "aux_fine_supported": false, "fine_with_full_aux_supported": false, "historical_E4_conclusion": "NO-GO UNCHANGED", "criterion": "BA/probAUC/SEN each positive in mean and every seed; descriptive internal consistency only"}。粗粒度教师近恒定，不能将辅助收益归因医学知识；原实验28 E4的NO-GO不变。

GPU任务与耗时：[{"config": "E10_COARSE_FULL", "seed": 0, "gpu": 0, "folds": 10, "status": "COMPLETE", "completed_folds": 10, "seconds": 695.9572685696185, "resumes": 0}, {"config": "E10_COARSE_FULL", "seed": 1, "gpu": 1, "folds": 10, "status": "COMPLETE", "completed_folds": 10, "seconds": 676.8862256519496, "resumes": 0}, {"config": "E10_COARSE_FULL", "seed": 2, "gpu": 0, "folds": 10, "status": "COMPLETE", "completed_folds": 10, "seconds": 730.6162389665842, "resumes": 0}]；总墙钟秒 808.959539。

核心源码SHA {"main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67", "ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763"}；缓存SHA 2a3cf9332ad3d1dbdba70731a46acee43f3f7e52c92fe3ab2366fa90b0d54420；历史保护 {"status": "UNCHANGED_SHA256", "files": 1028}。预检 {"passed": true, "completed_utc": "2026-10-20T04:02:10.921241+00:00", "checks": {"P0_unpatched_init_forward_two_adam_first_update_rng": "EXACT_PASS", "P0_seed0_fold1_initial_hash": "b464f86ee750e5550a6330942ee102a5d9d1f3ab4967a385e23655741c92ae0f", "E10_E4_structure_original_new_parameters_optimizer_rng": "BITWISE_EXACT", "E10_E1_E4_zero_alpha_output": "BITWISE_EXACT", "E10_E2_same_student_semantic_loss": "BITWISE_EXACT", "total_parameters": 1491474, "additional_parameters": 28517, "heldout_label_flip_full_train_update_models_both_adams_rng": "BITWISE_EXACT", "first_step_gradient_norms": {"gfr": 1.7359182958520136, "pln_original": 2.0574789457950238, "alpha": 0.004164975586912834, "fc_encoder": 0.0015731278535086523, "context_encoder": 0.00018808754404105524, "demo_encoder": 0.0, "anat_gate": 0.0, "func_gate": 0.0, "decode": 0.0, "active_parameters": 751311}, "second_step_gradient_norms": {"gfr": 1.5987937536028982, "pln_original": 3.4424123207985153, "alpha": 0.00016862177183014633, "fc_encoder": 0.0005641556965762557, "context_encoder": 0.00022434364488380088, "demo_encoder": 9.49557618791706e-06, "anat_gate": 2.815128581226507e-07, "func_gate": 2.786521469425497e-07, "decode": 0.0002703180729481091, "active_parameters": 760755}, "auxiliary_conditional_reaches_classification": "PASS", "disk_checkpoint_model_two_adams_full_rng_next_update_prediction": "BITWISE_EXACT", "next_fold_initialization_rng_resume": "BITWISE_EXACT", "device": "_CudaDeviceProperties(name='PPU-ZW810E', major=8, minor=0, total_memory=98304MB, multi_processor_count=64, uuid=019e4201-0591-0030-0000-000080387166, pci_bus_id=99, pci_device_id=0, pci_domain_id=0, L2_cache_size=64MB)", "all10fold_E10_E2_teacher_active_weights_loss": "BITWISE_EXACT", "all10fold_E10_E4_parameters_aux_forward_nonzero_alpha": "BITWISE_EXACT", "teacher_forward_semantic_loss_source": "BYTE_IDENTICAL_FUNCTIONS", "frozen_coarse_cache": "REUSED_NO_CLIP_ENCODING", "GPU1_visible_logical0": "PPU-ZW810E"}}。

报告 /tmp/experiment28_supplement_e10_coarse_full/results/report.md。保留test=val、前CV RFE及全图transductive的历史乐观偏差；后续仅建议独立队列/训练内选模验证，不追加追分参数。无自动Git提交或推送。


<!-- EXPERIMENT28_SUPPLEMENT_E10_IO_RECOVERED -->
### I/O恢复、交付完成与GitHub归档

更新日期：2026-10-20T05:48:00.853249+00:00。I/O修复后已将此前暂存的E10补充实验记录追加至项目根handoff.md，并完成写入、fsync和完整历史前缀SHA核验。新增训练仍为3任务/30折，未重新训练历史300折。

关键判读：E10相对E2的ACC/BA分别提高0.26820/0.25853个百分点，但概率AUC降低0.32976个百分点；E4相对E10的ACC/BA/概率AUC分别为−0.19157/−0.17443/−0.02361个百分点。ACC和BA的2×2交互分别为−0.34483/−0.30390个百分点，三个seed均为负；概率AUC交互均值+0.14223个百分点，但seed方向不一致。当前证据不支持细粒度Prompt在相同辅助结构下提供稳定独立增益，实验28原NO-GO结论不变。

GitHub归档：此前按用户明确上传指令，将补充实验目录13个文件上传至https://github.com/tangaii/SPromptGL/tree/5e8049b40260b3090cddfeea94c7820b5b3c2ed0/experiment/experiment28_supplement_e10_coarse_full；提交5e8049b40260b3090cddfeea94c7820b5b3c2ed0，各文件SHA256核验一致。该提交保留当时handoff待I/O恢复的历史状态。本次更新完成本地handoff及对应交付元数据同步，尚未再次提交或推送GitHub。


<!-- EXPERIMENT29_CONNECTIVITY_SEMANTIC_SELECTIVE_AUXILIARY_FINAL -->
## 实验二十九：CORR主导的双层疾病语义对齐与选择性辅助修正

更新时间：2026-10-20T07:59:45.926015+00:00。STATUS=COMPLETE；NO-GO。

本轮问题：真实疾病语义的类别对齐与真实FC连接身份匹配是否增加诊断信息；CORR条件化软稀疏辅助是否优于Dense及同站点错误患者配对。原SPromptGL已有GFR/CAM/SPGS/RES，MMGPL已有概念加权/Token图；实验28平均教师高度同向且文本不直接进入推理，本轮保留主干但以原始CORR64维Anchor、全局CE与局部InfoNCE及推理语义响应替代。

精确方法：{"semantic_dim": 64, "top_k": 16, "temperature": 0.2, "lambda_global": 0.05, "lambda_local": 0.02, "lambda_sparse": 0.005, "residual_coordinate_bound": 0.1, "new_weight_decay": 0.0, "init_seed": 290000, "edge_id_dim": 16, "query_input": 83, "query_hidden": 64, "semantic_hidden": 64, "stability": "unweighted site sign agreement / (1 + weighted site effect heterogeneity)", "candidate": "ReLU(sign(d)*r - .5) * clip(abs(d),0,2) * stability; stable descending top16; active iff >0", "attention": "cos(query,key)/.2 + tanh(candidate_evidence)", "gating": "continuous sigmoid soft sparse; no hard threshold; L_sparse=(mean_feature_gate+mean_router)/2 over active modalities, outer TRAIN only", "residual": "0.10*tanh(tanh(alpha_sem)*semantic_decode + tanh(alpha_aux)*aux_decode)", "local_loss": "Within-patient selected-set InfoNCE; valid train connections only; repeated coarse keys imply constant log(candidate_count) and zero useful query gradient", "fc_encoder": "256->64 + LayerNorm + GELU; no CAM or auxiliary input to semantic losses", "auxiliary": "independent selectors [h_fc,x]->dim; biasless dim->64 GELU encoder; router [h,u,h*u]->1 sigmoid; biasless 64->64 value; biasless concat192->36 decoder"}。

20组冻结设计：{"C0": {"fine": false, "semantic": true, "auxiliary": [], "sparse": false, "shuffled": false, "text": "REAL"}, "C1": {"fine": false, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": false, "shuffled": false, "text": "REAL"}, "C2": {"fine": false, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "REAL"}, "C3": {"fine": false, "semantic": false, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "REAL"}, "C4": {"fine": false, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": true, "text": "REAL"}, "C5": {"fine": false, "semantic": true, "auxiliary": ["PHENO"], "sparse": true, "shuffled": false, "text": "REAL"}, "C6": {"fine": false, "semantic": true, "auxiliary": ["ANAT"], "sparse": true, "shuffled": false, "text": "REAL"}, "C7": {"fine": false, "semantic": true, "auxiliary": ["FUNC"], "sparse": true, "shuffled": false, "text": "REAL"}, "C8": {"fine": false, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "NUMERIC"}, "F0": {"fine": true, "semantic": true, "auxiliary": [], "sparse": false, "shuffled": false, "text": "REAL"}, "F1": {"fine": true, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": false, "shuffled": false, "text": "REAL"}, "F2": {"fine": true, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "REAL"}, "F3": {"fine": true, "semantic": false, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "REAL"}, "F4": {"fine": true, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": true, "text": "REAL"}, "F5": {"fine": true, "semantic": true, "auxiliary": ["PHENO"], "sparse": true, "shuffled": false, "text": "REAL"}, "F6": {"fine": true, "semantic": true, "auxiliary": ["ANAT"], "sparse": true, "shuffled": false, "text": "REAL"}, "F7": {"fine": true, "semantic": true, "auxiliary": ["FUNC"], "sparse": true, "shuffled": false, "text": "REAL"}, "F8": {"fine": true, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "WRONG_ROI"}, "F9": {"fine": true, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "SWAP_DIRECTION"}, "F10": {"fine": true, "semantic": true, "auxiliary": ["PHENO", "ANAT", "FUNC"], "sparse": true, "shuffled": false, "text": "NUMERIC"}}。

正式完成20/20任务、200/200fold；seed仅0，每任务连续十折；两GPU完成fold数{"0": 100, "1": 100}；正式墙钟秒1980.191423。历史P0/E1只读，不重复训练。

全部四指标（每组seed0十折均值±population SD，比例值）：{"P0": {"acc": {"mean": 0.9172936259143156, "sd_population": 0.029505272171989984}, "auc": {"mean": 0.9156249858983327, "sd_population": 0.029444404947036207}, "sen_asd": {"mean": 0.8934146339245939, "sd_population": 0.0366765124677748}, "spe_td": {"mean": 0.9378353374499598, "sd_population": 0.03674878136204607}}, "E1": {"acc": {"mean": 0.9126959247648904, "sd_population": 0.031421413426393055}, "auc": {"mean": 0.9108974583154712, "sd_population": 0.030777916250821154}, "sen_asd": {"mean": 0.8883536583161179, "sd_population": 0.03732031110470889}, "spe_td": {"mean": 0.9334412578949582, "sd_population": 0.04992138002127053}}, "C0": {"acc": {"mean": 0.9081112852664578, "sd_population": 0.026297437508786776}, "auc": {"mean": 0.9058438437760881, "sd_population": 0.025034828569596934}, "sen_asd": {"mean": 0.8782926827088919, "sd_population": 0.043577046161031065}, "spe_td": {"mean": 0.933395004425968, "sd_population": 0.05946116861717495}}, "C1": {"acc": {"mean": 0.8908829676071056, "sd_population": 0.033507562225153914}, "auc": {"mean": 0.8886335766341011, "sd_population": 0.03313396820074038}, "sen_asd": {"mean": 0.8587195119819737, "sd_population": 0.04092281186489906}, "spe_td": {"mean": 0.9185476408768505, "sd_population": 0.04630083143134144}}, "C2": {"acc": {"mean": 0.9000914315569487, "sd_population": 0.027291565701158583}, "auc": {"mean": 0.8976880034295256, "sd_population": 0.027979096823506806}, "sen_asd": {"mean": 0.8660975607606469, "sd_population": 0.049888483599551614}, "spe_td": {"mean": 0.929278445684908, "sd_population": 0.036355609136584394}}, "C3": {"acc": {"mean": 0.9184430512016719, "sd_population": 0.028892371995820263}, "auc": {"mean": 0.9169286004377157, "sd_population": 0.029110594625625066}, "sen_asd": {"mean": 0.8959756095336948, "sd_population": 0.043679311028620695}, "spe_td": {"mean": 0.93788159091895, "sd_population": 0.0365829853312548}}, "C4": {"acc": {"mean": 0.902364158829676, "sd_population": 0.03269314544551906}, "auc": {"mean": 0.9006950993885517, "sd_population": 0.03141172204485268}, "sen_asd": {"mean": 0.8808536583179929, "sd_population": 0.03269930422524085}, "spe_td": {"mean": 0.920536540043895, "sd_population": 0.05674899165727291}}, "C5": {"acc": {"mean": 0.9046630094043886, "sd_population": 0.03380129415852216}, "auc": {"mean": 0.9020557128674895, "sd_population": 0.03493402447153708}, "sen_asd": {"mean": 0.8639634144196279, "sd_population": 0.07000013276787259}, "spe_td": {"mean": 0.9401480108999364, "sd_population": 0.04133735432328575}}, "C6": {"acc": {"mean": 0.909260710553814, "sd_population": 0.0362082559368514}, "auc": {"mean": 0.9079858024412808, "sd_population": 0.035034132291310374}, "sen_asd": {"mean": 0.8933536583148681, "sd_population": 0.03146890904124111}, "spe_td": {"mean": 0.9226179461489201, "sd_population": 0.05945974741005344}}, "C7": {"acc": {"mean": 0.904649947753396, "sd_population": 0.03643746483913649}, "auc": {"mean": 0.9018643814444619, "sd_population": 0.03647892909980789}, "sen_asd": {"mean": 0.8637195119807238, "sd_population": 0.04243153317936547}, "spe_td": {"mean": 0.9400092504929655, "sd_population": 0.03579435803004943}}, "C8": {"acc": {"mean": 0.9069618599791012, "sd_population": 0.03276470566644689}, "auc": {"mean": 0.9043643814444622, "sd_population": 0.0321992911259094}, "sen_asd": {"mean": 0.8687195119794738, "sd_population": 0.04077171431215434}, "spe_td": {"mean": 0.9400092504929655, "sd_population": 0.046846361924799115}}, "F0": {"acc": {"mean": 0.9058124346917451, "sd_population": 0.030835149937132385}, "auc": {"mean": 0.904544967396945, "sd_population": 0.030735954749873562}, "sen_asd": {"mean": 0.888414633925844, "sd_population": 0.045751215339677874}, "spe_td": {"mean": 0.9206753004508659, "sd_population": 0.04674267196623808}}, "F1": {"acc": {"mean": 0.8908568443051201, "sd_population": 0.03616616199807024}, "auc": {"mean": 0.890957164775163, "sd_population": 0.035813493481723135}, "sen_asd": {"mean": 0.893292682705142, "sd_population": 0.044092576061845364}, "spe_td": {"mean": 0.8886216464336641, "sd_population": 0.05020741089429996}}, "F2": {"acc": {"mean": 0.9035266457680249, "sd_population": 0.036116738069965695}, "auc": {"mean": 0.9012055797477494, "sd_population": 0.035712272760177674}, "sen_asd": {"mean": 0.8710975607593969, "sd_population": 0.0359282449577086}, "spe_td": {"mean": 0.9313135983209427, "sd_population": 0.04335371477155311}}, "F3": {"acc": {"mean": 0.9184430512016719, "sd_population": 0.028892371995820263}, "auc": {"mean": 0.9169286004377157, "sd_population": 0.029110594625625066}, "sen_asd": {"mean": 0.8959756095336948, "sd_population": 0.043679311028620695}, "spe_td": {"mean": 0.93788159091895, "sd_population": 0.0365829853312548}}, "F4": {"acc": {"mean": 0.8989158829676072, "sd_population": 0.030849753243132622}, "auc": {"mean": 0.8979222039213918, "sd_population": 0.031301979767658916}, "sen_asd": {"mean": 0.885853658316743, "sd_population": 0.04741048736084083}, "spe_td": {"mean": 0.9099907491117986, "sd_population": 0.03885027081625861}}, "F5": {"acc": {"mean": 0.9035135841170323, "sd_population": 0.032263708356091214}, "auc": {"mean": 0.9018103438550575, "sd_population": 0.03291988251744757}, "sen_asd": {"mean": 0.8785975607575219, "sd_population": 0.049563419662114444}, "spe_td": {"mean": 0.9250231265368771, "sd_population": 0.032617354720642235}}, "F6": {"acc": {"mean": 0.9081243469174505, "sd_population": 0.03174514362810706}, "auc": {"mean": 0.9056679959838452, "sd_population": 0.03272806356700432}, "sen_asd": {"mean": 0.8712804875885748, "sd_population": 0.05328685712092282}, "spe_td": {"mean": 0.9400555039619558, "sd_population": 0.025201307745625994}}, "F7": {"acc": {"mean": 0.9035135841170323, "sd_population": 0.03538835169206356}, "auc": {"mean": 0.9013234132803862, "sd_population": 0.035543644607291695}, "sen_asd": {"mean": 0.8734146339295938, "sd_population": 0.0499048767849903}, "spe_td": {"mean": 0.9292321922159177, "sd_population": 0.04235262227340977}}, "F8": {"acc": {"mean": 0.9035135841170323, "sd_population": 0.03575974000612225}, "auc": {"mean": 0.9017387637914307, "sd_population": 0.03523776702553811}, "sen_asd": {"mean": 0.8762804875873249, "sd_population": 0.04452760340131206}, "spe_td": {"mean": 0.927197039579883, "sd_population": 0.049075867876760874}}, "F9": {"acc": {"mean": 0.9035005224660397, "sd_population": 0.03463603091278066}, "auc": {"mean": 0.9023152399539722, "sd_population": 0.03452171421193343}, "sen_asd": {"mean": 0.8860365851459211, "sd_population": 0.04384095456318793}, "spe_td": {"mean": 0.9185938943458407, "sd_population": 0.04300975756758576}}, "F10": {"acc": {"mean": 0.9023641588296762, "sd_population": 0.03929969485834418}, "auc": {"mean": 0.9008001286072064, "sd_population": 0.039958351667336536}, "sen_asd": {"mean": 0.8786585363672479, "sd_population": 0.06424931453210478}, "spe_td": {"mean": 0.9229417204318523, "sd_population": 0.045324246057837234}}}。

全部预设配对（pp及逐折胜/平/负/差）：[{"left": "F0", "right": "C0", "metrics": {"acc": {"delta_pp": -0.2298850574712663, "wins": 2, "ties": 5, "losses": 3, "per_fold_delta_pp": [0.0, -1.1494252873563315, -1.1494252873563315, 0.0, 0.0, 1.1494252873563204, 1.1494252873563204, 0.0, -2.298850574712641, 0.0]}, "auc": {"delta_pp": -0.12988763791430658, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [0.15568240788791377, -1.3520678685047693, -1.21951219512193, 0.37234042553192515, -0.18617021276594592, 1.4361702127659415, 1.6223404255318985, 0.0, -1.9414893617021312, -0.18617021276596812]}, "sen_asd": {"delta_pp": 1.0121951216951963, "wins": 5, "ties": 1, "losses": 4, "per_fold_delta_pp": [2.439024389649014, -4.878048779298039, -2.439024389649014, 4.999999998750004, -2.499999999375002, 4.999999998749994, 7.4999999981250065, 0.0, 2.499999999375002, -2.499999999375002]}, "spe_td": {"delta_pp": -1.271970397510207, "wins": 3, "ties": 2, "losses": 5, "per_fold_delta_pp": [-2.1276595740153836, 2.173913043005671, 0.0, -4.2553191480307895, 2.1276595740153836, -2.1276595740153947, -4.255319148030779, 0.0, -6.382978722046174, 2.1276595740153947]}}}, {"left": "F1", "right": "C1", "metrics": {"acc": {"delta_pp": -0.002612330198535906, "wins": 2, "ties": 5, "losses": 3, "per_fold_delta_pp": [2.272727272727282, 2.298850574712652, -1.1494252873563204, 0.0, 0.0, -1.1494252873563204, 0.0, 0.0, -2.298850574712652, 0.0]}, "auc": {"delta_pp": 0.2323588141061805, "wins": 5, "ties": 2, "losses": 3, "per_fold_delta_pp": [2.5947067981318073, 2.5715800636267216, -0.9544008483563071, 0.18617021276596812, 0.744680851063817, -0.8776595744680815, 0.0, 0.18617021276596812, -2.127659574468088, 0.0]}, "sen_asd": {"delta_pp": 3.4573170723168136, "wins": 7, "ties": 3, "losses": 0, "per_fold_delta_pp": [7.317073168947063, 7.3170731689470525, 2.439024389649025, 2.499999999375002, 9.999999997499998, 2.499999999374991, 0.0, 2.499999999375002, 0.0, 0.0]}, "spe_td": {"delta_pp": -2.9925994443186332, "wins": 0, "ties": 2, "losses": 8, "per_fold_delta_pp": [-2.1276595740153836, -2.173913043005671, -4.347826086011342, -2.1276595740153947, -8.510638296061568, -4.2553191480307895, 0.0, -2.1276595740153947, -4.2553191480307895, 0.0]}}}, {"left": "F2", "right": "C2", "metrics": {"acc": {"delta_pp": 0.34352142110762873, "wins": 6, "ties": 1, "losses": 3, "per_fold_delta_pp": [1.1363636363636354, 1.1494252873563315, -4.597701149425292, -2.298850574712641, -2.298850574712641, 2.298850574712652, 4.597701149425282, 1.1494252873563204, 2.298850574712641, 0.0]}, "auc": {"delta_pp": 0.35175763182238473, "wins": 6, "ties": 0, "losses": 4, "per_fold_delta_pp": [1.2195121951219412, 1.3520678685047693, -4.74549310710497, -2.3138297872340674, -2.6861702127659592, 2.313829787234012, 4.62765957446808, 1.2500000000000067, 2.8723404255319274, -0.37234042553189184]}, "sen_asd": {"delta_pp": 0.4999999998749993, "wins": 6, "ties": 0, "losses": 4, "per_fold_delta_pp": [2.439024389649025, 4.878048779298039, -7.3170731689470525, -2.499999999375002, -7.4999999981250065, 2.499999999374991, 4.999999998750004, 2.499999999375002, 9.999999997499998, -4.999999998750004]}, "spe_td": {"delta_pp": 0.20351526360348318, "wins": 4, "ties": 2, "losses": 4, "per_fold_delta_pp": [0.0, -2.173913043005671, -2.173913043005671, -2.1276595740153947, 2.1276595740153947, 2.1276595740153947, 4.2553191480307895, 0.0, -4.2553191480307895, 4.255319148030779]}}}, {"left": "C2", "right": "C0", "metrics": {"acc": {"delta_pp": -0.8019853709508895, "wins": 1, "ties": 4, "losses": 5, "per_fold_delta_pp": [-2.272727272727282, -1.1494252873563315, 0.0, 1.1494252873563204, 0.0, 0.0, -1.1494252873563204, 0.0, -3.4482758620689613, -1.1494252873563204]}, "auc": {"delta_pp": -0.8155840346562582, "wins": 3, "ties": 1, "losses": 6, "per_fold_delta_pp": [-2.283341982355991, -1.8822905620360597, 0.2651113467656341, 1.4361702127659859, 0.18617021276596812, 0.0, -1.0638297872340496, -0.18617021276595702, -3.750000000000009, -0.8776595744681037]}, "sen_asd": {"delta_pp": -1.2195121948245091, "wins": 4, "ties": 2, "losses": 4, "per_fold_delta_pp": [-2.439024389649025, -14.634146337894116, 4.878048779298039, 4.999999998750004, 2.499999999375002, 0.0, 0.0, -2.499999999375002, -7.499999998124995, 2.499999999375002]}, "spe_td": {"delta_pp": -0.4116558741059939, "wins": 2, "ties": 2, "losses": 6, "per_fold_delta_pp": [-2.1276595740153836, 10.869565215028354, -4.347826086011342, -2.1276595740153947, -2.1276595740153947, 0.0, -2.1276595740153947, 2.1276595740153947, 0.0, -4.255319148030779]}}}, {"left": "F2", "right": "F0", "metrics": {"acc": {"delta_pp": -0.22857889237199447, "wins": 5, "ties": 0, "losses": 5, "per_fold_delta_pp": [-1.1363636363636465, 1.1494252873563315, -3.4482758620689613, -1.1494252873563204, -2.298850574712641, 1.1494252873563315, 2.298850574712641, 1.1494252873563204, 1.1494252873563204, -1.1494252873563204]}, "auc": {"delta_pp": -0.33393876491956687, "wins": 5, "ties": 0, "losses": 5, "per_fold_delta_pp": [-1.2195121951219634, 0.821845174973479, -3.260869565217406, -1.2500000000000067, -2.313829787234045, 0.8776595744680704, 1.9414893617021312, 1.0638297872340496, 1.0638297872340496, -1.0638297872340274]}, "sen_asd": {"delta_pp": -1.731707316644706, "wins": 0, "ties": 4, "losses": 6, "per_fold_delta_pp": [-2.439024389649014, -4.878048779298039, 0.0, -2.499999999375002, -2.499999999375002, -2.499999999375002, -2.499999999375002, 0.0, 0.0, 0.0]}, "spe_td": {"delta_pp": 1.0638297870076963, "wins": 5, "ties": 2, "losses": 3, "per_fold_delta_pp": [0.0, 6.521739129017012, -6.521739129017012, 0.0, -2.1276595740153836, 4.2553191480307895, 6.382978722046174, 2.1276595740153947, 2.1276595740153836, -2.1276595740153947]}}}, {"left": "C2", "right": "C1", "metrics": {"acc": {"delta_pp": 0.920846394984326, "wins": 5, "ties": 2, "losses": 3, "per_fold_delta_pp": [-1.1363636363636354, 1.1494252873563204, 3.4482758620689724, 3.4482758620689613, 3.4482758620689613, -1.1494252873563204, -2.298850574712641, 0.0, 0.0, 2.298850574712641]}, "auc": {"delta_pp": 0.9054426795424309, "wins": 5, "ties": 0, "losses": 5, "per_fold_delta_pp": [-1.0638297872340385, 0.821845174973479, 3.791092258748663, 3.377659574468117, 4.122340425531911, -1.0638297872340163, -2.500000000000002, -0.18617021276595702, -0.7446808510638392, 2.499999999999991]}, "sen_asd": {"delta_pp": 0.7378048778673041, "wins": 4, "ties": 2, "losses": 4, "per_fold_delta_pp": [0.0, -4.878048779298039, 9.756097558596078, 2.499999999375002, 12.499999996875, 0.0, -4.999999998750004, -2.499999999375002, -9.999999997499998, 4.999999998750004]}, "spe_td": {"delta_pp": 1.0730804808057526, "wins": 4, "ties": 2, "losses": 4, "per_fold_delta_pp": [-2.1276595740153836, 6.521739129017012, -2.173913043005671, 4.255319148030779, -4.255319148030779, -2.1276595740153947, 0.0, 2.1276595740153947, 8.510638296061568, 0.0]}}}, {"left": "F2", "right": "F1", "metrics": {"acc": {"delta_pp": 1.2669801462904906, "wins": 7, "ties": 2, "losses": 1, "per_fold_delta_pp": [-2.272727272727282, 0.0, 0.0, 1.1494252873563204, 1.1494252873563204, 2.298850574712652, 2.298850574712641, 1.1494252873563204, 4.597701149425292, 2.298850574712641]}, "auc": {"delta_pp": 1.024841497258635, "wins": 7, "ties": 1, "losses": 2, "per_fold_delta_pp": [-2.4390243902439046, -0.3976670201484733, 0.0, 0.8776595744680815, 0.6914893617021356, 2.127659574468077, 2.127659574468077, 0.8776595744680815, 4.255319148936176, 2.1276595744680993]}, "sen_asd": {"delta_pp": -2.2195121945745098, "wins": 0, "ties": 5, "losses": 5, "per_fold_delta_pp": [-4.878048779298039, -7.3170731689470525, 0.0, -2.499999999375002, -4.999999998750004, 0.0, 0.0, -2.499999999375002, 0.0, 0.0]}, "spe_td": {"delta_pp": 4.269195188727869, "wins": 8, "ties": 2, "losses": 0, "per_fold_delta_pp": [0.0, 6.521739129017012, 0.0, 4.255319148030779, 6.382978722046184, 4.2553191480307895, 4.2553191480307895, 4.2553191480307895, 8.510638296061568, 4.255319148030779]}}}, {"left": "C2", "right": "C3", "metrics": {"acc": {"delta_pp": -1.8351619644723105, "wins": 1, "ties": 1, "losses": 8, "per_fold_delta_pp": [-3.409090909090917, -2.298850574712652, 0.0, 1.1494252873563204, -1.1494252873563204, -2.298850574712652, -4.597701149425282, -1.1494252873563204, -2.298850574712641, -2.298850574712641]}, "auc": {"delta_pp": -1.9240597008190217, "wins": 2, "ties": 0, "losses": 8, "per_fold_delta_pp": [-3.502854177477943, -2.4390243902439046, 0.2651113467656341, 1.4361702127659859, -1.2499999999999845, -2.313829787234012, -5.18617021276595, -1.2500000000000067, -2.6861702127659814, -2.3138297872340563]}, "sen_asd": {"delta_pp": -2.9878048773048014, "wins": 2, "ties": 0, "losses": 8, "per_fold_delta_pp": [-4.878048779298039, -4.878048779298039, 4.878048779298039, 4.999999998750004, -2.499999999374991, -2.499999999374991, -12.499999996875, -2.499999999375002, -7.499999998124995, -2.499999999375002]}, "spe_td": {"delta_pp": -0.860314523404212, "wins": 2, "ties": 3, "losses": 5, "per_fold_delta_pp": [-2.1276595740153836, 0.0, -4.347826086011342, -2.1276595740153947, 0.0, -2.1276595740153947, 2.1276595740153836, 0.0, 2.1276595740153947, -2.1276595740153836]}}}, {"left": "F2", "right": "F3", "metrics": {"acc": {"delta_pp": -1.4916405433646818, "wins": 0, "ties": 4, "losses": 6, "per_fold_delta_pp": [-2.272727272727282, -1.1494252873563204, -4.597701149425292, -1.1494252873563204, -3.4482758620689613, 0.0, 0.0, 0.0, 0.0, -2.298850574712641]}, "auc": {"delta_pp": -1.5723020689966374, "wins": 1, "ties": 2, "losses": 7, "per_fold_delta_pp": [-2.283341982356002, -1.0869565217391353, -4.480381760339336, -0.8776595744680815, -3.9361702127659437, 0.0, -0.5585106382978711, 0.0, 0.18617021276594592, -2.686170212765948]}, "sen_asd": {"delta_pp": -2.4878048774298023, "wins": 2, "ties": 3, "losses": 5, "per_fold_delta_pp": [-2.439024389649014, 0.0, -2.439024389649014, 2.499999999375002, -9.999999997499998, 0.0, -7.499999998124995, 0.0, 2.499999999375002, -7.4999999981250065]}, "spe_td": {"delta_pp": -0.6567992598007288, "wins": 3, "ties": 2, "losses": 5, "per_fold_delta_pp": [-2.1276595740153836, -2.173913043005671, -6.521739129017012, -4.2553191480307895, 2.1276595740153947, 0.0, 6.382978722046174, 0.0, -2.1276595740153947, 2.1276595740153947]}}}, {"left": "C2", "right": "C4", "metrics": {"acc": {"delta_pp": -0.22727272727272818, "wins": 3, "ties": 3, "losses": 4, "per_fold_delta_pp": [-2.272727272727282, -2.298850574712652, 4.597701149425292, 3.4482758620689613, 1.1494252873563204, 0.0, -3.4482758620689613, 0.0, -3.4482758620689613, 0.0]}, "auc": {"delta_pp": -0.30070959590262514, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [-2.283341982355991, -2.836691410392378, 4.612937433722153, 3.750000000000009, 1.4361702127659637, 0.0, -3.563829787234052, -0.37234042553192515, -3.936170212765977, 0.18617021276594592]}, "sen_asd": {"delta_pp": -1.475609755734607, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [-2.439024389649025, -12.195121948245092, 4.878048779298039, 7.4999999981250065, 4.999999998750004, 0.0, -4.999999998750004, -4.999999998750004, -9.999999997499998, 2.499999999375002]}, "spe_td": {"delta_pp": 0.8741905641012981, "wins": 4, "ties": 2, "losses": 4, "per_fold_delta_pp": [-2.1276595740153836, 6.521739129017012, 4.347826086011342, 0.0, -2.1276595740153947, 0.0, -2.1276595740153947, 4.2553191480307895, 2.1276595740153947, -2.1276595740153836]}}}, {"left": "F2", "right": "F4", "metrics": {"acc": {"delta_pp": 0.46107628004179674, "wins": 5, "ties": 1, "losses": 4, "per_fold_delta_pp": [-1.1363636363636465, -1.1494252873563204, -1.1494252873563204, 2.298850574712641, -1.1494252873563204, 2.298850574712652, 0.0, 1.1494252873563204, 1.1494252873563204, 2.298850574712641]}, "auc": {"delta_pp": 0.3283375826357682, "wins": 5, "ties": 0, "losses": 5, "per_fold_delta_pp": [-1.2195121951219634, -1.3520678685047804, -1.2195121951219523, 2.499999999999991, -1.4361702127659637, 2.127659574468077, -0.5585106382978711, 1.2500000000000067, 0.8776595744680815, 2.3138297872340563]}, "sen_asd": {"delta_pp": -1.475609755734606, "wins": 3, "ties": 1, "losses": 6, "per_fold_delta_pp": [-2.439024389649014, -4.878048779298039, -2.439024389649014, 4.999999998750004, -4.999999998750004, 0.0, -7.499999998124995, 2.499999999375002, -2.499999999375002, 2.499999999375002]}, "spe_td": {"delta_pp": 2.13228492091442, "wins": 6, "ties": 4, "losses": 0, "per_fold_delta_pp": [0.0, 2.173913043005671, 0.0, 0.0, 2.1276595740153947, 4.2553191480307895, 6.382978722046174, 0.0, 4.255319148030779, 2.1276595740153947]}}}, {"left": "C5", "right": "C0", "metrics": {"acc": {"delta_pp": -0.34482758620689724, "wins": 4, "ties": 2, "losses": 4, "per_fold_delta_pp": [0.0, 0.0, -3.4482758620689724, 1.1494252873563204, -2.298850574712641, 1.1494252873563204, 2.298850574712641, 2.298850574712641, -2.298850574712641, -2.298850574712641]}, "auc": {"delta_pp": -0.3788130908598608, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [0.0, -0.7953340402969244, -3.923647932131491, 1.2500000000000067, -2.127659574468077, 1.4361702127659415, 2.500000000000002, 2.1276595744680993, -2.499999999999991, -1.7553191489361741]}, "sen_asd": {"delta_pp": -1.4329268289264196, "wins": 4, "ties": 3, "losses": 3, "per_fold_delta_pp": [0.0, -14.634146337894116, -12.195121948245092, 2.499999999375002, 0.0, 4.999999998749994, 4.999999998750004, 0.0, -4.999999998749994, 4.999999998750004]}, "spe_td": {"delta_pp": 0.6753006473968393, "wins": 3, "ties": 4, "losses": 3, "per_fold_delta_pp": [0.0, 13.043478258034025, 4.347826086011342, 0.0, -4.2553191480307895, -2.1276595740153947, 0.0, 4.255319148030779, 0.0, -8.510638296061568]}}}, {"left": "C6", "right": "C0", "metrics": {"acc": {"delta_pp": 0.11494252873563092, "wins": 4, "ties": 3, "losses": 3, "per_fold_delta_pp": [0.0, -1.1494252873563315, -3.4482758620689724, 0.0, 0.0, 2.298850574712652, 2.298850574712641, 1.1494252873563204, 1.1494252873563204, -1.1494252873563204]}, "auc": {"delta_pp": 0.2141958665192567, "wins": 5, "ties": 2, "losses": 3, "per_fold_delta_pp": [0.0, -1.4846235418875864, -2.9957582184517495, 0.18617021276596812, 0.0, 2.499999999999991, 2.686170212765948, 1.0638297872340496, 1.2499999999999956, -1.0638297872340496]}, "sen_asd": {"delta_pp": 1.506097560597598, "wins": 5, "ties": 4, "losses": 1, "per_fold_delta_pp": [0.0, -7.317073168947063, 4.878048779298039, 2.499999999375002, 0.0, 4.999999998749994, 7.4999999981250065, 0.0, 2.499999999375002, 0.0]}, "spe_td": {"delta_pp": -1.0777058277047802, "wins": 2, "ties": 4, "losses": 4, "per_fold_delta_pp": [0.0, 4.347826086011342, -10.869565215028354, -2.1276595740153947, 0.0, 0.0, -2.1276595740153947, 2.1276595740153947, 0.0, -2.1276595740153947]}}}, {"left": "C7", "right": "C0", "metrics": {"acc": {"delta_pp": -0.34613375130616464, "wins": 3, "ties": 4, "losses": 3, "per_fold_delta_pp": [1.1363636363636354, 0.0, -3.4482758620689724, 0.0, -2.298850574712641, 2.298850574712652, 0.0, 1.1494252873563204, 0.0, -2.298850574712641]}, "auc": {"delta_pp": -0.39794623316260935, "wins": 5, "ties": 1, "losses": 4, "per_fold_delta_pp": [1.2195121951219523, -0.7953340402969244, -3.5259809119830288, 0.0, -2.499999999999991, 2.313829787234012, 0.18617021276595702, 1.2500000000000067, 0.18617021276596812, -2.313829787234045]}, "sen_asd": {"delta_pp": -1.457317072816815, "wins": 5, "ties": 1, "losses": 4, "per_fold_delta_pp": [2.439024389649014, -14.634146337894116, -4.878048779298039, 0.0, -4.999999998750004, 2.499999999374991, 2.499999999375002, 2.499999999375002, 2.499999999375002, -2.499999999375002]}, "spe_td": {"delta_pp": 0.6614246066997564, "wins": 2, "ties": 4, "losses": 4, "per_fold_delta_pp": [0.0, 13.043478258034025, -2.173913043005671, 0.0, 0.0, 2.1276595740153947, -2.1276595740153947, 0.0, -2.1276595740153947, -2.1276595740153947]}}}, {"left": "F5", "right": "F0", "metrics": {"acc": {"delta_pp": -0.22988505747126298, "wins": 2, "ties": 5, "losses": 3, "per_fold_delta_pp": [0.0, 2.298850574712652, -2.298850574712641, 0.0, -1.1494252873563204, 0.0, 0.0, 0.0, 1.1494252873563204, -2.298850574712641]}, "auc": {"delta_pp": -0.2734623541887604, "wins": 2, "ties": 2, "losses": 6, "per_fold_delta_pp": [-0.15568240788791377, 2.0413573700954424, -2.3064687168610987, 0.0, -1.0638297872340496, 0.0, -0.18617021276594592, -0.18617021276595702, 1.0638297872340496, -1.9414893617021312]}, "sen_asd": {"delta_pp": -0.9817073168322066, "wins": 1, "ties": 4, "losses": 5, "per_fold_delta_pp": [-2.439024389649014, -2.439024389649025, -2.439024389649025, 0.0, 0.0, 0.0, -2.499999999375002, -2.499999999375002, 0.0, 2.499999999375002]}, "spe_td": {"delta_pp": 0.4347826086011331, "wins": 5, "ties": 2, "losses": 3, "per_fold_delta_pp": [2.1276595740153836, 6.521739129017012, -2.173913043005671, 0.0, -2.1276595740153836, 0.0, 2.1276595740153836, 2.1276595740153947, 2.1276595740153836, -6.382978722046174]}}}, {"left": "F6", "right": "F0", "metrics": {"acc": {"delta_pp": 0.23119122257053262, "wins": 5, "ties": 2, "losses": 3, "per_fold_delta_pp": [-1.1363636363636465, 1.1494252873563315, -1.1494252873563204, 1.1494252873563204, 0.0, 0.0, 1.1494252873563204, 1.1494252873563204, -1.1494252873563204, 1.1494252873563204]}, "auc": {"delta_pp": 0.11230285869001211, "wins": 5, "ties": 1, "losses": 4, "per_fold_delta_pp": [-1.2195121951219634, 0.5567338282078449, -1.2195121951219634, 0.8776595744680815, -0.18617021276594592, 0.0, 1.2500000000000067, 1.0638297872340496, -1.4361702127659526, 1.4361702127659637]}, "sen_asd": {"delta_pp": -1.7134146337269118, "wins": 2, "ties": 2, "losses": 6, "per_fold_delta_pp": [-2.439024389649014, -9.756097558596078, -2.439024389649025, -2.499999999375002, -2.499999999375002, 0.0, 2.499999999375002, 0.0, -4.999999998750004, 4.999999998750004]}, "spe_td": {"delta_pp": 1.9380203511089922, "wins": 5, "ties": 4, "losses": 1, "per_fold_delta_pp": [0.0, 10.869565215028354, 0.0, 4.2553191480307895, 2.1276595740153947, 0.0, 0.0, 2.1276595740153947, 2.1276595740153836, -2.1276595740153947]}}}, {"left": "F7", "right": "F0", "metrics": {"acc": {"delta_pp": -0.22988505747126298, "wins": 3, "ties": 2, "losses": 5, "per_fold_delta_pp": [0.0, 2.298850574712652, -2.298850574712641, -1.1494252873563204, 0.0, 1.1494252873563315, 1.1494252873563204, -1.1494252873563315, -1.1494252873563204, -1.1494252873563204]}, "auc": {"delta_pp": -0.32215541165587447, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [0.15568240788791377, 2.1739130434782594, -2.3064687168610987, -1.4361702127659526, 0.0, 0.8776595744680704, 0.8776595744681037, -1.2499999999999956, -1.0638297872340385, -1.2500000000000067]}, "sen_asd": {"delta_pp": -1.4999999996250013, "wins": 1, "ties": 3, "losses": 6, "per_fold_delta_pp": [2.439024389649025, 0.0, -2.439024389649025, -4.999999998750004, 0.0, -2.499999999375002, -2.499999999375002, -2.499999999375002, 0.0, -2.499999999375002]}, "spe_td": {"delta_pp": 0.8556891765051844, "wins": 4, "ties": 3, "losses": 3, "per_fold_delta_pp": [-2.1276595740153947, 4.347826086011342, -2.173913043005671, 2.1276595740153947, 0.0, 4.2553191480307895, 4.255319148030779, 0.0, -2.1276595740153947, 0.0]}}}, {"left": "C2", "right": "C8", "metrics": {"acc": {"delta_pp": -0.6870428422152575, "wins": 2, "ties": 3, "losses": 5, "per_fold_delta_pp": [-2.272727272727282, -1.1494252873563315, 2.298850574712652, 1.1494252873563204, 0.0, -2.298850574712652, -2.298850574712641, 0.0, -2.298850574712641, 0.0]}, "auc": {"delta_pp": -0.6676378014936479, "wins": 4, "ties": 0, "losses": 6, "per_fold_delta_pp": [-2.283341982355991, -0.9544008483563404, 2.4390243902438935, 1.4361702127659859, 0.18617021276596812, -2.313829787234012, -2.127659574468077, -0.5585106382978711, -2.6861702127659814, 0.18617021276594592]}, "sen_asd": {"delta_pp": -0.26219512188269567, "wins": 5, "ties": 1, "losses": 4, "per_fold_delta_pp": [-2.439024389649025, 2.439024389649014, 4.878048779298039, 4.999999998750004, 2.499999999375002, -2.499999999374991, 0.0, -7.4999999981250065, -7.499999998124995, 2.499999999375002]}, "spe_td": {"delta_pp": -1.0730804808057504, "wins": 2, "ties": 1, "losses": 7, "per_fold_delta_pp": [-2.1276595740153836, -4.347826086011342, 0.0, -2.1276595740153947, -2.1276595740153947, -2.1276595740153947, -4.2553191480307895, 6.382978722046184, 2.1276595740153947, -2.1276595740153836]}}}, {"left": "F2", "right": "F8", "metrics": {"acc": {"delta_pp": 0.0013061650992696183, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [-1.1363636363636465, 1.1494252873563315, -1.1494252873563204, -1.1494252873563204, -2.298850574712641, 1.1494252873563315, 2.298850574712641, 0.0, 3.4482758620689613, -2.298850574712641]}, "auc": {"delta_pp": -0.05331840436813562, "wins": 4, "ties": 1, "losses": 5, "per_fold_delta_pp": [-1.0638297872340496, 1.4846235418875642, -1.0869565217391353, -1.0638297872340496, -2.8723404255319163, 1.2499999999999956, 2.127659574468077, 0.0, 3.005319148936181, -2.313829787234023]}, "sen_asd": {"delta_pp": -0.5182926827927958, "wins": 2, "ties": 5, "losses": 3, "per_fold_delta_pp": [0.0, 7.3170731689470525, 0.0, 0.0, -9.999999997499998, 2.499999999374991, 0.0, 0.0, -2.499999999375002, -2.499999999375002]}, "spe_td": {"delta_pp": 0.4116558741059961, "wins": 3, "ties": 2, "losses": 5, "per_fold_delta_pp": [-2.1276595740153836, -4.347826086011342, -2.173913043005671, -2.1276595740153947, 4.2553191480307895, 0.0, 4.2553191480307895, 0.0, 8.510638296061568, -2.1276595740153947]}}}, {"left": "F2", "right": "F9", "metrics": {"acc": {"delta_pp": 0.002612330198537016, "wins": 3, "ties": 4, "losses": 3, "per_fold_delta_pp": [-2.272727272727282, 0.0, -1.1494252873563204, 1.1494252873563204, 0.0, 2.298850574712652, -1.1494252873563204, 1.1494252873563204, 0.0, 0.0]}, "auc": {"delta_pp": -0.11096602062228178, "wins": 4, "ties": 0, "losses": 6, "per_fold_delta_pp": [-2.283341982356002, -0.13255567338282814, -0.9544008483563071, 1.0638297872340496, -0.18617021276595702, 2.127659574468077, -1.4361702127659859, 0.8776595744680815, 0.18617021276594592, -0.37234042553189184]}, "sen_asd": {"delta_pp": -1.4939024386524014, "wins": 2, "ties": 2, "losses": 6, "per_fold_delta_pp": [-2.439024389649014, -2.439024389649014, 2.439024389649025, 0.0, -2.499999999375002, 0.0, -4.999999998750004, -2.499999999375002, 2.499999999375002, -4.999999998750004]}, "spe_td": {"delta_pp": 1.2719703975102081, "wins": 7, "ties": 0, "losses": 3, "per_fold_delta_pp": [-2.1276595740153836, 2.173913043005671, -4.347826086011342, 2.1276595740153836, 2.1276595740153947, 4.2553191480307895, 2.1276595740153947, 4.2553191480307895, -2.1276595740153947, 4.255319148030779]}}}, {"left": "F2", "right": "F10", "metrics": {"acc": {"delta_pp": 0.11624869383490057, "wins": 3, "ties": 2, "losses": 5, "per_fold_delta_pp": [-1.1363636363636465, 0.0, -1.1494252873563204, -1.1494252873563204, -3.4482758620689613, 0.0, 2.298850574712641, 3.4482758620689724, 3.4482758620689613, -1.1494252873563204]}, "auc": {"delta_pp": 0.040545114054284515, "wins": 3, "ties": 0, "losses": 7, "per_fold_delta_pp": [-1.2195121951219634, -0.13255567338282814, -0.6892895015906619, -1.0638297872340496, -3.749999999999998, -0.18617021276597923, 2.500000000000002, 3.1914893617021267, 3.377659574468095, -1.6223404255318985]}, "sen_asd": {"delta_pp": -0.7560975607850973, "wins": 3, "ties": 2, "losses": 5, "per_fold_delta_pp": [-2.439024389649014, -2.439024389649014, 7.317073168947063, 0.0, -7.4999999981250065, -2.499999999375002, 4.999999998750004, 0.0, 2.499999999375002, -7.4999999981250065]}, "spe_td": {"delta_pp": 0.8371877889090727, "wins": 5, "ties": 3, "losses": 2, "per_fold_delta_pp": [0.0, 2.173913043005671, -8.695652172022683, -2.1276595740153947, 0.0, 2.1276595740153947, 0.0, 6.382978722046184, 4.255319148030779, 4.255319148030779]}}}, {"left": "C2", "right": "P0", "metrics": {"acc": {"delta_pp": -1.7202194357366787, "wins": 1, "ties": 1, "losses": 8, "per_fold_delta_pp": [-3.409090909090917, -2.298850574712652, 0.0, 1.1494252873563204, -2.298850574712641, -2.298850574712652, -3.4482758620689613, -1.1494252873563204, -1.1494252873563204, -2.298850574712641]}, "auc": {"delta_pp": -1.7936982468807106, "wins": 2, "ties": 0, "losses": 8, "per_fold_delta_pp": [-3.502854177477943, -2.4390243902439046, 0.13255567338281704, 1.2500000000000178, -2.313829787234045, -2.313829787234023, -3.3776595744680837, -1.2500000000000067, -1.8085106382978777, -2.3138297872340563]}, "sen_asd": {"delta_pp": -2.731707316394704, "wins": 2, "ties": 0, "losses": 8, "per_fold_delta_pp": [-4.878048779298039, -4.878048779298039, 2.439024389649025, 2.499999999375002, -2.499999999374991, -2.499999999374991, -2.499999999375002, -2.499999999375002, -9.999999997499998, -2.499999999375002]}, "spe_td": {"delta_pp": -0.8556891765051844, "wins": 1, "ties": 3, "losses": 6, "per_fold_delta_pp": [-2.1276595740153836, 0.0, -2.173913043005671, 0.0, -2.1276595740153947, -2.1276595740153947, -4.2553191480307895, 0.0, 6.382978722046174, -2.1276595740153836]}}}, {"left": "F2", "right": "P0", "metrics": {"acc": {"delta_pp": -1.3766980146290497, "wins": 2, "ties": 2, "losses": 6, "per_fold_delta_pp": [-2.272727272727282, -1.1494252873563204, -4.597701149425292, -1.1494252873563204, -4.597701149425282, 0.0, 1.1494252873563204, 0.0, 1.1494252873563204, -2.298850574712641]}, "auc": {"delta_pp": -1.4419406150583258, "wins": 2, "ties": 2, "losses": 6, "per_fold_delta_pp": [-2.283341982356002, -1.0869565217391353, -4.612937433722153, -1.0638297872340496, -5.000000000000004, -1.1102230246251565e-14, 1.2499999999999956, 0.0, 1.0638297872340496, -2.686170212765948]}, "sen_asd": {"delta_pp": -2.2317073165197043, "wins": 1, "ties": 5, "losses": 4, "per_fold_delta_pp": [-2.439024389649014, 0.0, -4.878048779298028, 0.0, -9.999999997499998, 0.0, 2.499999999375002, 0.0, 0.0, -7.4999999981250065]}, "spe_td": {"delta_pp": -0.6521739129017012, "wins": 2, "ties": 4, "losses": 4, "per_fold_delta_pp": [-2.1276595740153836, -2.173913043005671, -4.347826086011342, -2.1276595740153947, 0.0, 0.0, 0.0, 0.0, 2.1276595740153836, 2.1276595740153947]}}}, {"left": "C2", "right": "E1", "metrics": {"acc": {"delta_pp": -1.2604493207941492, "wins": 2, "ties": 0, "losses": 8, "per_fold_delta_pp": [-3.409090909090917, -1.1494252873563315, 1.1494252873563315, 2.298850574712641, -1.1494252873563204, -2.298850574712652, -2.298850574712641, -1.1494252873563204, -2.298850574712641, -2.298850574712641]}, "auc": {"delta_pp": -1.3209454885945715, "wins": 2, "ties": 0, "losses": 8, "per_fold_delta_pp": [-3.502854177477943, -1.2195121951219523, 0.954400848356296, 2.6861702127659814, -1.2499999999999956, -2.313829787234023, -2.313829787234045, -1.2500000000000067, -2.8723404255319274, -2.1276595744680993]}, "sen_asd": {"delta_pp": -2.2256097555471044, "wins": 1, "ties": 1, "losses": 8, "per_fold_delta_pp": [-4.878048779298039, -2.439024389649014, -2.439024389649014, 7.4999999981250065, -2.499999999374991, -2.499999999374991, -2.499999999375002, -2.499999999375002, -9.999999997499998, 0.0]}, "spe_td": {"delta_pp": -0.4162812210050215, "wins": 2, "ties": 3, "losses": 5, "per_fold_delta_pp": [-2.1276595740153836, 0.0, 4.347826086011342, -2.1276595740153947, 0.0, -2.1276595740153947, -2.1276595740153947, 0.0, 4.2553191480307895, -4.255319148030779]}}}, {"left": "F2", "right": "E1", "metrics": {"acc": {"delta_pp": -0.9169278996865204, "wins": 1, "ties": 5, "losses": 4, "per_fold_delta_pp": [-2.272727272727282, 0.0, -3.4482758620689613, 0.0, -3.4482758620689613, 0.0, 2.298850574712641, 0.0, 0.0, -2.298850574712641]}, "auc": {"delta_pp": -0.9691878567721867, "wins": 3, "ties": 3, "losses": 4, "per_fold_delta_pp": [-2.283341982356002, 0.13255567338281704, -3.791092258748674, 0.37234042553191404, -3.936170212765955, -1.1102230246251565e-14, 2.313829787234034, 0.0, 0.0, -2.499999999999991]}, "sen_asd": {"delta_pp": -1.7256097556721048, "wins": 3, "ties": 3, "losses": 4, "per_fold_delta_pp": [-2.439024389649014, 2.439024389649025, -9.756097558596066, 4.999999998750004, -9.999999997499998, 0.0, 2.499999999375002, 0.0, 0.0, -4.999999998750004]}, "spe_td": {"delta_pp": -0.21276595740153834, "wins": 3, "ties": 4, "losses": 3, "per_fold_delta_pp": [-2.1276595740153836, -2.173913043005671, 2.173913043005671, -4.2553191480307895, 2.1276595740153947, 0.0, 2.1276595740153947, 0.0, 0.0, 0.0]}}}]。

科研筛选：{"C2": {"baseline": false, "semantic_off": false, "numeric": false, "correct_patient": false, "dense": true}, "F2": {"baseline": false, "semantic_off": false, "numeric": false, "correct_patient": false, "dense": false, "roi": false, "direction": false}}。NO-GO。仅一个seed，不声称跨seed稳定；不能用最高ACC或有效梯度替代真实医学文本、正确患者配对及Sparse必要性。

详细机制、逐折四指标和反事实见/data/adodas/SPromptGL-main/experiment/experiment29_connectivity_semantic_selective_auxiliary/results/report.md；诊断200条目标，包含文本几何/Query匹配、损失梯度、门控及站点关系、各模态关闭及数值/错ROI/反向文本。全部20组新增注册参数71277；C3/F3关闭语义后数学相同仍独立运行。局部Token非ROI，ANAT/FUNC为QC；不把效应方向当病灶真值。

代码/数据SHA：{"core": {"main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67", "ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763"}, "data": {"/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1", "/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv": "4f9cd719d27c1e8b9a6cf8933534c0b796101cacf248e509fcfd3c5868fa3ebf", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_256_mapping.csv": "53c927ebf8a594108942146992b14bc0945f0271b713a669ff00ce37e5f8788d", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv": "b383308bb413a94d1646511a9bea5bc16f54c975b938ac4d947d676e357c5eb2"}, "code": {"method.py": "60580bd8fcb632052090d6c425d602734b8ac8e74acc7c68cf2bc55a2e77aecf", "prompt_bank.py": "7d472804b075a7438de84a2b3add0a51fece03a1ca02313b9e36a9bb6a312049", "integration.py": "dd12ec4aa297676069fe30134478f07c9bdddcdff59bccce5bbbb0c2dd69d1ec", "run.py": "77c9709cf5af2570d91bec44c49251f882954253fb9daa8e69e858bc9e9921a4", "integration.patch": "93ba4192e779437931bebf2fc1aeac74d6912d51a5553d772b300d21ea767d81"}, "cache": "b7ba891518fadc8b8f1e03916177b26f092f4a7ceb743f1889f6f38b9d9474a8"}。历史保护：{"status": "UNCHANGED_SHA256", "files": 1028}。预检：{"passed": true, "completed_utc": "2026-10-20T07:26:31.164164+00:00", "checks": {"P0_untouched_initial_state_forward_two_adam_update_rng": "BITWISE_EXACT", "P0_seed0_fold1_initial_trunk_sha256": "b464f86ee750e5550a6330942ee102a5d9d1f3ab4967a385e23655741c92ae0f", "SPGS_OFF_RES_matches_historical_E28_E1": "BITWISE_EXACT", "all20_original_and_new_parameter_initialization_rng": "BITWISE_EXACT", "additional_parameters_all20": 71277, "ROI252_unknown4_excluded_same_selection_all20": "PASS", "heldout_label_flip_statistics_forward_full_train_two_adams_rng": "BITWISE_EXACT", "first_step_gradient_norms": {"gfr": 2.306002378463745, "pln_original": 2.9831485748291016, "fc_encoder": 0.043160244822502136, "edge_id": 0.0006165030063129961, "query": 0.021496789529919624, "semantic_decode": 0.0, "selectors": 0.0005485737347044051, "encoders": 2.658269295352511e-05, "routers": 0.0017742698546499014, "values": 0.0, "aux_decode": 0.0, "alpha_sem": 0.0014647948555648327, "alpha_aux": 5.3242849389789626e-05, "aux_PHENO": 0.0010458370670676231, "aux_ANAT": 0.0011031266767531633, "aux_FUNC": 0.0010672437492758036}, "second_step_gradient_norms": {"gfr": 0.6294006109237671, "pln_original": 1.3194774389266968, "fc_encoder": 0.03206922486424446, "edge_id": 0.0005675334250554442, "query": 0.018681423738598824, "semantic_decode": 4.155246278969571e-05, "selectors": 0.0005503208958543837, "encoders": 2.6089823222719133e-05, "routers": 0.0017845187103375793, "values": 2.055876620943309e-06, "aux_decode": 3.538464397934149e-06, "alpha_sem": 0.0009057074785232544, "alpha_aux": 2.0153636796749197e-05, "aux_PHENO": 0.0010471101850271225, "aux_ANAT": 0.0011050545144826174, "aux_FUNC": 0.0010818780865520239}, "no_valid_local_fallback_and_zero_aux_no_bias_bypass": "PASS", "same_checkpoint_text_intervention_REAL_restore_weights_rng": "BITWISE_EXACT", "two_step_intervention_probability_changes": {"REAL": 0.0, "SEMANTIC_OFF": 1.9961396446888102e-06, "PHENO_OFF": 2.4978692181321094e-07, "ANAT_OFF": 4.5657259306608466e-07, "FUNC_OFF": 4.6584517576775397e-07, "AUX_OFF": 8.975767968877335e-07, "ALL_OFF": 2.713944468268892e-06, "NUMERIC_TEXT": 1.027658299790346e-06, "WRONG_ROI": 1.7762491211215092e-07, "SWAP_DIRECTION": 1.4286123928286543e-07, "GLOBAL_CLASS_SWAP": 0.0, "AUX_SHUFFLED": 5.994338607706595e-07}, "global_local_loss_do_not_supervise_auxiliary": "PASS", "epoch25_disk_both_adams_rng_next_update_prediction": "BITWISE_EXACT", "fold_boundary_next_initialization_rng": "BITWISE_EXACT", "device": "_CudaDeviceProperties(name='PPU-ZW810E', major=8, minor=0, total_memory=98304MB, multi_processor_count=64, uuid=019e4201-0591-0030-0000-000080387166, pci_bus_id=99, pci_device_id=0, pci_domain_id=0, L2_cache_size=64MB)", "GPU1_visible_logical0": "_CudaDeviceProperties(name='PPU-ZW810E', major=8, minor=0, total_memory=98304MB, multi_processor_count=64, uuid=019e4201-8920-0208-0000-0000c0f15256, pci_bus_id=234, pci_device_id=0, pci_domain_id=0, L2_cache_size=64MB)"}}。

历史协议test=val、外层前RFE/标准化、871全图transductive及跨轮探索保留；指标偏乐观，非独立无泄漏泛化。后续唯一建议为真实互补测量及训练内选模/独立队列验证；本轮不再追分或追加超参。无自动Git提交/推送。


<!-- EXPERIMENT29_FINAL_DELIVERY_GITHUB_HANDOFF_UPDATE -->
### 实验29最终交付复核与GitHub归档

更新时间：2026-10-20T08:40:37+00:00。工程状态 **COMPLETE**；科学判断 **NO-GO**。本节补充最终收尾、代码版本及已获用户授权的发布记录，以上历史正文逐字保留。

- 完成20/20个独立任务、200/200个唯一训练fold及200条唯一机制诊断；仅seed0，每任务连续10折。GPU0、GPU1各完成100折，正式训练墙钟1980.191秒（约33.00分钟）；历史P0/E1未重训。
- 最优Checkpoint诊断包含2400个条件记录，保存在diagnostics.csv的反事实字段内，不计作额外训练。临时`.runtime`、权重和训练工作目录已清理；最终实验目录11个文件均与发布提交逐文件Git blob SHA一致。
- 此次复核确认六核心源码、数据、历史参考、最终实验代码、Prompt缓存及结果哈希与manifest一致；1028个受保护历史文件逐个SHA256通过。未新增训练、改动模型或追试参数。

下表沿用正式报告，数值为百分比，±为**seed0十折间population SD**，不是三seed间SD；AUC为原始硬标签AUC（等价于BA，微小分母舍入除外），不是连续概率AUC。三seed历史P0约91.691%的口径不可与本表seed0的91.72936%混用。

|组|ACC %|硬标签AUC %|SEN-ASD %|SPE-TD %|ASD FN|TD FP|
|---|---:|---:|---:|---:|---:|---:|
|P0（历史seed0）|91.72936±2.95053|91.56250±2.94444|89.34146±3.66765|93.78353±3.67488|43|29|
|E1（历史SPGS OFF）|91.26959±3.14214|91.08975±3.07779|88.83537±3.73203|93.34413±4.99214|45|31|
|C2（粗粒度主方法）|90.00914±2.72916|89.76880±2.79791|86.60976±4.98885|92.92784±3.63556|54|33|
|F2（细粒度主方法）|90.35266±3.61167|90.12056±3.57123|87.10976±3.59282|93.13136±4.33537|52|32|
|C3/F3（语义OFF）|91.84431±2.88924|91.69286±2.91106|89.59756±4.36793|93.78816±3.65830|42|29|

最终判读：C2相对同seed P0的ACC/硬AUC降低1.72022/1.79370个百分点，F2降低1.37670/1.44194个百分点；ASD漏诊由P0的43例升至54/52例。C2/F2相对从头训练的语义OFF对照分别降低1.83516/1.49164个百分点ACC。C2不及数值码C8；F2相对错误ROI、方向交换及数值码的ACC差仅+0.00131/+0.00261/+0.11625个百分点，且ASD灵敏度均更低。Selective相对Dense虽改善ACC/硬AUC，F2的灵敏度反而下降，不能据此宣称识别了患者特异医学互补信息。

机制澄清：C2/F2语义残差L2均值0.572990/0.581562，接近理论L2上界0.6，分支响应并非近零。这里限制是36维残差**每坐标≤0.10**，不是相对原特征范数10%。同Checkpoint关闭语义后ACC分别下降8.95507/16.42241个百分点，只证明已训练模型的计算依赖；完整训练的语义OFF组表现更好，故不能据关闭分支的降分认定医学知识有效。语义解码器还读取CORR Anchor，OFF同时移除了FC容量。粗粒度重复Key没有连接身份辨识信息，其极小梯度/有效秩可能来自浮点累加；F2全局类别共同置换响应为零是加权和算子的置换不变性。数值对照只在最终归一化前匹配均值/协方差，最终类别夹角及Gram不完全相同，不宣称已经隔离纯文案含义。

最终代码可追溯性：实际训练时`run.py` SHA256为`77c9709cf5af2570d91bec44c49251f882954253fb9daa8e69e858bc9e9921a4`；交付版本为`d44c1bad64959af8be1776fac317b4fa4fcd02761d342310265fb58ba31e0592`。训练结束后仅修复NFS控制器锁释放/清理顺序、已完成CSV的refresh/finalize校验和报告表述；完整diff与理由见manifest的`post_training_changes`，未改变训练数学、随机流或200折成绩。收尾曾发生目录非空错误，已于2026-10-20T08:03:35Z修复并完成最终验收，没有重训；manifest正确标记COMPLETE，cleanup_status=REMOVED。manifest的handoff_sha256保留训练完成时的历史快照，本次追加不回写已发布实验元数据。

GitHub发布已按用户随后明确的上传要求完成，仅新增实验29目录的11个文件，未上传其他本地改动。提交：[d3c5814500dff71f996108d6c2f89435b9bf9ed8](https://github.com/tangaii/SPromptGL/commit/d3c5814500dff71f996108d6c2f89435b9bf9ed8)；父提交为`5e8049b40260b3090cddfeea94c7820b5b3c2ed0`。固定归档：[实验29完整目录](https://github.com/tangaii/SPromptGL/tree/d3c5814500dff71f996108d6c2f89435b9bf9ed8/experiment/experiment29_connectivity_semantic_selective_auxiliary)。本次更新前再次核验远端main为该提交；这条发布记录补充此前“实验运行不自动提交/推送”的时点说明，上传由后续独立用户指令授权。

最终结果SHA256：fold_metrics.csv=`36d403c0438d8515e0e06a04015479d71473bf6d12a5edd077387f6d09591820`；diagnostics.csv=`86a03485ebe0554f023de994bc461e8ec3bc7cec74786d06012fae9179002ca0`；report.md=`81e46202f7a80f7a2f3b5c8d9eaa1c0e14cd3eae80a988ac9b8ecd8938e9ca28`；Prompt缓存=`b7ba891518fadc8b8f1e03916177b26f092f4a7ceb743f1889f6f38b9d9474a8`。完整逐折数据、预设比较和机制边界见[实验29中文报告](experiment/experiment29_connectivity_semantic_selective_auxiliary/results/report.md)。

本轮不改变科研结论：保留P0；C3/F3约+0.115个百分点的单seed领先不足以支持医学语义创新。历史test=val选模、CV前RFE/标准化、全871人transductive和多轮探索仍有乐观偏差。后续优先获得真实互补测量，并采用训练内选模或独立队列验证，不追加本轮超参数搜索。

历史保护：本次追加前handoff为272818字节，SHA256=`bd7e07ea3801594418e5a60a17e6b67c0402d65960cc590d6e04d9d4f1894c2d`；原有全部字节作为新文件完整前缀保留。本次仅更新本地根handoff.md，未执行新的Git提交或推送。


<!-- EXPERIMENT30_CLASS_CONDITIONAL_SEMANTIC_SHARED_REPAIR_FINAL -->
## 实验三十：类别条件语义共享修复

更新时间：2026-10-20T14:35:16.064518+00:00；STATUS=COMPLETE；正式训练折 180/180；任务 18/18。

目的：以raw[:,64:320] CORR为诊断Anchor，使用ASD/TD类别竞争语义，并只把与CORR共享的PHENO年龄/性别28维、ANAT六项、FUNC十项辅助表示通过含null的Router注入PLN-CORR。与原SPGS相比，Z/C/F全部跳过原SPGS；P0只读审计原SPGS。与实验29相比，本轮删除edge_id/连接ID InfoNCE和每坐标固定0.1残差，改为训练折可靠度Top-16、同连接ASD/TD文本对、类别CE和相对CAM-CORR范数约束。

预注册18组（seed0，每组原始连续10折）：Z0/Z1；C0–C6（粗粒度文本，依次含无辅助、shared、类别混排、站点内辅助错配、dense、private、无配对损失）；F0–F8（细粒度真实ROI文本，含shared主组、错ROI、反方向、文本混排、站点内辅助错配、dense、private、无配对损失）。没有依据早期成绩删组或调整参数。

GPU分配：GPU0完成 90/90折（Z0,C0,C2,C4,C6,F0,F2,F4,F6）；GPU1完成 90/90折（Z1,C1,C3,C5,F1,F3,F5,F7,F8）；正式墙钟 7185.60秒。

正式四指标及错误数（均值±十折population SD；AUC是原始one-hot硬预测口径）：

|组|ACC %|Hard-AUC %|SEN-ASD %|SPE-TD %|ASD漏诊|TD误报|
|---|---:|---:|---:|---:|---:|---:|
|P0 (历史只读)|91.72936±2.95053|91.56250±2.94444|89.34146±3.66765|93.78353±3.67488|43|29|
|E1 (历史只读)|91.26959±3.14214|91.08975±3.07779|88.83537±3.73203|93.34413±4.99214|45|31|
|Z0 (本轮)|91.03971±3.33031|90.81375±3.41084|87.62195±5.35135|94.00555±3.01118|50|28|
|Z1 (本轮)|91.72936±3.03876|91.60763±2.97138|90.07927±3.31542|93.13599±4.73086|40|32|
|C0 (本轮)|91.61442±3.09283|91.44286±3.06366|89.09756±3.49809|93.78816±3.89794|44|29|
|C1 (本轮)|91.72936±3.03876|91.58112±3.00735|89.59146±3.23620|93.57077±3.73939|42|30|
|C2 (本轮)|91.49817±3.18333|91.34154±3.06607|89.56098±3.35452|93.12211±5.73461|42|32|
|C3 (本轮)|91.61442±2.77776|91.47422±2.74204|89.81707±3.96210|93.13136±4.74306|41|32|
|C4 (本轮)|91.61442±3.00618|91.37933±3.01864|88.34146±4.71849|94.41721±4.36080|47|26|
|C5 (本轮)|91.38454±3.38681|91.23083±3.38939|89.10366±4.68807|93.35800±4.35172|44|31|
|C6 (本轮)|91.61442±2.96190|91.35767±2.98615|88.08537±4.30042|94.62997±3.80486|48|25|
|F0 (本轮)|91.49948±3.22477|91.34995±3.13947|89.57317±3.67769|93.12673±5.38420|42|32|
|F1 (本轮)|91.49948±3.18353|91.32649±3.16100|89.09146±3.84895|93.56152±4.23828|44|30|
|F2 (本轮)|91.49948±3.05650|91.35278±3.04743|89.34756±3.80001|93.35800±3.79612|43|31|
|F3 (本轮)|91.61442±3.04981|91.48263±2.97214|89.82927±3.01565|93.13599±4.62128|41|32|
|F4 (本轮)|91.61442±3.09283|91.46401±3.01846|89.57927±3.81723|93.34875±5.15894|42|31|
|F5 (本轮)|91.61442±2.72978|91.32580±2.67591|87.59146±2.49184|95.06013±3.64698|50|23|
|F6 (本轮)|91.49948±3.26548|91.28367±3.26265|88.35366±4.11655|94.21369±3.85580|47|27|
|F7 (本轮)|91.61442±3.17711|91.44003±3.14885|89.32317±4.33059|93.55689±4.96979|43|30|
|F8 (本轮)|91.72936±2.85957|91.54619±2.87455|89.09146±4.02305|94.00093±3.57944|44|28|

关键预设配对差，单位百分点（ACC / Hard-AUC / SEN-ASD / SPE-TD）：

|比较|ACC|Hard-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|
|F1−P0|-0.2299|-0.2360|-0.2500|-0.2220|
|F1−Z0|+0.4598|+0.5127|+1.4695|-0.4440|
|F1−F0|-0.0000|-0.0235|-0.4817|+0.4348|
|F1−F2|-0.0000|-0.0263|-0.2561|+0.2035|
|F1−F3|-0.1149|-0.1561|-0.7378|+0.4255|
|F1−F4|-0.1149|-0.1375|-0.4878|+0.2128|
|F1−F5|-0.1149|+0.0007|+1.5000|-1.4986|
|F1−F6|+0.0000|+0.0428|+0.7378|-0.6522|
|F1−F7|-0.1149|-0.1135|-0.2317|+0.0046|
|F1−F8|-0.2299|-0.2197|+0.0000|-0.4394|
|C1−P0|-0.0000|+0.0186|+0.2500|-0.2128|
|C1−Z0|+0.6897|+0.7674|+1.9695|-0.4348|
|C1−C0|+0.1149|+0.1383|+0.4939|-0.2174|
|C1−C2|+0.2312|+0.2396|+0.0305|+0.4487|
|C1−C3|+0.1149|+0.1069|-0.2256|+0.4394|
|C1−C4|+0.1149|+0.2018|+1.2500|-0.8464|
|C1−C5|+0.3448|+0.3503|+0.4878|+0.2128|
|C1−C6|+0.1149|+0.2234|+1.5061|-1.0592|

机制概况（十折诊断均值）：

|组|有效连接数|总残差/原CORR|语义残差/原CORR|辅助残差/原CORR|null路由|pair|reconstruction|orthogonality|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Z0|16.000000|0.000000|0.000000|0.000000|1.000000|0.000000|0.000000|0.000000|
|Z1|16.000000|0.036901|0.000000|0.036901|0.074105|3.891378|0.091393|0.005563|
|C0|16.000000|0.038909|0.038909|0.000000|1.000000|0.000000|0.000000|0.000000|
|C1|16.000000|0.074218|0.039705|0.038032|0.081947|3.854106|0.090905|0.006650|
|C2|16.000000|0.067804|0.038640|0.038279|0.090620|3.868803|0.102132|0.006825|
|C3|16.000000|0.074545|0.039073|0.037924|0.036604|3.875253|0.094357|0.006023|
|C4|16.000000|0.053781|0.038586|0.037196|0.000000|3.861755|0.114094|0.006940|
|C5|16.000000|0.074367|0.039199|0.039133|0.070377|3.830007|0.118744|0.016266|
|C6|16.000000|0.075049|0.039494|0.038946|0.072482|0.000000|0.091413|0.005118|
|F0|16.000000|0.038613|0.038613|0.000000|1.000000|0.000000|0.000000|0.000000|
|F1|16.000000|0.069477|0.038621|0.037315|0.145677|3.860978|0.090553|0.005943|
|F2|16.000000|0.069402|0.038892|0.038327|0.128082|3.866997|0.095667|0.006407|
|F3|16.000000|0.071748|0.039233|0.038362|0.141326|3.865016|0.097446|0.006649|
|F4|16.000000|0.059793|0.038516|0.036494|0.109596|3.861664|0.089847|0.006230|
|F5|16.000000|0.073763|0.038508|0.037813|0.105396|3.883037|0.097728|0.005765|
|F6|16.000000|0.054431|0.038609|0.038282|0.000000|3.862971|0.120916|0.007105|
|F7|16.000000|0.070856|0.038777|0.037798|0.113915|3.813260|0.112233|0.012950|
|F8|16.000000|0.068983|0.038568|0.037893|0.100781|0.000000|0.097951|0.005221|

预检和完整性：764条RN50冻结文本、252条有效连接、P0 bitwise parity、Z0零残差、训练/测试标签边界、5%/10%范数上界、私有梯度边界及25 epoch双Adam/RNG磁盘恢复通过。源、数据、历史参考、缓存及代码SHA见 /data/adodas/SPromptGL-main/experiment/experiment30_class_conditional_semantic_shared_repair/manifest.json；根核心源码与实验1–29的保护哈希未变。

科研判定：**NO-GO**。GO requires both principal groups to exceed their frozen baseline, semantic and auxiliary controls by more than 0.1 pp in ACC and hard-AUC, with at least 7/10 paired-fold wins; otherwise NO-GO. 只有完整重训对照可支持机制归因；同checkpoint反事实仅用于描述依赖。兼容度不是临床标定扫描质量，训练折效应方向不是病灶真值。

限制：val=test参与Checkpoint选择，RFE和标准化在外层划分前完成，GFR仍在871人上transductive传播；本结果是原协议内部可比指标，不代表无泄漏独立泛化、临床因果或病灶发现。

完整报告：/data/adodas/SPromptGL-main/experiment/experiment30_class_conditional_semantic_shared_repair/results/report.md。未执行Git commit/push。


<!-- EXPERIMENT31_SCORE_FINAL -->
## 实验三十一：SCORE 个体化脑连接语义证据与可靠性调制

更新时间：2026-10-21T07:54:54.342749+00:00；状态：COMPLETE；正式训练折 180/180；18组各seed=0连续10折。

审计：核对原始main/model/network/layers/utils、原始shell、实验11/13/28/29/30与SPromptGL论文；验证CORR256、252条ROI连接映射、四个未知连接、PHENO48的年龄26+性别2+站点20、ANAT6/FUNC10、原SPGS位置与val=test checkpoint协议。RTGMFF只借鉴患者特异性ROI状态报告和年龄/性别FiLM，未移植AAL-116、ALFF/fALFF/ReHo、原始影像Wavelet-Mamba或其BioBERT路径。

方法：每个外层训练折仅用CORR训练样本拟合median/MAD和训练站点内/(内+间)方差可靠性；252条已验证连接按五个参考状态生成1260条冻结RN50文本。SCORE以数值Query与当前患者状态文本Value交互聚合证据，PHENO年龄/性别只作Query FiLM，FUNC/ANAT只作证据门控；15%训练连接的masked状态预测不接收目标z/状态/文本，修正只作用CAM-CORR且严格小于5%。A0–A11及B0–B5全部从头训练，B组包含错误ROI、方向/程度、同站点错配、几何数值码和原8句固定Prompt控制。

正式四指标（均值±十折population SD）见实验目录results/report.md；该报告只把ACC、Hard-AUC、SEN-ASD、SPE-TD列为论文指标，诊断另存diagnostics.csv。

科研判定：**NO-GO**。GO requires A5 to exceed frozen baseline and every prespecified semantic, patient-matching, auxiliary and masked control by >0.1 pp in ACC and Hard-AUC with at least 7/10 paired-fold wins; otherwise NO-GO. 正确文本、患者对应关系、质量门控和masked机制只有在预设配对差及机制诊断共同支持时才可归因；单seed且val=test的内部结果不能宣称无偏泛化或临床可用。

完整交付：/data/adodas/SPromptGL-main/experiment/experiment31_subject_specific_connectomic_evidence；GPU0/GPU1折数和墙钟见manifest.json；根目录源码保护哈希、缓存哈希、工程预检与断点恢复记录均在manifest及runtime审计日志中。未执行Git commit/push。

<!-- EXPERIMENT32_DCER_FINAL -->
## 实验三十二：DCER 疾病对比证据路由

更新时间：2026-10-21T10:33:50.083810+00:00；状态：COMPLETE；正式折数：120/120；GPU0=60、GPU1=60。正式墙钟：1439.78秒。

科研问题：原 SPGS 对 ASD/TD 两类 Prompt 的最大概率无法表达反向证据，也没有 NULL 拒绝；RES 缺少证据充分性/必要性约束。DCER 在原 PEN 位置将 8 条冻结 RN50 文本分成 TD/ASD 两组，以差异分数和 NULL 路由竞争，跨 16 个潜在 CAM 子空间硬选至多 4 个，再以共享原 OutputLayer 评估完整、保留和删除视图。16 个 Token 是潜在表征子空间，不是解剖 ROI 或病灶；原 GPT-4 句子多数是一般脑结构/功能描述，不能当成可靠 ASD 病理先验。

对照含义：B0 删除 PEN，B1 保留原 RES，B2 从头重训完整原 SPromptGL，D0-D8 分别检验无文本、无符号方向、稠密路由、无充分/必要约束、类别身份混排、随机数值文本、关闭 NULL、随机同预算路由。论文报告的 91.73/92.08/92.30 是独立论文值，不与本地训练结果混记。

正式四指标均值±十折 population SD（百分比）见下表：

|组|ACC|Hard-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|
|B0|92.0742±2.6000|91.8546±2.5815|89.0793±3.7283|94.6300±4.0457|
|B1|91.2696±3.1421|91.0897±3.0778|88.8354±3.7320|93.3441±4.9921|
|B2|91.7294±2.9505|91.5625±2.9444|89.3415±3.6677|93.7835±3.6749|
|D0|91.6144±3.1771|91.4803±3.0794|89.8293±3.4050|93.1314±5.3846|
|D1|91.8430±3.0666|91.6707±3.0592|89.5671±4.0211|93.7743±4.2842|
|D2|91.8430±2.6507|91.6544±2.6009|89.3171±3.1945|93.9917±4.2078|
|D3|91.6157±2.6800|91.5191±2.6585|90.3232±2.8246|92.7151±3.4059|
|D4|91.1546±3.0506|91.0362±2.9748|89.5793±3.8330|92.4931±5.2301|
|D5|91.9579±3.0901|91.7590±3.1302|89.0915±4.3086|94.4265±3.2394|
|D6|91.6157±2.6800|91.4352±2.6307|89.0915±3.8411|93.7789±4.5987|
|D7|91.6144±3.0498|91.4671±3.0174|89.5854±3.6383|93.3488±4.2787|
|D8|91.7294±2.8130|91.5776±2.7545|89.8110±3.8125|93.3441±4.9006|

D4 相对 B0/B1/B2 与 D0-D8 的逐折配对差见 `/data/adodas/SPromptGL-main/experiment/experiment32_disease_contrastive_evidence_routing/results/report.md`。证据忠实性与负对照结论：NO-GO；D4 选中证据删除相对等预算随机删除的真类 log-prob 下降差为 +0.043672。稀疏注意力本身不构成创新证据。

接入审计：B2 相对未修改 PLN bitwise exact；B0 为 CAM 直接 flatten，B1 为原 RES 文本；D 组完整替换 SPGS+RES。双 Adam、模型、最佳 checkpoint、连续组 RNG、D8 random_step 均写入 `.runtime/` 原子断点。原模型协议与根源码未改，未执行 Git commit/push。

实验目录：`/data/adodas/SPromptGL-main/experiment/experiment32_disease_contrastive_evidence_routing`。代码/保护源码、RN50 Prompt 和控制矩阵 SHA，以及恢复与墙钟信息见 `manifest.json`。限制：val=test 选 checkpoint、外层前 RFE/标准化、871 人 transductive GFR 均沿用历史协议，需独立验证后才能讨论投稿或临床泛化。

## CLIP Encoder Equivalence Audit — 2026-10-21

目的：固定上游源码版本，验证 RN50 预训练 CLIP 在作者 `ModalFusion.apply(my_weight_init)` 顺序下是否被覆盖，并核验当前缓存特征与原始 checkpoint 的等价性。上游版本为 `ba3b9446fa451fde6864d6555bdf324de0c4b64a`；来源文件 SHA、Checkpoint/Tokenizer SHA 和所有数值均见审计 `audit.json`。

结果：作者代码将完整 CLIP 注册为 `PLN.ViT_model`，优化器建立后整棵 ModalFusion 调用 Xavier 初始化；实测总计 40 个 CLIP Linear 权重被重置，其中文本 Transformer 36 个（12层各自的 attention `out_proj.weight`、MLP `c_fc.weight`、`c_proj.weight`），另有视觉 attention pool 4 个投影权重。MHA `in_proj_weight`、bias、Linear bias、`text_projection` 等直接参数未变。A 与 B 初始化前逐位一致；B 初始化后对 A 的逐句余弦均值 0.1029401570558548，最小 0.017229026183485985，最大 0.15867231786251068，最大绝对误差 10.995849609375，文本相似矩阵 Frobenius 差 0.45915505290031433。A/C 逐位相同、最大误差 0；当前缓存复用、buffer 设备移动、dtype、模式、token 顺序均通过。

主干初始化：相同 seed 下 B/C 的 GFR、CAM、OutputLayer 参数哈希全部一致；PEN MLP 的 4 个张量不同，初始化后的 CUDA RNG 状态也不同，原因是上游注册式 CLIP 参与整树 Linear 初始化并消耗随机数。B 的 PLN 有 363 个可训练参数张量、102,582,258 个参数元素，其中 Adam 收录 324 个 CLIP 张量；C 有 39 个参数张量、575,121 个元素，CLIP 特征为 buffer。

影响：实验32 B0 前向绕过整个 PEN 的 SPGS+RES 文本分支，但 PLN 构造仍创建 MLP、加载/注册缓存 buffer，optimizer 仍包含 MLP 参数（B0 前向不给这些参数梯度）；所以它是功能上移除 PEN 路径，不是物理删除 PEN 模块。CLIP 表示差异不直接影响 B0 输出。B1、B2、D4 在本地运行/对照路径使用当前缓存实现，A=C 证明该路径正确，无需重训。B2 应标记为原架构加当前缓存协议，不能与固定上游注册式 CLIP 初始化结果混称。没有证据要求重跑实验1–32，保留现有结果及其协议记录。后续 Prompt 实验固定 RN50 checkpoint/Token SHA，eval + no_grad 编码、冻结缓存，并禁止自定义初始化触及 CLIP。

根目录代码未修改。完整报告：`/data/adodas/SPromptGL-main/experiment/clip_encoder_equivalence_audit/report.md`；机器结果与完整 8×8 相似度矩阵：`/data/adodas/SPromptGL-main/experiment/clip_encoder_equivalence_audit/audit.json`；复现脚本：`/data/adodas/SPromptGL-main/experiment/clip_encoder_equivalence_audit/audit_clip.py`。未训练、未创建实验33，未执行 Git commit/push。

## B0 Physical PEN Removal Equivalence Audit — 2026-10-21

- 状态：完成。只使用 GPU0、真实 ABIDE 全量特征和 StratifiedKFold 第1折训练索引；未运行正式 epoch、未创建实验33、未执行 Git commit/push。
- B0-A：实验32实际运行时 B0。GFR→原始 CAM→flatten→OutputLayer；PLN 构造仍实例化 `MLP(4096,128,9)`、`prompt_text_features` buffer 和 RES 专用 `avg_pool`。
- B0-B：独立审计 PLN 从构造阶段省去 MLP、Prompt buffer、avg_pool；原 GFR、CAM、OutputLayer、simple-2 损失和两个 Adam 优化器保持不变。B 构造期间 RN50加载、`get_prompt_text_features`、MLP 初始化、AvgPool 初始化调用均为0。
- 参数：B0-A PLN 注册575,121（39张量），其中 MLP 为525,577（4张量）；B0-B PLN注册49,544（35张量）。两者 simple-2 反向均有49,538个 PLN 参数元素产生梯度；B0-A MLP四个参数梯度均为None。两边 `fusion_weight` 的6个元素也无梯度。B0-A PLN Adam收录575,121个元素，B0-B收录49,544个；GFR Adam各收录887,836个。
- 同 seed 0 自然初始化时，69个共有参数已逐位相等；显式按名称/形状复制后状态SHA仍相等。构造后的CPU/CUDA RNG状态不同，可能改变后续Dropout随机轨迹；本审计通过保存并恢复完整RNG隔离该效应。
- 前向：eval和train模式下GFR、CAM、分类输出均 `torch.equal`，最大误差0；train Dropout RNG消耗匹配。`FORWARD: EXACT`。
- 反向：分类损失1.032340168952942，图学习损失-1.1294156312942505，总损失-0.0970754623413086；共有参数49个非空梯度张量逐位一致。`BACKWARD: EXACT`。
- Adam：按实验32超参数执行3步实际双优化器更新；共有参数、梯度、step、exp_avg和exp_avg_sq每一步逐位相等，最大误差0。MLP参数未更新且没有Adam状态。`ADAM: EXACT`。
- 负对照：扰动B0-B分类器有效权重令输出改变（最大误差2.8560815）；扰动B0-A闲置MLP权重后输出逐位不变。根核心源码、数据、实验32源码/manifest/报告/120折结果和CLIP专项审计哈希均未改变。
- 判定：`PEN_PHYSICALLY_REMOVED: YES`；`B0_HISTORY_VALID: YES`。保留实验32 B0 `ACC=92.0742%` 作为历史结果；物理无PEN的B0-B可作后续结构基座，但等价审计不意味着新的完整训练必然获得相同分数。B0/B2同折20折实验不是回答本结构问题所必需；性能优越性需独立评估回答。
- 文件：`experiment/b0_pen_physical_removal_equivalence_audit/audit.py`、`results.json`、`report.md`。完整结果、损失/逐参数/RNG/hash记录及复现命令见报告与JSON。

## B0-B Seed 0 Full 10-Fold Retraining — 2026-10-21

- 状态：`COMPLETE`，seed=0，完整10折；GPU0执行fold 1–5，GPU1执行fold 6–10。
- 实验目录：`experiment/b0_b_seed0_full10fold`。正式运行代码、协议/输入哈希、双卡预检、逐折四项指标、均值±总体标准差与B0-A逐折比较见 `run.py`、`manifest.json`、`results/fold_metrics.csv`、`results/summary.json`、`results/report.md`。
- 实现：基于 `b0_pen_physical_removal_equivalence_audit` 已验证的B0-B结构，构造时不创建PEN MLP、CLIP prompt buffer或RES avg_pool；PLN注册49,544参数。两卡预检及训练期追踪均确认RN50/CLIP模型加载0次、prompt特征调用0次；未载入或复制B0-A训练Checkpoint/参数，也未按92.0742%调随机轨迹。
- 协议：沿用实验32原始ABIDE输入及预处理、`[48,6,10,256]`模态维度、`StratifiedKFold(10, shuffle=True, random_state=10)`、全部超参数、simple-2 `ClsLoss + mean(4个图损失)`、双Adam、`val_idx=test_idx`及严格ACC提升的Checkpoint选择。指标为ACC、one-hot hard-label ROC-AUC、SEN-ASD、SPE-TD。
- 两卡并行时每个GPU worker各自以seed=0初始化一次，并在所分配的5折内连续使用RNG；两worker为独立流，因此后半折不强行匹配实验32单worker随机流。B0-A对比仅读取实验32历史 `fold_metrics.csv`；其历史ACC均值92.0742%保持不变。
- B0-B：ACC 91.7294±2.8596%，Hard-AUC 91.6046±2.8059%，SEN 90.0732±3.5117%，SPE 93.1360±4.5354%。相对B0-A逐折配对均值差：ACC -0.3448pp、Hard-AUC -0.2500pp、SEN +0.9939pp、SPE -1.4940pp。该结果是独立随机训练结果，不覆盖实验32历史结果。
- 实验32、ABIDE输入及根核心源码的运行前后SHA-256一致；未修改任何历史实验。未执行Git commit或push。

## Experiment 33 — QCAR Conditional Cross-Modal Alignment

- 状态：`COMPLETE`，seed=0，8组×10折=80/80折；判定`NO-GO`。实验目录：`experiment/experiment33_qcar_conditional_alignment`。
- 研究问题：ANAT6/FUNC10质量代理在控制年龄、性别和站点后，能否预测可由采集条件解释的CORR256成分，并在保留ASD疾病信号的情况下改善预测？这些数据仅构成候选QC关联，不能视为已验证的伪影标签。
- 研究方向：实验18/23/24/25/27/28/29/30/31的语义对齐、辅助校准与疾病证据路由没有稳定优于强基线。本实验转向可检验的质量条件关联，并将FC校准放在GFR前，因为GFR直接用四模态输入构建患者图；校准后再建图可以避免候选QC共享成分先进入图传播。论文/仓库审计参考[SPromptGL MICCAI 2025论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[作者仓库](https://github.com/wanxixi11/SPromptGL)及[当前仓库](https://github.com/tangaii/SPromptGL)。本地实验32当前`simple-2`代码优先于论文公式：保留四图正则均值和分类损失，`ALoss`虽计算但不加进损失；保留GFR中的`i=1`覆盖`i=0`行为、双Adam、`val_idx=test_idx`和严格ACC提升选模。
- 输入：既有871×320 ABIDE数据，ASD=403、TD=468，模态[PHENO48, ANAT6, FUNC10, CORR256]；核对的PHENO列为age[0:26]、site[26:46]、sex[46:48]。数据预处理、RFE、列顺序、标签和`StratifiedKFold(10, shuffle=True, random_state=10)`未更改。只有CORR256在运行时可被校准。
- 核心方法：outer-train内以float64、ridge λ=1拟合C→Q、C→F并残差化，再拟合Q-residual→F-residual。三折训练OOF比较C→F与C+Q→F连接级误差，`gain=(MSE_C−MSE_C+Q)/(MSE_C+1e-12)`、`w=clip(max(gain,0),0,1)`。QCAR以单个gamma从0开始，前向限制[0,0.25]，修正每受试者不超过当前FC范数的15%；PHENO/ANAT/FUNC保持原值。统计拟合不接收标签、只用外层训练样本，不用outer-test FC拟合；GFR→原CAM→原OutputLayer维持不变。
- 八组：G0原始B0-A；G1站点独立预测；G2未经C条件残差化的C+Q预测；G3全连接权重w=1；G4完整条件残差+OOF可靠度；G5 G4预测方向在站点内错配；G6 FC自身C残差收缩；G7与G4相反符号。每个config单独seed=0，fold1–10由同一worker连续执行。
- B0-A复现：G0与实验32历史B0-A逐折初始主干参数哈希10/10相同、初始RNG哈希10/10相同、best epoch 10/10相同、四项指标10/10相同。两者ACC均值92.074190%=92.0742%。未加载或复用历史训练Checkpoint，也没有调整随机轨迹追分。实验32的`B0_seed0.pt`归档状态是`BOUNDARY`，只含结果行和RNG、不含最终模型权重，因此没有声称历史模型张量级比较。
- 双卡工作量：GPU0运行G0/G2/G4/G6共40折、2,102.1 worker秒；GPU1运行G1/G3/G5/G7共40折、1,964.7 worker秒。总计4,066.8 worker秒。第一worker启动至完成的端到端时间921.3秒（15分21秒，包含安全恢复）；最终恢复批次约513秒。训练先后遇到线程入口和诊断行键冲突，7个组的首折训练状态已周期性Checkpoint保存；G0在epoch边界安全停止。修复只调整线程调用、诊断行拼接和早停Checkpoint恢复，Checkpoint签名迁移保留模型/双Adam/RNG/已提交折数据；最终代码重做双卡预检，80折无已成功epoch或fold重训。详情见manifest的`zero_fold_startup_repairs`、`in_progress_repairs`及`checkpoint_signature_migration`。
- 预检：两卡G0首折初始化哈希与历史完全一致；gamma=0时G4/G0输入、GFR、图、CAM、分类输出、损失、共有梯度和双Adam首次更新逐位一致；MLP无梯度；gamma零点梯度0.00029827；84条正可靠度连接的非零gamma影响FC、GFR图和分类输出。outer-test标签与FC扰动不改变G4统计拟合；病人FC修正上界、参数Adam登记、checkpoint往返均验证通过。

| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|
| G0_B0A | 92.0742±2.6000% | 91.8546±2.5815% | 89.0793±3.7283% | 94.6300±4.0457% |
| G1_SITE_ONLY | 91.2709±3.0992% | 91.0740±3.1637% | 88.3598±4.6579% | 93.7882±2.8197% |
| G2_UNCONDITIONAL | 91.3845±3.2677% | 91.2436±3.2079% | 89.5732±4.0032% | 92.9140±5.3294% |
| G3_NO_RELIABILITY | 91.3845±3.1017% | 91.2038±3.0990% | 88.8415±4.2932% | 93.5661±4.1156% |
| G4_QCAR_FULL | 91.4995±3.0130% | 91.3365±2.9863% | 89.0976±3.4981% | 93.5754±3.8430% |
| G5_WRONG_PAIR | 91.6144±3.0498% | 91.4087±3.0473% | 88.6037±3.4759% | 94.2137±3.2156% |
| G6_FC_ONLY | 92.1904±2.5132% | 92.0936±2.5156% | 90.8293±3.6819% | 93.3580±3.5440% |
| G7_REVERSE_SIGN | 91.3845±3.2677% | 91.1691±3.2314% | 88.5732±4.5190% | 93.7650±5.2773% |

- 配对差值均为G4−对照平均百分点，括号为逐折胜/平/负：

| 对照 | ACC | Hard-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|
| G0_B0A | -0.5747 (2/3/5) | -0.5181 (2/3/5) | +0.0183 (2/6/2) | -1.0546 (2/4/4) |
| G1_SITE_ONLY | +0.2286 (4/4/2) | +0.2625 (6/1/3) | +0.7378 (4/5/1) | -0.2128 (3/3/4) |
| G2_UNCONDITIONAL | +0.1149 (3/5/2) | +0.0929 (3/3/4) | -0.4756 (2/5/3) | +0.6614 (3/4/3) |
| G3_NO_RELIABILITY | +0.1149 (2/6/2) | +0.1327 (3/4/3) | +0.2561 (4/4/2) | +0.0093 (2/6/2) |
| G5_WRONG_PAIR | -0.1149 (1/8/1) | -0.0722 (3/5/2) | +0.4939 (3/6/1) | -0.6383 (1/6/3) |
| G6_FC_ONLY | -0.6910 (1/2/7) | -0.7572 (1/2/7) | -1.7317 (0/6/4) | +0.2174 (3/4/3) |
| G7_REVERSE_SIGN | +0.1149 (3/5/2) | +0.1674 (4/3/3) | +0.5244 (4/3/3) | -0.1896 (2/5/3) |

- 机制：G4 OOF平均gain为−0.06071，仅3/10折正；每折平均96.2/256连接具有非零w（范围54–150），平均w=0.01447。G4 gamma十折均到0.25，但均值修正仅FC范数0.127%（范围0.088–0.185%）；高QC残差患者0.233%，其余0.100%。G4比错配G5更低0.1149pp ACC及0.0722pp Hard-AUC，正确配对未显示优势；G6数值FC-only反而领先0.6910pp ACC及0.7572pp Hard-AUC。G4的SEN与G0近似持平，但SPE低1.0546pp。
- 预注册NO-GO：G4低于本轮及历史B0-A的92.0742% ACC（91.4995%，−0.5747pp），Hard-AUC低0.5181pp，未达到+0.23pp ACC目标；G4未超过G5或不弱于G6；总体OOF gain为负。SEN-ASD非劣和实际非零校准虽通过，不能抵消这些失败条件。不得把G1/G2/G7改名为主方法，也没有进行事后调参。
- 交付：`README.md`、`qcar.py`、`integration.py`、`integration.patch`、`run.py`、`manifest.json`、`results/fold_metrics.csv`、`results/diagnostics.csv`、`results/summary.json`、`results/report.md`。已删除八组最终epoch训练Checkpoint和锁文件；保留隔离源码、日志、双卡预检、逐折指标和统计诊断。全部80折唯一、结果代码/数据/历史保护SHA验证通过，handoff此前298216字节前缀SHA `bc6c83769510b54a8eab5e22c781a093ee64226766738f625db4ac73b418c64d` 保持。本次未改根模型、数据或实验1–32，未执行Git commit/push。


## Experiment 42 — Four-link Evidence Validation


科研状态：**DIAGNOSTIC_COMPLETE_NO_GO**。交付状态：**TMP_ONLY**。

本轮没有训练HCER-v3或新的集成分类器。按授权验证并复用E40不可变B0-A十折初始状态及最佳Checkpoint，逐折重算指标；新B0训练0折，验证复用10/10。Q1–Q4各10/10独立fold任务完成，共40个诊断任务。训练内探针及选择器不冒充外层分类实验。

|链|结论|主要对照|下一步|
|---|---|---|---|
|Q1 文献Prompt|NO-GO|P0–P3/数值/中性/交换/错配/Gram匿名|该链尚不足以支持集成创新|
|Q2 FC评分|NO-GO|随机/Top-abs(z)/训练判别/冻结B0导师|该链尚不足以支持集成创新|
|Q3 反事实约束|NO-GO|C0/C1/C2/C3与固定CF权重诊断|该链尚不足以支持集成创新|
|Q4 适用性和QC|GO|正向/反向/随机/站点置乱/恒一|可独立继续验证|

## 正式四指标：仅完整十折B0-A

单位%，mean ± population SD；没有以micro替代折均值。Hard-AUC为原始one-hot硬标签ROC-AUC。ASD=0，TD=1。

|来源|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E32历史训练（只读参照）|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E42验证复用E40最佳模型（非新训练）|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|

逐fold混淆、best epoch、executed epochs、初始化和入口RNG SHA均与E32 B0一致；详见fold_metrics.csv。原PEN完整构造但前向跳过SPGS+RES；原始未调用参数无梯度。

## Q1：文献到FC与四种文本

13个假说中，训练条件效应在至少7折通过预注册方向/站点/幅值规则的数量：3/13。网络级来源不等于指定HO边的直接证据；全部39条的edge_specific_verified仍为false。

|假说|PASS折数|平均条件效应SMD|inner方向稳定率|
|---|---:|---:|---:|
|H01|10|+0.2383|0.9600|
|H02|0|-0.1456|0.1400|
|H03|8|+0.1508|0.8600|
|H04|0|-0.0849|0.2600|
|H05|10|+0.3534|0.9600|
|H06|1|+0.0840|0.7000|
|H07|0|-0.0341|0.4400|
|H08|0|-0.0978|0.2400|
|H09|0|-0.0862|0.3400|
|H10|0|-0.1059|0.1800|
|H11|0|-0.1241|0.2400|
|H12|4|+0.1066|0.7400|
|H13|0|-0.0522|0.3600|

P0是旧文本逐字冻结；P1平衡句式；P2逐条完整HO双端短句分别编码后按假说平均，未截断；P3可反驳条件及exploratory适用背景。四族均有26个概念文本槽，P2每槽含数条边级短句。CLIP RN50固定、eval/no_grad、只编码一次；不是GPT-4 API生成记录。

下面是outer-train内嵌套5折OOF探针，**不是外层正式分类结果**。所有探针相同30维输入/31参数/150 Adam步；共同数值来源和软方向metadata相同。

|风格|train OOF ACC|train OOF Hard-AUC|SPE|SEN|胜过全部匹配控制|
|---|---:|---:|---:|---:|---|
|P0|59.7653|59.0532|68.5895|49.5169|False|
|P1|59.2933|58.5877|68.0432|49.1322|False|
|P2|60.0204|59.3063|68.8742|49.7384|False|
|P3|59.4337|58.7278|68.1857|49.2700|False|

每族对控制的同fold差（pp）及ACC/AUC W/T/L：

|族|控制|ΔACC|ΔHard-AUC|ACC W/T/L|AUC W/T/L|
|---|---|---:|---:|---|---|
|P0|NEUTRAL|+0.1275|+0.1011|4/0/6|4/0/6|
|P0|SWAP_ASD_TD|-0.1787|-0.1934|2/0/8|2/0/8|
|P0|DERANGED_HYPOTHESIS|+3.0234|+3.3075|10/0/0|10/0/0|
|P0|GRAM_MATCHED_ANONYMOUS|-0.1532|-0.1678|4/2/4|4/2/4|
|P0|NUMERIC_ONLY|-2.8065|-3.0622|0/0/10|0/0/10|
|P1|NEUTRAL|-0.3445|-0.3645|3/0/7|3/0/7|
|P1|SWAP_ASD_TD|-0.0510|-0.0512|4/1/5|4/0/6|
|P1|DERANGED_HYPOTHESIS|+2.5643|+2.8789|10/0/0|10/0/0|
|P1|GRAM_MATCHED_ANONYMOUS|-0.7400|-0.7401|0/0/10|0/0/10|
|P1|NUMERIC_ONLY|-3.2785|-3.5277|0/0/10|0/0/10|
|P2|NEUTRAL|+0.3827|+0.3541|7/0/3|7/0/3|
|P2|SWAP_ASD_TD|+0.0382|+0.0260|4/2/4|4/2/4|
|P2|DERANGED_HYPOTHESIS|+2.8448|+3.1284|10/0/0|10/0/0|
|P2|GRAM_MATCHED_ANONYMOUS|+0.0382|+0.0240|5/3/2|6/1/3|
|P2|NUMERIC_ONLY|-2.5513|-2.8091|0/0/10|0/0/10|
|P3|NEUTRAL|-0.2040|-0.2243|4/1/5|4/0/6|
|P3|SWAP_ASD_TD|+0.1275|+0.1263|4/2/4|5/1/4|
|P3|DERANGED_HYPOTHESIS|+3.2404|+3.5289|10/0/0|10/0/0|
|P3|GRAM_MATCHED_ANONYMOUS|-0.5996|-0.5981|0/0/10|0/0/10|
|P3|NUMERIC_ONLY|-3.1381|-3.3876|0/0/10|0/0/10|

可识别性限制：对允许自由线性系数的低容量读出，静态交换ASD/TD特征槽可被系数重参数化吸收；交换控制接近不等于医学类别正确。四风格只是冻结候选，训练内选择不支持绝对“最优Prompt”。结合文献连接级不确定性，不能把可编码/有梯度当作疾病信息。

## Q2：冻结原始分类器的FC依赖

|图算子|训练内选定风格删除|随机删除|Top-abs(z)删除|相对随机差/胜折|相对Top-abs(z)差/胜折|
|---|---:|---:|---:|---|---|
|live|0.079938|0.033208|0.097882|+0.046730 / 9/10|-0.017943 / 2/10|
|fixed|0.079940|0.033211|0.097887|+0.046729 / 9/10|-0.017947 / 2/10|

表为K=8、完整TD条件mu替换的真实类logp下降。score_fidelity.csv保留K=4/8、三随机掩码均值来源、四种S4风格、S1/S2/S3、逐患者原预测正确/错误、预测类/真实类效应、半强度及匹配训练TD替换。S3只用16名训练者逐边探测，评价在不重叠的outer-test人群；不是个人oracle的自评。

每次只修改目标人的raw_X[person,64+edge_id]，其余人的输入和其它281维保持不变，完整重过GFR→CAM→OutputLayer。固定邻接关开用于分解图重连，关闭时严格相同。selected-keep只保留39候选中的选中边；其它217条FC和其它模态始终存在，不能说“只用8条FC预测”。

## Q3：选择器反事实学习，B0始终冻结

|组|heldout真实类删除影响|与C0差|方向导师rho|与C0差|删除/方向胜折|
|---|---:|---:|---:|---:|---|
|C0|0.112415|+0.000000|-0.00350|+0.00000|0/10 / 0/10|
|C1|0.112358|-0.000056|-0.00359|-0.00009|1/10 / 4/10|
|C2|0.112522|+0.000107|-0.00316|+0.00034|2/10 / 5/10|
|C3|0.112310|-0.000105|-0.00342|+0.00009|2/10 / 4/10|
|C3_FIXED_DIAGNOSTIC|0.111467|-0.000948|-0.00493|-0.00143|2/10 / 3/10|

C0拟合训练B0单边依赖导师；C1加keep，C2加delete，C3两者。错误/低置信训练预测不参加必要性CF；原分类器所有参数无优化步骤，无法用破坏分类器换删除下降。5折OOF按训练导师患者划分，未用同人拟合后在同人报告选择定位。

C1–C3预注册将纯CF梯度最多缩到teacher范数的0.25；C3_FIXED_DIAGNOSTIC保留原.10/.05固定权重、相同步数和患者预算，用于量级比较，不据外层test选权重。hard TopK的真实有限差分是分段常数，STE是代理梯度；数值有限差分验收针对连续松弛与固定图，而非伪称hard导数等于STE。

|组|步骤|探测次数/有效|原CF/teacher平均比|应用后平均比|梯度cos均值|
|---|---:|---|---:|---:|---:|
|C0|1500|0/0|0.00000|0.00000|0.00000|
|C1|1500|600/460|0.13464|0.05257|0.01794|
|C2|1500|600/460|0.10227|0.07384|0.01029|
|C3|1500|600/460|0.22994|0.09522|0.01728|
|C3_FIXED_DIAGNOSTIC|1500|600/460|0.22920|0.22920|0.00595|

## Q4：背景适用性与质量角色

|角色|控制后风险/FC偏离rho|风险/计算不稳定rho|满足代理方向折数|平均选择数|弃权率|支持|
|---|---:|---:|---:|---:|---:|---|
|R1|-0.02558|-0.16327|0/10|3.350|0.3812|False|
|R2|+0.14558|+0.21235|8/10|4.925|0.1250|True|
|R3|-0.00741|-0.06206|0/10|5.325|0.0688|False|

R1只以训练TD背景支持度调节参考可比性/弃权，不产生疾病支持；R2只用已验证方向的DVARS/outlier/mean FD；R3只用CNR/FBER/QI1/SNR，未将EFC/FWHM/GSR统一当坏质量。原始16列QC与871人CSV匹配后的正仿射关系在preflight逐列验收。FUNC GSR是ghost-to-signal ratio，不是是否做global signal regression。

门控仅进入可解释的证据keep/弃权诊断，没有新分类残差；其影响可从选边数、拒绝及冻结模型keep输出核验，不被单位范数抵消。它不改变原B0常规预测，不能据此宣称分类增益。FC-only噪声不改变真实QC，可靠性不能假装感知物理头动；压力测试只评价计算稳定和过度置信风险。R4只在训练内角色通过全十折门槛后组合，未通过的角色不纳入。

## 审计、资源、限制与判断

设备：0, PPU-ZW810E, 98304 MiB
1, PPU-ZW810E, 98304 MiB。诊断任务GPU0/GPU1={'0': 20, '1': 20}；四阶段各卡各5fold；B0额外验证10fold在GPU0，原训练复用未计新训练。各卡诊断任务墙钟之和（并发，不能相加当总耗时）={'0': 1597.0092051029205, '1': 1569.2408752441406}秒。本次总墙钟1669.86秒。

源码/数据/Prompt/CLIP/token/split/anchor SHA、每任务日志、锁、原子检查点、resume记录见manifest及runtime。每个训练选择器保存optimizer/RNG/step/fit/valid/config SHA；已完成任务按唯一(phase,fold)跳过。根模型、E32/E38/E39/E40和数据只读，最终SHA不变。

局限：原val=test选模、CV前RFE和标准化、全队列transductive图均保留；诊断模型已受历史测试选模影响。10fold训练集重叠，W/T/L是筛选描述，不是10个独立数据集的显著性。参考是统计TD条件估计，遮蔽/替换有分布偏移，不是临床正常线或脑区因果试验。

逐环节结果优先于创新叙述；任何一环NO-GO时不训练集成分类模型。局部模型依赖、语言几何或质量相关均不足以证明正确医学语义带来独立诊断收益。

文献用于定义检验边界：[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[ERASER](https://aclanthology.org/2020.acl-main.408/)、[GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html)、[ROAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)、[ABIDE多站点](https://pmc.ncbi.nlm.nih.gov/articles/PMC3782703/)、[FC可复现性](https://pubmed.ncbi.nlm.nih.gov/31285817/)、[非复制](https://pubmed.ncbi.nlm.nih.gov/31916675/)、[QC字段定义](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)。PMC6591952和PMC7268009页面遇验证码，实际采用对应PubMed身份/摘要核验，不伪称已读取被拦截全文。

实际工作目录：`/tmp/SPromptGL-E42-stage/experiment/experiment42_fourlink_evidence_validation`。/tmp不是持久存储保障，交付状态与科研完成分开记录。


### Experiment 42 — 最终质量与交付验收补充

E42_FINAL_QA_ce03fb5f5c9a05f7ca53033b9e5aea1a13d59827691d62240b6b940a06ba9289

四链结论{'Q1': 'NO-GO', 'Q2': 'NO-GO', 'Q3': 'NO-GO', 'Q4': 'GO'}；Q1–Q4各10折，共40/40诊断任务，GPU0/GPU1各20。补充QA各5任务，十折选择器下一Adam更新逐值复演通过。B0为10折验证复用，新增正式分类训练0折。最终追加前总墙钟3600.72秒。

Q1仅3/13假说稳定受支持；Q2选中删除0.079938、随机0.033208、Top-|z|0.097882，胜Top-|z|仅2/10折。Q3 C3对C0删除差−0.00010488，仅2/10折胜，未改善定位。Q4仅FUNC质量代理通过8/10折，PHENO/ANAT均0/10；没有多模态门控联合分类收益证明。结论DIAGNOSTIC_COMPLETE_NO_GO，不训练集成候选。

公开更正诊断标签缺陷：R0/R2/R3不读取SITE，site-only置乱应恒等；原运行误用random rank，原始行保留，最终CSV按恒等性质重建并标注，训练与GO判定不变。补充恢复检查首次失败为克隆Adam CPU step字典别名，深拷贝和RNG恢复后通过，正式checkpoint未修改。

最终工作目录/tmp/SPromptGL-E42-stage/experiment/experiment42_fourlink_evidence_validation；正式交付目录/data/adodas/SPromptGL-main/experiment/experiment42_fourlink_evidence_validation。初次/data健康检查EIO，主体同步时已恢复；本次对更新产物做有界SHA校验同步。保留/tmp唯一完整副本及恢复证据，不自动git操作。完整结果见results/report.md、manifest.json及各诊断CSV。


<!-- HANDOFF_ALL_EXPERIMENTS_RECONCILED_20261026 -->
## 实验1–42交接总账与缺项补齐（2026-10-26）

本节是追加的统一索引与最终状态校正，不删除、不改写上方历史正文。核对时间：2026-10-26T08:55:20.301596+00:00。实际读取各实验本地或/tmp的README、最终报告、manifest及可用逐折CSV；不是新训练，也不是重新发布GitHub。所有新表直接从现有CSV重算，仅展示ACC / Hard-AUC / SPE-TD / SEN-ASD。补齐原先缺少独立章节的实验6/7/20与34–41；实验42在此前正文基础上登记最终交付状态。上方阶段记录、早期失败或待同步记录仍保留，其最终状态以带日期的最终验收和本节为准。

更新前原handoff：316,990字节，SHA256=`cd658893c6b837e4e621761cde89409de0d213306c6e3a31722f32244dac60fb`。原文完整备份及本次来源/重算核验位于`/tmp/SPromptGL-handoff-reconcile-9ixtl42v`；本节仅追加，不重写历史。

### 统计、基线和解释边界

- 实验1–28多数采用seed=0/1/2；下方实验6/7/20的“±”是先对每seed十折平均，再对3个seed均值计算population SD，绝不是30fold SD。实验29起此处各正式表是seed0十折均值±十折population SD，单位%。早期CSV保留的四舍五入精度不人为补造。
- 指标顺序固定ACC、原代码硬预测one-hot Hard-AUC、SPE-TD、SEN-ASD。ASD=0，TD=1；部分早期CSV把TD识别率存于`specificity_asd`，这里按其实际混淆含义统一叫SPE-TD。旧报告中的prob-AUC/F1/BA记录仍保留，但本节不把它们增加为新的正式指标。
- E32 B0即B0-A：完整构造原GFR/PLN及原PEN闲置层、avg_pool与8条CLIP缓存，保持初始化/RNG轨迹；CAM后绕过SPGS+RES，144维直接进入原OutputLayer。未调用PEN层只是兼容脚手架，不是参与前向的新模块。物理删除PEN的B0-B/旧E38 B1与完整SPromptGL B2/P0均不得冒充B0-A。
- 原871×320、PHENO48/ANAT6/FUNC10/CORR256、ASD403/TD468、全患者transductive图、GFR `i==1`覆盖、分类损失＋四图GraphConstructLoss均值、ALoss不反传、双Adam均按历史记录。ANAT/FUNC是QC/头动摘要，不是疾病病灶；GFR输出已混合，CAM4×36不是ROI。
- `val=test`严格首次最佳ACC选模、CV前RFE/标准化和transductive传播带来乐观估计风险。十折训练集重叠；这些结果用于历史协议内部比较，不是10个独立数据集、独立临床泛化或生物因果证据。
- /tmp来源表示本次确实能读取的临时副本，不代表已经安全同步至/data；/tmp没有持久保障。本次只更新handoff，不移动实验包、不清理唯一副本、不重训、不commit/push。旧运行时的I/O失败仍作为历史事实保留。


### 全实验唯一索引（含停止、诊断和未完成状态）

|编号|研究|最终或当前可证实状态|主要结论|
|---|---|---|---|
|E01|MASPE 模态感知语义Prompt|COMPLETE；120新折，3 seeds|P1相对原P0小幅变化，未证明医学语义必要性|
|E02|CSMP 对比状态Prompt|COMPLETE；60新折，3 seeds|V2/V2S未稳定超过冻结P1，状态文字与对应关系缺少独立收益|
|E03|PEPCA Prompt与原型塌缩诊断|COMPLETE；150折，3 seeds|识别Softmax→Max、原型趋同与梯度路径问题；诊断不是创新成立证据|
|E04|DAFCP 疾病感知FC Prompt|COMPLETE；60新折，3 seeds|正确疾病语义未稳定胜过原P1及匹配中性文本|
|E05|SPCR 语义投影塌缩修复|COMPLETE；120折，3 seeds|调整投影/正则未建立稳定独立医学收益|
|E06|语义方向保持门控|COMPLETE；60新折，3 seeds|有符号门控、幅值门控均未超过冻结P1|
|E07|双主题语义融合|COMPLETE；120新折，3 seeds|真实双主题未胜均值/错配等控制；医学对应必要性未建立|
|E08|SDPA 共享疾病原型对齐|COMPLETE；150新折，3 seeds|疾病对齐与现有信息冗余，未取得稳健互补收益|
|E09|PDAI 疾病/对齐交互|COMPLETE；240新折，3 seeds|E9为后续强参照之一，正确医学语义的不可替代性未证明|
|E10|DSSC 疾病语义特异性|COMPLETE；60新折，3 seeds|真实ASD、ADHD错误语义与中性控制差异微小且seed混合|
|E11|FC256/受试者身份恢复|COMPLETE；映射审计，无新分类训练|256/256数值身份；252双端命名，195/225/232/236未知|
|E12|AGESA ROI语义对齐可行性|PILOT_COMPLETE；工程PASS，语义NOT_SUPPORTED|不能将可编码、可反传当作医学语义有效；非正式外层完整分类矩阵|
|E13|个体化FC Prompt|COMPLETE；240折，3 seeds|患者对应、ROI身份及独立语义价值均NOT_SUPPORTED|
|E14|层级语义Prompt|COMPLETE；修正后的420折，3 seeds|以修正后14组结果为准；层级语义未建立独立收益|
|E15|AARSP 解剖锚定ROI提示|COMPLETE；240折，3 seeds|原始FC decoder旁路/门控饱和；文本变化常未传至预测|
|E16|ROI语义通路修复|COMPLETE；300折，3 seeds|修复后支路实际生效但更弱；不是正确语义独立贡献|
|E17|DS3A-v2 选择性共享/特有对齐|COMPLETE；360折，3 seeds|类别原型趋同、真实文本交换零翻转；NO-GO|
|E18|DS3A-v3 判别性保护对齐|COMPLETE；300新折，3 seeds|辅助人口学信息不足、稳定互补证据未建立；NO-GO|
|E19|AG-CDEA 解剖锚定疾病对比|COMPLETE；420折，3 seeds|自由Text-Key可重参数化；真实先验未过中性/错误对照|
|E20|疾病概念与连接图对齐|COMPLETE；600折，3 seeds|局部/连接图/E9叠加三线均未过正确概念必要性门槛|
|E21|证据约束自适应对齐|COMPLETE；600折，3 seeds|三线对照仍不支持独立语义收益；旧checkpoint清理后不可做精确事后分解|
|E22|概念条件化有符号FC|COMPLETE；540新折，3 seeds|有符号证据与语义图未过稳定性控制；不重复当新方案|
|E23|GC-SRA 增益校准关系对齐|COMPLETE；240折，3 seeds|最终为COMPLETE/阴性；后补早期PREFLIGHT_BLOCKED不是最终状态|
|E24|CARA 冲突感知关系对齐|PILOT_STOP；48/48 pilot折|冻结门槛全失败；剩余192计划折未训练，不能记240折完成|
|E25|EPAGC 非对称图校准|COMPLETE；180新折＋30只读参考折|方向/强度校准未超过强参照；终止该路线|
|E26|SDER 有符号疾病证据路由|COMPLETE；180新折，3 seeds|SCIENTIFIC_NEGATIVE；I/O曾恢复并完成历史补记|
|E27|EGRP-CAM 医学关系可识别性|STOP_UNGROUNDED_INPUTS；0训练折|Gate 0只找到0条独立跨疾病测量关系，低于2条要求；科学NOT_EVALUABLE|
|E28|IA-FDSA 影像锚定细粒度语义|COMPLETE；300折＋E10补充30折，3 seeds|粗/细教师及辅助交互仍NO-GO；E10不推翻原判断|
|E29|连接语义选择性辅助修正|COMPLETE；200新折，seed0|正确ROI/方向/患者及数值控制未共同通过；NO-GO|
|E30|类别条件语义共享修复|COMPLETE；180新折，seed0|F1/C1均未通过全部冻结强对照；NO-GO|
|E31|SCORE 个体连接证据|COMPLETE；180折，seed0|患者、文本、数值及辅助控制不足以支持独立价值；NO-GO|
|E32|DCER 疾病对比证据路由|COMPLETE；120折，seed0|确立B0-A强参照；D4潜在Token竞争与删除仍NO-GO|
|E33|QCAR QC–FC条件校准|COMPLETE；80折，seed0|G4不胜B0-A，错误对应和FC-only可更好；NO-GO|
|E34|SeRCA 语义关系差分|COMPLETE；100折，seed0|残差极弱，FULL等同CORR-only的ACC，医学互补未证实|
|E35|SeRCA-v2 联合异常关系模式|COMPLETE；120折，seed0|尺度修复后仍未胜数值/随机/错配及强基座；NO-GO|
|E36|PCPR 表型条件患者路由|COMPLETE；220折，seed0|T0 ACC等于B0-A，文本关闭不改变类别；NO-GO|
|E37|LSPCR 标签监督表型路由|COMPLETE；140折，seed0|训练对监督有梯度但G05不胜基线/文本和标签错配；NO-GO|
|E38|LCHER 原版＋B0-A锚定重跑|两版各100唯一折；分别归档|原版物理删PEN参照不当；严格重跑仍NO-GO，禁止混算两版|
|E39|HASE-MoE 异质机制专家|COMPLETE；修正后100折，seed0|E9明显低于B0-A/普通MoE/患者错配；NO-GO|
|E40|HCER-v2 竞争性FC证据|COMPLETE；100折，seed0|分类、医学语义及Top-abs(z)证据特异性均NO-GO|
|E41|UCSMA-v2 不确定性语义多尺度|PREFLIGHT_PASS；当前0正式完成折|仅有预检证据；未完成E0十折验收和正式结果，不能给出科学成败|
|E42|四链独立机制验证|DIAGNOSTIC_COMPLETE_NO_GO；40诊断任务|Q1/Q2/Q3 NO-GO；Q4仅FUNC代理局部GO；复用验证10锚点、新分类训练0折|


### 本次实际读取的来源目录

下列路径是本机现有文件的来源，不把远程仓库文件假称为本机已执行结果。逐折明细、best epoch、混淆计数和原始任务状态保留在各来源目录；此处不重复存储数千行原始结果。


- E01：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt`


- E02：`/data/adodas/SPromptGL-main/experiment/experiment2_contrastive_state_prompts`


- E03：`/data/adodas/SPromptGL-main/experiment/experiment3_prompt_effectiveness_diagnostics`


- E04：`/data/adodas/SPromptGL-main/experiment/experiment4_disease_aware_fc_prompt`


- E05：`/data/adodas/SPromptGL-main/experiment/experiment5_semantic_projection_collapse`


- E06：`/data/adodas/SPromptGL-main/experiment/experiment6_semantic_direction_preserving_gate`


- E07：`/data/adodas/SPromptGL-main/experiment/experiment7_dual_theme_semantic_fusion`


- E08：`/data/adodas/SPromptGL-main/experiment/experiment8_scheme1_shared_disease_alignment`


- E09：`/data/adodas/SPromptGL-main/experiment/experiment9_prompt_disease_adversarial_interaction`


- E10：`/data/adodas/SPromptGL-main/experiment/experiment10_disease_semantic_specificity`


- E11：`/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery`


- E12：`/data/adodas/SPromptGL-main/experiment/experiment12_roi_semantic_alignment_feasibility`


- E13：`/data/adodas/SPromptGL-main/experiment/experiment13_patient_specific_fc_prompt`


- E14：`/data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt`


- E15：`/data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt`


- E16：`/data/adodas/SPromptGL-main/experiment/experiment16_roi_semantic_pathway_repair`


- E17：`/data/adodas/SPromptGL-main/experiment/experiment17_disease_semantic_selective_alignment`


- E18：`/data/adodas/SPromptGL-main/experiment/experiment18_discriminability_preserving_selective_alignment`


- E19：`/data/adodas/SPromptGL-main/experiment/experiment19_contrastive_disease_evidence_alignment`


- E20：`/tmp/spromptgl-publish26/experiment/experiment20_grounded_concept_graph_alignment`


- E21：`/data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment`


- E22：`/data/adodas/SPromptGL-main/experiment/experiment22_concept_conditioned_signed_connectivity`


- E23：`/tmp/spromptgl-publish26/experiment/experiment23_gain_calibrated_selective_relational_alignment`


- E24：`/data/adodas/SPromptGL-main/experiment/experiment24_conflict_aware_relational_alignment`


- E25：`/data/adodas/SPromptGL-main/experiment/experiment25_evidence_preserving_asymmetric_graph_calibration`


- E26：`/tmp/spromptgl-publish26/experiment/experiment26_signed_disease_evidence_routing`


- E27：`/data/adodas/SPromptGL-main/experiment/experiment27_evidence_grounded_relational_prompt_cam`


- E28：`/data/adodas/SPromptGL-main/experiment/experiment28_imaging_anchored_finegrained_semantic_alignment`


- E29：`/data/adodas/SPromptGL-main/experiment/experiment29_connectivity_semantic_selective_auxiliary`


- E30：`/data/adodas/SPromptGL-main/experiment/experiment30_class_conditional_semantic_shared_repair`


- E31：`/data/adodas/SPromptGL-main/experiment/experiment31_subject_specific_connectomic_evidence`


- E32：`/data/adodas/SPromptGL-main/experiment/experiment32_disease_contrastive_evidence_routing`


- E33：`/data/adodas/SPromptGL-main/experiment/experiment33_qcar_conditional_alignment`


- E34：`/tmp/SPromptGL-E34-SeRCA-stage-20261022/experiment/experiment34_serca_semantic_relational_alignment`


- E35：`/tmp/SPromptGL-E35-SeRCA-v2-stage/experiment/experiment35_serca_v2_relational_pattern_alignment`


- E36：`/tmp/SPromptGL-E36-PCPR-stage/experiment/experiment36_phenotype_conditioned_patient_routing`


- E37：`/tmp/SPromptGL-E37-LSPCR-stage/experiment/experiment37_label_supervised_phenotype_routing`


- E38：`/tmp/SPromptGL-E38-B0A-stage/experiment/experiment38_lcher_b0a_anchor_rerun`


- E39：`/tmp/SPromptGL-E39-HASE-stage/experiment/experiment39_heterogeneity_aware_semantic_experts`


- E40：`/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence`


- E41：`/tmp/SPromptGL-E41-UCSMA-stage/experiment/experiment41_uncertainty_calibrated_semantic_multiscale_alignment`


- E42：`/data/adodas/SPromptGL-main/experiment/experiment42_fourlink_evidence_validation`


- E38原版另存：`/tmp/SPromptGL-E38-LCHER-stage/experiment/experiment38_lcher_hypothesis_evidence_routing`，与B0-A重跑相互独立。
- E28补充E10：`/tmp/experiment28_supplement_e10_coarse_full`；历史正文已记录30新增折、原E28 NO-GO不变。


### Experiment 6 — 语义方向保持门控（补齐）

冻结P1/E5-E0的30参考折，新增B1有符号门控、B2幅值门控，各3seed×10fold，共60新折和240条诊断。两者均沿用原Prompt/分类器与历史协议。下表B0是只读参考，不是本轮新增训练。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|B0|91.8047 ± 0.1084|91.6163 ± 0.1210|94.0641 ± 0.2637|89.1687 ± 0.4237|
|B1|91.7283 ± 0.1622|91.5867 ± 0.1785|93.3457 ± 0.1756|89.8293 ± 0.4082|
|B2|91.7670 ± 0.1951|91.5240 ± 0.1913|94.7071 ± 0.1985|88.3415 ± 0.1993|


三seed总体B1相对B0的ACC/Hard-AUC约−0.08/−0.03pp，B2约−0.04/−0.09pp；方向和幅值调整都未证明稳定增益。门控非零、符号变化不等于疾病语义具有临床解释力。结论：该修复未建立独立收益。


### Experiment 7 — 双主题语义融合（完整补记）

F0为冻结E5-E0/P1的30参考折；F1真实双主题、F2模态主题错配、F3均值主题、F4固定0.5融合。F1–F4共12任务/120新增折，GPU0/GPU1各60折，480诊断、150反事实；统一新增9,378参数。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|F0|91.8047 ± 0.1084|91.6163 ± 0.1210|94.0641 ± 0.2637|89.1687 ± 0.4237|
|F1|91.6913 ± 0.2861|91.4930 ± 0.3012|94.0641 ± 0.2050|88.9228 ± 0.4728|
|F2|91.6140 ± 0.1626|91.4583 ± 0.2153|93.4983 ± 0.7317|89.4187 ± 1.1093|
|F3|91.7673 ± 0.0542|91.5993 ± 0.0764|93.7820 ± 0.3550|89.4167 ± 0.5012|
|F4|91.5370 ± 0.0544|91.3320 ± 0.0689|94.0672 ± 0.2703|88.5955 ± 0.3983|


F1−F0 ACC约−0.11pp、Hard-AUC约−0.12pp；30配对折ACC W/T/L=6/15/9、Hard-AUC=6/10/14。固定checkpoint主题交换平均类别翻转约0.04%，关闭分支同样接近不变。真实文本未优于均值/错配控制，不能把微小隐表示响应解释为医学对应必要性。结论：NO-GO。


### Experiment 20 — Grounded Concept Graph Alignment（完整补记）

实际完成20组×3seed×10fold=600/600折，60/60任务、600诊断、1,080反事实；GPU0/1各300折，墙钟3,608.47秒。使用E19四概念（72/56/45/32条、允许重叠）与E11可信FC映射；4条未命名连接不进入可解释分支。局部证据节点和line graph节点对应FC连接，不是把患者图冒充ROI图。无自由Text-Key以降低语义身份被重参数化的问题。

T00为P0，T01为P3，T02为P3+D/E9，T03加GRL；T04–T09检验数值/真实/中性/错误解剖/等距/匿名局部证据；T10–T13检验数值/真实/中性/错误图证据；T14–T16为E9＋数值/真实/中性局部，T17–T19为E9＋数值/真实/中性图证据。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|T00|91.6910 ± 0.0542|91.5394 ± 0.0841|93.4181 ± 0.5102|89.6606 ± 0.6456|
|T01|91.5378 ± 0.2167|91.3101 ± 0.2156|94.3540 ± 0.5641|88.2663 ± 0.5102|
|T02|91.8826 ± 0.1954|91.6915 ± 0.1929|94.1351 ± 0.2707|89.2480 ± 0.1193|
|T03|91.8443 ± 0.0939|91.6835 ± 0.1218|93.7064 ± 0.2687|89.6606 ± 0.5111|
|T04|91.5378 ± 0.1084|91.3633 ± 0.1232|93.5600 ± 0.3512|89.1667 ± 0.4686|
|T05|91.6527 ± 0.1084|91.4914 ± 0.1176|93.5661 ± 0.3456|89.4167 ± 0.4686|
|T06|91.5387 ± 0.1435|91.3355 ± 0.1573|93.9963 ± 0.4589|88.6748 ± 0.6187|
|T07|91.4995 ± 0.1626|91.3216 ± 0.1905|93.6371 ± 0.2716|89.0061 ± 0.6458|
|T08|91.4228 ± 0.0542|91.2693 ± 0.0530|93.2054 ± 0.2028|89.3333 ± 0.2016|
|T09|91.4228 ± 0.1084|91.2580 ± 0.1215|93.3472 ± 0.5231|89.1687 ± 0.6531|
|T10|91.8439 ± 0.1882|91.6555 ± 0.1636|94.1381 ± 0.4372|89.1728 ± 0.1195|
|T11|91.8064 ± 0.1438|91.5998 ± 0.1487|94.3540 ± 0.2050|88.8455 ± 0.3421|
|T12|91.5003 ± 0.1622|91.3193 ± 0.1541|93.6386 ± 0.2651|89.0000 ± 0.2329|
|T13|91.5761 ± 0.1084|91.3790 ± 0.1220|93.9978 ± 0.3064|88.7602 ± 0.5022|
|T14|91.8826 ± 0.0542|91.6935 ± 0.1010|94.1351 ± 0.5627|89.2520 ± 0.7643|
|T15|91.8447 ± 0.1632|91.7120 ± 0.1441|93.3488 ± 0.5250|90.0752 ± 0.3521|
|T16|91.6527 ± 0.1434|91.4470 ± 0.1678|94.2152 ± 0.1756|88.6789 ± 0.5091|
|T17|91.6910 ± 0.0542|91.5525 ± 0.0682|93.2778 ± 0.2741|89.8272 ± 0.3507|
|T18|91.6527 ± 0.0542|91.4510 ± 0.0527|94.0626 ± 0.5015|88.8394 ± 0.5372|
|T19|91.4999 ± 0.1619|91.3400 ± 0.1848|93.4258 ± 0.2050|89.2541 ± 0.5072|


三条研究线均未建立正确医学语义不可替代性。T15真实局部相对E9/T02的ΔACC=−0.0379pp、ΔHard-AUC=+0.0205pp，SEN约+0.8272pp但SPE约−0.7863pp；T18真实图更弱。真实局部/图注意力熵约0.96–0.97；中性文本交换可不改变预测，而患者FC错配影响更大。结论：NO-GO，不把局部FC利用或概念图本身再次包装成新创新。


### Experiment 34 — SeRCA Semantic-guided Relational Complementary Alignment

100/100折，10组seed0十折；GPU0/GPU1=50/50；墙钟1,377.80秒，worker累计4,764.56秒。复用E19四个冻结概念与可信连接；在原四张患者图支持上形成疾病条件消息，取`M_sem−M_base`，再将小残差注入原CAM路径；不新学患者主图、不强迫模态特征相等。B0-A保留原构造、绕过PEN。S0重新训练并与E32逐折10/10匹配。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|S0_B0A|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|S1_NUMERIC_GLOBAL|92.0742 ± 2.6503|91.8972 ± 2.6233|94.2091 ± 3.8720|89.5854 ± 3.4710|
|S2_RANDOM_CONCEPTS|91.6157 ± 2.8704|91.4377 ± 2.7959|93.5523 ± 4.9698|89.3232 ± 3.5531|
|S3_SINGLE_CONCEPT|91.7294 ± 2.9505|91.5066 ± 2.9408|94.4218 ± 4.3370|88.5915 ± 4.3332|
|S4_CORR_REL_ONLY|91.9592 ± 2.5772|91.7529 ± 2.5794|94.4265 ± 3.3763|89.0793 ± 3.5652|
|S5_AUX_REL_ONLY|91.5008 ± 3.0122|91.3421 ± 2.9315|93.3488 ± 4.8764|89.3354 ± 3.3396|
|S6_UNIFORM_RELATION|91.3845 ± 3.0154|91.2433 ± 3.0129|93.1452 ± 3.1907|89.3415 ± 3.2911|
|S7_DIRECT_MESSAGE|91.2696 ± 2.8788|91.0230 ± 2.8346|94.2044 ± 4.7581|87.8415 ± 4.1970|
|S8_PATIENT_PERMUTED|92.0729 ± 3.2341|91.8870 ± 3.2237|94.4265 ± 3.6345|89.3476 ± 3.7841|
|S9_SERCA_FULL|91.9592 ± 2.4726|91.7722 ± 2.4317|94.2091 ± 3.5038|89.3354 ± 2.7208|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|S9_SERCA_FULL − S0_B0A|-0.1149|-0.0824|-0.4209|+0.2561|1/7/2|2/4/4|
|S9_SERCA_FULL − S1_NUMERIC_GLOBAL|-0.1149|-0.1250|+0.0000|-0.2500|1/7/2|2/6/2|
|S9_SERCA_FULL − S2_RANDOM_CONCEPTS|+0.3435|+0.3345|+0.6568|+0.0122|5/3/2|5/3/2|
|S9_SERCA_FULL − S3_SINGLE_CONCEPT|+0.2299|+0.2656|-0.2128|+0.7439|4/4/2|5/3/2|
|S9_SERCA_FULL − S4_CORR_REL_ONLY|+0.0000|+0.0194|-0.2174|+0.2561|3/4/3|4/2/4|
|S9_SERCA_FULL − S5_AUX_REL_ONLY|+0.4585|+0.4302|+0.8603|+0.0000|5/3/2|6/2/2|
|S9_SERCA_FULL − S6_UNIFORM_RELATION|+0.5747|+0.5289|+1.0638|-0.0061|5/4/1|6/1/3|
|S9_SERCA_FULL − S7_DIRECT_MESSAGE|+0.6897|+0.7493|+0.0046|+1.4939|7/2/1|7/2/1|
|S9_SERCA_FULL − S8_PATIENT_PERMUTED|-0.1136|-0.1148|-0.2174|-0.0122|3/3/4|3/2/5|


机制与结论：S9实际相对残差约0.0354%；ACC与仅CORR来源S4相同，未证明辅助关系提供互补增量。S9不胜B0-A、纯数值S1或患者错配S8；NO-GO。后续审计指出S8置乱的训练折统计边界不足，S7直接消息与差分组实际剂量相差约67.6倍，因此不能将S9>S7归因于差分形式本身。训练结束时/data实验目录与根handoff写入均遭EIO；当前可读取完整/tmp副本，上述路径不等于已同步正式目录。


### Experiment 35 — SeRCA-v2 Disease-Semantic Relational Pattern Alignment

120/120折，12组seed0十折；GPU0/GPU1=60/60；墙钟1,339.17秒，worker累计5,196.23秒。冻结关系为D2–D4和D3–D4，来自历史Prompt与文献的保守人工综合，不伪称GPT-4逐字生成。概念偏离乘积是co-deviation，不是实际跨网络神经连接测量。Ridge alpha10、signed z/clip3；新增Top8语义发送者仅outer-train、lambda=.25；单独行归一化消息、来源明确的Linear+LN值投影、null路由、训练RMS尺度校准与逐token5%上限。E0独立重训10/10复现B0-A。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|E0_B0A|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E10_DIRECT_MATCHED|92.0742 ± 2.8427|91.8763 ± 2.7870|94.4172 ± 4.3608|89.3354 ± 3.3306|
|E11_SERCA_V2_FULL|91.2696 ± 2.9692|91.0235 ± 3.0337|94.4311 ± 3.0759|87.6159 ± 4.8988|
|E1_SERCA34_CAL|91.5008 ± 2.7843|91.2969 ± 2.7630|93.9963 ± 3.9613|88.5976 ± 3.6938|
|E2_NUMERIC_GLOBAL|91.6144 ± 3.0498|91.3793 ± 3.0553|94.4172 ± 4.0275|88.3415 ± 4.3028|
|E3_FIRST_ORDER|91.8443 ± 3.0233|91.6691 ± 2.9546|93.7650 ± 5.1908|89.5732 ± 3.8439|
|E4_ALL_PAIR|91.6144 ± 2.9170|91.5091 ± 2.8368|92.7012 ± 4.9769|90.3171 ± 3.2605|
|E5_RANDOM_RULE|91.7307 ± 2.8586|91.5492 ± 2.8742|94.0009 ± 2.8784|89.0976 ± 3.4981|
|E6_PATIENT_SHUFFLED|91.6144 ± 3.0062|91.4484 ± 2.9532|93.5615 ± 4.4468|89.3354 ± 3.3396|
|E7_ORIGINAL_EDGES|91.7294 ± 3.1245|91.5362 ± 3.0900|93.9870 ± 4.4272|89.0854 ± 3.7260|
|E8_CORR_ONLY|91.4982 ± 3.4617|91.2783 ± 3.4655|94.2091 ± 3.9922|88.3476 ± 4.4039|
|E9_AUX_ONLY|91.6131 ± 3.0927|91.4026 ± 3.1096|94.2137 ± 3.6134|88.5915 ± 4.1720|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|E11_SERCA_V2_FULL − E0_B0A|-0.8046|-0.8312|-0.1989|-1.4634|0/5/5|0/5/5|
|E11_SERCA_V2_FULL − E10_DIRECT_MATCHED|-0.8046|-0.8528|+0.0139|-1.7195|0/3/7|0/3/7|
|E11_SERCA_V2_FULL − E1_SERCA34_CAL|-0.2312|-0.2735|+0.4348|-0.9817|2/5/3|2/4/4|
|E11_SERCA_V2_FULL − E2_NUMERIC_GLOBAL|-0.3448|-0.3559|+0.0139|-0.7256|1/6/3|2/5/3|
|E11_SERCA_V2_FULL − E3_FIRST_ORDER|-0.5747|-0.6456|+0.6660|-1.9573|0/5/5|0/4/6|
|E11_SERCA_V2_FULL − E4_ALL_PAIR|-0.3448|-0.4857|+1.7299|-2.7012|1/6/3|1/5/4|
|E11_SERCA_V2_FULL − E5_RANDOM_RULE|-0.4611|-0.5258|+0.4302|-1.4817|2/4/4|2/1/7|
|E11_SERCA_V2_FULL − E6_PATIENT_SHUFFLED|-0.3448|-0.4250|+0.8696|-1.7195|2/4/4|2/3/5|
|E11_SERCA_V2_FULL − E7_ORIGINAL_EDGES|-0.4598|-0.5127|+0.4440|-1.4695|1/4/5|1/3/6|
|E11_SERCA_V2_FULL − E8_CORR_ONLY|-0.2286|-0.2548|+0.2220|-0.7317|1/6/3|2/3/5|
|E11_SERCA_V2_FULL − E9_AUX_ONLY|-0.3435|-0.3791|+0.2174|-0.9756|2/4/4|3/1/6|


E11路由均值(null/PHENO/ANAT/FUNC/CORR)≈[.0959,.1616,.1339,.1849,.4238]；实际残差各折最大值的均值4.6862%，最坏5.0000012%（浮点容差）。旧报告约.99的量是cap乘数，不是残差比例。固定checkpoint逐来源关闭均未改变类别；同RMS直接消息替换平均翻转约0.57%。正确规则不胜数值、全交互、错误规则、患者错配和仅CORR，NO-GO。该版训练TD用OOF、训练ASD用full-TD fit的变换分支不对称，后来E40/E42统一为所有训练者cross-fit；不把后来的修复假写回E35。原交付为OSSFS阻塞、/tmp完整保留。


### Experiment 36 — PCPR Phenotype-Conditioned Patient Relational Routing

220/220正式折、22/22组，seed0；GPU0/GPU1各110折，调度墙钟1,114.79秒。通过E11身份链将871行与1112行真实CSV按SUB_ID一一匹配，独立核对人口学和QC；不按行号或SUB_IN_SMP筛选。新增真实年龄/IQ/利手来自匹配CSV，与原PHENO48主要为年龄bin/site/sex的事实区分。诊断字段、ADOS/ADI-R等不进入文本。ANAT/FUNC只描述质量/运动，CORR仅特征槽分块统计，不冒称ROI。BioLORD-2023固定revision=`167aab527b238a50ca65224e6319215d2ff4fc9f`，768维mean pooling＋L2、冻结/缓存，manifest记录用户许可确认。

B0保留原GFR覆盖行为并绕过PEN；B1新四图全部有效传播、共享GCN、softmax融合与行归一化；B2先融合为单传播图；B3只用CORR图但输入仍320维。Q/K=64，beta_max=.25/init=.10，只调既有非自环边。N0–N3数值、T0–T3真实文本、S0–S3池内错配、C0/C1常量、H0/H1共享临床、M0/M1去临床；四图与单图不冒称原B0同构。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|B0_ORIGINAL_B0A|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|B1_FOUR_GRAPH_BASE|91.2696 ± 3.2251|91.0819 ± 3.2037|93.5661 ± 4.4333|88.5976 ± 4.1562|
|B2_ONE_GRAPH_BASE|91.7294 ± 2.7656|91.5564 ± 2.7835|93.7835 ± 3.6749|89.3293 ± 4.1741|
|B3_CORR_ONLY_BASE|91.2696 ± 3.1839|91.0867 ± 3.1174|93.3441 ± 5.1626|88.8293 ± 3.9209|
|C0_FOUR_SYM_CONSTANT|91.4995 ± 3.3057|91.2814 ± 3.3047|94.2091 ± 3.8772|88.3537 ± 4.1166|
|C1_ONE_SYM_CONSTANT|91.8443 ± 2.9793|91.6898 ± 3.0017|93.7882 ± 3.6583|89.5915 ± 4.2536|
|H0_FOUR_SYM_SHARED_CLINICAL|91.4995 ± 2.9239|91.3444 ± 2.8707|93.3534 ± 4.3743|89.3354 ± 3.3306|
|H1_ONE_SYM_CLINICAL_ONLY|91.3845 ± 3.2677|91.1587 ± 3.2904|94.2137 ± 3.6134|88.1037 ± 4.3801|
|M0_FOUR_SYM_MODAL_ONLY|91.6144 ± 3.1353|91.3528 ± 3.1235|94.8520 ± 3.6394|87.8537 ± 3.7148|
|M1_ONE_SYM_MODAL_ONLY|91.4995 ± 3.2248|91.3367 ± 3.1306|93.3441 ± 5.5239|89.3293 ± 3.6978|
|N0_FOUR_SYM_NUMERIC|91.8430 ± 3.2344|91.7242 ± 3.1579|93.1314 ± 5.1984|90.3171 ± 3.6154|
|N1_FOUR_DIR_NUMERIC|91.6144 ± 2.8249|91.3763 ± 2.7914|94.4172 ± 4.6618|88.3354 ± 4.1849|
|N2_ONE_SYM_NUMERIC|91.0397 ± 2.9517|90.8584 ± 2.8945|93.1314 ± 4.9302|88.5854 ± 3.8934|
|N3_ONE_DIR_NUMERIC|91.2696 ± 3.0134|91.0416 ± 3.0050|93.9917 ± 4.1984|88.0915 ± 4.1374|
|S0_FOUR_SYM_SHUFFLED|91.7294 ± 2.9950|91.6579 ± 3.0375|92.7243 ± 3.2175|90.5915 ± 4.3635|
|S1_FOUR_DIR_SHUFFLED|91.7294 ± 2.7656|91.5036 ± 2.7805|94.4218 ± 4.1230|88.5854 ± 4.4898|
|S2_ONE_SYM_SHUFFLED|91.3845 ± 3.0154|91.1508 ± 3.0410|94.4357 ± 3.4770|87.8659 ± 4.5555|
|S3_ONE_DIR_SHUFFLED|91.8443 ± 3.1095|91.6214 ± 3.1100|94.6392 ± 3.8838|88.6037 ± 4.2816|
|T0_FOUR_SYM_TEXT|92.0742 ± 2.4963|91.9423 ± 2.4248|93.5615 ± 4.2383|90.3232 ± 2.8246|
|T1_FOUR_DIR_TEXT|91.3845 ± 3.3079|91.2627 ± 3.2920|92.9278 ± 4.5321|89.5976 ± 4.4887|
|T2_ONE_SYM_TEXT|91.1560 ± 3.0064|90.9864 ± 2.9491|93.1314 ± 4.9302|88.8415 ± 3.8785|
|T3_ONE_DIR_TEXT|91.1546 ± 2.9178|90.8816 ± 2.9273|94.4218 ± 3.5259|87.3415 ± 3.7838|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|T0_FOUR_SYM_TEXT − B0_ORIGINAL_B0A|+0.0000|+0.0877|-1.0685|+1.2439|1/8/1|3/5/2|
|T0_FOUR_SYM_TEXT − B1_FOUR_GRAPH_BASE|+0.8046|+0.8605|-0.0046|+1.7256|7/2/1|8/1/1|
|T0_FOUR_SYM_TEXT − N0_FOUR_SYM_NUMERIC|+0.2312|+0.2181|+0.4302|+0.0061|3/5/2|4/4/2|
|T0_FOUR_SYM_TEXT − S0_FOUR_SYM_SHUFFLED|+0.3448|+0.2844|+0.8372|-0.2683|4/5/1|5/2/3|
|T0_FOUR_SYM_TEXT − C0_FOUR_SYM_CONSTANT|+0.5747|+0.6610|-0.6475|+1.9695|5/4/1|6/3/1|
|T0_FOUR_SYM_TEXT − H0_FOUR_SYM_SHARED_CLINICAL|+0.5747|+0.5980|+0.2081|+0.9878|6/3/1|8/1/1|
|T0_FOUR_SYM_TEXT − M0_FOUR_SYM_MODAL_ONLY|+0.4598|+0.5895|-1.2905|+2.4695|5/4/1|7/2/1|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|T1_FOUR_DIR_TEXT − B0_ORIGINAL_B0A|-0.6897|-0.5919|-1.7021|+0.5183|2/2/6|3/1/6|
|T1_FOUR_DIR_TEXT − B1_FOUR_GRAPH_BASE|+0.1149|+0.1809|-0.6383|+1.0000|2/7/1|5/3/2|
|T1_FOUR_DIR_TEXT − N1_FOUR_DIR_NUMERIC|-0.2299|-0.1136|-1.4894|+1.2622|3/3/4|4/2/4|
|T1_FOUR_DIR_TEXT − S1_FOUR_DIR_SHUFFLED|-0.3448|-0.2409|-1.4940|+1.0122|2/4/4|3/3/4|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|T2_ONE_SYM_TEXT − B0_ORIGINAL_B0A|-0.9182|-0.8682|-1.4986|-0.2378|1/2/7|1/2/7|
|T2_ONE_SYM_TEXT − B2_ONE_GRAPH_BASE|-0.5734|-0.5700|-0.6522|-0.4878|0/6/4|1/5/4|
|T2_ONE_SYM_TEXT − B3_CORR_ONLY_BASE|-0.1136|-0.1003|-0.2128|+0.0122|2/4/4|3/2/5|
|T2_ONE_SYM_TEXT − N2_ONE_SYM_NUMERIC|+0.1162|+0.1280|+0.0000|+0.2561|3/5/2|3/5/2|
|T2_ONE_SYM_TEXT − S2_ONE_SYM_SHUFFLED|-0.2286|-0.1644|-1.3043|+0.9756|1/6/3|3/4/3|
|T2_ONE_SYM_TEXT − C1_ONE_SYM_CONSTANT|-0.6883|-0.7034|-0.6568|-0.7500|1/2/7|1/2/7|
|T2_ONE_SYM_TEXT − H1_ONE_SYM_CLINICAL_ONLY|-0.2286|-0.1723|-1.0823|+0.7378|2/4/4|3/3/4|
|T2_ONE_SYM_TEXT − M1_ONE_SYM_MODAL_ONLY|-0.3435|-0.3503|-0.2128|-0.4878|1/4/5|2/2/6|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|T3_ONE_DIR_TEXT − B0_ORIGINAL_B0A|-0.9195|-0.9730|-0.2081|-1.7378|1/4/5|3/2/5|
|T3_ONE_DIR_TEXT − B2_ONE_GRAPH_BASE|-0.5747|-0.6748|+0.6383|-1.9878|1/5/4|2/3/5|
|T3_ONE_DIR_TEXT − N3_ONE_DIR_NUMERIC|-0.1149|-0.1599|+0.4302|-0.7500|2/6/2|3/4/3|
|T3_ONE_DIR_TEXT − S3_ONE_DIR_SHUFFLED|-0.6897|-0.7398|-0.2174|-1.2622|2/3/5|3/2/5|


T0 ACC与B0-A相同92.0742%，Hard-AUC略高至91.9423%；T0最佳checkpoint关闭文本后10/10折均无类别变化，四图融合仍接近.25。新单图邻接过密、表现明显较弱；未证明方向性或真实个体文本具有独立收益。数值对照的稀疏零填充与稠密BioLORD表示不完全匹配，不能简单据此宣称语言优势。常量文本理论应产生恒等调制，旧C0与B1差异不是“常量语义有效”证据，E37进一步用两步同RNG/Adam严格检验。结论NO-GO；原根交付和handoff受EIO阻塞，完整stage保留。


### Experiment 37 — LSPCR Label-Supervised Phenotype-Conditioned Patient Routing

140/140折、14/14组，seed0；GPU0/GPU1各70折，墙钟2,045.31秒。独立实现E36对称四图PCPR；FULL与HYBRID分别保留/去掉ANAT/FUNC/CORR重复临床背景，HYBRID严格对应E36 `modal_full[:,0] + modal_only[:,1:4]`。沿用同一BioLORD缓存。新增关系监督只用train-train非自环Q/K实际pre-tanh中心化对称分数，same/different软目标.9/.1、类别平衡BCE；SUPPORT/ALL、ranking、.03/.10/.30权重、文本错配、辅助标签站点内错配、数值、原GraphLearn监督、site-balanced控制均完整训练。推理从不读标签，不增加缺失边；原ALoss仍不进入目标。G00重放E36 T0的指标、混淆、best epoch与初始化/RNG均10/10 MATCH。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|G00_T0_REPLAY|92.0742 ± 2.4963|91.9423 ± 2.4248|93.5615 ± 4.2383|90.3232 ± 2.8246|
|G01_HYBRID_BASE|91.4995 ± 3.0994|91.3474 ± 3.0600|93.3534 ± 4.5766|89.3415 ± 3.9788|
|G02_FULL_SUPPORT_BCE|91.6144 ± 2.9170|91.4033 ± 2.8817|94.2091 ± 4.1041|88.5976 ± 3.6775|
|G03_HYBRID_SUPPORT_BCE|91.6144 ± 2.9170|91.3980 ± 2.9011|94.2044 ± 4.1193|88.5915 ± 4.0194|
|G04_FULL_ALL_BCE|91.7294 ± 2.9950|91.5120 ± 2.9877|94.4265 ± 3.7570|88.5976 ± 3.8515|
|G05_HYBRID_ALL_BCE|91.2696 ± 3.0569|91.0446 ± 3.0914|93.9917 ± 4.3141|88.0976 ± 5.0618|
|G06_HYBRID_RANK|91.7294 ± 2.9505|91.5199 ± 2.9177|94.2044 ± 4.3427|88.8354 ± 3.7320|
|G07_HYBRID_ALL_BCE_LOW|91.6144 ± 2.9170|91.4826 ± 2.8498|93.1360 ± 4.7309|89.8293 ± 3.4050|
|G08_HYBRID_ALL_BCE_HIGH|91.3845 ± 3.2270|91.1191 ± 3.2356|94.6346 ± 4.2390|87.6037 ± 4.6946|
|G09_HYBRID_TEXT_SHUFFLE|91.4995 ± 3.1835|91.3048 ± 3.1424|93.7743 ± 4.2842|88.8354 ± 3.3805|
|G10_HYBRID_AUX_LABEL_SHUFFLE|91.6144 ± 3.0498|91.4219 ± 3.0711|93.9963 ± 3.7257|88.8476 ± 4.3125|
|G11_HYBRID_NUMERIC_BCE|91.6144 ± 2.9619|91.4987 ± 2.9836|93.1499 ± 3.8147|89.8476 ± 4.4613|
|G12_RAW_GRAPH_LABEL_SUP|91.4995 ± 2.5884|91.3094 ± 2.5605|93.7835 ± 3.6749|88.8354 ± 3.3716|
|G13_HYBRID_SITE_BALANCED|91.4995 ± 3.0994|91.2459 ± 3.0572|94.4126 ± 4.4777|88.0793 ± 3.5256|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|G05_HYBRID_ALL_BCE − G00_T0_REPLAY|-0.8046|-0.8977|+0.4302|-2.2256|0/5/5|1/2/7|
|G05_HYBRID_ALL_BCE − G01_HYBRID_BASE|-0.2299|-0.3028|+0.6383|-1.2439|2/4/4|2/4/4|
|G05_HYBRID_ALL_BCE − G02_FULL_SUPPORT_BCE|-0.3448|-0.3587|-0.2174|-0.5000|1/6/3|2/4/4|
|G05_HYBRID_ALL_BCE − G03_HYBRID_SUPPORT_BCE|-0.3448|-0.3533|-0.2128|-0.4939|0/7/3|2/4/4|
|G05_HYBRID_ALL_BCE − G04_FULL_ALL_BCE|-0.4598|-0.4674|-0.4348|-0.5000|1/4/5|1/4/5|
|G05_HYBRID_ALL_BCE − G06_HYBRID_RANK|-0.4598|-0.4753|-0.2128|-0.7378|1/4/5|1/4/5|
|G05_HYBRID_ALL_BCE − G07_HYBRID_ALL_BCE_LOW|-0.3448|-0.4380|+0.8557|-1.7317|3/2/5|3/2/5|
|G05_HYBRID_ALL_BCE − G08_HYBRID_ALL_BCE_HIGH|-0.1149|-0.0745|-0.6429|+0.4939|1/7/2|2/6/2|
|G05_HYBRID_ALL_BCE − G09_HYBRID_TEXT_SHUFFLE|-0.2299|-0.2602|+0.2174|-0.7378|2/4/4|4/2/4|
|G05_HYBRID_ALL_BCE − G10_HYBRID_AUX_LABEL_SHUFFLE|-0.3448|-0.3773|-0.0046|-0.7500|1/5/4|3/1/6|
|G05_HYBRID_ALL_BCE − G11_HYBRID_NUMERIC_BCE|-0.3448|-0.4541|+0.8418|-1.7500|1/5/4|2/2/6|
|G05_HYBRID_ALL_BCE − G12_RAW_GRAPH_LABEL_SUP|-0.2299|-0.2648|+0.2081|-0.7378|1/6/3|2/4/4|
|G05_HYBRID_ALL_BCE − G13_HYBRID_SITE_BALANCED|-0.2299|-0.2013|-0.4209|+0.0183|2/4/4|2/3/5|


G05相对E36 B0-A的ΔACC/ΔHard-AUC约−0.8046/−0.8100pp，相对T0约−0.8046/−0.8977pp；不胜文本错配、辅助标签错配或同源数值。Q/K辅助梯度在10/10折非零、参数实际更新，但真实文本替换为错配文本的held-out类别翻转为0、logit绝对变化均值约5.58e−7；关闭条件同样无类别翻转，融合权重≈[.249956,.250069,.249980,.249995]。标签监督原患者关系早在原论文已有，不能将首次使用标签作创新。空支持正负类与G09身份断言曾修复并从检查点恢复，最终140唯一键无重复；结论NO-GO，训练关系可分性没有传递成可用分类增益。


### Experiment 38 — LCHER 原版与B0-A严格锚定重跑（必须分开）

两版各完成100唯一折、GPU各50折，均为seed0。原版墙钟2,785.95秒；严格重跑含恢复墙钟7,341.55秒。原版P0完整原PEN、B1物理删PEN与E1–E8的结果只作历史归档，不是本轮B0-A统一比较。原版E4比低B1高约.2299pp ACC，但仍低于真正B0-A，且PEN构造改变了RNG轨迹；这种结果不能证明创新成功。原版完整表：


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|B1_NO_PEN|91.3845 ± 2.7399|91.2270 ± 2.8180|93.3626 ± 3.6535|89.0915 ± 5.2263|
|E1_ANONYMOUS_NUMERIC|91.7294 ± 2.8596|91.5120 ± 2.8637|94.4265 ± 3.3763|88.5976 ± 3.6857|
|E2_REAL_NO_CF|91.3832 ± 3.1438|91.2326 ± 3.1604|93.1360 ± 4.5354|89.3293 ± 4.8469|
|E3_REAL_KEEP_ONLY|91.6144 ± 2.7778|91.4982 ± 2.6898|92.9232 ± 4.5446|90.0732 ± 2.7299|
|E4_LCHER_FULL|91.6144 ± 2.8713|91.3919 ± 2.8382|94.2044 ± 4.3427|88.5793 ± 3.7390|
|E5_WRONG_EDGE_PAIR|91.6144 ± 2.8249|91.4663 ± 2.8043|93.3534 ± 3.8116|89.5793 ± 3.3078|
|E6_NEUTRAL_TEXT|91.6144 ± 2.9170|91.4801 ± 2.9098|93.3626 ± 3.5274|89.5976 ± 3.5901|
|E7_NO_AUX_CONDITION|91.3845 ± 2.9264|91.2270 ± 2.9051|93.3626 ± 3.5274|89.0915 ± 3.3263|
|E8_NO_RIVAL_CONTRAST|91.4995 ± 2.6390|91.3071 ± 2.6118|93.7789 ± 4.0573|88.8354 ± 3.5606|
|P0_FULL_ORIGINAL|91.7294 ± 2.9505|91.5625 ± 2.9444|93.7835 ± 3.6749|89.3415 ± 3.6677|


严格重跑`B0A_ANCHOR`按E32完整构造＋PEN前向绕过，单进程连续10fold重训后逐折指标/混淆/best/执行epoch/主干初始化SHA/入口RNG全匹配；E1–E9恢复对应不可变fold入口并隔离新层初始化，非历史best微调。13组26条文献综合Prompt、39条可信FC，H14–H16仅归档；不是本轮GPT-4 API调用。训练TD条件参照、Top8、5%有界CAM残差、逐人rawFC keep/delete后完整重跑GFR、CF冻结主干与正式分类更新分开。E1匿名组与E4同CF预算；E5错配/E6中性/E9类别交换完整重训。严格重跑表：


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
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


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|E4_LCHER_FULL − B0A_ANCHOR|-0.6897|-0.6748|-0.8557|-0.4939|0/5/5|0/4/6|
|E4_LCHER_FULL − E1_ANON_FULL|+0.0000|-0.0342|+0.4255|-0.4939|2/6/2|4/3/3|
|E4_LCHER_FULL − E2_REAL_NO_CF|-0.0013|+0.0025|-0.2266|+0.2317|3/4/3|4/3/3|
|E4_LCHER_FULL − E3_REAL_KEEP_ONLY|-0.0013|+0.0181|-0.4394|+0.4756|2/6/2|4/3/3|
|E4_LCHER_FULL − E5_WRONG_EDGE_FULL|-0.3461|-0.3533|-0.2128|-0.4939|2/3/5|2/3/5|
|E4_LCHER_FULL − E6_NEUTRAL_FULL|-0.2312|-0.3137|+0.8603|-1.4878|2/4/4|3/0/7|
|E4_LCHER_FULL − E7_NO_AUX_FULL|-0.3448|-0.3268|-0.6475|-0.0061|1/5/4|1/5/4|
|E4_LCHER_FULL − E8_NO_RIVAL_FULL|-0.5747|-0.5852|-0.6522|-0.5183|2/2/6|2/2/6|
|E4_LCHER_FULL − E9_SWAP_CLASS_FULL|-0.3448|-0.4194|+0.6429|-1.4817|1/5/4|1/5/4|


严格E4 ΔACC/Hard-AUC相对B0-A=−0.6897/−0.6748pp；错误映射、中性文本和去竞争组均不弱，NO-GO。E4残差均值0.1746%、NULL概率.3126；文本交换绝对logp变化.001258、类别翻转.001149；selected-delete−random约+.0737、8/10折正，但Top-abs(z)本身已很强，不能据此认定医学文本有效。后续E40审计确认本版summary部分绕过TopK、错误映射物理量关联不干净、key缺边身份、强共同context冲淡query、TD/ASD训练参考分支不一致等问题；本版结论不升级，保留为阴性历史。一次E2-fold1因best文件不完整按相同锚点重放，指标和best一致，恢复不计新增唯一折。两版原根同步/handoff均EIO；当前stage是完整可读来源。


### Experiment 39 — HASE-MoE Heterogeneity-aware Semantic Experts

修正后100/100正式折，GPU0/GPU1=50/50，调度墙钟1,124.30秒、worker累计约61.5分钟。E0独立重训B0-A并10/10复演历史。四机制FC掩码为E19 D1–D4（72/56/45/32、并集165，允许重叠）；不等同完整规范脑网络或临床亚型。四FC专家＋CAM上下文/机制响应/年龄箱性别/QC路由，新增残差零起点、逐人10%上限，站点不直接进入新增路由。E1容量匹配、E2普通MoE、E3均匀、E4仅GFR路由、E5错误掩码、E6训练数值分组、E7患者错配、E8无专门化、E9 FULL。E9新增154,816参数，E1为154,452。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|E0_B0A|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E1_CAPACITY|90.4663 ± 2.5824|90.3561 ± 2.6314|91.6512 ± 3.4232|89.0610 ± 4.2502|
|E2_GENERIC_MOE|90.3514 ± 3.3862|90.3302 ± 3.2951|90.5689 ± 5.6488|90.0915 ± 4.1087|
|E3_MED_UNIFORM|87.3615 ± 3.9591|87.3181 ± 3.9965|88.0204 ± 5.6844|86.6159 ± 6.7266|
|E4_MED_GFR_ROUTE|87.0337 ± 4.4209|87.0440 ± 4.3004|86.7160 ± 7.4640|87.3720 ± 5.6142|
|E5_WRONG_MASK|84.8485 ± 5.8678|84.6515 ± 5.8980|87.1323 ± 6.2018|82.1707 ± 7.5147|
|E6_DATA_DRIVEN|85.9835 ± 5.2223|85.9567 ± 5.2250|86.5171 ± 6.4528|85.3963 ± 6.8588|
|E7_PATIENT_MISMATCH|89.7766 ± 4.3518|89.6959 ± 4.2792|90.5504 ± 6.6164|88.8415 ± 4.8778|
|E8_NO_SPEC|85.9862 ± 3.9889|85.8074 ± 3.9576|88.0111 ± 6.9960|83.6037 ± 7.0005|
|E9_HASE_FULL|85.7654 ± 4.1109|85.6017 ± 3.9648|88.0389 ± 6.7336|83.1646 ± 5.0081|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|E9_HASE_FULL − E0_B0A|-6.3088|-6.2529|-6.5911|-5.9146|0/0/10|0/0/10|
|E9_HASE_FULL − E1_CAPACITY|-4.7009|-4.7544|-3.6124|-5.8963|1/0/9|1/0/9|
|E9_HASE_FULL − E2_GENERIC_MOE|-4.5859|-4.7284|-2.5301|-6.9268|0/1/9|1/0/9|
|E9_HASE_FULL − E3_MED_UNIFORM|-1.5961|-1.7164|+0.0185|-3.4512|4/2/4|4/1/5|
|E9_HASE_FULL − E4_MED_GFR_ROUTE|-1.2683|-1.4422|+1.3228|-4.2073|4/1/5|4/0/6|
|E9_HASE_FULL − E5_WRONG_MASK|+0.9169|+0.9502|+0.9066|+0.9939|5/2/3|5/1/4|
|E9_HASE_FULL − E6_DATA_DRIVEN|-0.2181|-0.3550|+1.5217|-2.2317|5/0/5|5/0/5|
|E9_HASE_FULL − E7_PATIENT_MISMATCH|-4.0112|-4.0942|-2.5116|-5.6768|1/0/9|1/0/9|
|E9_HASE_FULL − E8_NO_SPEC|-0.2207|-0.2056|+0.0278|-0.4390|3/2/5|3/0/7|


E9相对B0-A ΔACC/Hard-AUC=−6.3088/−6.2529pp，ACC十折0/0/10；低于普通MoE/数值分组/患者错配。专家中心化余弦接近0、路由有差异及固定checkpoint关闭分支会改变预测，只能说明实际依赖，不证明机制正确。残差均值9.90%接近10%cap，路由站点η²均值约.234，存在队列/采集分布混杂的可能。首轮因错误在E2–E7启用L_spec及入口边界问题已全部排除；正式结果是修正后从头100折，不能把两轮混算。重要限制：各配置仅fold1共享主干/RNG，后续因各自连续随机流和早停长度分流，并非E38重跑式逐fold锚定。最终模型权重已按原验收清理；不能承诺仅凭当前结果可恢复旧最佳模型的精确logit。根同步仍为TEMP_ONLY_OSSFS_EIO。结论NO-GO。


### Experiment 40 — HCER-v2 Hypothesis-Constrained Evidence Routing

100/100折，GPU0/GPU1=50/50；正式墙钟1,696.69秒、总执行2,728.39秒。B0连续十折重训，与E32混淆/指标/best/执行epoch/入口SHA和RNG逐折EXACT；E1–E9共同fold锚点初始化。canonical edge_id统一raw/z/mu/sigma/selector/CF，E7只改query关联；加入真实ROI端点CLIP文本、显式跨边attention/温度、统一所有训练者OOF参照、方向核验QC、可弃权≤8、正门控和5%cap、单患者完整GFR反事实以及固定图/固定残差分解。没有干扰E39。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|B0 B0A_ANCHOR|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E1 Z_TOPK_FULL|91.7294 ± 2.9950|91.5758 ± 2.9531|93.5661 ± 4.1059|89.5854 ± 3.2768|
|E2 ANONYMOUS_FULL|88.0865 ± 8.3045|87.9569 ± 8.1011|89.5236 ± 11.2868|86.3902 ± 5.9814|
|E3 REAL_NO_CF|82.4190 ± 14.7069|81.8434 ± 15.3405|89.2599 ± 11.9230|74.4268 ± 25.6341|
|E4 REAL_KEEP_ONLY|85.2926 ± 12.4563|84.7548 ± 13.2601|91.6559 ± 8.9107|77.8537 ± 24.8420|
|E5 REAL_FULL|82.8788 ± 14.1519|82.1048 ± 14.9529|92.5023 ± 8.3901|71.7073 ± 27.0535|
|E6 ANATOMY_NEUTRAL_FULL|78.0878 ± 13.9343|77.2811 ± 14.3596|88.8853 ± 9.4876|65.6768 ± 22.0198|
|E7 WRONG_PAIR_FULL|81.7385 ± 14.7831|80.8543 ± 15.8096|92.9463 ± 4.7751|68.7622 ± 29.9727|
|E8 SWAPPED_CLASS_FULL|87.1317 ± 10.6609|86.7958 ± 10.9513|91.6281 ± 6.9235|81.9634 ± 16.8416|
|E9 NO_RIVAL_NORMMATCH_FULL|87.7129 ± 8.9954|87.5672 ± 8.5493|89.7502 ± 14.6085|85.3841 ± 5.1964|


|配对（左减右）|ΔACC pp|ΔHard-AUC pp|ΔSPE-TD pp|ΔSEN-ASD pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|E5 − B0|-9.1954|-9.7498|-2.1277|-17.3720|1/4/5|3/2/5|
|E5 − E1|-8.8506|-9.4709|-1.0638|-17.8780|2/3/5|3/0/7|
|E5 − E2|-5.2077|-5.8521|+2.9787|-14.6829|5/2/3|5/0/5|
|E5 − E3|+0.4598|+0.2614|+3.2424|-2.7195|3/5/2|4/2/4|
|E5 − E4|-2.4138|-2.6500|+0.8464|-6.1463|2/4/4|3/2/5|
|E5 − E6|+4.7910|+4.8238|+3.6170|+6.0305|5/4/1|7/1/2|
|E5 − E7|+1.1403|+1.2505|-0.4440|+2.9451|5/1/4|5/1/4|
|E5 − E8|-4.2529|-4.6910|+0.8742|-10.2561|6/2/2|7/0/3|
|E5 − E9|-4.8341|-5.4624|+2.7521|-13.6768|5/2/3|5/1/4|


分类明显失败，E5−B0 ΔACC/Hard-AUC约−9.1954/−9.7498pp；SEN降至71.7073%，不能以医疗语义改善叙述。无CF E3也在3/6/9折严重下降，因此不能单因CF解释。E5 score与abs(z)折均Spearman约−.698；selected-delete正确类logp下降.01897011，Top-abs(z)=.06718485，固定图Top-abs(z)=.06725948，固定残差selected≈.01878109。失败不只是图重连或残差自响应。E5/E9残差均值约.9406%/1.0392%，没有E38数量级剂量差，但单位范数校准使同患者强弱证据实际同幅，且训练后空mask≈0，弃权没有被有效使用。纯CF/分类梯度另行重算，在线累加梯度不冒充纯CF；部分fold CF query梯度很大但无CF也弱，只能列为优化风险。结论同时为NO-GO-CLASSIFICATION / NO-GO-SEMANTICS / NO-GO-EVIDENCE-SPECIFICITY。报告后处理bool_序列化错误已修复，没有重训或伪造指标；原/data交付与handoff阻塞，完整tmp保留。


### Experiment 41 — UCSMA-v2（当前阶段登记，非完成报告）

读取时间2026-10-26T08:55:20.301596+00:00；manifest prepared_utc=2026-10-26T08:44:04.573102+00:00、status=`PREFLIGHT_PASS`、completed_folds=0、workers与fold_status均空、e0_history_verification为空。当前没有`results/report.md`或`results/fold_metrics.csv`，不能输出任何已训练的四指标或GO/NO-GO科学结论。

已记录的预检：871×320、403/468；252可信连接覆盖110/111 HO ROI，91个degree≥3、19个degree1–2；4未知边排除语义解释但保留主干。首折初始化SHA=`e385468ea7f55107af54d22e6b4b2830607ad744cde58593d0dbf140b5253dc6`，与E32首折相符；参考测试标签不变性、零残差、两Adam/RNG下一步恢复、错误映射敏感、拒绝路径、两步梯度均PASS。预检首折相符不等于E0十折复现通过。

设计为训练TD条件参照、ROI覆盖/可靠度与保守解剖概念的可拒绝证据、共享残差adapter；新增182,016可训练参数、实际证据维2,700，预检最大cap比约.050000008。上述只是设计和预检观测，不是已成立的分类或医学收益。后续须由该实验自身完成E0精确十折闸门后才能登记正式结果；README记录`python run.py train`及`python run.py report`。本次仅只读登记，不启动、不恢复、不修改E41。


### Experiment 42 — 四环节独立可证伪验证（最终状态统一）

科研DIAGNOSTIC_COMPLETE_NO_GO；正式包已COPIED_AND_HASH_VERIFIED至/data的E42目录，最终540个普通文件逐文件SHA与/tmp相符，stage保留。上方首次追加曾写TMP_ONLY，最终验收已恢复交付，不把旧临时状态当当前结论。40/40 phase×fold任务、GPU0/GPU1各20；追加QA十折各卡5任务，不计分类fold。新B0训练0、不可变E40锚点重新核验并复用10/10、集成分类训练0。总墙钟4,042.26秒；主任务墙钟1,669.86秒，两个口径不同。


|组别|ACC %|Hard-AUC %|SPE-TD %|SEN-ASD %|
|---|---:|---:|---:|---:|
|B0A_VALIDATED_REUSE|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|


|链|结论|关键证据与局限|
|---|---|---|
|Q1 文献/Prompt|NO-GO|只有H01/H03/H05（3/13）在至少7fold训练内方向/站点规则受支持；39边均无直接edge-specific验证。四风格CLIP可区分，但没有风格在train-OOF同时胜全部匹配负对照；可学线性读出吸收类别槽交换也是探针识别性局限。|
|Q2 FC评分|NO-GO|冻结B0，K8主评分selected-delete真实类logp降.079938，随机.033208、Top-abs(z).097882；胜随机9/10、胜Top-abs(z)2/10，fixed graph基本相同。|
|Q3 CF约束|NO-GO|B0始终冻结；C0删除效应.112415，C3 .112310，Δ−.000105、胜2/10，rank改善4/10。每CF组600 attempted/460有效probe，不能误写600个唯一人。fold5无满足原正确且置信≥.6者，按规则不做CF。|
|Q4 条件/QC|局部GO：仅FUNC|R2训练内风险与绝对z的partial rho≈.14558、扰动不稳定性≈.21235，8/10折过代理门槛，平均选4.925、弃权.125；R1 PHENO/R3 ANAT均0/10。R4退化单R2，不构成多模态组合增益或正常B0分类增益。|

固定模型诊断和质量代理不等于独立临床疾病证据。首要瓶颈是当前39候选连接的医学先验没有稳定边级支持、Prompt评分不胜简单Top-abs(z)；CF也未改善定位。仅FUNC代理有效不足以组成整条创新链，因此没有启动HCER-v3或额外集成分类试验。

QA记录：R0/R2/R3本来不读站点，早期“site-permuted”误用随机排名；最终CSV按恒等关系作诊断性校正，原runtime行保留，决策不变。恢复测试发现内存Adam.load_state_dict复用step tensor，改为deepcopy并恢复RNG后精确复演；正式Checkpoint与指标未改。本次更新不重做这些实验、不删历史失败。E42根handoff先前主追加及QA追加均已前缀验证成功。


### 统一B0-A逐折复核与后续使用规则

本次只读重算E32-B0，并逐项比较E33/E34/E35/E36/E38严格重跑/E39/E40的基线和E42复用验证记录。四指标、混淆、best epoch、executed epoch、历史主干初始化SHA、入口RNG均10/10一致（E36用其明确记录的historical_initial_trunk_sha256，不能拿含新增结构的总参数hash比较）。E37是对E36-T0的重放，不冒称B0-A十折重放；E41目前仅首折初始化预检。E组跨折是否逐fold锚定必须按各实验自己的实现说明，E39后续fold不具备这一条件。


|fold|TP-ASD|FN-ASD|FP-TD|TN-TD|best epoch(0-based)|executed epochs|
|---|---:|---:|---:|---:|---:|---:|
|1|37|4|0|47|322|450|
|2|35|6|4|42|149|351|
|3|38|3|7|39|428|450|
|4|36|4|1|46|243|445|
|5|37|3|2|45|428|450|
|6|38|2|2|45|124|326|
|7|33|7|1|46|154|356|
|8|34|6|3|44|280|450|
|9|35|5|3|44|169|371|
|10|36|4|2|45|251|450|


B0-A固定统计为ACC92.0741901776%、Hard-AUC91.8546202703%、SPE94.6299722479%、SEN89.0792682927%；这些是十折非加权均值，全队列802/871正确、FN44、FP25是另一种样本加权口径，不能互相替换。完整原始SPromptGL的论文ABIDE ACC92.30%、本地E32-B2 ACC91.7294%、物理删除PEN旧E38-B1 ACC91.3845%分别属于不同来源/构造，不可混写。


### 本次来源SHA256清单与交付说明

仅记录实际读到的本地文件；“-”表示该类产物不存在或该实验不是分类训练。阶段仍可能被其独立任务更新（特别E41），以下SHA对应本次读取快照。既有实验文件没有由本次回填修改；本次不声称旧实验完整checkpoint仍可恢复，原清理状态须看各manifest。


|来源|最终报告SHA256|逐折CSV SHA256|
|---|---|---|
|E01|`fe9151b357dabdb2e118b0781d237f512a9cdb1031ebd45dcc8bf5020899269c`|`161713d671bd4a86dbd3fe52edd6b48a254035261ba4cf8c7a064ad95a1cf137`|
|E02|`ff0957432f80434ee6d1f79f3f71f4948401bf66f429f1f3a6d432477b8cd993`|`f79de97ade05f93e2af80f20db75db243bdc19249b750f37728fcff1ac2749e9`|
|E03|`c0f48f499f0b3f852b6d80af58a5d761f326aecdd201ace05bf0e151282cf14e`|`d627f8b9317027b05c2f9e73eb20061f5fa836dc6318cb3f403207f45c647d64`|
|E04|`eb14f344d29330c6e672881c4d0930d2f1fe6255977911e876c8e1ae74895da0`|`35e17a6bca3cba30197d358eb610f5c4976aa1e592b7ee19ed1aa934aabe8468`|
|E05|`092a12a8d6efc3d22070efc7f8f39971358bfd7586968552e187a090d4eeb839`|`735c614dfd621ac49a1cee1b53e13f7ff833ef81670bf2a83c939a9154f1c44a`|
|E06|`2b4315568ef32385c622cc05672276b8426693f8f35135fd17666cff92211795`|`62436ebe9651097cd64f677ae152a1fc926fe1fe94cb6815763e26a21d450ed3`|
|E07|`d51536f6e758240d3bf857444405152302e28f564736f4a9386c9b6d097d4029`|`56f225f4cc5a6b01f5832bd7925cf74f9617e0838f73e5c2728d65d2074c7a71`|
|E08|`932952b8a78752710a5614a25e2b5d9dba13922df18efb6de43d4ffef5fbecd2`|`36fdc054c096a88e9bfcd2ed7f02b2270ce96949814a949655f8c0f7668fe532`|
|E09|`ae0649a81484b7f642b6c5bf05680bb13fe45ef6661c0d4513f1136b4d15a905`|`669b59f2313dd559d5ce5fb5820ee6e9b5fc6da76fca76cdd2b2097a58b2e638`|
|E10|`64152ef7ed5cadc26e8f4f999faca222a65a2a096b0c8c48792fd315f96cc3ae`|`8a241ed11bbcd1e81fe4f6f50c648602e3ef805bf00e977cbbdda173b4505fe9`|
|E11|`a6358416262962415aceb99c02948e97ec6a15832229f41ccc9d3dd505534f88`|`-`|
|E12|`b847853f68485b0b57039e062b56d8ed724a8e875a8f04bf26b55b333e7ea869`|`-`|
|E13|`579dc6fbbbb9275a27e3146936bea608c827b040b1f549efc656053c5851ae68`|`8118658fa728a7a239a7648426f0e8b6b097589bd9c3d4b2f8edfc4544f143ad`|
|E14|`00da17b6ad80bae91a14770cdd735dd4908202e0e0932c36f0bfb364d0d9d195`|`bbd94b7089d7937f52d3590c9511f9aa60cac715690f09b18477ab05b5a89af7`|
|E15|`2b68c0fe32894f8d9eea243caf03ffac9f18a6872ccfd8d1cc1c5afff8f56886`|`f00f44effa8cbfb9dc9222ef5d8cce62aa61784a528590084e81384aef342d96`|
|E16|`df2b9478c042af87be2ac44806c156e873f3e09d35dce4c9768c77cc7b3bb1a0`|`3fc430915aad8d00a3ed0004e21a7fde1d6fa117c4f602f3ac0ffad21b040d56`|
|E17|`e6b44eb0ab1c37ff2e964f95ba73f3a231bc850b8b06bf547150fc867609c320`|`99ba41beb3843c254ca7c2382d599855c4eefabea42758907f2d002a550503b1`|
|E18|`3f556214fc610dafb184389e5fb6fcb81700678a772b46d2426dda18f42334c1`|`ee0b9babd7da2a517b7ee0d32caf5f470b8795e1234642a3ebf49da2fe225fa0`|
|E19|`9de20fe5c10e0f559c0cc33e8b4a81c312e9e9df7dc30ec8a3de72261b564ecf`|`e15d7c60dfd6bee29344708eda9ed4c5135f1d2851c534d1408cf49572305e2c`|
|E20|`3c2bd14ecaa4ebd0c8b4415d06459489ff0f8593c5315103f313143a56dd9cc3`|`dc4e2682a545aa6993c6bb11b3fa00d395c2f49e0a863adee64239bb25765cc0`|
|E21|`8076343fa518cf52c7d1c291770505eb8ad8e0169a438dfe8f84b493b95b9490`|`32ec5228dff4c5a3f496ae85844b8639ffe62029e83bf2e39c5f8e4c9c9d20c6`|
|E22|`cdb26831b50541c5dcae2f7cb03fcd69f538ab2a236de7c6202fbf457e14c6af`|`358c3ffe59ca767decca7cbafeff7b6a84aba634349997747b3f318f6c2b8eb9`|
|E23|`23aa65b98d8329bbae397327e47b1f2471da8251e6f2c5945b62fbfc90e58f02`|`a27ba1589578898e7629eb2eef1f398e2babb1f149448b57fbbada5b1a826ecf`|
|E24|`202cf507e55121e4e17e86701de8a70e67e9d4bed3965ceba3eb3d28c3dbc407`|`d922bf7d02cdf1948ec92329a4c835ee28a04b3f424664e620524d7a66fc2052`|
|E25|`a62e05213189f1ddc39f06690b586685a97c45d2347e39697e22a2f9f297db2f`|`08b4150e8c4a6ddead96afb59c48129c5c70aea1d89b92804785c2fa23b99e1a`|
|E26|`ca3f3a1ca2c643c392dcf779931dfb8e895e4fa6fd5ffc61c4d58b08ca10627a`|`e01c0c407cc0739359bb833c6bf0bfd40f81779b259250cc14c5708e90f18029`|
|E27|`7d0c9e8df67e9be10e5836140ee106ff6abe9d9d577b67feccf2ffe9f28ed696`|`-`|
|E28|`ff3abb967da6499ef4f9d624816f35b11481ac1bd3feb178c645318d3da9e6b7`|`1c1af3e5277c7b270f01c93d64d7bd1ed3d4b079d9e03849582be530091a8fb5`|
|E29|`81e46202f7a80f7a2f3b5c8d9eaa1c0e14cd3eae80a988ac9b8ecd8938e9ca28`|`36d403c0438d8515e0e06a04015479d71473bf6d12a5edd077387f6d09591820`|
|E30|`be3a4931b9f4d1bc1fd0dbcc4e928b4242ac73e7638cb1e459c8b0bfcaf8ec1e`|`5cbc6ffa3b63025e364bb1fa74b16a5fab2530be2b0aab70aa80b38ed309fa14`|
|E31|`6ba9d9d826c42a1e647e390bf6c0e061f6e4da2f9f6c49ad88db5f9a6e52d4cc`|`797cc22f21786830dc6fa177b4e8cbdab7687bab019815b34612631b8dc9f213`|
|E32|`d90d97a7b0f01f33dd84e42c64b0fec622eff74f6e12d97bce34188ea6d5df19`|`0133bb2fd5810beeb777c54b923c8d9b6e9ae2211ed225116ca7db2e3609ffec`|
|E33|`17e35073dd6495eaeb51de0ffcfb6825235300cc6a4dc83863700b6b1f30589e`|`d5dfd3ba8f3b2317ba490833ee6a3d53f8e6d6818c7957a1e65836db099e972c`|
|E34|`b3db01fd3d1c685e850a670e1e7af670fc9092c20a8f40e59490e18526681e1d`|`57b5c55060a64ccf4cf6e556c2b6d5234ab4f6d728e13b9350bc3a02e2e33897`|
|E35|`f8204d871999c11251234cdcf7a7b320303ee5c3b265046e7c549211c727f116`|`c9b417c036a3c22633ff0be120cc622a1adfbc587d7c79fba6762202de6a56e8`|
|E36|`e7cdc459eff1195b72a92f9919d71b1725984300e8f10a0f53750e6eccf9443f`|`e3e320714407861256be5fbb8b9b0b7b38e9884cfe521f2858f5032b15f676d3`|
|E37|`c7d179551f47d20b5f487e9cb549a654732a007754913b7b0be6c63dc45e5d5d`|`fb8416e2a6091bc8e4cdc96cc415567bbdc74845ff5bdd74659887d03bfdb717`|
|E38|`5b866f18f65a6e7f82aee04e9d6e50e0058611c9d56772b563f93e18d9291e23`|`20b4f31412b95d9c5010df2ead4b6336fd7660eaef4952cb06b4ad3957be8abf`|
|E39|`3054424ac69ce3e6a6c969931ffac3e1e384330c93bf410c82f8daf798e7deec`|`7be7bd0965acf4e100f096bc7fa4219e6401f7d8740fe4634b0248539566d51c`|
|E40|`7b14edb7b73ee7699fde5080372b174c2502c5da67dd92d384146625ecf42fc0`|`53a3e91d56d37b630073d7e3dedcb181863129580754bf2690699efc20df404e`|
|E41|`-`|`-`|
|E42|`4511ee48e30bab97400de0864fdd7013aeaa3ef4b704733ea12a6a6d05b7aff2`|`1ecb2119b1cc8793790d01843ba5c8a3c7017763acb9c6b2d93360eb1a8621b6`|
|E38原版|`d84356f30c21a2815d92aa59c8fc0f4920bbba27062c5a7b7dfdc3ef4797363a`|`26cf7b6657dda20233c3b5f2282bb52c15663aab3482c4683ccf7a88c5f40a04`|
|E41当前manifest|`2e30d739d132be37219ccb342ba8db7fd14a0cc7d2a93e1f8ab574c934276e6c`|无正式fold CSV|


本次handoff回填已覆盖E01–E42；这是记录完整性，不等于42项方法成功或所有计划训练都完成。已完成研究以阴性为主，E24按门槛止损、E27科学前提不足、E41尚未正式训练须单列。下一步若继续研究，先处理可验证先验和简单数值证据的增量，再预注册新方法；不得将当前NO-GO改写为语义或临床机制已经成立。



<!-- EXPERIMENT43_CORR_ANCHORED_GFR_FINAL -->
## Experiment 43 — CORR-anchored GFR Audit

# Experiment 43 — GFR结构审计、密度校准与CORR锚定可靠性精炼

状态：COMPLETE；完成100/100唯一正式折；GPU0=50、GPU1=50；B0-A=ANCHOR_REPLAY_PASS。
科研判定：**NO-GO**。正式调度墙钟2139.61秒；总执行2815.80秒。

所有组seed0；四指标为十折mean±population SD，单位%。未完成十折的组不生成均值。
|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E00_B0A_REPLAY|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E01_AGGREGATION_FIX|91.6157 ± 3.0054|91.4219 ± 3.0104|93.9963 ± 3.3354|88.8476 ± 3.6875|
|E02_SAFE_SUPLOSS|91.6144 ± 2.8249|91.4105 ± 2.7922|93.9917 ± 4.5191|88.8293 ± 3.9209|
|E03_DENSITY_CAL_ONLY|91.8443 ± 3.1095|91.6635 ± 3.0980|93.9917 ± 4.5366|89.3354 ± 4.5465|
|E04_MODALITY_SPECIFIC_PROP|91.3858 ± 2.7391|91.2194 ± 2.7095|93.3534 ± 4.0520|89.0854 ± 3.5458|
|E05_FIX_PLUS_CAL|91.6144 ± 3.0062|91.4268 ± 2.9731|93.7743 ± 4.6921|89.0793 ± 4.0646|
|E06_CORR_RELIABLE_REFINE|91.4995 ± 3.3057|91.3240 ± 3.3027|93.7882 ± 3.9081|88.8598 ± 4.2458|
|E07_CORR_UNIFORM_AUX|91.4995 ± 3.0565|91.2837 ± 3.0511|94.2137 ± 3.8661|88.3537 ± 4.1019|
|E08_CORR_SHUFFLED_AUX|91.4995 ± 3.0565|91.2783 ± 3.0424|94.2091 ± 3.6307|88.3476 ± 3.4922|
|E09_CORR_ANCHOR_ONLY|91.8443 ± 2.9346|91.6742 ± 2.9234|94.0009 ± 3.3169|89.3476 ± 3.2788|

## 预注册配对

|左减右|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|Hard-AUC W/T/L|
|---|---:|---:|---:|---:|---:|---:|
|E01_AGGREGATION_FIX − E00_B0A_REPLAY|-0.4585|-0.4327|-0.6337|-0.2317|1/4/5|2/3/5|
|E02_SAFE_SUPLOSS − E00_B0A_REPLAY|-0.4598|-0.4441|-0.6383|-0.2500|2/3/5|2/3/5|
|E03_DENSITY_CAL_ONLY − E00_B0A_REPLAY|-0.2299|-0.1911|-0.6383|+0.2561|2/4/4|3/3/4|
|E04_MODALITY_SPECIFIC_PROP − E01_AGGREGATION_FIX|-0.2299|-0.2026|-0.6429|+0.2378|2/4/4|3/3/4|
|E05_FIX_PLUS_CAL − E01_AGGREGATION_FIX|-0.0013|+0.0048|-0.2220|+0.2317|3/4/3|4/2/4|
|E05_FIX_PLUS_CAL − E03_DENSITY_CAL_ONLY|-0.2299|-0.2367|-0.2174|-0.2561|2/4/4|3/2/5|
|E06_CORR_RELIABLE_REFINE − E00_B0A_REPLAY|-0.5747|-0.5307|-0.8418|-0.2195|2/3/5|2/3/5|
|E06_CORR_RELIABLE_REFINE − E05_FIX_PLUS_CAL|-0.1149|-0.1028|+0.0139|-0.2195|2/6/2|3/4/3|
|E06_CORR_RELIABLE_REFINE − E07_CORR_UNIFORM_AUX|-0.0000|+0.0403|-0.4255|+0.5061|1/8/1|3/5/2|
|E06_CORR_RELIABLE_REFINE − E08_CORR_SHUFFLED_AUX|-0.0000|+0.0456|-0.4209|+0.5122|1/8/1|4/4/2|
|E06_CORR_RELIABLE_REFINE − E09_CORR_ANCHOR_ONLY|-0.3448|-0.3503|-0.2128|-0.4878|0/7/3|1/5/4|

完整逐fold四指标差见manifest.paired；混淆、best/执行epoch、预测hash、入口SHA/RNG、best checkpoint SHA见fold_metrics.csv。完成状态要求任务原子提交和best文件读回SHA，不能只数CSV行。

## 原始结构、损失和机制核验

原始GFR i==1覆盖i==0；每支传播全320维是论文设计而非代码错误。GraphLearn.p实际参与cosine；w/t/threshold和combined_weight未进入有效前向。原simple-2只有分类＋四图GraphConstructLoss平均，ALoss不优化。E01只修聚合；E02只加train-train无序对MSE，lambda=1；E03只改mutual96与训练pair median/IQR；E04相对E01只保留对应模态输入块；E05为聚合修复＋密度校准。
E06–E09全部以CORR mutual支持为上限；辅助只在exp(±.25*q)内改边权。可靠性来自原始特征空间、subject-disjoint内层逻辑回归的同诊断pair AUC增量；正式精炼使用当前GraphLearn投影空间。这个pair AUC是机制代理，不是第五项分类指标。所有新增标签计算只接收outer-train标签。E09 q=0的辅助GraphLearn实际可能没有梯度，须结合E07/E08，不能单据E06>E09宣称医学互补。

|组|模态|平均邻居|分类消息梯度均值|分支关闭logit变化|分支关闭翻转|
|---|---|---:|---:|---:|---:|
|E00_B0A_REPLAY|PHENO|191.0997|0|0|0|
|E00_B0A_REPLAY|ANAT|330.5819|0.01488296|0.001988814|0.002298851|
|E00_B0A_REPLAY|FUNC|155.4087|0.01511156|0.003370667|0.003448276|
|E00_B0A_REPLAY|CORR|93.21929|0.02422786|0.002980936|0.002298851|
|E01_AGGREGATION_FIX|PHENO|157.2716|0.04333887|0.003015258|0|
|E01_AGGREGATION_FIX|ANAT|321.809|0.03066894|0.001610261|0|
|E01_AGGREGATION_FIX|FUNC|147.293|0.01983336|0.0004721207|0|
|E01_AGGREGATION_FIX|CORR|109.2925|0.02385266|0.0002277157|0.001149425|
|E02_SAFE_SUPLOSS|PHENO|41.78324|0|0|0|
|E02_SAFE_SUPLOSS|ANAT|78.63927|0.02247717|0.0002441087|0|
|E02_SAFE_SUPLOSS|FUNC|50.08565|0.007239667|3.03141e-05|0|
|E02_SAFE_SUPLOSS|CORR|50.85557|0.02579056|0.001970441|0.001149425|
|E03_DENSITY_CAL_ONLY|PHENO|62.55017|0|0|0|
|E03_DENSITY_CAL_ONLY|ANAT|71.81355|0.008732718|0.0002013107|0|
|E03_DENSITY_CAL_ONLY|FUNC|70.49897|0.006883501|6.23235e-05|0|
|E03_DENSITY_CAL_ONLY|CORR|39.03284|0.003676799|7.014427e-05|0|
|E04_MODALITY_SPECIFIC_PROP|PHENO|206.471|0.0001735027|2.407452e-07|0|
|E04_MODALITY_SPECIFIC_PROP|ANAT|317.6356|0.000182473|7.341464e-07|0|
|E04_MODALITY_SPECIFIC_PROP|FUNC|154.7727|0.0003149034|2.455425e-06|0|
|E04_MODALITY_SPECIFIC_PROP|CORR|140.8753|0.01696857|0.001798124|0.001149425|
|E05_FIX_PLUS_CAL|PHENO|63.42388|0.04584816|0.001847199|0|
|E05_FIX_PLUS_CAL|ANAT|71.50448|0.05085582|0.003351345|0.002298851|
|E05_FIX_PLUS_CAL|FUNC|71.37704|0.02323161|0.0007489153|0|
|E05_FIX_PLUS_CAL|CORR|42.28703|0.0323972|0.001060701|0|
|E06_CORR_RELIABLE_REFINE|PHENO|48.47003|0.01598307|0.000577338|0|
|E06_CORR_RELIABLE_REFINE|ANAT|48.47003|0.009461891|0.0002891289|0|
|E06_CORR_RELIABLE_REFINE|FUNC|48.47003|0.005748666|4.645115e-05|0|
|E06_CORR_RELIABLE_REFINE|CORR|48.47003|0.009679734|6.189074e-05|0|
|E07_CORR_UNIFORM_AUX|PHENO|46.92813|0.008824734|0.0001437671|0|
|E07_CORR_UNIFORM_AUX|ANAT|46.92813|0.0232604|0.001923675|0|
|E07_CORR_UNIFORM_AUX|FUNC|46.92813|0.02834097|0.0009249105|0|
|E07_CORR_UNIFORM_AUX|CORR|46.92813|0.01728621|0.001362457|0.001149425|
|E08_CORR_SHUFFLED_AUX|PHENO|48.33318|0.004197494|3.995572e-05|0|
|E08_CORR_SHUFFLED_AUX|ANAT|48.33318|0.004596488|5.333065e-05|0|
|E08_CORR_SHUFFLED_AUX|FUNC|48.33318|0.006210049|7.996308e-05|0|
|E08_CORR_SHUFFLED_AUX|CORR|48.33318|0.008664581|0.0004581926|0|
|E09_CORR_ANCHOR_ONLY|PHENO|46.37681|0.009491048|0.000684936|0.001149425|
|E09_CORR_ANCHOR_ONLY|ANAT|46.37681|0.01000478|0.0002385477|0|
|E09_CORR_ANCHOR_ONLY|FUNC|46.37681|0.01565027|0.00143669|0.001149425|
|E09_CORR_ANCHOR_ONLY|CORR|46.37681|0.02452446|0.004067353|0.003448276|

## 可靠性门控（训练内估计）

|fold|PHENO Δpair-AUC|ANAT Δpair-AUC|FUNC Δpair-AUC|q(PHENO/ANAT/FUNC)|
|---|---:|---:|---:|---|
|1|+0.0002844|-0.0004886|-0.0000350|[1.0, 0.0, 0.0]|
|2|-0.0001253|-0.0001078|+0.0010551|[0.0, 0.0, 1.0]|
|3|-0.0003346|-0.0002351|+0.0003257|[0.0, 0.0, 1.0]|
|4|+0.0003362|-0.0001471|+0.0028957|[0.104034, 0.0, 0.895966]|
|5|+0.0000071|+0.0000363|+0.0001326|[0.040073, 0.206464, 0.753463]|
|6|-0.0001277|+0.0003153|+0.0030137|[0.0, 0.09472, 0.90528]|
|7|+0.0005206|-0.0001892|+0.0023181|[0.183377, 0.0, 0.816623]|
|8|+0.0001793|+0.0002810|+0.0031169|[0.050127, 0.078561, 0.871313]|
|9|-0.0007090|+0.0001872|+0.0056980|[0.0, 0.03181, 0.96819]|
|10|+0.0002925|-0.0000015|-0.0000014|[1.0, 0.0, 0.0]|

## 四类问题的独立结论

- 分支覆盖修正：E01_AGGREGATION_FIX−E00_B0A_REPLAY ΔACC=-0.4585pp、ΔHard-AUC=-0.4327pp。
- 论文监督意图：E02_SAFE_SUPLOSS−E00_B0A_REPLAY ΔACC=-0.4598pp、ΔHard-AUC=-0.4441pp。
- 密度及传播策略：E03_DENSITY_CAL_ONLY−E00_B0A_REPLAY ΔACC=-0.2299pp、ΔHard-AUC=-0.1911pp；E04_MODALITY_SPECIFIC_PROP−E01_AGGREGATION_FIX ΔACC=-0.2299pp、ΔHard-AUC=-0.2026pp；E05_FIX_PLUS_CAL−E01_AGGREGATION_FIX ΔACC=-0.0013pp、ΔHard-AUC=+0.0048pp；E05_FIX_PLUS_CAL−E03_DENSITY_CAL_ONLY ΔACC=-0.2299pp、ΔHard-AUC=-0.2367pp。
- CORR锚定辅助可靠性：E06_CORR_RELIABLE_REFINE−E00_B0A_REPLAY ΔACC=-0.5747pp、ΔHard-AUC=-0.5307pp；E06_CORR_RELIABLE_REFINE−E05_FIX_PLUS_CAL ΔACC=-0.1149pp、ΔHard-AUC=-0.1028pp；E06_CORR_RELIABLE_REFINE−E07_CORR_UNIFORM_AUX ΔACC=-0.0000pp、ΔHard-AUC=+0.0403pp；E06_CORR_RELIABLE_REFINE−E08_CORR_SHUFFLED_AUX ΔACC=-0.0000pp、ΔHard-AUC=+0.0456pp；E06_CORR_RELIABLE_REFINE−E09_CORR_ANCHOR_ONLY ΔACC=-0.3448pp、ΔHard-AUC=-0.3503pp。

预注册E06门槛：{"E00_B0A_REPLAY": false, "E05_FIX_PLUS_CAL": false, "E07_CORR_UNIFORM_AUX": false, "E08_CORR_SHUFFLED_AUX": false, "E09_CORR_ANCHOR_ONLY": false, "bounded_true_auxiliary_response": true}。实现修复、监督项恢复或稀疏图收益分别解释，不自动计为疾病语义创新。

## 初始化、恢复、文献与局限

本轮没有调用GPT-4、CLIP或编码新文本。保留原PLN完整参数注册与旧8条buffer，以E40已验证的E32训练前初始模型/两个Adam/四种RNG重建每fold；缓存替代原CLIP构造的随机消耗后显式恢复post-construction RNG。加载的是未训练fold入口，不是历史最佳模型微调。E00按相同入口分两卡各五折重训；只在10/10指标、混淆、best/执行epoch、初始化/RNG/split及最佳预测全匹配后启动90折。
原始simple-2/双Adam/450epochs/early>200/lr.0038/wd.11/dropout.35/hidden18/head2/th.9均沿用。阈值前score额外缓存不改变E00返回值；数值/梯度/两Adam两步一致与磁盘恢复后第三步一致见.runtime/preflight.json。每fold独立task/lock/原子checkpoint，10epoch覆盖保存，完成后仅保留可追溯best和入口。
论文来源：[SPromptGL MICCAI2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)；对照[作者源码](https://github.com/wanxixi11/SPromptGL)。论文本身已有标签监督图及全模态传播；本轮E02是安全的阈值前MSE实现，不声称与原ALoss数值等价。网络论文没有证明ANAT/FUNC QC是病理信号。
历史val=test选模、CV前全队列RFE/标准化、871节点transductive和重叠训练集使结果只适合内部配对筛选；不作独立泛化、临床因果、显著性或SOTA主张。

失败/恢复：{}

源码、数据、官方来源、Prompt文件指令、入口和结果hash见manifest。原保护文件需在结束时逐字节SHA仍相同。

产物目录：`/data/adodas/SPromptGL-main/experiment/experiment43_corr_anchored_gfr_audit`；复现：`python run.py all`，恢复：`python run.py resume`，报告：`python run.py report`。
源码SHA：{"run.py": "01270c207211e2f6dff903cbbd213082d01cba884b5734d1dafa89b5aab0d1ba", "method.py": "db28a3243318a6247d41585ec2f82c54f312ee9524d245bd418a7591ff924030", "integration.py": "13ebfce1fc575a459536e66f446343b0da16e4a1c9c7e6dd23c117a3c084fcac", "config.json": "5ad87e1314bee262d506881f9046760adfb049935e40a4de310d2591b5ccd8d0", "integration.patch": "9c4ce656efaa22cdd8907ad4961b479ac6cc8e9132031dad7a1d224099c083d8"}
输入SHA：b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1

<!-- EXPERIMENT43_AUXILIARY_FIXED_MODEL_AUDIT -->
## 完成后的独立核对与辅助权重干预

100/100唯一折、两卡50/50、所有best SHA、四项指标与保存预测/混淆均重新核对通过。保护源码与输入SHA保持不变；两物理设备初始校准四图逐位相同。未增加任何训练组或修改训练公式。
全部组保持同一参数注册：{"GFR": 887836, "PLN": 575121}；E09部分辅助GraphLearn无有效梯度是预注册控制的实际结果。
E02监督无序训练配对数：{"1": 306153, "2": 306936, "3": 306936, "4": 306936, "5": 306936, "6": 306936, "7": 306936, "8": 306936, "9": 306936, "10": 306936}；新增监督不接收外层test标签。E05诊断中估计q只供参照，未用于其前向；E07实际q固定1/3。
关闭整条GCN消息不能单独证明辅助关系有用。因此另外固定E06最佳模型，仅将q设0、设等权或置乱辅助C，保留所有GCN和CORR支持。恢复配置后输出、图和模型参数逐值恢复；结果如下。此处log-prob/翻转仅为机制诊断。
|fold|仅关闭辅助精炼：平均绝对log-prob变化|类别翻转数/测试数|
|---|---:|---:|
|1|8.8898979e-08|0/88|
|2|3.2890674e-07|0/87|
|3|2.4496995e-07|0/87|
|4|1.7567741e-07|0/87|
|5|1.2106872e-06|0/87|
|6|2.8524161e-07|0/87|
|7|2.1545613e-06|0/87|
|8|3.0463455e-06|0/87|
|9|5.3716974e-08|0/87|
|10|1.3702217e-09|0/87|

仅关闭辅助精炼：十折平均绝对log-prob变化=7.59037586e-07，合计类别翻转0/871。边权调制实现成立，但这种弱响应不能证明有独立分类价值；不能用非零q或梯度替代负对照。

|组|TP-ASD|FN-ASD|FP-TD|TN-TD|正确总数（混淆计数）|
|---|---:|---:|---:|---:|---:|
|E00_B0A_REPLAY|359|44|25|443|802|
|E01_AGGREGATION_FIX|358|45|28|440|798|
|E02_SAFE_SUPLOSS|358|45|28|440|798|
|E03_DENSITY_CAL_ONLY|360|43|28|440|800|
|E04_MODALITY_SPECIFIC_PROP|359|44|31|437|796|
|E05_FIX_PLUS_CAL|359|44|29|439|798|
|E06_CORR_RELIABLE_REFINE|358|45|29|439|797|
|E07_CORR_UNIFORM_AUX|356|47|27|441|797|
|E08_CORR_SHUFFLED_AUX|356|47|27|441|797|
|E09_CORR_ANCHOR_ONLY|360|43|28|440|800|

上述为871人的混淆合计；主表仍是十折均值，并未将折均值替换为微平均。
审计源码：audit.py；详细证据：.runtime/final_check.json及diagnostics.csv。再生成主报告后可运行`CUDA_VISIBLE_DEVICES=1 python audit.py`恢复本附录。

<!-- EXPERIMENT43_SCIENTIFIC_INTERPRETATION -->
## 四个问题的科学判断

1. 分支覆盖：原E00的PHENO消息分类梯度和关闭效应均严格为0，E01恢复四支有效梯度。但E01相对E00的ACC/Hard-AUC为−0.4585/−0.4327pp；修复实现问题并未带来分类增益。
2. 监督图项：E02的阈值前MSE确实作用于四个GraphLearn投影，新增监督只包含train-train无序非对角配对；相对E00为−0.4598/−0.4441pp。有效监督梯度没有转化成性能提升，它也不是原仓库未优化ALoss的数值复现。
3. 密度与传播：E03相对E00为−0.2299/−0.1911pp；E05相对E01为−0.0013/+0.0048pp，未见同时改善。E04相对E01为−0.2299/−0.2026pp，不支持仅传播对应模态。原全320维传播是原设计，本轮不能将其定性为错误。
4. CORR锚定可靠性：E06相对E00为−0.5747/−0.5307pp，相对E09纯CORR为−0.3448/−0.3503pp；相对等权E07、错配E08的ACC均无均值增益。全部主要比较均未达到>0.1pp和至少7折胜的门槛，结论NO-GO。
原E00四图平均邻居191.100/330.582/155.409/93.219；E06四图均为48.470，辅助精炼没有改变CORR支持。四个GCN消息可有非零分类梯度，但E06各分支贡献相对输入平均仅约2.74e−5/1.83e−5/2.76e−6/4.19e−6；仅关闭辅助精炼无类别翻转，不能据此宣称跨关系互补。
固定模型干预与从头训练消融回答不同问题，不能以前者替代E07–E09。强主干残差、归一化对边权改动的压缩和原权重衰减都可能参与消息衰减，但本轮未分别验证这些原因，不作因果归因。建议保留B0-A强基座，先定位消息传播的实际决策贡献，避免继续堆叠门控或将微小代理AUC增量包装成可靠医学关系。
本轮没有Prompt或疾病语义新实验。以上均为单seed历史val=test、CV前RFE/标准化及transductive协议下的内部比较；不支持独立泛化、显著性或临床因果主张。


## Experiment 41 — UCSMA-v2

<!-- experiment41_ucsma_v2_completion_v1 -->

验收时间：2026-10-26T11:55:29.973012+00:00。计算状态：**100/100有效正式折完成；NO-GO**。

E0独立重训，十折四指标、最佳epoch、执行epoch、初始参数SHA256及fold入口RNG全部精确匹配Experiment32 B0-A；另逐层核对原E32 B0 forward，GFR、四图邻接、CAM、attention与输出均10/10逐位一致。PEN保留构造但完全旁路前向。

有效正式任务：GPU0 E0/E2/E4/E6/E8=50折；GPU1 E1/E3/E5/E7/E9=50折；每卡最多4 worker，CPU/BLAS线程1。设备实际为两张CUDA兼容PPU-ZW810E，各98,304MiB，未修改其他任务或杀进程。

训练调度累计墙钟645.39秒（E0 455.71秒、初次增量100.11秒、更正调度89.56秒）；不含审计/报告。自E0开始至验收10498.00秒，包含人工中断与等待，不等于连续计算时间。有效worker耗时合计684.06秒；runtime约200.51MiB。

最终复核修复了条件参考mean FD索引（EFC→全局60列）、概念拒绝门控及ROI覆盖支持传递。初次E3–E9共70折作废并排除；更正预检后重算，E0/E1/E2保留，E2输入逐位等价。总执行170次fold拟合，其中仅100次进入正式结果；GPU0额外作废30折、GPU1额外作废40折。Prompt、先验、超参、GO规则均未按测试成绩更改。排除记录及旧哈希见results/correction_audit.json。

四指标：十折均值±population SD（%），AUC为硬标签one-hot ROC-AUC，二分类等于BA。

| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|
| E0_B0A | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E1_RAW_CAPACITY | 91.9592 ± 2.5255 | 91.7482 ± 2.5067 | 94.4172 ± 4.0374 | 89.0793 ± 3.7283 |
| E2_GLOBAL_NORM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E3_COND_NORM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E4_ROI_NORM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E5_MULTISCALE_NUM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E6_REAL_NO_REJECT | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E7_WRONG_SEMANTICS | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E8_DATA_PRIOR | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E9_UCSMA_FULL | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |

核心机制：条件参考TD OOF MSE由0.049654降至0.045132，但E9−E0/E5/E7/E8的ACC及Hard-AUC均为0.0000pp，逐折0胜/10平/0负；ASD灵敏度无增益。E9平均未匹配质量0.9545；正确/错误医学映射、关闭语义、全部残差关闭均未改变硬标签，患者错配仅改变2个外层测试预测。真实医学语义没有显示超过多尺度数值、错误或数值先验的独立增益。

新增组注册参数182016；有效输入连接参数E1=182016，E2/E3=116736，E4=158976，E5=162048，E6=163840，E7/E8/E9=163904；元素级梯度范围和剂量饱和见报告。E7不精确匹配重叠/覆盖，E8簇规模不同，不能宣称严格等容量或等覆盖。

结论：**NO-GO，不作为当前MICCAI 2027第一创新点主推**。不把内部单seed结果当作跨seed、独立中心或临床泛化；继承val=test选模、CV前RFE/标准化与871人transductive图限制。下一步仅建议重新评估问题与数据覆盖，不在本轮调参、联合微调、追加seed或启动实验42。

计算目录：`/tmp/SPromptGL-E41-UCSMA-stage/experiment/experiment41_uncertainty_calibrated_semantic_multiscale_alignment`；指定交付目录：`/data/adodas/SPromptGL-main/experiment/experiment41_uncertainty_calibrated_semantic_multiscale_alignment`（同步状态VERIFIED）；中文报告：`/data/adodas/SPromptGL-main/experiment/experiment41_uncertainty_calibrated_semantic_multiscale_alignment/results/report.md`。

报告SHA256：`1bc880f2957cca0c67e423d0102baca35aaaa18ca7fad683fb918716fec854e6`；run.py：`2b92a245edf5db77e5dbec59790f4a51a410620cecb971edc04811e89d298c3c`；ucsma.py：`8b34151fef678bc69a41e6fd64f2da6902f3a79ee7c6a7a9cf044f823b119af7`；Prompt：`0531b4753ef262242244ab117ff74f4329c0d1966fecd2cb14a8e0060ea8f493`；ontology：`c7cffee92b4d945f9928601ab639e78b9b84c6df7c8809daec86638dcaf85f6c`。

原始核心源码、数据、原论文、历史结果及CLIP缓存SHA256详见manifest；交付前复核受保护核心源码、两份数据及所有已读历史报告均未变化。processed_standard_data.csv SHA256=`b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`；modal_feat_dict.npy SHA256=`8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`。

I/O与恢复：计算阶段无OSSFS I/O故障；双Adam/RNG及增量Adam/RNG下一步重放预检均通过。最终同步：VERIFIED；无自动Git提交/推送。

<!-- /experiment41_ucsma_v2_completion_v1 -->


<!-- EXPERIMENT43_1_B0B_GFR_RETEST_FAIL -->
## Experiment 43.1 — B0-B GFR复验执行阻塞

AssertionError()
路径：/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest；未完成折不生成指标；修复后由resume继续。


<!-- EXPERIMENT43_1_B0B_GFR_RETEST_FAIL -->
## Experiment 43.1 — B0-B GFR复验执行阻塞

TypeError("dict() got multiple values for keyword argument 'lambda_sup'")
路径：/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest；未完成折不生成指标；修复后由resume继续。


<!-- EXPERIMENT43_1_B0B_GFR_RETEST_FAIL -->
## Experiment 43.1 — B0-B GFR复验执行阻塞

FileNotFoundError(2, 'No such file or directory')
路径：/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest；未完成折不生成指标；修复后由resume继续。


<!-- EXPERIMENT43_1_B0B_GFR_RETEST_FAIL -->
## Experiment 43.1 — B0-B GFR复验执行阻塞

AssertionError()
路径：/tmp/SPromptGL-E43_1-stage/experiment/experiment43_1_b0b_gfr_mechanism_retest；未完成折不生成指标；修复后由resume继续。


<!-- EXPERIMENT43_1_B0B_GFR_RETEST -->
## Experiment 43.1 — B0-B GFR五组优化验证


完成 60/60 唯一正式折。正式四指标为十折 mean ± population SD，单位%。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E00_B0B_REPLAY|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|E01_AGGREGATION_FIX_V2|91.6144 ± 2.9619|91.4694 ± 2.9274|93.3534 ± 4.1526|89.5854 ± 3.4710|
|E02_SAFE_LABEL_ADJ_SUP_V2|91.6144 ± 3.0498|91.4375 ± 3.0446|93.7835 ± 3.4255|89.0915 ± 3.5092|
|E03_SUPPORT_PRESERVING_CAL_V2|91.9592 ± 2.7747|91.7296 ± 2.7559|94.6300 ± 4.4709|88.8293 ± 4.2134|
|E04_MODALITY_LOCAL_PROP_V2|91.4995 ± 2.8321|91.2590 ± 2.8721|94.4265 ± 3.3763|88.0915 ± 4.4156|
|E05_FIX_PLUS_SUPPORT_CAL_V2|91.9592 ± 2.7747|91.7296 ± 2.7559|94.6300 ± 4.4709|88.8293 ± 4.2134|

## 与同折 B0-B 配对

|组别−B0-B|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|判定|
|---|---:|---:|---:|---:|---|---|---|
|E01_AGGREGATION_FIX_V2|-0.1149|-0.1352|+0.2174|-0.4878|1/7/2|2/4/4|NO-GO|
|E02_SAFE_LABEL_ADJ_SUP_V2|-0.1149|-0.1671|+0.6475|-0.9817|2/5/3|3/3/4|NO-GO|
|E03_SUPPORT_PRESERVING_CAL_V2|+0.2299|+0.1250|+1.4940|-1.2439|4/4/2|4/3/3|NO-GO|
|E04_MODALITY_LOCAL_PROP_V2|-0.2299|-0.3456|+1.2905|-1.9817|2/5/3|2/3/5|NO-GO|
|E05_FIX_PLUS_SUPPORT_CAL_V2|+0.2299|+0.1250|+1.4940|-1.2439|4/4/2|4/3/3|NO-GO|

## 基线身份及设计边界

B0-B物理不构造PEN/CLIP；PLN49,544参数，GFR887,836参数。原PLN fusion_weight的6个闲置参数是原有注册，不属于PEN。本轮未使用任何B0-A/E40锚点。B0-B历史为两条seed0连续随机流：GPU0折1–5，GPU1折6–10。保存新基线的首次更新前模型、双Adam与完整四种RNG；其余组逐折恢复这些未训练锚点。
历史未保留预测向量或完整Python/NumPy RNG，不能宣称历史预测SHA或完整四类历史RNG匹配；逐折实际核验所有历史已有的参数SHA、CPU/CUDA入口SHA、混淆、四指标、best/执行epoch。本轮另存完整RNG与预测向量及SHA。
E03与E05按用户定义完全同构；保留两组正式训练作为重复性核验，不能解释成两个独立改进或组合协同。
E02是仓库非平方Frobenius ALoss的训练诱导子图安全实现；论文式(1)为平方范数，故不声称数值复现论文损失。预测与同类目标均先限制train后加I归一化；原对角约1，加I约2；标签同类对角为1，加I为2。lambda只由首个同一次训练前向的四个p.weight/p.bias梯度计算，冻结不调参。原非优化ALoss继续不进入基础损失。
E03/E05校准仅作用传播图，原四GraphConstructLoss继续使用原始GraphLearn输出；每次forward逐元素验证支持、对角不变，没有mutual/topk/新边。E04输入和输出两端均限制对应模态块，参数形状仍320→320。

## 机制诊断

|组别|模态|邻居均值|分类消息梯度|贡献/输入范数|关闭分支log-prob变化|关闭翻转率|
|---|---|---:|---:|---:|---:|---:|
|E00_B0B_REPLAY|PHENO|186.6588|0|0|0|0|
|E00_B0B_REPLAY|ANAT|329.2041|0.01771266|4.43303e-05|0.00132518|0|
|E00_B0B_REPLAY|FUNC|148.6896|0.01516256|1.580155e-05|0.0003896123|0|
|E00_B0B_REPLAY|CORR|112.9917|0.01271634|1.705884e-05|9.154602e-05|0|
|E01_AGGREGATION_FIX_V2|PHENO|175.0762|0.004720359|2.787186e-05|0.0005778442|0|
|E01_AGGREGATION_FIX_V2|ANAT|318.2882|0.002432996|4.567415e-06|9.220062e-05|0|
|E01_AGGREGATION_FIX_V2|FUNC|160.5033|0.002469831|1.216164e-05|0.0001269725|0|
|E01_AGGREGATION_FIX_V2|CORR|132.3947|0.004268034|5.559833e-06|9.540566e-05|0|
|E02_SAFE_LABEL_ADJ_SUP_V2|PHENO|193.2652|0.006353941|4.899822e-05|0.0008988324|0.001149425|
|E02_SAFE_LABEL_ADJ_SUP_V2|ANAT|314.921|0.004722037|9.719336e-06|5.781378e-05|0|
|E02_SAFE_LABEL_ADJ_SUP_V2|FUNC|150.8181|0.002325226|2.65743e-06|1.64456e-05|0|
|E02_SAFE_LABEL_ADJ_SUP_V2|CORR|85.80712|0.002303517|2.115489e-06|1.792822e-05|0|
|E03_SUPPORT_PRESERVING_CAL_V2|PHENO|177.301|0.02134018|7.204938e-05|0.0004739036|0|
|E03_SUPPORT_PRESERVING_CAL_V2|ANAT|332.9706|0.00776689|2.18313e-05|0.0004830677|0.001149425|
|E03_SUPPORT_PRESERVING_CAL_V2|FUNC|149.1242|0.007835815|2.442027e-05|0.0002646228|0|
|E03_SUPPORT_PRESERVING_CAL_V2|CORR|164.4034|0.01224673|3.5402e-05|0.0009848337|0.001149425|
|E04_MODALITY_LOCAL_PROP_V2|PHENO|201.44|2.083407e-05|7.54589e-09|1.166153e-08|0|
|E04_MODALITY_LOCAL_PROP_V2|ANAT|333.7984|3.339858e-06|3.311837e-09|8.389543e-09|0|
|E04_MODALITY_LOCAL_PROP_V2|FUNC|157.6195|5.424794e-06|6.580259e-09|1.303759e-08|0|
|E04_MODALITY_LOCAL_PROP_V2|CORR|164.0225|0.06367716|0.0001099578|0.004562376|0.002298851|
|E05_FIX_PLUS_SUPPORT_CAL_V2|PHENO|177.301|0.02134018|7.204938e-05|0.0004739036|0|
|E05_FIX_PLUS_SUPPORT_CAL_V2|ANAT|332.9706|0.00776689|2.18313e-05|0.0004830677|0.001149425|
|E05_FIX_PLUS_SUPPORT_CAL_V2|FUNC|149.1242|0.007835815|2.442027e-05|0.0002646228|0|
|E05_FIX_PLUS_SUPPORT_CAL_V2|CORR|164.4034|0.01224673|3.5402e-05|0.0009848337|0.001149425|

## 基线逐折审计

|fold|参数SHA|入口CPU/CUDA RNG|混淆/四指标|best epoch|epochs|
|---|---|---|---|---|---|
|1|MATCH|MATCH|MATCH|157|359|
|2|MATCH|MATCH|MATCH|239|441|
|3|MATCH|MATCH|MATCH|180|382|
|4|MATCH|MATCH|MATCH|213|415|
|5|MATCH|MATCH|MATCH|219|421|
|6|MATCH|MATCH|MATCH|155|357|
|7|MATCH|MATCH|MATCH|187|389|
|8|MATCH|MATCH|MATCH|248|450|
|9|MATCH|MATCH|MATCH|206|408|
|10|MATCH|MATCH|MATCH|266|450|

## 统计边界与结论

历史B0-A（仅次级参照，非本轮锚点）：ACC92.0742、Hard-AUC91.8546、SPE94.6300、SEN89.0793%。若仅超过B0-B而未超过B0-A，只能声称相对B0-B的候选改善。
原val=test严格首次最佳、CV前全队列RFE/标准化、全871节点transductive与重叠训练集保留；不构成无偏独立泛化、统计显著性、医学语义或临床因果证据。本轮不调用GPT/CLIP，不下载模型。
所有支持/对角/局部传播及首步梯度等工程证据见.runtime/preflight.json，逐折图统计与分支关闭响应见diagnostics.csv，监督权重及损失见loss_trace.csv。

科研结论：NO-GO。E03/E05十折最佳模型、预测、epoch与四指标逐值相同。GPU0/GPU1正式折数：{'0': 30, '1': 30}；其中创新折分别25/25，基线各5。总墙钟2358.27秒。

产物：`/tmp/SPromptGL-E43_1-stage/experiment/experiment43_1_b0b_gfr_mechanism_retest`；复现：`python -B run.py all`，精确续跑：`python -B run.py resume`。


<!-- EXPERIMENT43_1_FINAL_VERIFIED_DELIVERY -->
### Experiment 43.1 最终验收与交付

<!-- E43_1_INDEPENDENT_FINAL_AUDIT -->
## 独立读回、单因素配对及交付审计

60/60最佳Checkpoint均已重新前向、重算四指标与混淆并核对预测SHA；全部held-out标签置换后的log-prob逐位不变。十个锚点的双Adam均为空的未训练初态，源/数据/split/RNG绑定通过。根源码、输入及受保护历史产物SHA保持不变。

|相对声明前置组|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|
|---|---:|---:|---:|---:|---|---|
|E02_SAFE_LABEL_ADJ_SUP_V2 - E01_AGGREGATION_FIX_V2|-0.0000|-0.0319|+0.4302|-0.4939|2/6/2|3/3/4|
|E03_SUPPORT_PRESERVING_CAL_V2 - E01_AGGREGATION_FIX_V2|+0.3448|+0.2602|+1.2766|-0.7561|3/7/0|3/4/3|
|E04_MODALITY_LOCAL_PROP_V2 - E01_AGGREGATION_FIX_V2|-0.1149|-0.2104|+1.0731|-1.4939|1/7/2|2/2/6|
|E05_FIX_PLUS_SUPPORT_CAL_V2 - E03_SUPPORT_PRESERVING_CAL_V2|+0.0000|+0.0000|+0.0000|+0.0000|0/10/0|0/10/0|

E03/E05定义重复；十折最佳模型、预测和epoch完全一致，不是两个独立改进。校准相对B0-B提高均值，但仅4/10折严格胜，未达7折门槛；相对历史B0-A的ACC仍低约0.1149pp、Hard-AUC低约0.1250pp。其SEN相对B0-B下降约1.2439pp，SPE增加约1.4940pp，须同时披露。

|固定最佳模型同时关闭四支消息|测试类别翻转/871|平均绝对log-prob变化|
|---|---:|---:|
|E00_B0B_REPLAY|0/871|0.0017995839|
|E01_AGGREGATION_FIX_V2|0/871|0.000812296847|
|E02_SAFE_LABEL_ADJ_SUP_V2|1/871|0.000962398892|
|E03_SUPPORT_PRESERVING_CAL_V2|1/871|0.00210632998|
|E04_MODALITY_LOCAL_PROP_V2|2/871|0.00456238045|
|E05_FIX_PLUS_SUPPORT_CAL_V2|1/871|0.00210632998|

以上只验证固定权重下的依赖，不能替代从头训练的无GFR组，也不能据此声称患者图无用。E01恢复四支分类梯度，但没有分类增益。E04辅助消息相对输入约1e−9量级，CORR约1e−4；本轮未支持限制传播到模态块。消息微弱的原因没有被单独隔离，不归因为某个未经对照的超参数。

E02十折lambda范围0.00378618–0.01027879，首步缩放监督/分类投影梯度比均为0.10。第1折PHENO监督梯度仅4.34e−9；非零不能视作实质监督。首步相对E01的参数差异见`.runtime/supervision_update_audit.json`，CORR最明显；Adam首步对梯度幅值的归一化意味着不能仅凭其他模态首步参数差很小就断言整个训练无监督作用。

正式训练区间（首锚点落盘至最后结果提交，含存储恢复）为1703.67秒；本次总执行墙钟见manifest.total_wall_seconds，起点为存储预检。GPU0/GPU1各30折，其中创新各25折、基线各5折。预检峰值显存0.4684GiB，不冒称它是全部正式worker的全程峰值。

存储初检PASS，后续OSSFS原子替换任务JSON时发生瞬时ENOENT，导致汇总器退出；原worker完成基线，无完成折重训。全部文件字节校验后迁移到/tmp继续，同一代码/训练签名不变。最终模型相等性比较复用单卡runtime，首次双卡可见控制器在报告阶段断言失败；以`CUDA_VISIBLE_DEVICES=1`重启控制器后全部已完成折直接跳过，报告成功；子进程仍按各自物理GPU分配。早期预检的计数/诊断键修正发生在任何正式训练之前。失败记录均保留，不计为额外训练折。

科研状态SCIENTIFIC_COMPLETE，结论NO-GO。正式目标`/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest`；本地保护副本`/tmp/SPromptGL-E43_1-stage/experiment/experiment43_1_b0b_gfr_mechanism_retest`。交付以manifest.delivery与delivery_sync为准；/tmp不具持久存储保障。未提交/推送Git。

本次有界同步成功：240文件，274008808字节，逐文件SHA完全一致；正式目录已交付。


<!-- EXPERIMENT43_2_GFR_INCREMENTAL_VALUE -->
## Experiment 43.2 — GFR Incremental Value Audit


正式完成：20/20折。仅seed0；四指标为十折mean ± population SD，单位%。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E00_FULL_GFR_B0B_REPLAY|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|E01_RAW_INPUT_NO_GFR|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|

## 同折配对 E00−E01

|fold|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|E00 best/epochs|E01 best/epochs|
|---|---:|---:|---:|---:|---|---|
|1|+0.0000|+0.1557|-2.1277|+2.4390|157/359|141/343|
|2|-1.1494|-1.2195|+0.0000|-2.4390|239/441|137/339|
|3|+1.1494|+1.4846|-4.3478|+7.3171|180/382|192/394|
|4|+0.0000|+0.0000|+0.0000|+0.0000|213/415|165/367|
|5|-1.1494|-0.8777|-4.2553|+2.5000|219/421|367/450|
|6|-1.1494|-1.2500|+0.0000|-2.5000|155/357|103/305|
|7|+1.1494|+1.6223|-4.2553|+7.5000|187/389|139/341|
|8|+0.0000|+0.0000|+0.0000|+0.0000|248/450|191/393|
|9|+0.0000|+0.0000|+0.0000|+0.0000|206/408|158/360|
|10|+1.1494|+1.2500|+0.0000|+2.5000|266/450|139/341|
|平均|+0.0000|+0.1165|-1.4986|+1.7317|—|—|

ACC W/T/L：3/4/3；Hard-AUC W/T/L：4/3/3。

预注册判定：**NO-GO / 增量价值未获充分支持**。门槛为两项均≥+0.1pp，且各自≥7折严格胜。未达门槛时不等价于证明所有患者图无效。

## 历史复现与初始化审计

E00重新完整训练10折；所有可用历史混淆、四指标、best_epoch、epochs_executed、初始化参数和CPU/CUDA入口RNG逐折匹配。E43.1另有完整四类RNG及预测SHA，本轮逐折匹配；独立读回进一步比较十折最佳模型所有张量，完全一致。历史B0-B没有保存全部预测和完整四类RNG，未伪称核验不存在的历史字段。
十个未训练锚点按E43.1原signature验证，未将旧signature误要求为新代码signature。共同PLN初始化、双Adam初态和完整RNG逐值相同；每折E00/E01的共同训练epoch后完整RNG SHA全部匹配。两组停止时间不同允许后续未共有epoch不作比较。E01全程不调用GFR/GCN/图损失，GFR参数与Adam状态SHA保持初始化值；从头训练的是同一PLN，而非最佳GFR模型微调。

## 每折混淆：TP-ASD/FN-ASD/FP-TD/TN-TD

|fold|E00|E01|
|---|---|---|
|1|38/3/1/46|37/4/0/47|
|2|35/6/4/42|36/5/4/42|
|3|38/3/8/38|35/6/6/40|
|4|36/4/1/46|36/4/1/46|
|5|38/2/5/42|37/3/3/44|
|6|37/3/1/46|38/2/1/46|
|7|37/3/3/44|34/6/1/46|
|8|34/6/4/43|34/6/4/43|
|9|34/6/3/44|34/6/3/44|
|10|36/4/2/45|35/5/2/45|

## GFR机制

|模态|分类消息梯度范数|图构建正则梯度范数|有效消息/输入范数|关闭分支logp变化|关闭类别翻转总数|
|---|---:|---:|---:|---:|---:|
|PHENO|0|0.033677608|0|0|0|
|ANAT|0.017712657|0.07748503|4.4330299e-05|0.0013251801|0|
|FUNC|0.015162565|0.052283533|1.5801545e-05|0.00038961227|0|
|CORR|0.012716344|0.19433281|1.7058839e-05|9.1546021e-05|0|

最佳模型GFR−raw相对范数：十折均值6.9216286e-05，范围[1.7786647e-07, 0.00038127572]。同时关闭四支消息类别翻转0/871；平均绝对logp变化0.0017995839。
PHENO中间消息虽可非零，最终直接贡献与分类消息梯度均为0，因为i==1覆盖i==0。图构建正则仍可更新PHENO图学习参数。固定模型关闭消息只解释依赖，不能替代本轮E01独立重训。

## 工程验收、时间与限制

GPU0=10/10折（fold1–5两组），GPU1=10/10折（fold6–10两组）。正式训练墙钟661.11秒；本次存储保护记录起至报告1599.93秒。每GPU一个worker、每worker2 CPU线程；不调用CLIP/GPT、不下载模型、未Git提交/推送。
20个最佳checkpoint独立重载：预测、四指标、混淆、模型/预测SHA及split/anchor全通过。预检含两组两步后磁盘恢复第三步逐值复演、E00原始步梯度/双Adam/RNG等价、E01输入恒等和GFR零调用、测试标签置换前向不变。受保护根源码、原数据、E43.1及历史基线文件SHA保持不变。
原val=test选首次最佳、CV前全数据标准化/RFE、全节点transductive和重叠训练集保持。Hard-AUC由硬标签计算，不能当作概率ROC-AUC；不额外报BA等第五指标。该配对比较不构成独立泛化、显著性或临床因果证据。差异归于整体GFR包（传播和必要的图正则），本轮未隔离它们的单独作用。

产物：`/data/adodas/SPromptGL-main/experiment/experiment43_2_gfr_incremental_value_audit`。训练恢复：`python -B run.py resume`；独立复核：`python -B run.py audit`；报告重建：`python -B run.py report`。

<!-- E43_2_RECOVERY_AND_FINAL_DELIVERY_AUDIT -->
## 存储恢复与最终交付补充审计

正式期间两次OSSFS任务JSON原子替换窗口返回瞬时ENOENT：首次影响E00阶段控制器，第二次影响E01阶段恢复控制器。两次均仅汇总控制器退出，独立训练worker继续完成当前阶段。使用同一冻结源码/配置有限恢复两次；最终稳定任务记录重核验后全部完成折跳过。没有完成折重训、没有改动训练公式/参数/RNG、没有将历史最佳模型用于初始化。故障堆栈和恢复记录保存在`.runtime/tasks/failure_*.json`、controller日志及manifest.storage_events。常规训练前storage probe为PASS，不能据此声称运行期间从无故障。

两组共有的3592个训练epoch，Python/NumPy/Torch CPU/CUDA完整RNG摘要逐个匹配。20个唯一正式任务各仅有一次FOLD_DONE；20个best checkpoint齐全，冗余latest checkpoint均已清理。每个best保留对应最佳epoch的模型、双Adam和完整RNG；任务记录保留结束epoch、等待/恢复证据和预测。

全体测试折混淆计数合计仅作审计（不替代十折均值）：E00 {'asd_tp': 363, 'asd_fn': 40, 'td_fp': 32, 'td_tn': 436}；E01 {'asd_tp': 356, 'asd_fn': 47, 'td_fp': 25, 'td_tn': 443}。GFR多正确识别7名ASD，同时少正确识别7名TD；两组总正确人数均为799/871，未新增第五项分类指标。

正式训练区间包含两次控制器故障与稳定结果重核对；训练最后一折落盘时间另见manifest.training_complete_at_last_record_unix。设备内存数值仅在预检/独立验收实测，不冒称它们是正式训练全程峰值。诊断中scalar转换产生一次PyTorch提醒，模型/优化器不变的断言通过；不是训练数值故障。

## Experiment 41.1 — SAGE-Align Final

<!-- experiment41_1_sage_align_final_v1 -->

验收时间：2026-10-26T14:33:41.110442+00:00。100/100有效正式折完成；GPU0执行 E0/E2/E4/E6/E8=50折，GPU1执行 E1/E3/E5/E7/E9=50折；每卡最多一个 worker，实际设备为两张 PPU-ZW810E。

E0 B0-A 精确复现：True（10/10）。适配学习率均为 0.00038；残差上限固定 8%；SAGE 分支 5,760 参数，CAM 仅作查询门控，零有效医学证据时残差严格为零。

| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|
| E0_B0A | 92.0742 ± 2.6000% | 91.8546 ± 2.5815% | 94.6300 ± 4.0457% | 89.0793 ± 3.7283% |
| E1_HEAD_ONLY | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E2_NUMERIC_ONLY | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E3_GLOBAL_REFERENCE | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E4_NO_QC | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E5_WRONG_SEMANTICS | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E6_DATA_PRIOR | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E7_FORCED_ALIGNMENT | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |
| E8_FROZEN_HEAD | 92.0742 ± 2.6000% | 91.8546 ± 2.5815% | 94.6300 ± 4.0457% | 89.0793 ± 3.7283% |
| E9_SAGE_FULL | 91.9592 ± 2.7267% | 91.7482 ± 2.7033% | 94.4172 ± 4.1480% | 89.0793 ± 3.7283% |

E9 未匹配质量与残差/8%上限见 `results/diagnostics.csv`；固定 E9 反事实见 `results/counterfactual.csv`。医学 Prompt 使用 Ilioska DOI 10.1038/s44220-026-00656-y 与 Kobbersmed DOI 10.1016/j.pneurobio.2026.102906 的保守区域级先验，未使用 CLIP/GPT-4。

最终科研结论：**FINAL NO-GO**。本轮若未通过 E0、E1、数值先验或错误映射必要性，不能作为 MICCAI 2027 第一创新点继续；按任务要求不启动 41.2。

结果目录：`/data/adodas/SPromptGL-main/experiment/experiment41_1_sage_align_final`。总训练墙钟：713.76 秒；报告 SHA256：`aa7143bf3711b8395ce7cfabeaed67d821596ba959d9254bdf0b960917bafae7`；run.py SHA256：`5cc9823312819e261196347c355b741b59222526873891f983cb14d4df73bc65`；sage_align.py SHA256：`a64ccf904c299aa92c42f18426d9afcc0a2ceef327e0a0245c2f7b235dccc99d`；prompts.json SHA256：`8d895ea4e422d776c6d1970307cb5801bcd9446e33d2d05d8c9a96d4ebd51a23`；ontology.json SHA256：`67205f37e0f98217518d1705e75f794a23bd2696c946422d4c9533e80958e230`。

沿用 val=test、CV 前 RFE/标准化和 871 人 transductive 图，结果属于历史协议下的内部比较。

<!-- /experiment41_1_sage_align_final_v1 -->


<!-- EXPERIMENT43_3_PAPER_GFR_ABLATION -->
## Experiment 43.3 — 原论文GFR消融复现专项核查


状态：SCIENTIFIC_COMPLETE；实际80/80条唯一折记录。新增50，核验复用30。

正式四指标：seed0十折mean ± population SD，单位%；ACC为折均值，并非全871人加权micro。后者单独列在summary.csv。

|组别|来源|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---|---:|---:|---:|---:|
|H-A-G|REUSED_VERIFIED|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|H-A-R|NEWLY_TRAINED|91.9592 ± 2.9591|91.7651 ± 2.9755|94.4265 ± 3.2394|89.1037 ± 3.9659|
|H-B-G|REUSED_VERIFIED|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|H-B-R|REUSED_VERIFIED|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|
|P-A-G|NEWLY_TRAINED|91.3845 ± 3.0589|91.1587 ± 3.0667|94.2137 ± 3.6134|88.1037 ± 4.0848|
|P-A-R|NEWLY_TRAINED|91.4995 ± 3.2655|91.2893 ± 3.2137|93.9870 ± 4.8354|88.5915 ± 3.8608|
|P-B-G|NEWLY_TRAINED|91.4995 ± 3.0130|91.2806 ± 3.0387|94.2137 ± 3.2156|88.3476 ± 3.9856|
|P-B-R|NEWLY_TRAINED|91.2709 ± 2.9685|91.0686 ± 2.9495|93.7835 ± 3.5496|88.3537 ± 3.2888|

## 原论文与双基座比较

|来源/协议|基座|无GFR ACC|有GFR ACC|ΔACC|无GFR AUC|有GFR AUC|ΔAUC|
|---|---|---:|---:|---:|---:|---:|---:|
|Figure 2图中标注|作者未公开完整消融实现|86.48|91.73|5.25|86.40|91.54|5.14|
|H|B0-A|91.9592|92.0742|+0.1149|91.7651|91.8546|+0.0896|
|H|B0-B|91.7294|91.7294|+0.0000|91.4880|91.6046|+0.1165|
|P|B0-A|91.4995|91.3845|-0.1149|91.2893|91.1587|-0.1306|
|P|B0-B|91.2709|91.4995|+0.2286|91.0686|91.2806|+0.2120|

Figure 2(c) +RES：ACC92.08、AUC91.93；(d) +PEN：ACC92.30、AUC92.22。以上全部来自PDF第7页原图数字标签，非本轮结果；本轮不训练(c)/(d)。正文Sec3.3明确baseline只用CAM。作者公开树中没有Figure2专用消融实现、消融初始化/训练顺序/种子。标记：PAPER_ABLATION_IMPLEMENTATION_UNAVAILABLE。

## 同协议同基座GFR配对

|比较|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|决策|
|---|---:|---:|---:|---:|---|---|---|
|H-A-G − H-A-R|+0.1149|+0.0896|+0.2035|-0.0244|4/4/2|4/4/2|NO-GO|
|H-B-G − H-B-R|+0.0000|+0.1165|-1.4986|+1.7317|3/4/3|4/3/3|NO-GO|
|P-A-G − P-A-R|-0.1149|-0.1306|+0.2266|-0.4878|1/7/2|1/6/3|NO-GO|
|P-B-G − P-B-R|+0.2286|+0.2120|+0.4302|-0.0061|4/3/3|4/3/3|NO-GO|
|H-A-G − H-B-G|+0.3448|+0.2500|+1.4940|-0.9939|4/5/1|5/3/2|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|
|H-A-R − H-B-R|+0.2299|+0.2770|-0.2081|+0.7622|5/3/2|6/2/2|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|
|P-A-G − P-B-G|-0.1149|-0.1220|-0.0000|-0.2439|2/6/2|2/5/3|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|
|P-A-R − P-B-R|+0.2286|+0.2207|+0.2035|+0.2378|3/5/2|3/3/4|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|

GFR准入要求：两项增益均≥0.1pp且两项各≥7/10严格胜。A/B差别含构造与初始化随机路径，不能解释为PEN语义收益；H/P同时改变多个参数，不能归因单一参数。训练折相互重叠，这些胜平负不是独立数据集显著性检验。

## 参数映射与确定源码证据

见paper_ablation_audit.json逐项paper_text_original/repo_argument/mapping_evidence/confidence/ambiguity。H为官方Shell运行参数；P=dropout .75、lr .0023、n_hidden10、heads4、nlayer4，明确为PAPER_TEXT_MAPPED_PROXY。原文“number of hidden layers10”与代码n_hidden表示Q/K/V每头宽度存在语义歧义；nlayer实际控制Encoder+FeedForward重复次数。P的alpha记为1以对应原文，但代码该参数仅存储，simple-2中原始特征加法硬编码为1；H默认alpha .5与P1均不改变该路径。reg .11、th .9、450/200及其它未明参数继承Shell/default，不是论文确认。

官方main/model/layers/shell与本地逐字相同。GraphLearn/GCN/GFR的AST逐值一致；均含i==1覆盖PHENO消息。simple-2的ALoss计算而不优化。官方network把RN50作为ViT_model注册在PLN，且在MLP之前加载；用户network先构造MLP再调用一次文本缓存，缓存8条1024维buffer，不注册整个RN50。这改变构造与apply(my_weight_init)所遍历的参数/RNG。官方utils.one_hot在CPU创建eye，本地改到索引device以兼容新PyTorch；数学指标口径相同。上述是确定实现/配置差别，尚不能单独解释原作者私有消融的5.25pp。

B0-A复用E43已逐折精确匹配E32的10个未训练状态、完整RNG与best；本轮又重新加载这些best并计算预测/四指标/混淆。B0-B复用E43.2已独立验收模型并再次重算。新增无GFR组只恢复未训练入口，不加载有GFR best作为初始化。历史A best只保留模型，best时Adam未保存；入口两个Adam空状态完整存在。该历史限制明确保留，不补造缺失状态。

## GFR机制

|有GFR组|输出−输入相对范数(均值)|有效消息/输入ANAT,FUNC,CORR|图密度PHENO,ANAT,FUNC,CORR|关闭全部消息logp绝对变化|类别翻转(871)|
|---|---:|---|---|---:|---:|
|H-A-G|0.00067748303|0.000336662, 0.000193902, 0.000270931|0.2197, 0.3800, 0.1786, 0.1071|0.008307849|7|
|H-B-G|6.9216286e-05|4.43303e-05, 1.58015e-05, 1.70588e-05|0.2146, 0.3784, 0.1709, 0.1299|0.001799584|0|
|P-A-G|5.0197642e-08|3.91979e-08, 1.39013e-08, 9.14792e-09|0.2237, 0.3316, 0.1606, 0.1525|4.275136e-07|0|
|P-B-G|1.5715236e-07|8.78368e-08, 1.02896e-07, 6.62973e-08|0.2276, 0.3320, 0.1719, 0.1460|9.391726e-07|0|

逐分支消息范数、分类GCN梯度、图投影梯度、有效融合、密度/度数及单支关闭logp/翻转见mechanism_diagnostics.csv。PHENO消息虽计算但被覆盖，分类GCN梯度为0；图正则仍优化PHENO GraphLearn。数值有消息、固定模型依赖消息、从头训练带来分类增益是三件事。

## 论文数值兼容性与核心回答

- H/B0-A：CAM-only距86.48为+5.4792pp，GFR距91.73为+0.3442pp，GFR增量距+5.25为-5.1351pp。
- H/B0-B：CAM-only距86.48为+5.2494pp，GFR距91.73为-0.0006pp，GFR增量距+5.25为-5.2500pp。
- P/B0-A：CAM-only距86.48为+5.0195pp，GFR距91.73为-0.3455pp，GFR增量距+5.25为-5.3649pp。
- P/B0-B：CAM-only距86.48为+4.7909pp，GFR距91.73为-0.2305pp，GFR增量距+5.25为-5.0214pp。

数值接近不等于严格重现作者私有消融；没有原始消融实现与完整初始化记录，不能断言论文错误或作者不当行为。B0-A/B0-B只能在各自协议内作配对增量；跨基座最好/最差拼接禁止。根据四个预注册增量与机制审计判断是否值得继续以GFR作创新基座：若证据不足，优先保留它作为历史兼容比较，并以CAM-only为必要强对照，不以论文5.25pp作为已证实前提。

## 训练、恢复与独立验收

- H-A-G：best_epoch=[322, 149, 428, 243, 428, 124, 154, 280, 169, 251]；epochs_executed=[450, 351, 450, 445, 450, 326, 356, 450, 371, 450]；来源REUSED_VERIFIED。
- H-A-R：best_epoch=[172, 124, 259, 154, 333, 139, 204, 389, 147, 203]；epochs_executed=[374, 326, 450, 356, 450, 341, 406, 450, 349, 405]；来源NEWLY_TRAINED。
- H-B-G：best_epoch=[157, 239, 180, 213, 219, 155, 187, 248, 206, 266]；epochs_executed=[359, 441, 382, 415, 421, 357, 389, 450, 408, 450]；来源REUSED_VERIFIED。
- H-B-R：best_epoch=[141, 137, 192, 165, 367, 103, 139, 191, 158, 139]；epochs_executed=[343, 339, 394, 367, 450, 305, 341, 393, 360, 341]；来源REUSED_VERIFIED。
- P-A-G：best_epoch=[186, 182, 376, 218, 247, 167, 157, 388, 229, 256]；epochs_executed=[388, 384, 450, 420, 449, 369, 359, 450, 431, 450]；来源NEWLY_TRAINED。
- P-A-R：best_epoch=[178, 188, 201, 155, 312, 142, 180, 292, 253, 364]；epochs_executed=[380, 390, 403, 357, 450, 344, 382, 450, 450, 450]；来源NEWLY_TRAINED。
- P-B-G：best_epoch=[166, 230, 224, 190, 311, 206, 173, 353, 129, 248]；epochs_executed=[368, 432, 426, 392, 450, 408, 375, 450, 331, 450]；来源NEWLY_TRAINED。
- P-B-R：best_epoch=[147, 206, 227, 357, 245, 286, 190, 313, 318, 162]；epochs_executed=[349, 408, 429, 450, 447, 450, 392, 450, 450, 364]；来源NEWLY_TRAINED。
- 新训GPU0=25折，GPU1=25折；正式墙钟1611.43s；本轮新训fold wall总和3072.64s（并行和I/O，不等于墙钟）。
- 新训每10epochs原子保存model、两Adam、best、当前epoch/wait、四类RNG、源/数据/配置/anchor/fold签名；完成后独立进程重载best重新预测及指标，并检查原输入/no-GFR零调用零梯度零更新。
- loss/训练ACC/testACC轨迹在新训task的loss_trace，10epoch抽样；历史保留的可读轨迹作为历史证据，不虚构未保存训练曲线。strict first best与wait>200均未更改。
- 原始代码/数据/历史文件protected SHA前后核对；无Git commit/push。

## 限制

val=test参与checkpoint选择、全队列标准化及CV前RFE、全871节点transductive协议保持，仅用于内部历史可比；不能视为无泄漏独立临床泛化。单seed与重叠训练折不支持显著性或因果结论。ANAT/FUNC是扫描质量/运动摘要；图是患者图，CAM潜token不是ROI。


<!-- E43_3_INDEPENDENT_SCIENCE -->

## 独立公式终验、训练曲线与弱消息来源

额外audit.py为与冻结训练入口分离的独立验收实现：用布尔混淆计数重算ACC/Hard-AUC/SPE/SEN，跨物理GPU重载全部80个best；检查初始化/预测/最佳模型SHA、no-GFR零调用/零更新、闲置PEN MLP不更新及heldout标签置换不改变forward。80/80通过；没有训练或修改原模型。

|组|最佳epoch均值|执行epoch均值|训练eval ACC|测试eval ACC|训练−测试pp|初始→末次分类loss|最佳图loss/分类loss绝对比|预测坍塌折|
|---|---:|---:|---:|---:|---:|---|---:|---:|
|H-A-G|254.8|409.9|96.0325|92.0742|3.9584|0.96447→0.37106|13.4615|0|
|H-A-R|212.4|390.7|96.6323|91.9592|4.6730|1.01835→0.24767|0.0000|0|
|H-B-G|207.0|407.2|96.7086|91.7294|4.9792|1.15263→0.29365|15.1335|0|
|H-B-R|173.2|363.3|96.4917|91.7294|4.7624|1.14408→0.19865|0.0000|0|
|P-A-G|240.6|415.0|96.2622|91.3845|4.8776|1.45016→0.13771|13.7185|0|
|P-A-R|226.5|405.6|95.8031|91.4995|4.3036|1.45872→0.31015|0.0000|0|
|P-B-G|223.0|408.2|96.1600|91.4995|4.6606|1.40253→0.21171|13.1266|0|
|P-B-R|245.1|418.9|96.6701|91.2709|5.3992|1.42571→0.30648|0.0000|0|

训练eval ACC是最佳checkpoint的诊断，非新增第五项正式指标。训练loss轨迹为train-mode、按原保存频率抽样，初始/末次不是同一dropout mask；不能把此表当作无偏过拟合估计。图loss包含负degree项，因此另列绝对比，不能把负总loss误判数值故障。

|有GFR组|ANAT GCN权重范数(best/init)|FUNC|CORR|GCN图正则梯度|
|---|---:|---:|---:|---:|
|H-A-G|0.008536212|0.005118586|0.005999795|0（逐张量独立核验）|
|H-B-G|0.003129299|0.001938282|0.003398192|0（逐张量独立核验）|
|P-A-G|0.0001685754|0.0001665667|0.0001091897|0（逐张量独立核验）|
|P-B-G|0.0003383632|0.0002986017|0.0002877412|0（逐张量独立核验）|

这些范数收缩和弱消息是实际观察；GFR的GraphConstructLoss只作用于构图投影，不直接监督GCN权重。原weight_decay=.11是候选机制解释，但本轮未新增去正则组，不能将收缩原因单独归因于weight decay。PHENO GCN因覆盖无分类梯度，其消息不进入最终输出。

补充确定源码/数据证据：官方固定commit的预处理CSV及modal_feat_dict.npy与本地逐字节相同。用户远端main七个核心文件也与本地相同。官方PEN硬编码CAM4×36，而论文代理10×4产生4×40，不能直接执行当前公开PEN；本轮两个基座都跳过PEN，因此代理四组维度合法。此证据限制论文配置精确映射，不影响冻结H/P。Eq4的CAM残差在InputLayer硬编码系数1，hyperpm.alpha未传入；GFR raw-X系数1是另一条独立skip。

H明确指公开Shell超参配合用户指定的历史A/B构造，不声称E32缓存式构造等于官方注册整个RN50的原始构造。强行换成后者会破坏本轮要求的E32-B0-A身份，因此仅做源码对照并保留未公开消融实现的不确定性。官方与用户都存在PHENO覆盖及ALoss未优化，不能将这两点说成用户独有改坏。

### A–F结论

A. 公开Shell/用户兼容初始化协议：B0-A ΔACC=+0.1149pp、ΔAUC=+0.0896pp，NO-GO；B0-B ΔACC=+0.0000pp、ΔAUC=+0.1165pp，NO-GO。
B. 论文文本代理协议：B0-A ΔACC=-0.1149pp、ΔAUC=-0.1306pp，NO-GO；B0-B ΔACC=+0.2286pp、ΔAUC=+0.2120pp，NO-GO。

C. 本轮A/B差异中最大ACC绝对差为0.3448pp；该量级不足以解释原论文5.25pp的同基座GFR增量。两者本来不执行PEN，不能归因于疾病文本。

D. CAM-only实测范围91.2709–91.9592%；与86.48最近差距4.7909pp。是否数值接近应看这个差距，不能用不同协议组拼接。

E. 确定差异：正文/Shell超参不一致；代码n_hidden语义与自然语言不一致；官方/用户RN50注册、构造顺序、缓存和初始化遍历不同；公式图监督与实际loss不同；论文未公开消融专用脚本/种子/完整初始化。上述证据并未证明其中某一项单独造成论文差距。评估偏倚与原Torch1.12/RTX3090环境也已披露，不能在未执行反事实配置时断言为根因。

F. 研究建议：继续把GFR作为历史对照可以，但任何以患者图为创新基座的工作都需同时对CAM-only证明增量，并检查实际消息与初始化。若配对四组未达门槛，本轮不给“GFR具有稳定额外分类价值”的GO，也不据此证明全部患者图无效。


H-B历史G/R共同epoch完整四类RNG又逐项核对：3592个全部相同。P-A/P-B共同epoch trace由run.py report逐项核对；H-A历史未保存逐epoch完整trace，不伪称具有该层证据。A/B两基座的入口状态各自匹配，跨基座不作逐参数相同主张。


<!-- E43_3_FINAL_AUDIT_TABLES -->

## 表四：论文原文、公开Shell及唯一代理映射

|字段|论文正文|H Shell/default|P代理|源码含义及证据状态|
|---|---|---|---|---|
|dropout|the dropout rate is 0.75|0.35|0.75|main.py parser; layers.Attention/VariLengthInputLayer/EncodeLayer/FeedForwardLayer; HIGH|
|lr|the learning rate is 0.0023|0.0038|0.0023|model.py Adam lr=hyperpm.lr; HIGH|
|n_hidden|the number of hidden layers is 10|18|10|network.PLN d_q=d_k=d_v=hyperpm.n_hidden; NOT hidden layer count; LOW; ASSUMED_MAPPING|
|n_head|number of attention heads is 4|2|4|network.PLN n_head and layer reshape; HIGH|
|nlayer|the encoding layer is 4|1|4|network.PLN for i in range(self.n_layer), Encoder+FeedForward pair; MEDIUM; ASSUMED_MAPPING|
|alpha|weight of original multi-modal features alpha=1|0.5|1.0|Eq4 CAM residual: layers.VariLengthInputLayer q += residual hardcodes1; EncodeLayer also uses residual1. hyperpm.alpha only stored in EvalHelper, never passed to CAM/GFR simple-2. GFR raw-X residual also independently hardcodes1.; LOW; ASSUMED_MAPPING|
|reg|not disclosed|0.11|0.11|Shell --reg .11; Adam weight_decay; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|th|not disclosed as numeric in implementation detail|0.9|0.9|main default .9; GraphLearn score>.9 detached mask; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|nepoch|not disclosed|450|450|main default450; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|early|not disclosed|200|200|main wait_cnt>hyperpm.early; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|loss|Eq7 lambda=eta=1; Eq1 graph smoothness+supervised adjacency|simple-2|simple-2|model.py loss=cls_loss+all_graph_loss; ALoss absent; CODE_DIFFERS_FROM_PAPER|
|optimizer|not specified in paper|two Adam|two Adam|model EvalHelper optimizer_MP0/optimizer_MF; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|seed|not disclosed|0|0|main default0; historical continuous RNG; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|data|ABIDE871 four modalities|same historical preprocessed|same|main loads processed_standard_data.csv & modal_feat_dict.npy; no new preprocessing; HIGH; ASSUMED_MAPPING|
|evaluation|10-fold ACC/AUC/SEN/SPE|val=test strict first ACC; hard onehot AUC|same|main.py split/early loop; model.test/roc_auc_score; PAPER_UNDISCLOSED_DETAIL; ASSUMED_MAPPING|

## 表五补充：有效消息、分类梯度与图正则路径

|有GFR组|输入范数|输出范数|相对差异%|ANAT/FUNC/CORR分类GCN梯度范数|ANAT/FUNC/CORR best/init权重范数|关闭消息翻转|
|---|---:|---:|---:|---|---|---:|
|H-A-G|236.3112|236.2978|0.0677483|0.014883, 0.0151116, 0.0242279|0.00853621, 0.00511859, 0.00599979|7/871|
|H-B-G|236.3112|236.3107|0.006921629|0.0177127, 0.0151626, 0.0127163|0.0031293, 0.00193828, 0.00339819|0/871|
|P-A-G|236.3112|236.3112|5.019764e-06|0.00030141, 0.000228785, 0.000125074|0.000168575, 0.000166567, 0.00010919|0/871|
|P-B-G|236.3112|236.3112|1.571524e-05|0.000337561, 0.000494878, 0.000405342|0.000338363, 0.000298602, 0.000287741|0/871|

原始图的密度、邻居数及各分支关闭效应完整列于mechanism_diagnostics.csv。GraphConstructLoss→GCN参数梯度均为0；该loss优化构图投影，不能据此认为全部GFR参数都受到图正则直接训练。P最佳模型消息接近数值零，但从头训练G/R轨迹仍可能不同；不能把最终弱消息倒推为整个训练从未用图。

终验状态：80/80折，50 NEWLY_TRAINED、30 REUSED_VERIFIED；GPU0/GPU1各25新增折。正式训练与第一轮验收墙钟1611.43s；含论文/代码审计及独立公式终验2608.47s（截至此前独立终验）。本附表只整理既有记录，没有再次训练或评估。


## Experiment 41.1 — SAGE-Align Final

<!-- experiment41_1_sage_align_final_corrected_v2 -->

本更正段取代此前v1的有效折及墙钟声明；历史内容不删除。v1的100行结果因E5对应不一致、E2/E6重复、QC方向和恢复/预检不足而作废，隔离于`.runtime/rejected_protocol_audit_v1`。当前修正协议100/100有效唯一折完成；纠错前结果已观察，不声称盲法预注册。

完成UTC：2026-10-26T17:26:32.753226+00:00。GPU0=E0/E2/E4/E6/E8共50折，GPU1=E1/E3/E5/E7/E9共50折，每卡一个worker；E0四指标、best/执行epoch、初始模型和入口RNG逐折10/10精确复现E32 B0-A。

| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|
| E0_B0A | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E1_HEAD_ONLY | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E2_NUMERIC_ONLY | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E3_GLOBAL_REFERENCE | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E4_NO_QC | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E5_WRONG_SEMANTICS | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E6_DATA_PRIOR | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E7_FORCED_ALIGNMENT | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E8_FROZEN_HEAD | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E9_SAGE_FULL | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |

E9核心配对差（pp，ACC/Hard-AUC）：E0_B0A -0.1149/-0.1064；E1_HEAD_ONLY +0.0000/+0.0000；E2_NUMERIC_ONLY +0.0000/+0.0000；E3_GLOBAL_REFERENCE +0.0000/+0.0000；E4_NO_QC +0.0000/+0.0000；E5_WRONG_SEMANTICS +0.0000/+0.0000；E6_DATA_PRIOR +0.0000/+0.0000；E7_FORCED_ALIGNMENT +0.0000/+0.0000；E8_FROZEN_HEAD -0.1149/-0.1064。

E9未匹配质量=0.344745，有效残差/CAM=0.00121223，best epoch0=10/10。

主要修复：无bias证据必经分支5,760参数；同一TD OOF经验CDF；单向FUNC质量及严格拒绝；E5共同ROI/FC/族置换并公开匹配误差；E2固定数值与E6训练软先验区分；轻量分类头对照；能量正则、零残差预检及Adam/RNG精确恢复。P4因类型保持而不变，不能称完全公平；连续语义响应不等于独立疾病收益。

最终科研结论：{"GO_CLASSIFICATION": false, "GO_MEDICAL_SEMANTICS": false, "GO_RELIABILITY": false, "final": "FINAL NO-GO", "interpretation": "Internal descriptive screening; no statistical significance or independent generalization proof."}。永久停止当前SAGE-Align/UCSMA医学语义对齐路线；无足够证据作为MICCAI2027第一创新点，不提出41.2。

总墙钟=12210.46s；修正阶段墙钟=2527.19s（并行任务时间未求和冒充墙钟）。结果目录：`/data/adodas/SPromptGL-main/experiment/experiment41_1_sage_align_final`。

哈希：run.py `615081f5d13e3db4633aaa92c9b2d10cecd22cf8db770e6ffc4e1f959cd86f4b`；sage_align.py `1a4cc2ce60f259b9e9b128052d69b9df27aace209d7fbf8d7876c75fa8f7658c`；integration.py `95e0ff498bc9e34b5bdac91e7241dc11a9b1ed3eab1a8701ecc613646e1849d7`；prompts.json `6d585359a237ac841fa8f350f456fb4830e1dcc558e0fb55fb154b63c699f30b`；ontology.json `910d7dba0a01c22763974290f0cb2347eda89542f6ffb70ca8306c6eb8ec1155`；results/report.md `1a1b141cae6900a62c88bd5ae8875eb65ae8a44d8bfed7c844bc85081fe6ca28`。原始六源码、数据与100个正式best/anchor哈希见manifest。

沿用val=test、预CV RFE/标准化及871人transductive图；单seed内部比较，不是无偏独立泛化或临床证明。

<!-- /experiment41_1_sage_align_final_corrected_v2 -->

<!-- experiment41_1_corrected_v2_trajectory_addendum -->

Experiment 41.1优化轨迹补充：所选best为首次更新，其raw/有效残差未饱和；这不能推广到完整训练过程。E9后续10/10折出现饱和，每折有120–169个epoch存在raw残差范数大于3倍8%上限的训练患者；峰值患者比例为58.55%，最后100步的十折平均比例为13.29%。tanh上限保持有效残差有界，但未阻止raw残差的长期饱和。训练前短程梯度/恢复预检没有保证长期稳定；该稳定性目标没有通过正式轨迹验证。训练NLL下降不能弥补最佳epoch全部停在0，也不能称充分学习。这些是已冻结实验的科学与优化失败，按原规则保留100个正式折，不为此重新调参或选模。完整90折轨迹汇总见trajectory_audit.csv；最终维持FINAL NO-GO，永久停止本路线。

报告已补充上述失败，最新报告SHA256为`be93ae7aa0f252cba0162c460f7390d7f0d691d8dd312bdb757f14db442ae80a`，替代同节较早的报告哈希；训练设计、Prompt和100个checkpoint未改变。记录墙钟从2026-10-26T14:03:02.293569+00:00至2026-10-26T17:29:17.188891+00:00为12374.90秒；更早审计未完整计时，不伪称覆盖全部前置工作。


<!-- EXPERIMENT43_4_PHYSICAL_CAM_ONLY -->
## Experiment 43.4 — Physical CAM-only validation

# Experiment 43.4 — 物理移除GFR/PEN的CAM-only专项验证

状态：20/20折完成；E0/E1各10折。

|模型|具体做法|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---|---:|---:|---:|---:|
|E0_ANCHORED_PHYSICAL_CAM|物理删除GFR/PEN，恢复E43.2 E01 CAM anchor|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|
|E1_NATURAL_PHYSICAL_CAM|物理删除GFR/PEN，自然构造CAM|91.6144 ± 3.0062|91.4166 ± 3.0203|93.9917 ± 4.0988|88.8415 ± 4.4635|

## 43.2历史参考

|模型|ACC|Hard-AUC|SPE|SEN|
|---|---:|---:|---:|
|43.2 E01历史功能性删除|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|
 E1 ACC与历史E01差值：-0.1149pp。
 E1 HARD_AUC与历史E01差值：-0.0715pp。
 E1 SPE与历史E01差值：-0.6429pp。
 E1 SEN与历史E01差值：+0.5000pp。

## E0逐折等价性
E0的10个fold均通过：初始CAM参数/单Adam、前两步历史E01对照、最终best_epoch、epochs_executed、预测SHA、混淆计数和四指标均一致。E0不构造GFR/PEN，证明历史功能性删除与物理删除在该锚定状态下等价。

## 结构核验
PhysicalCAMOnly参数量49,544；每个模型只有一个CAM Adam。named_parameters/state_dict不存在GFR、GraphLearn、GCN、MessagePassing、PEN、SPGS、RES、CLIP、AvgPool、Prompt相关键。forward只有原始320维输入、CAM层和OutputLayer；没有图邻接、图归一化、GraphConstructLoss或GFR优化器。

## E1自然初始化解读
E1四指标为ACC 91.6144、Hard-AUC 91.4166、SPE 93.9917、SEN 88.8415。E1采用每卡一次seed=0后按五折连续构造的独立随机流，不读取历史anchor，因此与E0逐折差异不能解释为GFR因果效应。

## 结论
E0满足物理删除等价性；E1是否适合作为后续研究基座取决于其独立初始化稳定性和完整结构审计。本轮保留历史val=test、全局预处理/RFE和transductive图协议的评估偏倚，结果用于内部复现而非独立临床泛化。

GPU0/GPU1各完成10折；正式墙钟1510.32s。产物路径：`/data/adodas/SPromptGL-main/experiment/experiment43_4_physical_cam_only_validation`。


<!-- EXPERIMENT45_PID_DGAA -->
## Experiment 45 — PID-DGAA

# Experiment 45 — PID-DGAA

状态：E0复用10折，E1–E6新训练0/60折；PID准入=PID_RELIABLE_BUT_NONDISCRIMINATIVE。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E0_B0B|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|

## E4逐折配对

|比较|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|
|---|---:|---:|---:|---:|---|---|

## PID主结果

|关系|平均R|平均U1|平均U2|平均S|平均冗余占比|平均q|平均lambda|
|---|---:|---:|---:|---:|---:|---:|---:|
|PA|0.001534|0.005869|0.001659|0.005688|0.095397|—|—|
|PF|0.004388|0.003015|0.003865|0.011631|0.193797|—|—|
|PC|0.001801|0.005602|0.000045|0.013557|0.094020|—|—|
|AF|0.001237|0.001956|0.007016|0.005106|0.072863|0.000000|0.000000|
|AC|0.000688|0.002505|0.001158|0.007405|0.057990|0.000000|0.000000|
|FC|0.001738|0.006514|0.000107|0.012964|0.080196|0.000000|0.000000|

## 机制与限制
PID使用训练折聚类、I_min离散分解、12次bootstrap和19次标签置乱；未使用外层测试标签。E0来自实验43.2 verified B0-B，新增组从同折未训练anchor构造。图节点为患者，ANAT/FUNC是质量/头动摘要，PHENO分支的原始 i==1 覆盖行为保持。历史 val=test、CV前全局RFE/标准化、transductive图、单seed和PID有限样本估计误差意味着这些结果只支持内部机制比较。

## 判定
E4只有在同时超过E0、E3、E5且图消息对主分类有可测依赖时才满足预注册GO；否则报告NO-GO，不以辅助正则或单次均值提升替代PID关系证据。

产物：`/data/adodas/SPromptGL-main/experiment/experiment45_pid_guided_graph_adversarial_alignment`。

## 正式训练闸门
PID准入未达到 `PID_RELIABLE_AND_HETEROGENEOUS`（实际为 `PID_RELIABLE_BUT_NONDISCRIMINATIVE`），因此按预注册协议未启动E1–E6的60折训练；这不是缺失结果，而是PID关系没有通过非零且异质的训练内可用性门槛。


<!-- EXPERIMENT44_CUGL -->
## Experiment 44 — CU-GL Counterfactual Utility-Guided Patient Graph Learning

- 状态：FORMAL_COMPLETE；阶段A：PASS；正式记录：100/100。
- GPU0：教师任务1–5、正式E2/E4/E6/E8共40折；GPU1：教师任务6–10、正式E3/E5/E7/E9共40折；E0/E1各10折使用已核验历史锚点并保留折级文件。
- E0 B0-B精确复现：PASS；E1 CAM-only锚点核验：PASS；反事实教师任务：30/30。
- 结果/源码哈希：{"counterfactual.csv": "7a03d2993b6f24c9d3a6c1cd04a220dc9ee6877e53c9adf95c18ed755a0da79e", "diagnostics.csv": "8d9a9fbc89b5da1007edcbbccae21d2298a302909c9411f0b5a04648775d1bfd", "fold_metrics.csv": "32b066e07d420489eef353225fdd5598771d978f9b2920a240a74a70bbdffbf9", "manifest.json": "31ebd4b6d608ab87b650db07a78c27676a3b87435032efdc86ffeff13fc45e01", "report.md": "57058aa5e048e520ea80bd69b20ff1fc383fca5d4c32fb51a3697a92432a4028"}。
- 总墙钟：4388.2s；根源码/数据保护哈希已记录，未修改。
- 最终科研结论：FINAL NO-GO；反事实效用响应可计算且消息进入决策，但E9低于B0-B/CAM-only，且相对置乱效用和同质性监督无独立收益；永久停止该路线，不启动Experiment45。

# Experiment 44 — CU-GL 反事实效用驱动的定向患者图学习

状态：100/100条折级记录；E0/E1为已核验历史锚点，E2–E9为本实验私有实现。

本报告将反事实邻居效用定义为冻结预测模型删除一条有向边前后的真实类 log-prob 差异；它是模型层面的边干预响应，不是患者之间的临床因果作用。

## 结果（% mean ± population SD）

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|折数|
|---|---:|---:|---:|---:|---:|
|E0_B0B|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|10|
|E1_CAM_ONLY|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|10|
|E2_SYMMETRIC_KNN|90.6962 ± 3.5101|90.5290 ± 3.4578|92.7105 ± 4.7515|88.3476 ± 3.8256|10|
|E3_DIRECTED_ONLY|90.5825 ± 3.4464|90.4068 ± 3.3755|92.9417 ± 5.1436|87.8720 ± 4.4022|10|
|E4_DIRECTED_BLOCK|90.5825 ± 3.2082|90.2928 ± 3.2599|94.2137 ± 3.0716|86.3720 ± 4.6899|10|
|E5_CF_SYMMETRIC|90.6962 ± 3.6935|90.4255 ± 3.8111|94.2229 ± 3.5840|86.6280 ± 6.6881|10|
|E6_CF_NO_BLOCK|91.0397 ± 3.3303|90.7814 ± 3.3634|94.2091 ± 3.7533|87.3537 ± 4.7469|10|
|E7_SHUFFLED_CF|91.1573 ± 2.9604|90.9087 ± 3.0072|94.2137 ± 3.0716|87.6037 ± 4.4203|10|
|E8_HOMOPHILY|91.0423 ± 3.5213|90.7446 ± 3.6706|94.8612 ± 2.7550|86.6280 ± 6.6122|10|
|E9_CUGL_FULL|90.8111 ± 3.4929|90.6112 ± 3.5220|93.3626 ± 3.5274|87.8598 ± 4.5821|10|

## 逐折比较

- E9 − E0: ACC -0.9182pp (2/10 strict wins), HARD_AUC -0.9934pp (2/10 strict wins), SPE +0.2266pp (2/10 strict wins), SEN -2.2134pp (2/10 strict wins)
- E9 − E1: ACC -0.9182pp (1/10 strict wins), HARD_AUC -0.8768pp (2/10 strict wins), SPE -1.2720pp (1/10 strict wins), SEN -0.4817pp (4/10 strict wins)
- E9 − E4: ACC +0.2286pp (5/10 strict wins), HARD_AUC +0.3184pp (7/10 strict wins), SPE -0.8511pp (1/10 strict wins), SEN +1.4878pp (6/10 strict wins)
- E9 − E7: ACC -0.3461pp (2/10 strict wins), HARD_AUC -0.2975pp (3/10 strict wins), SPE -0.8511pp (1/10 strict wins), SEN +0.2561pp (2/10 strict wins)
- E9 − E8: ACC -0.2312pp (2/10 strict wins), HARD_AUC -0.1335pp (5/10 strict wins), SPE -1.4986pp (2/10 strict wins), SEN +1.2317pp (4/10 strict wins)

## 阶段A教师与候选图
- 阶段A状态：`PASS`；有效OOF边 334590，|U|均值 5.7577459e-06，正/负比例 0.4933/0.5001。
- 真实效用与固定置乱效用相关系数：0.0036；与类别同质性相关系数：0.0037。
- 候选集合是每个外层折训练发送者中的四模态余弦Top-16；测试患者不向测试患者发送消息。训练和推理使用相同接收行/发送列方向约定。
- E2–E9的新增分支为共享的小型方向评分器、显式NULL质量和两层真实患者消息残差；NULL全选时残差严格为零。

## 审计发现
- 原始GFR的PHENO分支在 `i==1` 覆盖此前聚合；43.1已恢复但未改善分类。43.2中消息相对变化约6.92e-5且关闭消息无类别翻转。CU-GL因此隔离自特征和真实跨患者残差，并记录NULL质量、残差范数和固定模型干预。
- E0沿用Experiment43.1已完成的B0-B物理PEN删除复现；E1沿用Experiment43.4已验证的物理CAM-only锚定结果。两者不与新分支混用。
- 原论文的四关系图、CAM和OutputLayer接口保持；本实验不重新引入PEN、SPGS、RES、CLIP或GPT。创新组没有把原无向GraphConstructLoss套在带NULL有向概率上，新增监督仅为固定规则的分类损失与交叉拟合边效用损失。
- 当前协议仍沿用 `val=test` checkpoint 选择、全队列预处理/RFE和transductive患者图，因此结果是历史协议下的内部比较，不能替代独立泛化评估。

## 机制审计与设计逐项记录
1. **论文与源码差异**：论文GFR写作中的关系图/消息传播与实际`GraphLearn`、两层`GCN`、`GFR`及`simple-2`训练接口并不完全一致；实际源码中四图正则进入分类损失，而`ALoss`只计算不反传。本轮以私有PEN-free CAM接口为基座，未改根源码。
2. **Experiment43教训**：PHENO在原`i==1`分支被覆盖；恢复聚合仍未改善。43.2测得GFR相对输入约6.92×10⁻⁵、关闭全部消息无类别翻转，因此本轮要求真实邻居残差进入CAM输入。
3. **相似有害邻居**：模态相似只表示输入接近，不表示发送者特征经过当前预测头后支持接收者真实类；异质性和错误传播会使相似边带来负的模型效用。
4. **已有方法边界**：UAI 2025选择性阻断、ICLR 2025连续边方向、NeurIPS 2025边编辑影响和图反事实综述分别覆盖阻断、方向和边干预；CU-GL的候选区别是标签隔离的个体级预测效用与四模态患者图及NULL动作的统一消融验证（不预称世界首次）。
5. **B0-B身份**：E0来自Experiment43.1物理删除PEN的10折已核验checkpoint；无PEN、SPGS、RES、CLIP，四项指标与B0-B历史值一致。E1来自Experiment43.4锚定CAM-only十折。
6. **四模态与方向**：PHENO48、ANAT6、FUNC10、CORR256保留；候选池为每外层折训练发送者中的模态余弦Top-16，矩阵约定为A[receiver,sender]，j→i，不使用患者ID或测试标签。
7. **交叉拟合教师**：每个外层训练折内3折；inner-train训练轻量定向候选图+原CAM头，inner-heldout标签只在冻结后选择真实类log-prob，30个教师任务、10个OOF缓存。
8. **反事实目标**：`U(i←j,m)=log p(y_i|WITH_EDGE)-log p(y_i|WITHOUT_EDGE)`；它是冻结模型边编辑的响应，不是脑疾病临床因果。
9. **定向评分与NULL**：共享小型receiver/sender差异-乘积评分器；E4/E5/E7/E8/E9引入显式NULL质量，E9平均NULL质量 0.0580，未强制所有患者传播。
10. **两层消息**：候选发送者的完整320维表示经两层低容量投影，显式分离自特征和跨患者残差；E9新增分支123,272参数，CAM保持49,544参数，alpha从0.10起始、上限0.25。
11. **十组设计**：E0 B0-B；E1 CAM-only；E2 对称稀疏KNN；E3 定向无拒绝；E4 定向+NULL无CF；E5 对称CF；E6 CF无NULL；E7 置乱CF；E8 同质性监督；E9 完整CU-GL。
12. **正式指标**：ACC、Hard-AUC（硬one-hot预测）、SPE-TD、SEN-ASD，均为十折mean±population SD；ASD=0、TD=1。
13. **同折配对**：报告列出E9−E0/E1/E4/E7/E8均值差及严格胜折数；E9相对E0 ACC/HARD-AUC分别−0.9182/−0.9934pp。
14. **效用与分类联系**：阶段A有效OOF边334,590，|U|均值5.76×10⁻⁶，正/负约0.493/0.500；与固定置乱相关0.0036、与同质性相关0.0037，说明有边删除响应但未形成可判别训练先验。
15. **关键消融**：E9相对E4 ACC +0.2286pp、Hard-AUC +0.3184pp（但仅5/10、7/10折严格胜）；相对置乱E7及同质性E8均无稳定优势，故不能归因于反事实效用。
16. **固定E9干预**：关闭全部消息平均造成6.8个测试预测翻转、真实类log-prob变化−0.0460；PHENO/ANAT/FUNC/CORR关闭分别平均1.2/1.5/3.1/1.0翻转；随机替换发送者平均1.4翻转；原始评分方向不对称度均值 0.0387。
17. **失败机制与开销**：真实消息进入CAM且梯度非零，但E9分类低于B0-B；教师效用数量级很小且与置乱/同质性近零相关，表明当前教师响应不能提供可泛化的边选择信号。正式80创新折耗时约2106.3秒，另有30个教师任务。
18. **GO/NO-GO**：阶段A信息量门通过，但分类门、E9相对E0/E1及机制必要性门均未通过，最终为FINAL NO-GO。
19. **MICCAI创新证据**：本轮证明了结构可实现、方向不对称且NULL非零，但没有证明反事实效用带来独立疾病判别收益；不足以作为MICCAI 2027第一创新点。
20. **协议局限**：沿用`val=test`、历史全队列预处理/RFE及transductive患者图，结果仅是内部可比性；未来需独立中心、严格inductive图、独立验证集和预注册统计检验。

## 固定模型干预摘要
|条件|平均真实类log-prob变化|平均绝对变化|平均翻转数|
|---|---:|---:|---:|
|ALL_MESSAGES_OFF|-0.045992|0.141790|6.80|
|PHENO_OFF|-0.001333|0.036725|1.20|
|ANAT_OFF|-0.003418|0.039392|1.50|
|FUNC_OFF|-0.014914|0.072024|3.10|
|CORR_OFF|-0.001775|0.033109|1.00|
|RANDOM_SENDER_REPLACED|0.002715|0.052897|1.40|

## GO/NO-GO

- `GO_CLASSIFICATION`: **FAIL**（E9−E0 ACC -0.9182pp，Hard-AUC -0.9934pp，ACC逐折严格胜 2/10）。
- `GO_COUNTERFACTUAL_UTILITY`: **FAIL**；需要真实效用预检和E4/E7/E8独立差异同时成立。
- `GO_DIRECTIONAL_SELECTIVITY`: **FAIL**；本轮要求固定模型方向/NULL干预与从头消融共同支持。
- `GO_INDEPENDENT_GRAPH_VALUE`: **FAIL**；比较CAM-only锚点。
- `FINAL_DECISION`: **FINAL NO-GO**。若提升只来自普通分类或消息尺度，不能称为反事实效用机制。

## 已有工作边界与未来验证
UAI 2025选择性阻断、ICLR 2025连续边方向、NeurIPS 2025边编辑影响和图反事实综述均说明相关组件本身并非新颖；CU-GL的候选贡献是把标签隔离的个体级边干预效用与四模态患者图的有向传播及NULL拒绝统一，并用错误效用、同质性和无效用对照证伪。未来投稿需在独立中心、严格inductive协议、独立验证集和预注册统计检验下复核。

产物目录：`/data/adodas/SPromptGL-main/experiment/experiment44_counterfactual_utility_graph_learning`；总墙钟（截至报告）：4360.2s。


<!-- EXPERIMENT44_1_CUGL_R -->
## Experiment 44.1 — Reliability-Calibrated CU-GL Repair

- 状态：PREFLIGHT_SCIENTIFIC_NO_GO；阶段A：PREFLIGHT_SCIENTIFIC_NO_GO；新增正式折：0（预检停机）。
- 原Experiment44错误：四模态评分切片错误、效用监督未校准、NULL近似均匀softmax、恢复未保存完整CUDA/Python/NumPy RNG。
- 修复：显式模态切片、outer/inner训练统计、60 epoch交叉拟合边/关系教师、噪声/MAD校准、接收者留出排序、独立NULL门、完整恢复状态。
- 教师任务：6；结果哈希：{"README.md": "9047c17d3d470276e209dceb20d69ea847d711468a8014811fd6eca752700b80", "config.json": "10dccda487d615cdef4d9b27256db50e073236cb946aea7da5d6185284507024", "cugl_r.py": "3fc2f2f93b3cf033ad386e7f623659ee2d19a5c43da7b92fa52aae022481672f", "integration.py": "3efd4855cc824c5dd63aa08fe825fee3ab164b90df80ba474d8be89b0a09551a", "manifest.json": "3563110dc46b9ba5b39cfe8a4145a3af62bdfb0a75795cedf5fbd0782d440a08", "results/counterfactual.csv": "9b3378b32f4806543a267dc6fe46a70f6db0473e722d26af2bebfae816e2514c", "results/diagnostics.csv": "26ef99cfa497f3c837590d8e69e70a875205018cca22f4ee33f27c72448c8f51", "results/fold_metrics.csv": "91d457c59ce47c2b0717c33e66cc3cdfb97006196cfbdfa1661dfdd426a315d0", "results/report.md": "ae2ec12f3af32622244d6125888e39afe3307d3e9c2170c500a0def4e03118d0", "run.py": "0321b78cdc270bc7f4be2ae8bb7541d8ec40724aeabc9ea749ff195e6d3402e3", "teacher.py": "2be1e9b6e06a6864125b2d480fff4ce9060d2229d65f57831018c4a5daec5a50"}。
- GPU计划：fold1在GPU0、fold6在GPU1的阶段A；因预检停机未启动E3–E10的80折。
- 最终科研结论：PREFLIGHT_SCIENTIFIC_NO_GO；不支持继续作为MICCAI 2027第一创新点，不启动Experiment44.2。
- 详细报告：`/data/adodas/SPromptGL-main/experiment/experiment44_1_cugl_reliability_calibrated/results/report.md`。
# Experiment 44.1 — Reliability-Calibrated CU-GL Repair

状态：阶段A `PREFLIGHT_SCIENTIFIC_NO_GO`；正式新增训练组 E3–E10 未启动（预检终止规则）。

## 论文、源码和历史审计
SPromptGL论文定义四模态关系图和DGCN消息传播，公开源码的实际GraphLearn/GCN/GFR、CAM和simple-2接口与论文图示不完全相同；原GFR存在PHENO聚合覆盖，Experiment43证明简单恢复不足。Experiment44的确定错误是关系评分器用`[..., :dim]`而不是四模态真实起止列，且效用监督均值约5.8e-6未经校准。源码证据、私有源哈希和保护文件哈希见`manifest.json`。

## 历史参照（只读）

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|折数|
|---|---:|---:|---:|---:|---:|
|E0_B0B|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|10|
|E1_CAM_ONLY|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|10|
|E2_E44_ORIGINAL|90.8111 ± 3.4929|90.6112 ± 3.5220|93.3626 ± 3.5274|87.8598 ± 4.5821|10|

E0为B0-B物理删除PEN历史复现；E1为CAM-only锚定；E2为Experiment44旧CU-GL完整模型。它们不是本轮修正版的训练结果。

## 阶段A修复与科学预检
本轮统一使用 `[0:48]`、`[48:54]`、`[54:64]`、`[64:320]`，候选图归一化统计只由outer-train拟合，inner teacher再只由inner-train拟合。教师固定60 epoch，在冻结后计算边级 `U_edge` 和关系级 `U_gate`；两者是模型干预响应，不是临床因果。效用尺度由训练内MAD/分位数和FP32 no-op噪声下限确定，学生排序按接收者留出验证。NULL是独立接受门，拒绝关系时消息严格为零。

|外层折|有效边|有效关系|mean|U_edge||mean|U_gate||scale|noise floor|真实排序|随机|余弦|接收者数|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|33477|3132|7.6568335e-07|7.4852198e-07|9.5367432e-07|1.19e-07|0.4677|0.5039|0.4304|311|
|6|32965|3136|5.3364454e-07|5.2589314e-07|7.0035458e-07|1.19e-07|0.4685|0.5207|0.4218|286|

### 工程验收
- 正确切片：`PASS`；逐模态无关扰动不改变评分，相关扰动改变评分。
- 方向评分：`PASS`；评分角色交换不是候选数组反转。
- NULL严格关闭：最大残差 `0.0`。
- RNG/优化器/模型恢复逐值：`PASS`，最大参数差 `0.0`。
- 教师任务：6；两折均同时满足科学预检：`False`。

## 11组正式设计（本轮未越过阶段A）
E3仅修复四模态切片；E4修正版教师+校准效用+对称评分+NULL；E5修正版架构无CF；E6原始未校准CF；E7错配CF；E8同质性监督；E9无NULL门；E10完整CU-GL-R。若预检通过，这些组共享候选池、CAM未训练锚点、初始化、随机流、优化和val=test checkpoint规则；本轮按预注册停机条件未启动任何新增fold。

## GO/NO-GO
- `GO_CLASSIFICATION`: **FAIL / 未执行**；没有外层新增分类结果，不能用历史E2或阶段A教师结果替代。
- `GO_COUNTERFACTUAL_UTILITY`: **FAIL**；未见接收者排序、噪声与门控条件必须先通过。
- `GO_DIRECTIONAL_SELECTIVITY`: **FAIL / 未完成正式消融**；工程方向测试不等于分类必要性。
- `GO_NULL_REJECTION`: **FAIL / 未完成正式消融**；独立门控已实现但没有正式对照。
- `GO_INDEPENDENT_GRAPH_VALUE`: **FAIL / 未执行**；不能从历史协议推断。
- `FINAL_DECISION`: **PREFLIGHT_SCIENTIFIC_NO_GO**；不启动E3–E10，也不启动Experiment44.2。

## 失败解释
如果预检失败，结论是效用响应虽可计算，但在未见接收者上不能稳定预测真实边收益，或响应无法显著超过数值噪声/简单相似度；这比“模型可能过拟合”更具体。若工程切片或恢复测试失败，则优先归因于实现错误并停止；若工程通过而学生排序失败，则是科学监督不可学习。

## 局限
沿用历史val=test、全队列预处理/RFE和transductive队列，仅支持内部可比性；单seed不提供统计显著性或跨中心泛化证据。U_edge/U_gate是冻结预测模型对边/关系编辑的响应，不是临床因果。

产物目录：`/data/adodas/SPromptGL-main/experiment/experiment44_1_cugl_reliability_calibrated`。


<!-- EXPERIMENT44_1_CUGL_R_SUPPLEMENT -->
### Experiment 44.1 交付补充

- 总墙钟：439.8s。
- 阶段A：fold1、fold6各完成3个60 epoch教师，共6/30预期教师；E3–E10新增正式折0/80，因科学预检停机。
- 独立交付审计：PASS；根源码与数据保护哈希复核PASS；`.runtime`恢复状态、候选图和教师缓存保留在实验目录。
- 审计摘要：`{
  "status": "PASS",
  "protected": [
    {
      "file": "main.py",
      "before": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2",
      "after": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2",
      "pass": true
    },
    {
      "file": "model.py",
      "before": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda",
      "after": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda",
      "pass": true
    },
    {
      "file": "network.py",
      "before": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec",
      "after": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec",
      "pass": true
    },
    {
      "file": "layers.py",
      "before": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
      "after": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
      "pass": true
    },
    {
      "file": "utils.py",
      "before": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67",
      "after": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67",
      "pass": true
    },
    {
      "file": "ABIDE-simple-2-concat-weighted-cosine.sh",
      "before": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763",
      "after": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763",
      "pass": true
    },
    {
      "file": "data/ABIDE/processed_standard_data.csv",
      "before": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1",
      "after": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1",
      "pass": true
    },
    {
      "file": "data/ABIDE/modal_feat_dict.npy",
      "before": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e",
      "after": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e",
      "pass": true
    }
  ],
  "stageA_status": "PREFLIGHT_SCIENTIFIC_NO_GO",
  "teacher_tasks": 6,
  "formal_new_folds": 0,
  "historical_rows": 30
}`


<!-- EXPERIMENT44_1_CUGL_R_HASH_UPDATE -->
### Experiment 44.1 最终源码哈希补充

- README源码证据表已补齐；最终源码/结果SHA256：{"README.md": "396ceb29578aa8f638ec149305082f5b3f1f47c5640218aa9a8fb445be922c1e", "audit.py": "ced7ec6ec86eed13953ea442125c832c698cfddcd33e5c23783edd0abf48659a", "config.json": "10dccda487d615cdef4d9b27256db50e073236cb946aea7da5d6185284507024", "cugl_r.py": "3fc2f2f93b3cf033ad386e7f623659ee2d19a5c43da7b92fa52aae022481672f", "integration.py": "3efd4855cc824c5dd63aa08fe825fee3ab164b90df80ba474d8be89b0a09551a", "manifest.json": "97957babad0fc3fbae15735b0be8957b6860562f7e5570418f32e1cb6abbdd9e", "results/counterfactual.csv": "9b3378b32f4806543a267dc6fe46a70f6db0473e722d26af2bebfae816e2514c", "results/diagnostics.csv": "26ef99cfa497f3c837590d8e69e70a875205018cca22f4ee33f27c72448c8f51", "results/fold_metrics.csv": "91d457c59ce47c2b0717c33e66cc3cdfb97006196cfbdfa1661dfdd426a315d0", "results/report.md": "ae2ec12f3af32622244d6125888e39afe3307d3e9c2170c500a0def4e03118d0", "run.py": "0321b78cdc270bc7f4be2ae8bb7541d8ec40724aeabc9ea749ff195e6d3402e3", "teacher.py": "2be1e9b6e06a6864125b2d480fff4ce9060d2229d65f57831018c4a5daec5a50"}。


<!-- EXPERIMENT45_1_PID_CORRECTION_REAUDIT -->
## Experiment 45.1 - PID correction and independent re-audit

# Experiment 45.1 - PID correction and independent re-audit

状态：PID重算完成，E1-E6正式训练折数为 0（NOT_TRAINED_BY_DESIGN）。
PID准入：`PID_GATE_GO_FOR_FUTURE_TRAINING`。站点映射：`VERIFIED`。

## 完成状态

- Experiment 45 历史文件、根模型源码和数据仅读复核，SHA256 前后不变。
- Bootstrap、站点条件置乱、BROJA PID 数值求解和对抗单步测试已执行。
- E1-E6 未训练；本报告不填充新的 ACC/AUC/SPE/SEN。

## Q1 Bootstrap 错误

旧实现保留训练折局部模态聚类标签，却用全局 `y[tr[ix]]`，所以模态行与疾病标签不再来自同一患者。样本ID测试命中率从 0/48 变为 48/48；带噪冗余数据的配对 Bootstrap 中位 R 为 0.6943，标签独立数据中位 R 为 0.00019。下表 `q_legacy same-null` 用同一批新全局置乱，仅换回错误 Bootstrap，隔离配对修复的影响。

|fold|pair|q旧|q错配同零分布|q修正|旧 boot Q10|错配 boot Q10|配对 boot Q10|旧全局 perm Q95|新全局 perm Q95|旧lambda|新lambda|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|AF|0.000000|0.000000|0.000000|0.000099|0.000249|0.000542|0.002440|0.001857|0.00000000|0.00000000|
|1|AC|0.000000|0.000000|0.000000|0.000098|0.000125|0.001199|0.003218|0.001819|0.00000000|0.00000000|
|1|FC|0.000000|0.000000|0.000000|0.000340|0.000642|0.001474|0.003153|0.002850|0.00000000|0.00000000|
|2|AF|0.000000|0.000000|0.000000|0.000154|0.000341|0.000394|0.002721|0.001614|0.00000000|0.00000000|
|2|AC|0.000000|0.000000|0.000000|0.000420|0.000217|0.000502|0.002980|0.001439|0.00000000|0.00000000|
|2|FC|0.000000|0.000000|0.000000|0.000116|0.000499|0.002362|0.002640|0.002906|0.00000000|0.00000000|
|3|AF|0.000000|0.000000|0.396411|0.000197|0.000190|0.004191|0.002420|0.001643|0.00000000|0.00373180|
|3|AC|0.000000|0.000000|0.000000|0.000254|0.000984|0.001619|0.003450|0.002417|0.00000000|0.00000000|
|3|FC|0.000000|0.000000|0.000000|0.000313|0.000538|0.001057|0.001999|0.002828|0.00000000|0.00000000|
|4|AF|0.000000|0.000000|0.000000|0.000327|0.000225|0.000144|0.002475|0.001288|0.00000000|0.00000000|
|4|AC|0.000000|0.000000|0.000000|0.000376|0.000595|0.000675|0.001436|0.002698|0.00000000|0.00000000|
|4|FC|0.000000|0.000000|0.000000|0.000250|0.000276|0.000336|0.004401|0.003587|0.00000000|0.00000000|
|5|AF|0.000000|0.000000|0.000000|0.000123|0.000291|0.000338|0.003506|0.002371|0.00000000|0.00000000|
|5|AC|0.000000|0.000000|0.000000|0.000227|0.000212|0.000601|0.002705|0.001937|0.00000000|0.00000000|
|5|FC|0.000000|0.000000|0.000000|0.000498|0.000963|0.001485|0.005439|0.004399|0.00000000|0.00000000|
|6|AF|0.000000|0.000000|0.000000|0.000417|0.000301|0.000724|0.002343|0.002453|0.00000000|0.00000000|
|6|AC|0.000000|0.000000|0.000000|0.000494|0.000414|0.000369|0.003254|0.003289|0.00000000|0.00000000|
|6|FC|0.000000|0.000000|0.000000|0.000449|0.000295|0.000597|0.002834|0.002258|0.00000000|0.00000000|
|7|AF|0.000000|0.000000|0.000000|0.000308|0.000490|0.000923|0.002431|0.002129|0.00000000|0.00000000|
|7|AC|0.000000|0.000000|0.000000|0.000374|0.000637|0.000248|0.003828|0.002664|0.00000000|0.00000000|
|7|FC|0.000000|0.000000|0.000000|0.000281|0.000386|0.001155|0.003757|0.002246|0.00000000|0.00000000|
|8|AF|0.000000|0.000000|0.000000|0.000347|0.000267|0.000850|0.002830|0.002604|0.00000000|0.00000000|
|8|AC|0.000000|0.000000|0.000000|0.000453|0.000367|0.000757|0.003119|0.001744|0.00000000|0.00000000|
|8|FC|0.000000|0.000000|0.000000|0.000630|0.000505|0.000938|0.002954|0.003258|0.00000000|0.00000000|
|9|AF|0.000000|0.000000|0.000000|0.000310|0.000161|0.000294|0.002020|0.001655|0.00000000|0.00000000|
|9|AC|0.000000|0.000000|0.000000|0.000196|0.000316|0.000650|0.001815|0.003836|0.00000000|0.00000000|
|9|FC|0.000000|0.000000|0.000000|0.000295|0.000138|0.001796|0.002581|0.002307|0.00000000|0.00000000|
|10|AF|0.000000|0.000000|0.000000|0.000413|0.000742|0.000586|0.002059|0.002176|0.00000000|0.00000000|
|10|AC|0.000000|0.000000|0.000000|0.000246|0.000387|0.000783|0.003211|0.003363|0.00000000|0.00000000|
|10|FC|0.000000|0.000000|0.000000|0.000178|0.000535|0.000423|0.004883|0.003172|0.00000000|0.00000000|

## Q2-Q3 修正后 PID

|关系|R|U1|U2|S|冗余占比|q均值|lambda均值|
|---|---:|---:|---:|---:|---:|---:|---:|
|PA|0.00153373|0.00586871|0.00165864|0.00568794|0.09539710|-|-|
|PF|0.00438743|0.00301501|0.00386521|0.01163038|0.19377466|-|-|
|PC|0.00177130|0.00563114|0.00007381|0.01352820|0.09234278|-|-|
|AF|0.00123676|0.00195561|0.00701588|0.00510567|0.07286411|0.03964113|0.00037318|
|AC|0.00068758|0.00250479|0.00115753|0.00740520|0.05798873|0.00000000|0.00000000|
|FC|0.00173846|0.00651418|0.00010665|0.01296357|0.08019556|0.00000000|0.00000000|

### K=4 旧值与修正值

|关系|旧 R|新 R|旧冗余占比|新冗余占比|
|---|---:|---:|---:|---:|
|PA|0.00153373|0.00153373|0.09539707|0.09539710|
|PF|0.00438788|0.00438743|0.19379705|0.19377466|
|PC|0.00180053|0.00177130|0.09401967|0.09234278|
|AF|0.00123675|0.00123676|0.07286307|0.07286411|
|AC|0.00068759|0.00068758|0.05799035|0.05798873|
|FC|0.00173846|0.00173846|0.08019556|0.08019556|

逐折复核显示，给定当前相同输入，历史 PID 源码与 45.1 BROJA 求解器四个分量最大差异低于 1.3e-17；与实验45已落盘结果相比有 11/60 条 K=4 记录超过 1e-12，最大差异约 2.92e-4。历史 manifest 未记录 PID 源码 SHA 或 NumPy/SciPy/scikit-learn 版本，因此无法唯一归因；当前环境为 NumPy 1.26.0/SciPy 1.11.3/scikit-learn 1.3.2，与根项目 requirements.txt 中的版本不同。这更符合历史离散化/数值运行状态或历史结果源码未留 hash 的差异，而不是本轮数学目标更改。


### K 敏感性

|K|关系|平均 R|平均冗余占比|
|---:|---|---:|---:|
|3|PA|0.00067163|0.08947306|
|3|PF|0.00389136|0.20328577|
|3|PC|0.00078779|0.07104578|
|3|AF|0.00054282|0.04568222|
|3|AC|0.00038327|0.11334237|
|3|FC|0.00078850|0.05474116|
|4|PA|0.00153373|0.09539710|
|4|PF|0.00438743|0.19377466|
|4|PC|0.00177130|0.09234278|
|4|AF|0.00123676|0.07286411|
|4|AC|0.00068758|0.05798873|
|4|FC|0.00173846|0.08019556|
|5|PA|0.00012350|0.02764758|
|5|PF|0.00460231|0.12517823|
|5|PC|0.00191551|0.07419677|
|5|AF|0.00177490|0.05908614|
|5|AC|0.00071004|0.03691200|
|5|FC|0.00181712|0.05133307|

K=3/4/5 共 180 条主 PID 记录；K=4 的 AF/AC/FC 共 30 条诊断记录。Bootstrap 只在冻结训练折的离散化结果上重采样，不包含重新拟合 scaler/KMeans 的不确定性。

## Q4 站点混杂诊断

站点列依据 `modal_feat_dict.npy` 中的 PHENO 特征名和 `processed_standard_data.csv` 的精确列名恢复：20 个 `SITE_ID_*` 列；每个患者恰有一个 1。站点条件置乱从原始训练标签开始，在每个训练折站点内部置换，保持总计数和每站点 ASD/TD 计数。它是混杂敏感性诊断，不替代冻结 q 公式的全局置乱零分布。

|关系|平均全局置乱 Q95(R)|平均站点置乱 Q95(R)|
|---|---:|---:|
|AF|0.00197908|0.00309138|
|AC|0.00252056|0.00301757|
|FC|0.00298108|0.00382890|

站点敏感性 q 使用同一 Bootstrap、只将 q 的零分布替换为站点条件置乱；它不参与准入。

|关系|全局 q 正值折数|站点敏感性 q 正值折数|全局 q 均值|站点敏感性 q 均值|
|---|---:|---:|---:|---:|
|AF|1|0|0.03964113|0.00000000|
|AC|0|0|0.00000000|0.00000000|
|FC|0|0|0.00000000|0.00000000|

### Fold lambda

|关系|10折均值|最小值|最大值|
|---|---:|---:|---:|
|AF|0.000373180|0.000000000|0.003731801|
|AC|0.000000000|0.000000000|0.000000000|
|FC|0.000000000|0.000000000|0.000000000|

旧 lambda_bar 均值=0.000000000；修正后逐折 `mean(lambda_pair)` 的均值=0.000124393，最大值=0.001243934。旧实现用 `0.03*mean(q)*mean(r)`，修正版逐折求三条 lambda 后取均值。


## Q5 PID 定义与求解

历史代码名为 I_min，但实际求解的是 BROJA 式两源离散 PID：在保持 `P(X1,Y)` 与 `P(X2,Y)` 的耦合集合中最小化 `I_Q((X1,X2);Y)`，再由 `I1/I2/Ijoint/Ijoint_min` 分解 R、U1、U2、S。主解用解析梯度 SLSQP；每个解检查边缘约束、目标值复算、恒等式和分量非负性。无 CVXPY 时，低维审计用独立参数化的 SciPy trust-constr 求解器交叉核验。数值恒等式通过不代表有限样本统计可靠。

## q 门槛功效模拟

样本量与 ASD/TD 比例匹配 fold 1 训练折，K=4，每对使用 12 次 Bootstrap 和 19 次全局置乱，重复三次。模拟结果只诊断冻结门槛，不改阈值。

|合成机制|GO次数/3|平均 q|平均 lambda|
|---|---:|---:|---:|
|strong_redundancy|3/3|0.932968|0.02778636|
|weak_redundancy|3/3|0.672506|0.01884468|
|unique_dominant|1/3|0.003396|0.00000026|
|synergy_dominant|0/3|0.000000|0.00000000|
|label_independent|0/3|0.000000|0.00000000|

强冗余正对照若未稳定触发 GO，说明当前 12/19 次重采样门槛检验功效有限；零值不能解释为连续模态不存在疾病相关共享信息。

## Q6 对抗修复与逻辑一致性

关系 i/j 的表示沿样本维构造 `[z_i; z_j]`，输入为 `[2*N_train,64]`；来源 0/1 分别是第一个/第二个关系图。条件标签是同一训练患者标签的两份复制，判别器输入固定为 66 维。GRL 实际梯度满足 `grad_GRL = -lambda * grad_plain`；E2/E3 使用相同逐折 `lambda_bar`，E4 使用 AF/AC/FC 各自 lambda，E5 只循环置换归属，E6 保留辅助/对抗/GFR结构而仅把主分类输入旁路到原始 X。GPU0/GPU1 均通过 E1-E6 单步、双步 B0-B 等价、来源学习和 GRL 梯度检查。状态机 dry-run 接受有效 PID_GO 签名下的 TRAINING 并拒绝过期签名；正式 worker 未启动。

## Q7 科研结论

修复后的准入为 `PID_GATE_GO_FOR_FUTURE_TRAINING`：30 条 AF/AC/FC 关系-折记录中 1 条 lambda 为正，唯一正值来自 AF fold 3；AC 与 FC 十折均为零。按冻结的‘至少一个正值且全体 lambda 范围大于 1e-6’规则，Experiment 45 原 NO-GO 不再成立，建议进入后续正式对照训练，但只作为可执行的探索性实验，不代表 PID-DGAA 创新成立。当前稀疏通过和功效模拟必须在解释中保留。E1-E6 本轮仍为 0/60 折。

下一阶段应保留 E0/E1/E2/E3/E4/E5/E6 同折对照，报告每折 lambda 与领域分类/特征梯度诊断，检查 E4 相对 E3/E5 的逐折增量及 E6 图旁路；完成后再独立评估多 seed 与无偏验证。历史 val=test、CV 前全局预处理/RFE 和 transductive 图仍限制外部泛化解释。

## 交付

实际主 PID 估计数：180；有效关系诊断：30；正式新增训练折：0；结果目录：`/data/adodas/SPromptGL-main/experiment/experiment45_1_pid_correction_reaudit/results`。


<!-- EXPERIMENT46_PGIV -->
## Experiment 46 — Patient Graph Incremental Value Probe

- 状态：FORMAL_COMPLETE；新增探针折：80/80；历史参考折：20。
- GPU0：fold1–5；GPU1：fold6–10；冻结CAM特征和候选图每fold缓存一次。
- 最终判定：NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE。
- 源码、结果和manifest哈希：{"README.md": "43a84799aa9a22050ecb8f5569cf8f267074b3ef8909ea4277ad025ea70f20ec", "audit.py": "40b79722ff1575718be6849fadd8c5f2d6f25aa847b16aef8c621cc88e4d4733", "config.json": "9e7466b49633fba5b379cb375ee71bae4066bddf0b76030fa203fffc35fe8ef6", "integration.py": "39a41152aa1deb81849e62003fb35797adaa46715e88c3d5145943bcaec996a7", "manifest.json": "65e7ed83cc1130b9f1a5010f9a1e21cac9070ad59407f9fd563afc73a08425f4", "probe.py": "492f3e38b6b2add76644d1edabed5621041d3e802df61c5fe012e70ca43d2708", "results/diagnostics.csv": "fe0f750b0e0cd527d2de9ba27ecca048a220c25204a1be3a6bef1602f0b1c836", "results/fold_metrics.csv": "741f6e66c555fdbf6116f007aa68a8f53b3d3abbc23c68d562f30be9ff4c5e2d", "results/report.md": "01478d297fe301161d922aab898a86e9e3f78b14a10d63ecbdcd553632d35598", "run.py": "a72f98b70cc021e6b094f1de88f2acb43492e212645fa3020240d786a4dec852"}。
- 详细报告：`/data/adodas/SPromptGL-main/experiment/experiment46_patient_graph_incremental_value_probe/results/report.md`。
# Experiment 46 — Patient Graph Incremental Value Probe

状态：80/80新增探针折，20条历史参考折。

## 方法与协议
冻结Experiment43.4 E0_ANCHORED_PHYSICAL_CAM，实际提取OutputLayer.mlp_head后的36维hidden；并行使用原始320维输入。四模态候选为outer-train发送者的训练折Top-16余弦关系，训练接收者排除自身，测试接收者只接收outer-train消息。每个Arm使用LayerNorm→Linear(5d,36)→GELU→Dropout→Linear(36,2)探针，四组共享逐张量初始化。历史val=test、预CV RFE、原始标准化和transductive协议均保留。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|折数|
|---|---:|---:|---:|---:|---:|
|R0_B0B_REFERENCE|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|10|
|R1_CAM_REFERENCE|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|10|
|H0_SELF|90.1202 ± 3.0589|90.0282 ± 3.0002|91.2211 ± 4.4633|88.8354 ± 3.1903|10|
|H1_TRUE|90.3501 ± 2.6412|90.2650 ± 2.6034|91.4385 ± 3.7317|89.0915 ± 2.9470|10|
|H2_RANDOM|90.9261 ± 2.4446|90.7752 ± 2.4258|92.7151 ± 3.5251|88.8354 ± 3.1997|10|
|H3_PERMUTED|90.5799 ± 2.9970|90.3954 ± 3.0022|92.9371 ± 4.0806|87.8537 ± 4.5758|10|
|X0_SELF|88.7448 ± 3.8476|88.6437 ± 3.8310|89.9399 ± 4.4298|87.3476 ± 4.0719|10|
|X1_TRUE|83.4653 ± 4.9416|83.3321 ± 5.0688|85.2313 ± 5.6818|81.4329 ± 8.6534|10|
|X2_RANDOM|79.9046 ± 5.0284|79.7942 ± 5.1725|81.6189 ± 4.7158|77.9695 ± 8.5075|10|
|X3_PERMUTED|83.0029 ± 4.7748|82.9008 ± 4.7938|84.1674 ± 5.2291|81.6341 ± 5.7022|10|

## 核心同折配对
|比较|ΔACC|ΔHard-AUC|ΔSPE|ΔSEN|ACC胜/平/负|Hard-AUC胜/平/负|
|---|---:|---:|---:|---:|---:|---:|
|H1_TRUE−H0_SELF|+0.2299|+0.2367|+0.2174|+0.2561|2/7/1|4/3/3|
|H1_TRUE−H2_RANDOM|-0.5760|-0.5102|-1.2766|+0.2561|1/3/6|2/2/6|
|H1_TRUE−H3_PERMUTED|-0.2299|-0.1304|-1.4986|+1.2378|3/3/4|4/2/4|
|X1_TRUE−X0_SELF|-5.2795|-5.3116|-4.7086|-5.9146|0/2/8|1/1/8|
|X1_TRUE−X2_RANDOM|+3.5606|+3.5379|+3.6124|+3.4634|7/2/1|8/0/2|
|X1_TRUE−X3_PERMUTED|+0.4624|+0.4313|+1.0638|-0.2012|4/1/5|5/0/5|

## 固定 TRUE checkpoint 干预
|Arm|条件|平均log-prob变化|平均绝对变化|平均翻转|
|---|---|---:|---:|---:|
|H|H0_SELF|-0.014977|0.212723|1.70|
|H|H2_RANDOM|+0.010385|0.116468|1.40|
|H|H3_PERMUTED|+0.005003|0.111367|1.20|
|X|X0_SELF|-0.114345|0.506557|19.70|
|X|X2_RANDOM|+0.048725|0.311044|9.80|
|X|X3_PERMUTED|+0.010648|0.331492|10.00|

## GFR审计与8组定义
原论文GFR通过四模态相似关系图和两层GCN消息传播把患者间信息送入CAM；本轮没有重新训练GraphLearn、GCN、GAT、Transformer、PEN或任何文本模块。实验43.2显示历史GFR输出相对输入约6.92e-5且关闭消息无类别翻转，实验44/44.1也没有证明反事实效用监督有独立增益。因此本轮冻结实验43.4物理CAM-only编码器，只训练同结构轻量探针，直接检验固定患者关系的增量价值。
H0/H1/H2/H3分别是CAM36自身重复槽位、真实Top-16四模态邻居、固定随机训练患者和真实支持下的双射错配；X0/X1/X2/X3在原始320维空间使用相同四种输入方式。每个Arm的探针均为LayerNorm→Linear(5d,36)→GELU→Dropout(0.35)→Linear(36,2)，H参数量6,950，X参数量60,910；同一Arm同一fold四组逐张量共享初始化、优化器和训练随机流。
邻居候选严格由训练折发送者构建：PHENO 0:48、ANAT 48:54、FUNC 54:64、CORR 64:320，训练拟合标准化后取余弦Top-16；训练接收者排除自身，测试接收者只从outer-train接收。RANDOM和PERMUTED均不读取标签。

## 同折预测纠错
|Arm|对照→TRUE|对照错且TRUE对|对照对且TRUE错|
|---|---|---:|---:|
|H|SELF→TRUE|10|8|
|H|RANDOM→TRUE|7|12|
|H|PERMUTED→TRUE|10|12|
|X|SELF→TRUE|22|68|
|X|RANDOM→TRUE|87|56|
|X|PERMUTED→TRUE|44|40|

## 关系与站点诊断
|Arm|邻居方式|同类比例|站点内比例|
|---|---|---:|---:|
|H|TRUE|0.5210|0.5528|
|H|RANDOM|0.5014|0.0810|
|H|PERMUTED|0.5023|0.0811|
|X|TRUE|0.5210|0.5528|
|X|RANDOM|0.5014|0.0810|
|X|PERMUTED|0.5023|0.0811|

## 预注册判定
- `GO_PREDICTIVE_INCREMENT`: **FAIL**；X1必须同时超过X0/X2/X3并满足7/10折门槛。
- `GO_REPRESENTATION_SPACE_ONLY`: **FAIL**。
- `GO_PRACTICAL_B0B_IMPROVEMENT`: **FAIL**。
- `FINAL_DECISION`: **NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE**。

## 科研结论与局限
H1相对H0只有+0.2299pp ACC且仅2/10折严格胜；H1低于RANDOM和PERMUTED，说明CAM36空间的真实相似邻居没有表现出稳定的独立增量。RAW320中X1为83.4653% ACC，低于X0的88.7448%，虽高于随机上下文但没有超过自身特征基线和错配对照，且SEN-ASD下降5.91pp。固定TRUE checkpoint替换邻居会改变连续输出（H平均1.70、X平均19.70次翻转），证明探针实际读取邻居槽位，但“产生影响”不等于“提供有益增量”。
冻结CAM checkpoint本身经过历史val=test选择；本轮探针继续使用outer-test ACC选最佳epoch。原始RFE/标准化和transductive队列均保留，单seed结果不构成独立泛化或严格因果证据。结果支持暂停围绕固定Top-16患者相似图继续堆叠GFR创新：当前受控协议未发现稳定的真实患者图增量价值，不应启动Experiment47。

产物目录：`/data/adodas/SPromptGL-main/experiment/experiment46_patient_graph_incremental_value_probe`。


<!-- EXPERIMENT46_PGIV_FAIRNESS_CORRECTION -->
## Experiment 46 — PGIV fairness-corrected final result

- 先前追加的初步PGIV块发现探针组间未恢复统一post-initialization RNG；该初步数字不作为最终比较依据，历史记录保留以便审计。
- 修正：每个Arm/fold/组在训练前恢复同一anchor start RNG；80/80新增fold已从头重跑，20个Arm×fold的start_rng_sha各自一致，80/80 checkpoint重载审计通过。
- 最新最终判定：`NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE`；完整四指标和配对诊断见 `/data/adodas/SPromptGL-main/experiment/experiment46_patient_graph_incremental_value_probe/results/report.md`。
- 修正后源码与结果哈希：{"README.md": "43a84799aa9a22050ecb8f5569cf8f267074b3ef8909ea4277ad025ea70f20ec", "audit.py": "40b79722ff1575718be6849fadd8c5f2d6f25aa847b16aef8c621cc88e4d4733", "config.json": "9e7466b49633fba5b379cb375ee71bae4066bddf0b76030fa203fffc35fe8ef6", "integration.py": "39a41152aa1deb81849e62003fb35797adaa46715e88c3d5145943bcaec996a7", "probe.py": "492f3e38b6b2add76644d1edabed5621041d3e802df61c5fe012e70ca43d2708", "results/diagnostics.csv": "eb2aaa3e98214ba1ef1b2a678d9d3ca1e6eed688fc17e4813fad5e7ce92acf47", "results/fold_metrics.csv": "e6d9cc05bb8b4328091ae68896090daa16c323d68e84008a57fb82b776a8fdad", "results/report.md": "dcd710c1d97cb474f08829f165d6c7511f8a1fd6a437b360927be073adbfff8f", "run.py": "f6fe2de14ad669dac606fa11c9e1016dbe1b99086eda673035cee3ade0de96ab"}

## Experiment 47.3 — Patient Graph Relation-Specific Value Verification

- 状态：COMPLETE；在 `/tmp/SPromptGL-E47/experiment47_3_patient_graph_relation_value_verification` 执行，根目录源码/数据/历史 checkpoint 未修改。
- 目标：修正 47.2 尺度审计，验证原 D2 实际扰动，并用 D4 站点分层节点重标记、D5 站点分层消息接收置换检验真实患者关系特异性。
- 历史复用：D1/D2/D3 共 30 个已核验折；新增 D4/D5 共 20/20 完整训练折。GPU0 folds1–5，GPU1 folds6–10；GPU1 通过 `CUDA_VISIBLE_DEVICES=1` 映射。
- 47.2 修正：B0-B alpha=1 测试 ACC 91.7294%；T1 alpha=1 71.2970%；T1 alpha=0 54.2045%；T1 centered alpha=1 66.3623%。
- 强负控制有效性：D4 10/10 折聚合消息距离≥0.25，均值 1.0524；D5 10/10 折，均值 1.2050；均无 NaN。D4 的对称性/边数/站点对边数/train-test 分层检查通过；D5 消息均值和中心化范数保持。
- 十折结果（ACC / Hard-AUC / SPE / SEN，mean±population SD，%）：D1 90.6962±2.9809 / 90.6190±3.0133 / 91.6466±3.4323 / 89.5915±4.1041；D2 90.4676±2.8717 / 90.2928±2.8866 / 92.7197±3.6359 / 87.8659±4.2934；D3 88.3986±3.4762 / 88.3317±3.4310 / 89.0842±4.9485 / 87.5793±4.1998；D4 91.3858±3.0146 / 91.1936±3.0325 / 93.7835±3.5496 / 88.6037±4.1403；D5 90.5812±3.2908 / 90.4436±3.3060 / 92.2895±4.0005 / 88.5976±4.4536。
- 配对：D1−D2 ACC +0.2286 pp（4/4/2），Hard-AUC +0.3263（7/1/2）；D1−D3 ACC +2.2975（8/2/0），Hard-AUC +2.2873（9/0/1）；D1−D4 ACC −0.6897（1/3/6），Hard-AUC −0.5746（3/1/6）；D1−D5 ACC +0.1149（4/3/3），Hard-AUC +0.1755（5/2/3）。
- 结论：D1 相对 D3 支持有限的一般跨患者传播作用，但 D4 反而高于 D1、D5 与 D1 相近；没有真实患者—消息对应的稳定独立增量，且 D1 低于 CAM-only 91.7294%。
- 最终判断：`NO-GO`（当前原始 GFR 固定患者图的关系特异性 MICCAI 第一创新点）；不启动 47.4/48。该结论不等同于所有患者图方法永远无效。
- 关键结果目录：`/tmp/SPromptGL-E47/experiment47_3_patient_graph_relation_value_verification/results`；代码、输入、源快照和受保护文件 SHA 见 `manifest.json`。
- 新增训练 wall time：约 460.9 秒；OSSFS handoff 同步状态以实际追加尝试为准。

- SHA补充：results/report.md=8ed91aad3c26128579333f15e0bc5d32d467e48c6f6fe761e21d0a07f8946fca；results/fold_metrics.csv=f5706210094284dffaec4791e9e90bada5a7a9051847219c731e8ca79db65769；results/strong_null_effectiveness.csv=68d26ae551d87bd7f5818002e78526d7d54732aee1db908865c650374c9b0b8f；results/paired_analysis.csv=18a04faee26dc5dda9864e03ebe1cbee42fc40ba14cad50f401511002adc112e；manifest.json=ef767aef719e18bf42682f3ba7fe00ea9a024e3d500dedda863026a1324e2f84。源码快照、输入数据及受保护根文件 SHA 详见本实验 `manifest.json`。


<!-- EXPERIMENT48_DPA_GFR_SELECTIVE_ADVERSARIAL_ALIGNMENT -->
## Experiment 48 — DPA-GFR：判别性保持的选择性对抗患者图学习

- **实验路径与状态**：隔离执行于 `/tmp/experiment48_dpa_gfr_selective_adversarial_alignment`；8组×10折共80/80完成。E0使用Experiment43.1已核验B0-B最佳检查点并在独立进程读回；E1–E7共70个新训练折从同折B0-B锚点开始。加速期间发现折2/7的新增组队列重叠，14个受影响任务已删除并以单GPU队列清洁重跑，未将竞态结果计入最终报告。
- **目标与假设**：检验真实非自环患者关系是否具有超出自身容量的疾病判别增量（H1）、CORR-FUNC选择性对齐是否优于全部关系对齐（H2）、PCGrad是否在对抗梯度冲突下保持分类方向（H3）。保留原始B0-B GFR/CAM、双Adam、reg=0.11、450轮上限、wait>200、历史 `val_idx=test_idx` checkpoint规则；PEN物理移除。新增图发送者仅为outer-train，K=16，测试患者不参与构图或编码器/域判别器标签训练。
- **论文/源码审计**：核对SPromptGL GFR式(1)、DGCN式(2)、CAM式(4)、PEN/RES式(5–6)、总损失式(7)；DeepASD模态判别器min-max式(1–5)、余弦图式(6)、S2GC及联合损失式(9)；AAAI 2026 PID约束 `I=R+U1+U2+S`、`I1=R+U1`、`I2=R+U2`；PCGrad负内积冲突与法向投影规则。源码位置、历史Experiment43.1/43.2/43.4/46/45/45.1证据和论文SHA见 `.runtime/literature_audit.json` 与 `.runtime/source_history_audit.json`。
- **四指标（十折 mean ± population SD，%）**：

|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E0_B0B_REFERENCE|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|E1_SELF_CAPACITY|91.6144 ± 3.1353|91.5494 ± 3.0933|92.2757 ± 5.1619|90.8232 ± 4.1719|
|E2_TRUE_GRAPH_ONLY|90.8124 ± 3.1326|90.6828 ± 3.1214|92.2803 ± 3.8116|89.0854 ± 3.6935|
|E3_FULL_ADVERSARIAL|90.4663 ± 3.4190|90.2579 ± 3.4298|93.1499 ± 3.3062|87.3659 ± 4.1229|
|E4_SELECTIVE_ADVERSARIAL|90.4676 ± 3.6059|90.3458 ± 3.5865|91.8501 ± 4.1297|88.8415 ± 3.6895|
|E5_DPA_GFR_FULL|90.3514 ± 3.2264|90.2285 ± 3.2209|91.8594 ± 3.8519|88.5976 ± 3.9804|
|E6_RANDOM_GRAPH_CONTROL|91.9592 ± 2.7747|91.7722 ± 2.7337|94.2091 ± 3.7586|89.3354 ± 2.9313|
|E7_WRONG_PAIR_CONTROL|90.0065 ± 4.1526|89.9629 ± 3.9541|90.5782 ± 7.2976|89.3476 ± 4.0938|

- **核心配对差（百分点，ACC / Hard-AUC）**：E2−E1 `−0.8020 / −0.8666`（ACC 1/2/7胜平负）；E3−E2 `−0.3461 / −0.4250`；E4−E3 `+0.0013 / +0.0879`；E4−E2 `−0.3448 / −0.3370`；E5−E4 `−0.1162 / −0.1173`；E5−E0 `−1.3780 / −1.3761`；E5−E6 `−1.6079 / −1.5437`；E5−E7 `+0.3448 / +0.2656`。
- **机制证据**：真实图消息非零、K=16且每折0自环；E2 graph-only ACC均值约74.5%，但最终图依赖没有转化为正增量。E5域判别器末期准确率约1.00，梯度冲突约77.6%，PCGrad投影非零；E5仍低于E4，说明机制触发不等于分类收益。E6随机图反而高于E5，否定真实相似邻居在该新增通路中的可靠独立价值。E5相对E7略高，但仍低于B0-B，不能据此声称疾病相关选择性对齐成立。
- **合成/工程控制**：真实同类人工图与标签相关系数1.0，随机图0.2227；人工模态来源域判别准确率1.0；GRL符号、非自环、PCGrad投影均PASS。预检PASS；独立80/80读回PASS；根源码、数据和历史锚点保护SHA未变。
- **冻结GO/NO-GO**：`GO_GRAPH=NO-GO`、`GO_SELECTIVE_ALIGNMENT=NO-GO`、`GO_GRADIENT_PROTECTION=NO-GO`、`GO_FINAL_MICCAI_CANDIDATE=NO-GO_FOR_CURRENT_DPA_GFR`。当前不建议把DPA-GFR作为MICCAI 2027第一创新点提交，也不建议在不改变科学问题和无偏验证协议前继续堆叠对抗/图模块。该结论只针对本数据、历史评估偏倚和seed=0协议，不证明所有患者图或条件对齐方法普遍无效。
- **资源/耗时**：GPU0 folds1–5、GPU1 folds6–10；E1–E7训练任务累计墙钟约4115秒，正式任务窗口约2296秒（并行）；E0为已核验检查点读回。无额外seed、PEN、GFR重训练或PID硬门槛。
- **交付文件与SHA**：报告 `[report.md](/tmp/experiment48_dpa_gfr_selective_adversarial_alignment/results/report.md)` SHA `2e4b49d457b3baec0ddb14fca3934938f7722e14801d6990bf4decd6491f36c2`；逐折 `[fold_metrics.csv](/tmp/experiment48_dpa_gfr_selective_adversarial_alignment/results/fold_metrics.csv)` SHA `0117427cef981c2d12d87a1620a8415b5ba490d9466e82e6fbbec74a258f2362`；汇总 `[summary.csv](/tmp/experiment48_dpa_gfr_selective_adversarial_alignment/results/summary.csv)`；代码签名 `91cd94f5e30fca5db86faa38f7d444e324cb83562754f6686d6f94253dc480f7`；manifest `[manifest.json](/tmp/experiment48_dpa_gfr_selective_adversarial_alignment/manifest.json)`。完整训练检查点、任务状态、预检、论文审计和独立读回在该目录 `.runtime/`。


### Experiment48最终审计补充（站点敏感性与最终哈希）

- 只读站点诊断已完成：`results/site_diagnostics.csv` 共120行，覆盖E2真实图、E5真实图、E6随机图的4关系×10折；真实E2/E5同站点边比例约48.92%/48.45%，E6为8.05%。CORR同类边约70.0%、跨站点约87.5%，FUNC同类边约52.3%、跨站点约47.4%，因此同类富集包含站点与标签混合相关性，不能视作ASD生物学证明。
- 最终报告SHA更新为 `3b0807905c1f9389ddb171623c3a1969529def786535d477666c0153dd7387b1`；站点诊断SHA为 `2c9afc14d6019509854cbc9889994f8dbcaa21a1c8acfc1e9a0197e6ff88d156`；manifest已记录 `site_diagnostics=PASS_120_rows_E2_E5_E6`。

<!-- EXPERIMENT47_MISSING_ENTRIES_FILLED_2026-10-28 -->


## Experiment 47 — GFR Fundamental Problem Discovery（缺项补齐，2026-10-28）

- **执行状态**：Stage A 静态原图审计与 Stage B 固定 checkpoint 干预、条件信息探针均完成，10/10 折；未启动 Stage C 重训。GPU0 负责 folds 1–5，GPU1 负责 folds 6–10。
- **基线核验**：B0-B 与 Experiment43.1 逐折预测、混淆计数和四项指标一致：ACC 91.7294±2.8596%、Hard-AUC 91.6046±2.8059%、SPE-TD 93.1360±4.5354%、SEN-ASD 90.0732±3.5117%；CAM-only 与 Experiment43.4 一致：ACC 91.7294±3.0819%、Hard-AUC 91.4880±3.0586%、SPE-TD 94.6346±3.7878%、SEN-ASD 88.3415±3.3369%。
- **源码/机制发现**：四路 GFR 均从完整 320 维输入传播；`i==1` 的 `combined_feat=f_w*feat1+raw_features` 覆盖 PHENO 消息；simple-2 实际优化分类损失与四图 GraphConstructLoss 均值，`ALoss` 只计算不反传。原始 GFR 最终消息/输入约 `7e-05`，关闭全部消息无稳定类别翻转。
- **问题归因**：P1 图疾病结构为 `INCONCLUSIVE`（CORR/FUNC 有小而跨折稳定的统计关系，不能据此证明条件增量）；P2 站点聚集为 `OBSERVED_ASSOCIATION_ONLY`；P3 条件邻居增量为 `SUPPORTED_BUT_NOT_FULLY_ATTRIBUTED`（真实邻居不超过自身特征）；P4 过平滑为 `HYPOTHESIS_NOT_SUPPORTED`；P5 图分支不可识别/优化捷径为 `VERIFIED_MAJOR_GFR_PROBLEM`；P6 测试 cohort 依赖为 `HYPOTHESIS_NOT_SUPPORTED`；P7 跨模态冲突为 `OBSERVED_ASSOCIATION_ONLY`。
- **科学结论**：原始 GFR 的图关系并非完全没有统计结构，但在残差直通、PHENO 覆盖和损失/梯度路径下没有形成可辨认的增量决策通路。该问题支持后续问题导向归因研究，不支持宣称已有解决方案或 MICCAI 2027 第一创新点成立；不应把低同质性、站点聚集或固定干预单独包装成疾病机制。
- **限制与交付**：沿用 `val=test`、外层前 RFE/标准化、871 人 transductive 图和单 seed；结论是内部机制证据，不是无偏独立泛化或临床因果证据。执行目录为 `/tmp/SPromptGL-E47/experiment47_gfr_fundamental_problem_discovery`；报告 SHA256=`c78df52537498ec6af37061cf754cd3c4e7c48543714ce37657014722fec3905`，manifest SHA256=`154d72fec99b79a285ff5d398b3f60f2070c0d8ed7a37c9154d21fdcc3feece5`。


## Experiment 47.1 — GCN Message Amplitude Collapse Attribution（缺项补齐，2026-10-28）

- **执行状态**：Stage A 十折逐层静态分解、Stage B 预定 fold1/fold6 的 B0-B 精确轨迹复演、Stage C 单因素训练均完成；T0 两折、T2 两折为 pilot，T1 扩展至 10 折，共 14 个完整训练任务。GPU0 负责 folds 1–5，GPU1 负责 folds 6–10。
- **精确复演**：T0 fold1/fold6 首步损失、双 Adam 状态、RNG、best epoch（157/155）、预测和四项指标逐值匹配历史 B0-B；未混用 B0-A 或 PEN checkpoint。
- **幅值归因**：ANAT/FUNC/CORR 初始化 W1/W2 范数约 25，训练后 W1/W2 比例约 0.0023–0.0040/0.0015–0.0026；最佳模型邻接归一化比例约 0.93–0.99，不能解释多数量级衰减。固定“初始图+初始 W”可恢复百量级消息，而“最佳图+初始 W”恢复、“初始图+最佳 W”不恢复，说明主要幅值损失来自训练后的 GCN W1/W2 收缩，ReLU 稀疏化为次要贡献。
- **单因素结果**：T1 只取消 ANAT/FUNC/CORR 六个 GCN W 参数的 weight decay，十折 ACC 71.2970±4.7123%、Hard-AUC 70.9726±4.8432%、SPE-TD 75.4209±5.8353%、SEN-ASD 66.5244±8.6896%；T0/T2 仅两折 pilot，不生成十折均值。图正则不直接对 GCN W1/W2 反传；T2 pilot 未形成一致的图正则因果链。
- **消息与分类关系**：T1 的消息确实进入决策，但其分类显著低于 B0-B；T0 原始消息关闭无类别翻转。去掉 weight decay 只恢复幅值，不恢复真实患者关系的疾病增量。
- **最终判定**：`VERIFIED_GCN_WEIGHT_DECAY_ATTRIBUTION`（仅针对幅值衰减）；`AMPLITUDE_RESTORED_NO_INCREMENTAL_VALUE`；图目标归因为未证实；`FINAL_NO_GO_FOR_AMPLITUDE_ONLY_GFR_INNOVATION`。不建议把“放大消息/去 weight decay”作为 MICCAI 2027 第一创新点。
- **限制与交付**：仍沿用历史 `val=test`、预 CV RFE/标准化、transductive 图和单 seed；T1 十折是归因扩展，T0/T2 仅 pilot。执行目录为 `/tmp/SPromptGL-E47/experiment47_1_gcn_message_collapse_attribution`；报告 SHA256=`7e6c4331eb726a582b0581f95cc68a1f8cf884a7a9e6256d1243db9503820255`，manifest SHA256=`a118345377f90c3b622232e43fcf9b8cd12326bbd9d683be557a07dee186adb1`。原先追加曾受 OSSFS OSError(5) 阻塞，本次在根 handoff 完成补记。


## Experiment 47.2 — Active Patient Graph Failure Attribution（缺项补齐，2026-10-28）

- **执行状态**：Stage A 十折只读诊断、fold1/fold6 的 D0–D3 pilot、Gate A/B 训练侧准入及剩余八折扩展全部完成，共 40/40 个新增训练折；GPU0 负责 folds 1–5，GPU1 负责 folds 6–10。扩展未使用 outer-test ACC 选择。
- **方法核验**：固定 `rho=0.25`，只用 outer-train 估计消息中心化与尺度；D0 动态真实图继续优化 GraphLearn/原图正则，D1/D2/D3 冻结图且只优化同容量 GCN/CAM。D2 保持对称性、每节点非自环度数、总边数、自环、权重多重集和站点对边数等预注册结构约束；D3 为仅 self-loop。
- **正式四指标（十折 mean±population SD，%）**：R0 B0-B=`91.7294±2.8596 / 91.6046±2.8059 / 93.1360±4.5354 / 90.0732±3.5117`；R1 CAM-only=`91.7294±3.0819 / 91.4880±3.0586 / 94.6346±3.7878 / 88.3415±3.3369`；R2 T1=`71.2970±4.7123 / 70.9726±4.8432 / 75.4209±5.8353 / 66.5244±8.6896`；D0=`90.4676±2.7781 / 90.3614±2.6960 / 91.6374±4.5046 / 89.0854±2.7628`；D1 固定真实图=`90.6962±2.9809 / 90.6190±3.0133 / 91.6466±3.4323 / 89.5915±4.1041`；D2 匹配重连=`90.4676±2.8717 / 90.2928±2.8866 / 92.7197±3.6359 / 87.8659±4.2934`；D3 仅 self-loop=`88.3986±3.4762 / 88.3317±3.4310 / 89.0842±4.9485 / 87.5793±4.1998`（顺序均为 ACC / Hard-AUC / SPE-TD / SEN-ASD）。
- **配对证据**：D1−D2 的 ACC `+0.2286 pp`（4/4/2）、Hard-AUC `+0.3263 pp`（7/1/2），未同时达到预注册七折门槛；D1−D3 的 ACC `+2.2975 pp`（8/2/0）、Hard-AUC `+2.2873 pp`（9/0/1），说明跨患者传播相对纯 self-loop 有模型依赖，但不能排除一般图传播/尺度训练效应。D0 的训练 NLL相对 T1 降至同折约 0.309/0.383 倍，支持尺度/优化失衡确实存在。
- **科学结论**：`SCALE_OPTIMIZATION_FAILURE_VERIFIED`（优化层面）和 `ACTIVE_GRAPH_NONDISCRIMINATIVE`（当前真实关系增量检验）。D1 与 D2 几乎相同，真实患者关系没有稳定超越结构匹配错误关系；D1 低于历史 B0-B/CAM-only。动态 GraphLearn 的独立失败原因未闭合，不应继续只增大消息、取消 weight decay 或堆叠门控。
- **限制与交付**：固定干预存在分布偏移；历史 `val=test`、全队列 RFE/标准化、871 人 transductive 图和单 seed 仍在。执行目录为 `/tmp/SPromptGL-E47/experiment47_2_active_graph_failure_attribution`；报告 SHA256=`907cdbe7e84b293b4c55b10ef03823745f695614639c5eb92def1b2d8ed75f43`，manifest SHA256=`e590933fa2feb46b89647a88fdd91577436b7851a3491fdd20c2d8c12c664c81`。


<!-- EXPERIMENT48_1_DGAA_R_DISEASE_CONDITIONAL_ALIGNMENT -->
## Experiment 48.1 — DGAA-R：疾病判别保持的多关系图对抗对齐

- **执行状态**：`COMPLETE_AUDITED`。80/80组折完成；E0为Experiment43.1已核验B0-B十折checkpoint独立读回，E1–E7共70折从同折B0-B锚点重新训练；独立读回80/80通过，最终审计`PASS`。未按中间测试结果减少任务。
- **目标与假设**：检验四路原始GFR的疾病辅助监督、普通关系来源对抗、ASD/TD条件关系对抗、CORR–FUNC选择性对齐、CORR优先辅助权重，以及真实关系/置乱条件负控制能否改善最终GFR→CAM判别。保留原GFR/CAM、PHENO覆盖行为、完整320维GCN输入、ALoss不反传、原weight_decay、历史`val_idx=test_idx`规则；没有新增Top-K患者图、PEN或PCGrad。
- **论文/源码审计**：已读取SPromptGL、DeepASD、AAAI 2026选择性对齐/PID及条件对抗材料；核对根`main.py/model.py/network.py/layers.py/utils.py`，并复核43.1/43.2/43.4/45/45.1/46/47.3及可定位的48历史记录。47.1/47.2目录缺失状态已记录，未将其伪称为源码复核证据。审计明细见`.runtime/source_history_audit.json`。
- **固定配置**：seed=0，ABIDE A 871例（ASD403/TD468），10折StratifiedKFold(shuffle=True, random_state=10)，450轮上限、wait>200；新增共享320→128→64投影、共享疾病头、三组来源头，新增lr=0.00038、`lambda_adv_max=0.03`及条件分组规则见`config.json`。

### 正式指标（十折 mean ± population SD，%）

|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E0 E0_B0B_REFERENCE|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|E1 E1_AUX_ONLY|91.6144 ± 3.1353|91.3972 ± 3.1155|94.2091 ± 4.2082|88.5854 ± 4.0433|
|E2 E2_UNCONDITIONAL_DGAA|91.3845 ± 3.0589|91.1347 ± 3.0951|94.4218 ± 3.8972|87.8476 ± 4.7679|
|E3 E3_DISEASE_CONDITIONAL_DGAA|91.7294 ± 2.8596|91.5697 ± 2.8546|93.5661 ± 4.4333|89.5732 ± 4.3041|
|E4 E4_SELECTIVE_CONDITIONAL_DGAA|91.6144 ± 2.8713|91.4398 ± 2.8778|93.7882 ± 3.2660|89.0915 ± 3.6830|
|E5 E5_CORR_PRIORITY_DGAA_R|91.6144 ± 2.8249|91.4586 ± 2.7355|93.3441 ± 4.9006|89.5732 ± 3.1266|
|E6 E6_SITE_STRATIFIED_RELATION_NULL|91.6144 ± 2.8713|91.4051 ± 2.8335|93.9870 ± 4.8354|88.8232 ± 4.0863|
|E7 E7_PERMUTED_DISEASE_CONDITION|91.8443 ± 2.4968|91.6628 ± 2.4560|93.9963 ± 3.9613|89.3293 ± 3.3427|

### 关键配对差（百分点；胜/平/负为十折严格比较）

|比较|ΔACC|胜/平/负|ΔHard-AUC|胜/平/负|ΔSPE|ΔSEN|
|---|---:|---|---:|---|---:|---:|
|E1-E0|-0.1149|1/7/2|-0.2074|2/2/6|1.0731|-1.4878|
|E2-E1|-0.2299|2/5/3|-0.2625|4/2/4|0.2128|-0.7378|
|E3-E2|0.3448|3/5/2|0.4350|4/3/3|-0.8557|1.7256|
|E3-E1|0.1149|3/5/2|0.1724|5/2/3|-0.6429|0.9878|
|E4-E3|-0.1149|1/6/3|-0.1298|2/4/4|0.2220|-0.4817|
|E4-E1|0.0000|2/6/2|0.0426|3/4/3|-0.4209|0.5061|
|E5-E4|0.0000|2/6/2|0.0188|4/3/3|-0.4440|0.4817|
|E5-E0|-0.1149|2/5/3|-0.1459|3/3/4|0.2081|-0.5000|
|E5-E6|0.0000|4/3/3|0.0535|4/3/3|-0.6429|0.7500|
|E5-E7|-0.2299|3/3/4|-0.2041|4/2/4|-0.6522|0.2439|

### 机制与归因
- E1相对E0：`GO_AUX=NO-GO`，ACC −0.1149pp、Hard-AUC −0.2074pp；辅助视角测试ACC约48.1/50.3/52.7/50.3%，没有转化为最终CAM增量。
- E2相对E1：普通无条件对抗继续下降（ACC −0.2299pp、Hard-AUC −0.2625pp）。E3相对E2为ACC +0.3448pp、Hard-AUC +0.4350pp，达到预注册的有限`GO_CONDITIONAL`，但只3/10与4/10折严格胜，不能据此声称稳健最终增益。
- E4相对E3：ACC −0.1149pp、Hard-AUC −0.1298pp；选择CORR–FUNC没有优于全关系条件对抗。E5相对E4：ACC 0.0000pp、Hard-AUC +0.0188pp，CORR优先辅助监督不成立。
- E5相对E0：ACC −0.1149pp、Hard-AUC −0.1459pp，SEN −0.5000pp；相对E6真实关系身份差异ACC 0.0000pp、Hard-AUC +0.0535pp，相对E7置乱条件ACC −0.2299pp、Hard-AUC −0.2041pp。真实关系和真实疾病条件没有提供预注册的稳定独立增量。
- 原始E0四路GCN输出均值约PHENO 241.69、ANAT 0.018、FUNC 0.006、CORR 0.007（相对输入约1.02、7.7e−5、2.7e−5、2.9e−5）；PHENO关闭干预为零，其他关系关闭的预测翻转率为零。E1–E7的辅助/来源梯度探针确实非零，但消息幅值和CAM依赖仍未形成可靠最终增量。
- 条件来源头在最佳checkpoint读回中多为接近0的来源准确率，显示来源判别训练存在反向/退化现象；对抗头接近随机或反向均不能解释为医学表征改善。
- 站点诊断120行：E2/E5/E6全关系平均同站点边比例44.11%/44.61%/44.94%，跨站点同诊断比例50.52%/50.55%/50.44%；站点/标签混合相关性只能作为混杂敏感性解释。

### GO/NO-GO
- `GO_AUX=NO-GO`；`GO_CONDITIONAL=GO（有限机制证据）`；`GO_SELECTIVE=NO-GO`；`GO_CORR_PRIORITY=NO-GO`；`GO_REAL_RELATION=NO-GO`；`GO_DISEASE_CONDITION=NO-GO`；`GO_FINAL_MICCAI_CANDIDATE=NO-GO`。
- 研究结论：DGAA-R在当前ABIDE、历史评估协议和单seed下不具备成为MICCAI 2027第一创新点的充分证据；不建议继续通过堆叠对抗权重/图模块推进48.2。可继续的方向应先修复无偏验证、关系消息可识别性和来源判别训练退化，并用独立数据/多seed验证。
- 该NO-GO只针对本实验协议，不等同于所有疾病条件关系对齐方法都无效；E3的有限相对改善值得作为机制线索保留，不能作为最终分类收益声明。

### 资源、路径与SHA
- GPU分配：GPU0负责fold 1,3,5与2,4两个不重叠worker；GPU1负责fold 6,8,10与7,9两个不重叠worker。E1–E7累计任务wall time约11099.4秒（并行运行），无额外seed。
- 实验目录：`/data/adodas/SPromptGL-main/experiment/experiment48_1_dgaa_r_disease_conditional_alignment`；报告：`results/report.md`；逐折：`results/fold_metrics.csv`；汇总：`results/summary.csv`；配对：`results/paired_comparisons.csv`；机制：`results/diagnostics.csv`；站点：`results/site_diagnostics.csv`；manifest：`manifest.json`；最终审计：`.runtime/final_audit.json`。
- 结果SHA：report=073051e36601cf81d2c979f0183e96d337e0c73f6c86e0e53f820b932715e593；fold_metrics=c7594e66e70f7fb4d4bd270f5633830087289b8030158fb8350d02b6a77d584c；summary=c4fb143c14a88f9187bd4aff882c15110107a8079ab2411748c55df84a373877；paired=53eb80defbbeb2f93fb4c2cd7339d24405047bb5ebcaa240e7927402fc1e4a45；diagnostics=e98961a6ad24fdd1f3e3481ccea39c326b4cae5f7f91d387292a6f48396dc2ca；site=94cf2596f1edee38a6723c438f7b0ab0bfea1e7be2ff9505da746dea8e97498a；manifest=d1662a0e2992b702c19076afd50af3a625613583d91f50013a395f644d732206；code_signature_final=249fc1c2b29b1d307df3f7811ef6a8c46994c26c56639eafc4393704461544d8。
- 训练过程中曾发现并修复了评估模式泄漏（评估后未恢复train导致dropout关闭）；修复后清理旧新增任务并从冻结锚点全量重跑，修复前结果未计入正式报告。
- 根源码、原始数据及历史锚点保护SHA前后不变；本条记录写入前缀锁定并验证唯一标记。


## Experiment 49 — B0-B Dual-Seed Robustness Study

- 状态：20/20组、200/200 folds完成；GPU0执行E01、E03、E06、E08、E09、E11、E14、E16、E17、E19，GPU1执行其余10组；固定组合内fold1–10连续消费单一训练RNG。
- B0-B协议：物理移除PEN，GFR+CAM、原始四图/两层GCN、PHENO覆盖、GraphConstructLoss、weight_decay=0.11、simple-2和val=test均保留；没有使用47.2/47.3的rho、中心化、固定图或关系错配。
- 结果文件：`experiment/experiment49_b0b_dual_seed_robustness/results/fold_metrics.csv`（200行）、`group_summary.csv`（20行）、`factor_summary.csv`、`oof_predictions.csv`和`report.md`。
- 历史参考：seed=0、cv_random_state=10 的 B0-B 为 ACC 91.7294%、Hard-AUC 91.6046%、SPE-TD 93.1360%、SEN-ASD 90.0732%；20组最高值只作描述性观察，不能作为无偏性能或方法增益。
- 审计：20组配置唯一；200/200完整；相同CV seed的split SHA一致、不同CV seed不同；GFR/CAM参数量887,836/49,544；PEN/CLIP文本路径不存在；weight_decay=0.11；OOF覆盖与checkpoint独立读回通过。
- 限制：val=test选模、外层前RFE/标准化、transductive全队列、单数据集和共享训练患者均保留，结果是内部稳健性描述，不是无偏独立泛化证据，也没有证明GFR相对CAM-only的独立增量。
- 详细报告：`experiment/experiment49_b0b_dual_seed_robustness/results/report.md`。
- 交付存储：200个checkpoint在`/tmp/SPromptGL-E49/experiment49_b0b_dual_seed_robustness/.runtime/checkpoints`已完成读回审计；因`/data` OSSFS逐文件同步阻塞，正式目录保留源码、结果、任务状态和轻量审计，未保留不完整checkpoint。


## Experiment 50 — C²R-Align

- 状态：E0历史参考读回、E1–E7新增70折训练和独立审计已完成；实际目录 `/tmp/SPromptGL-E50/experiment50_c2r_align`。
- 设计：B0-B物理删除PEN，原始GFR/CAM、weight_decay=0.11、simple-2、val=test保留；C²R以CORR/FC为锚点，加入条件门控、NULL、训练折内疾病关系监督、受限残差和E6/E7负控制。
- 结果：E5 ACC=0.8909±0.0322%，Hard-AUC=0.8883±0.0315%；E0 ACC=0.9173±0.0286%。完整ACC/Hard-AUC/Probability-AUC/SPE/SEN表和逐折配对见`results/report.md`。
- GPU：所有E1–E7的fold1–5使用GPU0，所有E1–E7的fold6–10使用GPU1；共70个新增折，固定每卡最多3个并发worker。
- 判定：以E5相对容量控制、FC-only、无条件对齐、无显式对齐、错配关系和重连关系的逐折结果及固定checkpoint干预共同决定GO/NO-GO；不把单项指标提升视为机制成立。
- 限制：历史val=test、CV前RFE/标准化、transductive协议和单seed保留；不构成无偏临床泛化或医学因果结论。
- 文件SHA：manifest=`6f200c2dd263709db500b84a692b0b45dec381833f84b6223786c64b05e5566f`，report=`496a0c46e3ccf4178493427eb5bd70531e80b7d5d32689c83ac267e745fb8fb5`，fold_metrics=`819e34274308af3e714127e8ce52c86fca546ec226d5ab52f528bd406a40f640`。
- 八组十折均值（ACC/Hard-AUC/SPE/SEN，%）：E0 91.7294/91.6046/93.1360/90.0732；E1 91.4995/91.2753/94.2091/88.3415；E2 88.2889/88.1141/90.3562/85.8720；E3 89.2085/89.1403/90.1526/88.1280；E4 89.8929/89.7242/91.6374/87.8110；E5 89.0935/88.8269/92.2757/85.3780；E6 89.7766/89.6190/91.6466/87.5915；E7 90.0065/89.9456/91.0315/88.8598。
- 机制：E5原始GFR消息范数均值约0.001012、GFR范数236.311，C²R残差/GFR约3.15%；NULL均值0.2382，门控未全拒绝。固定检查点关闭C²R平均使ACC下降2.4099pp，但重训E5低于E0 2.6358pp、E1 2.4060pp、E6 0.6831pp和E7 0.9130pp，未证明真实关系特异性收益。
- 最终：NO-GO；C²R残差确实进入决策，但关系对齐没有带来超过容量控制、基线或强负控制的独立疾病分类增益。
- 交付：正式目录 `/data/adodas/SPromptGL-main/experiment/experiment50_c2r_align` 已同步代码、配置、源码快照、10个锚点、fold缓存和全部结果；70个唯一best checkpoint约1.1GB保留在只读scratch `/tmp/SPromptGL-E50/experiment50_c2r_align/.runtime/checkpoints`，以避免向OSSFS复制冗余大文件，独立重载审计已在scratch完成。
- 工程记录：首轮调度发现GPU计数条件错误并立即停止，未将其结果计入；随后修正并固定每卡并发上限。另修正了best模型CAM键名兼容和训练/评估模式显式传参；70个正式任务均以修正后的可读回路径完成，失败状态未计入最终结果。

## Experiment 50.1 — C²R-Align Corrected & Validated

### 状态与执行
- 状态：COMPLETE；E0历史参考10折，E1–E8共80个新增折，合计90条正式折级记录；E8在审计后发现错误继承无显式对齐条件，删除其临时状态并从同一未训练锚点重跑十折，最终E8为修正结果。
- 运行路径：完整训练/最佳checkpoint/重载审计位于`/tmp/SPromptGL-E501/experiment50_1_c2r_corrected`；正式代码与结果位于`/data/adodas/SPromptGL-main/experiment/experiment50_1_c2r_corrected`。
- GPU：GPU0负责所有新增组fold1–5，GPU1负责fold6–10；控制器最终80/80完成、失败任务0、失败进程0；训练墙钟约559.35秒（并行控制器记录）。
- 12项语义测试在GPU0/GPU1均通过；B0-B参数量GFR=887,836、CAM=49,544。

### 修复与语义审计
- FC锚点固定为CORR/FC index=3，辅助模态固定为PHENO/ANAT/FUNC index=0/1/2；门控差分均相对FC。
- 对齐损失使用辅助关系对停止梯度的FC关系，辅助分支获得非零梯度，FC锚点未被错误detach。
- NULL只拒绝辅助关系；NULL=1时最终关系、邻居消息和残差路径与FC-only恒等，最大绝对误差0。
- E6只置乱辅助模态且保持FC分数不变；E7在重连邻居上重新计算四模态score、gate、relation、softmax权重和value；语义测试报告重连改变6968个候选并产生非零输出差异。
- 站点关系校正只使用训练折配对统计、停止梯度，E8仅关闭该校正；测试标签隔离、训练/评估模式切换、优化器weight_decay=0.11、主干初始化和checkpoint/RNG恢复均通过。

### 九组正式结果（十折 mean±population SD，百分比）
|组|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|---:|
|E0|91.7294±2.8596|91.6046±2.8059|93.2322±3.2881|93.1360±4.5354|90.0732±3.5117|
|E1|91.4995±2.6885|91.2620±2.6818|93.6590±2.4351|94.4265±3.3763|88.0976±3.4542|
|E2|88.1727±3.1766|88.1365±3.2106|91.4617±3.7948|88.6633±3.6128|87.6098±4.4067|
|E3|88.2863±3.0413|88.0070±3.0526|91.7270±3.7794|91.6420±4.5110|84.3720±5.2688|
|E4|88.8597±3.3812|88.6779±3.3132|91.8612±3.1319|91.0083±5.9671|86.3476±5.1620|
|E5|88.8623±3.7818|88.8090±3.7694|91.4583±4.1411|89.5143±4.1398|88.1037±3.9364|
|E6|88.3986±3.2401|88.3274±3.2885|91.2825±2.6113|89.5328±3.8522|87.1220±5.5142|
|E7|89.6656±2.1866|89.4354±2.3225|92.3104±2.7690|92.7428±2.7107|86.1280±5.6566|
|E8|89.2032±3.4265|89.2067±3.3113|92.5809±3.1517|89.0842±5.5557|89.3293±3.6650|

### E5核心逐折比较（均值差，百分点；胜/平/负）
|比较|ACC|Hard-AUC|Probability-AUC|SPE|SEN|
|---|---:|---:|---:|---:|---:|
|E5−E0|-2.8670 (0/1/9)|-2.7956 (1/0/9)|-1.7739 (1/0/9)|-3.6216|-1.9695|
|E5−E1|-2.6371 (1/1/8)|-2.4530 (1/1/8)|-2.2007 (1/0/9)|-4.9121|+0.0061|
|E5−E2|+0.6897 (4/4/2)|+0.6725 (5/2/3)|-0.0034 (4/0/6)|+0.8511|+0.4939|
|E5−E3|+0.5760 (6/2/2)|+0.8020 (7/1/2)|-0.2687 (3/0/7)|-2.1277|+3.7317|
|E5−E4|+0.0026 (3/2/5)|+0.1311 (5/0/5)|-0.4029 (6/0/4)|-1.4940|+1.7561|
|E5−E6|+0.4637 (5/2/3)|+0.4816 (6/1/3)|+0.1758 (6/0/4)|-0.0185|+0.9817|
|E5−E7|-0.8033 (2/3/5)|-0.6264 (2/3/5)|-0.8521 (4/0/6)|-3.2285|+1.9756|
|E5−E8|-0.3409 (3/1/6)|-0.3977 (3/0/7)|-1.1226 (3/0/7)|+0.4302|-1.2256|

### 机制结论
- E5相对B0-B：ACC -2.8670pp，Hard-AUC -2.7956pp；没有超过91.7294%历史基线。
- E5相对Self-only E1：ACC -2.6371pp；相对FC-only E2为+0.6897pp，但Probability-AUC几乎无差；这不足以证明辅助关系具有稳定增量。
- E5相对E3（固定均匀对齐）ACC +0.5760pp、SEN为+3.7317pp，但SPE下降；相对E4（无显式对齐）ACC仅+0.0026pp且Probability-AUC下降，显式对齐的独立作用未证实。
- E5相对E6（辅助错配）ACC +0.4637pp，属于有限局部优势；相对E7（患者关系重连）ACC -0.8033pp、Hard-AUC为-0.6264pp；相对E8（无站点校正）ACC -0.3409pp。真实关系特异性没有通过全部强对照。
- 固定E5 checkpoint中关闭C²R平均使ACC下降3.5619pp、重连使ACC下降5.0549pp，证明新增分支和关系路径进入了决策；但固定干预依赖不等于独立重训收益，且E7/E8独立重训更高。
- 原始GFR消息仍接近零（E5审计均值约0.001227，相对GFR范数约236.311），C²R残差相对GFR约3.18%；分支确实有界进入CAM，但没有转化为超过强基线的分类增量。

### 科研判定
- **最终：NO-GO（作为MICCAI 2027第一创新模块候选）**。工程错误已修复并通过语义审计，分支依赖可测；但E5低于B0-B、Self-only、关系重连E7和去站点校正E8，且未同时胜过所有预注册机制对照。
- 本结果不支持“FC锚定的选择性多模态关系对齐已产生可归因疾病判别增益”；不再基于同一方案追加超参数或新模块。
- 限制：历史`val=test`选模、CV前RFE/标准化、transductive全队列、单seed和单数据集保留；结果是内部协议比较，不是无偏临床泛化或医学因果证据。

### 文件与SHA
- 正式报告：`experiment/experiment50_1_c2r_corrected/results/report.md`；fold_metrics、summary、paired_comparisons、diagnostics、interventions均已同步。
- manifest SHA：`82d6b02e970018bf523eb7ae7fc16a0f9b84278436f0392aa57b86e41e8cb7a0`；report SHA：`af399044647501f52de30f036a11d5f02619bbd9529ce410a3e3599387afe4f1`；fold_metrics SHA：`295ce939d49670326a97de7d0d70bc32bc9b0667211fee14e9ad0eab22cdd73c`；summary SHA：`335161b9dcc256a3d50cf91cfb0314b1b5b7fbec335abf7bd8d8a1262919d811`；paired SHA：`155435f10e2d3c136e98ac187be3df7de9490898c756514018393ebb39b831eb`。
- 根目录main.py/model.py/network.py/layers.py/utils.py、Shell、原始数据SHA复核不变；正式目录未复制1.2GB checkpoint，完整最佳checkpoint仍在scratch且已独立重载审计。


<!-- EXPERIMENT48_2_CARE_GFR -->
## Experiment 48.2 — CARE-GFR

- **科学目标**：验证疾病关系证据学习（DREL）、冲突感知可信共识（CATA）和受限反事实关系攻击（ACRL）能否使多模态患者关系为B0-B/CAM提供可靠的ASD/TD增量；不把图依赖、对齐程度或辅助头性能当作疾病增量。
- **审计结论**：已读取并核对可缓存的SPromptGL/DeepASD/AAAI2026及可定位的43.2、43.3、43.4、47.3、48.1历史结果；RCMVL、UMCGL和ICML 2025图对抗论文PDF在服务器审计目录缺失，未声称完成原文逐式核验。确认原GFR完整320维传播、PHENO覆盖、ALoss未反传和GCN weight_decay消息收缩；修复只放入E1工程对照，没有计入CARE创新。缺失的47.1/47.2历史目录在`.runtime/source_history_audit.json`中如实记录。
- **方法与对照**：E0历史B0-B；E1 repaired GFR；E2 DREL；E3 DREL+CATA；E4 FULL CARE-GFR；E5去共识；E6特征GRL控制；E7站点/outer-split关系错配；E8置乱DREL对标签；E9同预算随机边扰动。所有E1–E9从匹配初始化独立训练，训练后仅用外层测试标签读回评估。

### 正式结果（mean±population SD，%）

|组|定义|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|
|---|---|---:|---:|---:|---:|---:|
| E0 | E0_B0B_REFERENCE | 91.7294±2.8596 | 91.6046±2.8059 | 93.1360±4.5354 | 90.0732±3.5117 | 93.2322±3.2881 |
| E1 | E1_REPAIRED_GFR | 89.0870±3.3514 | 89.0124±3.3025 | 90.1711±5.6452 | 87.8537±5.3440 | 91.4491±3.2009 |
| E2 | E2_DREL_ONLY | 60.7301±2.9966 | 59.9731±3.0140 | 70.0925±11.1047 | 49.8537±12.0436 | 57.4217±7.1387 |
| E3 | E3_DREL_CATA | 59.4684±2.6698 | 57.9241±2.9446 | 77.7567±15.4971 | 38.0915±17.9339 | 55.7787±4.6950 |
| E4 | E4_FULL_CARE_GFR | 58.7800±3.2205 | 58.0860±2.6682 | 66.8733±20.9786 | 49.2988±20.6846 | 57.6049±4.3274 |
| E5 | E5_NO_CONSENSUS_ALIGNMENT | 60.6165±3.3292 | 59.1669±4.2817 | 77.9741±11.1369 | 40.3598±18.5297 | 60.4353±8.2210 |
| E6 | E6_FEATURE_GRL_CONTROL | 59.8145±2.5419 | 59.2717±2.7238 | 66.0130±16.3693 | 52.5305±18.2614 | 57.7942±6.5370 |
| E7 | E7_WRONG_RELATION_CONTROL | 60.0431±2.2133 | 59.3621±2.6452 | 67.7058±16.3965 | 51.0183±19.2206 | 57.2845±7.1236 |
| E8 | E8_SHUFFLED_PAIR_LABEL_CONTROL | 59.1262±2.4388 | 58.4163±2.1912 | 67.0583±24.5644 | 49.7744±25.9154 | 56.2219±4.1558 |
| E9 | E9_RANDOM_PERTURB_CONTROL | 58.4483±3.5245 | 57.0560±3.5604 | 74.9352±15.9066 | 39.1768±17.2910 | 52.9498±6.5538 |

- E0逐折历史checkpoint读回与参考一致：ACC 91.7294±2.8596%、Hard-AUC 91.6046±2.8059%。E4 ACC 58.7800±3.2205%，因此初筛为`SCREEN_ACC_FAIL`。
- 关键配对：E4-E0: ΔACC=-32.9493pp (0/0/10)；E4-E1: ΔACC=-30.3069pp (0/0/10)；E4-E3: ΔACC=-0.6883pp (4/2/4)；E4-E5: ΔACC=-1.8365pp (2/3/5)；E4-E6: ΔACC=-1.0345pp (3/3/4)；E4-E7: ΔACC=-1.2631pp (1/3/6)；E4-E8: ΔACC=-0.3461pp (5/1/4)；E4-E9: ΔACC=0.3318pp (1/8/1)。E4−E9仅+0.3318pp ACC（1/8/1），同时SPE−8.0620pp、SEN+10.1220pp；不能视为稳定ACRL收益。

### 机制和负控制

- DREL患者对AUC接近随机（E2四关系平均约0.5007），E2相对E1 ACC/Hard-AUC分别−28.3568/−29.0393pp；没有可识别的疾病关系证据。
- CATA不成立：E3−E2 ACC−1.2618pp，E4−E5 ACC−1.8365pp；一致性损失未带来独立收益，部分折损失尺度发散。
- 正确关系不成立：E4−E7 ACC−1.2631pp（1/3/6），错配关系反而更高；E4−E8 ACC−0.3461pp，真实DREL标签没有独立贡献。
- 图依赖可观测但无益：E4关闭跨患者消息后平均ACC约56.9410%，完整图约58.7800%，平均预测翻转约4.14%；这只说明分支改变了CAM输出，不证明关系的疾病有效性。
- 站点诊断中E4四关系同标签率约0.50，跨站点同标签率也约0.50；不能把站点统计相关性解释成ASD生物学证据。

### 判定、资源与交付

- `GO_DREL=NO-GO`，`GO_CATA=NO-GO`，`GO_ACRL=NO-GO`，`GO_VS_GRL=NO-GO`，`GO_REAL_RELATION=NO-GO`，`GO_DISEASE_RELATION_LABEL=NO-GO`。仅标记`GRAPH_DEPENDENCE_DETECTED`，不等价于疾病增量。
- **最终科研判定：NO-GO**，CARE-GFR没有超过91.73%，正确患者关系没有产生独立可重复的疾病诊断增量，反事实关系对抗没有证明优于普通控制或随机扰动；不继续以本配置启动后续方法。
- 完成100/100组折：E0历史复用核验10折，E1–E9实际训练90折；GPU0负责fold1–5，GPU1负责fold6–10；累计训练wall time 13007.1秒。向量化了逐候选GPU同步瓶颈后，清空并统一重跑E1–E9，90折代码签名一致。
- 工程独立审计：`AUDIT_PASS`，预检PASS，任务/预测SHA完整，根源码、数据和Shell SHA与独立历史manifest一致。`protected_sha256_match=true`。
- 产物：`/data/adodas/SPromptGL-main/experiment/experiment48_2_care_gfr`；报告 `results/report.md`；逐折 `results/fold_metrics.csv`；汇总 `results/summary.csv`；配对 `results/paired_comparisons.csv`；机制 `results/diagnostics.csv`；站点 `results/site_diagnostics.csv`；清单 `manifest.json`；审计 `.runtime/final_audit.json`。
- SHA与签名：训练代码签名 `4891127b596f1a65ebda893e502478e087372cdfe8152bb494e57f11541b3826`；当前manifest/report/fold_metrics/summary/paired SHA见实验目录`manifest.json`，根handoff最终SHA由`.runtime/handoff_append.json`记录。
- 主要限制：CV前全局RFE/标准化、`val=test`选择checkpoint、transductive 871节点、单seed和单数据集；这些结果是历史协议下的内部比较，不是无偏外部泛化或因果证据。

## Experiment 50.2 — PCE-Router

### 执行状态
- 状态：COMPLETE；E0历史参考10折，E1–E8有效新增80折，失败任务0、失败进程0。单GPU0，最多5个并行worker。
- E1首次实现与E2共享无条件mask路径，审计发现后只将E1修正为直接患者自身FC残差，从同一B0-B锚点重跑E1十折；旧E1结果不计入最终表。E2–E8未重复训练。
- 本轮未修改根模型源码、原始数据或历史目录；B0-B/GFR/CAM/OutputLayer、原始weight_decay=0.11、simple-2、val=test和CV前RFE/标准化均保留。
- 论文/源码审计覆盖SPromptGL、DeepASD、To Align or Not to Align、跨注意力患者图工作、FC亚型工作及43–50.1历史失败证据；PCE不引入患者图、Prompt、CLIP、GPT或反事实教师。

### PCE-Router定义
- PCE插在原始GFR输出与CAM之间，只读取GFR后FC slice 64:320；前64维逐值保留。ASD/TD证据头输出有符号256维证据，差异生成FC mask；条件输入为PHENO年龄26、性别2、ANAT6、FUNC10，共44维，不含站点；残差上限5%，alpha初始化为0。
- E6在训练/测试分组内按站点分层置乱条件；E7按患者固定置乱256维证据位置；E8只关闭证据辅助CE。证据选择与删除均不读取测试标签。

### 九组正式结果（十折 mean±population SD，百分比）
|组|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|---:|
|E0|91.7294±2.8596|91.6046±2.8059|93.2322±3.2881|93.1360±4.5354|90.0732±3.5117|
|E1|91.4995±2.8321|91.2992±2.8557|93.3541±2.9303|94.0009±3.1775|88.5976±3.8515|
|E2|91.8443±2.9793|91.6556±2.9761|93.5043±2.8565|94.2137±3.3535|89.0976±3.4981|
|E3|91.7294±3.0819|91.5495±3.0288|93.0865±3.2021|93.7697±4.8044|89.3293±3.7140|
|E4|91.4995±2.9239|91.2699±2.9368|93.4727±3.0608|94.2044±4.3335|88.3354±4.6178|
|E5|91.6144±2.8249|91.4245±2.7437|93.5316±2.4608|93.7697±5.0870|89.0793±3.7283|
|E6|91.6144±3.0498|91.4268±2.9618|93.5857±2.9371|93.7743±4.5946|89.0793±2.5536|
|E7|91.7294±3.0819|91.5122±3.0441|93.5460±3.2955|94.1952±5.1451|88.8293±4.5003|
|E8|91.4995±2.9688|91.3125±2.9477|93.2977±2.9360|93.7835±4.0325|88.8415±3.7139|

### E5配对结果（差值百分点；胜/平/负）
|比较|ACC|Hard-AUC|Probability-AUC|SPE|SEN|
|---|---:|---:|---:|---:|---:|
|E5−E0|-0.1149 (2/6/2)|-0.1801 (2/4/4)|+0.2994 (5/0/5)|+0.6337|-0.9939|
|E5−E1|+0.1149 (2/7/1)|+0.1252 (3/3/4)|+0.1774 (5/0/5)|-0.2313|+0.4817|
|E5−E2|-0.2299 (1/6/3)|-0.2312 (1/5/4)|+0.0272 (5/0/5)|-0.4440|-0.0183|
|E5−E3|-0.1149 (1/7/2)|-0.1250 (2/5/3)|+0.4451 (8/0/2)|+0.0000|-0.2500|
|E5−E4|+0.1149 (3/5/2)|+0.1546 (3/4/3)|+0.0588 (3/0/7)|-0.4348|+0.7439|
|E5−E6|-0.0000 (3/3/4)|-0.0023 (3/2/5)|-0.0542 (4/0/6)|-0.0046|-0.0000|
|E5−E7|-0.1149 (2/5/3)|-0.0878 (3/3/4)|-0.0144 (5/0/5)|-0.4255|+0.2500|
|E5−E8|+0.1149 (3/5/2)|+0.1120 (4/2/4)|+0.2339 (4/0/6)|-0.0139|+0.2378|

### 机制与证据诊断
- E5证据头训练/测试诊断准确率均值=49.33%/48.67%；mask熵=5.5438，接近256维均匀熵5.5452。竞争证据没有形成可重复的疾病区分。
- E5 alpha均值约-1.938e-06，残差/GFR约1.926e-07；固定checkpoint关闭PCE、alpha=0、均匀mask、错误条件和错误证据位置均未改变测试预测。
- 固定GFR输出Top-16/随机/幅值匹配遮蔽ACC差值均值=-4.8237/-6.1978/-5.0522pp；Top-16没有比随机负控造成更大损害。原始FC重算小样本平均概率变化=0.2476/0.2194/0.1985。
- ROI映射使用Experiment11已核验的256维表；本轮Top-16映射均有可验证端点时才记录名称，未知端点不补写脑区解释。

### 科研判定
- E5 ACC=91.6144%，低于B0-B=91.7294%、E2=91.8443%和E7=91.7294%；E6错误条件与E5 ACC相同。E5只在E1/E4/E8等局部对照上有小幅优势，未满足完整机制门槛。
- **最终：NO-GO。** PCE-Router在当前历史训练协议下近似恒等映射，竞争证据接近均匀且不能区分疾病；没有形成超过容量、竞争、条件、错误位置和B0-B的可归因分类增量。
- 结论只适用于本轮PCE定义、单seed、ABIDE和历史val=test协议；不把失败解释为所有患者自身FC证据方法均无效。

### 文件与SHA
- 正式代码/结果目录：`experiment/experiment50_2_pce_router`；完整训练checkpoint、任务状态和审计运行位于`/tmp/SPromptGL-E502/experiment50_2_pce_router/.runtime`，正式交付可排除`.runtime`。
- manifest=15cbbb02f21a63cc5c0535a659b436312961e33d8554a64eee89f4523c94532e；report=4ef668059b361de94e6efe14481742c324c2080e993e3028b5e96bdee3974a46；fold_metrics=87eac9ebcec0ed642b0ea57c8725e101b13bde9ff2e0a06035dce9884a54a3e6；summary=8143b38dbf076551c9eab8618c0eda3f1e7c983cd87c6a1e0ec7775908c44d16；paired=11a668deb36e0044c57d5085d61973ba387ab9f10c4d3747fea3b66ce884ce68；diagnostics=fd52681a60ec7a65e4653686d0568afb609423d5e3961c1fe6c18edbf6bd9c4f；evidence=f39c8703bfea67389229dcd9224bb9d71a2a45211e556bd94fe0b95b542412e8；ROI mapping=b383308bb413a94d1646511a9bea5bc16f54c975b938ac4d947d676e357c5eb2。


<!-- EXPERIMENT48_3_APCR_GFR -->
## Experiment 48.3 — APCR-GFR

- **研究问题**：在保留B0-B GFR→CAM主路径的前提下，检验对称可信疾病关系（SRD）、选择性冲突感知对齐（SCA）和反事实关系扰动（CRA）能否提供真实跨患者诊断增量。48.2的整体CareGraph替换架构已弃用。
- **源码审计**：确认原GFR四路传播完整320维、PHENO覆盖、ALoss不反传、weight_decay=0.11消息收缩；48.2的错误normalize、非对称评分、偏置配对、无界一致性和clean/adv路径问题已记录。指定RCMVL/UMCGL/ICML图对抗论文未在服务器本地缓存，未声称完成逐式核验。
- **方法与组别**：E0历史B0-B；E1 anchor parity；E2 feature-only容量控制；E3真实关系残差；E4 SRD；E5 SRD+SCA；E6 FULL APCR；E7错误关系；E8置乱SRD标签；E9随机攻击。

### 正式结果（mean±population SD，%）

|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|
|---|---:|---:|---:|---:|---:|
|E0|91.7294±2.8596|91.6046±2.8059|93.1360±4.5354|90.0732±3.5117|不可由历史硬预测重建|
|E1|91.8443±2.7963|91.5914±2.7725|94.8474±3.7844|88.3354±3.3561|93.6388±2.7133|
|E2|91.8443±2.8431|91.6202±2.8507|94.4172±4.0374|88.8232±4.2222|93.4621±3.0641|
|E3|91.6144±2.9170|91.4166±2.8667|93.9917±4.0988|88.8415±3.1500|93.8019±2.6112|
|E4|91.7294±3.0388|91.4827±3.0363|94.6300±4.0358|88.3354±4.0328|93.8746±2.8611|
|E5|91.3845±3.1017|91.2589±3.1280|92.9325±3.2230|89.5854±4.0849|93.7804±2.9114|
|E6|91.9592±2.9141|91.7273±2.8980|94.6253±4.2741|88.8293±3.9209|93.7393±2.8417|
|E7|91.3845±3.1858|91.1829±3.1006|93.7743±4.7048|88.5915±2.7434|93.4079±2.8803|
|E8|92.0729±3.3148|91.7723±3.3378|95.7031±3.4971|87.8415±4.2256|93.2062±3.2145|
|E9|91.3845±3.2677|91.2069±3.2546|93.5661±4.1059|88.8476±4.1361|93.4101±2.6886|

- **初筛**：E6=91.9592%，严格超过91.73%。E6−E0为+0.2299pp ACC，但SEN下降1.2439pp。
- **关键差异**：E3−E2=−0.2299pp；E4−E3=+0.1149pp；E5−E4=−0.3448pp；E6−E5=+0.5747pp；E6−E7=+0.5747pp；E6−E8=−0.1136pp；E6−E9=+0.5747pp。

### 工程与科研判定

- 100/100组折完成，E1–E9共90折实际训练，统一源码签名，独立读回100/100。
- 数学预检通过，归一化和SRD对称性正确，SCA初始尺度有界。
- `ENGINEERING_FULL_PASS=NO-GO`：CRA使用两个固定候选替换路径，没有完成完整候选集合上的分类损失严格内层最大化；E1也没有完成逐元素前向、loss、梯度和第一步更新的独立parity证明；实际alpha约0.002而非声明的0.01；消息关闭干预尚未独立重算。
- `SCREEN_ACC_PASS`，但`GO_REAL_MESSAGE=NO-GO`、`GO_SCA=NO-GO`、`GO_TRUE_PAIR_LABEL=NO-GO`；CRA和真实关系仅部分支持，不能称为严格机制验证。
- **最终科研判定：NO-GO**。APCR-GFR超过指标门槛，但当前证据不足以支撑MICCAI 2027第一创新点；主要限制是关系增量弱、SCA无收益、CRA内层优化不完整和锚点parity未完成。

### 资源与产物

- 单GPU、最多5个逻辑worker；100/100任务完成。
- 产物目录：`/data/adodas/SPromptGL-main/experiment/experiment48_3_apcr_gfr`；报告、逐折指标、汇总、配对、诊断、manifest和`.runtime/final_audit.json`均已保存。
- 历史`val=test`、CV前RFE/全局预处理、transductive图和单seed限制仍然存在。

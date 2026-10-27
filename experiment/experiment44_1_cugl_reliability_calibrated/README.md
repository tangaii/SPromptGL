# Experiment 44.1 — Reliability-Calibrated CU-GL Repair

本目录是 Experiment 44 的一次独立修正与最终验证。所有新代码、缓存、审计和结果均位于本目录；`.runtime`只保存恢复状态、候选图、教师缓存和必要的校验文件。

## 论文与源码审计证据

SPromptGL 论文的 GFR 将每种模态定义为一类患者关系，并用余弦相似度、图正则和图卷积传播（Eq.(1)–(2)）；CAM/PLN随后处理融合后的患者表示。公开源码中的 `GraphLearn`、两层 `GCN`、`GFR`、`simple-2` 和实际 CAM 接口与论文图示并不完全等价，且原实现的 PHENO 聚合在 `i==1` 分支被覆盖。Experiment43 已验证简单恢复该分支没有带来独立增益，Experiment44 又显示其边效用均值约 `5.76e-6`。

Experiment44 的确定性错误是 `DirectedRelation.weights()`用 `[..., :dim]`切片，因而 ANAT/FUNC/CORR评分没有读取真实的 `[48:54]`、`[54:64]`、`[64:320]`字段；本轮所有前向、候选审计、教师、学生和反事实代码共用 `slice_for_modality()`，并用无关/对应模态扰动单元测试核验。第二个问题是原始效用监督未经尺度校准，微小响应会使评分器收缩；本轮用训练内噪声下限、MAD/分位尺度、可靠性权重和接收者留出排序验证处理。

UAI 2025 的选择性阻断、ICLR 2025 的连续边方向和 NeurIPS 2025 的边编辑影响函数分别说明阻断、方向和边干预组件本身不是新颖性；图反事实综述也强调反事实图学习并不自动等于临床因果。本轮只把“标签隔离的冻结模型边/关系干预效用”作为候选监督，并以随机效用、同质性、无效用、对称方向和无 NULL 对照证伪，绝不把 `U` 写成患者间疾病因果。

## 预注册执行规则

先核验 B0-B、CAM-only 和 Experiment44 历史结果，再对 outer-fold 1、6 做三折交叉拟合教师（固定60 epoch）。inner-heldout 标签只在教师冻结后选择真实类 log-prob；outer-test 标签不进入任何教师、候选、尺度或阈值。边级效用为

`U_edge(i,j,m) = log p(y_i | 保留 j→i) − log p(y_i | 删除 j→i)`，

关系级效用为

`U_gate(i,m) = log p(y_i | 完整消息) − log p(y_i | 关闭模态 m 消息)`。

在训练内 OOF 效用上估计数值噪声、MAD/分位尺度，使用 `clip(U/s,-3,3)` 与可靠性权重；只在可靠边上施加接收者内排序。学生评分在未参与训练的接收者上与随机评分、余弦相似度和同质性关系比较，按接收者聚合，不能把16条边当作16个独立患者。

NULL采用独立接受门 `g(i,m)`，邻居质量为 `g*softmax(score)`，NULL质量为 `1-g`。关闭关系时邻居残差严格为零。所有E3-E10共享候选池、CAM锚点、模型初始化、随机流、优化预算和checkpoint规则；E0-E2只作为历史参照，不与新替换GFR混淆。

若两个预检折的未见接收者效用排序不同时优于随机和余弦，或实际边/关系干预无法超过 no-op 数值噪声，则停止 E3-E10，写入 `PREFLIGHT_SCIENTIFIC_NO_GO`，不做事后调阈值或继续实验44.2。

协议仍沿用历史 `val=test` checkpoint 选择、全队列预处理/RFE 和 transductive 受试者图，因此任何结果都只是内部可比性证据，不是无偏独立泛化或临床因果证据。

## 逐项源码证据表

|审计对象|私有实现位置|本轮证据|结论|
|---|---|---|---|
|原论文GFR Eq.(1)–(2)、CAM接口|`README.md`、`.runtime/sources/network.py`、`.runtime/sources/layers.py`|论文与源码逐函数对照|论文叙述与公开实现不完全相同|
|原始PHENO聚合覆盖|`experiment43_1.../results/report.md`、`experiment43_3.../results/report.md`|`i==1`覆盖路径及修复后无增益|确定历史实现问题，非本轮主要创新|
|Experiment44错误切片|`cugl.py`旧实现与本轮`cugl_r.py:ModalityRelation`|旧版`[..., :dim]`；新版显式`lo,hi`|确定代码错误，已修复|
|候选图标准化|`integration.py:fit_block_normalizer/candidate_pool`|均值/尺度只在outer/inner训练索引拟合|通过标签隔离审计|
|边级反事实|`teacher.py:compute_interventions`|冻结教师、真实边删除、局部两层重算|可计算，但未见接收者泛化排序|
|关系级反事实|`teacher.py:compute_interventions`|冻结教师、整模态消息关闭|可计算，信号同样极小|
|效用校准|`run.py:robust_scale/calibration`|MAD/分位数、FP32 no-op噪声下限、clip和可靠性权重|规则已冻结，未改变失败判断|
|独立NULL门|`cugl_r.py:ModalityRelation.weights`|`g*softmax`与`1-g`，门关闭时残差精确为零|工程测试通过|
|完整恢复|`run.py:rng_capture/rng_restore/recovery_test`|Python、NumPy、Torch CPU/CUDA、双优化器和参数逐值复演|通过，最大差0|
|未见接收者效用验证|`run.py:train_student_eval/receiver_eval`|按接收者分组，真实/随机/余弦比较|两个预检折真实监督均低于随机|

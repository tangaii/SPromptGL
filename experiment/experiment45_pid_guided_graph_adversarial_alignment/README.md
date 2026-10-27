# Experiment 45 — PID-DGAA

本实验检验四模态患者图中疾病相关冗余、独有和协同信息是否能决定 GFR 关系对的疾病条件对抗强度。唯一主基座是实验43.2已验收的物理移除 PEN 的 B0-B（GFR+CAM）。E0 只复用该历史基线；E1–E6 从同一折的未训练 B0-B anchor 起步。

## 审计结论

根 `network.py:153–274` 定义 GraphLearn、两层 GCN 和 GFR。GFR 每个分支传播完整 320 维拼接特征，`i==1` 覆盖 `i==0`，之后 FUNC/CORR 分支累加；四张图保留，阈值为 0.9。`model.py:122–151` 的 `simple-2` 实际优化加权分类损失与四图 `GraphConstructLoss` 平均值，`ALoss` 只计算不反传。`layers.py` 的 CAM/OutputLayer 保持不变。PHENO48 是年龄 bin、站点、性别编码；ANAT6/FUNC10 是质量与头动摘要；CORR256 是连接特征。图节点是患者，不是 ROI。

论文与代码的关系依据：SPromptGL 论文描述多关系患者图、CAM/PEN 和标签邻接思想；公开 shell 使用 `lr=.0038, reg=.11, dropout=.35, n_hidden=18, n_head=2, nlayer=1`。DeepASD 给出多模态图对抗背景；AAAI 2026 的对齐分析强调冗余/独有/协同结构决定对齐收益；`pliang279/PID` 的 `synthetic/rus.py` 是本实验的上游 PID 参考。运行环境没有 `cvxpy`，因此使用 I_min 的等价离散两源分解，所有差异、合成恒等式和误差写入结果。

## PID实现与边界

每个外层训练折单独标准化、KMeans 离散化并使用训练标签。主粒度 K=4，K=3/5敏感性；12次训练内 bootstrap 和19次训练标签置乱仅用于可靠性。I_min 分解满足 `I(X1,X2;Y)=R+U1+U2+S`，且 `I(X1;Y)=R+U1`、`I(X2;Y)=R+U2`。`q` 和 `lambda` 在进入正式训练前冻结，不读取外层测试标签或测试性能。

## 组别

E0 为历史 B0-B；E1 只有共享 64 维图投影和训练标签辅助分类；E2 加无条件来源对抗；E3 加疾病条件但统一强度；E4 使用 PID 决定 ANAT–FUNC、ANAT–CORR、FUNC–CORR 三个关系对强度；E5 循环置换三个强度归属；E6 保留图分支辅助训练但 CAM 主分类直接读取原始 X。新增模块只使用 outer-train 标签，新增参数独立零 weight-decay 优化器。

## 可复现命令

`python -B run.py prepare` → `python -B run.py preflight` → `python -B run.py pid`。只有 manifest 中 PID 状态为可靠时才运行 `python -B run.py train`，随后 `python -B run.py audit report handoff`。正式训练保留历史 `val=test`、全局预处理/RFE 和 transductive 图协议，结果是内部比较，不是无偏泛化或临床因果证据。

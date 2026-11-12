# Experiment 78.2: DRAPS 结果报告

状态：`COMPLETE_AUDITED`。正式目标为4组×10折；完成任务 40/40，独立Checkpoint审计 40/40。公共RNG共有epoch轨迹共核对14,307条，逐值匹配14,307条，不匹配0。首次审计器在机制汇总阶段触发CUDA张量转NumPy错误；恢复审计器 `audit_recovery.py` 只修复该读取错误，未改变冻结训练源码、Checkpoint或预测。

## 论文、代码与基座审计

SPromptGL 与 BrainPrompt 本地全文已审计；HP²L 仅访问 DOI/ScienceDirect 摘要和网页预览，未将不可访问全文公式当作已读内容。GitHub 的 tangaii/SPromptGL、wanxixi11/SPromptGL 与 AngusMonroe/BrainPrompt 已检查；参考提交 d3e78fb 是 E78.1 添加提交。详细范围见 `literature/source_review.md`。
唯一训练基座是 `E74.4-G2-Positive-DCR`。每折从 SHA 核验的 E61 epoch-10 warmup 恢复完整 DARG/CAM 参数、三个原优化器和 RNG；E74.4 的 `PLNNoPEN` CAM144、历史损失与更新顺序保留。训练源 SHA-256：`afb561aec7b798b189195b17226f22a489a3033c0a4c7a354895a2f1854085ab`。

## 固定方法

DRAPS 使用 h∈R144→q∈R16、z∈R32 两个投影；训练支持内采用站点逆平方根权重、软责任度、精确 LOO、ESS=(Σmass)^2/(Σmass²+eps)、连续支持/站点可靠性和全局类别中心回退。疾病原型损失为 0.05·CE([S_ASD,S_TD]/0.25,y)，Prompt 语义损失为 0.02·CE([e_ASD,e_TD],y)，站点约束为 0.005·L_site。跨疾病参照按患者对对立类别全部候选原型的可靠路由聚合，不使用相同编号差分。共享 16→144 残差由 0–0.25 的六统计量线性患者门控控制；原 OutputLayer 每次前向只调用一次。
EMA 锚点按每 epoch 一次的 normalize(0.9·a+0.1·stopgrad(p)) 更新，保存于独立 Checkpoint；四组均使用相同规则、同一新增参数数量与训练超参数。

## 六指标结果

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E78.2-G1-SPromptGL-K1|91.8443 ± 2.8431|91.6123 ± 2.9122|93.3081 ± 2.9262|88.5854 ± 4.7601|94.6392 ± 3.3856|90.9299 ± 3.1708|
|E78.2-G2-SPromptGL-K3|91.4995 ± 3.0994|91.3520 ± 3.0968|93.6223 ± 2.9577|89.3415 ± 4.2955|93.3626 ± 4.2372|90.6892 ± 3.3452|
|E78.2-G3-SPromptGL-K5|91.8443 ± 2.7486|91.6490 ± 2.7280|93.7594 ± 2.5402|89.3110 ± 4.3699|93.9870 ± 4.8354|91.0329 ± 2.9346|
|E78.2-G4-Anonymous-K5|91.5008 ± 2.7843|91.3730 ± 2.7626|93.6110 ± 2.6477|89.5915 ± 3.4414|93.1545 ± 3.6839|90.7198 ± 2.9904|

## 历史参考与 E78.1-G2 对照

|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|E78.1-G1|92.1891 ± 2.8121|92.0480 ± 2.7367|93.8609 ± 2.6635|90.3171 ± 2.8515|93.7789 ± 4.5987|91.4991 ± 2.8997|
|E78.1-G2|91.6144 ± 2.9619|91.4531 ± 2.9713|93.5111 ± 3.3327|89.3354 ± 4.4613|93.5708 ± 4.3208|90.7977 ± 3.2084|

G3−E78.1-G2 的 ACC 差为 +0.2299 pp，W/T/L 4/4/2（排除 Fold 2：+0.1277 pp）；六指标逐折差值已追加到 `paired_comparisons.csv`。

平均每折训练耗时（秒）：E78.2-G1-SPromptGL-K1 82.76; E78.2-G2-SPromptGL-K3 90.42; E78.2-G3-SPromptGL-K5 99.04; E78.2-G4-Anonymous-K5 92.93

## 配对比较

W/T/L 与折2排除结果见 `paired_comparisons.csv`。
- E78.2-G2-SPromptGL-K3 - E78.2-G1-SPromptGL-K1：ACC -0.3448 pp，W/T/L 0/7/3；Prob-AUC 与其余指标见 CSV。
- E78.2-G3-SPromptGL-K5 - E78.2-G2-SPromptGL-K3：ACC 0.3448 pp，W/T/L 4/5/1；Prob-AUC 与其余指标见 CSV。
- E78.2-G3-SPromptGL-K5 - E78.2-G4-Anonymous-K5：ACC 0.3435 pp，W/T/L 4/5/1；Prob-AUC 与其余指标见 CSV。
- E78.2-G3-SPromptGL-K5 - E74.4-G2：ACC -0.3448 pp，W/T/L 0/7/3；Prob-AUC 与其余指标见 CSV。
- E78.2-G3-SPromptGL-K5 - E78.1-G1：ACC -0.3448 pp，W/T/L 0/7/3；Prob-AUC 与其余指标见 CSV。
- E78.2-G3-SPromptGL-K5 - E78.1-G2：ACC +0.2299 pp，W/T/L 4/4/2；Prob-AUC 与其余指标见 CSV。

## 机制问题

每折 `mechanism.csv` 记录候选/有效 K、N_eff、可靠性、站点覆盖、回退、原型分类正确率、ASD/TD 原型分数、路由、Prompt TV、门控分布、残差比例和训练 LOO 损失。
G3的可靠性规则在十折均保留5/5名义有效原型（ASD/TD），且最低N_eff仍高于或接近预设参考范围；但不加分离损失后原型几何明显回缩：平均余弦相似度 0.9313、cos>0.98原型对比例 84.0%。因此“有效K=5”只表示支持可靠性通过，不能解释为5个互异疾病子结构。
- E78.2-G1-SPromptGL-K1：有效K ASD 1.00、TD 1.00；训练原型分类 97.05%；门控均值 0.0448。
- E78.2-G2-SPromptGL-K3：有效K ASD 3.00、TD 3.00；训练原型分类 96.40%；门控均值 0.0461。
- E78.2-G3-SPromptGL-K5：有效K ASD 5.00、TD 5.00；训练原型分类 96.93%；门控均值 0.0562。
- E78.2-G4-Anonymous-K5：有效K ASD 5.00、TD 5.00；训练原型分类 94.80%；门控均值 0.0507。

## 固定 Checkpoint 干预

仅在 G3 最佳 Checkpoint 上执行 module_off、reliability_off、global_center、anonymous、prompt_class_swap、feature_residual_off；逐折翻转和概率变化见 `interventions.csv`。
- `anonymous`：ACC 91.8443%；平均概率绝对变化 0.001282；硬翻转 0。
- `feature_residual_off`：ACC 91.4995%；平均概率绝对变化 0.007596；硬翻转 3。
- `global_center`：ACC 91.3845%；平均概率绝对变化 0.006951；硬翻转 4。
- `module_off`：ACC 91.4995%；平均概率绝对变化 0.007596；硬翻转 3。
- `prompt_class_swap`：ACC 91.7294%；平均概率绝对变化 0.002500；硬翻转 1。
- `reliability_off`：ACC 91.8443%；平均概率绝对变化 0.000160；硬翻转 0。

## H1–H5 与 GO/NO-GO

- H1 疾病判别原型：`TRAINING_SUPPORTED_ONLY`；G3训练 LOO 原型准确率较高，但完整方法没有带来相对唯一基座的独立测试收益。
- H2 K=5 有效容量：`NOT_SUPPORTED_AS_DISTINCT_CAPACITY`；可靠性规则给出名义有效K=5，但G3平均原型余弦相似度约 0.9313，cos>0.98的原型对比例约 84.0%，提示候选容量再次明显坍塌。
- H3 无编号跨疾病竞争：`SUPPORTED`；包括索引置换和非对称 K 预检。
- H4 患者门控：`SUPPORTED`；门控范围固定为 0–0.25。
- H5 医学 Prompt 独立贡献：`WEAK_NOT_ESTABLISHED`；G3相对G4的独立训练差异较小，固定Checkpoint匿名替换无硬标签翻转，不能据此宣称稳定的独立诊断收益。
- 完整方法相对 E74.4-G2：`NOT_SUPPORTED`。

最终科研判定：`NO-GO`。

## 协议限制

结果保留历史 `val=test`、测试 ACC 选模、CV 前预处理/RFE、transductive 全患者图以及 Fold 2 历史重叠；这些协议限制泛化评估，不能当作无偏临床性能估计。没有根据测试结果调 K 或超参数，也不自动启动 E78.3。

## 文件

- `preflight.json`：10/10 base-off parity、全链梯度、文本几何、两步恢复。
- `unit_tests.json`：ESS、LOO、索引置换、非对称 K、回退、有限性。
- `fold_metrics.csv`（含TP/TN/FP/FN、best epoch和耗时）、`paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`、`checkpoint_audit.csv`、`audit_recovery.json`。

训练源 SHA-256：`afb561aec7b798b189195b17226f22a489a3033c0a4c7a354895a2f1854085ab`。

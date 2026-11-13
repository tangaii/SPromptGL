# Experiment 78.6 — PEN-SVA 原始SPromptGL疾病Prompt有效性全面审计

审计完成时间（UTC）：2026-11-13T07:42:13.743986+00:00

## 结论先行

**状态：`COMPLETE_AUDITED`。** 10/10 个 E60 `E60_ORIGINAL_PEN` 最佳 Checkpoint 已通过 SHA、身份、A0 指标/概率读回和 A0–A7 冻结重放。原始 Prompt 确实进入计算图，且文本替换会造成很小的概率变化；但没有出现硬分类翻转，也没有证据显示医学文本优于匿名、错配或数值对照。原始 SPGS 的类别最大值算子严格丢弃相似度方向。因此最终判定为：

- `COMPUTATION_ACTIVE`；
- `PREDICTION_DEPENDENCE_DETECTED_WEAK`；
- `EXPLICIT_CLASS_DIRECTION_NOT_PRESERVED_BY_SPGS`；
- `MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`；
- `PEN_NET_HARD_GAIN_NOT_DETECTED`。

这一区分了 Prompt 参与计算、改变预测概率和提供医学语义增益三个层次。

## 目录、来源与 Checkpoint 身份

首选目录因 OSSFS errno=5 不稳定而回退；唯一正式目录是 `/model/spromptgl/experiment/experiment78_6_pen_semantic_audit`。根目录 `network.py`、`model.py`、`layers.py`、`utils.py` 未修改，保护 SHA 记录在 `manifest.json`。源码审计确认 `GFR → CAM → SPGS → RES → OutputLayer`，8 条原始文本、冻结 RN50、4096→128→9 MLP、CAM 4×4×9 结构、class-axis softmax/max、AvgPool/RES 和 OutputLayer 均实际存在。

E32/E28历史目录没有可直接恢复的原始十折 best state_dict；E60 `checkpoint_audit.csv` 有 10/10 PASS，且当前 E60 `/data` runtime 中十个 best 文件均 SHA 匹配。它们被复制到 `/tmp/e78_6_e60_cp_run1` 作为只读稳定输入；没有进行参考重建，也没有使用 E78.x 权重。E60 是 E53.3 CADR/GFR 协议，不与 E32 B2 混写。每折最佳 epoch 为 123、159、153、207、213、154、134、190、122、215。

## 原始 SPGS 的严格数学结论

对两类余弦相似度 `s_ASD,s_TD`，原始代码先计算 `softmax([s_ASD,s_TD])`，然后在类别维取最大值。恒等式为：

`max(softmax([s_A,s_T])) = sigmoid(|s_A-s_T|)`。

由于余弦在 [-1,1]，权重理论范围为 [0.5, 0.8807970779]。独立数值单测 10,000 组的最大误差为 0。将已经由文本 MLP 产生的 ASD/TD 两个 9 维类别向量交换后，10/10 折的 SPGS 权重、PEN 特征、logits 和硬预测均逐值保持不变，最大误差均为 0。也就是说，该门控保留匹配强度，丢弃 `s_ASD-s_TD` 的符号；这不等价于声称整个 OutputLayer 没有类别判别能力。

## 固定 Checkpoint 的 A0–A7 结果

以下为 10 折均值 ± population SD，单位为 %；ASD=0，TD=1，Probability-AUC 使用 ASD 概率，Hard-AUC/BA 使用项目历史定义。

|干预|名称|ACC|Hard-AUC / BA|Probability-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---|---:|---:|---:|---:|---:|---:|
|A0|FULL_PEN|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A1|RES_ONLY|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1544 ± 3.2921|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A2|PEN_OFF|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1703 ± 3.2786|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A3|CLASS_VECTOR_SWAP|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A4|ANONYMOUS_TEXT|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A5|WRONG_SEMANTIC_TEXT|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A6|NUMERIC_TEXT_CONTROL|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|
|A7|NEUTRAL_SPGS|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|

|干预|名称|TP-ASD|FN-ASD|FP-TD|TN-TD|硬翻转|正确→错误|错误→正确|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|A0|FULL_PEN|359|44|32|436|0|0|0|
|A1|RES_ONLY|359|44|32|436|0|0|0|
|A2|PEN_OFF|359|44|32|436|0|0|0|
|A3|CLASS_VECTOR_SWAP|359|44|32|436|0|0|0|
|A4|ANONYMOUS_TEXT|359|44|32|436|0|0|0|
|A5|WRONG_SEMANTIC_TEXT|359|44|32|436|0|0|0|
|A6|NUMERIC_TEXT_CONTROL|359|44|32|436|0|0|0|
|A7|NEUTRAL_SPGS|359|44|32|436|0|0|0|

A0–A7 的总体混淆计数一致为 359 TP、44 FN、32 FP、436 TN；所有干预的硬预测均与 A0 相同。

|干预|非零概率患者(>1e-8)/871|平均|P95|最大|硬翻转|正确→错误|错误→正确|
|---|---:|---:|---:|---:|---:|---:|---:|
|A0 FULL_PEN|0|0.00000000|0.00000000|0.00000000|0|0|0|
|A1 RES_ONLY|871|0.00074737|0.00261056|0.01160836|0|0|0|
|A2 PEN_OFF|871|0.00028639|0.00101411|0.00380176|0|0|0|
|A3 CLASS_VECTOR_SWAP|0|0.00000000|0.00000000|0.00000000|0|0|0|
|A4 ANONYMOUS_TEXT|782|0.00004727|0.00022387|0.00145909|0|0|0|
|A5 WRONG_SEMANTIC_TEXT|773|0.00003977|0.00019125|0.00125852|0|0|0|
|A6 NUMERIC_TEXT_CONTROL|869|0.00005034|0.00023082|0.00154299|0|0|0|
|A7 NEUTRAL_SPGS|783|0.00004988|0.00023091|0.00154313|0|0|0|

A1/A2 的小概率变化来自 RES/PEN 特征路径；A4–A6 的文本替换只造成约 4–5×10^-5 的平均 ASD 概率变化，且无硬翻转。A3 的零变化是类别交换不变性的真实复现，不是探针失效。

|干预相对A0|ACC W/T/L|Hard-AUC W/T/L|Prob-AUC W/T/L|ΔProb-AUC均值(pp)|
|---|---:|---:|---:|---:|
|A1 RES_ONLY|0/10/0|0/10/0|1/5/4|-0.026528|
|A2 PEN_OFF|0/10/0|0/10/0|0/8/2|-0.010621|
|A3 CLASS_VECTOR_SWAP|0/10/0|0/10/0|0/10/0|+0.000000|
|A4 ANONYMOUS_TEXT|0/10/0|0/10/0|0/10/0|+0.000000|
|A5 WRONG_SEMANTIC_TEXT|0/10/0|0/10/0|0/10/0|+0.000000|
|A6 NUMERIC_TEXT_CONTROL|0/10/0|0/10/0|0/10/0|+0.000000|
|A7 NEUTRAL_SPGS|0/10/0|0/10/0|0/10/0|+0.000000|

完整逐折和逐患者记录在 `results/interventions.csv` 与 `results/patient_changes.csv`。

## 中间特征、文本 MLP 与梯度

|量|10折均值 ± population SD|
|---|---:|
|SPGS weight mean|0.5175 ± 0.0225|
|SPGS weight std|0.0122 ± 0.0158|
|signed margin mean|0.0014 ± 0.0253|
|absolute margin mean|0.0704 ± 0.0908|
|PEN相对残差范数|0.0586 ± 0.0030|
|PEN输入/输出余弦|1.0000 ± 0.0000|
|RES Vsim熵|2.1956 ± 0.0009|
|文本两类向量余弦|0.9004 ± 0.1698|
|文本MLP参数范数|0.1179 ± 0.0961|
|文本MLP训练标签 NLL 梯度范数|0.0103 ± 0.0170|
|CAM全参数训练标签 NLL 梯度范数|2.9189 ± 1.3398|

SPGS weight 约为 0.517，RES Vsim 熵约 2.196，接近 9 个通道的均匀熵 `ln(9)=2.1972`；因此原始 RES 的通道选择也非常平缓。PEN 的相对残差约 5.86%，但输入/输出余弦约 0.99998，说明它改变了潜在特征，却没有形成强分类边界移动。文本 MLP 参数在十折均非零；用合法训练标签只做梯度诊断时，9/10 折文本 MLP 梯度非零，10/10 折完整 CAM 参数梯度非零。冻结 CLIP 没有更新。此前将 CAM 首参数误当作路径参数的审计器错误已修正并重新完成 10 折重放。

确定性正向控制中，微扰 OutputLayer 参数或 PEN 中间特征均产生非零 logits 变化；A0 手工 PEN 细节与原模块逐值一致，排除了 Hook 或干预位置完全失效的解释。

## 历史交叉核查

|历史来源/配置|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E32 B0 CAM-only|92.0742 ± 2.6000|91.8546 ± 2.5815|未在保留摘要中提供|89.0793 ± 3.7283|94.6300 ± 4.0457|未在保留摘要中提供|
|E32 B1 RES-only|91.2696 ± 3.1421|91.0897（历史摘要）|未提供|88.8354（历史摘要）|93.3441（历史摘要）|未提供|
|E32 B2 full PEN|91.7294 ± 2.9505|91.5625（历史摘要）|未提供|89.3415（历史摘要）|93.7835（历史摘要）|未提供|
|E28 E0_P0（3-seed不同协议）|91.6911 ± 0.0542|91.5394 ± 0.0841|93.3352 ± 0.4154|89.6606 ± 0.6456|93.4181 ± 0.5102|90.9329 ± 0.1204|
|E43 E00_B0A_REPLAY（GFR审计协议）|92.0742 ± 2.6000|91.8546 ± 2.5815|未提供|89.0793 ± 3.7283|94.6300 ± 4.0457|未提供|
|E60 ORIGINAL_PEN（本次固定审计，E53.3协议）|91.2696 ± 3.2251|91.1160 ± 3.2072|93.1809 ± 3.2680|89.0915 ± 3.6830|93.1406 ± 3.9660|90.4495 ± 3.4753|

E32 的独立训练排序是 B0 CAM-only > B2 full PEN > B1 RES-only（ACC），支持继续检查 PEN 是否真的提供增量，但它不是当前 E60 固定 Checkpoint 的因果对照。E28 的原始粗 Prompt E0_P0 与其 E1_NO_SPGS/E4 等文本对照没有形成一致的医学语义优势；其三 seed 协议也不同。E43 的 E00_B0A 是 GFR/CORR 审计协议，不能与 E32/E60 作为同一训练轨迹合并；其结果同样没有替代当前 PEN 的固定 Checkpoint 证据。

## 科研判定

1. **Prompt 是否参与计算：是。** 源码路径、中间 Tensor、文本 MLP 参数和训练标签梯度支持 `COMPUTATION_ACTIVE`。
2. **Prompt 是否影响预测：弱。** A4–A6 在 871 个测试患者上多数产生非零概率变化，但平均仅约 4–5×10^-5，A4/A5/A6 均 0 硬翻转；因此是 `PREDICTION_DEPENDENCE_DETECTED_WEAK`。
3. **SPGS 是否保留疾病方向：否。** 类别向量交换 10/10 折严格不变，结论为 `EXPLICIT_CLASS_DIRECTION_NOT_PRESERVED_BY_SPGS`。
4. **医学文本是否具有独立诊断价值：未建立。** 医学、匿名、错误和数值文本都没有造成硬分类差异，且固定模型文本替换仅有微小概率变化；这支持 `MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`。它不能被解释为所有文本信息都无效，因为匹配强度仍可改变概率。
5. **PEN 是否有净硬增益：没有。** A0、A1、A2 的 ACC、Hard-AUC、SEN、SPE、F1 完全相同；完整 PEN 相对 PEN-off 只提高 Probability-AUC 0.0106 pp，未纠正或引入任何硬预测。A0 相对 RES-only 的 Probability-AUC 差为 +0.0265 pp，仍无硬增益。固定 Checkpoint 结果回答的是路径依赖，不替代从头重训消融。

因此本轮没有把原始疾病 Prompt 机制判为医学语义有效，也没有启动后续实验。最终科研标签：**`COMPUTATION_ACTIVE_BUT_SEMANTIC_VALUE_NOT_ESTABLISHED`**，并附 **`EXPLICIT_CLASS_DIRECTION_NOT_PRESERVED_BY_SPGS`**、**`PREDICTION_DEPENDENCE_DETECTED_WEAK`** 和 **`PEN_NET_HARD_GAIN_NOT_DETECTED`**。

## 协议限制与文件

历史 `val=test`、测试 ACC 参与选模、CV 前预处理/RFE、transductive 全患者图和单一历史协议限制无偏泛化解释。固定 Checkpoint 文本替换不是训练级医学/匿名对照；E60 与 E32/E28/E43 的训练协议不同。未使用测试标签选择干预、随机文本或结果。

核心文件：

- 审计实现：`/model/spromptgl/experiment/experiment78_6_pen_semantic_audit/audit.py`、`pen_probe.py`、`real_replay.py`
- 设计与身份：`design.md`、`manifest.json`、`checkpoint_inventory.json`
- 结果：`results/report.md`、`fold_metrics.csv`、`interventions.csv`、`feature_effects.csv`、`semantic_direction.csv`、`patient_changes.csv`、`audit_status.json`、`real_replay.json`
- handoff：`/model/spromptgl/experiment/handoff.md`

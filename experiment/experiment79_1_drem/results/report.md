# Experiment 79.1 — DREM动态错误恢复记忆学习

- **STATUS**：`COMPLETE_AUDITED`；**科研判定**：`NO_GO_DREM`。
- 正式组：G1 Memory-Dependent Error Learning、G2 Recovery-Aware Error Memory；seed=0，10折，20/20训练折完成。
- 运行目录：`/model/spromptgl/experiment/experiment79_1_drem`；单GPU worker峰值10；任务记录最大耗时约4.58 h；记录的单worker峰值显存约1.59 GiB；10 worker并发实测总占用约24.6 GiB。
- 独立审计：40/40 best（20 Method-Selected + 20 Base-Anchored）与10/10 latest PASS；`audit_status.json`为PASS。共享基座RNG三元组与历史任务共3917个Epoch逐值匹配，差异0（`rng_parity.json`）。

## 基座与审查

- E74.4-G2的DARG、PLNNoPEN、CAM144及OutputLayer保持只读；没有引入Prompt、PEN、OOF教师或新主干梯度。SAM 2只作为历史记忆写入/读取和Cross-Attention抽象参考，未移植视频跟踪结构。
- E79失败原因在设计中被直接处理：错误事件分区为Stable/Recent/Recovered，恢复方向采用真实训练恢复轨迹的类别中心漂移校正，不再使用E79的局部分类器梯度方向；Memory Value与Key绑定，增量方向只能来自记忆读出。
- 历史协议限制仍存在：val=test选模、CV前预处理、transductive图和Fold 2历史重叠，结果不能作为无偏外部泛化估计。

## 六指标（%，mean±population SD；列序ACC / Hard-AUC/BA / Prob-AUC / SEN-ASD / SPE-TD / ASD-F1）

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G1 Base-Anchored | 91.4995±2.9239 | 91.3655±2.8543 | 93.5162±2.8046 | 89.8171±3.6329 | 92.9140±5.0564 | 90.7672±3.0305 |
| G2 Base-Anchored | 91.4995±2.9239 | 91.3655±2.8543 | 93.4950±2.8213 | 89.8171±3.6329 | 92.9140±5.0564 | 90.7672±3.0305 |
| G1 Method-Selected | 92.0742±2.8427 | 91.9416±2.7896 | 93.5160±2.7284 | 90.3171±3.4469 | 93.5661±4.6502 | 91.3739±2.9678 |
| G2 Method-Selected | 92.0742±2.8427 | 91.9416±2.7896 | 93.5480±2.7804 | 90.3171±3.4469 | 93.5661±4.6502 | 91.3739±2.9678 |

配对差（均值pp；W/T/L；描述性95%区间）:
- Base-Anchored G1−基座 ACC：-0.6897 pp; W/T/L=0/6/4; 95% CI [-1.3426, -0.0367] pp; 排除Fold2 -0.7663 pp；Prob-AUC：+0.0216 pp; W/T/L=3/1/6; 95% CI [-0.0716, +0.1148] pp; 排除Fold2 +0.0413 pp
- Base-Anchored G2−G1 ACC：+0.0000 pp; W/T/L=0/10/0; 95% CI [+0.0000, +0.0000] pp; 排除Fold2 +0.0000 pp；Prob-AUC：-0.0212 pp; W/T/L=0/6/4; 95% CI [-0.0374, -0.0051] pp; 排除Fold2 -0.0236 pp
- Base-Anchored G2−基座 ACC：-0.6897 pp; W/T/L=0/6/4; 95% CI [-1.3426, -0.0367] pp; 排除Fold2 -0.7663 pp
- Method-Selected G1−基座 ACC：-0.1149 pp; W/T/L=0/9/1; 95% CI [-0.3287, +0.0988] pp; 排除Fold2 -0.1277 pp；G2−G1 ACC：+0.0000 pp; W/T/L=0/10/0; 95% CI [+0.0000, +0.0000] pp; 排除Fold2 +0.0000 pp；G2−基座 ACC：-0.1149 pp; W/T/L=0/9/1; 95% CI [-0.3287, +0.0988] pp; 排除Fold2 -0.1277 pp
- 完整六指标逐折差、排除Fold2结果见`paired_comparisons.csv`。

## 记忆、恢复与纠错

- 全部正式训练到各折结束的共享记忆轨迹累计写入/更新/移动/淘汰：166,200/234,041/331,017/298,584；恢复候选/确认/拒绝/失效：41,267/24,427/16,840/22,852。各Ledger在不同最佳Epoch的实际读取库另列于`memory_analysis.csv`。
- 有效Recovered记录（Base-Anchored G1去重）：253；同头margin gain均值0.5183，方向范数按定义约1。最终库没有被伪恢复事件填充，当前折容量和分区见`memory_analysis.csv`。
- 记忆事件只来自合法外层训练患者；查询自排除、测试患者不入库、Fold2 zero-based 376/406排除均通过审计。

- Base-Anchored G1：错误→正确 0，正确→错误 6，净纠错 -6（ASD纠正0，TD纠正0）；排除Fold2净纠错-3。
- Base-Anchored G2：错误→正确 0，正确→错误 6，净纠错 -6（ASD纠正0，TD纠正0）；排除Fold2净纠错-3。
- Method-Selected G1：错误→正确 6，正确→错误 0，净纠错 6（ASD纠正3，TD纠正3）；排除Fold2净纠错5。
- Method-Selected G2：错误→正确 7，正确→错误 0，净纠错 7（ASD纠正4，TD纠正3）；排除Fold2净纠错6。

## 同Checkpoint机制干预

平均ACC/净纠错/硬翻转（按10折；完整逐折见`intervention_analysis.csv`）：
- Base-Anchored G1：Full 91.4995%/-6/6；Memory-Off 92.1891%/+0/0；Key-Value-Association-Break 91.4995%/-6/6；Error-Type-Swap 91.3845%/-7/7；Recency-Off 91.4995%/-6/6；Recovery-Direction-Permuted 91.4995%/-6/6
- Base-Anchored G2：Full 91.4995%/-6/6；Memory-Off 92.1891%/+0/0；Key-Value-Association-Break 91.7294%/-4/4；Error-Type-Swap 91.2696%/-8/8；Recency-Off 91.4995%/-6/6；Recovery-Direction-Off 91.4995%/-6/6；Recovery-Direction-Permuted 91.4995%/-6/6
- Method-Selected G1：Full 92.0742%/+6/6；Memory-Off 91.3845%/+0/0；Key-Value-Association-Break 91.7294%/+3/7；Error-Type-Swap 91.8443%/+4/8；Recency-Off 91.9592%/+5/7；Recovery-Direction-Permuted 92.0742%/+6/6
- Method-Selected G2：Full 92.0742%/+7/7；Memory-Off 91.2696%/+0/0；Key-Value-Association-Break 91.8443%/+5/7；Error-Type-Swap 91.8443%/+5/9；Recency-Off 91.9592%/+6/8；Recovery-Direction-Off 92.0742%/+7/7；Recovery-Direction-Permuted 92.0742%/+7/7

## 门控、恢复方向与结论

- Base-Anchored机制均值：门控、相对增量范数、触及10%上限比例按组逐折见`mechanism.csv`；总体不再是E79式几乎全体饱和，但部分折仍有较高cap fraction。
- G2相对G1在正式硬分类ACC上没有任何折增益（两个Ledger的G2−G1 ACC均为0/10/0）；Method-Selected Prob-AUC仅有小幅+0.0319pp，未形成硬纠错证据。
- Key–Value Association Break在Base-Anchored G1几乎不改变硬结果，G2仅有限改变；Recovery-Direction-Off/Permuted不改变G2硬结果，说明真实恢复方向尚未显示可迁移独立价值。
- **科研判定：`NO_GO_DREM`。** 在严格Base-Anchored比较中，两组均低于E74.4-G2且产生净负硬变化；Method-Selected G1/G2接近但未超过基座，G2未优于G1。历史错误→正确恢复过程能够建立真实记忆事件，但没有提供稳定、可迁移的独立疾病信息。
- 由于训练期间为修复初始恢复字段异常并将等价Memory读出改为批处理而从最新状态恢复，训练Checkpoint保留了初始source inventory SHA；当前审计源SHA另记录在`audit_status.json`。这些改动未改变基座参数/RNG和重放预测，但该工程版本历史应在复现实验时一并保留。

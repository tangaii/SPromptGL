
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

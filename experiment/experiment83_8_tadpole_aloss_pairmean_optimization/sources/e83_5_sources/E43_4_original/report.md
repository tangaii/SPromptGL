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

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
4. **已有方法边界**：UAI 2025选择性阻断（https://proceedings.mlr.press/v286/choi25a.html）、ICLR 2025连续边方向（https://proceedings.iclr.cc/paper_files/paper/2025/hash/7422317f84e8c83e4c1ad2ff87e1e88e-Abstract-Conference.html）、NeurIPS 2025边编辑影响（https://proceedings.neurips.cc/paper_files/paper/2025/hash/f2ba7d0d1394609d5ce11853cfd1c9bf-Abstract-Conference.html）和图反事实综述（https://doi.org/10.1007/s11633-024-1519-z）分别覆盖阻断、方向和边干预；CU-GL的候选区别是标签隔离的个体级预测效用与四模态患者图及NULL动作的统一消融验证（不预称世界首次）。
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

产物目录：`/data/adodas/SPromptGL-main/experiment/experiment44_counterfactual_utility_graph_learning`；总墙钟（截至报告）：4721.4s。

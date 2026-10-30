# Experiment 53 — DRAP疾病判别驱动的对抗式患者图重构与消息传播

创建时间：2026-10-30T12:46:02.848775Z
执行目录：`/model/spromptgl/experiment/experiment53_drap_disease_relation_propagation`
数据/根源码只读来源：`/data/adodas/SPromptGL-main`

## 路径和保护审计

- `/model/spromptgl/experiment` 与 `/data/adodas/SPromptGL-main` 不是同一挂载点；`/data/adodas/SPromptGL-main` 提供ABIDE数据、43.1 anchors和根源码。
- 本目录是唯一实验53写入目录；根源码、数据、实验43–52和历史checkpoint不修改。
- 根 handoff 仅在最终完成后安全追加。

## 论文和源码审计

- SPromptGL MICCAI 2025：GraphLearn使用模态线性投影、归一化余弦相似度和阈值图；GraphConv先参数变换再邻接传播；历史simple-2公开实现实际训练分类损失和GraphConstructLoss。官方论文/源码：
  - https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf
  - https://papers.miccai.org/miccai-2025/0861-Paper5129.html
  - https://github.com/wanxixi11/SPromptGL
- DeepASD先用模态对抗正则编码器做特征分布对齐，再学习患者图和GNN；本实验只借鉴对抗思想，关系对表示和传播算子重新设计。来源：https://www.nature.com/articles/s41398-024-02972-2

## 历史失败边界

- 实验52的关系评分接近0.5，且原始GFR的强weight decay使GCN消息接近零；实验53新增关系参数使用AdamW矩阵1e-4、偏置/LayerNorm 0衰减，并用独立有符号传播绕过原始GCN消息瓶颈。
- 实验47.3真实图没有稳定优于站点分层强负控制；实验48对抗关系尝试和实验50/51关系/路由模块没有形成稳定可归因诊断增量。因此实验53必须训练E7错误关系和E8普通邻域控制。
- 所有关系可靠性、校准、阈值和传播预算只由outer-train内3折inner patient-disjoint OOF证据决定；不读取outer-test标签或ACC。

## 冻结方法

- E0为实验51 E1严格重放；E1–E8均从相同未训练GFR/CAM anchor和post-construction RNG构造。
- 原始GFR/CAM Adam保持lr=0.0038、weight_decay=0.11；DRAP新增关系/判别器AdamW lr=0.001，权重矩阵1e-4，偏置和LayerNorm 0。
- 关系编码器为每模态 `Linear(d,32)-GELU-LayerNorm`；患者对特征为绝对差、逐元素乘积、cosine；评分器为65→32→1。
- inner OOF固定3折、最多100 epoch、固定fold-local RNG；最终关系模型在outer-train pair上拟合，使用OOF可靠度和Logistic校准。
- 可信正边要求校准q≥0.55，可信负边要求q≤0.45；每名患者每种边最多12个；不确定边跳过。E6取消硬跳过并使用连续软权重。
- 正/负传播分别行归一化：`M_pos=sum A_pos_norm(X_j-X_i)`，`M_neg=sum A_neg_norm(X_j-X_i)`，`Delta=M_pos-0.25*M_neg`，每模态RMS-tanh有界，`H=H_E1+0.02*Delta`。
- E7保持E4关系损失但用固定fold级关系身份置乱传播；E8使用原候选图普通邻居和固定正/负预算，不用疾病关系决定邻居身份。
- 训练协议：ABIDE 871，48/6/10/256，10折StratifiedKFold(random_state=10)，seed0，450 epoch，wait>200，历史val=test，CV前预处理/RFE和transductive图保留，最多5 worker GPU0。

## 预注册9组

|组|传播|对抗|不确定边|关系身份|
|---|---|---|---|---|
|E0|无新增传播|无|无|E51 E1重放|
|E1|可信正边|无|跳过|真实疾病关系|
|E2|正边+负边|无|跳过|真实疾病关系|
|E3|可信正边|条件|跳过|真实疾病关系|
|E4|正边+负边|条件|跳过|真实疾病关系|
|E5|正边+负边|无条件|跳过|真实疾病关系|
|E6|正边+负边|条件|连续软权重|真实疾病关系|
|E7|正边+负边|条件|跳过|固定置乱关系|
|E8|普通邻域正/负预算|条件训练但传播不用关系|跳过|普通候选邻域|

## GO/NO-GO

- 关系判别GO：多数折至少一个可靠模态具有inner OOF AUC≥0.55且正负方向正确，并报告校准/覆盖。
- 有效传播GO：关闭传播、拆分正/负传播和关系置乱产生可测量输出/分类变化，且变化不是单独新增MLP残差。
- 诊断增量GO：E4同时超过E0、E7和E8的预注册配对门槛；单个AUC、非零梯度或判别器随机准确率不构成GO。

## 来源SHA（创建前快照）

- `/data/adodas/SPromptGL-main/main.py` SHA256 `dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2`
- `/data/adodas/SPromptGL-main/model.py` SHA256 `254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda`
- `/data/adodas/SPromptGL-main/network.py` SHA256 `484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec`
- `/data/adodas/SPromptGL-main/layers.py` SHA256 `4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a`
- `/data/adodas/SPromptGL-main/utils.py` SHA256 `95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67`
- `/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv` SHA256 `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`
- `/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy` SHA256 `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/config.json` SHA256 `0267c00adf69aef713216585e2466be4f3d9476f2f379f70aba59abc4f9572ca`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/run.py` SHA256 `2063f55147470fc4c335ab04044078824ed71fbb0dcaccbfb3b42da0072b7355`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/drap.py` SHA256 `e1207e5bec4a32724c72f49a1165ca6c8b32ec1237bce66601a750b2b0cec6c9`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/audit.py` SHA256 `551c2b93030d691f9cc358c94e3bed7205e46fae756294a79a52fad578d1ed1d`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/manifest.json` SHA256 `e91df6a814587544fa7cb232f971d4ae611cae397022c5f8d5cc9abf980f9209`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/results/report.md` SHA256 `55c2eb9afbe958bb78a4e12df09d2795e65a0510b0a3a69efb531b4e2cef0693`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/results/diagnostics.csv` SHA256 `ea2b3f475312c5dd0b8c26f37371870631b33a56939dba9618602192e86a7b57`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/results/fold_metrics.csv` SHA256 `82ed3719c067d53e9a713ea3af31cda64eba5f89b4d544fba5d3f9e2b67c1de9`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/manifest_wd_sweep.json` SHA256 `1ea9bb4c5e65aaef2edd4e3c76493f535f076cbcddefada8585d762815479a16`
- `/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/results/wd_sweep/report.md` SHA256 `ec2509584fcf4c227ade1e9f7e3f8a4e8595144cf6d2cfc9030b397bc935c840`

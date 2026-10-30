# Experiment 51.4 — D³-SAG正式报告

## 审计与设计冻结

本轮以Experiment51 E1为直接基座；新增分支只接收CORR与去站点PHENO。站点字段通过`modal_feat_dict.npy`名称精确排除，ANAT/FUNC只形成统计可靠性代理。疾病有效性由outer-train内部五折OOF风险增量、200次bootstrap单侧90%下界和站点残差CKA零分布冻结。

公开论文边界：SPromptGL定义患者关系图与GFR/CAM（[论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)，[代码](https://github.com/wanxixi11/SPromptGL)）；MGDR同时保留公共和私有模态表示（[论文](https://papers.miccai.org/miccai-2024/514-Paper2494.html)）；SMGD将Shared/Complementary/Incidental/Nuisance组织为诊断核心与非稳健信息，但其图是患者内部ROI图（[论文](https://proceedings.mlr.press/v306/shi26ag.html)）；AAAI 2026指出对齐收益依赖模态冗余（[论文](https://ojs.aaai.org/index.php/AAAI/article/view/39248)）。

所有组均使用seed=0、10折StratifiedKFold(random_state=10)、历史val=test和CV前处理；这些结果不代表无偏外部测试性能。未发现可核实的Experiment52（EXPERIMENT52_NOT_FOUND）。

## 正式结果（ACC / Hard-AUC / Probability-AUC / SPE / SEN，% mean±population SD）

|组|设计|ACC|Hard-AUC|Probability-AUC|SPE|SEN|
|---|---|---:|---:|---:|---:|---:|
|E0|B0-B_REFERENCE|91.7294±2.8596|91.6046±2.8059|93.2322±3.2881|93.1360±4.5354|90.0732±3.5117|
|E1|E1_AGGREGATION_REPLAY|91.9592±2.8684|91.7434±2.8950|93.1856±3.0236|94.6392±3.3856|88.8476±4.1651|
|E2|ORDINARY_DUAL_VIEW_CAPACITY|91.4995±2.7372|91.3336±2.6854|93.4091±3.1181|93.3441±5.1703|89.3232±4.3445|
|E3|SMGD_STYLE_FIXED_ALIGNMENT|91.2696±2.8788|91.1209±2.8461|93.4147±3.3192|92.9186±4.8583|89.3232±4.1982|
|E4|DISEASE_ROUTING_NO_ALIGNMENT|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E5|D3SAG_FULL|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E6|UNIFORM_ALIGNMENT_BUDGET|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E7|NO_COMPLEMENTARY_ROUTE|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E8|NO_NONDIAGNOSTIC_CONSTRAINT|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E9|WRONG_GEOMETRY_CORRESPONDENCE|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E10|WRONG_RELIABILITY_GATE|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E11|NO_QUALITY_CONDITION|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|
|E12|WRONG_AUXILIARY_PHENO|91.3845±3.0589|91.2140±3.0158|93.4452±3.0425|93.3488±4.9804|89.0793±4.2013|

## 机制证据

十折疾病有效性估计中，r_S非零折数=0/10，r_C非零折数=0/10；因此完整E5按预注册规则拒绝新增S/C诊断路由，不能把E5/E6等与E1相同的结果包装成共享对齐成功。ΔF平均=0.2071，ΔP平均=-0.0305，ΔC平均=-0.0941。

E2的普通双视图容量对照为91.4995% ACC；固定强对齐E3为91.2696%；E4–E12的新增路由因冻结的r_S/r_C均为0而在前向上严格写回零增量，因而各组收敛到同一套“E1结构、从E1锚点重新训练”的轨迹。它们不是历史E1最佳checkpoint的逐折复制（历史E1为91.9592%），因此不能把相同结果包装成主方法成功。该结果支持“当前训练折没有足够可复核的CORR–PHENO共享疾病证据”，不支持D³-SAG的主假设。

E0/E1与所有新增checkpoint均完成独立读回，`checkpoint_audit.csv`共110行、最大重算误差0。工程测试通过：站点字段排除、零路由恒等、固定关系块、训练标签隔离、初始化锚点和有限梯度。

## 科研判断

**NO-GO**。E5（完整D³-SAG）ACC和Hard-AUC低于E1，且疾病有效性路由在十折全部为零；E3固定对齐也低于E1，E2容量对照同样未超过E1。因而本轮没有证明疾病可靠性驱动的共享关系对齐、互补路由或质量门控可产生独立增量；应停止在同一患者关系选择性对齐路线继续堆叠51.5。

## 交付限制

训练在`/tmp/SPromptGL-E51_4/experiment51_4_d3sag`完成；`.runtime`含checkpoint且不纳入紧凑交付。正式`/data`目录同步及handoff追加需单独核验OSSFS写入状态。

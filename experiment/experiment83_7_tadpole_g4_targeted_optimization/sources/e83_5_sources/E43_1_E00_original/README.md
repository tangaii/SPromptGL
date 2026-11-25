# Experiment 43.1 — B0-B 基座上的 GFR 五组优化验证

本轮只研究GFR，不调用GPT、CLIP或新预训练模型。以真正物理移除PEN的B0-B为唯一正式锚点，检验分支聚合、安全标签邻接监督、保持支持的权重校准及模态专属传播。不能用B0-A的初始化/RNG或较强历史指标替换B0-B。

## 只读审计与实现证据

根仓库Git HEAD及逐文件SHA见manifest.json。实际查阅路径与关键位置如下，所有历史文件保持只读。

|来源|函数/位置|事实及本次处理|
|---|---|---|
|根README.md、ABIDE-simple-2-concat-weighted-cosine.sh|训练入口；shell第1行|维持lr=.0038、reg=.11、dropout=.35、hidden18/head2、nlayer1；450epoch、wait>200|
|main.py|set_rng_seed:44；train_and_eval:111及之后折循环|seed0、原预处理数据、SKFold10 shuffleTrue random_state10、全节点transductive|
|model.py|EvalHelper.__init__；run_epoch simple-2:122–151|两个Adam；ClsLoss+mean4 GraphConstructLoss；ALoss计算但不优化|
|network.py|GraphLearn:153；GCN:175；GFR:197、forward:235|四患者图，每个GCN传播完整320维；i==1覆盖i==0；阈值.9、原自环与normalize均不改基线|
|layers.py|InitialDataLayer:21；VariLengthInputLayer:80；EncodeLayer；GraphConv:192|原CAM/OutputLayer；GCN每支仅2个W参数；潜在4×36不是ROI|
|utils.py|GraphConstructLoss:31；ClsLoss:92；normalize_adj:101；my_weight_init:127|原smooth−.5degree图正则、加权NLL、对称归一化与Linear Xavier初始化|
|experiment/b0_pen_physical_removal_equivalence_audit/|audit.py build_pln_no_pen/build_b_agent及三步Adam检查；report.md；results.json|物理删除PEN、共有参数对齐时有效计算/Adam等价，但构造后RNG不同。没有复用其B0-A缓存或训练状态|
|experiment/b0_b_seed0_full10fold/|run.py make_pln_without_pen:103、make_agent:151、run_worker:287；manifest.json；逐折CSV与.runtime/*.json/log|本轮精确移植其物理PLN构造和Agent/指标函数；历史是GPU0折1–5、GPU1折6–10两条各seed0一次的连续随机流，并非单个十折流|
|experiment/experiment32_disease_contrastive_evidence_routing/|integration.py patch_network；run.py source_parity/worker；manifest、results/report.md、fold_metrics.csv|B0-A保留闲置PEN层及缓存构造，forward绕过；只作次级历史参照，不加载其锚点|
|experiment/experiment43_corr_anchored_gfr_audit/|run.py make/training_step/preflight/worker/report；method.py gfr_forward/calibrate/mutual_support；integration.py与patch；results/report.md|旧E02是阈值前MSE并非原ALoss；旧E03同时改mutual96支持；旧E04只屏蔽输入；本轮分别控制这些因素|
|本地已缓存原论文|experiment/experiment43_corr_anchored_gfr_audit/.runtime/paper.pdf、paper.txt（Sec2.1 Eq1–2、Sec2.2、消融）|本次在线PDF读取超时，使用已有本地论文；SHA和实际路径存manifest；不声称重新下载成功|

原论文：[SPromptGL, MICCAI2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)。论文式(1)为平方Frobenius；实际仓库ALoss是归一化图的非平方Frobenius，且未反传。本轮E02依据实际仓库公式构建安全训练诱导子图版本，不能称复现了论文平方损失。

## 预注册矩阵与唯一改动

|组|相对共同B0-B/声明前置组的改动|验收|
|---|---|---|
|E00_B0B_REPLAY|原GFR包括覆盖行为；物理不构造PEN|历史逐折参数SHA、CPU/CUDA入口SHA、混淆、四指标、best/执行epoch全部匹配|
|E01_AGGREGATION_FIX_V2|X+sum(sigmoid(w_i)*GCN_i(X,A_i))，四支都累加|四支分类梯度/消息/固定模型关闭响应|
|E02_SAFE_LABEL_ADJ_SUP_V2|E01＋训练诱导子图原仓库ALoss；lambda首步梯度比例校准|无test标签；投影梯度有效；逐折冻结lambda与比率|
|E03_SUPPORT_PRESERVING_CAL_V2|E01＋仅已有非对角边的median/IQR sigmoid校准|支持逐元素不变、对角原值保留、原自环/归一化|
|E04_MODALITY_LOCAL_PROP_V2|E01＋每支输入和输出均只保留对应模态块|块外严格零；同形状GCN，无缩放|
|E05_FIX_PLUS_SUPPORT_CAL_V2|按用户方案仍是E01＋相同支持保持校准|与E03数学重复，独立训练两组并核对逐值相等|

**方案内重复必须披露：E03已经以E01为基础，因此E05与E03没有可分离因素。** 不自行改变其中任何一组，也不把重复训练当作独立科学证据。两组完整各10折；E05为满足创新折25/25分配，在GPU0执行折1–5、GPU1执行折6–10。所有创新折逐折恢复配对锚点，不依赖组内连续RNG，所以该明确分配不改变数学实验。

输入固定871×320，模态48/6/10/256，ASD=0(403)、TD=1(468)。ANAT/FUNC是质量摘要，不是病理测量。B0-B PLN注册49,544元素（35参数张量），其中原有PLN fusion_weight6元素闲置；不创建MLP、avg_pool、prompt buffer，不导入CLIP。GFR887,836元素。

原B0-B历史只有指标/参数与CPU-CUDA RNG哈希，没有保存完整未训练锚点或逐人预测向量。因此本轮重新运行E00十折并存未训练锚点；不冒称可核对历史预测SHA或完整Python/NumPy历史RNG。已有历史字段要求逐折精确一致，本轮另保存完整RNG与预测向量SHA。

## 数学边界

E02：先提取outer-train诱导图，再各自加I并normalize；同类target含对角1，再加I为2。预测raw对角≈1，再加I≈2。四图非平方Frobenius取平均。只把train标签向量交给新增监督函数。首个同一次train-mode前向，针对全部四个GraphLearn.p.weight/p.bias测量分类与监督梯度的联合L2范数，lambda=min(1,0.1*g_cls/(g_sup+1e−12))，每折冻结。未使用test成绩标定；基础ALoss仍不参与基础优化。

E03/E05：分数统计来自全部无序train非对角pair；median/IQR detach，IQR下界.001。只改变原raw>0的非对角边；非边0，原对角逐位保留；每次forward assert支持和对角。**校准仅供消息传播，GraphConstructLoss始终接原始raw GraphLearn输出**，以免同时改变图正则。不会让训练后E03图与独立训练后E01图强行一致；支持不变的比较指同一参数/同一forward中校准前后。

E04：块[0:48]/[48:54]/[54:64]/[64:320]；先输入乘固定mask，再同形GCN，最后输出再乘mask；保留完整X残差，不放大消息。

## 随机数、恢复和运行

```bash
cd /data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest
python -B run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B run.py preflight
CUDA_VISIBLE_DEVICES=1 python -B -u run.py all
# 中断后，跳过经过SHA验证的完成折；折内恢复到最近10epoch checkpoint
CUDA_VISIBLE_DEVICES=1 python -B -u run.py resume
# 重建当前已完成结果，不重训
CUDA_VISIBLE_DEVICES=1 python -B run.py report
```

prepare生成单份隔离network/layers/utils与patch。run.py中的物理PLN/Agent函数为历史最小移植，无运行时历史Python实现依赖。根数据和源码只读。每卡2worker、每worker2CPU线程，实际设备PPU-ZW810E，98,304MB；无模型下载。E00先两卡各5连续折，历史精确验收后启动50创新折（每卡25）；全部最多60折。入口锚点保存原始模型、双Adam及Python/NumPy/CPU/CUDA四种RNG；新增无参数/无初始化，不消耗额外RNG。最佳模型机制诊断只读，所有RNG恢复到训练结束边界后再进入下一基线折。

.runtime/checkpoints覆盖式latest每10epoch，包含当前模型、双Adam、最佳状态、epoch/wait、lambda标定与RNG；完成时先存并读回校验best，再原子提交唯一(group,seed,fold)结果，删除冗余latest。基线每折保存边界RNG以便跳过完成折仍保持历史流。文件锁保护controller和每折。代码/配置、输入/历史文件和split/anchor/best签名均可审计。正式结束保留60个必要best及10初始锚点，不保存逐epoch模型。

预检包括历史fold1模型/RNG、独立原始训练step的双Adam首步等价、两步后磁盘恢复第三步等价、四支梯度、标签置换、有限性、原边支持/对角和局部传播、E03/E05首步完全一致。开发期间修正了参数张量与state_dict计数区别、GCN参数切片、诊断键重复、最终epoch提交窗口恢复；均在任何正式训练之前完成，详情manifest。严格失败即停止创新训练，不切换B0-A。

## 指标及筛选

ACC / one-hot hard-label AUC / SPE-TD / SEN-ASD，十折mean±population SD%，逐折混淆、epoch、预测/参数/RNG/split/best SHA完整保存；没有prob-AUC或第五正式指标。每组对B0-B四项pp差及ACC/AUC W/T/L。GO须两主指标都≥+0.1pp、两者都至少7/10严格胜且机制全部通过；否则NO-GO，不调参。B0-A92.0742/91.8546/94.6300/89.0793仅次级参照，不与B0-B混同。

val=test选checkpoint、CV前RFE/全局标准化、全871节点transductive及十折训练集重叠全部保留；这些结果只可用于历史协议内部比较，不是无偏独立泛化、显著性、医学因果或SOTA证据。原论文全320维传播是设计，不预设为bug；修复覆盖也不预设带来收益。

## 本次运行存储恢复

/data初检通过，但汇总进程读取正在原子替换的任务JSON时发生短暂ENOENT，原训练worker独立完成了十折基线。73文件（78,391,429字节）迁移到`/tmp/SPromptGL-E43_1-stage/experiment/experiment43_1_b0b_gfr_mechanism_retest`并逐文件SHA核验后，从同一源码resume；无模型/训练方法改动、无完成折重训。十折锚点的输入与源码绑定见`.runtime/anchors/index.json`。最终交付以manifest.delivery和逐文件同步核验为准；/tmp非持久存储保障。

## 独立交付审计入口

新增一个必要的`audit.py`隔离训练后验收与有界交付，避免修改已经冻结、签名绑定的训练代码。它可重新加载全部60个最佳模型重算四指标/预测、检查held-out标签推理不变性、同模型关闭四路消息后恢复、生成单因素配对与同步/安全handoff追加；不训练模型、不新增实验组。

```bash
CUDA_VISIBLE_DEVICES=1 python -B audit.py --verify
python -B audit.py --deliver
```

控制器需显式只可见单卡（上述命令已注明），因为最终模型相等性检查共用单卡runtime；其子进程会独立覆盖CUDA_VISIBLE_DEVICES，物理GPU分配仍为0/1各30正式折。未修改任何训练方法以解决该报告层断言。

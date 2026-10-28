# Experiment 47.1 — GCN有效消息幅值衰减根因归因

本轮在 `/tmp/SPromptGL-E47/experiment47_1_gcn_message_collapse_attribution`
完成。它是对 Experiment47 的 P5 归因，不是新的 GFR 模块。根目录模型、数据、
历史实验和 checkpoint 没有写入。Stage A 完成十折静态分解，Stage B 在预定的
fold1、fold6 完成原始 B0-B 训练复演，Stage C 完成 T0 两折、T1 十折、T2
两折，共14个完整训练任务。GPU0 执行 fold1–5，GPU1 执行 fold6–10；T1
扩展严格依照预注册机制门槛。

## 1. 论文、源码和基线身份

论文 Section 2.1/Eq.(1)–(2) 描述的是四关系患者图、两层图传播和图构建
目标。实际 immutable source snapshot 的关键路径是：`GraphConv.forward`
先做 `input @ W`，再做 `adj @ support`；`GCN.forward` 是第一层 ReLU 后的
第二层；`GFR.forward` 对每个模态从完整320维输入传播，且在 `i==1` 重新写入
`combined_feat = f_w * feat1 + raw_features`，覆盖此前的 PHENO 分支。四路
图构建使用 `GraphLearn.p` 后余弦阈值0.9。`GraphConstructLoss` 的输入是
模态原始 `parts` 和 GraphLearn 邻接，不含 GCN 的 `W1/W2`；simple-2 返回的
`ALoss` 不进入训练反向传播。

Experiment43.1 的十折 B0-B checkpoint、anchor、划分和输入 SHA 通过；本轮
T0 fold1/fold6 的训练损失首步（分类 1.1454628706、图损失 -1.1294156313、
总损失 0.0160472393）与历史一致，best epoch、执行 epoch、混淆计数、四项
指标和预测 SHA 均逐值一致：fold1 best=157，fold6 best=155。因此没有混用
B0-A 或 PEN 模型。

历史统一参照是 B0-B：ACC 91.7294±2.8596%、Hard-AUC 91.6046±2.8059%、
SPE-TD 93.1360±4.5354%、SEN-ASD 90.0732±3.5117%；CAM-only 的历史参照为
ACC 91.7294±3.0819%、Hard-AUC 91.4880±3.0586%、SPE-TD 94.6346±3.7878%、
SEN-ASD 88.3415±3.3369%。

## 2. Stage A：衰减发生在哪里

逐层分解严格采用原始 `normalize_adj(adj + I)`：`X`、`XW1`、
`A_norm(XW1)`、ReLU、第二层线性、`A_norm`、fusion sigmoid 和最终
`GFR-X`。十折最佳 checkpoint 的均值如下；`W` 和输出均为 Frobenius 范数。

|关系|初始 W1|最佳 W1|初始 W2|最佳 W2|最佳 XW1|最佳 A_norm XW1|最佳 ReLU|最佳 A_norm(HW2)|最佳 fusion 后消息|最佳 ReLU 零比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ANAT|25.3169|0.0896|25.2952|0.0662|11.9981|11.1346|0.7213|0.0182|0.0105|0.8372|
|FUNC|25.3111|0.0570|25.2998|0.0389|7.7014|7.2228|0.4838|0.0064|0.0037|0.8363|
|CORR|25.2880|0.1021|25.3059|0.0626|15.7266|15.1799|0.3932|0.0068|0.0040|0.8321|
|PHENO|25.2781|25.2781|25.3182|25.3182|—|—|—|—|142.5316|0.5017|

ANAT/FUNC/CORR 的 W1 最终/初始均值约为0.0023–0.0040，W2约为
0.0015–0.0026；初始消息并不小，训练后才收缩。图归一化不是主要的线性
收缩：最佳模型中 `A_norm(XW1)/XW1` 约为0.93、0.94、0.97，第二层
`A_norm(HW2)/(HW2)` 约为0.99；这与最终10^-2量级输出不相称。ReLU 零比例
由初始约0.48–0.52升到约0.83–0.84，会进一步削弱信号，但用初始权重替换
后仍能恢复百量级消息，因此不能把它单独称为根因。fusion sigmoid约0.6，
也不足以解释数百万倍下降。关键矩阵乘法另用 FP64 重算；最佳状态实际行的
FP32/FP64 相对范数误差均值约 `1.10×10^-5`、最大约 `1.98×10^-5`，远小于
W1/W2造成的多数量级变化，没有把浮点重复误差当成消息信号。

### 固定因素替换

四种未联合训练的诊断组合得到以下三路 fusion 后消息均值：

|组合|ANAT|FUNC|CORR|解释|
|---|---:|---:|---:|---|
|best adjacency + best W|0.0105|0.0037|0.0040|真实最佳状态|
|best adjacency + initial W|172.96|195.39|176.37|恢复初始权重后消息恢复|
|initial adjacency + best W|0.0106|0.0038|0.0041|换回初始图仍不能恢复|
|initial adjacency + initial W|184.06|211.63|216.71|初始状态|

因此主要幅值损失来自训练后的 GCN W1/W2 收缩，而非邻接归一化；W1、W2
共同作用，不能把全部下降归给第二层。PHENO 的中间消息仍很大，但 `i==1`
覆盖使它不形成最终独立 PHENO 增量。

## 3. Stage B：训练轨迹与优化器

fold1/fold6 的 T0 逐步复演与历史训练一致；独立原始操作顺序复演的前两步
模型、双 Adam 状态和 RNG 逐值一致。训练早期三路 W 从约25开始，
随着 Adam 更新持续下降；最佳 checkpoint 才出现上述小 W 和弱消息。对关键
GCN W 的一步 Adam 反事实更新显示，原始 T0/T2 使用 weight decay=0.11，
诊断的实际 weight-decay 分量约占 W 更新范数的0.34–0.36（首步因 Adam
偏置校正可高于1，不能把 `lr*wd*W` 当成实际更新）。T1 的六个有效 W 参数
组 weight_decay严格为0，其他 GFR、GraphLearn 和 CAM 参数仍为0.11。

图正则对 GCN W1/W2 的梯度在源码依赖上严格为零；`gradient_diagnostics.csv`
逐参数记录了这一点。图正则可以更新 GraphLearn 邻接，进而间接改变分类
梯度，但不能直接压缩 GCN W。T0 的全 GFR 分类梯度和图梯度均被记录，不能
仅凭全向量梯度大小宣称目标冲突。

## 4. Stage C 单因素训练

T0/T2 两折只作为 pilot，不生成十折均值；T1 满足两折幅值门槛后扩展至十折。

|组别|折|ACC|Hard-AUC|SPE-TD|SEN-ASD|best epoch|
|---|---|---:|---:|---:|---:|---:|
|T0 B0-B|1|95.4545|95.2776|97.8723|92.6829|157|
|T0 B0-B|6|95.4023|95.1862|97.8723|92.5000|155|
|T2 无图正则|1|95.4545|95.1220|100.0000|90.2439|122|
|T2 无图正则|6|96.5517|96.4362|97.8723|95.0000|443|

T1 十折正式均值±population SD（%）为：

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|T1 GCN_NO_WD|71.2970±4.7123|70.9726±4.8432|75.4209±5.8353|66.5244±8.6896|

T1 每折 ACC 为 71.5909、70.1149、66.6667、75.8621、71.2644、79.3103、
65.5172、66.6667、67.8161、78.1609%。它没有改善疾病分类，反而远低于
B0-B；这排除了“只要把消息放大就会得到患者图增益”的解释。

## 5. 固定 checkpoint 消息依赖

`interventions.csv` 对每个实际 checkpoint 重新执行两层传播；“严格置零”
和“self-loop only”分开实现，发送者错配为确定的列置换，不是换数组顺序。

|组别|原始 ACC|GCN严格置零|self-loop only|纯跨患者|发送者列置换|平均严格置零翻转数/折|
|---|---:|---:|---:|---:|---:|---:|
|T0（fold1/6）|95.43|95.43|95.43|95.43|95.43|0|
|T1（十折）|71.30|54.20|59.24|68.08|53.27|34.9|
|T2（fold1/6）|96.00|94.85|96.00|96.00|96.00|1个患者|

T1 中 GCN 消息确实进入决策：严格置零、self-loop 和纯跨患者干预都会
改变输出，发送者列置换通常更差；但这是对一个已经大幅降准确率的模型的依赖
诊断，不能证明真实关系是疾病有益关系。T0 的所有干预均无类别翻转，说明
原始 B0-B 的患者消息仍几乎不进入最终判别。T2 仍接近 T0，只有 CORR 关闭
在两折中产生一次翻转。

## 6. 原因—代码—轨迹—干预—分类影响—竞争解释

|候选原因|证据|结论|
|---|---|---|
|初始化已很小|初始化 W约25、三路消息约184–217|排除|
|邻接归一化本身收缩|逐层比例约0.93–0.99；初始图替换不能解释小输出|不支持为主因|
|ReLU激活抑制|最佳零比例约0.83，较初始升高|有贡献，但不是充分根因|
|W1/W2 weight decay收缩|训练轨迹持续收缩；T1去掉六个有效 W 的 decay 后两折均超10倍恢复，十折复现|**VERIFIED_GCN_WEIGHT_DECAY_ATTRIBUTION（对幅值，不是唯一因素）**|
|GraphConstructLoss直接压缩 W|源码不依赖 W；W 的 graph gradient为0|排除直接作用|
|图目标间接改变图学习|T2去图正则未一致恢复消息；仅有间接可能，未闭合因果链|未证实|
|分类器完全忽略图|T0严格置零无翻转；T1放大后确实有大量翻转|原始状态存在残差捷径，但不解释 T1 的关系价值|
|真实患者关系提供疾病增量|T1消息变大且跨患者干预有效，但分类下降、无错误关系重训对照|未证实|

## 7. 科研判定

本轮形成了一个闭合的**幅值退化归因**：三路 GCN 不是初始化为零，图归一化
也不是主要压缩；原 Adam 的0.11 weight decay 在两折机制 pilot 和十折 T1
扩展中得到一致支持，导致 W1/W2 收缩，ReLU进一步放大了稀疏化。T2 pilot
不支持“GraphConstructLoss 是直接原因”；由于图正则不对 W 反传，任何间接
影响都必须另行设计实验，不能在本轮包装为已证实。

但是，去掉有效 GCN 的 weight decay 只恢复了幅值，没有恢复疾病判别：T1
十折 ACC 71.2970%，远低于 B0-B 91.7294%，且固定模型干预提示它使用了
跨患者信息却没有显示真实关系特有的正向增益。因此最终状态为：

- `VERIFIED_GCN_WEIGHT_DECAY_ATTRIBUTION`：仅指 ANAT/FUNC/CORR 幅值衰减的主要可验证成因；
- `AMPLITUDE_RESTORED_NO_INCREMENTAL_VALUE`：幅值恢复不等于疾病分类价值；
- `GRAPH_OBJECTIVE_ATTRIBUTION`：未通过；
- `FINAL_NO_GO_FOR_AMPLITUDE_ONLY_GFR_INNOVATION`：不建议把“去 weight decay 放大消息”作为 MICCAI 2027 第一创新点。

真实患者关系是否能提供超出 CAM-only、错配关系和随机关系的疾病增量仍未被证明。
本轮没有设计新的 GFR 模块，也没有启动 47.2/48；若继续研究，必须先建立
关系价值的独立、标签隔离正/负控制，而不是继续调大消息或堆叠传播门控。

## 8. 局限和交付

仍沿用历史 `val=test` 选 best、外层 CV 前全队列 RFE/标准化和 871 人
transductive 关系，因此这些结果不是无偏独立泛化或临床因果证据。固定模型
的 self-loop、纯跨患者和发送者错配属于模型依赖诊断，不等同于从头重训的
反事实因果。T1 的十折是单因素归因扩展，T0/T2 仅两折 pilot，报告没有伪造
其十折统计。

结果文件：`layer_amplitudes.csv`（Stage A 240 行）、`training_trajectory.csv`、
`optimizer_diagnostics.csv`、`gradient_diagnostics.csv`、`fold_metrics.csv`
（T1十折）、`pilot_metrics.csv`、`best_amplitudes.csv` 和
`interventions.csv`。手工分解与hook在 `.runtime/preflight_checks.json` 核验。完整 source、anchor、历史 best 和受保护文件 SHA 在
`manifest.json`；恢复与复演检查在 `.runtime/replay_check.json`。

本次向 `/data/adodas/SPromptGL-main/handoff.md` 的幂等追加因 OSSFS
`OSError(5, Input/output error)` 未完成；待追加正文保存在
`/tmp/SPromptGL-E47/experiment47_1_cugl_handoff_append_pending.md`，没有将失败
状态伪称为已同步。

# 实验十四：Hierarchical Semantic Prompt Design and Cross-Level Interaction

14组固定设计，3 seeds×10 folds，42个独立任务，420折。仅在临时源码副本集成；不修改根源码、实验一至十三，不 commit/push。

## 论文、源码和历史审计

实际阅读根目录原论文PDF的Fig.1、2.2和消融，PyMuPDF可解析并渲染。CAM跨模态交互；PEN包括局部SPGS和RES；文本CLIP冻结，4096→128→9共享MLP；局部Softmax-Max权重后经RES加回全局表示、concat分类。论文消融支持模块组合，但不证明当前具体文本的必要性。论文ABIDE超参数与本地shell不同，本轮以已复现实验十三的本地shell为准。BrainPrompt/SABER本地PDF未发现，未以它们作为已读依据。

871人、48/6/10/256，ASD0/TD1。GFR为受试者图，保留原分支覆盖与未纳入总损失的ALoss。四个9维局部Token是潜在特征，没有ROI或功能网络一一对应关系。原论文图示和措辞不能提供源码中不存在的ROI映射。

实验一P1约91.80%，增益小且非三seed同时稳定；实验二至七的状态文本、MLP改造、标量方向门控、双主题残差未证明稳定的正确语义收益。实验八疾病分支与原分类概率高度相关；实验九P3+疾病E9略改善但没有疾病特异性对照。实验十真实ASD相对中性仅约0.077pp，固定监督类别代码仍是替代解释。实验十一256连接索引已恢复，252可安全命名，但本轮不重新引入连接级模块；实验十二pilot未支持静态ROI文本，实验十三240折个体化Prompt没有稳定优于P0，本轮不研究个体化。旧P1两个主题不解释为ASD/TD类别。

## 冻结矩阵

|ID|局部Prompt|疾病Prompt|新增参数|
|---|---|---|---:|
|E00|P0|OFF|0|
|E01|OLD_P1|OFF|0|
|E02|NEW_P1_SHARED|OFF|0|
|E03|NEW_P3_SHARED|OFF|0|
|E04|NEW_P3_SPECIFIC|OFF|0|
|E05|NEW_P3_SPECIFIC|HISTORICAL|36865|
|E06|NEW_P3_SPECIFIC|FC_CENTRIC|36865|
|E07|NEW_P3_SPECIFIC|BALANCED|36865|
|E08|NEW_P3_SPECIFIC|SELECTIVE|36865|
|E09|NEW_P3_SPECIFIC|CLINICAL_CORE|36865|
|E10|NEW_P3_SPECIFIC|NEUTRAL|36865|
|E11|P0|SELECTIVE|36865|
|E12|NEW_P1_SHARED|SELECTIVE|36865|
|E13|NEW_P3_MISMATCHED|SELECTIVE|36865|

P0八句含历史拼写Nromal，OLD_P1逐字32句；新共享版本8句、每类4句拼接4096，专属版本32句、[4,2,4096]通过同一MLP。E13只以[2,3,0,1]置换E08相同文本；没有新学习层。P0保留原matmul、旧P1保留历史einsum，Softmax-Max、RES和分类器不变。max门控对两类交换不变，不称为有向疾病证据。

疾病分支严格沿用E9：s=normalize(Linear36→1024(hidden))，C=normalize(mean(normalize(CLIP句向量)))，logits=sCᵀ/.2；CE仅train_idx、原权重、系数.1；output=log_softmax(base_logprob+.2tanh(alpha)×logits)。alpha0；新增36,865参数、Adam lr.0038/wd0；原组lr.0038/wd.11。全部疾病配置相同结构，仅冻结文本原型不同。

## 随机性、缓存与恢复

先正式训练E00三个连续十折，保留首次RN50原位置加载消耗，记录各折全部RNG入口。E00精确匹配实验十三30折六指标后，其他39任务恢复这些配对入口；新增疾病投影在原my_weight_init后fork_rng(140000+seed)，不改变公共初始化/dropout随机流。CLIP文本prepare时一次编码；训练编码被禁止。逐句检查BOS/EOS≤77，truncate=False。缓存与checkpoint存放本机/tmp临时state，通过实验目录state软链接查看；结束清理，复跑prepare重新构建。OSS目录只保留归档结果和日志。

checkpoint记录epoch/model/两Adam/best模型/早停/RNG/已完成逐折结果，原子写入并SHA256核验。最佳ACC严格提高、首次平局保留、wait>200、450上限；原第五折低ACC防护保留，触发会报任务失败。任务和scheduler持有独占锁；进程中断从完整epoch恢复，已完成折不重计。每GPU21任务/210折，默认4并行、每worker2线程；CUDA_VISIBLE_DEVICES物理绑定，逻辑卡0。

## 指标、反事实与限制

Hard-AUC原one-hot硬预测实现，在二分类中等于BA；Probability-AUC以ASD阳性并取output.exp()[:,0]。先seed内十折再3seed mean±population SD，30行共用fold不作为独立样本。E08同一最佳checkpoint做REAL、疾病融合OFF、疾病原型互换、中性疾病原型、局部模态错配5种推理，无重训练。OFF只移除最终融合，不撤销辅助训练。E10/E13另是重新训练对照。

历史val=test导致测试参与checkpoint选择，固定CSV上游全局标准化和RFE有共享信息，结果不宣称独立无偏泛化。共享→专属比较同时改变文本细化与独立句数，不能归因于模态对应单一因素。

## 运行

```bash
python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py prepare
CUDA_VISIBLE_DEVICES=0 python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py preflight
python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py all
python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py resume
python /data/adodas/SPromptGL-main/experiment/experiment14_hierarchical_semantic_prompt/run.py report
```

源码/数据/文本/CLIP/patch SHA256、Git版本、GPU UUID及状态见manifest；fold_metrics420、diagnostics420、counterfactual150、完整中文报告见results。完成审计后清理state、临时副本和checkpoint。

已完成的all/resume/preflight只校验并跳过，不重建临时cache。report可重新生成并更新报告哈希；独立重新准备时优先使用本实验冻结文本及reference，旧实验cache仅在存在时核验。首次正式运行的准备、文本修正和本机I/O迁移见manifest repairs。

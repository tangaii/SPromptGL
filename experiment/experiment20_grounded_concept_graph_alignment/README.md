# SABER Experiment20：固定疾病概念锚定与患者级证据图联合验证

冻结20组×3seed×10fold=600fold。基于原始P0及实验九P3+D/E9，新增模块只在Encoder→FeedForward之后、SPGS之前作用于CORR的36维潜在特征。根源码与实验一至十九只读。

文本完全来自实验九与十九；HO连接映射来自实验十一经过数值、标签、列序核验的两表。252个连接是线图节点，共享HO ROI定义固定连边；患者FC只调节边权，不改变物理脑连接。

`d=clip((FC-train_mean)/train_std,-6,6)`；参考统计仅使用当前训练折全体，不使用标签。四域固定prior直接进入候选softmax，禁止自由Key或额外证据诊断头。`3→32 + edgeID`编码，136→64→36读出；`0.1*tanh(alpha)*||base_CORR||*normalize(readout)`残差；alpha=0，绝对相对范数≤10%。

Local新增19429参数；Graph新增20517参数；其中没有进入候选读出或一跳邻居的ID参数可能无梯度，诊断报告实际可达连接。E9另加36865，E3判别器另加1388。新参数Adam lr=.0038、weight_decay=0；原两Adam和原始损失保留。

按用户最新授权，正式实验目录为 `/tmp/experiment20_grounded_concept_graph_alignment`。`/data` 仅作为已审计源码和数据的只读来源；完整更新后的handoff保存在 `/tmp/experiment20_grounded_concept_graph_alignment/handoff.md`，保留根handoff原有全部内容。

|配置|Prompt|D|GRL|先验|患者线图|
|---|---|---|---|---|---|
|T00|P0|False|False|无|False|
|T01|P3|False|False|无|False|
|T02|P3|True|False|无|False|
|T03|P0|True|True|无|False|
|T04|P0|False|False|NUMERIC|False|
|T05|P0|False|False|REAL|False|
|T06|P0|False|False|NEUTRAL|False|
|T07|P0|False|False|WRONG_ANATOMY|False|
|T08|P0|False|False|ISOMETRIC|False|
|T09|P0|False|False|ANONYMOUS|False|
|T10|P0|False|False|NUMERIC|True|
|T11|P0|False|False|REAL|True|
|T12|P0|False|False|NEUTRAL|True|
|T13|P0|False|False|WRONG_ANATOMY|True|
|T14|P3|True|False|NUMERIC|False|
|T15|P3|True|False|REAL|False|
|T16|P3|True|False|NEUTRAL|False|
|T17|P3|True|False|NUMERIC|True|
|T18|P3|True|False|REAL|True|
|T19|P3|True|False|NEUTRAL|True|

全部超参数、任务分卡、核心比较、输入文件与数值定义冻结在design.json；版本及哈希在manifest.json。正式训练前原实验九四路径初值/输出/三步Adam逐值一致，并以正式T00–T03的120折验证历史指标及可获得的epoch，过关后才释放新配置。T00还比对实验19 C0、实验13 F0。

每个seed进程内连续十折，只在任务开始播种；不同组早停造成后续折随机流分叉，不把相同fold误称全程同初始化。每25epoch及训练停止点原子保存模型、原两Adam、E3判别器Adam、所有RNG与图邻接；完成折的指标、诊断和反事实提交后，折边界检查点保留下一折所需的连续RNG与已提交结果。任务结束删除临时权重。

反事实只对T05/T11/T15/T18同一最佳checkpoint执行九条件；图条件每次重建邻接并逐值检查REAL恢复。Local的GRAPH_NUMERIC保留NA行，不把它算一次有效评估。

**历史协议局限**：val=test且用test ACC选checkpoint；全队列标准化与监督RFE先于外层CV；GFR在871人上transductive传播。结果有乐观估计风险，不能称独立无泄漏泛化。三个seed复用相同十折，不进行虚假的30独立样本检验。

MMGPL概念加权与语义图、MGDR多模态图分解、BrainPrompt多层语义注入均已有先例；本轮只能检验固定医学对应关系是否超越数值/中性/匿名/同Gram随机/错解剖控制，不能声称CLIP概念或动态图首次提出。MMGPL使用1029人MRI+FC、BiomedCLIP、5折×3seed，其72.39%不与本协议91.69%横比。

```bash
cd /data/adodas/SPromptGL-main
python /tmp/experiment20_grounded_concept_graph_alignment/run.py prepare
python /tmp/experiment20_grounded_concept_graph_alignment/run.py preflight
python /tmp/experiment20_grounded_concept_graph_alignment/run.py all
python /tmp/experiment20_grounded_concept_graph_alignment/run.py resume
python /tmp/experiment20_grounded_concept_graph_alignment/run.py refresh
python /tmp/experiment20_grounded_concept_graph_alignment/run.py report
```

首次prepare从经核验的根核心源码在临时目录应用实验九已验收diff，生成本实验唯一integration_patch.diff；任务各自重新复制根源码并应用本diff。语义缓存一次编码，训练新分支不加载CLIP；历史P3/疾病头仍按原实验九首次惰性编码顺序执行，保留其随机流。

审计参照提交：f36ed2e9f06b39e65d9ae70c549256dd39cf8e80；服务器HEAD：1167dd392837a02eb4f83481ae49f02607a9dae5。不执行git commit/push。

正式验收：20配置、60任务、600折、600诊断、1080反事实（1020有效、60 NA），四历史路径120折parity通过；GPU各300折。`results/report.md`包含自动统计及完成后的科学复核，`handoff.md`保留原handoff全部历史并追加本轮最终结果。

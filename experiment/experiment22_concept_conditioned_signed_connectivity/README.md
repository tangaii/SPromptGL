# SABER22：概念条件化有符号功能连接证据

固定18组、seed0/1/2各连续10折，共540新fold。额外导入实验21六组180历史fold，只用于paired-split比较。禁止自动commit/push。

三个指定Python模块逐字复制自根prompt；run.py由实验21审计模板一次复制并适配，不导入历史实验Python模块。训练只使用本目录semantic_cache.npz，不依赖临时实验20缓存。

数学：训练折全体FC估计center/scale；仅outer-train ASD/TD标签估计contrast，r=tanh(contrast/.4)，s=tanh(d/2)r。它是训练监督的弱疾病关联，不是临床症状或病理真值。
四概念正负系数各4标量，.35sigmoid、初值.1；attention=masked softmax(P/.8+(1+.5P)(a+max(s,0)-a−max(−s,0)))；仅train的.002KL。UNSIGNED只去attention修正符号，数值encoder/readout仍带符号；SHARED保留8参数并取均值。
图节点252条经HO端点核验的RFE保留FC连接；共享ROI决定固定line support，gate×概念profile余弦×绝对strength控制患者权重，单位self-loop+归一化聚合。不是新的物理脑FC拓扑。
仅CORR潜在slot：[871,4,36] Encoder/FeedForward后、原SPGS前追加≤10%归一化残差；alpha=0，fork_rng 220000+seed。local/graph各19437/20525新增参数；E9类别头另36865且完全沿用实验9数学。
E9互补组仅将delta方向正交于当前CORR潜在向量；不保证信息论互补。原四个9维局部token不是脑区。

|组|基座|prior|图|变体|正交|问题|
|---|---|---|---|---|---|---|
|M00|P0|REAL|False|SIGNED|False|P0 + disease-concept signed local|
|M01|P0|NUMERIC|False|SIGNED|False|P0 + numerical signed local|
|M02|P0|NEUTRAL|False|SIGNED|False|P0 + neutral signed local|
|M03|P0|WRONG_ANATOMY|False|SIGNED|False|P0 + wrong-ROI signed local|
|M04|P0|ISOMETRIC|False|SIGNED|False|P0 + isometric random signed local|
|M05|P0|REAL|False|UNSIGNED|False|P0 + real text; omit FC direction|
|M06|P0|REAL|False|SHARED|False|P0 + real text; one shared domain adaptation|
|M07|P0|REAL|True|SIGNED|False|P0 + true prior + signed evidence + magnitude-preserving graph|
|M08|P0|NUMERIC|True|SIGNED|False|P0 + numerical signed graph, no semantic factor|
|M09|P0|NEUTRAL|True|SIGNED|False|P0 + neutral signed magnitude graph|
|M10|E9|REAL|False|SIGNED|False|E9 + true signed local|
|M11|E9|NUMERIC|False|SIGNED|False|E9 + numerical signed local|
|M12|E9|NEUTRAL|False|SIGNED|False|E9 + neutral signed local|
|M13|E9|REAL|False|SIGNED|True|E9 + true signed local with orthogonal CORR residual|
|M14|E9|REAL|True|SIGNED|False|E9 + true signed magnitude graph|
|M15|E9|NUMERIC|True|SIGNED|False|E9 + numerical signed graph|
|M16|E9|NEUTRAL|True|SIGNED|False|E9 + neutral signed graph|
|M17|E9|REAL|True|SIGNED|True|E9 + true signed graph with orthogonal CORR residual|

原协议原样保留：871×321输入，特征48+6+10+256；ASD=0/403，TD=1/468；StratifiedKFold(10,shuffle=True,random_state=10)，450epoch/early200，lr.0038，原Adam WD.11，新分支WD0，dropout.35。
val=test、test ACC严格提升选最佳checkpoint、全局CV前监督RFE与标准化、全871 transductive传播存在乐观偏差；仅作历史协议比较，不是独立无泄漏泛化。原GFR i==1覆盖和ALoss不反传均保留，未加入新诊断BCE或GRL。
每卡固定27seed任务/270fold，起始上限4worker，每worker2CPU线程；根据cgroup/GPU显存下调。每25epoch和fold边界原子保存完整模型/Adam/RNG，可精确恢复。一个seed的10fold不重新播种；不同方法早停会分叉后续随机流。
补丁重建：读取根CORE→在临时oracle fuzz0应用实验9patch→CodeC生成新源码→保存相对根CORE的唯一integration_patch.diff。worker只复制根CORE并fuzz0应用本补丁一次，SHA严格核对。原目录只读。
六核心组同best checkpoint的13反事实=2340行，local三图操作270 NA；2070有效，无重训。图实时重算，恢复全部buffer/输入/权重/RNG及bitwise REAL预测。NUMERIC_ONLY保留REAL图架构，不代替独立数值组。

准备根HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；审计GitHub `c2f07d274a281861a79dd35b979449b6b25e992e`；缓存SHA `4f0f746ca4b65a638e8dbc0d937ad0f38145c9492d8f80688efa35efa910e96d`。源代码、RN50、数据、Prompt、映射、cache原始数组和历史参考签名见manifest。

临床身份链未建立（0/871可核验），未复制患者ID/原始临床记录。正确文本/数值/中性/随机/错解剖对照共同决定科学门槛，不由最高ACC单独决定。
[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)提出GFR+PLN/SPGS/RES；[MMGPL](https://arxiv.org/html/2312.14574v2)已有疾病概念加权与语义Token图。概念条件化GNN/语言Prompt本身不能作为首次创新。

```bash
cd /data/adodas/SPromptGL-main
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py prepare
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py preflight
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py all
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py resume
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py refresh
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py report
python experiment/experiment22_concept_conditioned_signed_connectivity/run.py validate
```

最终仅保留本README、四个Python文件（含run.py）、静态cache、design、manifest、唯一integration_patch和五个结果文件/简短scheduler日志；临时模型/源码/worker日志仅在/tmp，完整验证后清理。

诊断算子修订：CUDA float histc在确定性模式下不可用，已将两个直方图统计移到CPU；精确迁移检查点并补测后续跑。模型公式及超参未变，记录见manifest.engineering_revisions。完整实测科研复核见report末尾；七科学门槛均FAIL。

# DS³A-v2：实验十七

本实验在原SPromptGL P0之外增加读取真实原始模态的选择性Shared/Private辅助分支。正式范围固定12组×3 seeds×连续10 folds；不改变根源码、数据、既有实验，不自动commit/push。结果仅用于本项目历史协议内比较。

## 源码及数据审计

官方仓库`wanxixi11/SPromptGL`与本地的差异主要为CLIP文本缓存、标签one-hot设备适配；项目远端`tangaii/SPromptGL`六个核心文件与本地逐字节一致。审计提交、六文件哈希和所有数据/CLIP哈希见manifest。已逐项读取原论文Fig.1–3、Section2.1/2.2与Eq.(1)–(7)，本地PDF解析文本与官方相同。

原GFR在四个关系图上处理拼接320维，输出[871,320]已经混合，且循环中的已有覆盖行为保持不变；不能把输出分段视为纯模态。原`simple-2`反传`L_cls+L_graph`，GFR的ALoss虽计算并打印却未进入loss。PLN InputLayer/CAM产生[871,4,36]；此次注入在最后FeedForward之后、`text_features = self.prompt_text_features`之前。P0的8×1024→2×4096→MLP(4096,128,9)，产生两个非负9维原型；原cosine→softmax→max标量加权[871,4,4,9]，global平均池化与局部channel-softmax响应相加，分类器输出log-softmax。该实现不能直接解释成真实ROI或论文所有公式的完整实现。

CSV实测871×321，320个输入+标签；ASD原1→0(403人)，TD原2→1(468人)。PHENO48=26年龄分箱+20站点+2性别，新增PHENO仅由列名前缀提取28个年龄/性别字段；站点仍存在原主干。ANAT6=cnr/efc/fber/fwhm/qi1/snr，FUNC10=efc/fber/fwhm/dvars/outlier/quality/mean_fd/num_fd/perc_fd/gsr，均为预计算影像QC，不是脑体积或完整时间序列；`func_gsr`是Ghost-to-Signal Ratio，不能称为全局信号回归。QC方向不统一，不手设“越大越好”，sigmoid只称quality-conditioned confidence proxy。CORR256来自HO ROI相关矩阵、上三角Fisher-z及既有RFE，原数据处理用ConnectivityMeasure相关估计；当前不重算。列顺序、有限性、映射数量由prepare程序断言，详见data_audit。

## 历史实验复核

1：P1仅小幅超过P0；2：对比状态文本未稳定提高；3：固定/替换gate、语义投影诊断显示内容作用弱；4：新增ASD特定FC文本无稳定收益；5：取消ReLU/减衰减的MLP调整未优于P1；6：方向标量门控无稳定收益；7：双主题特征残差未证明正确主题优于容量控制；8：疾病Shared支路与原预测高度冗余；9：E9表现较好(ACC91.8826、Prob-AUC93.8227)，语言内容独立作用未证明；10：真实疾病文本相对中性优势很小；11：256FC列核实252具名、4未知，不能给潜在Token强加ROI名；12：ROI语义可实现但小规模试验不足；13：患者FC提示未稳定超过基线；14：层级文本投影趋同且敏感性弱；15：语义通路在强衰减下退化，反事实变化接近0；16：恢复通路有文本敏感性，但残差约32.33%、ACC下降。报告/真实逐折CSV与关键patch均在创建本模块前审计，全部既有文件哈希作为只读保护。

另读取`/data/adodas/MMGL-main/experiment`的alignment_validation、exp2_selective_alignment、exp3_disentangled_alignment、exp4_pairadaptive及exp4_legacy。最早严格基线ACC84.62，统一hybrid对齐86.34；后三轮严格3seed中selective-medium84.69未稳定超过uniform-strong84.88，disentangled-medium84.96但强对齐下降，pair-adaptive BA略增而Prob-AUC不增、权重近均匀。MMGL验证划分/训练协议不同，不是本轮性能基线。该目录无独立exp5，不能虚构“实验五MMGL”记录。

## 文献方法与本研究边界

- [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)：关系图、CAM和语义门控/残差；本轮以实际P0源码为不可变主路径，不声称补丁复现所有论文公式。
- [MGDR](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)：关系图表示、共性空间分解与特异表示/融合；本轮不是SVD或多模态特异注意力重建。
- [BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)：真实ROI层、个体文本建图、疾病文本对齐；我们的年龄/性别/QC与潜在Token没有同等症状及ROI身份。
- [PhenoBrain](https://raw.githubusercontent.com/mlresearch/v306/main/assets/meng26f/meng26f.pdf)：症状表型条件的低秩路由、多跳脑连接与表示学习；其症状表型不等于本项目人口学背景。
- [To Align or Not to Align](https://arxiv.org/pdf/2511.12121)：对齐强度应随冗余/特异信息变化，并通过可控信息分布验证；本轮匹配分数只是代理，不称精确互信息。
- [Class-Aware Semantic Decoupling](https://papers.miccai.org/miccai-2026/paper/1452_paper.pdf)：类别共享/特异、语义权重、实例和分布对齐与类感知融合；这些通用组件已有。本轮用全局32维Shared/Private、小型训练类别原型损失和可拒绝门控，研究不同输入角色，不等同于其类别特定分解。
- [CALM](https://arxiv.org/pdf/2607.01656)：非配对影像/遗传的类别对齐和独立配对验证；本项目没有遗传或独立配对验证，不能借用其证据。
- [CR-mCCAR](https://pmc.ncbi.nlm.nih.gov/articles/PMC12869378/)：联合优化跨模态典型相关、临床参考相关并压低协变量相关；本轮QC条件不等于混杂回归或因果校正。
- [SABER](https://arxiv.org/pdf/2607.01901)：实际ROI语义、多尺度脑图、个体条件残差与疾病对齐；本轮不加ROI图、复杂超图或新文本分类损失。

以上是文献已有方法；“CORR与年龄/性别有可对齐诊断信息”“QC有助于选择或拒绝”是本轮待检验假设，允许负结果。新贡献候选是明确数据角色、患者特异可拒绝决策与保留原判别路径的联合证据，不把Common/Private或类原型对齐单独称首创。

## 固定数学方案

所有辅助编码器Linear→GELU→LayerNorm；CORR256与PHENO28各有32维Shared和Private，QC16→16。冻CLIP RN50每条文本先归一化，每类4句平均再归一化，得2×1024原型；原P0八句不变。真实/中性文本逐字复用历史9/10，UNRELATED骨关节炎/2型糖尿病文本仅是背景控制；具体句子、长度、token数和编码哈希可核查prompts.json/manifest。

Shared单位向量经无bias的32→1024查询映射，与ASD/TD原型取cosine得到4个分数。Gate输入29维：Shared cosine、平滑单位表示差、两个范数/均值/方差(8维)+文本分数4维+QC16维+QC proxy1维；Gate Linear29→16→2+softmax生成Align/Skip。关闭文本时分数0，关闭QC时embedding0且proxy0.5，统一对齐组Align=1。

训练原型由本折train_idx按ASD/TD分组、用detach(Align)加权求均值后归一化；L_align是跨模态同类原型平方距离平均，D2另加Private原型距离，D11平均四模态六对距离。无效类别/极小分母妥善处理；forward不接受标签。Shared/Private去冗余为mean(relu(abs(cos)−0.5)^2)，统一系数0.01，不要求硬正交。L_total=L_original+0.1 L_align+0.01 L_private。D1的L_align=0，仍保留相同去冗余，故它是统一正则下容量控制，不能称完全无辅助正则。

融合输入[Align×s_corr,p_corr,Align×s_pheno,p_pheno,Align]共129维→64→72；输出两个36维方向，对PHENO位置0和CORR位置3增强。每个方向d归一化为d/sqrt(sum(d²)+1e-8)，delta=0.1*tanh(alpha)*detach(||base||)*direction。alpha为2个全零参数，故初始前向与P0逐值相同，每人每目标位置相对L2≤10%；这是预注册工程上限不是最优值。注入后完整执行原SPGS/RES及分类器，ANAT/FUNC位置不注入。

新增模块在原PLN完全注册并执行原my_weight_init之后，在fork_rng中统一seed=170000+seed创建；原网络/CPU/CUDA随机轨迹不变，无新dropout。原参数Adam lr0.0038/WD0.11；新增参数同lr/WD0固定。D8的两个随机单位类别向量用独立局部Generator初始化，不包含CLIP先验，多2048个可学习参数。

## 十二组与容量审计

|组别|名称|对齐|文本|选择性|QC|Private融合|新增参数|
|---|---|---|---|---|---|---|---:|
|D0|原始P0|none|OFF|False|False|True|0|
|D1|容量控制|none|OFF|False|False|True|98637|
|D2|Shared+Private强制对齐|shared_private|OFF|False|False|True|98637|
|D3|仅Shared统一对齐|shared|OFF|False|False|True|98637|
|D4|特征条件Align/Skip|shared|OFF|True|False|True|98637|
|D5|完整DS3A-v2|shared|REAL|True|True|True|98637|
|D6|中性文本|shared|NEUTRAL|True|True|True|98637|
|D7|无关疾病文本|shared|UNRELATED|True|True|True|98637|
|D8|可学习类别向量|shared|LEARNED|True|True|True|100685|
|D9|关闭QC条件|shared|REAL|True|False|True|98637|
|D10|不融合新增Private|shared|REAL|True|True|False|98637|
|D11|四模态统一Shared对齐|shared|OFF|False|False|True|98637|

所有D1–D11注册相同基础模块，D8追加2048参数。关闭组件仍进入优化器但可无梯度；D11激活另注册的ANAT/FUNC Shared编码器704参数，仅进入对齐损失。D10 Private仍参与固定去冗余但不进入增强融合；不能解释为删除整个原主干私有信息。D2同时Shared与Private对齐，总对齐项更多；D5−D4同时增加疾病语义和QC，不能独立归因。固定12组无无类别对齐训练对照，unconditional距离仅诊断，类别条件对齐的额外价值待后续独立预注册检验。

## 训练、恢复和复核

871人原始transductive四图；StratifiedKFold(10,shuffle=True,random_state=10)，seeds0/1/2，一次seed后同一进程连续10fold。最多450epoch，wait>200停止，严格测试折ACC提升才保存首次最佳；val_idx=tst_idx，不能宣称独立无泄漏泛化。既有全局预处理/RFE不重算，原GFR/CAM/SPGS/RES/分类头/损失和全部超参数保留。

```bash
python experiment/experiment17_disease_semantic_selective_alignment/run.py prepare
python experiment/experiment17_disease_semantic_selective_alignment/run.py all
python experiment/experiment17_disease_semantic_selective_alignment/run.py resume
python experiment/experiment17_disease_semantic_selective_alignment/run.py refresh
```

运行器以绝对目录定位，复制六个根文件至`/tmp/spromptgl17-state-<pathhash>/sources`，仅应用integration_patch.diff，CLIP/data用只读访问链接。不会import过去实验runner。临时源码清理后可按同一patch重建。调度器/任务独占锁、36个固定config×seed任务，两卡各18任务且每卡最多4并发；每worker两CPU线程。每25epoch及fold边界原子保存model/两个Adam/RNG/best-model/wait/已提交指标，带checksum；失败从完整checkpoint恢复，完成task跳过，任务完成文件签名隔离，汇总不混入未提交fold。完成后清理临时权重和源码，只保留日志、三份CSV、中文报告。

预检比较原始源码与D0两Adam步的权重/输出/RNG；3seeds×12初始化检查原路径与P0输出、参数、RNG相同；检查两步梯度、文本/QC/Skip/Private干预、训练标签隔离、站点辅助路径不变、10%界限、完整checkpoint第三步复演。仅为实现验证，不选模型或超参数。D0训练后的30fold与历史13/14/15/16六指标、初始化、14–16RNG与最佳epoch逐项比对；不同历史哈希格式都保留对应序列化。

D5每个最佳checkpoint做REAL/NEUTRAL/UNRELATED/SWAPPED/SKIP/NO_QC/NO_PRIVATE/RESIDUAL_OFF共240行。不重训、不选epoch，保持主干与注入前Token不变，最终恢复REAL；概率来自exp(log_softmax)。完整诊断每fold一行；Ridge probe仅用训练折拟合，作为训练后描述，不参与训练或选模。公开报告不保存大规模患者预测或永久权重。所有源码、原数据、CLIP、Prompt和历史文件SHA256以manifest为准。

## 注册容量与有效计算通路的区别

相同注册参数不等于相同有效参数。下表按配置开关审计结构连通性，排除永久零输入的Linear权重列；不把一次梯度为0等同于整个参数永久失效。训练实测梯度另见每折diagnostics.csv，component非零梯度计数只是上界。

|组别|注册新增参数|结构连通参数|
|---|---:|---:|
|D1|98637|31562|
|D2|98637|31562|
|D3|98637|31562|
|D4|98637|31756|
|D5|98637|97933|
|D6|98637|97933|
|D7|98637|97933|
|D8|100685|99981|
|D9|98637|97356|
|D10|98637|93837|
|D11|98637|32266|

D1关闭文本查询与选择门控，D5激活这些通路，因此D5−D1只能作为所设计容量控制的比较，不是严格相同有效参数量的因果估计。D6/D7与D5的结构连通数相同；D8多2048且文本几何不同。D10的Private仍可能从固定去冗余损失学习；D11新增704个实际使用的QC Shared参数。

完整训练结束后all/resume会校验并跳过。正式全量重跑需先备份本目录结果和manifest，保留审计、配置、哈希字段，将run_status改为PREPARED、preflight改为null、completed_folds设0、tasks状态和folds重置，并移除started_unix/started_utc及完成字段，然后重新all；运行器从prompts重新生成临时CLIP缓存与源码，无需原临时权重。该操作只用于用户明确要重跑时，不由本次任务自动执行。

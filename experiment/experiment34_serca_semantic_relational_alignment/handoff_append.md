
## Experiment 34 — SeRCA Semantic-guided Relational Complementary Alignment — 2026-10-22

- 状态：seed=0，10组×10折，100/100折完成；预注册判定 `NO-GO`。本轮只新增一个模块 SeRCA，不增加分类头、损失或患者图。
- 研究问题：检验冻结GPT-4疾病功能网络概念能否在保留GFR原有患者关系和主传播路径的前提下，形成不可由普通FC数值关系、随机概念或普通额外传播解释的跨关系互补分类信息。
- 文献/代码审计：阅读SPromptGL MICCAI 2025论文与作者仓库，并以当前仓库运行代码确定实现行为。四个GraphLearn结果是871名受试者间关系图；GraphLearn返回原始阈值图，GFR通过 `normalize_adj(A+I)` 传播完整320维拼接输入，保留 `i==1` 覆盖 `i==0` 的行为。PLN按原顺序构造，B0-A绕过PEN前向但保留其原构造及CLIP缓存访问；simple-2反传为分类损失加四图损失均值，ALoss不参与反传；CAM的4×36是混合潜在位置而非真实ROI。论文完整版ABIDE报告92.30%，不是本地B0-A复现值。参考：[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[作者仓库](https://github.com/wanxixi11/SPromptGL)、[当前仓库](https://github.com/tangaii/SPromptGL)。
- 历史依据：复用实验11核验的256维FC身份与实验19冻结的疾病功能概念/解剖映射。四个可信概念支持数为D1社会/颞叶-颞顶72条、D2额顶/执行56条、D3丘脑皮层/感觉运动45条、D4扣带-岛叶/显著性32条；两两重叠依次为14、8、7、4、5、2。未可靠命名的FC索引195、225、232、236保留于主干，不进入概念。概念连接只表示候选测量范围，不代表确诊病灶或统一异常方向。实验12–18、20–31的对齐、概念图、关系校准、CLIP原型及QC尝试提供了本轮必须设置的失败对照；尤其实验32正确Prompt未稳定胜过错误Prompt，实验33错误配对/FC-only不能由概念权重自动排除。
- 方法：用外层训练样本估计带符号FC的概念尺度、距离中位数 `tau_k` 和相似度中心；全871人按原transductive协议形成固定 `S^k_ij=exp(-||Z_i^k-Z_j^k||²/tau_k)`。对同一次GFR前向返回的原图 `A_m`，构造 `W_mk=diag(A_m)+offdiag(A_m)⊙exp(S^k-center_k)`，再用仓库归一化得到 `B_mk=normalize_adj(W_mk+I)`。在PLN InputLayer后取 `H∈R^(871×4×36)`，计算差分 `(B_mk-B_m)@H[:,t,:]`，由低容量全局softmax关系/概念权重汇总；最后以零初始化 `alpha_t` 加入有界残差 `0.10*tanh(alpha_t)`，逐患者/Token残差封顶为主干范数10%。不产生新边、不新建GraphLearn、不改变GFR主路径或图损失。
- 10组：S0 B0-A；S1全256维数值关系；S2支持数/重叠匹配的随机252条连接身份；S3仅D1；S4仅CORR关系源；S5仅PHENO/ANAT/FUNC关系源；S6均匀固定关系权重；S7直接条件消息而不减普通消息；S8按站点双射置乱患者概念对应；S9注册的完整SeRCA。S6冻结theta，只有4个可训练alpha；其余组参数差异按预注册定义记录于manifest。
- 训练协议：ABIDE 871人，ASD=403且为SEN正类，TD=468；原始实验32预处理/RFE/输入、10折 `StratifiedKFold(shuffle=True, random_state=10)`、seed=0、450 epoch、early=200、lr=.0038、原weight decay=.11、dropout=.35、hidden=18、head=2、1层、weighted-cosine/GCN/concat/simple-2、双Adam及 `val_idx=test_idx` 选模均保持。ACC严格增加才更新最佳Checkpoint，同分保留首次；AUC为原代码hard-label AUC。未加载历史训练Checkpoint或按历史分数对齐随机轨迹。
- B0-A重训核验：S0与实验32及33逐折初始主干参数哈希10/10相同、构造前RNG哈希10/10相同、best epoch 10/10相同、ACC/AUC/SEN/SPE 40/40数值相同。其他九组对每一折恢复S0保存的Python、NumPy、PyTorch CPU及CUDA入口RNG；10个fold内，10组的入口RNG哈希和原始主干初始化哈希均逐折一致。S0保持自然连续十折RNG；不声称S1–S9有自然的跨折连续RNG流。每个Worker的CLIP模型加载1次、原缓存特征访问10次（每折一次），SeRCA新增CLIP文本编码调用为0，PEN原构造保留且其前向绕过。
- 工程验收：双GPU均通过预检。alpha=0时GFR/CAM/OutputLayer关键张量、四原图与归一化图、两类损失、旧参数梯度及两个Adam首次更新均逐位等价；常量语义图退化回原关系、未生成额外边、有效差分消息非零、原GraphLearn梯度保留。alpha起始梯度非零，alpha更新后theta梯度非零；新增参数注册一次。原子Checkpoint恢复后下一次更新逐位相同并含双Adam/RNG/早停状态。正式完成后移除epoch Checkpoint和派生NPZ缓存，保留10份入口RNG快照、隔离源码、日志、预检证据、逐折结果及诊断。
- 双卡工作量：GPU0执行S0/S2/S4/S6/S8共50折，worker时间2331.1秒、GPU并行跨度924.4秒；GPU1执行S1/S3/S5/S7/S9共50折，worker时间2433.4秒、跨度969.7秒。正式调度总跨度1377.8秒（22分57.8秒），累积worker时间4764.6秒。每卡最多4个Worker。

| 组 | ACC % | Hard-AUC % | SEN-ASD % | SPE-TD % |
|---|---:|---:|---:|---:|
| S0_B0A | 92.0742±2.6000 | 91.8546±2.5815 | 89.0793±3.7283 | 94.6300±4.0457 |
| S1_NUMERIC_GLOBAL | 92.0742±2.6503 | 91.8972±2.6233 | 89.5854±3.4710 | 94.2091±3.8720 |
| S2_RANDOM_CONCEPTS | 91.6157±2.8704 | 91.4377±2.7959 | 89.3232±3.5531 | 93.5523±4.9698 |
| S3_SINGLE_CONCEPT | 91.7294±2.9505 | 91.5066±2.9408 | 88.5915±4.3332 | 94.4218±4.3370 |
| S4_CORR_REL_ONLY | 91.9592±2.5772 | 91.7529±2.5794 | 89.0793±3.5652 | 94.4265±3.3763 |
| S5_AUX_REL_ONLY | 91.5008±3.0122 | 91.3421±2.9315 | 89.3354±3.3396 | 93.3488±4.8764 |
| S6_UNIFORM_RELATION | 91.3845±3.0154 | 91.2433±3.0129 | 89.3415±3.2911 | 93.1452±3.1907 |
| S7_DIRECT_MESSAGE | 91.2696±2.8788 | 91.0230±2.8346 | 87.8415±4.1970 | 94.2044±4.7581 |
| S8_PATIENT_PERMUTED | 92.0729±3.2341 | 91.8870±3.2237 | 89.3476±3.7841 | 94.4265±3.6345 |
| S9_SERCA_FULL | 91.9592±2.4726 | 91.7722±2.4317 | 89.3354±2.7208 | 94.2091±3.5038 |

配对差值均为同折S9−对照的平均百分点；W/T/L为S9在ACC/Hard-AUC上的逐折胜/平/负数：

| 对照 | ΔACC | ΔAUC | ΔSEN | ΔSPE | ACC W/T/L | AUC W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| S0_B0A | -0.1149 | -0.0824 | +0.2561 | -0.4209 | 1/7/2 | 2/4/4 |
| S1_NUMERIC_GLOBAL | -0.1149 | -0.1250 | -0.2500 | +0.0000 | 1/7/2 | 2/6/2 |
| S2_RANDOM_CONCEPTS | +0.3435 | +0.3345 | +0.0122 | +0.6568 | 5/3/2 | 5/3/2 |
| S3_SINGLE_CONCEPT | +0.2299 | +0.2656 | +0.7439 | -0.2128 | 4/4/2 | 5/3/2 |
| S4_CORR_REL_ONLY | +0.0000 | +0.0194 | +0.2561 | -0.2174 | 3/4/3 | 4/2/4 |
| S5_AUX_REL_ONLY | +0.4585 | +0.4302 | +0.0000 | +0.8603 | 5/3/2 | 6/2/2 |
| S6_UNIFORM_RELATION | +0.5747 | +0.5289 | -0.0061 | +1.0638 | 5/4/1 | 6/1/3 |
| S7_DIRECT_MESSAGE | +0.6897 | +0.7493 | +1.4939 | +0.0046 | 7/2/1 | 7/2/1 |
| S8_PATIENT_PERMUTED | -0.1136 | -0.1148 | -0.0122 | -0.2174 | 3/3/4 | 3/2/5 |

- 机制结果：S0四图(PHENO/ANAT/FUNC/CORR)平均非对角边数166448/287937/135361/81194，密度21.97%/38.00%/17.86%/10.71%。S9每个原有支持边均被重加权，但没有新建边；归一化非对角权重质量平均改变10.02%/9.99%/9.56%/6.54%。真实概念相似度标准差0.1518/0.1552/0.1641/0.1583，S2随机概念为0.1397/0.1459/0.1447/0.1520，均非退化且S2扰动幅度相近。S9按目标通道alpha均值为0.3250/0.1730/0.0218/-0.2438；关系权重均值0.2624/0.2475/0.2545/0.2356，近似均匀；各源加权消息范数均非零，但残差/输入均值仅0.0561%/0.0268%/0.0250%/0.0338%，最大观测0.7099%（封顶10%）。S8站点内置换十折均为双射且无单例站点，平均改变97.78%患者身份，但S9没有超过S8。
- 科研结论：S9较S0 ACC -0.1149pp、Hard-AUC -0.0824pp、SEN +0.2561pp、SPE -0.4209pp；未达到ACC提升0.23pp，也未同时胜过S0。正确概念虽高于随机S2，却低于普通256D数值S1及患者错配S8；S9与只用CORR的S4 ACC打平、AUC仅+0.0194pp，不能证明跨关系互补；S9优于直接消息S7但仍不足以支持语义必要性。结论为 `NO-GO`，不支持“正确医学语义不可替代”或“跨模态关系互补已验证”，也不应标为NUMERIC_GAIN_ONLY（S9没有高于基线）。单seed筛选并非显著性检验，不宣称SOTA。
- 解释限制：遵循指定历史协议意味着 `val=test` 选Checkpoint、CV前RFE/预处理及全871人transductive图；这些分数不能作为无偏独立测试泛化证据。未来若继续，需先改进可证伪的语义设计和独立站点验证计划，不应依据本轮测试结果调参、换概念或追分。
- 交付目录：`experiment/experiment34_serca_semantic_relational_alignment`；源码 `serca.py`、`integration.py`、`integration.patch`、`run.py`；协议/概念及证据见 `manifest.json`、`concepts.json`、`README.md`；逐折正式指标与配对结果见 `results/fold_metrics.csv`、机制诊断见 `results/diagnostics.csv`、完整分析见 `results/report.md`；`.runtime/`保留隔离源码、入口RNG快照、训练日志和双卡预检。没有修改根模型、数据或实验1–33，没有安装框架、Git commit或push。
- 存储说明：本轮完整产物先写入 `/tmp/SPromptGL-E34-SeRCA-stage-20261022/experiment/experiment34_serca_semantic_relational_alignment`。`/data` 挂载为可读的 `ossfs`，但本轮先前对目标目录的创建/写入返回 `Input/output error`。根 `handoff.md` 当前305210字节前缀SHA-256为 `8039c875fbb00794fef975489a4f80a1682c2b9c4b641a65434214368ffce8a6`；追加文本在 `handoff_append.md`，待目标挂载可写时安全追加并验证前缀。
- GitHub交付：由于`/data`目录写入仍失败，本实验目录（排除`.runtime`）已通过独立临时克隆基于远端`main`提交并推送至`tangaii/SPromptGL`的`main`。本次仅包含上述11个实验文件；没有提交工作区其他变更。根目录`handoff.md`追加仍受OSSFS I/O故障阻止。

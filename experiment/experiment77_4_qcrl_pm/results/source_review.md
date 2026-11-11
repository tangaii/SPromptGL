# E77.4：论文、真实实现与继承链审计

## 阅读范围与证据边界

已阅读 [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) 全11页正文、图表、结论和参考文献，以及 [官方仓库](https://github.com/wanxixi11/SPromptGL) 的 network.py/model.py；远端 main 树 SHA 为 `ba3b9446fa451fde6864d6555bdf324de0c4b64a`。本地 PDF SHA 为 `219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89`。

原文针对跨模态互补和局部语义选择提出 GFR、CAM、PEN。GFR 每模态构造患者图，传播的是拼接后320维特征；官方实现为线性投影余弦、detach阈值mask、加自环/归一化和两层GCN。CAM 是患者内部四模态之间的信息交互；实际输入槽位维度不等，经 VariLengthInputLayer、四层Encode/FeedForward后在Outputlayer前形成144维。PEN将两组各4条RN50文本拼成4096维、MLP映射9维，对4×4个9维潜在局部块按文本相似性加权，再嵌入全局表示。这些块不具有已核实的解剖ROI标签。论文Eq1含监督图约束，而官方simple-2实际计算但没有把返回的ALoss加入反传总损失；原始优化顺序是GFR Adam再CAM Adam。硬预测one-hot的AUC实际上是二分类BA，不等于连续概率ROC-AUC。val_idx和tst_idx都来自test_index，严格ACC提升选模；因此不是独立测试估计。

原文Fig2控制GFR/RES/SPGS，支持这些组件在其协议下的总体作用；没有匿名医学文本、Query隔离或第三患者混淆关系对照。Fig3文字也承认TADPOLE某模态PEN下降。不能用其平均提升证明E77.4有效。

已阅读 [BrainPrompt论文](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf) 全11页及 [官方BrainPromptG/C分类器](https://github.com/AngusMonroe/BrainPrompt)，仓库树 `c85304a023636a2701d465e4cb0acb13b637127e`；PDF SHA `af251b7821106fa0d17346249e5b1e00c4254d0d9614c4f33bbda7239bc69c54`。它用ROI身份、人口属性和疾病类别三层文本，冻结LLM编码、按人口/图相似性构图，以及图/类别文本余弦交叉熵；不是持久混淆配对的第三患者监督。表3中单独疾病Prompt在ADNI反而下降，不能解释为文本在任何场景都增益。其8:1:1和不同样本数与本实验协议不同，不直接比较绝对ACC。公开C分类器还存在未定义hg引用，未将该代码作为可直接运行的本基座。

已读 [SABER全HTML](https://arxiv.org/html/2607.01901v1) 方法、表格和局限：Graph-CLS+图表示作为文本Query、患者条件检索和门控残差值得借鉴；其多尺度超图、额外Transformer和患者文本不引入本实验。其表I/II的完整语义组ABIDE ACC高于lite，但AUC低于lite；并未证明语义对所有指标均有正向贡献。E77.4只使用已有8条疾病文本，不虚构患者临床叙述。

[HP²L出版商摘要及公开预览](https://www.sciencedirect.com/science/article/pii/S1361841526001313) 可访问，全文403；未精读全文。可确认是层级ViT、优化Prompt与疾病原型细化，面向多层级多病种MRI分类。不能据摘要推断其完整训练损失、数据隔离或第三患者混淆关系设计。上述论文均不是QCRL-PM有效的证据。

## 真实历史代码

已核查E74.4的run.py、soft_positive_dcr.py、audit.py、config/manifest、逐折任务、最佳/latest检查点和结果。唯一基座仍是 `/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr`，真实CAM类 `dugsa.PLNNoPEN`，没有活动PEN。raw CORR256来自同一处理CSV的列64:320，模态字典顺序和871行严格比对；标签CSV减1后ASD=0/TD=1。原数据变换不重做，也不把256个特征解释为256个有解剖标签脑区。

E77的model.py/memory.py/run.py/audit.py/报告中，类别/FP/FN检索通过共享小残差回到CAM，固定错误来源方向只是弱证据。E77.1使用持续历史错误、leave-one-out中心、语义相似度和类别分数修正；其困难值ReLU与排序ReLU易截断。E77.2只读审计发现历史错误复现性，但CAM条件增量不稳定，不能把ReLU定性为唯一主因。

E77.3实际已迁移到工作区 `experiment/experiment77_3_ds_cbpm`；用户所给 `/model` 路径本次不存在。其manifest中训练源码/文本SHA与当前全部相符。已读ds_cbpm.py、memory.py、run.py、audit.py和完整结果：signed_pair_scores比较raw CORR投影余弦差；losses将配对端点展平为查询，再用自身配对列监督，存在自身匹配捷径。OutputLayer后Hook返回最终log_softmax，原分类损失直接经过新分支反传；新增aux与base合并并共用公共优化器追加组，引起共同适应。update_errors_and_pairs按历史患者重建全部配对，没有稳定pair_id和配对EMA。典型分支在四个模态循环中重复加入同一raw CORR相似度。G2自身基础头约85.1789%，不能拿即时去掉分支的损失当作相对历史基座92.1891%的增量。

本次全部取消端点排序监督。第三患者关系例：q不在(a,t)中，若投影余弦为0.7与0.6，tau=.2，则e=tanh(.5)=.4621；真实ASD的关系损失softplus(-.4621/.5)=.3341，TD则为1.2583。q的特征变化可以改变符号，FP/FN来源不能决定符号。与端点自比cos=1不同，该任务要把支持关系迁移到不同患者。独立Query标签只在损失入口使用。

## 路径与随机继承

635项历史来源与产物SHA核验通过，见protected_before.json。E74.4源码SHA：run.py `cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd`；soft_positive_dcr.py `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`。搬迁导致多个历史脚本的/model绝对路径失效；path_adapter只适配导入路径及模块Path全局变量，执行原文件原始字节，配置/源SHA字典不改。E51/E43从经过SHA核对的E72 vendor封存加载。没有改历史文件或创建软链接。

完整E61 epoch10 warmup → E69.3 Fold2添加历史两行 → E74.1 adaptive gate确定性初始化 → E74.4原forward/损失/优化器。这条链的模型、优化器、原4096个患者对/顺序、Python/NumPy/CPU/CUDA RNG和训练配置在10折均通过。新增模块在局部CPU/CUDA fork_rng中初始化，内部三路划分用独立NumPy Generator；正式训练不逐epoch重置seed。新模块独立AdamW，基座先按历史顺序执行一次训练/一次eval，新增损失不进入公共autograd图。

## 预检结论

最初预检发现历史指标函数需要CPU NumPy索引，已在正式训练前修复接口转换，记录于preflight_repairs.json，影响正式任务0。全部正确性预检通过：公共单步forward/backward/optimizer完全等价；新增公共梯度严格0；第三患者关系/分类对Key、CORR Value、语义及门控梯度非零；符号、空库、持久配对/EMA/过期、Query标签不影响Support库、测试排除和完整断点下一步重放通过。没有按预检测试ACC调参。10折G1/G2完整15-epoch parity在正式训练中核验。

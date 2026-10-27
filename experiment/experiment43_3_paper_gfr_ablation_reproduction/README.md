# Experiment 43.3 — Paper GFR ablation reproduction audit

研究问题：原论文ABIDE CAM→GFR增益5.25pp能否在官方Shell和唯一论文文本代理配置下、两个初始化基座内被复现？本轮无创新模块。

来源：[原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)；[官方固定commit](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a)；用户Git HEAD及逐文件SHA见manifest。PDF缓存实际查阅：experiment43_corr_anchored_gfr_audit/.runtime/paper.pdf。Figure2第7页ABIDE标注(a)86.48/86.40、(b)91.73/91.54、(c)92.08/91.93、(d)92.30/92.22，分别ACC/AUC%。全部为原图标签，不是实测。Sec3.3 baseline仅CAM；官方公开树没有对应消融代码、构造顺序、种子和完整配置。PAPER_ABLATION_IMPLEMENTATION_UNAVAILABLE。

|原文/机制|官方代码|用户/历史|处理|
|---|---|---|---|
|Eq1包含标签监督邻接|model.py:130–147计算ALoss但只cls+mean graph|model.py逐字相同|不加入ALoss|
|GFR四关系融合|network.GFR i==1覆盖i==0|AST逐值相同|不修复|
|图卷积Eq2|四分支均传播全部320维，normalize_adj(A+I)|保持硬阈值0.9与原归一化|不改GFR|
|CAM Eq4|layers.py InputLayer/Encoder/FF/OutputLayer|字节一致|不改|
|PEN初始化|官方PLN先注册完整RN50再MLP；随后apply遍历其Linear|用户先MLP再缓存8向量，不注册RN50|A严格沿用历史用户构造快照；B物理不构造PEN/CLIP|
|论文alpha1|main alpha默认.5仅存储；simple-2实际加原输入系数1|P记录alpha1但该路径无数值作用|不改代码|

|字段|H官方Shell|P论文文本代理|映射状态|
|---|---|---|---|
|dropout|0.35|0.75|HIGH: main.py parser; layers.Attention/VariLengthInputLayer/EncodeLayer/FeedForwardLayer|
|lr|0.0038|0.0023|HIGH: model.py Adam lr=hyperpm.lr|
|n_hidden|18|10|LOW: Paper natural language does not define code width; unique executable proxy only|
|n_head|2|4|HIGH: network.PLN n_head and layer reshape|
|nlayer|1|4|MEDIUM: Paper does not explicitly name argparse variable|
|alpha|0.5|1.0|LOW: No functional mapping from hyperpm.alpha to Eq4/GFR residual; effective raw residual coefficient already1|
|reg|0.11|0.11|INHERITED_UNDISCLOSED: Shell --reg .11; Adam weight_decay|
|th|0.9|0.9|INHERITED_UNDISCLOSED: main default .9; GraphLearn score>.9 detached mask|
|nepoch|450|450|INHERITED_UNDISCLOSED: main default450|
|early|200|200|INHERITED_UNDISCLOSED: main wait_cnt>hyperpm.early|
|loss|simple-2|simple-2|CODE_DIFFERS_FROM_PAPER: model.py loss=cls_loss+all_graph_loss; ALoss absent|
|optimizer|two Adam|two Adam|INHERITED_UNDISCLOSED: model EvalHelper optimizer_MP0/optimizer_MF|
|seed|0|0|INHERITED_UNDISCLOSED: main default0; historical continuous RNG|
|data|same historical preprocessed|same|HIGH: Private ablation input hash unavailable|
|evaluation|val=test strict first ACC; hard onehot AUC|same|PAPER_UNDISCLOSED_DETAIL: Paper no seed/init/fold-index/checkpoint full detail|

B0-A保留完整原PLN/PEN无用MLP、avg_pool及8旧Prompt buffer，forward从CAM直接原OutputLayer；B0-B物理不构造MLP/avg_pool/CLIP或Prompt缓存。H物理PLN49,544、GFR887,836；A/B构造随机消费不同，绝不声称跨基座参数匹配。P n_hidden10映射为每头QKV维，head4则CAM4×40→160；encoder4是四个Encoder/FF对。P不是作者私有消融精确配置。

历史实际查阅：E32 integration.py/run.py/manifest/results；E43 .runtime/anchors与E00 best；E43.1与E43.2源码、配置、报告、独立验收、未训练入口、best/预测；b0_b_seed0_full10fold及物理删除等价审计；/tmp/SPromptGL-E38-B0A-stage/...与/tmp/SPromptGL-E40-HCER-v2-stage/...报告；E42报告。缺失根目录E38/40路径不冒称已读，实际/tmp路径见manifest的history_reads。

H-A-G复用已精确重放E32的E43 E00；H-B-G/R复用E43.2 E00/E01。每折本轮重新加载best重算预测/四指标，核验来源SHA/分割/入口/初始状态。H-A-R从相同未训练A快照训练；P四组新训。从未加载训练后G模型为R初始值。

调度：H-A-R folds1–5 GPU0、6–10 GPU1；P-A-G/P-A-R连续10折GPU0，P-B-G/P-B-R连续10折GPU1。每个P G组原生seed0连续10fold并保存未训练入口；R逐fold恢复该入口而非自身自然连续随机流。新增GPU各25折；复用30折不计新训练。最多每GPU1正式worker，本轮无需4worker占满。

预检：同基座初始参数/两Adam/RNG一致；noGFR零调用、零梯度、零更新、原输入恒等；heldout标签改动不影响forward；原simple-2两Adam首步等价；两步保存中断第三步严格复演；P四层维度有限；所有配置指纹唯一。GFR模块forward不含随机Dropout(p=0)，配对G/R共同epochs随机流应相同；仅P可对逐epochtrace检查，H历史没有全部epoch RNG则如实标缺失。

机制：原GFR hook只读捕获四支消息，输出验证为X+sigmoid(f1)msg1+sigmoid(f2)msg2+sigmoid(f3)msg3；对best逐支/全部置零消息，仅诊断，不重选模型。分类和图正则梯度分开算。密度不含自环。数值消息、固定模型依赖、训练增量三层分开。

GFR正增量准入：ΔACC及ΔHard-AUC均≥0.1pp，且两项各≥7/10严格胜。未达则NO-GO。论文数值兼容单独列(a)/(b)绝对差与增量差；不选最低raw/最高GFR跨基座拼接。

仅seed0；SKF(10,shuffle=True,random_state=10)；450epoch wait>200；test=val首次严格ACC提高存best；原全队列标准化、CV前RFE、871节点传导图保留。四指标ACC/硬one-hot ROC-AUC/SPE-TD/SEN-ASD，标签ASD0/TD1，10fold mean±population SD。结果非独立临床泛化，非因果证据。

可复现命令（根目录）：
```bash
python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py anchors
CUDA_VISIBLE_DEVICES=0 python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py reuse --gpu 0
CUDA_VISIBLE_DEVICES=1 python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py reuse --gpu 1
CUDA_VISIBLE_DEVICES=0 python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py preflight --gpu 0
CUDA_VISIBLE_DEVICES=1 python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py preflight --gpu 1
python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py all
python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py resume
python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/run.py report
```

原子写带fsync、rename、SHA读回，四次有界重试ENOENT/EIO/ESTALE；每任务独占锁、源/配置/fold/anchor签名，两Adam完整恢复；完成只在best写入与预测读回通过后提交。独立audit-worker验收每个新best。根代码/数据/历史只读，唯一例外最后安全追加handoff。无commit/push。

补充实际审计路径：物理删除等价审计为experiment/b0_pen_physical_removal_equivalence_audit/report.md、results.json、audit.py（该目录无README或results/report.md）；已读取并记录SHA。原审计在受控同RNG/同参数下证明数学路径等价，但两模型构造后RNG不同，不外推成自然训练轨迹等价。

补充确定证据：官方network.py:96把PEN硬编码为871×4×4×9=125424元素，而论文代理n_hidden10×head4的CAM为871×4×40=139360元素，无法直接执行该PEN reshape。H18×2刚好满足36维。故公开完整PEN与论文正文映射不能同时按当前源码成立；本轮A/B都绕过PEN，P四个CAM/GFR组维度预检均通过，代理协议不因此改变。该差异只证明公开实现/文本不完全一致，不证明作者私有消融的具体配置。

参数语义进一步核实：论文Eq4的CAM残差在layers.VariLengthInputLayer以q += residual固定系数1实现；EvalHelper.alpha仅存储，未传入CAM。GFR的raw-X系数1是另一个独立skip，不应把论文alpha直接等同GFR系数。main.py的n_hidden/n_head帮助字符串互换、nlayer帮助写conv但实际控制PLN Encoder/FF重复；本轮按构造器真实语义映射。

额外audit.py只用于与训练入口分离的最终指标公式/曲线/权重收缩验收，不参与训练签名或方法。正式训练完成后：在实验目录中运行CUDA_VISIBLE_DEVICES=0 python -B audit.py worker --gpu 0，以及CUDA_VISIBLE_DEVICES=1 python -B audit.py worker --gpu 1；python -B run.py report后运行python -B audit.py finalize，补充独立科学验收并安全追加handoff，不修改任何训练结果。

官方ABIDE输入审计：固定作者commit ba3b9446fa451fde6864d6555bdf324de0c4b64a中的processed_standard_data.csv（4,698,784字节）与modal_feat_dict.npy（4,611字节）同本地逐字节/SHA相同。未保存重复数据副本，仅保存.runtime/official/official_input_audit.json。作者实际私有消融输入签名未披露，因此不能外推成已核实其训练用文件，但公开输入不存在已确认差异。用户远端main 45962c81d1ee5728581313af0967fcb9bfd2a4d2的七个核心文件也全部与服务器相同；GitHub API限流后通过一次有界git ls-remote+固定raw URL核实。

补丁谱系：冻结integration.patch为用户历史源→本轮隔离源（训练签名组成，不在运行中修改）；.runtime/official/network.py.diff及utils.py.diff为官方→用户差异；另保留.runtime/official/integration_vs_official.patch直接对比官方固定commit与本轮隔离源。B0-B的物理PLN构造器在integration.py明示，不修改root。

最终报告附表可用`python -B experiment/experiment43_3_paper_gfr_ablation_reproduction/audit.py tables`重建（先执行report及finalize）。该命令只把冻结参数映射和已验收机制CSV渲染为表四/表五补充，保留已执行worker/finalize的函数AST哈希不变；不重新加载模型或训练。每个报告追加块有唯一标记，根handoff仍只追加并核验原前缀。完整交付结果见results/report.md、summary.csv和paired_comparisons.csv。

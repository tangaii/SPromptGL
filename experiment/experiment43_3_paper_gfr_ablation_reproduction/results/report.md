# Experiment 43.3 — 原论文GFR消融专项复现核查

状态：SCIENTIFIC_COMPLETE；实际80/80条唯一折记录。新增50，核验复用30。

正式四指标：seed0十折mean ± population SD，单位%；ACC为折均值，并非全871人加权micro。后者单独列在summary.csv。

|组别|来源|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---|---:|---:|---:|---:|
|H-A-G|REUSED_VERIFIED|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|H-A-R|NEWLY_TRAINED|91.9592 ± 2.9591|91.7651 ± 2.9755|94.4265 ± 3.2394|89.1037 ± 3.9659|
|H-B-G|REUSED_VERIFIED|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|H-B-R|REUSED_VERIFIED|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|
|P-A-G|NEWLY_TRAINED|91.3845 ± 3.0589|91.1587 ± 3.0667|94.2137 ± 3.6134|88.1037 ± 4.0848|
|P-A-R|NEWLY_TRAINED|91.4995 ± 3.2655|91.2893 ± 3.2137|93.9870 ± 4.8354|88.5915 ± 3.8608|
|P-B-G|NEWLY_TRAINED|91.4995 ± 3.0130|91.2806 ± 3.0387|94.2137 ± 3.2156|88.3476 ± 3.9856|
|P-B-R|NEWLY_TRAINED|91.2709 ± 2.9685|91.0686 ± 2.9495|93.7835 ± 3.5496|88.3537 ± 3.2888|

## 原论文与双基座比较

|来源/协议|基座|无GFR ACC|有GFR ACC|ΔACC|无GFR AUC|有GFR AUC|ΔAUC|
|---|---|---:|---:|---:|---:|---:|---:|
|Figure 2图中标注|作者未公开完整消融实现|86.48|91.73|5.25|86.40|91.54|5.14|
|H|B0-A|91.9592|92.0742|+0.1149|91.7651|91.8546|+0.0896|
|H|B0-B|91.7294|91.7294|+0.0000|91.4880|91.6046|+0.1165|
|P|B0-A|91.4995|91.3845|-0.1149|91.2893|91.1587|-0.1306|
|P|B0-B|91.2709|91.4995|+0.2286|91.0686|91.2806|+0.2120|

Figure 2(c) +RES：ACC92.08、AUC91.93；(d) +PEN：ACC92.30、AUC92.22。以上全部来自PDF第7页原图数字标签，非本轮结果；本轮不训练(c)/(d)。正文Sec3.3明确baseline只用CAM。作者公开树中没有Figure2专用消融实现、消融初始化/训练顺序/种子。标记：PAPER_ABLATION_IMPLEMENTATION_UNAVAILABLE。

## 同协议同基座GFR配对

|比较|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|决策|
|---|---:|---:|---:|---:|---|---|---|
|H-A-G − H-A-R|+0.1149|+0.0896|+0.2035|-0.0244|4/4/2|4/4/2|NO-GO|
|H-B-G − H-B-R|+0.0000|+0.1165|-1.4986|+1.7317|3/4/3|4/3/3|NO-GO|
|P-A-G − P-A-R|-0.1149|-0.1306|+0.2266|-0.4878|1/7/2|1/6/3|NO-GO|
|P-B-G − P-B-R|+0.2286|+0.2120|+0.4302|-0.0061|4/3/3|4/3/3|NO-GO|
|H-A-G − H-B-G|+0.3448|+0.2500|+1.4940|-0.9939|4/5/1|5/3/2|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|
|H-A-R − H-B-R|+0.2299|+0.2770|-0.2081|+0.7622|5/3/2|6/2/2|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|
|P-A-G − P-B-G|-0.1149|-0.1220|-0.0000|-0.2439|2/6/2|2/5/3|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|
|P-A-R − P-B-R|+0.2286|+0.2207|+0.2035|+0.2378|3/5/2|3/3/4|DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE|

GFR准入要求：两项增益均≥0.1pp且两项各≥7/10严格胜。A/B差别含构造与初始化随机路径，不能解释为PEN语义收益；H/P同时改变多个参数，不能归因单一参数。训练折相互重叠，这些胜平负不是独立数据集显著性检验。

## 参数映射与确定源码证据

见paper_ablation_audit.json逐项paper_text_original/repo_argument/mapping_evidence/confidence/ambiguity。H为官方Shell运行参数；P=dropout .75、lr .0023、n_hidden10、heads4、nlayer4，明确为PAPER_TEXT_MAPPED_PROXY。原文“number of hidden layers10”与代码n_hidden表示Q/K/V每头宽度存在语义歧义；nlayer实际控制Encoder+FeedForward重复次数。P的alpha记为1以对应原文，但代码该参数仅存储，simple-2中原始特征加法硬编码为1；H默认alpha .5与P1均不改变该路径。reg .11、th .9、450/200及其它未明参数继承Shell/default，不是论文确认。

官方main/model/layers/shell与本地逐字相同。GraphLearn/GCN/GFR的AST逐值一致；均含i==1覆盖PHENO消息。simple-2的ALoss计算而不优化。官方network把RN50作为ViT_model注册在PLN，且在MLP之前加载；用户network先构造MLP再调用一次文本缓存，缓存8条1024维buffer，不注册整个RN50。这改变构造与apply(my_weight_init)所遍历的参数/RNG。官方utils.one_hot在CPU创建eye，本地改到索引device以兼容新PyTorch；数学指标口径相同。上述是确定实现/配置差别，尚不能单独解释原作者私有消融的5.25pp。

B0-A复用E43已逐折精确匹配E32的10个未训练状态、完整RNG与best；本轮又重新加载这些best并计算预测/四指标/混淆。B0-B复用E43.2已独立验收模型并再次重算。新增无GFR组只恢复未训练入口，不加载有GFR best作为初始化。历史A best只保留模型，best时Adam未保存；入口两个Adam空状态完整存在。该历史限制明确保留，不补造缺失状态。

## GFR机制

|有GFR组|输出−输入相对范数(均值)|有效消息/输入ANAT,FUNC,CORR|图密度PHENO,ANAT,FUNC,CORR|关闭全部消息logp绝对变化|类别翻转(871)|
|---|---:|---|---|---:|---:|
|H-A-G|0.00067748303|0.000336662, 0.000193902, 0.000270931|0.2197, 0.3800, 0.1786, 0.1071|0.008307849|7|
|H-B-G|6.9216286e-05|4.43303e-05, 1.58015e-05, 1.70588e-05|0.2146, 0.3784, 0.1709, 0.1299|0.001799584|0|
|P-A-G|5.0197642e-08|3.91979e-08, 1.39013e-08, 9.14792e-09|0.2237, 0.3316, 0.1606, 0.1525|4.275136e-07|0|
|P-B-G|1.5715236e-07|8.78368e-08, 1.02896e-07, 6.62973e-08|0.2276, 0.3320, 0.1719, 0.1460|9.391726e-07|0|

逐分支消息范数、分类GCN梯度、图投影梯度、有效融合、密度/度数及单支关闭logp/翻转见mechanism_diagnostics.csv。PHENO消息虽计算但被覆盖，分类GCN梯度为0；图正则仍优化PHENO GraphLearn。数值有消息、固定模型依赖消息、从头训练带来分类增益是三件事。

## 论文数值兼容性与核心回答

- H/B0-A：CAM-only距86.48为+5.4792pp，GFR距91.73为+0.3442pp，GFR增量距+5.25为-5.1351pp。
- H/B0-B：CAM-only距86.48为+5.2494pp，GFR距91.73为-0.0006pp，GFR增量距+5.25为-5.2500pp。
- P/B0-A：CAM-only距86.48为+5.0195pp，GFR距91.73为-0.3455pp，GFR增量距+5.25为-5.3649pp。
- P/B0-B：CAM-only距86.48为+4.7909pp，GFR距91.73为-0.2305pp，GFR增量距+5.25为-5.0214pp。

数值接近不等于严格重现作者私有消融；没有原始消融实现与完整初始化记录，不能断言论文错误或作者不当行为。B0-A/B0-B只能在各自协议内作配对增量；跨基座最好/最差拼接禁止。根据四个预注册增量与机制审计判断是否值得继续以GFR作创新基座：若证据不足，优先保留它作为历史兼容比较，并以CAM-only为必要强对照，不以论文5.25pp作为已证实前提。

## 训练、恢复与独立验收

- H-A-G：best_epoch=[322, 149, 428, 243, 428, 124, 154, 280, 169, 251]；epochs_executed=[450, 351, 450, 445, 450, 326, 356, 450, 371, 450]；来源REUSED_VERIFIED。
- H-A-R：best_epoch=[172, 124, 259, 154, 333, 139, 204, 389, 147, 203]；epochs_executed=[374, 326, 450, 356, 450, 341, 406, 450, 349, 405]；来源NEWLY_TRAINED。
- H-B-G：best_epoch=[157, 239, 180, 213, 219, 155, 187, 248, 206, 266]；epochs_executed=[359, 441, 382, 415, 421, 357, 389, 450, 408, 450]；来源REUSED_VERIFIED。
- H-B-R：best_epoch=[141, 137, 192, 165, 367, 103, 139, 191, 158, 139]；epochs_executed=[343, 339, 394, 367, 450, 305, 341, 393, 360, 341]；来源REUSED_VERIFIED。
- P-A-G：best_epoch=[186, 182, 376, 218, 247, 167, 157, 388, 229, 256]；epochs_executed=[388, 384, 450, 420, 449, 369, 359, 450, 431, 450]；来源NEWLY_TRAINED。
- P-A-R：best_epoch=[178, 188, 201, 155, 312, 142, 180, 292, 253, 364]；epochs_executed=[380, 390, 403, 357, 450, 344, 382, 450, 450, 450]；来源NEWLY_TRAINED。
- P-B-G：best_epoch=[166, 230, 224, 190, 311, 206, 173, 353, 129, 248]；epochs_executed=[368, 432, 426, 392, 450, 408, 375, 450, 331, 450]；来源NEWLY_TRAINED。
- P-B-R：best_epoch=[147, 206, 227, 357, 245, 286, 190, 313, 318, 162]；epochs_executed=[349, 408, 429, 450, 447, 450, 392, 450, 450, 364]；来源NEWLY_TRAINED。
- 新训GPU0=25折，GPU1=25折；正式墙钟1611.43s；本轮新训fold wall总和3072.64s（并行和I/O，不等于墙钟）。
- 新训每10epochs原子保存model、两Adam、best、当前epoch/wait、四类RNG、源/数据/配置/anchor/fold签名；完成后独立进程重载best重新预测及指标，并检查原输入/no-GFR零调用零梯度零更新。
- loss/训练ACC/testACC轨迹在新训task的loss_trace，10epoch抽样；历史保留的可读轨迹作为历史证据，不虚构未保存训练曲线。strict first best与wait>200均未更改。
- 原始代码/数据/历史文件protected SHA前后核对；无Git commit/push。

## 限制

val=test参与checkpoint选择、全队列标准化及CV前RFE、全871节点transductive协议保持，仅用于内部历史可比；不能视为无泄漏独立临床泛化。单seed与重叠训练折不支持显著性或因果结论。ANAT/FUNC是扫描质量/运动摘要；图是患者图，CAM潜token不是ROI。


<!-- E43_3_INDEPENDENT_SCIENCE -->

## 独立公式终验、训练曲线与弱消息来源

额外audit.py为与冻结训练入口分离的独立验收实现：用布尔混淆计数重算ACC/Hard-AUC/SPE/SEN，跨物理GPU重载全部80个best；检查初始化/预测/最佳模型SHA、no-GFR零调用/零更新、闲置PEN MLP不更新及heldout标签置换不改变forward。80/80通过；没有训练或修改原模型。

|组|最佳epoch均值|执行epoch均值|训练eval ACC|测试eval ACC|训练−测试pp|初始→末次分类loss|最佳图loss/分类loss绝对比|预测坍塌折|
|---|---:|---:|---:|---:|---:|---|---:|---:|
|H-A-G|254.8|409.9|96.0325|92.0742|3.9584|0.96447→0.37106|13.4615|0|
|H-A-R|212.4|390.7|96.6323|91.9592|4.6730|1.01835→0.24767|0.0000|0|
|H-B-G|207.0|407.2|96.7086|91.7294|4.9792|1.15263→0.29365|15.1335|0|
|H-B-R|173.2|363.3|96.4917|91.7294|4.7624|1.14408→0.19865|0.0000|0|
|P-A-G|240.6|415.0|96.2622|91.3845|4.8776|1.45016→0.13771|13.7185|0|
|P-A-R|226.5|405.6|95.8031|91.4995|4.3036|1.45872→0.31015|0.0000|0|
|P-B-G|223.0|408.2|96.1600|91.4995|4.6606|1.40253→0.21171|13.1266|0|
|P-B-R|245.1|418.9|96.6701|91.2709|5.3992|1.42571→0.30648|0.0000|0|

训练eval ACC是最佳checkpoint的诊断，非新增第五项正式指标。训练loss轨迹为train-mode、按原保存频率抽样，初始/末次不是同一dropout mask；不能把此表当作无偏过拟合估计。图loss包含负degree项，因此另列绝对比，不能把负总loss误判数值故障。

|有GFR组|ANAT GCN权重范数(best/init)|FUNC|CORR|GCN图正则梯度|
|---|---:|---:|---:|---:|
|H-A-G|0.008536212|0.005118586|0.005999795|0（逐张量独立核验）|
|H-B-G|0.003129299|0.001938282|0.003398192|0（逐张量独立核验）|
|P-A-G|0.0001685754|0.0001665667|0.0001091897|0（逐张量独立核验）|
|P-B-G|0.0003383632|0.0002986017|0.0002877412|0（逐张量独立核验）|

这些范数收缩和弱消息是实际观察；GFR的GraphConstructLoss只作用于构图投影，不直接监督GCN权重。原weight_decay=.11是候选机制解释，但本轮未新增去正则组，不能将收缩原因单独归因于weight decay。PHENO GCN因覆盖无分类梯度，其消息不进入最终输出。

补充确定源码/数据证据：官方固定commit的预处理CSV及modal_feat_dict.npy与本地逐字节相同。用户远端main七个核心文件也与本地相同。官方PEN硬编码CAM4×36，而论文代理10×4产生4×40，不能直接执行当前公开PEN；本轮两个基座都跳过PEN，因此代理四组维度合法。此证据限制论文配置精确映射，不影响冻结H/P。Eq4的CAM残差在InputLayer硬编码系数1，hyperpm.alpha未传入；GFR raw-X系数1是另一条独立skip。

H明确指公开Shell超参配合用户指定的历史A/B构造，不声称E32缓存式构造等于官方注册整个RN50的原始构造。强行换成后者会破坏本轮要求的E32-B0-A身份，因此仅做源码对照并保留未公开消融实现的不确定性。官方与用户都存在PHENO覆盖及ALoss未优化，不能将这两点说成用户独有改坏。

### A–F结论

A. 公开Shell/用户兼容初始化协议：B0-A ΔACC=+0.1149pp、ΔAUC=+0.0896pp，NO-GO；B0-B ΔACC=+0.0000pp、ΔAUC=+0.1165pp，NO-GO。
B. 论文文本代理协议：B0-A ΔACC=-0.1149pp、ΔAUC=-0.1306pp，NO-GO；B0-B ΔACC=+0.2286pp、ΔAUC=+0.2120pp，NO-GO。

C. 本轮A/B差异中最大ACC绝对差为0.3448pp；该量级不足以解释原论文5.25pp的同基座GFR增量。两者本来不执行PEN，不能归因于疾病文本。

D. CAM-only实测范围91.2709–91.9592%；与86.48最近差距4.7909pp。是否数值接近应看这个差距，不能用不同协议组拼接。

E. 确定差异：正文/Shell超参不一致；代码n_hidden语义与自然语言不一致；官方/用户RN50注册、构造顺序、缓存和初始化遍历不同；公式图监督与实际loss不同；论文未公开消融专用脚本/种子/完整初始化。上述证据并未证明其中某一项单独造成论文差距。评估偏倚与原Torch1.12/RTX3090环境也已披露，不能在未执行反事实配置时断言为根因。

F. 研究建议：继续把GFR作为历史对照可以，但任何以患者图为创新基座的工作都需同时对CAM-only证明增量，并检查实际消息与初始化。若配对四组未达门槛，本轮不给“GFR具有稳定额外分类价值”的GO，也不据此证明全部患者图无效。


H-B历史G/R共同epoch完整四类RNG又逐项核对：3592个全部相同。P-A/P-B共同epoch trace由run.py report逐项核对；H-A历史未保存逐epoch完整trace，不伪称具有该层证据。A/B两基座的入口状态各自匹配，跨基座不作逐参数相同主张。


<!-- E43_3_FINAL_AUDIT_TABLES -->

## 表四：论文原文、公开Shell及唯一代理映射

|字段|论文正文|H Shell/default|P代理|源码含义及证据状态|
|---|---|---|---|---|
|dropout|the dropout rate is 0.75|0.35|0.75|main.py parser; layers.Attention/VariLengthInputLayer/EncodeLayer/FeedForwardLayer; HIGH|
|lr|the learning rate is 0.0023|0.0038|0.0023|model.py Adam lr=hyperpm.lr; HIGH|
|n_hidden|the number of hidden layers is 10|18|10|network.PLN d_q=d_k=d_v=hyperpm.n_hidden; NOT hidden layer count; LOW; ASSUMED_MAPPING|
|n_head|number of attention heads is 4|2|4|network.PLN n_head and layer reshape; HIGH|
|nlayer|the encoding layer is 4|1|4|network.PLN for i in range(self.n_layer), Encoder+FeedForward pair; MEDIUM; ASSUMED_MAPPING|
|alpha|weight of original multi-modal features alpha=1|0.5|1.0|Eq4 CAM residual: layers.VariLengthInputLayer q += residual hardcodes1; EncodeLayer also uses residual1. hyperpm.alpha only stored in EvalHelper, never passed to CAM/GFR simple-2. GFR raw-X residual also independently hardcodes1.; LOW; ASSUMED_MAPPING|
|reg|not disclosed|0.11|0.11|Shell --reg .11; Adam weight_decay; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|th|not disclosed as numeric in implementation detail|0.9|0.9|main default .9; GraphLearn score>.9 detached mask; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|nepoch|not disclosed|450|450|main default450; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|early|not disclosed|200|200|main wait_cnt>hyperpm.early; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|loss|Eq7 lambda=eta=1; Eq1 graph smoothness+supervised adjacency|simple-2|simple-2|model.py loss=cls_loss+all_graph_loss; ALoss absent; CODE_DIFFERS_FROM_PAPER|
|optimizer|not specified in paper|two Adam|two Adam|model EvalHelper optimizer_MP0/optimizer_MF; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|seed|not disclosed|0|0|main default0; historical continuous RNG; INHERITED_UNDISCLOSED; ASSUMED_MAPPING|
|data|ABIDE871 four modalities|same historical preprocessed|same|main loads processed_standard_data.csv & modal_feat_dict.npy; no new preprocessing; HIGH; ASSUMED_MAPPING|
|evaluation|10-fold ACC/AUC/SEN/SPE|val=test strict first ACC; hard onehot AUC|same|main.py split/early loop; model.test/roc_auc_score; PAPER_UNDISCLOSED_DETAIL; ASSUMED_MAPPING|

## 表五补充：有效消息、分类梯度与图正则路径

|有GFR组|输入范数|输出范数|相对差异%|ANAT/FUNC/CORR分类GCN梯度范数|ANAT/FUNC/CORR best/init权重范数|关闭消息翻转|
|---|---:|---:|---:|---|---|---:|
|H-A-G|236.3112|236.2978|0.0677483|0.014883, 0.0151116, 0.0242279|0.00853621, 0.00511859, 0.00599979|7/871|
|H-B-G|236.3112|236.3107|0.006921629|0.0177127, 0.0151626, 0.0127163|0.0031293, 0.00193828, 0.00339819|0/871|
|P-A-G|236.3112|236.3112|5.019764e-06|0.00030141, 0.000228785, 0.000125074|0.000168575, 0.000166567, 0.00010919|0/871|
|P-B-G|236.3112|236.3112|1.571524e-05|0.000337561, 0.000494878, 0.000405342|0.000338363, 0.000298602, 0.000287741|0/871|

原始图的密度、邻居数及各分支关闭效应完整列于mechanism_diagnostics.csv。GraphConstructLoss→GCN参数梯度均为0；该loss优化构图投影，不能据此认为全部GFR参数都受到图正则直接训练。P最佳模型消息接近数值零，但从头训练G/R轨迹仍可能不同；不能把最终弱消息倒推为整个训练从未用图。

终验状态：80/80折，50 NEWLY_TRAINED、30 REUSED_VERIFIED；GPU0/GPU1各25新增折。正式训练与第一轮验收墙钟1611.43s；含论文/代码审计及独立公式终验2608.47s（截至此前独立终验）。本附表只整理既有记录，没有再次训练或评估。


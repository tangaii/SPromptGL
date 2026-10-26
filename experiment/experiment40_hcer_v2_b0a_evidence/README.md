# Experiment 40 — HCER-v2

正式问题：正确的文献疾病假说能否在同一个E32 B0-A上指导真实FC证据选择，并胜过数值异常、匿名、中性、错配和类别交换对照？全部参数与判定规则在`config.json`预注册；不根据测试成绩改变。

工作目录为`/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence`；首次OSSFS创建正式目录返回EIO。原源码、数据与实验39只读。`/tmp`不保证长期保存，训练完成与正式目录交付分别报告。

## 实际历史审计

读取根README、shell、main/model/network/layers/utils；E32 integration/run/manifest/report/fold_metrics；E38严格B0-A重跑的method/lcher_run/config/prompts/manifest/report/diagnostics；E11可信连接映射。实际路径与SHA在`.runtime/source_audit.json`。本地HEAD与GitHub E32文件校验均记录。E38只能在此前`/tmp/SPromptGL-E38-B0A-stage/...`访问，未伪称根目录存在。

[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)的监督邻接项与实际simple-2不同：源码计算ALoss但只优化分类与四图构图损失均值。四张图是871人患者图；每支传播完整320维；i==1覆盖i==0。本次全部保留。CAM的4×36是混合潜变量，原始FC身份仅对应raw_X的64+edge_id列。论文ABIDE参数及92.30%不冒充当前shell与E32的92.0742%。

|历史问题|本次修复|可证伪测试|
|---|---|---|
|E38 E4 ACC/AUC 91.3845/91.1798低于B0-A 92.0742/91.8546，E8/E5/E6分别91.9592/91.7307/91.6157 ACC|只接受新重训且逐折精确的E32锚点|混淆、4指标、best/executed epoch、参数/RNG SHA全10/10|
|E38两项summary绕过Top8|所有支持、未知和value聚合统一乘selector|mask全0时summary、残差逐值0|
|错误映射重排raw却未同步z/mu|所有数值始终按唯一canonical_edges；只改变query membership|E7 raw/z/mu/sigma/mask列身份不变|
|缺少连接身份|共享RN50投影的冻结真实HO双端描述|只交换端点表示而不交换数值会改变证据|
|相同context加到双Query且高cos|Query保留固定pair对比，共享context只进Key|原始/投影cos、query差、文本交换logit响应|
|竞争/非竞争残差相差数量级|共享逐患者范数校准与正gate|E5/E9相同gate下非空mask ratio误差≤1e-5|
|Top8总选8条，无弃权|train初始化分数75%分位阈值+至多8条|可返回0条且严格恢复B0-A|
|训练TD OOF而ASD fullfit|全部outer-train用同一5折过程|固定inner assignment后自己标签变化不改变自身reference fit/预测mu|
|QC绝对偏离不是质量方向|仅官方定义的DVARS、outlier、FD正向不利偏离|列名核对、单调可靠性测试|
|CF可走自身残差捷径|CF训练固定原输入残差并冻结主干，梯度仅走原始FC mask|主干CF梯度不变、mask梯度有限非零；离线分解live/fixed残差|
|只胜随机，未胜Top-abs(z)|E1正式训练及等预算固定模型反事实|selected-delete必须稳定胜过random及Top-abs(z)|

## 冻结方法

原样复制E38 prompts.json，主13组26文本39独立连接；3探索组只归档。文本来自历史文献综合，并非本轮GPT-4 API。网络级依据不证明精确HO单边效应。`canonical_edges`按原CORR索引排序，所有raw/z/mu/sigma/selector/CF严格同序。RN50一次缓存真实、解剖、中性文本；77 token严格检查。匿名bank采用固定带符号坐标置换，保持真实bank的全Gram矩阵与范数。所有人同时使用两类文本。

每outer fold内用固定StratifiedKFold(5,40000+fold)，每个inner held chunk的ASD/TD均由其它chunks的TD Ridge(alpha10)预测。测试只用全部outer-train TD模型；OOF-TD MAD尺度、floor1e-4、clip3。保存所有fit IDs与系数。年龄bin/性别/站点只使用历史PHENO；新增context排除站点，站点只作reference混杂控制。QC仅使用已核验FD/DVARS/outlier，单向有界可靠性≥0.5。

固定共享投影将pair中心化文本与解剖端点映射到64维；Query为`normalize(P(t-pair_mean+0.25*pair_mean))`。Key融合解剖、[raw,z,abs(z),0.25*direction*z,reliability]与低容量CAM/年龄性别context。对每hypothesis、每Query在其合法2–4条边上真正softmax attention。温度限定[0.10,0.50]，初始0.30。ASD、TD和NULL独立竞争，保持两个支持通道与未知/矛盾通道。方向为软先验而非标签真值。

选择按分数高于初始化outer-train 75%分位阈值，再最多取8；训练STE，推理硬mask。所有证据聚合先mask；bias-free融合输出只由选中证据产生。残差单位方向按患者CAM范数校准，`0.05*sigmoid(beta)`初值0.0075，实际逐患者≤5%；0条时严格0。E5/E9使用相同校准，避免强度混杂。gate关闭旁路要求与B0-A逐值相同。新参数只追加原MF Adam且WD=0；原MP/MF组不变，保留未使用PEN初始化脚手架。

训练每5epoch按固定轮转取2名train受试者，独立keep/delete rawFC，完整重跑GFR/CAM/原分类头；keep权重0.10，delete0.05，margin0.05。delete仅用于未干预预测正确且非空证据者。CF冻结主干、固定原输入残差，防止残差分支制造删除脆弱性；分类分支正常训练全部主干及新模块。最佳模型逐个test受试者独立审计selected/random/Top-abs(z)、live/fixed残差及固定/动态邻接，不以这些诊断重选checkpoint。

## 十组正式矩阵

|ID|组别|唯一改动|GPU|
|---|---|---|---|
|B0|B0A_ANCHOR|完整E32初始化，CAM直接分类|0|
|E1|Z_TOPK_FULL|无疾病Query，abs(z)>1及Top8；相同value/残差/CF；登记无效query参数容量差异|1|
|E2|ANONYMOUS_FULL|疾病bank正交匿名化，其余同E5|1|
|E3|REAL_NO_CF|正确疾病Query，无CF训练|0|
|E4|REAL_KEEP_ONLY|只加keep，不加delete|1|
|E5|REAL_FULL|唯一主方法，keep+delete|0|
|E6|ANATOMY_NEUTRAL_FULL|真实端点不变，中性解剖技术Query|1|
|E7|WRONG_PAIR_FULL|同组大小内query关联循环错配；物理数值不动|0|
|E8|SWAPPED_CLASS_FULL|仅交换ASD/TD Query通道；标签及软方向不动|1|
|E9|NO_RIVAL_NORMMATCH_FULL|平均Query logits形成一般异常支持；相同参数及残差剂量校准|0|

B0连续seed0十折，保留首次CLIP加载生命周期。E组逐fold恢复B0入口、模型及两个Adam，再隔离初始化新模块。共同入口配对不代表全训练同随机轨迹。所有组为450epoch、early200、原shell超参及严格首次best。4项正式指标仅ACC / hard-label one-hot AUC / SPE-TD / SEN-ASD，fold mean±population SD；总正确数另标micro计数。GO门槛见config；预检/锚点失败立即停止正式创新训练。

## 参考与边界

[ERASER](https://aclanthology.org/2020.acl-main.408/)提供保留/删除忠实度思路；[GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html)区分充分性与必要性；[ROAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)讨论特征移除评价及重训。它们不构成本次创新证明。[ABIDE官方QA](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)用于核验QC字段。

历史val=test、全队列标准化/RFE与transductive图保持；fold训练集重叠；单seed内部探索不等于独立泛化、统计显著或生物因果。

复现：`CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python run.py prepare`，随后同样绑定单张卡执行`python run.py preflight`，再执行自动绑卡的`python run.py all`。`resume`只恢复未完成状态。全部状态在`.runtime`，正式指标在results。最终更新handoff只追加且验证前缀；EIO时保存pending。

## 运行后报告入口与必要状态

100折已完成；训练后的原报告遇到NumPy bool_不能直接序列化为JSON。为保持训练签名及精确恢复，run.py/method.py/config均未修改；正式报告入口使用`PYTHONDONTWRITEBYTECODE=1 python audit.py --finalize`，只做完整性核验、原报告的NumPy标量转换、机制汇总和有界交付尝试，不训练。原异常与修复来源均保存在manifest。`audit.py --group E1/E5/E9`只读取相应组的10个最佳模型，分别隔离分类与CF梯度，不执行optimizer.step；必须绑定一张CUDA_VISIBLE_DEVICES并限制两线程。

本次保留的`.runtime/source_audit.json`、`storage_probe.json`、`handoff_prefix.json`可从公开manifest的同名字段重建；它们是prepare需要的来源与前缀保护记录。入口anchors、任务状态、best/latest checkpoint是精确resume必需产物，不能删除后仍声称可从训练中断点继续。全新重跑应在独立副本恢复来源记录并重置任务/manifest运行状态，不覆盖这次100折记录。参考与文本缓存可以由冻结方法、Prompt和原始只读输入再生。

交付状态为`PARTIAL_TMP_ONLY`：正式目录创建与根handoff追加均返回EIO。唯一完整副本在本目录；`handoff_append_pending.md`保留完整待追加摘要。/tmp不是持久存储，不应清理该唯一副本。未提交Git，未修改实验39。

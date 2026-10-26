# Experiment 42 — 四环节独立可证伪机制验证

Q1文本有无疾病区分证据；Q2评分能否选对真实FC；Q3冻结分类器时CF是否改善定位；Q4表型适用性与QC可靠性是否有独立作用。四链均执行，失败互不替代。本轮不预先训练HCER-v3。规则在config.json冻结，外层测试不用于设计/文本选择。实际目录位于/tmp；/data原子写读检查EIO，/tmp不保证持久。

## 文件与实际路径审计

源码/data/历史只读，SHA见runtime/source_audit.json。E38最新重跑与E40实际路径在/tmp，未伪称访问根目录不存在的版本；不修改实验39/40。

|文件:函数/行|实际行为及保持边界|
|---|---|
|main.py:111 train_and_eval|871×320，ASD403/label0，TD468/label1，SKF10 random_state10；一次seed0连续流|
|network.py:153 GraphLearn.forward|投影余弦、score>.9阈值；原始关系尚未加自环|
|network.py:197/225 GFR.forward|四张患者图，各支完整320维传播；i==1覆盖i==0，i==2/3累加，不修复|
|utils.py:101 normalize_adj|D^-1/2(A+I)D^-1/2，不改原传播|
|model.py:54 EvalHelper|val_idx=test_idx；测试ACC严格提高才保存首次最佳|
|model.py:76/78、125–148|原双Adam；simple-2分类+四图GraphConstructLoss均值；ALoss不反传|
|model.py:293 _print_acc|one-hot硬预测roc_auc_score，保留Hard-AUC|
|network.py:53–91 PLN构造|完整未用PEN MLP、pool、8prompt缓冲区和初始化；前向CAM后跳过SPGS/RES|
|network.py:93–152、layers.py:80/144/175|InputLayer/Encoder/FeedForward/OutputLayer保留；CAM4×36是混合潜在位置而非ROI|
|layers.py:192 GraphConv|adj乘输入，i接收j；fixed graph仅隔离诊断hook|
|ABIDE-simple-2-concat-weighted-cosine.sh:1|lr.0038/reg.11/dropout.35/hidden18/head2/nlayer1，与E32核验450/early200|
|phenotyic_processing.ipynb cells8–12|PHENO26年龄bin+20站点+2性别；ANAT6/FUNC10是质量/头动；QC StandardScaler保持方向|
|data_processing.ipynb|FC为收缩相关Fisher-z，RFE在CV前，原输入不重建|
|E11 roi_256_named_mapping.csv|39条unique numeric + both endpoints verified；connection_name用后续核验名，不用早期UNKNOWN猜测|
|E32 integration.py:patch_network B0 pass|原PEN完整构造、仅前向绕过，不能用物理删PEN低基线|

论文[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)Fig1、Sec2、Eq1–7：患者关系、完整拼接特征传播、CAM、SPGS/RES。纸面有标签邻接项，当前simple-2没有优化ALoss。源码是本轮执行依据。原始raw_X[:,64+edge_id]才有固定FC身份，GFR输出64:320不是原始连接，潜在Token不是ROI。

## 历史问题→修复→可证伪测试

|历史问题|本次处理|失败即不支持|
|---|---|---|
|E38删PEN改变RNG；其重跑仍NO-GO|复用经E32 exact验证的E40不可变anchor/best，在E42重算十折与构造SHA|任何混淆、四指标、epoch、参数或RNG不符停止依赖训练|
|E38 summary绕过Top8、错配raw/z顺序|统一canonical identity；错配只改membership；无新分类残差|索引或mask旁路FAIL|
|E40强制单位范数残差、空证据约0|冻结B0不注入残差；Q4独立动态预算/弃权|不能据内部响应宣称分类收益|
|E40 score与abs(z)约−.698；选中删除.01897低于Top-abs(z).06718|同预算输入替换与live/fixed图独立检验，不手调相关符号|不胜Top-abs(z)判Q2_NO_GO|
|E40第3/6/9折不稳，CF梯度可能过强|只更新47参数selector；teacher/CF梯度分开统计和控制|CF未提高OOF定位即NO_GO|
|旧参考TD OOF/ASD fullfit，QC绝对偏离不是坏质量|所有训练人相同交叉拟合；QC逐列正仿射/定义审核|自身fit路径或test标签依赖FAIL|

## 锚点、数据与参考

允许复用同数据/源码/划分/hash不可变锚点。E40十折已exact复演E32；本轮复制并校验anchor和best SHA、初始共同参数/入口RNG、best/executed epoch、混淆、四指标，重新前向评估。必须报告新B0训练0折、验证复用10折，不能将历史CSV冒充新训练。CLIP首次加载/后续缓存生命周期也核对构造SHA。preflight独立生成E32原补丁reference，核对输出/图/损失/梯度/原双Adam两步，以及中断后第三步逐值复演。

每outer-train所有ASD/TD走相同SKF5；当前chunk的人以其它chunks TD拟合Ridge10预测mu；test用全outer-train TD。sigma为train-TD交叉拟合残差1.4826 MAD，floor1e-4，clip3。保存membership、fit_subject_ID和系数。固定assignment时改变个人标签不得改变自身fit集合/mu；总体TD尺度本来依赖训练TD集合，不冒称整个z与训练标签无关。test标签置换对参考与推理必须逐值不变。

## 冻结四链

Q1A：13×39 registry保存network_supported/edge_specific_verified/方向冲突；没有精确边直接证据默认false。训练PHENO残差化后方向候选看signed z，either只看幅值；SMD≥.1、5内折方向≥.8、有效站点同向≥.6（站点每类≥5）。至少7假说在≥7/10折PASS才足够。Q1B：P0原文；P1平衡句式；P2完整HO双端逐边短句strict77分别编码后平均，未截断；P3条件可反驳，背景标exploratory。真实/中性/交换/错配/全Gram匿名/数值/结构诊断用同30输入31参数150步Adam，嵌套训练5折预处理，不读outer-test选择文本。每族相对全部控制ACC/AUC≥.25pp且≥7折同向胜。自由线性读出可能吸收类别槽交换，必须作为不可识别限制报告。

Q2：冻结B0；S0三组固定随机mask seed非训练seed，S1 Top-abs(z)，S2训练内调整效应，S3每类8训练者逐边B0依赖导师，S4四风格训练probe的逐边signed contribution绝对值。S3评价患者与导师训练不重叠；不能把自己oracle删边再自评。K4/8、TDmu全/半替换、训练TD匹配donor；live/fixed图；正确/错误原预测分层，预测类与真实类logp分开。主GO要求K8训练内选定风格胜随机和Top-abs(z)均值且各≥7折，fixed图仍支持。先fold1/6检查开销，再其它fold，不因结果改预算。

Q3：冻结B0，只在16个训练导师人群5折OOF训练共享46输入47参数选择器，30步。C0 teacher-only；C1keep；C2delete；C3两者。每5步2训练者；只对B0正确且置信≥.6者做CF。keep .10KL；delete .05softplus，相对同预算random/Top-abs(z) margin.05。C1–C3训练梯度CF cap为teacher范数.25；C3_FIXED_DIAGNOSTIC同预算原固定权重供对比，绝不用外层test选权重。记录梯度norm/cos/update、覆盖与unique IDs；half/full/fixed图。hard TopK有限差分本为分段常数，STE是代理；数值梯度检查连续松弛而非声称hard真实导数。GO要求删除忠实度和训练真实类单边导师排序均较C0提升、各≥7折。

Q4：真实[QC字典](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)及16列与匹配原CSV正仿射验收。R0无门控；R1训练TD背景支持→适用/弃权非疾病分数；R2 FUNC DVARS/outlier/meanFD正风险；R3 ANAT CNR/FBER/SNR负风险、QI1正风险，含糊EFC/FWHM等排除。GSR为ghost-to-signal非回归开关。rel按训练分位在.5–1，K=floor8rel，训练10%低可靠性弃权；原B0常规预测不变，作用经独立keep/选边/弃权诊断验证，不用单位归一化抹掉。对照反向/随机/恒一/site置乱。站点年龄性别控制后风险与FC偏离/计算不稳定rho>.05、胜随机、≥7fold才支持。R4只组合全fold训练验证通过角色。FC-only噪声不是物理头动模拟，QC未改时不假装能感知新运动。

## 执行与恢复

```bash
cd /tmp/SPromptGL-E42-stage/experiment/experiment42_fourlink_evidence_validation
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python run.py all
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python run.py resume
CUDA_VISIBLE_DEVICES=0 python run.py worker --phase q2 --fold 1
CUDA_VISIBLE_DEVICES=0 python run.py finalize
PYTHONDONTWRITEBYTECODE=1 python audit.py --all
PYTHONDONTWRITEBYTECODE=1 python audit.py --deliver
```

`audit.py`是唯一额外的小脚本：为保持预注册`run.py`字节冻结，独立补做训练者group-mask干预、替换分布代理、选择器精确恢复及结果审计。完成四阶段后执行`audit.py --all`（每卡最多2worker，已完成QA跳过）生成最终报告，再执行`audit.py --deliver`完成历史前缀保护追加和有界同步。它还公开更正R0/R2/R3站点置乱诊断的标签缺陷：这三种门控不读取站点，真正只置乱SITE应恒等；冻结原始运行误用了随机可靠性排序，错误原始行保留于runtime，最终CSV按此恒等性质重建并注明，训练结果、正反向/随机QC对照与决策均不变。`run.py finalize`会重建未校正的原始汇总，因此最终交付必须以这两条audit命令收尾。该脚本不重新训练分类器或选取超参数。首次补充恢复验收发现克隆Adam字典的CPU step引用复用，已以深拷贝和完整RNG恢复修正；原失败日志保留。

两CUDA兼容卡实际PPU-ZW810E，各98304MiB；每卡最多2worker，每worker2CPU线程。4阶段各10fold共40诊断任务，不伪造100分类fold。每任务锁、代码配置SHA、原子输出；Q2按患者边界续跑；Q3模型/optimizer/RNG/step与fit/valid写入恢复checkpoint。完成键精确跳过。任何链NO_GO不启动集成分类；四链全GO也需另注册具体集成模型，不能凭空新增。

文献：[ERASER](https://aclanthology.org/2020.acl-main.408/)、[GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html)、[ROAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)用于忠实度/充分必要/扰动边界，不是新颖证明。[ABIDE](https://pmc.ncbi.nlm.nih.gov/articles/PMC3782703/)、[King2019](https://link.springer.com/article/10.1186/s13229-019-0273-5)、[He2020](https://pubmed.ncbi.nlm.nih.gov/31916675/)用于站点/复制风险，不背书39条精确边。两PMC页遇验证码，实际读取对应出版社/PubMed，不冒称被阻全文已读。Prompt来自既有综合和确定模板，没有GPT-4新调用或医生认证。

保留val=test、CV前RFE/标准化、transductive和重叠fold局限；四指标是历史协议内部比较。FC替换可能OOD，是模型依赖而非临床因果。结果先于叙述，不声称SOTA/MICCAI录用/生物标志物。

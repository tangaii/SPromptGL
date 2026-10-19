# 实验二十七：EGRP-CAM 医学关系可识别性审计

`STATUS=STOP_UNGROUNDED_INPUTS` · `STAGE=GROUNDING_ONLY` · `SCIENCE=NOT_EVALUABLE`

按更新prompt的预注册Gate 0结束：可支撑的独立跨疾病测量关系为0，要求至少2。没有训练失败；也没有产生可供判断模型性能优劣的新试验。不能把输入前提未成立解释为所有关系路由方法普遍无效。

## 仓库、论文与源码核对

本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；远端main `bfd5f160314704c98106fd3d8abd9402d4f60f61`。提交不同，但六核心文件逐字节一致；config.py实际不存在，参数在main.py和历史shell中。原工作区既有修改保留。

MMGPL在概念响应上加权Token并构建Token图，经GCN形成Transformer图Prompt；已有相关机制，不能声称首次疾病概念建图。作者版本ABIDE使用1029对MRI/FC、BiomedCLIP及5折×3seed，72.39%不能与当前871人历史分数横向比较。[原文Fig1、§III-B/C/D、Eq7–14、TableI/Fig2](https://arxiv.org/pdf/2312.14574v2)。

SPromptGL将患者多关系GFR表示送入CAM，随后SPGS对局部Token加权、RES融合。论文的示意病变概念不能证明本地QA列测得病变。论文监督图损失与当前ALoss不反传实现有差异，论文ABIDE超参与历史shell也不同；本轮审计记录差异，未更改协议。[原文Fig1、§2、Eq1–7和ABIDE消融](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)。

|本地源码|SHA256|远端相等|
|---|---|---|
|main.py|dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2|是|
|model.py|254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda|是|
|network.py|484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec|是|
|layers.py|4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a|是|
|utils.py|95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67|是|
|ABIDE-simple-2-concat-weighted-cosine.sh|28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763|是|

原始Attention先计算QK，再除以全局abs(min)，L2 normalize后才softmax/dropout。此全局缩放和归一化不是标准缩放点积Attention。若未来通过前提，新增偏置只应加在归一化之后、softmax之前的InputLayer.attention；Encoder路径不动。本轮未创建e27_relation.py或e27_patch.py，更未运行网络。

原GFR i=1确实覆盖i=0的combined_feat；ALoss计算并打印，但simple-2最终loss只有原分类loss与四图正则均值。两个Adam、val_idx=test_idx、严格ACC提升选首个最佳Checkpoint、wait>early均由真实源码核实。

### 逐行源码锚点

|机制|文件:行|实际锚点|
|---|---|---|
|prompt_list|network.py:24|`attr_words = [`|
|clip_cached|network.py:33|`def get_prompt_text_features():`|
|clip_rn50|network.py:37|`clip_model, _ = clip.load('RN50', device)`|
|mlp_4096_128_9|network.py:86|`self.MLP = MLP(input_dim = 4096,hidden_dim = 128, output_dim = 9)`|
|first_cam|network.py:97|`x, _attn = self.InputLayer(x)`|
|encoder|network.py:101|`x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)`|
|latent_tokens|network.py:110|`part_x = x.view(871, 4, 4, 9)`|
|spgs_softmax|network.py:126|`similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)`|
|spgs_max|network.py:129|`similarity_max, _ = torch.max(similarity_matrix, dim=-1)`|
|spgs_weight|network.py:132|`part_x = similarity_max.unsqueeze(-1) * part_x`|
|res|network.py:147|`x = x + torch.cat(combine_part,dim=2)`|
|gfr_pheno|network.py:249|`combined_feat = f_w*feat0+torch.cat(x,dim=1)`|
|gfr_anat_overwrite|network.py:256|`combined_feat = f_w*feat1+torch.cat(x,dim=1)`|
|attention_min|layers.py:54|`attn = attn/abs(attn.min())`|
|attention_normalize_softmax|layers.py:55|`attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))`|
|input_layer|layers.py:80|`class VariLengthInputLayer(nn.Module):`|
|encode_layer|layers.py:144|`class EncodeLayer(nn.Module):`|
|test_as_val|model.py:54|`self.val_idx = np.array(test_index)`|
|test_index|model.py:55|`self.tst_idx = np.array(test_index)`|
|adam_mp|model.py:76|`self.optimizer_MP0 = optim.Adam(self.MessagePassing0.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)`|
|adam_mf|model.py:78|`self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)`|
|aloss_computed|model.py:131|`feat, adj, ALoss = self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)`|
|four_graph_losses|model.py:137|`all_graph_loss = all_graph_loss/4`|
|simple2_loss_no_aloss|model.py:143|`loss = cls_loss + all_graph_loss`|
|cls_loss|utils.py:94|`return F.nll_loss(output[idx], labels[idx], weight)`|
|graph_loss|utils.py:31|`def GraphConstructLoss(feat, adj, theta_smooth, theta_degree, theta_sparsity):`|
|historical_cv|main.py:134|`skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)#shang8xia4zuihou1`|
|preprocessed_input|main.py:116|`data = pd.read_csv(path + 'processed_standard_data.csv').values`|
|checkpoint_strict|main.py:153|`if cur_val_acc > best_val_acc:`|
|early_strict|main.py:163|`if wait_cnt > hyperpm.early:`|

## 真实输入和变换

当前数据871×320，403 ASD/468 TD。CSV列序与modal_feat_dict逐列一致；年龄/站点/性别独热各行严格一项为1；全表有限、无缺失。

|组|维数/内容|真实角色|
|---|---|---|
|PHENO|48=年龄26个两岁分箱+站点20独热+性别2独热|人口学与采集上下文，非临床症状量表|
|ANAT|6个anat_*字段|结构扫描QA，不是体积/厚度/白质结构|
|FUNC|10个func_*字段|功能扫描QA，不是激活图或独立功能影像流|
|CORR|256个RFE连接列|收缩协方差相关的Fisher-z功能连接测量|

16列QC采用历史全队列StandardScaler；PHENO独热未被数值标准化。CORR按HO111的6105条上三角候选连接计算相关、Fisher-z后选择256条，当前CSV无后续统一z标准化。Notebook中的RFE由旧5折最后一折697人拟合、随后在当前10折外使用；未保存原selector，实验11用数值指纹恢复索引。本轮没有重算RFE、变换训练输入或重排患者。

|QA字段|官方含义/解释边界|
|---|---|
|anat_cnr|灰/白质对比噪声比；不是灰白质体积|
|anat_efc|图像强度熵焦点指标；不是神经系统熵或厚度|
|anat_fber|头部前景/背景能量比|
|anat_fwhm|强度图像空间平滑度FWHM，原始体素单位；不是皮层厚度|
|anat_qi1|背景伪影体素比例|
|anat_snr|灰质信号/背景噪声比；不是脑容量|
|func_efc|平均功能图像强度熵焦点指标|
|func_fber|平均功能图像前景/背景能量比|
|func_fwhm|平均功能图像空间平滑度|
|func_dvars|标准化时间差分变动幅度|
|func_outlier|时间序列每帧异常体素比例的均值|
|func_quality|每帧对中位图像的秩相关距离|
|func_mean_fd|平均头动帧间位移，原始毫米|
|func_num_fd|超过头动阈值的体积/帧数|
|func_perc_fd|超过头动阈值的帧比例/百分比|
|func_gsr|ghost-to-signal ratio；不是global signal regression开关|

以上字段分类由[ABIDE PCP官方QA定义](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)核对；当前输入单位均为历史z分数，不能由字段名ANAT/FUNC推断测得脑病变。

## 患者身份与旁路临床量表复核

复用实验11的subject_row_mapping并独立重建871×64的PHENO/QC：871人唯一双射，最大绝对误差7.105427357601e-15，RMSE=2.42721525958011e-16。匹配不读诊断标签，标签仅在匹配完成后作一致性核查。当前256条连接映射编号/三角索引均通过，252条有可信名称，4条[195,225,232,236]不适合语义命名。沿用同CSV签名下实验11既有FC全量数值验证，未为本轮再重建整套FC。

**历史审计矛盾的具体来源：** 实验11建立了指纹身份链；实验21只在有限目录查找subject_IDs.txt，未读取实验11映射，故给出NOT_JOINABLE。本轮已验证现存映射，不能再把“0/871可链接”当作当前事实；同时，链接成功不等于量表已进入当前输入、量表版本可比或诊断泄漏已排除。历史文件保持不变。

以下是当前871人旁路表的可用性，只计非负观测，另记缺失/负值代码；不是合法量表范围、施测可靠度或跨版本有效性验收。覆盖明显依赖诊断与站点，不能用缺失模式替代疾病标签。

|旁路字段|ASD非负观测/403|TD非负观测/468|ASD缺失/负代码|TD缺失/负代码|已加入本轮输入|
|---|---:|---:|---:|---:|---|
|FIQ|376|435|12/15|16/17|否|
|ADI_R_SOCIAL_TOTAL_A|276|0|100/27|468/0|否|
|ADI_RRB_TOTAL_C|277|0|100/26|468/0|否|
|ADOS_TOTAL|282|19|107/14|444/5|否|
|SRS_RAW_TOTAL|163|137|238/2|297/34|否|
|SRS_COGNITION|26|30|377/0|438/0|否|

量表定义及版本/研究可靠度应继续参考[ABIDE官方字典](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf)。原表有部分ADOS/ADI-R/SRS/IQ，不含ADAS；这些字段不在320维。引入它们或MRI形态会改变输入任务，必须另立基线，不能套用旧P0证明本方法增益。

## Gate 0逐条候选账本

候选在e27_contract.py中冻结，没有做标签驱动的相关性搜索。完整中英文、两个模态、拟用/真实字段、单位、单调/非单调定义、混杂、文献支持范围见feasibility.json的grounding_ledger。尚无外部固定的有符号关系就记录“未固定”，不从文案生成医学定律。

|ID|候选关系|组对|类别|不满足依据|
|---|---|---|---|---|
|H01|脑体积与同患者FC的条件关系|ANAT↔Correlation|DISEASE_CANDIDATE|ANAT6无脑体积，CNR/SNR等不能替代体积。|
|H02|皮层厚度与同患者FC的条件关系|ANAT↔Correlation|DISEASE_CANDIDATE|ANAT6无皮层厚度；anat_fwhm是平滑度，不是厚度。|
|H03|白质微结构与功能连接的条件关系|ANAT↔Correlation|DISEASE_CANDIDATE|无DTI/FA/MD；信噪比不是白质完整性。|
|H04|患者症状量表与FC的条件关系|PHENO↔Correlation|DISEASE_CANDIDATE|PHENO48没有症状量表；原始旁路量表不能自动加入固定320维。|
|H05|脑结构侧化与FC的条件关系|ANAT↔Correlation|DISEASE_CANDIDATE|没有双侧形态端点；HO端点标签不提供形态数值。|
|H06|脑室形态与FC的条件关系|ANAT↔Correlation|DISEASE_CANDIDATE|ANAT6没有脑室体积或形状。|
|H07|激活模式与功能连接的条件关系|FUNC↔Correlation|DISEASE_CANDIDATE|FUNC10为QA统计，无任务激活；FC不等于激活图。|
|H08|年龄/性别与FC发育变化|PHENO↔Correlation|DEVELOPMENTAL_NON_DISEASE|人口学上下文不满足两种疾病/临床测量端点。|
|H09|头动/采集质量与FC伪相关|FUNC↔Correlation|ACQUISITION_NON_DISEASE|即使可测量，也属于采集混杂；不能充当第二疾病模态。|
|H10|结构与功能扫描的QA共变|ANAT↔FUNC|ACQUISITION_NON_DISEASE|两端都是QA，无法验证跨疾病测量关系。|

**Gate 0：STOP，0项符合≥2项独立关系要求。** 两个真实疾病/临床测量端点尚不存在；年龄/性别—FC即使有关联也是发育关系，头动/QC—FC是采集关系，ANAT-QA与FUNC-QA共变也不能改叫病理对应。CLIP相似度、共同ASD/TD标签或潜在Token相似度都无法补足缺失的测量端点与独立临床命题。

### 原始八句SPGS与输入映射

|原句（原拼写冻结）|能否由当前测量支撑为跨疾病模态关系|
|---|---|
|Nromal brain volume|无脑体积端点|
|Uniform cortical thickness|无皮层厚度端点|
|Symmetrical brain structure|无双侧形态端点|
|Normal ventricular morphology|无脑室形态端点|
|Altered brain volume|无脑体积端点|
|Abnormal functional connectivity|只对应FC测量自身，不能证明异常或第二疾病模态|
|White matter integrity issues|无DTI/白质完整性端点|
|Different activation patterns|无激活图；FC不是激活模式|

原Prompt列表从根network.py的AST直接提取并逐字对照，拼写Nromal保留；本轮不加载CLIP、不改Prompt。

## Gate 1与Stage 2的真实执行边界

`Gate1=NOT_RUN_G0_FAILED`：未计算outer-train关联、同站点置乱零分布、Holm p值或5fold OOF增益。不填0、不制造训练内量化成绩；没有用QA相关性越过疾病语义前提。

`Stage2=NOT_STARTED_G0_FAILED`：实际0/180fold、0/18任务、GPU0=0/GPU1=0，无模型副本/新关系模块/Checkpoint。GPU双Adam/RNG/偏置parity不适用；没有声称GPU工程预检通过。

|条件性配置|冻结问题|本轮状态|
|---|---|---|
|R0 NUMERIC_CONTEXT|同预算数值关联CAM偏置|Gate0未通过，未创建/训练|
|R1 MEDICAL_UNVERIFIED|有依据的候选关系不做患者拒绝|Gate0未通过，未创建/训练|
|R2 EGRP_CORE|医学关系+训练内实测核验及三路动作|Gate0未通过，未创建/训练|
|R3 NEUTRAL_TEXT|同患者验证及预算的中性文字|Gate0未通过，未创建/训练|
|R4 WRONG_RELATION|同预算预定错关系映射|Gate0未通过，未创建/训练|
|R5 ISOMETRIC_TEXT|同Gram固定旋转关系文字|Gate0未通过，未创建/训练|

### 历史完整30折只读来源核查

|历史参照|ACC %|BA %|ASD Prob-AUC %|来源|
|---|---:|---:|---:|---|
|P0|91.69105|91.53935|93.33520|/data/adodas/SPromptGL-main/experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv（F0）|
|E9|91.88262|91.69151|93.82272|/data/adodas/SPromptGL-main/experiment/experiment9_prompt_disease_adversarial_interaction/results/fold_metrics.csv（E9）|

两份CSV均包含seed0/1/2×fold1–10唯一30行；先每seed10折均值再三seed均值，与冻结参考一致。没有重训P0/E9，没有R2新成绩或配对差；实验24两fold的91.980%不作为完整基线。

## 历史经验与创新边界

|实验|当前采用的教训|
|---|---|
|11|FC/身份数值恢复可验证；潜在Token没有因此获得ROI身份。|
|17|PHENO Shared弱，类原型趋同，真实文本未稳定胜阴性控制。|
|18|判别保护/Private可读出信息不等于最终收益；两教师同时正确不是增益真值。|
|19|中性C5优于真实C4，自由Key可以吸收固定查询方向。|
|20|固定医学概念局部证据、连接线图及E9组合均未验证语义必要性。|
|21|语义、自适应、图及E9协同未稳定成立；临床审计未复用实验11身份链。|
|22|符号与概念幅度反事实响应极弱，正交残差没有提供E9互补。|
|23|增益排序弱；完整训练全部新组低于P0，同预算常量优于增益路由。|
|24|48折pilot按预注册规则停止，91.980%只来自前两折。|
|25|质量相似不是医学可靠度，真实患者QC对应未稳定胜置乱。|
|26|有符号读出进入决策，但真实文本不稳定胜无关/同Gram随机；FC-only有竞争力。|

[CSD](https://papers.miccai.org/miccai-2026/1073-Paper1452.html)已有类别语义共享/特异对齐；[OTCHA官方仓库](https://github.com/labhai/OTCHA)已有置信度部分匹配/dustbin；[AAAI To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/view/39248)已有非强制统一对齐的论证。本轮仅按官方摘要/README记录这些边界，未假称已读全部正文。新增门控、拒绝、OT或对比损失本身不构成创新。

原历史协议包含测试参与Checkpoint选择、外层CV前QC标准化/监督RFE、871人transductive传播，且同队列已被多轮探索。即使未来内部结果阳性，也不能称无泄漏外部泛化或患者因果获益；本轮更没有可评估的训练性能或已成立的MICCAI2027创新。

## 工程、签名、重跑与最小下一步

自动审计开始UTC 2026-10-19T08:11:05+00:00，结束UTC 2026-10-19T08:11:55+00:00；自动审计/取证计时31.97秒。之前的人工式论文/源码审阅未单独计时；没有GPU训练耗时。

保护清单包含963个根源/数据/历史文件；结束SHA对照结果：UNCHANGED_SHA256。没有git add/commit/push/pull/reset/checkout。审计完成后再次运行all/resume仅核验签名和唯一handoff标记，不重算或创建训练目录。

根handoff状态：UPDATED_FSYNC_VERIFIED；更新UTC：2026-10-19T08:11:55+00:00；旧前缀保留=True。

永久文件共6个：e27_contract.py、e27_feasibility.py、run.py、manifest.json、results/feasibility.json、results/report.md。不创建空白fold_metrics/diagnostics，不导出患者级量表或预测，不生成图片/README/训练源码或权重。

代码与主要输入SHA256（报告自身签名由manifest登记）：

|文件|SHA256|
|---|---|
|e27_contract.py|078eec43ad1c8d96d0b9d2c2384c5eab3c964eb700189d6597c730382d33043c|
|e27_feasibility.py|f3a58a30848a0607483ca0a622a2523c5b6c94456ff381e61f6c82b3d3297b05|
|run.py|cefd91fe1294ca39fcd2cf4c955ce6df542a020a0a9fbe29e9af328d9908d143|
|data/ABIDE/processed_standard_data.csv|b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1|
|data/ABIDE/modal_feat_dict.npy|8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e|
|data/ABIDE/data_processing.ipynb|348147e93555f765b4e78151b11221556a151b091f931d05b814dbcb70100c0f|
|data/ABIDE/phenotyic_processing.ipynb|15a46187b6f994ab3cde623fc0b5b2aa509efb41b21459e55fafe828240fdc7c|
|dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv|009f01a89697113e5375ded4fbcff4c85e89a4daa6d13e63c21ae5d57dda0865|
|历史P0 CSV|8118658fa728a7a239a7648426f0e8b6b097589bd9c3d4b2f8edfc4544f143ad|
|历史E9 CSV|669b59f2313dd559d5ce5fb5820ee6e9b5fc6da76fca76cdd2b2097a58b2e638|
|SPromptGL PDF|219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89|
|MMGPL PDF|535b9354c3faa187a2122e12f072b579f57ae1cdb028205bd629d982d84f18df|

**下一步的最小前提**：在已有可核验身份链上先审计真实量表版本、可靠度与不平衡缺失，或补充同患者定量MRI/DTI/PET等真实测量。先外部预定至少两项不同疾病/临床测量的可检验关系，再按训练内选模、独立站点/外队列重新预注册；症状量表若参与原ASD诊断，须避免把该诊断过程本身当预测输入。保留本次严格的不可评估结果，停止用QC改名或更换Prompt追分。

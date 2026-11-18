# Experiment 81.6.2 — SA-PRCL

Semantic-Anchored Prompt-conditioned Relational Learning (语义锚定的疾病Prompt条件化局部关系学习)。本实验只从每折真实恢复的 E74.4-G2 / E61 epoch-10 完整 warmup 起步。E81.6 与 E81.6.1 只读参考，不参与初始化。正式组固定为 G1 protected、G2 scrambled grouping protected、G3 joint，各10折。

## 编码前来源审查

复核了[原始SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)第2.1节GFR、第2.2节CAM/SPGS/RES及公式(3)、(5)、(6)、(7)，并逐行核对官方[`network.py`](https://github.com/wanxixi11/SPromptGL/blob/main/network.py)、[`layers.py`](https://github.com/wanxixi11/SPromptGL/blob/main/layers.py)和[`main.py`](https://github.com/wanxixi11/SPromptGL/blob/main/main.py)。实际源码保留8条原始Prompt并冻结CLIP RN50。原PEN将前4条、后4条分别拼接成4096维，经MLP(4096→128→9)映射到两个类别条件向量；CAM输入按4×4×9组织。原RES对每个连续36维槽用`AvgPool1d(4,4)`得到9维全局特征，再逐坐标执行`softmax(local*global)*local`并按原序加回。OutputLayer为LayerNorm144→Linear144,36→GELU→Dropout(0.35)→Linear36,2，接收但不使用`attn_embedding`。本实验只借鉴Prompt局部引导和RES融合，不恢复原PEN；SA-PRCL是新机制，不是原论文已有实现。GFR输出320维，经历史CAM映射后分类器入口为144维。

本次实际基座链经只读`path_adapter.py`加载E74.4 `prepare(fold)`，递归调用E74.1、E69.3、E61与E53的历史实现，逐折恢复E61第10轮完整模型、CAM、三个优化器moment、公用RNG、划分及4096训练关系对。E74.4 `soft_positive_dcr.py`的SHA256为`3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`；E74.4 `run.py`为`cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd`；配置为`a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a`。真正的PLNNoPEN及OutputLayer来自SHA已核验的历史vendor `dugsa.py` (`ee59de1eb00fad670a90c732356e2798c31481b9199701ed4bccc6c6c9a76f12`) 和`layers.py` (`4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a`)。每折的warmup、DARG来源文件、数据、历史任务身份与依赖SHA由`run.py`的动态来源清单记录。E74.4现成Best/Latest只作历史审计材料，绝不作为SA-PRCL的起点。

E81.6使用的pairwise PRCL-RES源码指定提交为[`75e11ac262a3789aa3595748800f4aed1ae3cc7a`](https://github.com/tangaii/SPromptGL/blob/75e11ac262a3789aa3595748800f4aed1ae3cc7a/experiment/experiment81_6_prcl_res/prcl_res.py)。E81.6.1按[`aabe8d5b146a59bfba7242fd93dce28a9eaf5164`](https://github.com/tangaii/SPromptGL/blob/aabe8d5b146a59bfba7242fd93dce28a9eaf5164/experiment/experiment81_6_1_contrastive_prcl_res/cprl_res.py)检查。它的42,336个新增可训练参数中，32,800个来自自由`Linear(1024,32)`，使8个文本条件可被任意重映射；另有中心化、视觉/语义关系评分、Pairwise Key/Value、Null、相对视觉增量、两组聚合、RES、10%残差、Hook及新增损失。本实验取消自由文本投影、文本中心化及Prompt专属索引，固定8条Prompt的完整Gram几何，改为共享64→8关系映射与8条固定语义向量点积；增加按患者关系响应对组内Prompt做Softmax选择；将保护组的新增监督与原DARG损失/优化更新分开。历史E81结果及其源码SHA在`runtime/source_review.json`锁定。旧组原始/打乱分组几乎相同，以及残差关闭下的DARG变化，需要本实验区分“训练梯度改变”与“选中epoch不同”。

[KD-Brain MICCAI 2026论文](https://papers.miccai.org/miccai-2026/paper/0073_paper.pdf)与官方提交[`180873a5dc2e6b5caf862528663f25e986b8f4f4`](https://github.com/LiuSy0731/KDBrain/blob/180873a5dc2e6b5caf862528663f25e986b8f4f4/KDBrain/models/model.py)已复核。`KDBrain/models/model.py`的`Attention_AggregateLayer.forward`使用`text_projector(768,out_shape)`、tanh与初值0.5的`text_gate`构造`guided_query`，使疾病先验进入关系评分。SA-PRCL只借鉴语义先验直接影响关系分数的思路；具体采用固定CLIP几何与共享pairwise关系投影相加。未复制真实脑子网络、BioMedBERT、病理一致性监督或额外图结构。该源码SHA256为`2e7f32693847e0794a9d0b2deb172dbcd285c686b74827eea31d6e30335a887d`。

[ItemizedCLIP论文](https://arxiv.org/abs/2512.11141)及官方提交[`1a8c13c9c5df722f9cc35dd251687cd1fd454e93`](https://github.com/MLNeurosurg/ItemizedCLIP/blob/1a8c13c9c5df722f9cc35dd251687cd1fd454e93/Radiology_Tasks/src/open_mri/my_model.py)已复核。`PureAttentionPoolingBlock.forward`保留逐条文本条件的独立Query表示。本实验只借鉴多文本条件分别保留的原则；ABIDE没有患者级逐条Prompt真值，因此不复制ILA/IIS/KTA/MPS或条目监督损失。`my_model.py`原始内容SHA256为`6318c8cad31236560ffda9713801bc29fe74b7bf3cf49b077223174e88dbb4f25`。

## 冻结语义与关系机制

文本严格保留官方8条及顺序，包括拼写`Nromal`。固定OpenAI CLIP RN50、权重SHA `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`，训练前仅编码一次。将每行归一化得到`T[8,1024]`，CPU float64薄SVD得到`E=U diag(S)[8,8]`；以每列`U`绝对值最大位置的符号固定SVD正负号。训练E buffer固定，Gram误差、数值秩、奇异值及范数在预检中报告。固定Gram只保持原始CLIP几何，不代表文本有经验证的医学诊断含义。

CAM144经`[N,4,4,9]`整理为16个9维潜在槽，再经共享Linear(9,32)+位置参数+LayerNorm。槽是网络坐标，不是标注脑区。对每个接收/发送槽，Key输入`[u_p*u_q,abs(u_q-u_p)]`为64维，Value输入`[u_q,u_p*u_q,u_q-u_p]`为96维；分别共享映射到32维Key/Value。每个接收槽使用其余15槽及一个Null，Null Key可学习、Null Value固定零。共120个唯一无向槽对、240个消息方向，不跨患者建新图。

视觉关系分数为`dot(LN(Wq*u_p),Wk*r_pq)/sqrt(32)`。语义Key为`normalize(Wsem*r_pq)`且`Wsem:64→8`对全部Prompt共享。每条Prompt以固定E行与语义Key点积，和视觉分数相加，语义系数固定1；Null语义分数为0。得到8条独立注意力和一条视觉基准注意力。逐条Prompt消息减视觉消息得到`delta_m[k]`。患者条件化证据只对15个真实候选计算`logsumexp(score_prompt)-logsumexp(score_visual)`；分别在预注册两组内Softmax成`beta`，不使用患者标签或模型预测。正常、异常两组加权增量相减得到`relation_contrast`，不再增加64→32的两组自由融合层。G2仍用原医学文本，只打乱固定分组为normal `[0,1,4,5]`、abnormal `[2,3,6,7]`，仅检验原始4+4语义组织是否更好。

`relation_contrast[16,32]`经零初始化无偏置Linear(32,9)成为局部关系。RES继续使用每槽`AvgPool1d(4,4)`与`softmax(local*global)*local`逐坐标加权，展平144维，不加全局Linear或多头注意力。残差按给定公式约束到`0.1*norm(CAM144)`以内。`L_raw`只在合法训练行计算，系数0.001。零初始化保证初始残差精确为零。

## 三组优化与随机数隔离

G1/G2对原OutputLayer输入注册只读Hook，仅捕获CAM144及Dropout执行前公共RNG。原始DARG完整前向执行一次且不替换分类器输入；历史`total`先反传。新增分支用`z_cam.detach()`，冻结的原LayerNorm/Linear/GELU/Dropout/Linear函数式参数计算增强预测。函数式Dropout在隔离的CPU/CUDA RNG快照中重放刚才原Dropout的掩码，之后恢复DARG公共RNG至历史前向结束状态。新增`ClsLoss + .001 L_raw`只反传SA-PRCL；比较所有DARG梯度后，按原顺序更新GFR、CAM、DARG，SA-PRCL优化器最后更新。折2新增分支仅屏蔽零基376/406残差及监督/正则行，原DARG损失和样本集不变。

G3使用相同结构和同折新增参数初值。其OutputLayer单次Hook将增强CAM送入原分类器，历史分类及辅助损失按原代码计算；`z_rel=detach(z)+0.1*(z-detach(z))`仅限制关系分支回传比例，增强残差恒等路径保持完整梯度。其总损失为历史总损失加`.001 L_raw`。因此G1/G3比较的是不同的损失路由与联合更新机制，不只是一个梯度缩放开关。

所有组沿用E74.4实际配置：seed0、10折、E61 epoch10 warmup、最多450 epoch、wait>200、test ACC严格提升选Best、平局保留较早epoch、val=test。沿用GFR/CAM Adam(.0038,.11)、DARG AdamW(.001,.01)，SA-PRCL为AdamW(.0005,.01)。每折新增参数用私有初始化salt `816200+fold`，同折三组逐元素相同；CLIP/SVD缓存构造在恢复公共RNG之前完成，不逐epoch重播种。

## 预注册分析与边界

来源和代码冻结后运行集中Preflight：验证文本缓存/SVD Gram、固定buffer与参数名单、形状/邻居、Prompt-off/Null归零、患者条件beta、RES逐元素、10%上限、冻结分类器与原Dropout掩码一致性、公共RNG不变、原DARG梯度不受分支影响、十折G1/G2连续两步同DARG-only参数/优化器/RNG重放、G3联合梯度、同折初始化及一次真实中断恢复。任一主干隔离或公共RNG审计失败则不得开始正式训练。

正式30任务完成后独立重载Best/Latest，逐折复算ACC、Hard-AUC/BA、Prob-AUC、SEN-ASD、SPE-TD、ASD-F1，列出全10折和去Fold2的9折配对差值、W/T/L及描述性配对区间。分析G1 vs DARG、G1 vs G2、G1 vs G3、G1 vs历史E81.6.1-G1。区分历史DARG最佳A、当前SA最佳epoch关闭残差B、完整SA模型C的A→B、B→C、A→C纠错/误伤。机制复核Prompt-off、Null-only、替换分组、单Prompt删除、RES global-off，残差投影与分类器Hidden/Logit margin变化。Prompt-off会关闭整个分支，其翻转不能单独证明医学语义价值；潜在槽不是脑区。Fold2的历史重叠、test=val、预CV预处理/RFE及transductive图仍保留，因此即使通过也只支持当前协议下的候选方法，不证明独立临床泛化。

科研判定按预注册规则：G1未同时超过DARG ACC和Hard-AUC则`NO_GO_SA_PRCL_BASELINE`；G1未明确超过G2则`SEMANTIC_GROUPING_INCREMENT_NOT_ESTABLISHED`；主干更新身份不一致则`BACKBONE_PROTECTION_AUDIT_FAIL`。仅完整性能、分组、保护与独立审计都支持时判`PRELIMINARY_GO_SA_PRCL`。

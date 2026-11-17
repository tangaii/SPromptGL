# Experiment 81.6.1 — CPRL-RES

差异化疾病Prompt引导的局部关系学习与全局–局部融合。唯一训练基座是E74.4-G2（DARG）。固定三组、seed=0、10折，不搜索超参数。所有历史文件只读；正式目录因首选OSSFS创建报errno=5，回退至当前目录，错误在runtime/storage_error.json。

## 编码前完成的来源复核

[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)第2.1、2.2节描述多关系患者图GFR、模态CAM、PEN内的SPGS与RES。公式(5)以局部与类别语义的最大相似度选择局部；公式(6)抽象表示局部–全局融合。公式不是实际张量实现的逐行说明。

[官方network.py](https://github.com/wanxixi11/SPromptGL/blob/main/network.py)的PLN.forward才是数值参照：RN50输出8×1024，先把前后各4条拼为两条4096维输入，经MLP映射到9维；CAM划为[N,4,4,9]，归一化余弦经两类Softmax后取最大值，缩放局部。每个连续36维槽位AvgPool1d(4,4)变为9维全局；循环计算softmax(part*global,dim=-1)*part，再按原局部顺序拼接加回。代码的逐坐标实现与论文矩阵写法明确区分。官方GFR还存在代码级模态累积细节；本实验调用历史DARG实际实现，不按论文重写。

[layers.py](https://github.com/wanxixi11/SPromptGL/blob/main/layers.py)的OutputLayer忽略attn_embedding，执行LayerNorm、Linear、GELU、Dropout和分类器。历史实例是144→36→2，Dropout=.35。[main.py](https://github.com/wanxixi11/SPromptGL/blob/main/main.py)固定StratifiedKFold(random_state=10)，严格ACC提升才覆盖Best，平局保留早期结果。本次不重新启用原PEN。

[KD-Brain源码](https://github.com/LiuSy0731/KDBrain/blob/main/KDBrain/models/model.py)的Attention_AggregateLayer.forward：text_projector(768,out_shape)经tanh，text_gate初值.5，先构造guided_query再进入关系评分。本次只借鉴语义先验直接参与关系评分；不复用真实子网络、BioMedBERT、多层聚合或其监督目标。本次使用独立归一化语义评分，固定lambda_sem=1，与其Query相加实现有区别。

[ItemizedCLIP模型](https://github.com/MLNeurosurg/ItemizedCLIP/blob/main/Radiology_Tasks/src/open_mri/my_model.py)的PureAttentionPoolingBlock.forward分别处理文本Query与视觉Key/Value；[损失源码](https://github.com/MLNeurosurg/ItemizedCLIP/blob/main/Radiology_Tasks/src/open_mri/itemizedclip_loss.py)的_loss_with_attn_pool生成每条文本条件化视觉表示。本次仅借鉴多个条件分别保留表示的思想。ABIDE没有患者逐条文本真值，不移植ILA、IIS、KTA、MPS、normalcy掩码或随机token掩码。

以上源码通过GitHub连接读取，保存原始内容、Git blob身份；论文SHA256与E81.6逐文件匹配证据在runtime/source_review.json、runtime/reviewed_sources.json。官方main分支文件按实际读取blob锁定。

## E81.6真实问题与本次改变

指定提交75e11ac262a3789aa3595748800f4aed1ae3cc7a的9个指定源码/配置/结果文件与/model内历史副本逐个Git blob相同。已读prcl_res.py、run.py、audit.py、path_adapter.py、config、design和报告/机制/审计状态。G1 ACC91.8443%、BA91.7615%，低于DARG92.1891%、92.0294%。10折机制均值：Prompt注意力非对角MAE .00158923；正常/异常消息MAE .01047334；Prompt-off概率MAE .000165961，硬预测翻转0；有效残差/CAM .09087165，raw范数1.09360363、CAM范数4.34497926；原文本非对角余弦约.769207。

旧法把gamma*文本投影附加到视觉Query，直接聚合携带大量共有视觉内容的消息，并使用competition。新法确定性分离文本共同中心，显式分开视觉和语义关系评分，保留两组相对视觉消息的增量，删除gamma、competition、L_geo。原来L_res约束受限delta，本次统一约束raw_delta，因此G1相对旧E81.6不是单变量对照。放大注意力差异或残差均不自动表示疾病语义有效。

## 冻结结构与三组

原8条及顺序保存在cprl_res.py与缓存，包括首条Nromal拼写。冻结RN50在任何历史RNG恢复前只编码一次，结果逐元素核对E81.6原医学向量。t=normalize(T)，共同中心mean(t)，centered=normalize(t-mean(t))；无标签。记录中心化前后Gram、非零范数与组方向。中心化只突出原有差别，没有生成医学知识。

z_rel=detach(z_cam)+.1*(z_cam-detach(z_cam))；CAM144→[N,4,4,9]→16个9维局部。共享Linear(9,32)+position、LayerNorm。接收p、发送q不自连：Key输入[u_p*u_q,abs(u_q-u_p)]，Value输入[u_q,u_p*u_q,u_q-u_p]，共享64→32、96→32投影。15个不同发送者加Null；Null Key可学习，Value恒零。120无向对、240方向；不新增视觉编码器或8个普通注意力头。

视觉Query=LayerNorm(Wq*u)，视觉分数=dot(Query,Key)/sqrt(32)。文本Query=normalize(Linear(1024,32,bias=False)(centered))；语义分数=dot(文本Query,normalize(Key))，Null语义分数恒0。两分数直接相加，lambda_sem固定1，不做逐患者标准差归一化。相同Key/Value/Null同时计算视觉及8条Prompt注意力。

逐条消息m_k与视觉m_v使用同一批次缩并保证相同注意力时数值严格相同；delta_m=m_k-m_v。两组各自平均后，[delta_normal,delta_abnormal]经共享无偏置64→32融合。差值delta_abnormal-delta_normal只诊断；若只用差值，视觉基准会相消，不能将减基准当作独立信息。本次保留两个独立32维增量，不使用第二个分类器。

关系经严格零初始化无偏置32→9输出，恢复[N,4,4,9]。全局按原槽位AvgPool1d(4,4)，逐坐标softmax(d*global)*d，按原序展平raw_delta144。没有144→32全局Linear、Global Cross-Attention或解剖脑区解释。

|组|分组|最终映射|
|---|---|---|
|G1 Medical-CPRL-10|normal=[0,1,2,3]，abnormal=[4,5,6,7]|bound=.1*norm(detach(CAM))；delta=bound*raw/sqrt(bound²+sum(raw²)+1e-12)|
|G2 Medical-CPRL-Unbounded|同G1|delta=raw_delta，完全无上限|
|G3 Scrambled-CPRL-10|normal=[0,1,4,5]，abnormal=[2,3,6,7]|同G1|

G3与G1向量、顺序、每条Query/注意力/消息及全部初始化相同；仅改变固定聚合索引。每组各混合原正常/异常各2条，不按结果选择。相较Gram匹配匿名向量，它避免可被自由投影吸收的正交变换混杂。但只检验原4+4组织的增量，不能证明医学语义完全正确，也不是无医学文本对照。

Prompt-off将全部语义分数严格置零，必须有attention_prompt==attention_visual、delta_m==0、两组增量==0、relation==0、raw_delta==0、enhanced==CAM，初始/训练后、非零输出头和最终Logp均测试。Null-only同样严格归零。这是结构依赖，需要G1/G3重训判断诊断价值。

三组统一L_total=历史total+.001*mean_legal_train(raw_ratio²)，raw_ratio=norm(raw_delta)/(norm(detach(CAM))+1e-12)。无L_geo。G1/G2在相同输入参数下raw正则严格相同；唯一最终残差映射不同。

## 完整历史训练继承与隔离

真实调用链：本次path_adapter→E74.4.run.prepare→E74.1.run.prepare→E69.3加载E61→E61.load_warmup_state→E53.3历史build_legacy_core、数据与anchor。恢复E61完整epoch10状态及三个优化器，沿用历史确定性gate配置、Fold2 additions与重新生成后SHA校验的4096关系对；不加载E81.*最佳状态作为起点，也不重训DARG或warmup。

原始数据为871×320，48/6/10/256四模态、ASD0/TD1；直接读历史processed_standard_data.csv、modal_feat_dict.npy，固定10折，预CV预处理/RFE不重做。原transductive图、test=val选模、Fold2行377/407重叠保留。训练只对新增残差的零基376/406清零并排除其raw正则；测试对所有行完整应用。此保护不消除历史信息泄漏。

原total=Cls+graph+.05*pair+.01*adv+.01*preserve，原实现与权重不变。原GFR/CAM Adam lr .0038 wd .11；DARG AdamW lr .001 wd .01；新增AdamW lr .0005 wd .01，最后更新。起点epoch10，训练到最多450，wait>200停止，严格test ACC提升选择Best、平局保早。每任务独立恢复Python、NumPy、Torch CPU、CUDA RNG，不逐epoch重置。

新增模块CPU私有fork_rng(devices=[])，salt=816100+fold；不扰动公共RNG，无新增Dropout或采样。同折三组所有可训练参数完全同值。每epoch记录训练前/训练后/Eval后公共RNG SHA，比较DARG及三组共有epoch，不以参数轨迹差异判断RNG失败。

原OutputLayer仅执行一次，通过前置Hook替换输入。历史模块仅只读调用；禁用pyc写入。源码清单含实际vendor依赖、数据、warmup、历史Best、结果、原根代码、CLIP/tokenizer与当前全部配置代码SHA。仅保存必要Best/Latest、紧凑标量/RNG日志、任务状态。

## 审计、冻结与判定

Preflight覆盖P1–P10：来源身份；逐折参数与优化器/RNG；全部维度；初始及非零头语义归零；文本独立性；官方RES数值；10折CAM/Logp/Hidden/loss/梯度/三优化器单步/RNG零残差恢复；首步/后续梯度、eta、无上限与正则；序列化两步恢复及四优化器/attention/增量/raw/effective状态；最后源码冻结。预检修正仅在正式冻结前允许，不依结果改模型或超参数。

先完成单worker真实训练测峰值，再依当前空闲显存调度，最多10 workers，每worker一CPU线程。每epoch原子替换日志并读回；Latest保留恢复RNG、四优化器、Best及完整日志，失败从Latest严格重放。G2数值发散不得偷偷加限幅、裁剪或门控。

独立重载30 Best/Latest，六指标独立计算、重验历史10折起点及同折初始化，检查Best严格选模、完整日志和RNG mismatch=0；训练后Prompt-off/Null恒等、Fold2保护、G2 raw==delta。历史DARG及E81.6-G1最佳Checkpoint只读重算。G1最佳Checkpoint额外一次unbounded推理仅作分布变化诊断，不能替代G2。

全部/去Fold2报告10/9折均值±population SD和逐折配对描述性区间。分开判断G1超过DARG、优于E81.6机制与分类、G2幅度效果、G1/G3原4+4组织价值。未同时超过DARG ACC/BA即NO_GO_CPRL_BASELINE。最终PRELIMINARY_GO_CPRL_RES还要求去Fold2支持、明确Prompt作用、分组有力增益、无严重主干共同适配及全审计通过；否则不预设正结果。单seed、重叠CV、test=val和预CV RFE不能当作独立泛化验证。

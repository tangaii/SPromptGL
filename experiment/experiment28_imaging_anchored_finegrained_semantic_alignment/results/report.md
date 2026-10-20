# 实验28：IA-FDSA影像锚定的细粒度疾病语义对齐

状态：**COMPLETE**；实际完成 **300/300折**；科学结论：**NO-GO**。

本轮验证细粒度FC文本、正确解剖身份/方向及正确辅助患者配对能否提供额外诊断信息。未按测试成绩修改Prompt、损失、网络或配置。

## 审计、来源与历史教训

本地HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；GitHub main `8217b36bfe7f51d878b793c070774129d11cad73`。六核心源码逐字节一致；本轮仅在独立运行副本应用 integration.patch。
原GFR四张受试者图、i==1覆盖、阈值0.9、四图正则平均、ALoss不反传、CAM特殊归一化及原RES/分类器均保留。CAM的4×36及4×4×9都是潜在Token，不能解释为具体ROI。ANAT6/FUNC10是扫描质量指标。
原SPGS把两类相似度Softmax后取max，类别位置交换不改变权重。此前原型趋同、自由Key重参数化、中性/随机控制不劣、PHENO强制对齐、ROI旁路、QC校准及患者增益路由失败，是本轮设立严格负对照的动机。
MMGPL已经提出概念权重和概念响应Token图（III-C/III-D，Eq.7–14）；SPromptGL提出GFR/CAM/SPGS/RES（Eq.1–7）。本轮不宣称上述机制首创，也不把文本可区分当作医学知识有效。
文献：[MMGPL](https://arxiv.org/html/2312.14574v2)；[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)。MMGPL输入/样本数与协议不同，不能直接横比其ACC。

## 冻结文本及教师

保留原8句（含Nromal拼写）。依据实验11最终connection_name、atlas端点和验证列，252连接×增强/减弱=504句；195/225/232/236四条仍在原CORR256中，但语义权重严格为0。完整文本、索引、文本SHA和原始/低维向量分别在prompt_banks.json/npz。
RN50冻结，P0八句独立编码并与实验12旧缓存逐值相同；512句共同单位归一化后做无中心SVD取64维，列符号固定，再单位归一化，不使用患者数据或标签。
E2两种ASD关联方向均使用原异常四句平均向量；原正常四句参与共同文本基底。它是刻意粗粒度/近恒定教师，FC降低不能等同正常。
数值对照E5：seed28005随机正交行坐标染色，最终归一化之前均值/协方差严格匹配文本，之后逐向量单位范数；个体配对Gram不相同，记录实际几何差异。E6 seed28006无固定点连接错配；E7仅交换方向。
每fold TD均值/总体标准差仅来自train TD；d优先在同站点ASD/TD各至少2人时估计，先截断标准化差异至±2，按有效样本/(有效样本+20)收缩，再按调和有效样本权重汇总；无合格站点时才使用同规则全train估计。报告站点异质性，不能称病灶真值。
r截断±3；a=ReLU(sign(d)r−0.5)×clip(|d|,0,2)。教师是252对应文本的加权单位向量，全部a为0时跳过。语义损失0.05×SmoothL1，仅train有效患者，不读取test标签。

## 模型、协议与预检

FC256→Linear64/LayerNorm/GELU；CAM CORR36→同类64维映射；anchor=FC+.25context。demo= tanh(Linear([年龄/性别28,anchor64]))；QA分别6/10→64 sigmoid；joint=anchor+.25qa×qf×demo。decode64→36；delta=.10tanh(alpha)tanh(decode)，alpha=0后安装。
仅CORR CAM位置增加残差，再进入无SPGS权重的原RES。每坐标绝对残差≤0.10，36维L2≤0.6；本公式不是相对原特征10%的约束。alpha=0与E1逐值相等，不能误称与P0相等。
E2–E9均注册28517个新参数，原1462957参数及原MLP注册顺序保留；E1–E9原MLP不参与前向。新分支在原my_weight_init之后fork_rng安装，新参数Adam lr=.0038/wd=0；原两Adam及原wd=.11保持。E2/E3的辅助模块注册但不参与有效贡献，实际梯度另报。
原始协议：871人，403 ASD(0)/468 TD(1)，48/6/10/256列；10fold StratifiedKFold(shuffle=True,random_state=10)，seed0/1/2，每任务只播种一次连续10折；450epochs、wait>200、首次严格test/val ACC最高选模；dropout.35、18hidden、2heads、1layer，GC weighted-cosine/MP GCN/MF concat/simple-2。
CLIP首次构建消耗的初始化RNG按实验13保留；之后使用固定缓存。新增模块不消耗原主干RNG；不同组早停后续折可能自然分叉，同seed/fold不代表完全同初始化。
预检结果：`{"passed": true, "completed_utc": "2026-10-19T14:33:22.647104+00:00", "checks": {"P0_unpatched_init_forward_two_adam_first_update_rng": "EXACT_PASS", "P0_seed0_fold1_initial_hash": "b464f86ee750e5550a6330942ee102a5d9d1f3ab4967a385e23655741c92ae0f", "new_groups_same_original_parameters_rng_zero_alpha_E1": "EXACT_PASS", "new_parameters_all_eight_groups": 28517, "unknown_edges_zero_semantic_weight": "PASS", "heldout_label_flip_teacher_full_train_update": "EXACT_PASS", "two_step_gradient_norms": {"gfr": 1.5985847042013066, "pln_original": 3.4424670623385034, "alpha": 0.006875439435169571, "fc_encoder": 0.0004854666678527849, "context_encoder": 0.0002692793248830235, "demo_encoder": 9.518379110455574e-06, "anat_gate": 2.6161175959745273e-07, "func_gate": 2.7062072764806413e-07, "decode": 0.00026451171444845427, "active_parameters": 760755}, "epoch_resume_model_adam_rng_prediction": "EXACT_PASS", "fold_boundary_next_initialization_rng": "EXACT_PASS", "counterfactual_restore_weights_rng_REAL": "EXACT_PASS", "device": "_CudaDeviceProperties(name='PPU-ZW810E', major=8, minor=0, total_memory=98304MB, multi_processor_count=64, uuid=019e4201-0591-0030-0000-000080387166, pci_bus_id=99, pci_device_id=0, pci_domain_id=0, L2_cache_size=64MB)"}}`。
完整P0历史parity：`{"passed": true, "rows": 30, "fields": ["acc", "ba", "probability_auc", "best_epoch", "initial_trunk_sha256"], "checked_utc": "2026-10-19T14:46:34.585349+00:00", "maximum_acc_ba_auc_error": 0.0, "initial_state_and_rng_preflight": true}`。只有通过完整30折ACC/BA/ProbAUC/best epoch/初始化逐项核验后才允许其他270折。

## 十组正式设计

|组|方法|SPGS|语义监督|辅助修正|
|---|---|---|---|---|
|E0_P0|Original P0|True|False|False|
|E1_NO_SPGS|No SPGS|False|False|False|
|E2_OLD_FC|Original coarse FC teacher|False|True|False|
|E3_FINE_FC|Fine FC teacher|False|True|False|
|E4_FINE_FULL|Fine full conditional fusion|False|True|True|
|E5_NUMERIC_FULL|Frozen numeric codes|False|True|True|
|E6_WRONG_ROI|Wrong ROI identity|False|True|True|
|E7_WRONG_DIRECTION|Wrong FC direction|False|True|True|
|E8_FINE_NO_ALIGN|No semantic training loss|False|False|True|
|E9_WRONG_AUX_PAIR|Wrong within-site auxiliary patient|False|True|True|

## 实测主结果（先seed十折平均，再三seed均值±population SD，%）

|组|ACC|BA/Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|
|---|---:|---:|---:|---:|---:|---:|
|E0_P0|91.69105±0.05418|91.53935±0.08408|93.33520±0.41541|89.66057±0.64562|93.41813±0.51023|90.93289±0.12038|
|E1_NO_SPGS|91.53779±0.19536|91.34905±0.20625|93.57039±0.30428|88.92073±0.71450|93.77737±0.60617|90.70334±0.21945|
|E2_OLD_FC|91.53779±0.05418|91.34309±0.06147|93.55219±0.26848|88.83943±0.19917|93.84675±0.10248|90.70297±0.07397|
|E3_FINE_FC|91.69105±0.05418|91.47256±0.07651|93.38635±0.15774|88.66667±0.65496|94.27845±0.53157|90.81486±0.10378|
|E4_FINE_FULL|91.61442±0.16255|91.42719±0.17620|93.19882±0.50570|89.00610±0.45706|93.84829±0.10466|90.77521±0.23647|
|E5_NUMERIC_FULL|91.46160±0.30108|91.27532±0.34054|93.35904±0.15532|88.91667±0.92844|93.63398±0.26743|90.62720±0.39481|
|E6_WRONG_ROI|91.42285±0.19536|91.23025±0.19801|93.25822±0.16141|88.75407±0.42892|93.70644±0.35952|90.57917±0.22862|
|E7_WRONG_DIRECTION|91.46160±0.21612|91.29334±0.26263|93.42765±0.15626|89.09146±0.88464|93.49522±0.36620|90.63693±0.32649|
|E8_FINE_NO_ALIGN|91.30834±0.14289|91.11195±0.14899|93.37431±0.19609|88.59146±0.52689|93.63244±0.44470|90.43667±0.18806|
|E9_WRONG_AUX_PAIR|91.57654±0.21612|91.38426±0.22948|93.49009±0.14858|88.91870±0.64688|93.84983±0.44094|90.74542±0.25832|

历史实验13 P0只读参照：acc 91.69105±0.05418；ba 91.53935±0.08408；probability_auc 93.33520±0.41541；sensitivity_asd 89.66057±0.64562；specificity_td 93.41813±0.51023；f1_asd 90.93289±0.12038。
历史实验9 E9：ACC91.88262%、BA91.69151%、Prob-AUC93.82272%，属于不同基座的历史描述性比较，绝不是本轮E9_WRONG_AUX_PAIR。

## 三个seed十折均值及病例计数

|组|seed|ACC|BA|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|ASD FN|TD FP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E0_P0|0|91.72936|91.56250|93.89057|89.34146|93.78353|90.92433|43|29|
|E0_P0|1|91.61442|91.42678|92.89162|89.07927|93.77428|90.78992|44|29|
|E0_P0|2|91.72936|91.62878|93.22340|90.56098|92.69658|91.08443|38|34|
|E1_NO_SPGS|0|91.26959|91.08975|93.20469|88.83537|93.34413|90.44780|45|31|
|E1_NO_SPGS|1|91.72936|91.59437|93.94965|89.83537|93.35338|90.98363|41|31|
|E1_NO_SPGS|2|91.61442|91.36303|93.55684|88.09146|94.63460|90.67859|48|25|
|E2_OLD_FC|0|91.49948|91.29462|93.45313|88.59756|93.99167|90.64297|46|28|
|E2_OLD_FC|1|91.49948|91.30482|93.28428|88.83537|93.77428|90.65876|45|29|
|E2_OLD_FC|2|91.61442|91.42982|93.91915|89.08537|93.77428|90.80718|44|29|
|E3_FINE_FC|0|91.72936|91.45795|93.17544|88.07317|94.84274|90.79285|48|24|
|E3_FINE_FC|1|91.61442|91.38701|93.42885|88.34756|94.42646|90.70020|47|26|
|E3_FINE_FC|2|91.72936|91.57271|93.55475|89.57927|93.56614|90.95153|42|30|
|E4_FINE_FULL|0|91.72936|91.54873|93.38848|89.32317|93.77428|90.93163|43|29|
|E4_FINE_FULL|1|91.38454|91.17803|92.50681|88.35976|93.99630|90.44101|47|28|
|E4_FINE_FULL|2|91.72936|91.55482|93.70116|89.33537|93.77428|90.95299|43|29|
|E5_NUMERIC_FULL|0|91.04101|90.79767|93.14914|87.60366|93.99167|90.07312|50|28|
|E5_NUMERIC_FULL|1|91.72936|91.56734|93.40791|89.57317|93.56152|90.96387|42|30|
|E5_NUMERIC_FULL|2|91.61442|91.46096|93.52005|89.57317|93.34875|90.84462|42|31|
|E6_WRONG_ROI|0|91.49948|91.26454|93.44069|88.32927|94.19981|90.61017|47|27|
|E6_WRONG_ROI|1|91.15465|90.97242|93.04821|88.59146|93.35338|90.28495|46|31|
|E6_WRONG_ROI|2|91.61442|91.45380|93.28576|89.34146|93.56614|90.84238|43|30|
|E7_WRONG_DIRECTION|0|91.61442|91.48031|93.61050|89.82927|93.13136|90.89239|41|32|
|E7_WRONG_DIRECTION|1|91.15596|90.92193|93.44369|87.84756|93.99630|90.17610|49|28|
|E7_WRONG_DIRECTION|2|91.61442|91.47778|93.22876|89.59756|93.35800|90.84228|42|31|
|E8_FINE_NO_ALIGN|0|91.26959|91.03441|93.11862|87.85976|94.20907|90.27133|49|27|
|E8_FINE_NO_ALIGN|1|91.15596|90.98105|93.40919|88.83537|93.12673|90.33894|45|32|
|E8_FINE_NO_ALIGN|2|91.49948|91.32039|93.59512|89.07927|93.56152|90.69975|44|30|
|E9_WRONG_AUX_PAIR|0|91.27090|91.06323|93.53274|88.34756|93.77891|90.39042|47|29|
|E9_WRONG_AUX_PAIR|1|91.72936|91.58596|93.64694|89.82317|93.34875|90.99760|41|31|
|E9_WRONG_AUX_PAIR|2|91.72936|91.50360|93.29058|88.58537|94.42183|90.84825|46|26|

## 全部预设配对（pp）

|比较|指标|均差pp|seed0/1/2差pp|逐折胜/平/负|
|---|---|---:|---|---|
|E3_FINE_FC − E2_OLD_FC|acc|+0.15326|+0.22989,+0.11494,+0.11494|8/16/6|
|E3_FINE_FC − E2_OLD_FC|ba|+0.12947|+0.16334,+0.08218,+0.14288|10/12/8|
|E3_FINE_FC − E2_OLD_FC|hard_auc|+0.12947|+0.16334,+0.08218,+0.14288|10/12/8|
|E3_FINE_FC − E2_OLD_FC|probability_auc|-0.16584|-0.27769,+0.14457,-0.36440|12/0/18|
|E3_FINE_FC − E2_OLD_FC|sensitivity_asd|-0.17276|-0.52439,-0.48780,+0.49390|6/17/7|
|E3_FINE_FC − E2_OLD_FC|specificity_td|+0.43170|+0.85106,+0.65217,-0.20814|7/18/5|
|E3_FINE_FC − E2_OLD_FC|f1_asd|+0.11189|+0.14987,+0.04144,+0.14435|10/12/8|
|E4_FINE_FULL − E3_FINE_FC|acc|-0.07663|+0.00000,-0.22989,+0.00000|9/12/9|
|E4_FINE_FULL − E3_FINE_FC|ba|-0.04536|+0.09077,-0.20898,-0.01788|11/9/10|
|E4_FINE_FULL − E3_FINE_FC|hard_auc|-0.04536|+0.09077,-0.20898,-0.01788|11/9/10|
|E4_FINE_FULL − E3_FINE_FC|probability_auc|-0.18753|+0.21304,-0.92204,+0.14641|14/1/15|
|E4_FINE_FULL − E3_FINE_FC|sensitivity_asd|+0.33943|+1.25000,+0.01220,-0.24390|7/19/4|
|E4_FINE_FULL − E3_FINE_FC|specificity_td|-0.43016|-1.06846,-0.43016,+0.20814|6/14/10|
|E4_FINE_FULL − E3_FINE_FC|f1_asd|-0.03965|+0.13879,-0.25919,+0.00146|11/9/10|
|E4_FINE_FULL − E5_NUMERIC_FULL|acc|+0.15282|+0.68835,-0.34483,+0.11494|13/9/8|
|E4_FINE_FULL − E5_NUMERIC_FULL|ba|+0.15187|+0.75106,-0.38932,+0.09386|15/5/10|
|E4_FINE_FULL − E5_NUMERIC_FULL|hard_auc|+0.15187|+0.75106,-0.38932,+0.09386|15/5/10|
|E4_FINE_FULL − E5_NUMERIC_FULL|probability_auc|-0.16022|+0.23934,-0.90110,+0.18111|15/1/14|
|E4_FINE_FULL − E5_NUMERIC_FULL|sensitivity_asd|+0.08943|+1.71951,-1.21341,-0.23780|14/8/8|
|E4_FINE_FULL − E5_NUMERIC_FULL|specificity_td|+0.21431|-0.21739,+0.43478,+0.42553|8/16/6|
|E4_FINE_FULL − E5_NUMERIC_FULL|f1_asd|+0.14801|+0.85851,-0.52286,+0.10838|15/5/10|
|E4_FINE_FULL − E6_WRONG_ROI|acc|+0.19157|+0.22989,+0.22989,+0.11494|10/15/5|
|E4_FINE_FULL − E6_WRONG_ROI|ba|+0.19694|+0.28419,+0.20561,+0.10102|14/10/6|
|E4_FINE_FULL − E6_WRONG_ROI|hard_auc|+0.19694|+0.28419,+0.20561,+0.10102|14/10/6|
|E4_FINE_FULL − E6_WRONG_ROI|probability_auc|-0.05940|-0.05220,-0.54140,+0.41540|13/0/17|
|E4_FINE_FULL − E6_WRONG_ROI|sensitivity_asd|+0.25203|+0.99390,-0.23171,-0.00610|10/15/5|
|E4_FINE_FULL − E6_WRONG_ROI|specificity_td|+0.14184|-0.42553,+0.64292,+0.20814|8/15/7|
|E4_FINE_FULL − E6_WRONG_ROI|f1_asd|+0.19605|+0.32146,+0.15607,+0.11061|14/10/6|
|E4_FINE_FULL − E7_WRONG_DIRECTION|acc|+0.15282|+0.11494,+0.22858,+0.11494|12/11/7|
|E4_FINE_FULL − E7_WRONG_DIRECTION|ba|+0.13385|+0.06841,+0.25610,+0.07704|13/8/9|
|E4_FINE_FULL − E7_WRONG_DIRECTION|hard_auc|+0.13385|+0.06841,+0.25610,+0.07704|13/8/9|
|E4_FINE_FULL − E7_WRONG_DIRECTION|probability_auc|-0.22883|-0.22202,-0.93689,+0.47240|15/0/15|
|E4_FINE_FULL − E7_WRONG_DIRECTION|sensitivity_asd|-0.08537|-0.50610,+0.51220,-0.26220|7/16/7|
|E4_FINE_FULL − E7_WRONG_DIRECTION|specificity_td|+0.35307|+0.64292,-0.00000,+0.41628|10/13/7|
|E4_FINE_FULL − E7_WRONG_DIRECTION|f1_asd|+0.13829|+0.03924,+0.26491,+0.11071|13/8/9|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|acc|+0.30608|+0.45977,+0.22858,+0.22989|10/15/5|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|ba|+0.31524|+0.51432,+0.19698,+0.23443|11/12/7|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|hard_auc|+0.31524|+0.51432,+0.19698,+0.23443|11/12/7|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|probability_auc|-0.17549|+0.26987,-0.90238,+0.10604|17/0/13|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|sensitivity_asd|+0.41463|+1.46341,-0.47561,+0.25610|7/18/5|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|specificity_td|+0.21585|-0.43478,+0.86957,+0.21277|9/14/7|
|E4_FINE_FULL − E8_FINE_NO_ALIGN|f1_asd|+0.33854|+0.66030,+0.10208,+0.25324|11/12/7|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|acc|+0.03788|+0.45846,-0.34483,+0.00000|10/13/7|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|ba|+0.04293|+0.48549,-0.40793,+0.05123|12/10/8|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|hard_auc|+0.04293|+0.48549,-0.40793,+0.05123|12/10/8|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|probability_auc|-0.29127|-0.14425,-1.14013,+0.41058|12/0/18|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|sensitivity_asd|+0.08740|+0.97561,-1.46341,+0.75000|9/15/6|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|specificity_td|-0.00154|-0.00463,+0.64755,-0.64755|7/16/7|
|E4_FINE_FULL − E9_WRONG_AUX_PAIR|f1_asd|+0.02979|+0.54121,-0.55658,+0.10474|12/10/8|
|E4_FINE_FULL − E1_NO_SPGS|acc|+0.07663|+0.45977,-0.34483,+0.11494|11/11/8|
|E4_FINE_FULL − E1_NO_SPGS|ba|+0.07814|+0.45898,-0.41634,+0.19179|12/10/8|
|E4_FINE_FULL − E1_NO_SPGS|hard_auc|+0.07814|+0.45898,-0.41634,+0.19179|12/10/8|
|E4_FINE_FULL − E1_NO_SPGS|probability_auc|-0.37157|+0.18380,-1.44284,+0.14432|11/2/17|
|E4_FINE_FULL − E1_NO_SPGS|sensitivity_asd|+0.08537|+0.48780,-1.47561,+1.24390|9/15/6|
|E4_FINE_FULL − E1_NO_SPGS|specificity_td|+0.07092|+0.43016,+0.64292,-0.86031|6/18/6|
|E4_FINE_FULL − E1_NO_SPGS|f1_asd|+0.07187|+0.48383,-0.54262,+0.27440|12/10/8|
|E4_FINE_FULL − E0_P0|acc|-0.07663|+0.00000,-0.22989,+0.00000|8/15/7|
|E4_FINE_FULL − E0_P0|ba|-0.11216|-0.01377,-0.24875,-0.07395|9/13/8|
|E4_FINE_FULL − E0_P0|hard_auc|-0.11216|-0.01377,-0.24875,-0.07395|9/13/8|
|E4_FINE_FULL − E0_P0|probability_auc|-0.13638|-0.50209,-0.38481,+0.47776|15/0/15|
|E4_FINE_FULL − E0_P0|sensitivity_asd|-0.65447|-0.01829,-0.71951,-1.22561|5/19/6|
|E4_FINE_FULL − E0_P0|specificity_td|+0.43016|-0.00925,+0.22202,+1.07771|8/17/5|
|E4_FINE_FULL − E0_P0|f1_asd|-0.15768|+0.00730,-0.34891,-0.13144|9/13/8|

E4相对历史实验9 E9（描述性）：acc -0.26820pp, ba -0.26432pp, probability_auc -0.62390pp。

## 文本、教师和辅助机制诊断

文本几何：`{"raw": {"norm_min": 9.794093608889419, "norm_mean": 10.376516266150633, "norm_max": 13.710279230266735, "centered_variance": 0.08364083685783404, "effective_rank": 29.051132463907784, "numerical_rank": 511, "pair_cos_quantiles": [0.5225082266013156, 0.8887505981167093, 0.925938721632477, 0.9547838639559869, 0.997640866201045]}, "projected": {"norm_min": 0.9999999999999997, "norm_mean": 1.0, "norm_max": 1.0000000000000002, "centered_variance": 0.08283143371247412, "effective_rank": 27.03397452500813, "numerical_rank": 64, "pair_cos_quantiles": [0.5227756606969707, 0.8896592910905219, 0.9267577005565006, 0.9555842301664368, 0.9977248729625023]}, "fine_raw": {"norm_min": 9.794093608889419, "norm_mean": 10.341716032927323, "norm_max": 10.950262550415212, "centered_variance": 0.07446059704075157, "effective_rank": 26.645063329620633, "numerical_rank": 503, "pair_cos_quantiles": [0.8180305190495268, 0.8934776470651763, 0.9269122223806735, 0.955178186173889, 0.997640866201045]}, "fine_projected": {"norm_min": 0.9999999999999997, "norm_mean": 1.0, "norm_max": 1.0000000000000002, "centered_variance": 0.07363510975977992, "effective_rank": 24.578394092096598, "numerical_rank": 64, "pair_cos_quantiles": [0.8193854799030947, 0.8943645588006898, 0.9277300330176337, 0.9559720976185135, 0.9977248729625023]}, "numeric": {"norm_min": 0.9999999999999997, "norm_mean": 1.0, "norm_max": 1.0000000000000002, "centered_variance": 0.07310281127799992, "effective_rank": 24.893767936593026, "numerical_rank": 64, "pair_cos_quantiles": [0.7435100119732655, 0.8950030887350967, 0.9303664250054291, 0.953917856577828, 0.9817661368891923]}, "fine_direction_cos_raw_mean": 0.9965796151420921, "fine_direction_cos_projected_mean": 0.9967063740706494, "coarse_normal_abnormal_cos": 0.9070489727901485, "wrong_roi_distance": 0.3794048138073989, "projected_energy_fraction": 0.9991192104919602, "numeric_pre_normalization_covariance_error": 2.6645352591003757e-14, "numeric_pair_gram_abs_difference": 0.026487362269749126}`。
统一短模板选择索引 0；最大token数 45；truncate=False。
|组|教师有效train比例|教师方差|student-teacher cos|alpha|残差L2|相对CAM残差|辅助/anchor|qa/qf患者STD|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|E2_OLD_FC|1.0000000|0.0000000|0.1267318|0.2362772|0.3278320|0.1771005|0.0000000|0.0096932/0.0118914|
|E3_FINE_FC|1.0000000|0.0034642|0.0601062|-0.0007680|0.3147545|0.2577995|0.0000000|0.0096932/0.0118914|
|E4_FINE_FULL|1.0000000|0.0034642|0.0573275|0.2117880|0.3228358|0.1541988|0.1773704|0.0552347/0.1015421|
|E5_NUMERIC_FULL|1.0000000|0.0022071|0.0948345|-0.1206408|0.2999731|0.1295598|0.2252518|0.0540333/0.1035489|
|E6_WRONG_ROI|1.0000000|0.0026789|0.0908512|0.2133827|0.2860861|0.1263476|0.2180009|0.0510710/0.1081470|
|E7_WRONG_DIRECTION|1.0000000|0.0033375|0.0824743|0.1185212|0.3059570|0.1828671|0.1745936|0.0541551/0.1017395|
|E8_FINE_NO_ALIGN|1.0000000|0.0034642|-0.0579852|0.1152349|0.3131919|0.1553784|0.1089248|0.0492386/0.0999235|
|E9_WRONG_AUX_PAIR|1.0000000|0.0034642|0.0612909|0.1432269|0.3120794|0.1515127|0.1930654|0.0572245/0.0727487|

每折diagnostics记录原始FC/FC编码/context/anchor/joint/teacher有效秩与类中心距离、teacher类余弦、语义/分类梯度比、各模块梯度、QA与年龄类别/站点/QC关联、站点差异稳定性及训练统计SHA。QA只是学习的条件系数，未经医学可靠度校准；高teacher类余弦或低方差意味着类别代码/近恒定目标，不能称个体医学知识成功。

## E4同Checkpoint反事实

|干预|ΔASD概率绝对均值|翻转率|ΔACC pp|ΔBA pp|ΔProbAUC pp|残差变化L2|
|---|---:|---:|---:|---:|---:|---:|
|REAL|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|
|SEMANTIC_TEACHER_OFF|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|
|SEMANTIC_RESIDUAL_OFF|0.01399688|0.01529955|-1.52995472|-1.54764013|-0.05681092|0.31380385|
|AUX_OFF|0.00130648|0.00114943|-0.11494253|-0.11157239|0.03004407|0.02787257|
|PHENO_OFF|0.00130648|0.00114943|-0.11494253|-0.11157239|0.03004407|0.02787257|
|ANAT_QC_OFF|0.00048068|0.00191571|-0.11494253|-0.12941852|-0.00530223|0.01085924|
|FUNC_QC_OFF|0.00051913|0.00153257|-0.07662835|-0.08775186|-0.00006957|0.01169262|
|AUX_PAIR_SHUFFLED|0.00061069|0.00038314|-0.03831418|-0.04166667|0.01409791|0.00965722|
|FC_ANCHOR_CONTEXT_OFF|0.00664742|0.00842041|-0.84204110|-0.84588100|0.01930800|0.14404302|

SEMANTIC_TEACHER_OFF只关闭训练损失，不改变推理，预期概率变化严格0；其净作用应看E4−E8完整重训。PHENO_OFF将demo条件修正置0；ANAT/FUNC_QC_OFF把指定门控置1而移除该调制。AUX_PAIR_SHUFFLED只错配新增28+6+10，同站点且train/test内独立，不更改原主干或FC。单Checkpoint关闭分支不是重新训练消融，可能分布外。

## 资源、恢复、完整性

完成任务 30/30，完成fold 300/300，diagnostics 300/300，counterfactual 270/270；GPU0/1 fold数 {'0': 150, '1': 150}。每任务连续十折、最多每GPU4个worker/每worker2线程；资源不足只下调并发。
25epoch及fold边界原子保存两个模型/两个Adam/所有RNG/最佳checkpoint/等待计数/签名及已提交行；fold状态先持久化再标记完成，不混合失败attempt。
逐任务耗时秒：`{"E0_P0_seed0": 758.4221819378436, "E0_P0_seed1": 675.4982658289373, "E0_P0_seed2": 715.5966301560402, "E1_NO_SPGS_seed0": 724.2013568766415, "E1_NO_SPGS_seed1": 730.7728206887841, "E1_NO_SPGS_seed2": 1598.8827958218753, "E2_OLD_FC_seed0": 1356.6969047486782, "E2_OLD_FC_seed1": 1293.9028919674456, "E2_OLD_FC_seed2": 971.613663174212, "E3_FINE_FC_seed0": 1623.1214179322124, "E3_FINE_FC_seed1": 1313.170266482979, "E3_FINE_FC_seed2": 1255.7978079989552, "E4_FINE_FULL_seed0": 834.8987557217479, "E4_FINE_FULL_seed1": 1486.7043199650943, "E4_FINE_FULL_seed2": 1159.8286404497921, "E5_NUMERIC_FULL_seed0": 1732.7575345486403, "E5_NUMERIC_FULL_seed1": 1018.8799349367619, "E5_NUMERIC_FULL_seed2": 1701.5514034181833, "E6_WRONG_ROI_seed0": 675.9066250547767, "E6_WRONG_ROI_seed1": 729.298168964684, "E6_WRONG_ROI_seed2": 745.6419029682875, "E7_WRONG_DIRECTION_seed0": 753.6403955854475, "E7_WRONG_DIRECTION_seed1": 733.8398070782423, "E7_WRONG_DIRECTION_seed2": 777.7721496336162, "E8_FINE_NO_ALIGN_seed0": 901.7212639078498, "E8_FINE_NO_ALIGN_seed1": 951.894286762923, "E8_FINE_NO_ALIGN_seed2": 804.2143075279891, "E9_WRONG_AUX_PAIR_seed0": 939.9437686949968, "E9_WRONG_AUX_PAIR_seed1": 819.9777272865176, "E9_WRONG_AUX_PAIR_seed2": 752.5718774758279}`。
异常/阻塞记录：`[{"time": "2026-10-19T14:56:49.750309+00:00", "stage": "all", "error": "[Errno 2] No such file or directory: '/data/adodas/SPromptGL-main/experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/tasks/E2_OLD_FC_seed2.json'"}, {"time": "2026-10-19T15:16:45.501899+00:00", "stage": "resume", "error": "Unterminated string starting at: line 825 column 7 (char 46112)"}]`。
源码SHA：`{"ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67"}`。
数据及ROI SHA：`{"/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e", "/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv": "4f9cd719d27c1e8b9a6cf8933534c0b796101cacf248e509fcfd3c5868fa3ebf", "/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv": "b383308bb413a94d1646511a9bea5bc16f54c975b938ac4d947d676e357c5eb2"}`。
RN50权重SHA `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`；Prompt缓存 `2a3cf9332ad3d1dbdba70731a46acee43f3f7e52c92fe3ab2366fa90b0d54420`；文本清单 `1643a70f75f51bf10eb36a5745449183c94d4a1949fbca13e816ed5d4a7b6b90`。
实验源码SHA：`{"e28_integration.py": "3b0c6a0114caf87d989e8af7bd905a26c3ab0661c358ecb8bdfc0869c9d585c2", "e28_method.py": "f162285872dcfda9d894ee4b163556a030bda152775720b90629dd26743c3a44", "e28_prompts.py": "0c5f09f23e3c0ca9db1f2292cfd957f04f779bab2eb213d5a97a7a947c6a4cb5", "integration.patch": "77cb8197e6af3329c04b4fa86e093be637d058d0a9b0695f0cf28a932caddb6c", "run.py": "47ae00a8c98015cd9cd82bcb63ef1394343cf1edc09153fc03d2fa3e6f36956a"}`。

## 科学判断、评价偏差与后续

本轮结论 **NO-GO**。预设主要配对须在BA和Prob-AUC同时改善且三个seed方向一致才列候选；单一最高ACC、有效梯度、低对齐损失、可解释连接词或单Checkpoint响应都不构成医学语义独立价值证据。
细粒度Prompt是否优于粗粒度、文本是否优于数值码、正确ROI/方向是否优于错配、语义损失及正确辅助配对是否必要，分别以E3−E2、E4−E5/E6/E7/E8/E9及E4−E3/E1/E0判断；不省略阴性控制。
本协议test=val且test ACC参与checkpoint选择，RFE在当前外层CV之前进行，QC有全队列预处理，全871人transductive传播，并已经历27轮探索，故结果是偏乐观内部证据。共享患者和fold的三个seed不是30个独立队列，不报告伪独立显著性、不称无泄漏泛化或临床因果获益。
FC是RFE保留的Fisher-z稀疏连接；训练差异和候选文本不是已验证个体ASD病灶；QA不等于独立结构/功能成像流。
若严格控制不成立，停止当前静态FC疾病文本路线并保留强基线，优先收集更互补的真实影像/临床模态并开展训练内选模的独立验证；若各项一致成立，也只能作为候选，仍需LOSO/独立队列验证。


## 针对本轮实测结果的逐项判读

与MMGPL的可学习概念权重及Token语义图不同，本轮使用固定真实FC连接文本，并由训练折参考与患者FC数值确定教师，监督64维FC影像锚；没有新增自由概念Query或语义图。与原SPromptGL两主题压缩成Softmax-Max标量不同，E1–E9去掉该权重，E2–E9以有界CORR残差参与原RES；因此必须同时比较无SPGS的E1和完整P0，不能把旁路结构变化归为文本收益。
按三seed均值，ACC最高为E0_P0，BA最高为E0_P0，连续Prob-AUC最高为E1_NO_SPGS；这些排名不改变预设主方法E4及其判据。
细粒度相对原8句粗粒度教师：ΔACC=+0.15326pp、ΔBA=+0.12947pp、ΔProb-AUC=-0.16584pp；BA三seed=+0.1633/+0.0822/+0.1429pp，Prob-AUC三seed=-0.2777/+0.1446/-0.3644pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
完整辅助条件相对CORR语义主锚：ΔACC=-0.07663pp、ΔBA=-0.04536pp、ΔProb-AUC=-0.18753pp；BA三seed=+0.0908/-0.2090/-0.0179pp，Prob-AUC三seed=+0.2130/-0.9220/+0.1464pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
正确医学文本相对数值码：ΔACC=+0.15282pp、ΔBA=+0.15187pp、ΔProb-AUC=-0.16022pp；BA三seed=+0.7511/-0.3893/+0.0939pp，Prob-AUC三seed=+0.2393/-0.9011/+0.1811pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
正确FC连接身份相对错配ROI：ΔACC=+0.19157pp、ΔBA=+0.19694pp、ΔProb-AUC=-0.05940pp；BA三seed=+0.2842/+0.2056/+0.1010pp，Prob-AUC三seed=-0.0522/-0.5414/+0.4154pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
正确增强/减弱方向相对反向绑定：ΔACC=+0.15282pp、ΔBA=+0.13385pp、ΔProb-AUC=-0.22883pp；BA三seed=+0.0684/+0.2561/+0.0770pp，Prob-AUC三seed=-0.2220/-0.9369/+0.4724pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
语义监督相对关闭对齐损失：ΔACC=+0.30608pp、ΔBA=+0.31524pp、ΔProb-AUC=-0.17549pp；BA三seed=+0.5143/+0.1970/+0.2344pp，Prob-AUC三seed=+0.2699/-0.9024/+0.1060pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
正确患者辅助配对相对同站点错配：ΔACC=+0.03788pp、ΔBA=+0.04293pp、ΔProb-AUC=-0.29127pp；BA三seed=+0.4855/-0.4079/+0.0512pp，Prob-AUC三seed=-0.1443/-1.1401/+0.4106pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
新模块相对删除SPGS：ΔACC=+0.07663pp、ΔBA=+0.07814pp、ΔProb-AUC=-0.37157pp；BA三seed=+0.4590/-0.4163/+0.1918pp，Prob-AUC三seed=+0.1838/-1.4428/+0.1443pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
新模块相对完整P0：ΔACC=-0.07663pp、ΔBA=-0.11216pp、ΔProb-AUC=-0.13638pp；BA三seed=-0.0138/-0.2487/-0.0740pp，Prob-AUC三seed=-0.5021/-0.3848/+0.4778pp。未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛。
E4的ASD漏诊累计133次，P0为125次；TD误报累计86次，P0为92次。这些为三个seed重复评估的病例事件数，并非三倍独立患者。

### 表示是否坍塌及训练判别性

|组|教师患者方差|教师ASD/TD中心cos|raw FC有效秩|anchor有效秩|teacher有效秩|anchor类分离比|joint类分离比|teacher类分离比|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E2_OLD_FC|2.048046e-11|1|53.57517|3.492497|4.448005|3.615643|3.615643|0.07532298|
|E3_FINE_FC|0.003464222|0.9999877|53.57517|5.054966|11.62161|3.154252|3.154252|0.08624723|
|E4_FINE_FULL|0.003464222|0.9999877|53.57517|4.851639|11.62161|3.219305|3.407472|0.08624723|
|E5_NUMERIC_FULL|0.002207144|0.9999843|53.57517|4.503897|15.73117|3.613063|3.782278|0.1222249|
|E6_WRONG_ROI|0.002678854|0.999982|53.57517|4.538313|11.39633|3.683496|3.843761|0.1186622|
|E7_WRONG_DIRECTION|0.00333746|0.9999882|53.57517|4.108578|12.48497|3.168264|3.348669|0.08574916|
|E8_FINE_NO_ALIGN|0.003464222|0.9999877|53.57517|6.069457|11.62161|2.385277|2.673066|0.08624723|
|E9_WRONG_AUX_PAIR|0.003464222|0.9999877|53.57517|4.117902|11.62161|3.37837|3.681664|0.08624723|
类分离比=两类训练中心距离/平均类内半径；此处全部以outer-train标签作事后诊断。anchor与joint是同一已训练模型的两个位置，不能据此推断相对训练前的因果提升。E2是同一粗粒度向量的加权平均，理论上没有患者语义差异；其极小非零方差及有效秩可由浮点归一化噪声产生，不能把噪声秩解读为丰富信息。高student-teacher余弦而低教师类差异，可能只是拟合共同文本方向。
E4教师患者方差=0.00346422，两类中心余弦=0.99998766，训练student-teacher余弦=0.05732746。教师并非逐患者完全相同，但两类中心高度同向；不能把非零患者方差或有效秩直接解释为强疾病语义区分性。

### 梯度、有效训练参数与残差

|组|已注册总参数|新增参数|采样有效参数范围|alpha梯度|FC梯度|context梯度|demo梯度|ANAT门梯度|FUNC门梯度|decode梯度|
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
|E0_P0|1462957|0|1257815–1257815|0|0|0|0|0|0|0|
|E1_NO_SPGS|1462957|0|732238–732238|0|0|0|0|0|0|0|
|E2_OLD_FC|1491474|28517|753651–753651|0.034037|0.15846|0.050515|0|0|0|0.041686|
|E3_FINE_FC|1491474|28517|753651–753651|0.037338|0.26587|0.061018|0|0|0|0.051334|
|E4_FINE_FULL|1491474|28517|760755–760755|0.041298|0.27906|0.059671|0.013342|0.00032364|0.00033341|0.033509|
|E5_NUMERIC_FULL|1491474|28517|760755–760755|0.041282|0.27912|0.05796|0.0099827|0.00025463|0.00023857|0.029598|
|E6_WRONG_ROI|1491474|28517|760755–760755|0.037254|0.13525|0.05833|0.0081196|0.00023807|0.00021263|0.025532|
|E7_WRONG_DIRECTION|1491474|28517|760755–760755|0.039692|0.33326|0.070712|0.016402|0.00042937|0.00040709|0.044105|
|E8_FINE_NO_ALIGN|1491474|28517|760755–760755|0.039143|0.38933|0.078838|0.021454|0.00047991|0.0004225|0.060856|
|E9_WRONG_AUX_PAIR|1491474|28517|760755–760755|0.039266|0.20425|0.048978|0.010728|0.00024984|0.00022605|0.030052|
梯度统计为每折前两epoch及每25epoch采样的模块梯度范数最大值，再跨30折平均；有效参数数目也仅为这些采样时刻非零梯度元素所属参数的最大计数，不等于全训练过程的精确活跃参数数目。E2/E3辅助模块注册但不参与有效贡献；E1–E9原MLP保留注册但不参与前向，故不能把注册参数全部算作实际训练容量。
E4残差平均L2=0.3228358，逐fold均值范围0.1364943–0.4767182，非零响应比例=1.0000000，最大坐标绝对值=0.0826021。平均相对CAM范数=0.1541988；该相对量只作诊断，数学约束仍是坐标≤0.10、L2≤0.6。未出现总体近零残差；分支被使用仍不等于正确医学文本有效。
E4语义/分类对anchor梯度范数比均值=0.04187544；语义梯度均值=0.0001402346，分类梯度均值=0.02205134。这是最佳Checkpoint的单次train-loss诊断，不是训练全过程累计贡献。

### 辅助门控与站点关联

|组|辅助/anchor|实际错配比例|QA/QF患者均值STD|QA/QF与年龄类别r|QA/QF站点η²|QA/QF最大QC绝对r|
|---|---:|---:|---|---|---|---|
|E4_FINE_FULL|0.1773704|0.0000000|0.055235/0.101542|0.200148/0.142639|0.603277/0.217883|0.766893/0.815348|
|E5_NUMERIC_FULL|0.2252518|0.0000000|0.054033/0.103549|0.192569/0.139180|0.618275/0.212469|0.787534/0.832116|
|E6_WRONG_ROI|0.2180009|0.0000000|0.051071/0.108147|0.159756/0.101583|0.613374/0.219532|0.780619/0.860948|
|E7_WRONG_DIRECTION|0.1745936|0.0000000|0.054155/0.101739|0.226311/0.130690|0.607717/0.244848|0.758921/0.823918|
|E8_FINE_NO_ALIGN|0.1089248|0.0000000|0.049239/0.099924|0.203284/0.102827|0.615994/0.211378|0.786906/0.846254|
|E9_WRONG_AUX_PAIR|0.1930654|0.9969001|0.057225/0.072749|0.127912/0.066379|0.643881/0.439607|0.531389/0.458632|
原始diagnostics.csv中的auxiliary_effective_shuffled_fraction记录各模型预生成置乱表的覆盖率，该表仅在E9训练或AUX_PAIR_SHUFFLED干预时启用；上表按真实训练配置显示应用比例，其余组为0。原始诊断不作回写。
十套fold训练统计中合格站点数范围=20–20，平均站点方向一致率=0.6288086，平均站点效应异质性=0.1477440。同fold在三个seed共享统计，此处只计一次；QA与年龄/站点/QC关联不能解释为校准后的医学可靠度。

### 反事实变化分布及实际依赖

|干预|逐fold平均概率变化的p50均值|p90均值|逐fold最大变化的最大值|QA均值|QF均值|辅助范数|
|---|---:|---:|---:|---:|---:|---:|
|REAL|0.00000000|0.00000000|0.00000000|0.58196415|0.58666833|0.60104087|
|SEMANTIC_TEACHER_OFF|0.00000000|0.00000000|0.00000000|0.58196415|0.58666833|0.60104087|
|SEMANTIC_RESIDUAL_OFF|0.00864569|0.03570587|0.26923534|0.58196415|0.58666833|0.60104087|
|AUX_OFF|0.00020647|0.00367661|0.07965964|0.58196415|0.58666833|0.00000000|
|PHENO_OFF|0.00020647|0.00367661|0.07965964|0.58196415|0.58666833|0.00000000|
|ANAT_QC_OFF|0.00006296|0.00126566|0.02699763|1.00000000|0.58666833|0.96511125|
|FUNC_QC_OFF|0.00006192|0.00140711|0.02373284|0.58196415|1.00000000|0.93304028|
|AUX_PAIR_SHUFFLED|0.00002344|0.00126227|0.12944350|0.58196416|0.58666833|0.59940553|
|FC_ANCHOR_CONTEXT_OFF|0.00230363|0.01984883|0.16219106|0.58196415|0.58666833|0.43493467|
关闭整条语义残差的ASD概率绝对变化=0.0139969，关闭辅助修正=0.0013065，关闭CAM上下文=0.0066474。这些值说明已训练模型依赖哪些计算路径；必须与E4−E3/E8/E9完整重训结果共同解释。SEMANTIC_TEACHER_OFF概率变化严格为0，是训练专用教师的预期性质；正确/错误文本只通过训练损失作用，不能把同Checkpoint的teacher-off当作医学文本必要性检验。

### 工程恢复与最终研究决策

墙钟执行时间=5690.65秒；开始2026-10-19T14:31:51.551269+00:00，结束2026-10-19T16:06:42.196700+00:00。各GPU累计task运行秒包含同卡并行任务的重叠时间，不能与墙钟时间直接相加。
GPU0：15任务/150折，累计task运行秒=13541.18；分配 E0_P0_seed0, E0_P0_seed2, E1_NO_SPGS_seed1, E2_OLD_FC_seed0, E2_OLD_FC_seed2, E3_FINE_FC_seed1, E4_FINE_FULL_seed0, E4_FINE_FULL_seed2, E5_NUMERIC_FULL_seed1, E6_WRONG_ROI_seed0, E6_WRONG_ROI_seed2, E7_WRONG_DIRECTION_seed1, E8_FINE_NO_ALIGN_seed0, E8_FINE_NO_ALIGN_seed2, E9_WRONG_AUX_PAIR_seed1。
GPU1：15任务/150折，累计task运行秒=16997.54；分配 E0_P0_seed1, E1_NO_SPGS_seed0, E1_NO_SPGS_seed2, E2_OLD_FC_seed1, E3_FINE_FC_seed0, E3_FINE_FC_seed2, E4_FINE_FULL_seed1, E5_NUMERIC_FULL_seed0, E5_NUMERIC_FULL_seed2, E6_WRONG_ROI_seed1, E7_WRONG_DIRECTION_seed0, E7_WRONG_DIRECTION_seed2, E8_FINE_NO_ALIGN_seed1, E9_WRONG_AUX_PAIR_seed0, E9_WRONG_AUX_PAIR_seed2。
已审计工程修复1：{"after_sha256": "37c8226f29853a84efd5159d83c7fb337addd709a3b76aaa50500bbb51ed68f2", "before_sha256": "6dd3f77fa4e0e4e7f78bc6d7c860ce258a97fab723ffb3092c82dd2c486cfc14", "reason": "Mounted filesystem transient FileNotFoundError while reading an atomically replaced task JSON; same finite retry pattern as accepted experiment13. No numerical/training code changed.", "time": "2026-10-19T14:58:49.967995+00:00", "unchanged_numerical_contract_signature": "fc8fe31bf13f5548ce42c9c9e43a7e0617c43c026aa0c7b04822abd4782ea567"}；逐行diff保留于manifest。
已审计工程修复2：{"after_sha256": "de917c9933d2555c277eb6360da4b11a8053d050f8e1e88140d8bb9c6178a957", "before_sha256": "37c8226f29853a84efd5159d83c7fb337addd709a3b76aaa50500bbb51ed68f2", "reason": "Restore original cudnn benchmark=False/deterministic=True when resuming in a fresh process. The original linear/graph model does not invoke convolution; verify exact update invariance. No algorithm, RNG state, text, teacher or protocol changed.", "time": "2026-10-19T15:03:59.862450+00:00", "verification": "REAL_GPU_EXACT_MODEL_BOTH_ADAM_AND_ALL_RNG under cudnn flag False vs True; no convolution in this training graph."}；逐行diff保留于manifest。
已审计工程修复3：{"after_sha256": "43dd3479bec2418b9e9baae1fea541bef564a8354df63bd132246670904e49fc", "before_sha256": "de917c9933d2555c277eb6360da4b11a8053d050f8e1e88140d8bb9c6178a957", "reason": "Observed transient unterminated JSON from an atomically replaced task status on the mounted filesystem. Add shared/exclusive reader/writer flock on a stable runtime inode plus bounded missing/JSON-decode retries; persistent invalid records remain errors. No numerical code or state format changed.", "time": "2026-10-19T15:20:05.367062+00:00", "verification": "Mounted-FS concurrent 40 writes/200 reads PASS; transient JSON retry PASS; persistent corruption rejection PASS; checksummed states restore all task summaries."}；逐行diff保留于manifest。
已审计工程修复4：{"after_sha256": "47ae00a8c98015cd9cd82bcb63ef1394343cf1edc09153fc03d2fa3e6f36956a", "before_sha256": "43dd3479bec2418b9e9baae1fea541bef564a8354df63bd132246670904e49fc", "phase": "POST_TRAIN_REPORT_ONLY", "reason": "Expand the completed Chinese report with actual negative-control conclusions, representation/gradient/QA tables, counterfactual distributions and audited recovery details. All 300 folds had already finished; no algorithm, prompt, protocol, training or checkpoint function changed.", "time": "2026-10-19T16:48:57.493709+00:00", "verification": {"added_functions": ["detailed_results"], "all_training_and_recovery_function_AST_unchanged": true, "changed_existing_functions": ["report"], "formal_fold_count_before_change": 300, "raw_diagnostics_unchanged": true, "shuffle_field_interpretation": "Raw field stores available derangement coverage; report applies it only for E9. E4-E8 training used original auxiliary pairing."}}；逐行diff保留于manifest。
最终科学决策：NO-GO。预设方法未通过全部正确文本、ROI身份、方向、语义监督和辅助患者配对的稳健对照；当前证据不支持将IA-FDSA作为MICCAI 2027第一创新点。保留完整阴性结果，停止本数据与历史协议下静态FC文本追分；优先增加真实互补影像/临床测量，并采用训练内选模、LOSO或独立队列检验。
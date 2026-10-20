# 实验28补充对照E10：粗粒度Prompt＋完整辅助条件修正

状态 **COMPLETE**。仅新增 **30/30折、3个任务**；历史300折只读。执行目录：`/tmp/experiment28_supplement_e10_coarse_full`。

本轮补全Prompt粒度×辅助条件修正2×2对照。E10与E4完整数学相同，只把fine教师换为原E2冻结coarse教师；不重编码CLIP、不改Prompt、student仍为anchor。
E2的单一粗粒度向量来自原异常四句投影后均值归一化，并广播到252×2个位置；原正常四句只参与共享SVD基底。该教师理论上近常量，不能把其正则效果解释为独立疾病知识。

## 历史与新增全部11组

先分别计算每seed十折均值，再三seed均值±population SD，单位%。Hard-AUC=BA。

|组|来源|ACC|BA/Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|
|---|---|---:|---:|---:|---:|---:|---:|
|E0_P0|HISTORICAL_READ_ONLY|91.69105±0.05418|91.53935±0.08408|93.33520±0.41541|89.66057±0.64562|93.41813±0.51023|90.93289±0.12038|
|E1_NO_SPGS|HISTORICAL_READ_ONLY|91.53779±0.19536|91.34905±0.20625|93.57039±0.30428|88.92073±0.71450|93.77737±0.60617|90.70334±0.21945|
|E2_OLD_FC|HISTORICAL_READ_ONLY|91.53779±0.05418|91.34309±0.06147|93.55219±0.26848|88.83943±0.19917|93.84675±0.10248|90.70297±0.07397|
|E3_FINE_FC|HISTORICAL_READ_ONLY|91.69105±0.05418|91.47256±0.07651|93.38635±0.15774|88.66667±0.65496|94.27845±0.53157|90.81486±0.10378|
|E4_FINE_FULL|HISTORICAL_READ_ONLY|91.61442±0.16255|91.42719±0.17620|93.19882±0.50570|89.00610±0.45706|93.84829±0.10466|90.77521±0.23647|
|E5_NUMERIC_FULL|HISTORICAL_READ_ONLY|91.46160±0.30108|91.27532±0.34054|93.35904±0.15532|88.91667±0.92844|93.63398±0.26743|90.62720±0.39481|
|E6_WRONG_ROI|HISTORICAL_READ_ONLY|91.42285±0.19536|91.23025±0.19801|93.25822±0.16141|88.75407±0.42892|93.70644±0.35952|90.57917±0.22862|
|E7_WRONG_DIRECTION|HISTORICAL_READ_ONLY|91.46160±0.21612|91.29334±0.26263|93.42765±0.15626|89.09146±0.88464|93.49522±0.36620|90.63693±0.32649|
|E8_FINE_NO_ALIGN|HISTORICAL_READ_ONLY|91.30834±0.14289|91.11195±0.14899|93.37431±0.19609|88.59146±0.52689|93.63244±0.44470|90.43667±0.18806|
|E9_WRONG_AUX_PAIR|HISTORICAL_READ_ONLY|91.57654±0.21612|91.38426±0.22948|93.49009±0.14858|88.91870±0.64688|93.84983±0.44094|90.74542±0.25832|
|E10_COARSE_FULL|NEW_30_FOLDS|91.80599±0.10837|91.60162±0.09888|93.22243±0.25061|88.92480±0.11357|94.27845±0.26579|90.96702±0.09444|

## E10三个seed及病例事件

|seed|ACC|BA|Prob-AUC|SEN|SPE|F1|ASD FN|TD FP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|91.72936|91.52295|92.87224|88.84146|94.20444|90.89432|45|27|
|1|91.72936|91.54083|93.44480|89.08537|93.99630|90.90636|44|28|
|2|91.95925|91.74108|93.35025|88.84756|94.63460|91.10040|45|25|

## 全部指定配对与逐fold胜/平/负

|比较|指标|均差pp|seed0/1/2差pp|W/T/L|
|---|---|---:|---|---|
|E10_COARSE_FULL − E2_OLD_FC|acc|+0.26820|+0.22989,+0.22989,+0.34483|10/17/3|
|E10_COARSE_FULL − E2_OLD_FC|ba|+0.25853|+0.22833,+0.23601,+0.31125|11/14/5|
|E10_COARSE_FULL − E2_OLD_FC|hard_auc|+0.25853|+0.22833,+0.23601,+0.31125|11/14/5|
|E10_COARSE_FULL − E2_OLD_FC|probability_auc|-0.32976|-0.58089,+0.16052,-0.56891|9/1/20|
|E10_COARSE_FULL − E2_OLD_FC|sensitivity_asd|+0.08537|+0.24390,+0.25000,-0.23780|6/18/6|
|E10_COARSE_FULL − E2_OLD_FC|specificity_td|+0.43170|+0.21277,+0.22202,+0.86031|7/21/2|
|E10_COARSE_FULL − E2_OLD_FC|f1_asd|+0.26405|+0.25134,+0.24759,+0.29322|11/14/5|
|E4_FINE_FULL − E3_FINE_FC|acc|-0.07663|+0.00000,-0.22989,+0.00000|9/12/9|
|E4_FINE_FULL − E3_FINE_FC|ba|-0.04536|+0.09077,-0.20898,-0.01788|11/9/10|
|E4_FINE_FULL − E3_FINE_FC|hard_auc|-0.04536|+0.09077,-0.20898,-0.01788|11/9/10|
|E4_FINE_FULL − E3_FINE_FC|probability_auc|-0.18753|+0.21304,-0.92204,+0.14641|14/1/15|
|E4_FINE_FULL − E3_FINE_FC|sensitivity_asd|+0.33943|+1.25000,+0.01220,-0.24390|7/19/4|
|E4_FINE_FULL − E3_FINE_FC|specificity_td|-0.43016|-1.06846,-0.43016,+0.20814|6/14/10|
|E4_FINE_FULL − E3_FINE_FC|f1_asd|-0.03965|+0.13879,-0.25919,+0.00146|11/9/10|
|E3_FINE_FC − E2_OLD_FC|acc|+0.15326|+0.22989,+0.11494,+0.11494|8/16/6|
|E3_FINE_FC − E2_OLD_FC|ba|+0.12947|+0.16334,+0.08218,+0.14288|10/12/8|
|E3_FINE_FC − E2_OLD_FC|hard_auc|+0.12947|+0.16334,+0.08218,+0.14288|10/12/8|
|E3_FINE_FC − E2_OLD_FC|probability_auc|-0.16584|-0.27769,+0.14457,-0.36440|12/0/18|
|E3_FINE_FC − E2_OLD_FC|sensitivity_asd|-0.17276|-0.52439,-0.48780,+0.49390|6/17/7|
|E3_FINE_FC − E2_OLD_FC|specificity_td|+0.43170|+0.85106,+0.65217,-0.20814|7/18/5|
|E3_FINE_FC − E2_OLD_FC|f1_asd|+0.11189|+0.14987,+0.04144,+0.14435|10/12/8|
|E4_FINE_FULL − E10_COARSE_FULL|acc|-0.19157|+0.00000,-0.34483,-0.22989|9/11/10|
|E4_FINE_FULL − E10_COARSE_FULL|ba|-0.17443|+0.02578,-0.36280,-0.18625|9/11/10|
|E4_FINE_FULL − E10_COARSE_FULL|hard_auc|-0.17443|+0.02578,-0.36280,-0.18625|9/11/10|
|E4_FINE_FULL − E10_COARSE_FULL|probability_auc|-0.02361|+0.51625,-0.93799,+0.35092|16/2/12|
|E4_FINE_FULL − E10_COARSE_FULL|sensitivity_asd|+0.08130|+0.48171,-0.72561,+0.48780|10/16/4|
|E4_FINE_FULL − E10_COARSE_FULL|specificity_td|-0.43016|-0.43016,-0.00000,-0.86031|4/19/7|
|E4_FINE_FULL − E10_COARSE_FULL|f1_asd|-0.19181|+0.03731,-0.46534,-0.14741|9/11/10|
|E10_COARSE_FULL − E3_FINE_FC|acc|+0.11494|-0.00000,+0.11494,+0.22989|9/16/5|
|E10_COARSE_FULL − E3_FINE_FC|ba|+0.12907|+0.06500,+0.15382,+0.16837|12/12/6|
|E10_COARSE_FULL − E3_FINE_FC|hard_auc|+0.12907|+0.06500,+0.15382,+0.16837|12/12/6|
|E10_COARSE_FULL − E3_FINE_FC|probability_auc|-0.16392|-0.30320,+0.01595,-0.20451|14/0/16|
|E10_COARSE_FULL − E3_FINE_FC|sensitivity_asd|+0.25813|+0.76829,+0.73780,-0.73171|7/18/5|
|E10_COARSE_FULL − E3_FINE_FC|specificity_td|-0.00000|-0.63830,-0.43016,+1.06846|6/19/5|
|E10_COARSE_FULL − E3_FINE_FC|f1_asd|+0.15217|+0.10147,+0.20616,+0.14887|12/12/6|
|E10_COARSE_FULL − E4_FINE_FULL|acc|+0.19157|-0.00000,+0.34483,+0.22989|10/11/9|
|E10_COARSE_FULL − E4_FINE_FULL|ba|+0.17443|-0.02578,+0.36280,+0.18625|10/11/9|
|E10_COARSE_FULL − E4_FINE_FULL|hard_auc|+0.17443|-0.02578,+0.36280,+0.18625|10/11/9|
|E10_COARSE_FULL − E4_FINE_FULL|probability_auc|+0.02361|-0.51625,+0.93799,-0.35092|12/2/16|
|E10_COARSE_FULL − E4_FINE_FULL|sensitivity_asd|-0.08130|-0.48171,+0.72561,-0.48780|4/16/10|
|E10_COARSE_FULL − E4_FINE_FULL|specificity_td|+0.43016|+0.43016,+0.00000,+0.86031|7/19/4|
|E10_COARSE_FULL − E4_FINE_FULL|f1_asd|+0.19181|-0.03731,+0.46534,+0.14741|10/11/9|

## 2×2交互描述

交互=(E4−E3)−(E10−E2)=(E4−E10)−(E3−E2)。逐fold两公式数值一致；这是因子交互的内部描述，不是医学因果协同证明。

|指标|交互均值pp|seed0/1/2 pp|正/零/负fold|
|---|---:|---|---|
|acc|-0.34483|-0.22989,-0.45977,-0.34483|6/13/11|
|ba|-0.30390|-0.13756,-0.44499,-0.32914|10/7/13|
|hard_auc|-0.30390|-0.13756,-0.44499,-0.32914|10/7/13|
|probability_auc|+0.14223|+0.79394,-1.08256,+0.71532|18/0/12|
|sensitivity_asd|+0.25407|+1.00610,-0.23780,-0.00610|9/14/7|
|specificity_td|-0.86186|-1.28122,-0.65217,-0.65217|6/12/12|
|f1_asd|-0.30370|-0.11256,-0.50678,-0.29176|11/6/13|

## 教师、辅助条件与有效梯度

|组|有效train/test比例|教师方差|ASD/TD中心cos|student-teacher train/test cos|辅助/anchor|alpha|残差L2|
|---|---|---:|---:|---|---:|---:|---:|
|E2_OLD_FC|1.000000/1.000000|2.04804566e-11|1|0.126732/0.127459|0.0000000|0.2362772|0.3278320|
|E3_FINE_FC|1.000000/1.000000|0.00346422219|0.99998766|0.060106/0.060118|0.0000000|-0.0007680|0.3147545|
|E4_FINE_FULL|1.000000/1.000000|0.00346422219|0.99998766|0.057327/0.057465|0.1773704|0.2117880|0.3228358|
|E10_COARSE_FULL|1.000000/1.000000|2.04804566e-11|1|0.186761/0.188302|0.2279424|-0.1507246|0.3236681|

E10 ANAT门均值0.5721210、患者均值SD的折平均0.0562398、患者均值方差的折平均0.003503802、逐维SD均值0.1643082；FUNC门均值0.5834698、患者均值SD的折平均0.1146842、患者均值方差的折平均0.014077448、逐维SD均值0.2270034。方差先逐fold计算SD平方，再跨fold平均，不能平方平均SD代替。
辅助修正L2均值0.6170630，相对anchor0.2279424。语义梯度到anchor为0.0002197478，分类梯度0.009706262。
各模块训练采样最大梯度范数的折平均：`{"alpha": 0.040148799490002975, "fc_encoder": 0.15842036074630544, "context_encoder": 0.06289763386665363, "demo_encoder": 0.007741899987978221, "anat_gate": 0.00026375933888887483, "func_gate": 0.00022687764653422582, "decode": 0.027414729453854546}`。
逐fold详细各模块梯度、alpha、残差非零率、坐标上界、QA、训练统计SHA均保存在diagnostics.csv。alpha=0首步decoder/辅助梯度为零符合原E4规律；第二步梯度连通。

## 结论A–D

A 辅助条件净价值：粗粒度E10−E2满足三seed BA/Prob-AUC/SEN一致正向：False；细粒度E4−E3：False。各指标和seed差如上，不以单一ACC决定。
B 细粒度独立价值：相同完整辅助结构的E4−E10满足上述一致门槛：False。若不满足，仅为阴性或指标权衡，不能声称细粒度全面有效。
C 交互方向按上表逐指标和逐seed判读；三seed重复同一队列，不视为30个独立外部样本，不报告伪独立P值。
D 原E4 ACC91.61442/BA91.42719/ProbAUC93.19882%，仍低于P0 91.69105/91.53935/93.33520%及历史Exp9 E9 91.88262/91.69151/93.82272%。原实验28 NO-GO判断保持。本轮历史E9_WRONG_AUX_PAIR不是历史Exp9 E9。
E2/E10粗粒度教师各有效患者理论方向一致，其近零方差及中心cos≈1不提供患者类别区分。若E10增益出现，应优先解释为辅助融合、优化或常量语义正则线索，不证明医学知识必要性。

## 工程、协议与完整性

原GFR i==1覆盖、阈值.9、ALoss不加入simple-2损失、四图正则平均、原RES/分类器及两Adam保持。SPGS按E4关闭，原MLP继续注册但不参与前向。新分支28517参数、总参数1491474，新增wd=0、原wd=.11、lr=.0038。alpha=0与E1逐值相同。残差每坐标≤.10，不是相对CAM范数10%。
871人403 ASD/468 TD，输入48/6/10/256。原10fold(seed划分10)、任务seed0/1/2，450epoch、wait>200、首次严格testACC最高Checkpoint。每任务只播种一次，连续10fold；早停后后续RNG自然分叉，不声称E4/E10各fold相同初始化。首fold入口RNG/主干hash与历史E4精确核对。
保留test=val选模、外层CV前RFE/全体标准化、全871人transductive传播；多轮探索和测试参与选模使结果偏乐观，不能宣称独立无泄漏泛化。潜在4×36与4×4×9不是ROI；ANAT/FUNC为质量特征。
预检：`{"passed": true, "completed_utc": "2026-10-20T04:02:10.921241+00:00", "checks": {"P0_unpatched_init_forward_two_adam_first_update_rng": "EXACT_PASS", "P0_seed0_fold1_initial_hash": "b464f86ee750e5550a6330942ee102a5d9d1f3ab4967a385e23655741c92ae0f", "E10_E4_structure_original_new_parameters_optimizer_rng": "BITWISE_EXACT", "E10_E1_E4_zero_alpha_output": "BITWISE_EXACT", "E10_E2_same_student_semantic_loss": "BITWISE_EXACT", "total_parameters": 1491474, "additional_parameters": 28517, "heldout_label_flip_full_train_update_models_both_adams_rng": "BITWISE_EXACT", "first_step_gradient_norms": {"gfr": 1.7359182958520136, "pln_original": 2.0574789457950238, "alpha": 0.004164975586912834, "fc_encoder": 0.0015731278535086523, "context_encoder": 0.00018808754404105524, "demo_encoder": 0.0, "anat_gate": 0.0, "func_gate": 0.0, "decode": 0.0, "active_parameters": 751311}, "second_step_gradient_norms": {"gfr": 1.5987937536028982, "pln_original": 3.4424123207985153, "alpha": 0.00016862177183014633, "fc_encoder": 0.0005641556965762557, "context_encoder": 0.00022434364488380088, "demo_encoder": 9.49557618791706e-06, "anat_gate": 2.815128581226507e-07, "func_gate": 2.786521469425497e-07, "decode": 0.0002703180729481091, "active_parameters": 760755}, "auxiliary_conditional_reaches_classification": "PASS", "disk_checkpoint_model_two_adams_full_rng_next_update_prediction": "BITWISE_EXACT", "next_fold_initialization_rng_resume": "BITWISE_EXACT", "device": "_CudaDeviceProperties(name='PPU-ZW810E', major=8, minor=0, total_memory=98304MB, multi_processor_count=64, uuid=019e4201-0591-0030-0000-000080387166, pci_bus_id=99, pci_device_id=0, pci_domain_id=0, L2_cache_size=64MB)", "all10fold_E10_E2_teacher_active_weights_loss": "BITWISE_EXACT", "all10fold_E10_E4_parameters_aux_forward_nonzero_alpha": "BITWISE_EXACT", "teacher_forward_semantic_loss_source": "BYTE_IDENTICAL_FUNCTIONS", "frozen_coarse_cache": "REUSED_NO_CLIP_ENCODING", "GPU1_visible_logical0": "PPU-ZW810E"}}`。
任务分配/耗时：`[{"config": "E10_COARSE_FULL", "seed": 0, "gpu": 0, "folds": 10, "status": "COMPLETE", "completed_folds": 10, "seconds": 695.9572685696185, "resumes": 0}, {"config": "E10_COARSE_FULL", "seed": 1, "gpu": 1, "folds": 10, "status": "COMPLETE", "completed_folds": 10, "seconds": 676.8862256519496, "resumes": 0}, {"config": "E10_COARSE_FULL", "seed": 2, "gpu": 0, "folds": 10, "status": "COMPLETE", "completed_folds": 10, "seconds": 730.6162389665842, "resumes": 0}]`；墙钟秒：808.959539。
历史/核心保护：`{"status": "UNCHANGED_SHA256", "files": 1028}`；缓存SHA：`2a3cf9332ad3d1dbdba70731a46acee43f3f7e52c92fe3ab2366fa90b0d54420`；源码SHA：`{"main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67", "ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763"}`；实验源码SHA：`{"e28_method.py": "9e9d0fe4de5287f40f22bacb41191e133e66f445ef8ce0e7182ea0cd26f8ba5e", "e28_prompts.py": "0c5f09f23e3c0ca9db1f2292cfd957f04f779bab2eb213d5a97a7a947c6a4cb5", "e28_integration.py": "3b0c6a0114caf87d989e8af7bd905a26c3ab0661c358ecb8bdfc0869c9d585c2", "run.py": "6063b44fff64dca89a7029629b7340b87c04a8357d4b6977051fc51d56968a30", "integration.patch": "77cb8197e6af3329c04b4fa86e093be637d058d0a9b0695f0cf28a932caddb6c"}`。
不重训历史300fold；新fold_metrics/diagnostics仅E10各30条，历史表按原CSV读取。完成后清理本实验.runtime，不自动Git提交/推送。

下一步建议：维持原强基线及实验28止损判断；若局部交互值得关注，仅在训练内选模和独立队列验证，不据本轮测试成绩改Prompt或搜损失权重。

## 实测机制判读与交付状态

E10−E2的ACC三个seed均改善(+0.22989/+0.22989/+0.34483pp)，BA也三个seed改善；但Prob-AUC两个seed降低，均值−0.32976pp。因此辅助修正在粗粒度条件下有硬分类阈值指标收益，未产生一致排序收益。
相同辅助结构的E4−E10：ACC两个seed降低、一个持平；BA两个seed降低；Prob-AUC两个seed上升、一个明显下降，均差−0.02361pp。ASD灵敏度均差+0.08130pp但方向不一致。没有支持细粒度教师全面优于粗粒度教师。
交互在ACC、BA、F1和特异度三个seed均为负；概率AUC交互均值为正但seed1为负，灵敏度交互也不同向。没有稳定的正向因子交互。
E10相对历史P0 ACC+0.11494pp、BA+0.06227pp，但Prob-AUC−0.11277pp、ASD灵敏度−0.73577pp；累计漏诊134次，P0为125次（重复三个seed的病例事件）。E10也未超过历史Exp9 E9的ACC/BA/Prob-AUC。单一最高ACC不能替代疾病语义必要性检验。
E10教师与E2完全一致，类别中心cos≈1、患者方差≈2.05e−11，属近常量语义方向；其优化或辅助融合增益不等于独立医学知识。新分支残差L2≈0.324且辅助/anchor≈22.79%，不是近零响应失败。
handoff状态：`PENDING_ROOT_IO`；整体交付状态：`PARTIAL_HANDOFF_IO`。
若为PENDING_ROOT_IO，根目录追加/fsync返回Errno5；新增章节保存在`handoff_addendum.md`，可读旧历史前缀SHA仍一致。未确认根文件新章节持久化，不能宣称handoff已更新；I/O修复后运行`python run.py handoff`重试。
正式训练完成后仅完善报告方差聚合、实测解释及handoff重试接口；训练冻结源码SHA和对应差异分别保存在manifest，训练/方法/协议/缓存/原始CSV均未改变。

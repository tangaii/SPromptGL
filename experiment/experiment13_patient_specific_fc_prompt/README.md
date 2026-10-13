# Experiment 13 — Patient-Specific FC Prompt

正式矩阵为 F0/F1/F2/F3/P4/P5/P6/P7 × seeds 0/1/2 × 10 folds，共 240 折。原源码、数据及实验一至十二只读，所有新增代码、缓存、日志、状态与结果只写入本目录。未自动 commit/push 或上传数据。

## 源码与历史审计

已逐文件检查根 `main.py`、`model.py`、`network.py`、`layers.py`、`utils.py`、原 Shell；实验一 Prompt 配置和运行器；实验八类别对齐、实验九交互与实验十疾病特异性报告及补丁；实验十一数值/解剖映射与报告；实验十二运行器、适配器、文本构建、预检和报告。

本地 HEAD 为 `1167dd392837a02eb4f83481ae49f02607a9dae5`，审计时远端 main 为 `b1c63941bd134495b713f0a0e05784f493563e5e`。六个核心源码逐字节相同，不强制重置仓库。prepare 再读取远端实际源码验证，manifest 记录版本与 SHA256。

历史结果的相关边界：实验一 P1 ACC 约91.80%，原 P0 约91.69%，差异很小；实验八 S1 疾病类别对齐约91.73%，未超过 P1，疾病分支与主分类器概率相关约0.958；实验九的疾病/对抗交互依赖基础 Prompt，P1 联合组未稳定受益；实验十正确 ASD 文本相对中性文本 ACC 只约+0.077个百分点，不能排除类别代码/容量作用。实验十一恢复了256/256条真实连接，确认110个标准解剖名称及一个未知 ROI82/atlas3455。实验十二仅单fold三分协议：F1静态医学名称未超过F0/错配/匿名对照，并出现投影共有方向增强；此次保留原结果，以指定原协议完整重跑。

原调用关系：CSV `[871,320]` → InitialDataLayer（48/6/10/256）→ GFR受试者图 → H `[871,320]` → 本适配器仅增量修改 H[:,64:320] → 原 PLN → 原 P0 Softmax-Max SPGS → 原 OutputLayer。AGESA 的数值配对使用 CSV 原始 Correlation，不把传播后 H 当作 Fisher-z。PLN 的每模态四个9维 Token 是潜在表示，不对应真实 ROI。

GFR 的第1模态覆盖前一分支、后续分支叠加等原实现保留；原 ALoss 仅打印而未加入最终损失，也不更改。原分类权重 Counter 顺序、NLL 与图损失计算均保留。完整标签张量按原代码保留，但 NLL 和图标签先验只实际利用 train_idx；预检直接验证修改测试标签不影响 GFR 输出和先验。

## 固定协议与配对初始化

StratifiedKFold(10,shuffle=True,random_state=10)，每折训练为除测试人员外的全部样本；val_idx=test_idx。第1折783训练/88验证测试。450 epochs，patience200，`wait_cnt > early` 停止，初始最佳ACC=0，严格ACC提高才更新checkpoint，相同ACC保留第一次。保留原第5折均值ACC<0.6终止保护，若触发则明确任务未完成。

原 Shell 参数：simple-2、weighted-cosine、concat、GCN、dropout .35、lr .0038、reg .11、hidden18、heads2、layers1。原模型1,462,957参数；除F0外各新增32,865参数，原Adam组不变，新增组lr .0038、weight_decay0。

首次 `clip.load(RN50)` 会创建模型并消耗CPU RNG；直接预置P0缓存会改变原始MLP之后的初始化。本运行器在原懒加载位置保留RN50载入，使用已验证的八句缓存替代无随机性的编码前向，禁止训练/preflight实际调用encode_text。prepare验证冻结编码不改变RNG。

F0在单个seed进程内连续十折，不重复每折seed。不同配置的早停长度可能不同，导致自然跨fold RNG轨迹不再相同；为保留原F0流程并满足配对初始化，F0发布实际各折入口RNG，其他组在相同seed/fold使用该入口状态。这些是连续原F0状态，不是每折重复同一seed。适配器仍在原初始化完成后以fork_rng(seed=120000+seed)注册；构造不扰动原状态。正式每折再核对主干和适配器参数初始化哈希。

原协议 `val=test` 有测试参与模型选择的问题；现有CSV上游全队列标准化与RFE特征选择也有信息共享风险。本实验是指定协议下的配对控制变量研究，不是无偏独立泛化估计；没有私自重划内部验证集、重做RFE或添加损失。

## 文本与机制

F1/F2/F3逐字、逐向量复用实验十二 REAL、seed1200无固定点SHUFFLED、ID_ONLY，保留252个有效连接位置。195/225/232/236四条异常连接保留原始输入，delta精确为0，不编造解剖名称。静态适配器与实验十二直接数值校验。

P4：真实ROI身份+本人状态。P5：保持ROI，逐连接在训练内部及测试内部独立固定置换状态，local NumPy Generator seed=130500+fold，不跨集合交换、不用标签、不改全局RNG，逐边状态频率保持。P6：匿名实际特征编号+本人状态。P7：复用F2 ROI无固定点置换，仅改变医学名称对应，仍选实际连接本人的状态。

每个fold仅以外层train_idx计算原始连接值mu/sigma，ddof=0。LOW为u<=-1，HIGH为u>=1，其余MID。sigma<1e-6时全部人该边为MID，记录退化数。三状态只描述与训练队列的定量差异，不包含ASD/TD真值，不称为临床异常或生物标志物。

只编码756条医学状态文本与756条匿名状态文本，各形成 `[252,3,1024]` 冻结RN50库。P5复用P4，P7重排P4的ROI轴。模板来源、哈希、权重哈希、最大token长度在cache和manifest中；严格77 token报错，不截断。文本/模板改变后缓存失效；训练每epoch编码为0。

`ProjectText=Linear(1024,32,bias=False)`；`ProjectValue=Linear(3,32,bias=False)`；phi(z)=[z,abs(z),sign(z)]；alpha为共享零初始化标量。先对库投影，再按患者状态索引 `[871,252,32]`，不展开 `[871,252,1024]`。

`s=normalize(Wt E)·normalize(Wv phi(z))`；`delta=.1*tanh(alpha)*z*tanh(s)`。维持实验十二乘法顺序。零alpha保持原模型初始预测；首步投影梯度为0、alpha有效，第二步两投影有梯度属于预期。

## 运行与恢复

```bash
cd /data/adodas/SPromptGL-main
python experiment/experiment13_patient_specific_fc_prompt/run.py prepare
python experiment/experiment13_patient_specific_fc_prompt/run.py preflight
python experiment/experiment13_patient_specific_fc_prompt/run.py all --workers-per-gpu 4
python experiment/experiment13_patient_specific_fc_prompt/run.py resume --workers-per-gpu 4
python experiment/experiment13_patient_specific_fc_prompt/run.py report
```

单scheduler持文件锁；每config×seed独立进程顺序10折。独立CUDA_VISIBLE_DEVICES绑定，OMP/MKL/OpenBLAS各2线程。起始每卡4个worker，F0优先并交错静态和患者任务；根据正式任务资源决定是否调整，不做并发基准搜索。

正式运行曾从每卡4增至8，再尝试将全部24任务同时运行；每卡12时发生明显CPU争用。宿主机虽显示256核和1.5TiB内存，容器实际配额为32核、128GiB。已回退每卡8个worker，依照原断点继续，全部任务最终仍严格按每卡12任务/120折分配。具体启动、暂停、恢复时间见scheduler/controller日志。

调度器读取替换中的任务JSON时曾出现一次FileNotFoundError。为read_json增加最多三次50ms重试，并在signature中保留已审计的原训练实现身份；异常JSON及持续缺失记录仍报错。另修正scheduler的收尾竞态：报告标记COMPLETE并清理临时状态后立即终止调度，避免下一轮误重试已结束的最后任务；接管进程时允许其在读取/proc期间自然退出。仅更换协调器，正在训练的进程不重启。模型、训练、评估函数及数值常量的AST完全一致，RNG流程未变。manifest分别保存当前实现哈希、原断点实现哈希和两次维护审计，preflight.json记录新增必要测试；24份原模型/Adam/RNG断点沿用，没有迁移或改写张量。

| Config | seed0 GPU | seed1 GPU | seed2 GPU |
|---|---:|---:|---:|
| F0 | 0 | 1 | 0 |
| F1 | 1 | 0 | 1 |
| F2 | 0 | 1 | 0 |
| F3 | 1 | 0 | 1 |
| P4 | 0 | 1 | 0 |
| P5 | 1 | 0 | 1 |
| P6 | 0 | 1 | 0 |
| P7 | 1 | 0 | 1 |

各卡12任务、静态6/患者6，共120折。正式日志逐任务保存。原子断点含当前epoch、最佳ACC/模型、两个Adam（新增组在MF Adam内）、Python/NumPy/CPU/CUDA RNG、输入签名、患者状态可重建信息、已完成折指标与诊断。每10epoch、ACC改善、早停或信号时保存；崩溃只回到最近完整断点，不重训已完成fold。fold完成时先安全写状态再写结果，避免写入间隙丢失连续RNG。死亡scheduler的存活worker可被续跑scheduler接管，避免重复领取。结果依据校验任务记录，不从失败日志混合解析。

## 评价与结论

主要ACC/Hard-label AUC复制原实现；附加BA、ASD SEN、TD SPE、Probability-AUC（exp(log-softmax)第0列）。每seed先求10折均值，三个seed汇总mean±population SD；逐折分布在CSV。配对比较相同seed/fold，不将30折当完全独立样本。

每折在最佳checkpoint记录残差、alpha、梯度最大范数、投影库cos/有效秩/中心化有效秩；患者组记录状态比例、多样性、退化连接和错配程度。P4的30个checkpoint各做REAL_PERSON/SHUFFLED_PERSON/ID_ONLY_PERSON/SHUFFLED_ROI/OFF，共150条无重训练反事实。只保存聚合变化，不保存患者预测表。

主要问题为P4相对F1/P5/P6/P7/F0的配对收益，结合BA和连续AUC、SEN/SPE、状态/名称错配与投影趋同。工程成功、语义敏感、独立诊断收益分别判断。不得仅凭ACC、梯度非零或推理敏感性证明医学语义正确；不自动扩展全局疾病模块。

本次是Patient-conditioned ROI-edge Prompts，并未实现BrainPrompt/SABER完整患者全局语义。报告由真实已完成结果生成，缺失折不填零；全量验收后manifest标记COMPLETE，删除临时权重、RNG状态与任务中间JSON，保留聚合CSV、必要日志和文本库。

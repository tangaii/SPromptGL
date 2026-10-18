# GC-SRA — Experiment 23

实验在 `/tmp/experiment23_gain_calibrated_selective_relational_alignment` 执行；项目根 `/data/adodas/SPromptGL-main` 只读（handoff仅安全追加）。/data 创建失败，按用户指令使用 /tmp。没有执行git commit/push/pull/reset/checkout。

## 冻结设计

G0原P0；G1统一关系桥接；G2同预算常量；G3同预算FC不确定度；G4 OOF增益；G5集合内置乱增益；G6同容量特征差分；G7仅PHENO关系构造错配。共24任务、240折，CUDA_VISIBLE_DEVICES隔离，逻辑cuda:0，两物理GPU交替分配各12任务/120折，每卡至多4 worker，每worker2 CPU线程。

严格复用六份用户参考源文件，SHA见manifest。算法未改写，运行器的原子checkpoint、RNG、data、训练loop/状态恢复由实验18移植并逐项替换其专用分支。只有复制到 `_runtime_sources/<task>/sources/model.py` 的源码有两个bridge前向点与构造后安装；network/layers/utils/main原样。禁止导入实验18教师/对齐分支。临时源码、完整模型和两个Adam状态在完成后清理；永久CSV足以校验任务及重跑。

## 数学与接口

训练折内5fold StandardScaler+LogisticRegression(C=1,lbfgs,1000)比较FC256与FC+PHENO28，delta=CE_F−CE_FP。Ridge(alpha=150)拟合clip(delta,−1.5,1.5)，meta5fold OOF只用于探索性诊断。gate=.02+.96 sigmoid(pred_gain/max(sd(clipped_target),.05))；训练gate为全训练meta拟合值，测试gate用outer-train全量教师概率。QC16为观测质量指标，站点20维不进辅助输入。

AF/AP仅以outer-train为key，train-only标准化、cosine/.3、各top12并集、排除自己。关系差分=(AP−AF)PHENOvalue；新增桥接只更新GFR输出的CORR256，前64维逐值保持。Decoder([difference,context])−Decoder([0,context])消除FC-only/bias旁路；零alpha；.5×.1×tanh(alpha)×gate×tanh(norm(difference))×detach(norm(CORR))×单位方向。新增30,081个注册参数，均在MF Adam新增组lr=.0038,wd=0，原参数wd=.11。

G6使用真实PHENO自身−AF邻域PHENO，训练平均L2匹配到关系信号，AP只提供幅度标尺且共享并集支持；不是无跨模态信息对照。G7只置乱构图PHENO，真实value、gain和主干输入不动；由于支持集为两图top12并集，错配AP也会影响AF在共享支持上的归一化，按参考实现披露此边界。decoder最后bias相减抵消，注册参数相同不代表每个单独参数都有非零梯度。

## 协议与审计

保持原871人、403ASD0/468TD1、320输入、SKFold10shuffle(random_state=10)、seed0/1/2连续十fold、450epoch/wait>200、严格valACC提升、原两Adam、图正则四图平均。原GFR i=1覆盖combined_feat，ALoss计算但不入loss；4×36潜在Token非ROI。P0原八句/CLIP与PLN保持。

历史偏差原样保留：val=test参与选模、CV前全队列标准化/监督RFE、全871人transductive、项目此前22轮探索。因此不是独立无偏/无泄漏泛化、临床或因果收益。meta OOF一阶教师训练池重叠，不是完全独立三层嵌套估计。

先自测/Stage0，四项同时成立才提前判停；否则全部八组训练。预检覆盖原P0每seed/fold初始化/前向/两Adam首步逐值一致，标签污染、预算/置乱、残差零恒等、真实梯度、epoch/fold完整恢复；正式G0逐折核对实验13–18的指标/epoch/初始化/RNG。E9仅历史描述参照。

每25epoch与fold边界原子checkpoint包含当前/最佳模型、两个Adam、RNG、best/wait/history、证据与源码签名、已提交行ID。完成任务校验后跳过，中断恢复不重置连续随机流。根与实验1–22SHA全程保护。每阶段尝试锁定单次O_APPEND写根handoff，保留此前字节；若/data I/O阻止则manifest记录实际失败。

## 复现

```bash
python /tmp/experiment23_gain_calibrated_selective_relational_alignment/run.py prepare
python /tmp/experiment23_gain_calibrated_selective_relational_alignment/run.py preflight
python /tmp/experiment23_gain_calibrated_selective_relational_alignment/run.py all
python /tmp/experiment23_gain_calibrated_selective_relational_alignment/run.py resume
python /tmp/experiment23_gain_calibrated_selective_relational_alignment/run.py refresh
python /tmp/experiment23_gain_calibrated_selective_relational_alignment/run.py report
```

all自动补齐prepare/preflight，已完成校验后跳过。迁移目录时SPROMPTGL_ROOT指定原始数据/根目录；若路径或代码变化，不能续用原签名checkpoint，应以冻结代码在新的完整副本从头重跑。参考六文件未使用接口补丁；若必须修bug，manifest必须列出逐行diff与原因并重新审计。

## 既有工作边界

[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)提供原GFR/CAM/SPGS/RES；[MGDR](https://papers.miccai.org/miccai-2024/514-Paper2494.html)已有共享/私有模态表示；[To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/view/39248)讨论对齐与冗余；[PhenoBrain](https://proceedings.mlr.press/v306/meng26f.html)已有表型条件化长程通信。本轮只检验OOF条件增益驱动的患者关系路由+FC锚+严格阴性对照组合，不声称门控/图或选择性对齐首创。

实验11有871行VERIFIED的数值身份恢复链；实验21仅因有界搜索未找到subject_IDs.txt判0/871，未复用前链。两者结论不一致，本轮不恢复身份、不引入临床量表监督。

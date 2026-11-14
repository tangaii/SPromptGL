# Experiment 79.3：CAM144疾病竞争证据记忆学习

## 状态

- 状态：`COMPLETE_AUDITED`
- 科研判定：`NO_GO_COMPETITIVE_MEMORY`
- 实际目录：`/model/spromptgl/experiment/experiment79_3_competitive_evidence_memory`
- 设备：`PPU-ZW810E`，10个fold worker；监控采样峰值显存约15,305 MiB / 98,304 MiB。
- 正式训练：20/20；最长worker 5,032.3 s，约83.9 min；任务累计CPU/GPU worker秒数为26,264.1 s。
- 独立审计：40/40最佳Checkpoint、10/10 latest；RNG不匹配0。

## 审查与基座

SPromptGL论文将GFR患者关系图、图卷积和CAM跨模态交互作为原方法组成；本实验确认项目DARG使用`PLNNoPEN`，CAM144是原OutputLayer输入。本实验不启用PEN、疾病Prompt或原始CORR256。SAM 2只借鉴有限记忆、Cross-Attention以及“读取旧记忆、完成当前预测、再更新记忆”的流式顺序；视频对象传播、掩码解码和时间位置假设没有移植。论文和源码入口见[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[SPromptGL官方代码](https://github.com/wanxixi11/SPromptGL)、[SAM 2论文](https://arxiv.org/abs/2408.00714)和[SAM 2 MemoryAttention源码](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_attention.py)。

唯一基座是只读E74.4-G2 Positive-DCR。每折从E61 epoch-10恢复原DARG、GFR、CAM、OutputLayer、三个原优化器、关系对、训练/测试划分和Python/NumPy/Torch CPU/CUDA RNG。新增分支只接收detach CAM144，原主干仍只由历史`L_base`更新。

## 两组方法

G1 CLDCM在合法外层训练患者中构建最多64个相似异类ASD/TD患者对。每对保存两侧锚点、局部双类别支持、方向、分离度和可靠性；查询时分别读取ASD和TD锚点，使用联合竞争Softmax产生两个不同的Memory Value，特征增量只由两个Value读出产生。

G2 CPDEM从合法训练内原分类正确患者提取最多48 ASD和48 TD条目。CAM144特征替换干预选出最多12个稳定正向margin维度，再用其他同类与异类训练患者验证跨患者一致性和局部区分，并保存证据Mask、联合margin下降、跨患者对照方向和综合可靠性。读取时按证据Mask检索两类记忆并竞争。

两组均采用64维Query/Key、Top-8、温度0.2、7.5%相对CAM范数上限和gate bias -1.5；无有效双类记忆时精确回退原CAM144。刷新间隔为5个epoch，第11个实际epoch结束后首次建立下一版本记忆。Fold 2 zero-based患者376、406以及所有外层测试患者均被排除。

## 六指标结果（百分比，mean ± population SD）

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 历史基座 | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |
| G1 Base-Anchored | 91.6144 ± 3.0928 | 91.4586 ± 3.0241 | 93.5477 ± 2.8555 | 89.5732 ± 3.5122 | 93.3441 ± 4.9921 | 90.8618 ± 3.1947 |
| G2 Base-Anchored | 91.9592 ± 2.6778 | 91.8166 ± 2.6429 | 93.5371 ± 2.8519 | 90.0671 ± 3.8862 | 93.5661 ± 4.6502 | 91.2252 ± 2.8081 |
| G1 Method-Selected | 91.8443 ± 3.1095 | 91.7433 ± 3.1081 | 93.3829 ± 2.8569 | 90.3415 ± 4.0185 | 93.1452 ± 3.8299 | 91.1274 ± 3.3660 |
| G2 Method-Selected | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.5690 ± 2.9695 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |

相对历史基座的Base-Anchored差值（ACC/Hard-AUC/Prob-AUC/SEN/SPE/F1，pp）为：G1 `-0.5747/-0.5707/+0.0531/-0.4939/-0.6475/-0.5972`，G2 `-0.2299/-0.2128/+0.0425/+0.0000/-0.4255/-0.2339`。Method-Selected差值为：G1 `-0.3448/-0.2860/-0.1117/+0.2744/-0.8464/-0.3317`，G2 `0.0000/0.0000/+0.0744/0.0000/0.0000/0.0000`。G2-G1在Base-Anchored ACC/Hard-AUC为`+0.3448/+0.3580`pp，在排除Fold 2后为`+0.1277/+0.1562`pp，但两者都没有超过历史基座。

排除Fold 2后，Base-Anchored G1/G2的ACC差分别为`-0.3831/-0.2554`pp，Method-Selected G1/G2分别为`-0.1277/0.0000`pp；因此结论不由Fold 2单独驱动。

## 纠错与记忆机制

在历史Base-Anchored Checkpoint上，G1错误→正确为0、正确→错误为5、净变化-5；G2为0、2、-2。排除Fold 2后分别为0、3、-3和0、2、-2。Method-Selected相对各自同epoch原始DARG，G1为6/3/+3，G2为2/0/+2（错误→正确/正确→错误/净变化）；这组数字不能替代相对历史基座的比较，因为Method-Selected epoch不同。

最佳Checkpoint中G1平均保存64对、平均可靠性约0.634；G2平均保存96条、平均可靠性约0.930、12维证据、联合margin下降约2.446。两类记忆均达到设计容量，但容量本身没有带来稳定硬分类收益。

Base-Anchored G1的平均gate约0.222、相对增量约4.82%、触及7.5%上限约43.4%；G2约0.194、6.03%、56.0%。因此门控已经比E79.2活跃，但G2仍有过半测试样本达到幅度上限，不能把增量饱和解释为可靠疾病纠错。

只读干预显示：Base-Anchored关闭全部记忆使G1硬预测变化5例、G2变化2例；但完整G1/G2相对历史基座的ACC仍分别下降0.5747/0.2299pp。G1 Pair-Association-Break产生2例硬变化，G2 Evidence-Mask-Break和Cross-Patient-Reliability-Off均为0例硬变化；G2 Memory-Content-Permuted在Method-Selected只有1例硬变化，Base-Anchored为0。记忆确实改变少量概率和局部预测，但具体跨患者证据与可靠性加权没有显示稳定、独立的硬分类贡献。

## 审计与限制

预检确认CAM形状`(871,144)`、空记忆严格等价、双路读出不同、Memory Value有梯度、原主干无新增梯度。独立审计确认：历史主干身份、Fold 2排除、测试患者不入库、查询患者自排除、Memory Value干预、Best读取旧记忆、Latest保存下一轮状态、预测和六指标独立复算均通过。历史val=test选模、CV前RFE/标准化、transductive图和Fold 2重叠仍限制无偏泛化结论。

结论是：CLDCM和CPDEM都没有在相同历史DARG最佳Checkpoint上产生可靠净纠错，也没有超过E74.4-G2。G2相对G1在硬指标上略好，但只是接近基座；证据内容干预的硬分类影响不足以支持“跨患者疾病证据已被有效学习”的主张。因此本轮按预注册规则判定`NO_GO_COMPETITIVE_MEMORY`，不追加G3/G4、Prompt、OOF教师或测试集调参。

from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np

EXP=Path(__file__).resolve().parent
RES=EXP/'results'
s=pd.read_csv(RES/'summary.csv')
p=pd.read_csv(RES/'paired_comparisons.csv')
m=pd.read_csv(RES/'mechanisms.csv')
i=pd.read_csv(RES/'interventions.csv')

def pct(x): return f'{100*float(x):.4f}'
def pm(x,y): return f'{100*float(x):.4f} ± {100*float(y):.4f}'
def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

names={
'E0':'B0-B历史参考','E1':'E1四路聚合重放','E2':'Self-only容量对照','E3':'普通邻域图传播','E4':'邻域诊断监督','E5':'学习式选择（无pair监督）','E6':'有效边pair监督（无对齐）','E7':'EC-SRGL完整方法','E8':'匹配预算统一关系','E9':'错误几何对应','E10':'错误新增患者图','E11':'关闭邻域辅助监督','E12':'去除私有残差'}
lines=[]
lines += ['# Experiment 51.3 — EC-SRGL：疾病证据约束的选择性关系图学习','',
'## 结论先行','',
'本轮完成E0–E12全部预定组别：E0历史B0-B读回10折，E1历史E1重放10折，E2–E12新增110折；GPU0单卡、最多5个并行worker，无失败任务、无NaN/OOM。独立checkpoint审计110/110通过，保存指标与重算指标最大绝对误差为0。按照预注册判断，E7为NO-GO：E7 ACC为91.5008%，低于E1的91.9592%（−0.4585个百分点），也低于B0-B的91.7294%（−0.2286个百分点）；E8匹配边预算统一关系的ACC为91.6144%，高于E7。',
'',
'本结果支持“新增邻域分支可以改变诊断输出并且邻域监督会影响优化路径”，但没有支持“非均匀患者对选择、正确跨模态几何对应和真实患者图身份带来稳定且独立的分类增量”。当前患者对选择性关系对齐路线按本轮预注册规则终止继续机械扩展。',
'',
'## 1. 审计范围与可复现身份','',
'- 本地审计了SPromptGL根源码、Experiment43.1、47.3、48–50.2、51以及51.2的实现和报告；E1基座使用Experiment51规定的`510000 + fold*100 + 1`初始化构造和E1锚点。',
'- SPromptGL本地论文PDF存在，公式与公开实现的差异以源码为准：四张患者图、两层GCN、四路聚合、原CAM和原GraphConstructLoss均保持在E1路径。MGDR、MM-GTUNets和SMGD的可用代码快照分别保存在本地临时审计目录；借鉴仅限共享/私有信息、模态不对称作用和关系几何思想，不复制其网络。',
'- 在线论文/GitHub端点本轮网络访问不可用，因此没有将不可核实的网页内容作为结果证据。实验52在本地仓库和本轮可访问临时目录中未发现，记录为`EXPERIMENT52_NOT_FOUND`。',
'- 受保护根源码和ABIDE输入SHA在`manifest.json`中记录并保持不变。ASD=0、TD=1；Hard-AUC按历史硬预测口径，Probability-AUC单独计算。',
'',
'## 2. 固定协议与方法','',
'基座是无PEN的E1：原始GraphLearn四图、完整320维四路GCN消息正确累加、原CAM和四图GraphConstructLoss。新增EC-SRGL只在原图支持内选择边：对称患者对评分得到`p_ij^m`，传播和CORR锚定几何对齐使用同源选择权重；训练pair只来自outer-train且属于对应图支持；邻域消息排除自身，无邻居时为零；邻域诊断头单独接受训练标签；图残差和私有残差独立写回48/6/10/256块。E8使用同一原图支持和相同边权预算的模态内统一缩放，E9只置换对齐距离对应，E10只置换新增分支邻居身份。',
'',
'训练保持seed=0、10-fold StratifiedKFold(shuffle=True, random_state=10)、历史`val=test`选模、450 epoch和`wait>200`、原始lr=0.0038/weight_decay=0.11；新增模块lr=0.001/weight_decay=0.001。评估协议保留CV前RFE/标准化和871节点transductive图，因此指标不能解释为无偏外部泛化估计。',
'',
'## 3. 正式分类结果（十折mean ± population SD，百分比）','',
'|组别|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in s.iterrows():
 lines.append(f"|{r['group']} {names.get(r['group'],r['name'])}|{pm(r.acc_mean,r.acc_sd)}|{pm(r.hard_auc_mean,r.hard_auc_sd)}|{pm(r.prob_auc_mean,r.prob_auc_sd)}|{pm(r.spe_mean,r.spe_sd)}|{pm(r.sen_mean,r.sen_sd)}|{pct(r.ba_mean)}|{pct(r.asd_f1_mean)}|")
lines += ['', '## 4. E7逐折配对比较','', '以下为E7减对照的百分点差值；W/T/L为E7严格胜/平/负折数。', '', '|比较|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔProbability-AUC|ΔSPE|ΔSEN|','|---|---:|---|---:|---|---:|---:|---:|']
for c in [f'E7_vs_E{i}' for i in [0,1,2,3,4,5,6,8,9,10,11,12]]:
 q=p[p.comparison==c].set_index('metric')
 a=q.loc['acc']; h=q.loc['hard_auc']; pr=q.loc['prob_auc']; sp=q.loc['spe']; se=q.loc['sen']
 lines.append(f"|{c}|{a.delta_pp:.4f}|{int(a.wins)}/{int(a.ties)}/{int(a.losses)}|{h.delta_pp:.4f}|{int(h.wins)}/{int(h.ties)}/{int(h.losses)}|{pr.delta_pp:.4f}|{sp.delta_pp:.4f}|{se.delta_pp:.4f}|")
lines += ['', '## 5. 机制诊断','',
'### 5.1 有效候选边和选择器','',
'E7最佳checkpoint的平均非零选择边数为PHENO 162,471、ANAT 238,060、FUNC 122,194、CORR 100,216；对应非零边权均值约0.487、0.473、0.484、0.392。选择器不是恒定0.5，但非恒定边权本身不能证明疾病关系正确。E7四模态训练pair判别AUC均值约为PHENO 0.903、ANAT 0.843、FUNC 0.792、CORR 0.109；CORR分支方向与同诊断代理标签不一致，整体不能解释为可靠疾病关系识别。E7训练/测试邻域头准确率均值约53.6%/53.8%，仅弱于强证据标准，且没有形成稳定的独立诊断证据。',
'',
'### 5.2 损失梯度与残差','',
'E7的新增分类梯度范数均值约1.086，pair、neighbor和align独立梯度范数约0.00345、0.00192和0.02799，说明这些路径在数值上确实连通；但连接存在不等于下游增益。E7图残差/输入范数约1.374%，私有残差/输入范数约0.834%，新增残差相对保守。固定E7 checkpoint关闭整个新增分支时平均ACC下降8.383个百分点，关闭图残差时下降7.923个百分点，平均硬预测翻转率分别10.45%和9.76%；这是固定路径敏感性，不能代替重训负控制，也不能证明真实关系具有因果价值。关闭私有残差平均ACC下降0.229个百分点、翻转率0.23%，其独立影响很小。',
'',
'### 5.3 关键消融含义','',
'- E7相对E1：ACC −0.4585 pp，Hard-AUC −0.2832 pp；完整EC-SRGL没有超过直接工程基座。',
'- E7相对E2 Self-only：ACC约持平，Hard-AUC +0.1686 pp但SPE下降2.1323 pp、SEN上升2.4695 pp；不能排除新增自身容量和阈值权衡。',
'- E7相对E3普通邻域图：ACC几乎相同（−0.0013 pp），Probability-AUC下降0.6598 pp；学习式选择没有形成稳定分类收益。',
'- E7相对E6（pair监督但无几何对齐）：ACC +0.4585 pp、Hard-AUC +0.5648 pp；几何对齐在这一对独立重训中有局部收益，但E7仍低于E1，且并非所有指标同步改善。',
'- E7相对E8匹配边预算统一关系：ACC −0.1136 pp、Hard-AUC −0.0387 pp；统一关系更高，选择性没有得到支持。',
'- E7相对E9错误几何对应：ACC +0.3448 pp、Hard-AUC +0.4459 pp，但ACC仅4/10严格胜，SPE下降1.0777 pp，证据不足以宣称正确几何对应带来稳定临床相关增量。',
'- E7相对E10错误新增患者图：ACC +0.2299 pp、Hard-AUC +0.3076 pp且Hard-AUC 7/10胜；Probability-AUC反而−0.2012 pp，且完整方法仍低于E1，因此这是有限路径差异而非完整假设成立。',
'- E7相对E11关闭邻域监督：ACC +1.0345 pp、Hard-AUC +1.1926 pp，说明邻域辅助监督改变训练结果；但E11同时去掉了一个训练约束，不能单独证明邻域消息本身是疾病特异的。',
'- E7相对E12关闭私有残差：ACC +0.1149 pp、Hard-AUC +0.4055 pp，但SPE下降3.6401 pp、SEN上升4.4512 pp；私有信息贡献不稳定且伴随明显阈值权衡。',
'',
'### 5.4 真实图、错误图和固定干预的边界','',
'E10只改变新增图分支，E1主干和GraphConstructLoss仍使用正确图；E9只置换对齐距离对应且不改变传播候选集合。因而E7/E9、E7/E10只支持各自计算路径的敏感性。E7优于E9/E10的部分指标不能被提升为“真实患者图已验证具有疾病因果关系”。固定checkpoint干预与独立重训消融已分开保存于`interventions.csv`和`paired_comparisons.csv`。',
'',
'## 6. 工程与独立审计','',
'- 语义测试10项全部通过：支持内选择、对称性、E8统一预算、E9对应置换、E10新增图置换、E12私有关闭隔离、outer-test标签隔离、有限梯度和输出路径等。',
'- 110/110新增最佳checkpoint独立读回通过；失败列表为空；最大指标重算误差0；30个E7固定干预完成。',
'- 根源码、历史实验和ABIDE数据未修改；保护SHA、fold SHA、源码SHA、结果SHA在`manifest.json`中记录。',
'- `.runtime/`未纳入交付目录，保留在scratch供审计；正式目录写入若受OSSFS Errno 5影响，交付状态与科研状态分开报告。',
'',
'## 7. 最终科研判断','',
'**NO-GO。** 本轮没有证明疾病证据约束的选择性患者邻域关系能在E1之上产生稳定、可归因的诊断增量。最直接的证据是E7低于E1，E8统一预算不低于E7，E7与普通图传播E3几乎持平；E7相对错误图的局部优势不足以抵消基座下降和指标权衡。下一步不应继续在当前患者相似图上机械增加选择器、对齐项或辅助头；若继续研究，应先改变关系证据来源或采用严格独立外部验证。',
'',
'## 8. 交付文件','',
'- `fold_metrics.csv`：130条逐折记录（E0/E1参考 + E2–E12 110折）。',
'- `summary.csv`：13组十折汇总。',
'- `paired_comparisons.csv`：E7与全部预注册对照的逐折配对。',
'- `mechanisms.csv`：残差、边权、邻域头、损失和E7独立梯度诊断。',
'- `interventions.csv`：E7固定checkpoint的base-only、graph-off、private-off干预。',
'- `manifest.json`：配置、源码、数据、fold、checkpoint及审计SHA。',
'- `semantic_tests.json`和`independent_audit.json`：工程测试与独立读回证据。',
'',
'报告生成时间：2026-10-30；结果目录为实验专属scratch，正式目录同步状态见最终交付说明。']
(RES/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('report written',len(lines),'lines',sha(RES/'report.md'))

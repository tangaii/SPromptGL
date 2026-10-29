#!/usr/bin/env python3
from pathlib import Path
import json, os, hashlib
import pandas as pd

EXP=Path(os.environ.get('DCSRA_EXP',Path(__file__).resolve().parent)); R=EXP/'results'
CFG=json.loads((EXP/'config.json').read_text()); S=pd.read_csv(R/'summary.csv'); P=pd.read_csv(R/'paired_comparisons.csv'); D=pd.read_csv(R/'diagnostics.csv'); I=pd.read_csv(R/'interventions.csv'); M=json.loads((EXP/'manifest.json').read_text()); A=json.loads((EXP/'.runtime/independent_audit.json').read_text())

def pct(m,sd): return f'{m*100:.4f} ± {sd*100:.4f}'
def avg(g,col):
 x=D[D.group==g][col]; return float(x.mean()) if len(x) else float('nan')
def find(g): return S[S.group==g].iloc[0]

lines=[]
lines += ['# Experiment 51.1 — DC-SRA-GFR：疾病条件选择性关系对齐','',
'## 执行状态','',
'- 状态：**COMPLETE_AUDITED**。E0和E1各10折独立checkpoint读回，E2–E11完成100/100个新增训练折。',
'- 资源：GPU0，最多5个并行worker；控制器完成后无失败任务。',
'- 审计：新增100/100 checkpoint独立读回，指标最大绝对误差为0；E7固定checkpoint干预30条记录。',
'- 根目录模型源码、数据和历史实验未修改；正式目录同步状态记录于manifest。','',
'## 审计依据与设计选择','',
'DC-SRA-GFR保留实验51的E1四路完整320维聚合，将新增模块限制为模态自身输入上的轻量graph/self视图、CORR锚定原型、患者级门控、关系几何约束和有界回写残差。设计参考了[SPromptGL MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)的多关系患者图和GCN传播、[MGDR MICCAI 2024](https://papers.miccai.org/miccai-2024/514-Paper2494.html)的共享/私有信息保留、[MM-GTUNets代码](https://github.com/NZWANG/MM-GTUNets)所体现的模态作用差异，以及[SMGD代码](https://github.com/greedisgood1000/SMGD)的关系几何与诊断相关性区分。SMGD的图是患者内部ROI关系，本实验只借鉴其组织思想，没有复制其图编码器。',
'',
'源码审计确认：公开GFR的四张患者图都传播完整320维输入，原实现`i==1`覆盖先前消息，`simple-2`实际反传分类损失和四图GraphConstructLoss均值，历史CAM为PEN-free结构；51.1不修改根实现，而在隔离目录复用GraphLearn和E1 GCN参数。PHENO的站点字段为PHENO切片中的20个`SITE_ID_*`列，ANAT/FUNC为质量摘要字段，CORR为256维功能连接特征。','',
'## 固定协议','',
'- ABIDE 871人，ASD=0（403）/TD=1（468），输入48/6/10/256。',
'- seed=0；10折StratifiedKFold，`shuffle=True, random_state=10`；历史`val=test`选择best，严格ACC提高保存，450 epoch、`wait>200`。',
'- E1主干GraphLearn/GCN和CAM使用历史weight decay 0.11、lr 0.0038；新增模块lr 0.001、weight decay 0.001，rho上限0.08、初始rho约0.02。',
'- 训练标签只进入outer-train分类、原型CE和训练侧损失；测试标签隔离测试通过。','',
'## E0–E11正式结果','',
'指标为十折mean ± population SD，百分比；Hard-AUC为历史硬预测口径，Probability-AUC单独报告。','',
'|组|定义|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in S.iterrows():
 lines.append(f"|{r.group}|{r['name']}|{pct(r.acc_mean,r.acc_sd)}|{pct(r.hard_auc_mean,r.hard_auc_sd)}|{pct(r.prob_auc_mean,r.prob_auc_sd)}|{pct(r.spe_mean,r.spe_sd)}|{pct(r.sen_mean,r.sen_sd)}|{pct(r.ba_mean,r.ba_sd)}|{pct(r.asd_f1_mean,r.asd_f1_sd)}|")
lines += ['', 'E0独立读回与历史B0-B一致：ACC 91.7294%、Hard-AUC 91.6046%、SPE 93.1360%、SEN 90.0732%。E1来自实验51完整scratch checkpoint读回：ACC 91.9592%，与根43.1的E1（91.6144%）不同，反映训练轨迹/初始化来源差异；不能把两者合并为一个基准。','', '## E7主方法逐折配对', '', '|比较|ΔACC pp|ACC胜/平/负|ΔHard-AUC pp|Hard胜/平/负|ΔProbability-AUC pp|ΔSPE pp|ΔSEN pp|', '|---|---:|---|---:|---|---:|---:|---:|']
for other in ['E0','E1','E2','E3','E4','E5','E6','E8','E9','E10','E11']:
 q=P[P.comparison==f'E7_vs_{other}']
 vals={r.metric:r for _,r in q.iterrows()}
 a=vals['acc']; h=vals['hard_auc']; pr=vals['prob_auc']; sp=vals['spe']; se=vals['sen']
 lines.append(f"|E7 vs {other}|{a.delta_pp:+.4f}|{int(a.wins)}/{int(a.ties)}/{int(a.losses)}|{h.delta_pp:+.4f}|{int(h.wins)}/{int(h.ties)}/{int(h.losses)}|{pr.delta_pp:+.4f}|{sp.delta_pp:+.4f}|{se.delta_pp:+.4f}|")
lines += ['',
'核心比较显示：E7完整方法为ACC 91.6157%，低于E1 0.3435个百分点、低于B0-B 0.1136个百分点；E7相对E5（无原型/对齐的自适应门控）也低0.3435个百分点。E7相对E6和E8分别有+0.3448和+0.3461个百分点ACC，但这两个对照同样没有超过E1，因此不能解释为稳定疾病增量。E7与错误对齐对应E9几乎相同（ACC +0.0013 pp，Hard-AUC +0.0030 pp），且低于错误新增图关系E10（ACC −0.2286 pp，Hard-AUC −0.2072 pp），没有形成正确患者关系优势。','',
'## 机制诊断','',
'### E1主干和新增图消息','',
'各新增组均从同一fold E1未训练主干锚点开始。E7的E1主干消息相对输入约为'+f"{avg('E7','base_message_ratio')*100:.4f}"+'%，新增残差相对原始320维输入约为'+f"{avg('E7','new_residual_ratio_input')*100:.4f}"+'%。四个新增graph/self分支的平均范数差异为'+f"{avg('E7','graph_self_delta_0'):.3f}, {avg('E7','graph_self_delta_1'):.3f}, {avg('E7','graph_self_delta_2'):.3f}, {avg('E7','graph_self_delta_3'):.3f}"+'，说明新增路径不是恒等空分支；但残差仍然很小。E10错误新增图关系与E7的独立训练结果差异不支持新增关系身份价值。','',
'### CORR疾病原型','',
'在E6–E11中，CORR原型没有退化成单类常数：平均训练原型准确率约'+f"{D[D.proto_active==1].proto_train_acc.mean()*100:.2f}"+'%，测试侧诊断准确率约'+f"{D[D.proto_active==1].proto_test_acc.mean()*100:.2f}"+'%，但这只是辅助原型任务表现，不能替代最终分类增量证据。E7原型距离均值约'+f"{avg('E7','prototype_distance'):.4f}"+'。原型条件本身并未把E7提升到E1之上。','',
'### 动态门控','',
'E7四路门控均值约为'+f"{avg('E7','gate_mean_all'):.4f}"+'，标准差约'+f"{avg('E7','gate_sd_all'):.4f}"+'；PHENO/ANAT/FUNC/CORR均值分别为'+f"{avg('E7','gate_mean_pheno'):.4f}, {avg('E7','gate_mean_anat'):.4f}, {avg('E7','gate_mean_func'):.4f}, {avg('E7','gate_mean_corr'):.4f}"+'。它们接近0.5且未显示强患者级选择。固定E7 checkpoint强制gate=0的平均预测翻转率为0，ACC/Hard-AUC变化为0；因此门控虽然有梯度路径，但没有形成可测的硬分类依赖。','',
'### 关系几何对齐和对应关系','',
'E7三组CORR锚定对齐权重均值约为'+f"{avg('E7','align_w_pheno'):.4f}, {avg('E7','align_w_anat'):.4f}, {avg('E7','align_w_func'):.4f}"+'，对齐raw均值约'+f"{avg('E7','align_raw'):.6g}"+'。E7相对E6（关闭对齐）ACC下降/上升数值见配对表，实际为+0.3448 pp；但E6仍低于E1，不能把该局部差异解释为方法成功。E7与E9错误辅助对应几乎相同，说明正确跨模态患者对应没有获得独立证据。','',
'### 模态互补分支与固定模型干预','',
'E7相对E11（关闭C直接贡献）ACC +0.1162 pp、Hard-AUC +0.1755 pp，属于小幅、跨折不稳定的差异。E7固定checkpoint关闭private分支时，平均硬预测ACC和Hard-AUC均变化0，Probability-AUC约−0.0053个百分点；说明新增残差实际进入CAM数值图，但幅度不足以稳定改变分类决策。','',
'## 竞争解释','',
'|解释|证据|判断|', '|---|---|---|',
'|新增参数容量|E2为91.7294%，与B0-B相同；E7未超过E2|不能把E7收益归因于容量|',
'|普通图传播|E3为91.7294%，E10错误新增图为91.8443%，高于E7|E7没有显示正确新增关系优势|',
'|疾病原型|E6为91.2709%，E7提高0.3448 pp但仍低于E1；原型测试准确率约85.5%|原型有可学习信号，但未产生完整方法增量|',
'|动态门控|E7门控接近0.5，gate-zero固定干预无硬预测变化|门控没有成为稳定分类控制信号|',
'|关系几何对齐|E7高于E6/E8的局部差异小，E7与E9相同|没有证明正确关系对齐的独立价值|',
'|传播尺度|新增残差约0.13%输入范数，固定干预不改硬预测|主要限制是新增路径过弱，而非主干崩溃|',
'|评估协议|历史val=test、CV前RFE/标准化和transductive图保留|限制外部泛化解释，不能据此宣称临床性能|',
'', '## 科研判定', '', '**NO-GO**。DC-SRA-GFR没有超过实验51 E1（ACC −0.3435 pp，Hard-AUC −0.2794 pp），也没有超过历史B0-B（ACC −0.1136 pp，Hard-AUC −0.1406 pp）。E7相对E6/E8的局部优势没有在E9/E10关系负控制中复现，E7与错误跨模态对应几乎相同，新增分支固定干预也没有改变硬分类。当前证据支持：CORR原型可以学到一定训练侧标签相关表示，但患者级门控、正确跨模态对应和新增患者图关系尚未形成可归因的疾病增量；新增残差路径存在但过弱。',
'', '该结论限于单seed、历史`val=test`、CV前预处理和871节点transductive图协议。它不构成无偏外部泛化或临床因果证明，也不支持继续在同一DC-SRA结构上做测试驱动的参数搜索。','',
'## 独立审计与交付','',
'- `independent_audit.json`：100/100 readback，30条E7固定checkpoint干预，失败数0，最大指标误差0。',
'- 语义测试：`.runtime/semantic_tests.json` 为PASS，覆盖E1四图、graph/self共享参数、非零残差、标签隔离、E9/E10关系扰动。',
'- 保护SHA及代码SHA见`manifest.json`。',
'- 运行位置：`/tmp/SPromptGL-E51_1/experiment51_1_dcsra_gfr`；正式`/data`目录同步和handoff追加需单独记录实际I/O状态。',
'- 结果文件：`results/fold_metrics.csv`（120条）、`results/summary.csv`（12组）、`results/paired_comparisons.csv`（55行）、`results/diagnostics.csv`/`mechanism.csv`（100条）、`results/interventions.csv`（30条）。']
(R/'report.md').write_text('\n'.join(lines)+'\n')
# update manifest after report exists
man=json.loads((EXP/'manifest.json').read_text()); man.update(report_sha=hashlib.sha256((R/'report.md').read_bytes()).hexdigest(),status='COMPLETE_AUDITED'); (EXP/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2))
print('REPORT_WRITTEN',hashlib.sha256((R/'report.md').read_bytes()).hexdigest())

from pathlib import Path
import pandas as pd, json, numpy as np, hashlib, csv, os
root=Path('/data/adodas/SPromptGL-main/experiment/experiment48_2_care_gfr')
res=root/'results'; sm=pd.read_csv(res/'summary.csv'); pa=pd.read_csv(res/'paired_comparisons.csv'); dg=pd.read_csv(res/'diagnostics.csv'); sd=pd.read_csv(res/'site_diagnostics.csv')
fmt=lambda x:f'{x:.4f}'
lines=[]
lines += ['# Experiment 48.2 — CARE-GFR正式报告','',
'## 执行状态','',
'本轮完成100/100个组折任务：E0为已核验历史B0-B checkpoint的10折读回，E1–E9为同一冻结实现下重新训练的90折。独立进程重新加载并核算100/100折预测、混淆矩阵、Hard-AUC和Probability-AUC，预测SHA全部通过。正式训练累计任务wall time为13007.1秒（约3.61小时；这是各折耗时之和，不等于两卡并行的日历时间）。GPU0/1分别承担fold1–5与fold6–10；没有遗留worker。','',
'训练遵循seed=0、10折StratifiedKFold（shuffle=True、random_state=10）、871节点transductive图、历史val=test checkpoint选择和全局预处理/CV前RFE。后四项属于历史协议偏倚，因此结果不是严格无偏的外部泛化估计。','',
'训练开始前发现逐候选`float(tensor)`造成的确定性GPU同步瓶颈；在不改变候选池、传播度数、攻击预算或选择目标的前提下将攻击选择向量化，并清空后统一重跑E1–E9。所有90个新增任务代码签名一致。','',
'## 四项正式指标（mean ± population SD，百分比）','',
'|组|定义|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|','|---|---|---:|---:|---:|---:|---:|']
for _,r in sm.iterrows():
 lines.append(f"|E{int(r.group_id)}|{r['group']}|{r.acc_mean_pct:.4f} ± {r.acc_sd_pct:.4f}|{r.hard_auc_mean_pct:.4f} ± {r.hard_auc_sd_pct:.4f}|{r.spe_mean_pct:.4f} ± {r.spe_sd_pct:.4f}|{r.sen_mean_pct:.4f} ± {r.sen_sd_pct:.4f}|{r.probability_auc_mean_pct:.4f} ± {r.probability_auc_sd_pct:.4f}|")
lines += ['', 'E0的91.7294% ACC、91.6046% Hard-AUC、93.1360% SPE和90.0732% SEN与历史参考逐项一致。E4 FULL_CARE_GFR的ACC为58.7800%，因此本轮初筛为`SCREEN_ACC_FAIL`。','',
'## 关键配对差（E4为主方案；百分点；W/T/L为十折严格胜/平/负）','',
'|比较|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSPE|ΔSEN|','|---|---:|---|---:|---|---:|---:|']
for _,r in pa.iterrows():
 lines.append(f"|{r['comparison']}|{r.acc_delta_pp:.4f}|{r.acc_win_tie_loss}|{r.hard_auc_delta_pp:.4f}|{r.hard_auc_win_tie_loss}|{r.spe_delta_pp:.4f}|{r.sen_delta_pp:.4f}|")
lines += ['', '## 论文、仓库与历史源码审计','', '- 已读取并核对可缓存的SPromptGL MICCAI 2025、DeepASD 2024和AAAI 2026论文，以及隔离快照中的network.py、layers.py、utils.py、model.py和训练入口；实现以实际B0-B代码路径为准。', '- RCMVL、UMCGL和ICML 2025图对抗论文PDF在本次服务器审计目录中不存在（.runtime/source_history_audit.json记录exists=false），因此不声称完成这些原文的逐式核验。', '- 该缺失不改变本轮训练结果，但限制了相关文献边界的完整复核。', '', '## 方法与审计对象','',
'- E1是工程修复对照：四路消息均进入融合、候选Top-24/传播Top-16和低衰减残差消息块；这些修复不计作CARE创新。E2加入仅用outer-train患者对监督的DREL；E3加入同患者对的CATA可信共识；E4加入受限候选边的ACRL困难替换和稳定性损失；E5去掉共识；E6为特征来源GRL控制；E7为站点/outer-split分层患者身份错配；E8只置乱DREL监督标签；E9为同预算随机合法边替换。',
'- 原始源码审计确认：原四路GCN都传播完整320维输入；`GFR.forward`中PHENO分支被`i==1`重新赋值覆盖；simple-2计算但不反传ALoss；原`weight_decay=0.11`导致GCN消息收缩。根源码、历史数据和既有实验保持只读，审计记录在`.runtime/source_history_audit.json`。',
'- E0没有把历史数值手工写入；其预测和四项指标从历史checkpoint独立读回。E1–E9的训练/恢复记录、best checkpoint、预测SHA和统一代码签名在`.runtime/tasks/`与`.runtime/checkpoints/`。','',
'## DREL、CATA与图依赖机制','']
# pair auc means
for gid,name in [(1,'E1'),(2,'E2 DREL'),(3,'E3 CATA'),(4,'E4 CARE'),(5,'E5 no-consensus'),(6,'E6 GRL'),(7,'E7 wrong relation'),(8,'E8 shuffled labels'),(9,'E9 random perturb')]:
 q=dg[dg.group_id==gid]
 vals=[]
 for z in q.pair_auc_trace.dropna():
  try: vals += [float(v) for v in json.loads(z)]
  except: pass
 pairmean=np.mean(vals) if vals else float('nan')
 pl=q.pair_loss.mean() if len(q) else float('nan'); cl=q.cons_loss.mean() if len(q) else float('nan'); rl=q.robust_loss.mean() if len(q) else float('nan')
 lines.append(f'- {name}：四关系患者对AUC均值={pairmean:.4f}；pair loss均值={pl:.4g}；consensus loss均值={cl:.4g}；robust loss均值={rl:.4g}。')
q4=dg[dg.group_id==4]; f4=sm[sm.group_id==4].iloc[0]
full=f4.acc_mean_pct; nog=q4.message_ablation_acc.mean()*100; flip=q4.prediction_flip_rate_vs_no_graph.mean()*100; wins=int((q4.acc_mean_pct if False else 0))
# merge for graph ablation wins
fm=pd.read_csv(res/'fold_metrics.csv'); qq=fm[fm.group_id==4][['fold','acc']].merge(q4[['fold','message_ablation_acc']],on='fold')
lines += [f'- E4完整图的ACC为{full:.4f}%，关闭全部跨患者消息后的ACC为{nog:.4f}%；7/10折完整图高于无图路径，平均预测翻转率为{flip:.2f}%。这证明CARE分支会改变CAM输出，但不能证明关系是疾病有效的；E4仍比E0低32.9493个百分点。',
'- DREL患者对AUC接近0.5，且E2相对E1的ACC/Hard-AUC分别下降28.3568/29.0393个百分点；当前关系监督没有产生可识别的疾病关系证据。',
'- CATA的E3相对E2继续下降1.2618个百分点，E4相对E5（去掉共识）下降1.8365个百分点。共识损失并未带来独立收益，且其数值在部分折显著增大，提示当前评分器/损失尺度不稳定。',
'- ACRL的E4相对E3下降0.6883个百分点；E4相对E9随机扰动仅+0.3318个百分点（ACC W/T/L=1/8/1，Hard-AUC W/T/L=5/3/2），同时SPE下降8.0620个百分点、SEN上升10.1220个百分点。该小幅差异不能支持“最坏关系对抗优于随机扰动”。',
'- E4相对E7真实关系身份错配为−1.2631个百分点（ACC W/T/L=1/3/6），说明正确患者关系没有可重复的独立诊断增量；错误关系控制反而更高。E4相对E8为−0.3461个百分点，真实DREL标签也没有独立贡献。',
'- E4相对E6特征GRL控制为−1.0345个百分点，未显示CARE反事实关系对抗优于普通控制。']
lines += ['', '## 站点与关系敏感性','']
for gid,label in [(4,'E4真实关系'),(7,'E7错配关系')]:
 q=sd[sd.group_id==gid].groupby('relation')[['same_site_rate','same_label_rate','cross_site_same_label_rate']].mean()
 lines.append(f'**{label}**（训练接收者边，按关系均值）：')
 for rel,r in q.iterrows(): lines.append(f'- {rel}：同站点率={r.same_site_rate:.4f}，同标签率={r.same_label_rate:.4f}，跨站点同标签率={r.cross_site_same_label_rate:.4f}。')
lines += ['这些边的同标签率约0.50，跨站点同标签率也约0.50，没有显示稳定疾病富集；E7错配确实改变了患者身份关系，但其性能没有低于E4以外的偶然折差异，不能把站点统计相关性解释为ASD生物学证据。完整明细在`results/site_diagnostics.csv`。','',
'## 预注册判定','',
'- `GO_DREL`: NO-GO；E2相对E1为强负增量且pair AUC接近随机。',
'- `GO_CATA`: NO-GO；E3不优于E2，E4也不优于去共识E5。',
'- `GO_ACRL`: NO-GO；E4不优于E3，且相对E9仅有不稳定的小幅均值差。',
'- `GO_VS_GRL`: NO-GO；E4不优于E6。',
'- `GO_REAL_RELATION`: NO-GO；E4不优于E7。',
'- `GO_DISEASE_RELATION_LABEL`: NO-GO；E4不优于E8。',
'- `GO_GRAPH_RELIANCE`: 仅可标记为`GRAPH_DEPENDENCE_DETECTED`：关闭消息会改变一部分输出，但这不是疾病增量或关系正确性的证明。',
'- `GO_FINAL_MICCAI_CANDIDATE`: NO-GO。E4没有超过91.73%，也没有通过关系身份、DREL标签、CATA和ACRL的归因对照。','',
'## 最终科研结论','',
'CARE-GFR在本冻结实现和历史评估协议下没有超过91.73% ACC。修复版E1本身已比B0-B低2.6424个百分点；加入DREL/CATA/ACRL后性能进一步坍塌，说明当前疾病关系评分和一致性/鲁棒损失没有形成可靠的患者边证据。正确患者关系没有产生独立疾病诊断增量，反事实关系对抗也没有被证明优于普通GRL控制或随机扰动。图消息对输出有可观测影响，但影响不能被解释为有益的疾病信息。',
'', '本实验不支持把CARE-GFR作为MICCAI 2027第一创新点直接推进。下一步若继续，应先独立解决关系评分的概率校准、损失尺度和训练标签可识别性，并在不使用外层测试标签的严格协议下重新验证；本轮不进行事后调参。','',
'## 交付与可追溯性','',
'- 实验目录：`/data/adodas/SPromptGL-main/experiment/experiment48_2_care_gfr`。',
'- 结果：`results/fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`diagnostics.csv`、`site_diagnostics.csv`、`report.md`。',
'- 任务状态：`.runtime/tasks/`共100个COMPLETE；`.runtime/preflight.json`为PASS；`.runtime/sources/`保存隔离源码快照。',
'- 统一训练签名：`4891127b596f1a65ebda893e502478e087372cdfe8152bb494e57f11541b3826`；独立读回100/100通过。',
'- 该报告保留历史transductive、全局预处理/CV前RFE、val=test和单seed限制，不能替代严格无偏的外部验证。']
(res/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('wrote',res/'report.md', 'sha',hashlib.sha256((res/'report.md').read_bytes()).hexdigest())

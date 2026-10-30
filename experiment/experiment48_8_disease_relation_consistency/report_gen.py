#!/usr/bin/env python3
import csv,glob,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; RT=HERE/'.runtime'; RES=HERE/'results'
active=[json.loads(Path(f).read_text()) for f in glob.glob(str(RT/'tasks/E*_f*.json'))]
active=[r for r in active if r.get('status')=='COMPLETE']
e0=[json.loads(Path(f).read_text()) for f in glob.glob(str(HERE/'e0/.runtime/tasks/E11_f*.json'))]
names={0:'E0_G-F-05_RETRAIN',1:'E1_CANDIDATE_SRD',2:'E2_RELIABLE_SCA',3:'E3_BOUNDARY_CRA',4:'E4_SRD_SCA',5:'E5_SRD_CRA',6:'E6_SCA_CRA',7:'E7_FULL_APCR_DRC',8:'E8_SHUFFLED_DISEASE_RELATION',9:'E9_RANDOM_LEGAL_ATTACK',10:'E10_PSEUDO_ATTACK_NO_EDGE_CHANGE'}
groups={0:e0}
for r in active: groups.setdefault(int(r['group_id']),[]).append(r)
metric_keys=['acc','hard_auc','spe','sen','probability_auc']
def pct(r,k): return 100*float(r[k])
def summarize(gid,rs):
 row={'group_id':gid,'group':names[gid],'n':len(rs)}
 for k in metric_keys:
  v=np.array([pct(r,k) for r in rs]);row[k+'_mean']=float(v.mean());row[k+'_sd_pop']=float(v.std(ddof=0))
 row['best_epoch_mean']=float(np.mean([r['best_epoch'] for r in rs]));row['epochs_executed_mean']=float(np.mean([r['epochs_executed'] for r in rs]))
 ds=[r.get('diagnostics',{}) for r in rs]
 row['alpha_best_mean']=float(np.mean([d.get('alpha_best',.0015) for d in ds]));row['residual_ratio_mean']=float(np.mean([d.get('relative_residual_norm',0.) for d in ds]))
 row['srd_auc_mean']=float(np.mean([np.mean((d.get('best_epoch_trace') or {}).get('pair_stats',{}).get('auc',[.5]*4)) for d in ds]))
 row['srd_candidate_pairs_best_mean']=float(np.mean([((d.get('best_epoch_trace') or {}).get('pair_stats') or {}).get('stats',{}).get('candidate_pairs_used',0) for d in ds]))
 row['srd_top16_edge_auc_best_mean']=float(np.mean([((d.get('best_epoch_trace') or {}).get('pair_stats') or {}).get('stats',{}).get('top16_fit_edge_auc',.5) for d in ds]))
 row['sca_active_fraction_mean']=float(np.mean([int(((d.get('best_epoch_trace') or {}).get('sca_diagnostics') or {}).get('teacher_pairs',0)>0) for d in ds]))
 row['cra_target_ce_gain_mean']=float(np.mean([(((d.get('best_epoch_trace') or {}).get('cra') or {}).get('adv_target_nll',0)-((d.get('best_epoch_trace') or {}).get('cra') or {}).get('clean_target_nll',0)) for d in ds]))
 return row
srows=[summarize(gid,groups[gid]) for gid in range(11)]
RES.mkdir(parents=True,exist_ok=True)
with open(RES/'summary.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(srows[0]));w.writeheader();w.writerows(srows)
frows=[]
for gid in range(11):
 for r in groups[gid]:
  d=r.get('diagnostics',{});frows.append({'group_id':gid,'group':names[gid],'fold':r['fold'],**{k:pct(r,k) for k in metric_keys},'best_epoch':r['best_epoch'],'epochs_executed':r['epochs_executed'],'best_sha':r.get('best_sha'),'prediction_sha256':r.get('prediction_sha256'),'initial_bundle_sha':r.get('initial_bundle_sha'),'signature':r.get('signature'),'alpha_best':d.get('alpha_best'),'relative_residual_norm':d.get('relative_residual_norm')})
with open(RES/'fold_metrics.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(frows[0]));w.writeheader();w.writerows(sorted(frows,key=lambda x:(x['group_id'],x['fold'])))
by={(r['group_id'],r['fold']):r for r in frows}; pairs=[]
for ga in range(11):
 for gb in range(ga+1,11):
  aa=[by[(ga,f)] for f in range(1,11)];bb=[by[(gb,f)] for f in range(1,11)];row={'a_id':ga,'a':names[ga],'b_id':gb,'b':names[gb],'n':10}
  for k in metric_keys:
   da=np.array([x[k]-y[k] for x,y in zip(aa,bb)]);row[f'delta_{k}_a_minus_b_mean']=float(da.mean());row[f'delta_{k}_a_minus_b_sd']=float(da.std(ddof=0));row[f'{k}_win_tie_loss_a_vs_b']=f'{int((da>0).sum())}/{int((da==0).sum())}/{int((da<0).sum())}'
  pairs.append(row)
with open(RES/'paired_comparisons.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
def getpair(a,b):
 for x in pairs:
  if x['a_id']==a and x['b_id']==b:return x
  if x['a_id']==b and x['b_id']==a:return x
 raise KeyError((a,b))
def delta(a,b,k='acc'):
 x=getpair(a,b); key=f'delta_{k}_a_minus_b_mean'; return x[key] if x['a_id']==a else -x[key]
def wtl(a,b,k='acc'):
 x=getpair(a,b); v=x[f'{k}_win_tie_loss_a_vs_b']
 return v if x['a_id']==a else '/'.join(v.split('/')[::-1])
inter=list(csv.DictReader(open(RES/'interventions.csv')))
def iagg(gid,c):
 v=np.array([float(r[c]) for r in inter if int(r['group_id'])==gid]);return float(v.mean()) if len(v) else float('nan')
mech=list(csv.DictReader(open(RES/'mechanism_diagnostics.csv')))
def fmt(r,k): return f"{r[k+'_mean']:.4f}±{r[k+'_sd_pop']:.4f}"
lines=['# Experiment 48.8 — Disease-Relation Consistent APCR','', '## 执行与审计','', '- E0 G-F-05 重训 10/10 折通过逐折 ACC、Hard-AUC、SPE、SEN、Probability-AUC、best epoch、执行 epoch、预测 SHA、初始化 bundle SHA 和 split SHA 的历史核验。','- E1–E10 完成 100/100 折；正式训练使用统一 v3 源码签名 `a6e3b13a7d99a56676a07b276e2d88900a4757c9eb3028f97108c0f59d49630f`。','- 独立进程重新加载 100 个最佳 checkpoint，重算五项指标并核对预测 SHA；`audit_status.json` 为 100/100、failed=0、independent_recompute=true。','- 早期旧签名的 partial tasks 以及 E10_f01、E10_f02–f04 的旧签名记录均归档并排除；所有正式 E10 结果均来自统一 v3 签名。','- /data 写入探针仍报 `Errno 5 Input/output error`，因此当前交付暂存于 `/tmp/SPromptGL-E48.8`，handoff 另存 pending。','', '## 五项正式指标（百分比，10折 population SD）','', '|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|','|---|---:|---:|---:|---:|---:|']
for r in srows: lines.append(f"|E{r['group_id']} {r['group']}|{fmt(r,'acc')}|{fmt(r,'hard_auc')}|{fmt(r,'spe')}|{fmt(r,'sen')}|{fmt(r,'probability_auc')}|")
lines += ['', '## 关键配对差（前者减后者，pp；胜/平/负）','']
for a,b in [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,7),(9,7),(10,7),(4,1),(4,2),(5,1),(5,3),(6,2),(6,3)]:
 lines.append(f"- E{a}−E{b}: ACC {delta(a,b):+.4f} ({wtl(a,b)}), Hard-AUC {delta(a,b,'hard_auc'):+.4f} ({wtl(a,b,'hard_auc')}), SPE {delta(a,b,'spe'):+.4f}, SEN {delta(a,b,'sen'):+.4f}, Prob-AUC {delta(a,b,'probability_auc'):+.4f}.")
lines += ['', '## 机制诊断','', '|组|SRD AUC均值|候选对(best epoch)|Top-16 edge AUC|SCA active折比例|CRA局部CE增量|残差/GFR|α关闭 ΔACC|错配 ΔACC|评分错配 ΔACC|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for gid in range(1,11):
 r=next(x for x in srows if x['group_id']==gid);mm=[x for x in mech if int(x['group_id'])==gid];edge=np.mean([float(x['srd_top16_fit_edge_auc']) for x in mm]);cand=np.mean([float(x['srd_candidate_pairs']) for x in mm]);sca=np.mean([int(float(x['sca_teacher_pairs'])>0) for x in mm]);cg=np.mean([float(x['cra_target_ce_gain']) for x in mm]);
 lines.append(f"|E{gid}|{r['srd_auc_mean']:.4f}|{cand:.1f}|{edge:.4f}|{sca:.3f}|{cg:.6f}|{r['residual_ratio_mean']:.5f}|{iagg(gid,'delta_acc_alpha_off_pp'):+.4f}|{iagg(gid,'delta_acc_wrong_relation_pp'):+.4f}|{iagg(gid,'delta_acc_score_perturbed_pp'):+.4f}|")
lines += ['', 'Candidate-aware SRD 的 best epoch 记录显示全局 1024 + 候选 1024、四类各约 256 的采样；候选 Top-16 edge AUC 约 0.53–0.55，尚未形成强疾病关系判别。E7 的 SCA 教师覆盖约 97% 训练 epoch，但这只证明约束被执行，不证明方向正确。E7 CRA 的局部 CE 增量约 1e-4，攻击边变化率约 0.36%，而 E10 无边变化；因此 CRA 的实际关系扰动很弱。','', '## 固定 checkpoint 干预','', f"APCR 关闭保留 G-F-05 的 LegacyFixedGFR 与 CAM；E7 关闭 APCR 后平均 ACC 变化为 {iagg(7,'delta_acc_alpha_off_pp'):+.4f} pp、预测翻转均值 {iagg(7,'flip_alpha_off'):.2f}；统一节点错配后的 E7 平均 ACC 变化为 {iagg(7,'delta_acc_wrong_relation_pp'):+.4f} pp，评分与患者 ID 错配后的变化为 {iagg(7,'delta_acc_score_perturbed_pp'):+.4f} pp。低 α 残差确实存在（E7 残差/GFR 约 {next(x for x in srows if x['group_id']==7)['residual_ratio_mean']:.4f}），但没有形成可检测的硬预测依赖。",'', '## 科学判断','', '- `BASE_REPRODUCTION_STATUS=PASS`：E0 严格复现 G-F-05。','- `TRAINING_STATUS=100/100_COMPLETE`；`AUDIT_STATUS=PASS`。',f"- 候选 SRD、可靠教师 SCA、边界 CRA 均在代码和单元测试层面执行，但 E1–E7 没有超过 E0；E7 相对 E0 ACC 为 {delta(7,0):+.4f} pp，Hard-AUC 为 {delta(7,0,'hard_auc'):+.4f} pp。",f"- E7 相对 E8（疾病标签置乱）ACC 为 {delta(7,8):+.4f} pp、相对 E9（随机合法攻击）为 {delta(7,9):+.4f} pp；E10 伪攻击不改边却仍改变训练目标。因此真实疾病关系监督、最坏攻击选择及三模块正向交互均未得到分类证据。",'- 这些结果支持“关系证据模块确实运行”的工程结论，不能支持“患者关系传播产生独立 ASD 诊断增量”的科研结论。`MICCAI_CANDIDATE_STATUS=NO-GO`。','', '## 限制','', '沿用全局预处理/CV 前 RFE、transductive 871 节点图、`val=test` checkpoint 选择和单 seed；因此这些十折结果不是严格无偏的外部泛化估计。11组、110折和机制选择仍存在多重比较风险。','', '产物：`/tmp/SPromptGL-E48.8/results/`；旧实现和未纳入统计的任务见 `.runtime/formal_v2_pre_fix/`。']
(RES/'report.md').write_text('\n'.join(lines)+'\n')
print({'summary_rows':len(srows),'pair_rows':len(pairs),'report':str(RES/'report.md')})

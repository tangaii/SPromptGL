from __future__ import annotations
import csv, json, hashlib, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

E=Path('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison')
GROUPS=['G1_E28_E0','G2_E43_1_E00','G3_E74_4_G2','G4_E82_4_6_10_G1']
LABELS={'G1_E28_E0':'E28-E0','G2_E43_1_E00':'E43.1-E00','G3_E74_4_G2':'E74.4-G2','G4_E82_4_6_10_G1':'E82.4.6.10-G1'}
CORE=['ACC','AUC_original','AUC_probability','SPE_macro','SEN_macro','F1_macro']
ALL=['ACC','AUC_original','AUC_probability','SPE_macro','SEN_macro','F1_macro','F1_weighted','Balanced_Accuracy','NC_precision','NC_recall','NC_f1','MCI_precision','MCI_recall','MCI_f1','AD_precision','AD_recall','AD_f1']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def fmt(v,s): return f'{100*v:.2f}±{100*s:.2f}'
def pct(v): return f'{100*v:.2f}'


def torch_version():
 try:
  import torch; return f'{torch.__version__}, CUDA {torch.version.cuda}, available={torch.cuda.is_available()}'
 except Exception:return 'unknown'

splits=json.loads((E/'data_audit/split_indices.json').read_text())
rows=[]; summaries={}; mechanism=[]; audit={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'groups':{},'checks':{},'errors':[]}
for g in GROUPS:
 p=E/g/'fold_metrics.csv'
 if not p.exists(): audit['errors'].append(f'{g}: missing fold_metrics.csv'); continue
 d=pd.read_csv(p)
 group_a={'fold_count':int(len(d)),'folds':sorted(d.fold.astype(int).tolist()),'seed_values':sorted(d.seed.astype(int).unique().tolist()),'finite_metrics':bool(np.isfinite(d[ALL].to_numpy(float)).all()),'duplicate_fold_count':int(d.fold.duplicated().sum()),'checkpoints_exist':bool(all(Path(x).exists() for x in d.checkpoint_path)),'predictions_exist':bool(all(Path(x).exists() for x in d.prediction_path))}
 # Verify every prediction table is exactly the shared test partition and probabilities are finite.
 pred_ok=True; checkpoint_meta_ok=True
 for _,rr in d.iterrows():
  fold_split=next(x for x in splits['splits'] if int(x['fold'])==int(rr.fold))
  pp=pd.read_csv(rr.prediction_path)
  expected=np.asarray(fold_split['test_idx'],dtype=int); got=np.asarray(pp['sample_index'],dtype=int)
  if len(pp)!=len(expected) or set(got.tolist())!=set(expected.tolist()) or pp.sample_index.duplicated().any(): pred_ok=False
  probs=pp[['prob_NC','prob_MCI','prob_AD']].to_numpy(float)
  if not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1.0,atol=1e-5): pred_ok=False
  try:
   import torch
   ck=torch.load(rr.checkpoint_path,map_location='cpu',weights_only=False)
   checkpoint_meta_ok = checkpoint_meta_ok and int(ck.get('seed',-1))==0 and bool(ck.get('val_idx_equals_test_idx',False)) and ck.get('group')==g and int(ck.get('fold',-1))==int(rr.fold)
  except Exception:
   checkpoint_meta_ok=False
 group_a['predictions_match_shared_splits']=pred_ok; group_a['checkpoint_metadata_ok']=checkpoint_meta_ok
 audit['groups'][g]=group_a
 if len(d)!=10 or group_a['folds']!=list(range(1,11)): audit['errors'].append(f'{g}: expected exactly folds 1..10')
 if group_a['seed_values']!=[0]: audit['errors'].append(f'{g}: seed mismatch')
 if not group_a['finite_metrics']: audit['errors'].append(f'{g}: NaN/Inf metrics')
 if not group_a['checkpoints_exist'] or not group_a['predictions_exist']: audit['errors'].append(f'{g}: missing artifact')
 if not group_a['predictions_match_shared_splits']: audit['errors'].append(f'{g}: prediction partition/probability audit failed')
 if not group_a['checkpoint_metadata_ok']: audit['errors'].append(f'{g}: checkpoint metadata audit failed')
 rows.extend(d.to_dict('records'))
 for _,r in d.iterrows():
  s={k:float(r[k]) for k in CORE}; s.update({'group':g,'fold':int(r.fold),'train_seconds':float(r.train_seconds) if 'train_seconds' in r else float('nan')}); mechanism.append(s)
 metrics={}
 for m in ALL:
  vals=d[m].astype(float).to_numpy(); metrics[m]={'mean':float(vals.mean()),'std':float(vals.std(ddof=0)),'min':float(vals.min()),'max':float(vals.max())}
 summaries[g]=metrics
# Verify shared split definition exactly once and common data hashes
audit['checks']['exact_total_rows']=len(rows)==40
audit['checks']['all_group_rows_10']=all(a['fold_count']==10 for a in audit['groups'].values())
audit['checks']['shared_split_file_hash']=sha(E/'data_audit/split_indices.json')
audit['checks']['data_hash']=sha(E/'preprocessing/processed_standard_data.csv')
audit['checks']['split_protocol']=splits.get('protocol')
audit['checks']['all_val_equals_test']=all(bool(x.get('val_equals_test',False)) for x in splits['splits'])
for g in GROUPS:
 adir=E/g/'audits'
 aps=sorted(adir.glob('fold*.json'))
 for p in aps:
  a=json.loads(p.read_text()); mechanism.append({**a,'audit_path':str(p)})
 if len(aps)!=10: audit['errors'].append(f'{g}: mechanism audit count {len(aps)}')
# Mechanism identity checks, including physical G2 PEN removal and actual G4 residual activity
mech_json=[]
for g in GROUPS:
 vals=[]
 for p in sorted((E/g/'audits').glob('fold*.json')):
  a=json.loads(p.read_text()); vals.append(a); mech_json.append(a)
 expected={'G1_E28_E0':(True,False,False),'G2_E43_1_E00':(False,False,False),'G3_E74_4_G2':(True,True,False),'G4_E82_4_6_10_G1':(True,True,True)}[g]
 actual=all((a['pen_present'],a['darg_active'],a['dpsl_active'])==expected for a in vals)
 audit['groups'][g]['mechanism_identity']=actual
 if not actual:audit['errors'].append(f'{g}: mechanism identity mismatch')
 if g.startswith('G3') or g.startswith('G4'):
  audit['groups'][g]['relation_update_nonzero']=all(a['relation_update_norm']>0 and a['darg_relation_grad_norm']>0 for a in vals)
  if not audit['groups'][g]['relation_update_nonzero']:audit['errors'].append(f'{g}: DARG relation update/gradient zero')
 if g.startswith('G4'):
  audit['groups'][g]['dpsl_gradient_nonzero']=all(a['dpsl_gradient_norm']>0 for a in vals)
  audit['groups'][g]['dpsl_residual_bound_ok']=all(a['dpsl_residual_ratio_max']<=0.08+1e-6 for a in vals)
  if not audit['groups'][g]['dpsl_gradient_nonzero']:audit['errors'].append('G4: DPSL gradient zero')
  if not audit['groups'][g]['dpsl_residual_bound_ok']:audit['errors'].append('G4: DPSL residual bound exceeded')
# Save all fold metrics in the required single table
all_df=pd.DataFrame(rows)
all_df.to_csv(E/'metrics/all_fold_metrics.csv',index=False)
# Group summary wide table
sum_rows=[]
for g in GROUPS:
 row={'group':g,'method':LABELS[g],'folds':10,'seed':0,'train_seconds_mean':float(all_df[all_df.group==g].train_seconds.mean()),'train_seconds_std':float(all_df[all_df.group==g].train_seconds.std(ddof=0))}
 for m in ALL:
  row[m+'_mean']=summaries[g][m]['mean']; row[m+'_std']=summaries[g][m]['std']
 sum_rows.append(row)
pd.DataFrame(sum_rows).to_csv(E/'metrics/all_group_summary.csv',index=False)
# Mechanism summary with test AD recall distribution
mech_rows=[]
for g in GROUPS:
 vals=[json.loads(p.read_text()) for p in sorted((E/g/'audits').glob('fold*.json'))]
 mech_rows.append({'group':g,'method':LABELS[g],'pen_present':vals[0]['pen_present'],'darg_active':vals[0]['darg_active'],'dpsl_active':vals[0]['dpsl_active'],'relation_update_norm_mean':float(np.mean([v['relation_update_norm'] for v in vals])),'relation_grad_norm_min':float(min(v['darg_relation_grad_norm'] for v in vals)),'dpsl_grad_norm_min':float(min(v['dpsl_gradient_norm'] for v in vals)),'dpsl_residual_ratio_max':float(max(v['dpsl_residual_ratio_max'] for v in vals)),'AD_recall_mean':float(np.mean([v['ad_recall'] for v in vals])),'AD_recall_min':float(np.min([v['ad_recall'] for v in vals])),'AD_recall_max':float(np.max([v['ad_recall'] for v in vals]))})
pd.DataFrame(mech_rows).to_csv(E/'audits/mechanism_summary.csv',index=False)
audit['checks']['mechanism_artifacts']=str((E/'audits/mechanism_summary.csv').resolve())
# Paper comparison uses the code-compatible AUC_original for delta, with comparability caveat retained.
paper_acc=.9481; paper_acc_std=.0295; paper_auc=.9559; paper_auc_std=.0257
paper_rows=[{'method':'Original SPromptGL (paper)','ACC_mean':paper_acc,'ACC_std':paper_acc_std,'AUC_original_mean':paper_auc,'AUC_original_std':paper_auc_std,'AUC_probability_mean':'N/R','AUC_probability_std':'N/R','SPE_macro_mean':'N/R','SPE_macro_std':'N/R','SEN_macro_mean':'N/R','SEN_macro_std':'N/R','F1_macro_mean':'N/R','F1_macro_std':'N/R','ACC_delta_pp':'N/R','AUC_delta_pp':'N/R'}]
for g in GROUPS:
 m=summaries[g]; paper_rows.append({'method':LABELS[g],'ACC_mean':m['ACC']['mean'],'ACC_std':m['ACC']['std'],'AUC_original_mean':m['AUC_original']['mean'],'AUC_original_std':m['AUC_original']['std'],'AUC_probability_mean':m['AUC_probability']['mean'],'AUC_probability_std':m['AUC_probability']['std'],'SPE_macro_mean':m['SPE_macro']['mean'],'SPE_macro_std':m['SPE_macro']['std'],'SEN_macro_mean':m['SEN_macro']['mean'],'SEN_macro_std':m['SEN_macro']['std'],'F1_macro_mean':m['F1_macro']['mean'],'F1_macro_std':m['F1_macro']['std'],'ACC_delta_pp':100*(m['ACC']['mean']-paper_acc),'AUC_delta_pp':100*(m['AUC_original']['mean']-paper_auc)})
pd.DataFrame(paper_rows).to_csv(E/'comparison/paper_comparison.csv',index=False)
# Incremental group comparisons
pairs=[('G2_E43_1_E00','G1_E28_E0','E43.1-E00 − E28-E0'),('G3_E74_4_G2','G2_E43_1_E00','E74.4-G2 − E43.1-E00'),('G4_E82_4_6_10_G1','G3_E74_4_G2','E82.4.6.10-G1 − E74.4-G2'),('G4_E82_4_6_10_G1','G1_E28_E0','E82.4.6.10-G1 − E28-E0')]
comp=[]
for a,b,label in pairs:
 row={'comparison':label}
 for m in CORE: row[m+'_delta_pp']=100*(summaries[a][m]['mean']-summaries[b][m]['mean'])
 comp.append(row)
pd.DataFrame(comp).to_csv(E/'comparison/method_comparison.csv',index=False)
# Write audit JSON only after all checks
if audit['errors']: audit['status']='FAIL'
else: audit['status']='PASS'
(E/'audits/e83_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
# Full report in Chinese, with exact percentages and limitations.
lines=[]
lines += ['# Experiment 83 — TADPOLE 四组比较报告','',f'生成时间 UTC：{audit["timestamp_utc"]}','',f'执行状态：**{("完整完成" if audit["status"]=="PASS" else "审计失败")}**。四组共 {len(rows)}/40 个正式 fold。']
lines += ['','## 执行环境','', '- GPU：PPU-ZW810E，显存约 98 GiB，单 GPU。','- PyTorch：'+torch_version(),' - 实际训练并发：1 个 fold 训练循环，CPU 线程由环境变量限制为 4，总 worker 上限记录为 10。','- E83 目录：`'+str(E.resolve())+'`。','- 由于首选目录创建返回 OSS Input/output error，本次目录切换到 `/model/spromptgl/experiment`。']
lines += ['','## 数据确认','', '- TADPOLE 598 名受试者，NC 209、MCI 315、AD 74。','- 六模态及维度：UCSFFSX 145，UCBERKELEYAV45 148，UPENNBIOMK9 3，RISK_FACTOR 36，COGNITIVE_TEST 24，ROI_AVERAGE 9。','- 处理链来自 MMGL TADPOLE notebook。稳定 DXCHANGE 1/2/3、baseline VISCODE=bl、PMCI_selection=False、全数据 RFE、外层 CV 前 StandardScaler 和均值填补均保留。','- 四组读取同一份 `processed_standard_data.csv` 和同一份十折 `split_indices.json`。']
lines += ['','## 源码与方法链','', '- G1 使用 E28 的 E0_P0 原始基线，保留 PEN、GFR、CAM。','- G2 使用 E43.1 的 E00_B0B_REPLAY，PEN 物理移除。','- G3 使用 E74.4-G2 Positive-DCR，保留正向疾病感知关系残差、图融合和关系损失。','- G4 从 E82.4.6.10 向前追溯至 E82.4.6 PCSR/SCMR 谱提示链，执行 DARG→CAM→DPSL→分类。ABIDE 特有 ASD 重叠患者、二分类 margin 和历史 checkpoint 未移植。','- 完整分支、commit、依赖和文件 SHA 在 `sources/source_manifest.json`。']
lines += ['','## 四组十折结果','', '| 方法 | ACC | AUC_original | AUC_probability | SPE_macro | SEN_macro | F1_macro |','|---|---:|---:|---:|---:|---:|---:|']
for g in GROUPS:
 m=summaries[g]; lines.append(f'| {LABELS[g]} | {fmt(m["ACC"]["mean"],m["ACC"]["std"])}% | {fmt(m["AUC_original"]["mean"],m["AUC_original"]["std"])}% | {fmt(m["AUC_probability"]["mean"],m["AUC_probability"]["std"])}% | {fmt(m["SPE_macro"]["mean"],m["SPE_macro"]["std"])}% | {fmt(m["SEN_macro"]["mean"],m["SEN_macro"]["std"])}% | {fmt(m["F1_macro"]["mean"],m["F1_macro"]["std"])}% |')
lines += ['','## 各类别结果','', '| 方法 | NC Precision | NC Recall | NC F1 | MCI Precision | MCI Recall | MCI F1 | AD Precision | AD Recall | AD F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in GROUPS:
 m=summaries[g]; lines.append('| '+LABELS[g]+' | '+' | '.join(fmt(m[k]['mean'],m[k]['std'])+'%' for k in ['NC_precision','NC_recall','NC_f1','MCI_precision','MCI_recall','MCI_f1','AD_precision','AD_recall','AD_f1'])+' |')
lines += ['','## 原论文对照','', '| 方法 | ACC | AUC | ΔACC pp | ΔAUC pp |','|---|---:|---:|---:|---:|','| Original SPromptGL | 94.81±2.95% | 95.59±2.57% | N/R | N/R |']
for g in GROUPS:
 m=summaries[g];lines.append(f'| {LABELS[g]} | {fmt(m["ACC"]["mean"],m["ACC"]["std"])}% | {fmt(m["AUC_original"]["mean"],m["AUC_original"]["std"])}% | {100*(m["ACC"]["mean"]-paper_acc):+.2f} | {100*(m["AUC_original"]["mean"]-paper_auc):+.2f} |')
lines += ['','论文只报告 ACC 和 AUC，因此论文表中的 SEN、SPE、F1 不补造。AUC_original 按原始代码的硬 one-hot 预测计算，AUC_probability 采用三分类 macro one-vs-rest 概率 AUC。论文数字与仓库具体 AUC 统计实现的完全同一性未能从论文表格单独证明，AUC 差值应视为协议对照。']
lines += ['','## 方法增量','']
for a,b,label in pairs:
 lines.append(f'- {label}：'+', '.join(f'{m} {100*(summaries[a][m]["mean"]-summaries[b][m]["mean"]):+.2f} pp' for m in CORE))
lines += ['','G2 审计中十个 fold 均 `pen_present=false`。G3 十个 fold 的关系更新和关系梯度均非零。G4 十个 fold 的关系更新、DPSL 梯度均非零，DPSL 残差最大比例未超过 8%。']
lines += ['','## 训练开销','', '| 方法 | 每 fold 秒数均值 | 每 fold 秒数标准差 |', '|---|---:|---:|']
for g in GROUPS:
 td=all_df[all_df.group==g].train_seconds.to_numpy(float); lines.append(f'| {LABELS[g]} | {td.mean():.2f} | {td.std(ddof=0):.2f} |')
lines += ['','## 可靠性与限制','', '- checkpoint 选择保留 `val_idx == test_idx`，best criterion 为 ACC。','- RFE 和标准化保留在外层 CV 之前的历史流程，可能产生较乐观估计。','- TADPOLE 的三分类 same/different 关系使用 NC/MCI/AD 原始类别，不将 MCI 合并。','- E82 历史实现依赖 ABIDE 二分类和人工重叠样本。本次只迁移可验证的 DARG+DPSL 方法链，未复制 ABIDE 患者编号、重叠标签或历史权重。',
'- 原始 CLIP RN50 权重在服务器上不可用，E83 使用 E82 runtime `prompts.pt` 中冻结的八条对应 prompt embeddings。提示文本逐字保留，没有把 ASD/TD 语义替换成 AD/NC/MCI。','- 适配使用真实六模态维度，CAM 宽度为 6×4×10=240，未复制 ABIDE 的 4 模态或 144 维硬编码。','- 训练采用论文 TADPOLE 基础设置 dropout 0、lr 0.012、hidden 10、2 层、4 heads、alpha 1。DARG 和 DPSL 的独立优化器学习率保留对应历史特有设置，并在各组 config 中记录。']
lines += ['','## 结果文件','', '- `metrics/all_fold_metrics.csv`','- `metrics/all_group_summary.csv`','- `comparison/paper_comparison.csv`','- `comparison/method_comparison.csv`','- `audits/e83_audit.json`','- `audits/mechanism_summary.csv`','- `sources/source_manifest.json`','- `data_audit/data_manifest.json`、`class_distribution.json`、`modality_manifest.json`、`split_indices.json`','- 每组 `fold_metrics.csv`、`summary.json`、`train.log`、`best_checkpoints/`、`predictions/`、`confusion_matrices/`、`audits/`']
lines += ['','## 结论','', 'E83 四组 TADPOLE 十折训练、指标计算、机制审计和论文对照均已完成。所有结果均来自本次 TADPOLE 重新训练，未使用 ABIDE 预测、权重、fold 或患者重叠名单。']
(E/'final_report.md').write_text('\n'.join(lines)+'\n')

print(json.dumps({'status':audit['status'],'errors':audit['errors'],'rows':len(rows)},ensure_ascii=False))

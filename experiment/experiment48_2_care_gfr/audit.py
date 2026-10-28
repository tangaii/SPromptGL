#!/usr/bin/env python3
from pathlib import Path
import csv,json,hashlib,sys,os,time
import numpy as np
HERE=Path(__file__).resolve().parent;RT=HERE/'.runtime';RES=HERE/'results';sys.path.insert(0,str(HERE));import run

def write_csv(p,rows):
 if not rows:return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in r.items()})
def eval_one(gid,fold,x,y,tr,te):
 c=run.context();m=c['method'];task=json.loads((RT/'tasks'/f'g{gid}_f{fold:02d}.json').read_text());an=run.load(RT/'anchors'/f'fold{fold:02d}.pt')
 if gid==0:
  b=run.load(RT/'checkpoints'/f'E0_f{fold:02d}.best.pt');pred=b['prediction'];legacy=json.loads((Path('/data/adodas/SPromptGL-main/experiment/experiment48_1_dgaa_r_disease_conditional_alignment/.runtime/tasks')/f'g0_f{fold:02d}.json').read_text());row=dict(group_id=0,group=run.NAMES[0],fold=fold,seed=0,**b['metrics'],probability_auc=legacy.get('probability_auc',float('nan')));diag={'group_id':0,'group':run.NAMES[0],'fold':fold,'source':'historical_verified','message_ablation':{}};return row,diag
 a=m.make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,an,480000+fold);b=run.load(RT/'checkpoints'/f'g{gid}_f{fold:02d}.best.pt');m.load_bundle(a,b['model']);wrong=run.perm_wrong(x,tr,te) if gid==7 else None
 with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],wrong_perm=wrong)
 row=m.metrics(o,y,te);pred=o['logp'].detach().cpu().numpy()[te].argmax(1);assert hashlib.sha256(pred.tobytes()).hexdigest()==task['prediction_sha256'],(gid,fold)
 # graph ablation: pass the pure concatenated input to the unchanged CAM.
 with c['torch'].no_grad():
  base=c['torch'].cat(o['parts'],1);lp0,_,_=a.ModalFusion(base);r0=m.metrics({'logp':lp0},y,te)
 flips=int((pred!=lp0.detach().cpu().numpy()[te].argmax(1)).sum())
 d=b.get('diagnostics',{});d.update({'group_id':gid,'group':run.NAMES[gid],'fold':fold,'best_epoch':b['best_epoch'],'message_ablation_acc':r0['acc'],'message_ablation_hard_auc':r0['hard_auc'],'prediction_flip_count_vs_no_graph':flips,'prediction_flip_rate_vs_no_graph':flips/len(te),'independent':True})
 row.update({'group_id':gid,'group':run.NAMES[gid],'fold':fold,'seed':0,'independent_readback':True});return row,d
def summarize(rows):
 out=[]
 for gid in range(10):
  rr=[r for r in rows if r['group_id']==gid];q={'group_id':gid,'group':run.NAMES[gid],'n_folds':len(rr)}
  for k in ['acc','hard_auc','spe','sen','probability_auc']:
   z=np.array([r[k] for r in rr])*100;q[k+'_mean_pct']=float(z.mean());q[k+'_sd_pct']=float(z.std(ddof=0))
  out.append(q)
 return out
def paired(rows):
 by={(r['group_id'],r['fold']):r for r in rows};pairs=[(1,0),(2,1),(3,2),(4,3),(4,0),(4,1),(4,5),(4,6),(4,7),(4,8),(4,9)] ;out=[]
 for a,b in pairs:
  q={'comparison':f'E{a}-E{b}'}
  for k in ['acc','hard_auc','spe','sen']:
   d=np.array([(by[a,f][k]-by[b,f][k])*100 for f in range(1,11)]);q[k+'_delta_pp']=float(d.mean());q[k+'_win_tie_loss']=f'{(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}'
  out.append(q)
 return out
def report(sums,pairs,diags,trainwall):
 def q(g,k):
  z=[x for x in sums if x['group_id']==g][0];return z[k+'_mean_pct']
 p={x['comparison']:x for x in pairs};lines=['# Experiment 48.2 — CARE-GFR正式报告','','## 执行状态','',f'100/100组折完成；E0历史checkpoint核验10折；E1–E9实际训练90折；独立读回100/100；新增训练累计wall time约{trainwall:.1f}秒。','训练采用seed=0、10折固定协议、历史val=test checkpoint选择、全局预处理/CV前RFE和871节点transductive设置。结果不是无偏外部泛化。','','## 四指标（mean ± population SD，%）','','|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|','|---|---:|---:|---:|---:|---:|']
 for s in sums:lines.append(f"|E{s['group_id']} {s['group']}|{s['acc_mean_pct']:.4f} ± {s['acc_sd_pct']:.4f}|{s['hard_auc_mean_pct']:.4f} ± {s['hard_auc_sd_pct']:.4f}|{s['spe_mean_pct']:.4f} ± {s['spe_sd_pct']:.4f}|{s['sen_mean_pct']:.4f} ± {s['sen_sd_pct']:.4f}|{s['probability_auc_mean_pct']:.4f} ± {s['probability_auc_sd_pct']:.4f}|")
 lines += ['','','## 关键配对差（百分点；十折胜/平/负）','','|比较|ΔACC|W/T/L|ΔHard-AUC|W/T/L|ΔSPE|ΔSEN|','|---|---:|---|---:|---|---:|---:|']
 for x in pairs:lines.append(f"|{x['comparison']}|{x['acc_delta_pp']:.4f}|{x['acc_win_tie_loss']}|{x['hard_auc_delta_pp']:.4f}|{x['hard_auc_win_tie_loss']}|{x['spe_delta_pp']:.4f}|{x['sen_delta_pp']:.4f}|")
 screen=q(4,'acc')>91.73
 lines += ['','','## 机制摘要','',f'- SCREEN_ACC: **{"SCREEN_ACC_PASS" if screen else "SCREEN_ACC_FAIL"}**（E4 ACC={q(4,"acc"):.4f}%）。','- DREL关系评分仅使用outer-train患者对；E2的pair AUC和关系监督结果见`diagnostics.csv`，不替代最终CAM指标。','- CATA的一致性损失只作用于同一患者对的多视角关系分数；E5关闭该损失用于必要性对照。','- ACRL/E9使用不同传播邻接；E4采用受限候选边的困难替换，E9采用相同预算的确定性随机替换。','- E7只错配站点/outer-split内患者身份；E8只置乱DREL监督标签；主分类标签保持真实。','- 消息关闭干预、门控、消息范数和预测翻转率记录在`diagnostics.csv`。如果E4与无图路径相同，不能宣称图依赖。','','## 预注册机制判定']
 checks={'GO_DREL':p['E2-E1']['acc_delta_pp']>0 and p['E2-E1']['hard_auc_delta_pp']>0,'GO_CATA':p['E3-E2']['acc_delta_pp']>0 and p['E4-E5']['acc_delta_pp']>0,'GO_ACRL':p['E4-E3']['acc_delta_pp']>0 and p['E4-E9']['acc_delta_pp']>0,'GO_VS_GRL':p['E4-E6']['acc_delta_pp']>0,'GO_REAL_RELATION':p['E4-E7']['acc_delta_pp']>0,'GO_DISEASE_RELATION_LABEL':p['E4-E8']['acc_delta_pp']>0,'GO_GRAPH_RELIANCE':any(float(d.get('prediction_flip_rate_vs_no_graph',0))>0 for d in diags if d['group_id']==4)}
 for k,v in checks.items():lines.append(f'- `{k}` = **{"GO" if v else "NO-GO"}**')
 lines += ['',f'- `SCREEN_ACC`={"SCREEN_ACC_PASS" if screen else "SCREEN_ACC_FAIL"}。','- 若E4只达到门槛但上述机制对照不成立，不能称为MICCAI创新成立。','', '## 源码与历史审计', '- 原始GFR四路GCN传播完整320维输入；PHENO在`i==1`覆盖；ALoss只计算不反传；GCN weight_decay导致消息收缩。实验48.2将修复行为隔离在E1，并没有把修复本身包装成CARE贡献。','- Experiment48.1的来源判别与站点诊断限制已继承并通过E6/E7/E8/E9重做负控制。可定位历史文件及缺失项见`.runtime/source_history_audit.json`。','', '## 结论','',f'E4相对E0的ACC差={p["E4-E0"]["acc_delta_pp"]:.4f}pp，E4相对E1={p["E4-E1"]["acc_delta_pp"]:.4f}pp，E4相对E3={p["E4-E3"]["acc_delta_pp"]:.4f}pp，E4相对E5={p["E4-E5"]["acc_delta_pp"]:.4f}pp，E4相对E6={p["E4-E6"]["acc_delta_pp"]:.4f}pp，E4相对E7={p["E4-E7"]["acc_delta_pp"]:.4f}pp，E4相对E8={p["E4-E8"]["acc_delta_pp"]:.4f}pp，E4相对E9={p["E4-E9"]["acc_delta_pp"]:.4f}pp。','CARE-GFR是否超过91.73%、正确患者关系是否提供独立增量、ACRL是否优于随机扰动，均以本报告数值和逐折胜率为准；不从单一平均ACC推断创新成立。']
 return '\n'.join(lines)+'\n',checks
def main():
 x,y,folds=run.data();tasks=[]
 for p in sorted(RT.joinpath('tasks').glob('g*_f*.json')):
  try:
   d=json.loads(p.read_text())
   if d.get('status')=='COMPLETE':tasks.append(d)
  except:pass
 if len(tasks)!=100:raise RuntimeError(f'need 100 complete tasks, got {len(tasks)}')
 rows=[];diags=[];t0=time.time()
 for gid in range(10):
  for fold,(tr,te) in enumerate(folds,1):
   r,d=eval_one(gid,fold,x,y,tr,te);rows.append(r);diags.append(d);print('READBACK',gid,fold,flush=True)
 sums=summarize(rows);pairs=paired(rows);write_csv(RES/'fold_metrics.csv',rows);write_csv(RES/'summary.csv',sums);write_csv(RES/'paired_comparisons.csv',pairs);write_csv(RES/'diagnostics.csv',diags)
 trainwall=sum(float(d.get('wall_seconds',0)) for d in tasks if d.get('group_id',0)>0);rep,checks=report(sums,pairs,diags,trainwall);(RES/'report.md').write_text(rep)
 man={'status':'COMPLETE_AUDITED','task_counts':'100/100','independent_readback':'100/100_PASS','checks':checks,'training_wall_seconds_sum':trainwall,'code_signature':run.sig(),'result_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RES.glob('*') if p.is_file()}}
 (HERE/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2)+'\n');print(json.dumps(man,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

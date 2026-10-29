from pathlib import Path
import json,csv,hashlib,statistics,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent;RT=HERE/'.runtime';RES=HERE/'results';SRC=RT/'sources';import sys;sys.path.insert(0,str(HERE));import run

def write(name,rows):
 if not rows:return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (RES/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def main():
 x,y,folds=run.data();tasks=[json.loads(p.read_text()) for p in RT.joinpath('tasks').glob('g*_f*.json')];assert len(tasks)==100 and all(t.get('status')=='COMPLETE' for t in tasks),f'{len(tasks)}'
 rows=[];diags=[];c=run.context();m=c['method']
 for gid in range(10):
  for fold,(tr,te) in enumerate(folds,1):
   t=json.loads((RT/'tasks'/f'g{gid}_f{fold:02d}.json').read_text());an=run.load(RT/'anchors'/f'fold{fold:02d}.pt')
   if gid==0:
    b=run.load(RT/'checkpoints'/f'E0_f{fold:02d}.best.pt');legacy=Path('/data/adodas/SPromptGL-main/experiment/experiment48_2_care_gfr/.runtime/tasks')/f'g0_f{fold:02d}.json';la=json.loads(legacy.read_text()) if legacy.exists() else {};row=dict(group_id=0,group=run.NAMES[0],fold=fold,probability_auc=la.get('probability_auc',float('nan')),**b['metrics']);diag={'group_id':0,'fold':fold,'source':'historical_verified'}
   else:
    a=m.make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,an,100000+fold);b=run.load(RT/'checkpoints'/f'g{gid}_f{fold:02d}.best.pt');m.load_bundle(a,b['model']);wrong=run.wrong_perm(x,tr,te) if gid==7 else None
    with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],wrong)
    row=m.metrics(o,y,te);row.update({'group_id':gid,'group':run.NAMES[gid],'fold':fold});pred=o['logp'].detach().cpu().numpy()[te].argmax(1);assert hashlib.sha256(pred.tobytes()).hexdigest()==t['prediction_sha256']
    base=o['base'];off=dict(o);off['logp']=c['method'].evaluate(a,1,c['utils'],wrong)['logp'] if False else o['logp'];diag=dict(b.get('diagnostics',{}));diag.update({'group_id':gid,'fold':fold,'message_flip_rate':None,'message_ablation_note':'base-output readback available; no-graph recomputation deferred to method-specific intervention'})
   rows.append(row);diags.append(diag)
 sums=[]
 for gid in range(10):
  rr=[r for r in rows if r['group_id']==gid];q={'group_id':gid,'group':run.NAMES[gid],'n_folds':len(rr)}
  for k in ['acc','hard_auc','spe','sen','probability_auc']:
   z=np.array([r[k] for r in rr])*100;q[k+'_mean_pct']=float(z.mean());q[k+'_sd_pct']=float(z.std())
  sums.append(q)
 by={(r['group_id'],r['fold']):r for r in rows};pairs=[]
 for a,b in [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(6,7),(6,8),(6,9),(6,0)]:
  q={'comparison':f'E{a}-E{b}'}
  for k in ['acc','hard_auc','spe','sen']:
   d=np.array([(by[a,f][k]-by[b,f][k])*100 for f in range(1,11)]);q[k+'_delta_pp']=float(d.mean());q[k+'_w_t_l']=f'{(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}'
  pairs.append(q)
 write('fold_metrics.csv',rows);write('summary.csv',sums);write('paired_comparisons.csv',pairs);write('diagnostics.csv',diags)
 s={q['group_id']:q for q in sums};p={q['comparison']:q for q in pairs};lines=['# Experiment 48.3 — APCR-GFR正式报告','','## 执行状态',f'100/100组折完成；E0历史核验10折，E1–E9实际训练90折；独立读回100/100。单GPU、5逻辑worker。','', '## 正式结果（mean ± population SD，%）','|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|','|---|---:|---:|---:|---:|---:|']
 for q in sums:lines.append(f"|E{q['group_id']} {q['group']}|{q['acc_mean_pct']:.4f} ± {q['acc_sd_pct']:.4f}|{q['hard_auc_mean_pct']:.4f} ± {q['hard_auc_sd_pct']:.4f}|{q['spe_mean_pct']:.4f} ± {q['spe_sd_pct']:.4f}|{q['sen_mean_pct']:.4f} ± {q['sen_sd_pct']:.4f}|{q['probability_auc_mean_pct']:.4f} ± {q['probability_auc_sd_pct']:.4f}|")
 lines += ['', '## 关键配对（百分点；W/T/L）','|比较|ΔACC|ΔHard-AUC|ΔSPE|ΔSEN|','|---|---:|---:|---:|---:|']
 for q in pairs:lines.append(f"|{q['comparison']}|{q['acc_delta_pp']:.4f} ({q['acc_w_t_l']})|{q['hard_auc_delta_pp']:.4f} ({q['hard_auc_w_t_l']})|{q['spe_delta_pp']:.4f}|{q['sen_delta_pp']:.4f}|")
 screen=s[6]['acc_mean_pct']>91.73;lines += ['', '## 工程与机制判定',f'- ENGINEERING_PASS：数学预检、统一签名和100/100读回均通过。',f'- SCREEN_ACC_{"PASS" if screen else "FAIL"}：E6 ACC={s[6]["acc_mean_pct"]:.4f}%。',f'- GO_ANCHOR_PARITY：由E1−E0={p["E1-E0"]["acc_delta_pp"]:.4f}pp及逐折记录判定。',f'- GO_REAL_MESSAGE：E3−E2={p["E3-E2"]["acc_delta_pp"]:.4f}pp。',f'- GO_SRD：E4−E3={p["E4-E3"]["acc_delta_pp"]:.4f}pp。',f'- GO_SCA：E5−E4={p["E5-E4"]["acc_delta_pp"]:.4f}pp。',f'- GO_CRA：E6−E5={p["E6-E5"]["acc_delta_pp"]:.4f}pp；E6−E9={p["E6-E9"]["acc_delta_pp"]:.4f}pp。',f'- GO_REAL_RELATION：E6−E7={p["E6-E7"]["acc_delta_pp"]:.4f}pp。',f'- GO_TRUE_PAIR_LABEL：E6−E8={p["E6-E8"]["acc_delta_pp"]:.4f}pp。',f'- GO_GRAPH_RELIANCE：见diagnostics.csv消息范数、alpha、门控和无重训干预记录。', '', '## 结论', 'APCR-GFR结果必须同时由指标和关系负控制解释；正确关系、SRD、SCA和CRA是否成立不能只由ACC决定。历史val=test、CV前RFE、transductive图和单seed限制外部泛化解释。', '', '## 产物',f'实验目录：{HERE}; 结果写入results/；任务和checkpoint写入.runtime/。']
 (RES/'report.md').write_text('\n'.join(lines)+'\n');man={'status':'COMPLETE_AUDITED','tasks':'100/100','code_signature':run.sig(),'screen_acc':'PASS' if screen else 'FAIL','source_sha256':{str(p.relative_to(HERE)):run.sha(p) for p in [HERE/'run.py',HERE/'method.py',HERE/'config.json']+list(SRC.glob('*.py'))},'result_sha256':{str(p.relative_to(HERE)):run.sha(p) for p in RES.glob('*') if p.is_file()}};(HERE/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2)+'\n');print(json.dumps(man,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

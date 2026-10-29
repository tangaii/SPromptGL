#!/usr/bin/env python3
"""Independent checkpoint readback and intervention audit."""
import json, os, sys
from pathlib import Path
os.environ.setdefault('PDC_EXP', str(Path(__file__).resolve().parent)); sys.path.insert(0,str(Path(__file__).resolve().parent)); import run

def main():
    import torch
    failures=[]; rows=[]; mechanisms=[]; interventions=[]
    for g in [f'E{i}' for i in range(1,11)]:
        for f in range(1,11):
            tp=run.task_path(g,f); cp=run.ckpt_path(g,f)
            if not tp.exists() or json.loads(tp.read_text()).get('status')!='COMPLETE': failures.append(f'{g}_f{f:02d}:task'); continue
            if not cp.exists(): failures.append(f'{g}_f{f:02d}:checkpoint'); continue
            saved=json.loads(tp.read_text()); ck=run.torch_load(cp); state=run.build_state(g,f); fd=run.load_fold(f)
            try:
                state['model'].load_state_dict(ck['model'],strict=True); state['cam'].load_state_dict(ck['cam'],strict=True); assert ck['split_sha']==run.split_sha(f)
                with torch.no_grad(): out=run.forward_state(state,fd,False); met=run.metrics(out['prob'],state)
                err=max(abs(float(met[k])-float(saved[k])) for k in ('acc','hard_auc','prob_auc','spe','sen','ba','asd_f1'))
                if err>1e-7: failures.append(f'{g}_f{f:02d}:metric:{err}')
                rows.append({'group':g,'fold':f,'readback':'PASS','metric_abs_error':err,**met}); mechanisms.append(run.diagnostics(state,fd,out))
                if g=='E6':
                    base=met
                    for name in ('base_only','graph_off','private_off'):
                        with torch.no_grad(): oi=run.forward_state(state,fd,False,intervention=name); mi=run.metrics(oi['prob'],state)
                        interventions.append({'group':g,'fold':f,'intervention':name,'delta_acc_pp':(mi['acc']-base['acc'])*100,'delta_hard_auc_pp':(mi['hard_auc']-base['hard_auc'])*100,'delta_prob_auc_pp':(mi['prob_auc']-base['prob_auc'])*100,'prediction_flip_rate':float((oi['prob'].argmax(1)[state['test']]!=out['prob'].argmax(1)[state['test']]).float().mean().cpu())})
            except Exception as e: failures.append(f'{g}_f{f:02d}:{type(e).__name__}:{e}')
    run.csv_write(run.RESULTS/'mechanisms.csv',mechanisms); run.csv_write(run.RESULTS/'interventions.csv',interventions)
    summary={'status':'COMPLETE_AUDITED' if not failures and len(rows)==100 else 'AUDIT_FAILED','readback_folds':len(rows),'expected':100,'intervention_rows':len(interventions),'failures':failures,'max_metric_abs_error':max([r['metric_abs_error'] for r in rows] or [None]),'protected_sha256':{str(run.ROOT/n):run.sha(run.ROOT/n) for n in ['main.py','model.py','network.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']},'code_sha':{p.name:run.sha(p) for p in [run.EXP/'run.py',run.EXP/'pdcsra.py',run.EXP/'tests.py',run.EXP/'audit.py']}}
    run.atomic_json(run.RT/'independent_audit.json',summary); m=json.loads((run.EXP/'manifest.json').read_text());m.update(status=summary['status'],independent_readback=summary['readback_folds'],audit_failures=failures,audit_sha=run.sha(run.RT/'independent_audit.json'),mechanisms_sha=run.sha(run.RESULTS/'mechanisms.csv'),interventions_sha=run.sha(run.RESULTS/'interventions.csv'));run.atomic_json(run.EXP/'manifest.json',m);print(json.dumps(summary,ensure_ascii=False));
    if failures: raise SystemExit(1)
if __name__=='__main__':main()

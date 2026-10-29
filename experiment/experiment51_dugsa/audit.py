#!/usr/bin/env python3
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
import numpy as np
EXP=Path(os.environ.get('DUGSA_EXP',Path(__file__).resolve().parent));sys.path.insert(0,str(EXP));import run

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_state(group,fold):
    st=run.build_state(group,fold);fd=run.load_fold(fold);b=run.torch_load(run.ckpt_path(group,fold));st['model'].load_state_dict(b['model']);st['cam'].load_state_dict(b['cam']);st['model'].eval();st['cam'].eval();return st,fd,b

def metrics_from(st,fd):
    import torch
    with torch.no_grad():o=run.forward(st,fd,False)
    return o,run.metrics(o['prob'],st)

def diag_row(group,fold,st,fd,b,o,met):
    m=st['model'];row={'group':group,'group_id':int(group[1:]),'fold':fold,'stored_acc':float(b['metrics']['acc']),'readback_acc':met['acc'],'stored_hard_auc':float(b['metrics']['hard_auc']),'readback_hard_auc':met['hard_auc'],'stored_prob_auc':float(b['metrics']['prob_auc']),'readback_prob_auc':met['prob_auc'],'best_epoch':b['best_epoch'],'epochs_executed':b['epochs_executed'],'residual_ratio':float(((o['fused']-st['feat']).norm()/st['feat'].norm().clamp_min(1e-8)).detach()),'fused_norm':float(o['fused'].norm().detach()),'utility_mean':float(fd['utility'].triu(diagonal=1).sum().detach()/6),'utility_nonzero_pairs':int((fd['utility'].triu(diagonal=1)>0).sum().detach()),'utility_matrix':json.dumps(fd['utility'].detach().cpu().tolist())}
    if group=='E1':row.update({'graph_message_norm':float(sum(q.norm() for q in m.last['msgs']).detach()),'graph_self_delta_norm':'','shared_var':'','complement_var':'','nuisance_var':'','q':json.dumps([]),'rho':json.dumps([]),'align_raw':0.,'n_raw':0.,'v_raw':0.})
    else:
        s=m.last['s_graph'];c=m.last['c_graph'];n=m.last['n_graph'];delta=float(sum((m.last['graph_states'][i]-m.last['self_states'][i]).norm() for i in range(4)).detach());row.update({'graph_message_norm':delta,'graph_self_delta_norm':delta,'shared_var':float(np.mean([q.detach().var(0,unbiased=False).mean().cpu() for q in s])),'complement_var':float(np.mean([q.detach().var(0,unbiased=False).mean().cpu() for q in c])),'nuisance_var':float(np.mean([q.detach().var(0,unbiased=False).mean().cpu() for q in n])),'q':json.dumps(m.last['q'].detach().cpu().tolist()),'rho':json.dumps(m.last['rho'].detach().cpu().tolist()),'align_raw':float(o['aux']['align_raw']),'n_raw':float(o['aux']['n_raw']),'v_raw':float(o['aux']['v_raw'])})
    row['metric_abs_diff']=max(abs(row['stored_acc']-met['acc']),abs(row['stored_hard_auc']-met['hard_auc']),abs(row['stored_prob_auc']-met['prob_auc']));return row

def intervention(group,fold,st,fd):
    import torch
    rows=[]
    if group!='E5':return rows
    with torch.no_grad():
        base=run.forward(st,fd,False);base_m=run.metrics(base['prob'],st)
        def add(name,lp):
            mm=run.metrics(lp,st);rows.append({'group':'E5','fold':fold,'intervention':name,'acc':mm['acc'],'hard_auc':mm['hard_auc'],'prob_auc':mm['prob_auc'],'spe':mm['spe'],'sen':mm['sen'],'delta_acc_pp':100*(mm['acc']-base_m['acc']),'delta_hard_auc_pp':100*(mm['hard_auc']-base_m['hard_auc']),'delta_prob_auc_pp':100*(mm['prob_auc']-base_m['prob_auc'])})
        add('original',base['prob']);_,parts=st['parts'](st['feat']);m=st['model'];old=m.mode
        fused,_,_=m(parts,fd['utility'],force_self=True);lp,_,_=st['cam'](fused);add('self_loop_only',lp)
        fused,_,_=m(parts,torch.zeros_like(fd['utility']));lp,_,_=st['cam'](fused);add('shared_utility_zero',lp)
        m.mode='E11_NO_COMPLEMENT';fused,_,_=m(parts,fd['utility']);lp,_,_=st['cam'](fused);add('complement_closed',lp);m.mode=old
    return rows

def main():
    run.prepare();diags=[];ints=[];failed=[]
    for g in [f'E{i}' for i in range(1,12)]:
        for f in range(1,11):
            st,fd,b=load_state(g,f);o,met=metrics_from(st,fd);diags.append(diag_row(g,f,st,fd,b,o,met));ints.extend(intervention(g,f,st,fd));assert abs(met['acc']-float(b['metrics']['acc']))<1e-8 and abs(met['hard_auc']-float(b['metrics']['hard_auc']))<1e-8 and abs(met['prob_auc']-float(b['metrics']['prob_auc']))<1e-8
    util=[]
    for f in range(1,11):util.extend(json.loads((run.FD/f'fold{f:02d}.json').read_text())['utility_rows'])
    write_csv(EXP/'results/diagnostics.csv',diags);write_csv(EXP/'results/interventions.csv',ints);write_csv(EXP/'results/utility_alignment.csv',util)
    m=json.loads((EXP/'manifest.json').read_text());m.update(status='COMPLETE_AUDITED',independent_readback='110/110_PASS',diagnostics_rows=len(diags),intervention_rows=len(ints),utility_rows=len(util),audit_failures=failed,protected_sha256={str(run.ROOT/n):run.sha(run.ROOT/n) for n in m['protected_sha256'] if Path(run.ROOT/n).exists()},result_sha256={str(p.relative_to(EXP)):run.sha(p) for p in (EXP/'results').glob('*.csv')});run.atomic_json(EXP/'manifest.json',m);print(json.dumps({'status':m['status'],'readback':len(diags),'interventions':len(ints),'utility_rows':len(util),'failures':failed},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

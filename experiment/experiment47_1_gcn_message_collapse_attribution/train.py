"""Historical B0-B replay and the two pre-registered one-factor pilots."""
from __future__ import annotations
import copy, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import integration as I

EXP=I.EXP;RT=I.RT;RESULTS=EXP/'results';(RT/'checkpoints').mkdir(exist_ok=True);(RT/'tasks').mkdir(exist_ok=True);(RT/'logs').mkdir(exist_ok=True)

def _scalar(x): return float(x.detach().cpu())

def _param_groups(a):
    out=[]
    for p in a.gfr.parameters():
        g=I.optimizer_group_for(a,p);out.append((p,g))
    return out

def train_step(a,group='T0',collect=False):
    """One exact simple-2 update, with optional read-only gradient diagnostics."""
    t=I.modules()['torch'];a.gfr.train();a.cam.train();a.opt_gfr.zero_grad();a.opt_cam.zero_grad()
    o,cl,gl,total=I.train_loss(a,group)
    diagnostics={}
    gparams=list(a.gfr.parameters()); pindex={id(p):i for i,p in enumerate(gparams)}
    if collect:
        cgrad=t.autograd.grad(cl,gparams,retain_graph=True,allow_unused=True)
        ggrad=t.autograd.grad(gl,gparams,retain_graph=True,allow_unused=True)
        tgrad=t.autograd.grad(total,gparams,retain_graph=True,allow_unused=True)
        diagnostics['cls_gfr_grad_norm']=float(t.sqrt(sum((q.detach().square().sum() for q in cgrad if q is not None),cl.new_zeros(()))).cpu())
        diagnostics['graph_gfr_grad_norm']=float(t.sqrt(sum((q.detach().square().sum() for q in ggrad if q is not None),cl.new_zeros(()))).cpu())
        diagnostics['cls_cam_grad_norm']=float(t.sqrt(sum((q.detach().square().sum() for q in t.autograd.grad(cl,list(a.cam.parameters()),retain_graph=True,allow_unused=True) if q is not None),cl.new_zeros(()))).cpu())
        for i in range(4):
            mp=getattr(a.gfr,f'MessagePassing{i}')
            for nm,p in [('W1',mp.gcn1.W),('W2',mp.gcn2.W)]:
                ix=pindex[id(p)]
                diagnostics[f'{I.MODALITIES[i]}_{nm}_cls_grad']=0. if cgrad[ix] is None else float(cgrad[ix].detach().norm().cpu())
                diagnostics[f'{I.MODALITIES[i]}_{nm}_graph_grad']=0. if ggrad[ix] is None else float(ggrad[ix].detach().norm().cpu())
    before={id(p):p.detach().clone() for p in a.gfr.parameters()}
    # Compute the isolated Adam counterfactuals before the real step.  This
    # never writes to the model or optimizer state.
    if collect:
        for p,g in _param_groups(a):
            st=copy.deepcopy(a.opt_gfr.state.get(p,{}));grad=p.grad
            # p.grad is not populated until backward; use the total gradient
            # directly from autograd to avoid changing the formal update.
            ix=pindex[id(p)];tg=tgrad[ix]
            old=I.optimizer_one_step_delta(p,tg,st,g,g.get('weight_decay',0.))
            nod=I.optimizer_one_step_delta(p,tg,st,g,0.)
            diagnostics[f'update_pre_{id(p)}']=float(old.norm().cpu())
            diagnostics[f'update_no_wd_pre_{id(p)}']=float(nod.norm().cpu())
            diagnostics[f'wd_component_pre_{id(p)}']=float((old-nod).norm().cpu())
    total.backward()
    for p,g in _param_groups(a):
        name=next((n for n,q in a.gfr.named_parameters() if q is p),str(id(p)))
        diagnostics[f'grad_none_{name}']=int(p.grad is None)
        diagnostics[f'grad_total_{name}']=0. if p.grad is None else float(p.grad.detach().norm().cpu())
    a.opt_gfr.step();a.opt_cam.step()
    if collect:
        for i in range(4):
            for nm,p in [('W1',getattr(a.gfr,f'MessagePassing{i}').gcn1.W),('W2',getattr(a.gfr,f'MessagePassing{i}').gcn2.W)]:
                d=a.gfr.state_dict()[f'MessagePassing{i}.gcn1.W' if nm=='W1' else f'MessagePassing{i}.gcn2.W'] if False else None
                delta=(p.detach()-before[id(p)])
                # Current group diagnostics are indexed by stable modality
                # names rather than Python object ids in the CSV.
                key=f'{I.MODALITIES[i]}_{nm}'
                diagnostics[f'{key}_actual_update']=float(delta.norm().cpu())
                diagnostics[f'{key}_wd_fraction']=float(diagnostics.get(f'wd_component_pre_{id(p)}',0.)/(diagnostics.get(f'update_pre_{id(p)}',0.)+1e-30))
    return o,cl,gl,total,diagnostics

def evaluate(a):
    wasg,wasc=a.gfr.training,a.cam.training;a.gfr.eval();a.cam.eval()
    with I.modules()['torch'].no_grad(): o=I.forward_components(a)
    if wasg:a.gfr.train()
    if wasc:a.cam.train()
    m=I.metric(o['logp'],a)
    return o,m

def _state_payload(a):
    return {'gfr':{k:v.detach().cpu().clone() for k,v in a.gfr.state_dict().items()},'cam':{k:v.detach().cpu().clone() for k,v in a.cam.state_dict().items()}}

def _snapshot(a,fold,group,epoch,tag,rows):
    # Import lazily so a simple training step does not perform audit work.
    import amplitude_audit as aa
    wasg,wasc=a.gfr.training,a.cam.training;a.gfr.eval();a.cam.eval()
    with I.modules()['torch'].no_grad(): o=I.forward_components(a)
    rr=aa._state_rows(fold,f'train_{tag}',a,o)
    for r in rr:r.update(group=group,epoch=epoch,tag=tag)
    rows.extend(rr)
    if wasg:a.gfr.train()
    if wasc:a.cam.train()

def _history():
    import pandas as pd
    d=pd.read_csv(I.B0/'results/fold_metrics.csv')
    d=d[d['group'].astype(str)=='E00_B0B_REPLAY']
    return {int(r.fold):r for r in d.itertuples()}

def _history_diff(fold,row):
    ref=_history()[fold];bad={}
    for k in ['acc','hard_auc','sen','spe','asd_tp','asd_fn','td_fp','td_tn']:
        if abs(float(getattr(ref,k))-float(row[k]))>1e-12:bad[k]=[float(getattr(ref,k)),row[k]]
    for k in ['best_epoch','epochs_executed']:
        if int(getattr(ref,k))!=int(row[k]):bad[k]=[int(getattr(ref,k)),row[k]]
    if str(ref.prediction_sha256)!=row['prediction_sha256']:bad['prediction_sha256']=[str(ref.prediction_sha256),row['prediction_sha256']]
    return bad

def run_task(group,fold,device='cuda:0',max_epoch=450):
    t=I.modules()['torch'];start=time.monotonic();a,anchor=I.load_anchor_agent(fold,device,group)
    bestacc=0.;bestepoch=-1;wait=0;best=None;trajectory=[];optrows=[];amprows=[]
    selected={0,1,2,5,10,20,50,100,150,200,450}
    for epoch in range(max_epoch):
        _,cl,gl,total,diag=train_step(a,group,collect=(epoch in selected or epoch%10==0))
        o,m=evaluate(a);acc=m['acc']
        if acc>bestacc:bestacc=acc;bestepoch=epoch;wait=0;best=_state_payload(a)
        else:wait+=1
        if epoch in selected or epoch%10==0:
            row={'group':group,'fold':fold,'epoch':epoch,'cls_loss':_scalar(cl),'graph_loss':_scalar(gl),'total_loss':_scalar(total),**{k:v for k,v in diag.items() if not k.startswith(('update_pre_','update_no_wd_pre_','wd_component_pre_'))}}
            trajectory.append(row)
            for i in range(4):
                for nm in ['W1','W2']:
                    key=f'{I.MODALITIES[i]}_{nm}'
                    if f'{key}_actual_update' in diag:
                        optrows.append({'group':group,'fold':fold,'epoch':epoch,'parameter':key,
                            'actual_update':diag[f'{key}_actual_update'],'wd_fraction':diag[f'{key}_wd_fraction'],
                            'grad_total':diag.get(f'grad_total_MessagePassing{i}.gcn{1 if nm=="W1" else 2}.W',np.nan),
                            'cls_grad':diag.get(f'{key}_cls_grad',np.nan),'graph_grad':diag.get(f'{key}_graph_grad',np.nan)})
            _snapshot(a,fold,group,epoch,'interval',amprows)
        if (epoch+1)%10==0 or wait>200 or epoch==max_epoch-1:
            latest=RT/'checkpoints'/f'{group}_f{fold:02}.latest.pt'
            t.save({'group':group,'fold':fold,'epoch':epoch,'best_epoch':bestepoch,'bestacc':bestacc,'wait':wait,
                    'model':_state_payload(a),'best':best,'gfr_opt':a.opt_gfr.state_dict(),'cam_opt':a.opt_cam.state_dict(),
                    'rng':I.rng()},latest)
        if wait>200:break
    if best is None:raise RuntimeError('no best checkpoint')
    I.set_model(a,best);o,m=evaluate(a);p=o['logp'][a.test_idx].argmax(1).detach().cpu().numpy().astype(np.int64)
    row={'group':group,'fold':fold,'seed':0,'acc':m['acc'],'hard_auc':m['hard_auc'],'spe':m['spe'],'sen':m['sen'],
         'asd_tp':m['asd_tp'],'asd_fn':m['asd_fn'],'td_fp':m['td_fp'],'td_tn':m['td_tn'],'best_epoch':bestepoch,
         'epochs_executed':epoch+1,'prediction_sha256':hashlib.sha256(p.tobytes()).hexdigest(),
         'wall_seconds':time.monotonic()-start,'status':'COMPLETE','device':device}
    if group=='T0':
        row['historical_differences']=_history_diff(fold,row)
        if row['historical_differences']:
            raise AssertionError(f'B0-B replay mismatch fold {fold}: {row["historical_differences"]}')
    outp=RT/'checkpoints'/f'{group}_f{fold:02}.best.pt';t.save({'group':group,'fold':fold,'model':best,'prediction':p,'metrics':m,'best_epoch':bestepoch,'epochs_executed':epoch+1,'opt_gfr':a.opt_gfr.state_dict(),'opt_cam':a.opt_cam.state_dict(),'rng':I.rng()},outp)
    # CSV fragments are one file per task to avoid concurrent writers.
    pref=RESULTS/f'pilot_{group}_f{fold:02}'
    _write_csv(pref.with_suffix('.trajectory.csv'),trajectory)
    _write_csv(pref.with_suffix('.optimizer.csv'),optrows)
    _write_csv(pref.with_suffix('.amplitude.csv'),amprows)
    _write_csv(pref.with_suffix('.metrics.csv'),[row])
    latest=RT/'checkpoints'/f'{group}_f{fold:02}.latest.pt'
    if latest.exists():latest.unlink()
    print('TASK_DONE',group,fold,row['acc'],bestepoch,flush=True)
    return row

def _write_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['T0','T1','T2'],required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=450);args=p.parse_args()
    I.seed_all(0);run_task(args.group,args.fold,args.device,args.epochs)

"""Independent audits and final aggregation for Experiment 47.1."""
from __future__ import annotations
import csv, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import integration as I
import amplitude_audit as AA

EXP=I.EXP;RT=I.RT;RESULTS=EXP/'results';RESULTS.mkdir(exist_ok=True)

def _write(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def _best_agent(group,fold,device='cuda:0'):
    a,_=I.load_anchor_agent(fold,device,group);s=torch.load(RT/'checkpoints'/f'{group}_f{fold:02}.best.pt',map_location='cpu',weights_only=False);I.set_model(a,s['model']);return a,s

def _branches(a,o,mode='actual'):
    raw=o['raw'];g=a.gfr;out=[]
    for i in range(4):
        out.append(I.layer_manual(raw,o['adjs'][i],getattr(g,f'MessagePassing{i}'),g.fusion_weight[i]))
    return out

def _cam_logp(a,feat):
    was=a.cam.training;a.cam.eval()
    with torch.no_grad():logp,hid=a.cam(feat)
    if was:a.cam.train()
    return logp

def _metric_row(a,logp,group,fold,name,base):
    m=I.metric(logp,a);p=m.pop('pred');
    yy=a.y_np[a.test_idx];correct=(p==yy)
    return dict(group=group,fold=fold,intervention=name,acc=m['acc'],hard_auc=m['hard_auc'],spe=m['spe'],sen=m['sen'],
                logp_delta_mean=float((logp[a.test_idx]-base[a.test_idx]).abs().mean().detach().cpu()),
                true_class_logp_delta=float((logp[a.test_idx,torch.as_tensor(yy,device=logp.device)]-base[a.test_idx,torch.as_tensor(yy,device=logp.device)]).mean().detach().cpu()),
                class_flip=int((p!=I.metric(base,a)['pred']).sum()),correct_count=int(correct.sum()),
                asd_tp=m['asd_tp'],asd_fn=m['asd_fn'],td_fp=m['td_fp'],td_tn=m['td_tn'])

def _offdiag_norm(adj):
    # A strict cross-patient normalized operator: no diagonal/self loop is
    # inserted.  Zero-degree rows remain zero rather than becoming NaN.
    t=adj.new_zeros(adj.shape);t.copy_(adj);t.fill_diagonal_(0)
    d=t.sum(1);inv=torch.where(d>0,d.rsqrt(),torch.zeros_like(d));return inv[:,None]*t*inv[None,:]

def run_one(group,fold,device='cuda:0'):
    a,s=_best_agent(group,fold,device);o=I.eval_components(a);base=o['logp'];raw=o['raw'];g=a.gfr;branches=_branches(a,o)
    rows=[]
    def add(name,feat):rows.append(_metric_row(a,_cam_logp(a,feat),group,fold,name,base))
    add('ORIGINAL',o['feat'])
    add('GCN_OUTPUT_ZERO_STRICT',raw)
    # Self-loop only uses the source normalization with A=I, hence no
    # messages from other subjects but retains the learned GCN transforms.
    ident=torch.eye(raw.size(0),device=raw.device,dtype=raw.dtype);selfb=[]
    for i in range(4):selfb.append(I.layer_manual(raw,ident,g and getattr(g,f'MessagePassing{i}'),g.fusion_weight[i]))
    add('SELF_LOOP_ONLY',I.manual_fusion(raw,selfb))
    # Strict cross-patient operator (no diagonal) for every branch.
    cross=[]
    for i in range(4):
        mp=getattr(g,f'MessagePassing{i}');an=_offdiag_norm(o['adjs'][i]);z1=an@(raw@mp.gcn1.W);h=torch.relu(z1);z2=an@(h@mp.gcn2.W);cross.append({'scaled':torch.sigmoid(g.fusion_weight[i])*z2})
    add('CROSS_PATIENT_ONLY',I.manual_fusion(raw,cross))
    for i,mod in enumerate(I.MODALITIES):
        z=[dict(x) for x in branches];z[i]=dict(z[i]);z[i]['scaled']=torch.zeros_like(z[i]['scaled']);add('BRANCH_OFF_'+mod,I.manual_fusion(raw,z))
    # A deterministic sender-column permutation preserves row count and edge
    # weights but breaks the learned sender identity.  It is a fixed-model
    # dependence diagnostic, not a retrained counterfactual result.
    perm=torch.roll(torch.arange(raw.size(0),device=raw.device),1)
    mism=[]
    for i in range(4):
        adj=o['adjs'][i][:,perm];mism.append(I.layer_manual(raw,adj,getattr(g,f'MessagePassing{i}'),g.fusion_weight[i]))
    add('SENDER_COLUMN_PERMUTED',I.manual_fusion(raw,mism))
    return rows

def interventions(groups):
    rows=[]
    for group,folds in groups.items():
        for fold in folds:
            rows.extend(run_one(group,fold,'cuda:0'));print('INTERVENTION',group,fold,flush=True)
    _write(RESULTS/'interventions.csv',rows);return rows

def merge_training_csv():
    out={}
    for suffix,name in [('metrics','pilot_metrics.csv'),('trajectory','training_trajectory.csv'),('optimizer','optimizer_diagnostics.csv'),('amplitude','pilot_amplitude.csv')]:
        rows=[]
        for p in sorted(RESULTS.glob(f'pilot_*.{suffix}.csv')):
            with p.open(newline='') as f:rows+=list(csv.DictReader(f))
        _write(RESULTS/name,rows);out[name]=len(rows)
    return out

def mechanism_summary():
    import pandas as pd
    metrics=pd.read_csv(RESULTS/'training_trajectory.csv') if (RESULTS/'training_trajectory.csv').exists() else pd.DataFrame()
    # Best-checkpoint amplitude is recomputed directly, avoiding the sampled
    # trajectory's epoch dependence.
    rows=[]
    groups={'T0':[1,6],'T1':list(range(1,11)),'T2':[1,6]}
    for g,fs in groups.items():
        for f in fs:
            a,s=_best_agent(g,f,'cuda:0');o=I.eval_components(a);rr=AA._state_rows(f,'best_'+g,a,o)
            for r in rr:
                if r['modality'] in ['ANAT','FUNC','CORR']:rows.append(dict(group=g,**r))
    _write(RESULTS/'best_amplitudes.csv',rows);return rows

def gradient_audit(groups):
    rows=[];t=I.modules()['torch']
    for group,folds in groups.items():
        for fold in folds:
            a,s=_best_agent(group,fold,'cuda:0');a.gfr.eval();a.cam.eval()
            o=I.forward_components(a);cl=I.classification_loss(a,o['logp']);gl=I.graph_loss(a,o['adjs'],o['parts'])
            gp=list(a.gfr.parameters());cg=t.autograd.grad(cl,gp,retain_graph=True,allow_unused=True);gg=t.autograd.grad(gl,gp,retain_graph=True,allow_unused=True)
            ix={id(p):i for i,p in enumerate(gp)}
            for name,p in a.gfr.named_parameters():
                i=ix[id(p)];c=0. if cg[i] is None else float(cg[i].norm().detach().cpu());q=0. if gg[i] is None else float(gg[i].norm().detach().cpu())
                rows.append(dict(group=group,fold=fold,parameter=name,cls_grad=c,graph_grad=q,graph_to_cls_ratio=q/(c+1e-30),gradient_cosine=float((cg[i].flatten()@gg[i].flatten()).detach().cpu()/(cg[i].norm()*gg[i].norm()+1e-30)) if cg[i] is not None and gg[i] is not None and gg[i].norm()>0 else np.nan))
            rows.append(dict(group=group,fold=fold,parameter='__GFR_TOTAL__',cls_grad=float(t.sqrt(sum((q.detach().square().sum() for q in cg if q is not None),cl.new_zeros(()))).cpu()),graph_grad=float(t.sqrt(sum((q.detach().square().sum() for q in gg if q is not None),cl.new_zeros(()))).cpu()),graph_to_cls_ratio=np.nan,gradient_cosine=np.nan))
            print('GRADIENT',group,fold,flush=True)
    _write(RESULTS/'gradient_diagnostics.csv',rows);return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['merge','interventions','summary','gradients']);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.command=='merge':print(merge_training_csv())
    elif args.command=='summary':print(len(mechanism_summary()))
    elif args.command=='gradients':print(len(gradient_audit({'T0':[1,6],'T1':list(range(1,11)),'T2':[1,6]})))
    else:print(len(interventions({'T0':[1,6],'T1':list(range(1,11)),'T2':[1,6]})))

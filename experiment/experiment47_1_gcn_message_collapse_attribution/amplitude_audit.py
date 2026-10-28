"""Stage-A amplitude and fixed-factor audits for Experiment 47.1.

This file only evaluates the historical GFR.  It does not define a new
network or alter any checkpoint.  The manual factorization mirrors the two
``GraphConv`` calls in the source snapshot and uses the source
``normalize_adj`` implementation verbatim.
"""
from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import numpy as np
import integration as I

EXP=I.EXP; RESULTS=EXP/'results'; RESULTS.mkdir(exist_ok=True)

def _f(x): return float(x.detach().float().norm().cpu())
def _meanabs(x): return float(x.detach().float().abs().mean().cpu())
def _maxabs(x): return float(x.detach().float().abs().max().cpu())
def _zero(x): return float((x.detach().float()==0).float().mean().cpu())
def _var_nodes(x): return float(x.detach().float().var(dim=0,unbiased=False).mean().cpu())

def _state_agent(fold, state, device):
    a,_=I.load_anchor_agent(fold,device,'T0')
    if state=='best':
        b=I.best_checkpoint(fold)
        I.set_model(a,{'gfr':b['model']['gfr'],'cam':b['model']['pln']})
    elif state!='initial':
        raise ValueError(state)
    return a

def _source_parts(a, state):
    # state dictionaries are kept on the same device as the model for exact
    # matrix products.  Only tensors used by the audit are returned.
    g=a.gfr
    out=[]
    for i in range(4):
        mp=getattr(g,f'MessagePassing{i}')
        out.append(dict(w1=mp.gcn1.W.detach().clone(),w2=mp.gcn2.W.detach().clone(),
                        fusion=g.fusion_weight[i].detach().clone()))
    return out

def _adj_from(o): return [z.detach().clone() for z in o['adjs']]

def _one_branch(x, adj, src):
    z=I.layer_manual(x,adj,src,src['fusion'])
    return z

def _factor_rows(fold, device, initial_agent, best_agent, init_o, best_o):
    """Evaluate the four pre-registered adjacency/weight combinations."""
    t=I.modules()['torch']; raw=best_o['raw'].detach()
    init_src=_source_parts(initial_agent,'initial');best_src=_source_parts(best_agent,'best')
    choices=[('best_adj_best_weights',best_o['adjs'],best_src),
             ('best_adj_initial_weights',best_o['adjs'],init_src),
             ('initial_adj_best_weights',init_o['adjs'],best_src),
             ('initial_adj_initial_weights',init_o['adjs'],init_src)]
    rows=[]
    for name,adjs,srcs in choices:
        branches=[]
        for i in range(4):
            z=_one_branch(raw,adjs[i],srcs[i]);branches.append(z)
        fused=I.manual_fusion(raw,branches)
        for i,z in enumerate(branches):
            rows.append(dict(fold=fold,state='factor',factor=name,modality=I.MODALITIES[i],
                             x_norm=_f(raw),xw1_norm=_f(z['xw1']),agg1_norm=_f(z['agg1']),
                             relu_norm=_f(z['relu']),x2w2_norm=_f(z['x2w2']),agg2_norm=_f(z['agg2']),
                             scaled_norm=_f(z['scaled']),branch_effective_norm=_f(z['scaled']),
                             gfr_minus_x_norm=_f(fused-raw),
                             relu_zero_fraction=_zero(z['relu']),node_variance_ratio=_var_nodes(z['agg2'])/(_var_nodes(raw)+1e-30),
                             w1_norm=_f(srcs[i]['w1']),w2_norm=_f(srcs[i]['w2']),w1_spectral=I.spectral(srcs[i]['w1']),w2_spectral=I.spectral(srcs[i]['w2']),
                             fusion_sigmoid=float(t.sigmoid(srcs[i]['fusion']).detach().cpu()),
                             adj_norm_norm=_f(z['an']),adj_nonzero=int((adjs[i].detach()!=0).sum().cpu()),
                             adj_density=float((adjs[i].detach()!=0).float().mean().cpu()),adj_max=_maxabs(adjs[i]),
                             fp64_rel_error=np.nan))
    return rows

def _state_rows(fold,state,a,o):
    t=I.modules()['torch'];rows=[];raw=o['raw'].detach();g=a.gfr
    for i,mod in enumerate(I.MODALITIES):
        mp=getattr(g,f'MessagePassing{i}')
        z=I.layer_manual(raw,o['adjs'][i],mp,g.fusion_weight[i])
        # Independent FP64 products for the three sensitive matrix products.
        x64=raw.detach().double();a64=o['adjs'][i].detach().double();
        eye=t.eye(a64.size(0),device=a64.device,dtype=a64.dtype)
        an64=I.modules()['utils'].normalize_adj(a64+eye)
        w164=mp.gcn1.W.detach().double();w264=mp.gcn2.W.detach().double()
        h164=an64@(x64@w164);r64=t.relu(h164);q64=an64@(r64@w264)
        fp32=[z['xw1'],z['agg1'],z['agg2']];fp64=[x64@w164,h164,q64]
        fp64err=[float((u.float()-v.float()).norm().cpu()/(v.float().norm().cpu()+1e-30)) for u,v in zip(fp32,fp64)]
        rows.append(dict(fold=fold,state=state,factor='actual',modality=mod,
            x_norm=_f(raw),xw1_norm=_f(z['xw1']),agg1_norm=_f(z['agg1']),relu_norm=_f(z['relu']),
            x2w2_norm=_f(z['x2w2']),agg2_norm=_f(z['agg2']),scaled_norm=_f(z['scaled']),
            branch_effective_norm=_f(z['scaled']),gfr_minus_x_norm=_f(o['feat']-raw),
            relu_zero_fraction=_zero(z['relu']),node_variance_ratio=_var_nodes(z['agg2'])/(_var_nodes(raw)+1e-30),
            w1_norm=_f(mp.gcn1.W),w2_norm=_f(mp.gcn2.W),w1_spectral=I.spectral(mp.gcn1.W),w2_spectral=I.spectral(mp.gcn2.W),
            fusion_sigmoid=float(t.sigmoid(g.fusion_weight[i]).detach().cpu()),adj_norm_norm=_f(z['an']),
            adj_nonzero=int((o['adjs'][i].detach()!=0).sum().cpu()),adj_density=float((o['adjs'][i].detach()!=0).float().mean().cpu()),
            adj_max=_maxabs(o['adjs'][i]),fp64_rel_error=float(max(fp64err))))
    return rows

def run_fold(fold,device='cuda:0'):
    initial=_state_agent(fold,'initial',device);best=_state_agent(fold,'best',device)
    io=I.eval_components(initial);bo=I.eval_components(best)
    rows=_state_rows(fold,'initial',initial,io)+_state_rows(fold,'best',best,bo)
    rows+=_factor_rows(fold,device,initial,best,io,bo)
    # Add explicit initial/final ratios to actual rows for direct attribution.
    by={(r['fold'],r['state'],r['modality']):r for r in rows if r['factor']=='actual'}
    for mod in I.MODALITIES:
        ir=by[(fold,'initial',mod)];br=by[(fold,'best',mod)]
        for k in ['xw1_norm','agg1_norm','relu_norm','x2w2_norm','agg2_norm','scaled_norm','branch_effective_norm','gfr_minus_x_norm','w1_norm','w2_norm','adj_norm_norm']:
            br['ratio_'+k]=float(br[k]/(ir[k]+1e-30))
    return rows

def write_rows(rows,path):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folds',default='1,2,3,4,5,6,7,8,9,10');p.add_argument('--gpu',default='0');p.add_argument('--output',default=str(RESULTS/'layer_amplitudes.csv'));args=p.parse_args()
    I.seed_all(0);allrows=[]
    for fold in map(int,args.folds.split(',')):
        allrows.extend(run_fold(fold,f'cuda:0'))
        print('STAGE_A_FOLD',fold,flush=True)
    write_rows(allrows,Path(args.output))
    print('WROTE',len(allrows),flush=True)

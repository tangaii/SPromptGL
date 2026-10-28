"""Experiment 47.3 integration built on the verified 47.2 source snapshot.

The project source is imported from ``.runtime/sources`` through
``integration_base``. This module adds only the pre-registered D4/D5 relation
controls and corrected scale audit; it does not alter SPromptGL code.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
import integration_base as B

ROOT=B.ROOT
EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime'
RESULTS=EXP/'results'
DIMS=B.DIMS
SLICES=B.SLICES
MODALITIES=B.MODALITIES
rho=0.25
eps=1e-8

modules=B.modules
seed_all=B.seed_all
rng=B.rng
restore=B.restore
set_model=B.set_model
make_cam=B.make_cam
layer_manual=B.layer_manual
classification_loss=B.classification_loss
graph_loss=B.graph_loss
def state_payload(a):
    return {'gfr':{k:v.detach().cpu().clone() for k,v in a.gfr.state_dict().items()},
            'cam':{k:v.detach().cpu().clone() for k,v in a.cam.state_dict().items()}}

def load_payload(a,payload):
    a.gfr.load_state_dict(payload['gfr']);a.cam.load_state_dict(payload['cam'])

def metric_indices(logp,a,idx):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    p=logp.detach().cpu().numpy()[idx].argmax(1);y=a.y_np[idx]
    tp,fn,fp,tn=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    return {'acc':float((p==y).mean()),
            'hard_auc':float(roc_auc_score(np.eye(2)[y],np.eye(2)[p])),
            'spe':float(tn/(tn+fp)) if tn+fp else 0.,
            'sen':float(tp/(tp+fn)) if tp+fn else 0.,
            'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,
            'pred':p.astype(np.int64)}

def weighted_nll(logp,a,idx):
    return float(torch.nn.functional.nll_loss(logp[idx],a.y[idx],weight=a.class_weight).detach().cpu())

def train_metrics(logp,a):
    tr=metric_indices(logp,a,a.train_idx);te=metric_indices(logp,a,a.test_idx)
    tr['nll']=weighted_nll(logp,a,a.train_idx);te['nll']=weighted_nll(logp,a,a.test_idx)
    return tr,te

def data(): return B.data()

def load_anchor_agent(fold,device='cuda:0',fixed=True):
    a,s=B.load_anchor_agent(fold,device,group='T1')
    if fixed:
        for n,p in a.gfr.named_parameters():
            if n.startswith('GraphConstruct'): p.requires_grad_(False)
    return a,s

def make_optimizer(a):
    t=modules()['torch'];h=a.hp
    target_names=('MessagePassing1.gcn1.W','MessagePassing1.gcn2.W',
                  'MessagePassing2.gcn1.W','MessagePassing2.gcn2.W',
                  'MessagePassing3.gcn1.W','MessagePassing3.gcn2.W')
    target=[];other=[]
    for n,p in a.gfr.named_parameters():
        if n in target_names: target.append(p)
        elif p.requires_grad: other.append(p)
    return t.optim.Adam([{'params':other,'weight_decay':h.reg},
                         {'params':target,'weight_decay':0.0}],lr=h.lr)

def raw_parts(a):
    ps=a.parts_layer(a.x)[1]
    return ps,torch.cat(ps,dim=1)

def branch_outputs(a,adjs,raw):
    return [layer_manual(raw,adjs[i],getattr(a.gfr,f'MessagePassing{i}'),a.gfr.fusion_weight[i]) for i in range(4)]

def load_controls(fold,device='cpu'):
    q=torch.load(RT/'relation_controls'/f'fold{fold:02}.pt',map_location=device,weights_only=False)
    base=torch.load(RT/'fixed_graphs'/f'fold{fold:02}.pt',map_location=device,weights_only=False)
    out={}
    for k in ('true','D4'):
        if k in q: out[k]=[z.to(device) for z in q[k]]
    for k in ('rewire','self'):
        if k in base: out[k]=[z.to(device) for z in base[k]]
    for k in ('sites','split','groups'):
        if k in q: out[k]=q[k]
    out['perm']=torch.as_tensor(q['perm'],device=device,dtype=torch.long)
    out['meta']=q.get('meta',{})
    return out

def graph_sha(adjs):
    h=hashlib.sha256()
    for z in adjs: h.update(z.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def effective_message(a,adjs,perm=None):
    ps,raw=raw_parts(a)
    bs=branch_outputs(a,adjs,raw)
    M=bs[1]['scaled']+bs[2]['scaled']+bs[3]['scaled']
    if perm is not None: M=M[perm]
    return ps,raw,bs,M

def scaled_forward(a,group,controls=None,compute_cam=True):
    ps,raw=raw_parts(a)
    if controls is None: controls=load_controls(a.fold,a.x.device)
    if group=='D4': adjs=controls['D4']; perm=None
    elif group=='D5': adjs=controls['true']; perm=controls['perm']
    elif group=='D1': adjs=controls['true']; perm=None
    elif group=='D2': adjs=controls['rewire']; perm=None
    elif group=='D3': adjs=controls['self']; perm=None
    else: raise ValueError(group)
    adjs=[z.to(raw.device) for z in adjs]
    bs=branch_outputs(a,adjs,raw)
    M=bs[1]['scaled']+bs[2]['scaled']+bs[3]['scaled']
    if perm is not None: M=M[perm.to(raw.device)]
    tr=torch.as_tensor(a.train_idx,device=raw.device,dtype=torch.long)
    mu_x=raw[tr].mean(0).detach();mu_m=M[tr].mean(0).detach()
    xc=raw[tr]-mu_x;mc_train=M[tr]-mu_m
    scale=(torch.tensor(rho,device=raw.device,dtype=raw.dtype)*xc.norm()/(mc_train.norm()+eps)).detach()
    Z=raw+scale*(M-mu_m)
    logp=hid=None
    if compute_cam: logp,hid=a.cam(Z)
    cl=classification_loss(a,logp) if logp is not None else None
    aux={'raw':raw,'parts':ps,'adjs':adjs,'branches':bs,'M':M,'Z':Z,
         'mu_x':mu_x,'mu_m':mu_m,'scale':scale,'raw_norm':raw.norm(),
         'M_norm':M.norm(),'Mc_norm':(M-mu_m).norm(),'Z_norm':Z.norm(),
         'Xc_train_norm':xc.norm(),'Mc_train_norm':mc_train.norm(),
         'train_M_mean_norm':M[tr].mean(0).norm(),
         'patient_var':(M[tr]-M[tr].mean(0)).pow(2).mean().sqrt()}
    return logp,hid,cl,torch.zeros((),device=raw.device),adjs,aux

def train_group_is_fixed(group): return group in ('D4','D5')

def seed_for_task(seed=0): seed_all(seed)

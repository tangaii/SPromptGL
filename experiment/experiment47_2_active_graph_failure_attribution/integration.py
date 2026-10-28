"""Private integration for Experiment 47.2.

The original SPromptGL modules are imported through the immutable 47.1 source
snapshot.  This file only exposes the pre-registered scale-controlled forward
and fixed-graph diagnostic paths; it does not modify the project source.
"""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch

import integration_base as I

# Re-export the immutable integration primitives used by the audit scripts.
modules=I.modules
seed_all=I.seed_all
rng=I.rng
restore=I.restore
classification_loss=I.classification_loss
graph_loss=I.graph_loss
layer_manual=I.layer_manual
set_model=I.set_model
forward_components=I.forward_components
best_checkpoint=I.best_checkpoint
make_cam=I.make_cam

ROOT=I.ROOT
EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime'
RESULTS=EXP/'results'
DIMS=I.DIMS
SLICES=I.SLICES
MODALITIES=I.MODALITIES
rho=0.25
eps=1e-8

def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def make_agent(fold, device='cuda:0'):
    return I.make_agent(fold, device)

def load_anchor_agent(fold, device='cuda:0', fixed=False):
    a,s=I.load_anchor_agent(fold, device, group='T1')
    # 47.2 starts every group from the same untrained B0-B anchor, but the
    # six effective W matrices use the pre-registered zero-decay optimizer.
    if fixed:
        for n,p in a.gfr.named_parameters():
            if n.startswith('GraphConstruct'):
                p.requires_grad_(False)
    return a,s

def fixed_true_graphs(fold, device='cpu'):
    p=RT/'fixed_graphs'/f'fold{fold:02}.pt'
    s=torch.load(p,map_location=device,weights_only=False)
    return {k:[q.to(device) for q in v] for k,v in s.items() if k in ('true','rewire','self')}

def _effective_targets():
    return ('MessagePassing1.gcn1.W','MessagePassing1.gcn2.W',
            'MessagePassing2.gcn1.W','MessagePassing2.gcn2.W',
            'MessagePassing3.gcn1.W','MessagePassing3.gcn2.W')

def make_optimizer(a):
    t=I.modules()['torch'];h=a.hp
    targets=[];others=[]
    for n,p in a.gfr.named_parameters():
        if n in _effective_targets(): targets.append(p)
        elif p.requires_grad: others.append(p)
    # GraphConstruct is included only for dynamic D0. For fixed groups it is
    # frozen before this function is called and therefore excluded.
    return t.optim.Adam([{'params':others,'weight_decay':h.reg},
                         {'params':targets,'weight_decay':0.0}],lr=h.lr)

def raw_parts(a):
    ps=a.parts_layer(a.x)[1]
    return ps, torch.cat(ps,dim=1)

def branch_outputs(a, adjs, raw):
    bs=[]
    for i in range(4):
        bs.append(I.layer_manual(raw,adjs[i],getattr(a.gfr,f'MessagePassing{i}'),a.gfr.fusion_weight[i]))
    return bs

def scaled_forward(a, group, fixed_adjs=None, training=True, compute_cam=True):
    """Return scaled Z and losses, preserving the historical PHENO overwrite.

    The effective message is branch ANAT+FUNC+CORR (indices 1,2,3).  The
    PHENO branch is evaluated for audit completeness but is excluded from Z,
    exactly matching the original i==1 overwrite.
    """
    ps,raw=raw_parts(a)
    if fixed_adjs is None:
        adjs=[getattr(a.gfr,f'GraphConstruct{i}')(ps[i]) for i in range(4)]
    else:
        adjs=[q.to(raw.device) for q in fixed_adjs]
    bs=branch_outputs(a,adjs,raw)
    M=bs[1]['scaled']+bs[2]['scaled']+bs[3]['scaled']
    tr=torch.as_tensor(a.train_idx,device=raw.device,dtype=torch.long)
    mu_x=raw[tr].mean(dim=0).detach()
    mu_m=M[tr].mean(dim=0).detach()
    xc=raw[tr]-mu_x
    mc_train=M[tr]-mu_m
    scale=(torch.tensor(rho,device=raw.device,dtype=raw.dtype)*xc.norm()/(mc_train.norm()+eps)).detach()
    Z=raw+scale*(M-mu_m)
    logp=hid=None
    if compute_cam:
        logp,hid=a.cam(Z)
    cl=None;gl=raw.new_zeros(())
    if logp is not None: cl=I.classification_loss(a,logp)
    if group=='D0': gl=I.graph_loss(a,adjs,ps)
    aux={'raw':raw,'parts':ps,'adjs':adjs,'branches':bs,'M':M,'Z':Z,
         'mu_x':mu_x,'mu_m':mu_m,'scale':scale,'raw_norm':raw.norm(),
         'M_norm':M.norm(),'Mc_norm':(M-mu_m).norm(),'Z_norm':Z.norm(),
         'Xc_train_norm':xc.norm(),'Mc_train_norm':mc_train.norm(),
         'train_M_mean_norm':M[tr].mean(0).norm(),
         'patient_var':(M[tr]-M[tr].mean(0)).pow(2).mean().sqrt()}
    return logp,hid,cl,gl,adjs,aux

def metric_indices(logp,a,idx):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    p=logp.detach().cpu().numpy()[idx].argmax(1); y=a.y_np[idx]
    tp,fn,fp,tn=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    return {'acc':float((p==y).mean()),
            'hard_auc':float(roc_auc_score(np.eye(2)[y],np.eye(2)[p])),
            'spe':float(tn/(tn+fp)) if tn+fp else 0.,
            'sen':float(tp/(tp+fn)) if tp+fn else 0.,
            'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,
            'pred':p.astype(np.int64)}

def weighted_nll(logp,a,idx):
    return float(I.modules()['torch'].nn.functional.nll_loss(logp[idx],a.y[idx],weight=a.class_weight).detach().cpu())

def train_metrics(logp,a):
    tr=metric_indices(logp,a,a.train_idx);te=metric_indices(logp,a,a.test_idx)
    tr['nll']=weighted_nll(logp,a,a.train_idx);te['nll']=weighted_nll(logp,a,a.test_idx)
    return tr,te

def state_payload(a):
    return {'gfr':{k:v.detach().cpu().clone() for k,v in a.gfr.state_dict().items()},
            'cam':{k:v.detach().cpu().clone() for k,v in a.cam.state_dict().items()}}

def load_payload(a,payload):
    a.gfr.load_state_dict(payload['gfr']);a.cam.load_state_dict(payload['cam'])

def graph_sha(adjs):
    h=hashlib.sha256()
    for a in adjs: h.update(a.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train_group_is_fixed(group): return group in ('D1','D2','D3')

def seed_for_task(seed=0):
    I.seed_all(seed)


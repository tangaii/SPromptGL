"""Isolated, read-only integration for Experiment 47.

The implementation deliberately imports the PEN-free source snapshot from
``.runtime/sources``.  It never imports the project root network module, and
all model calls are evaluation-only unless a diagnostic explicitly asks for
autograd gradients.
"""
from __future__ import annotations
import copy, hashlib, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path('/data/adodas/SPromptGL-main')
EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'
DIMS = [48, 6, 10, 256]
MODALITIES = ['PHENO', 'ANAT', 'FUNC', 'CORR']
SLICES = [(0,48), (48,54), (54,64), (64,320)]
B0_DIR = ROOT / 'experiment/experiment43_1_b0b_gfr_mechanism_retest'
B0_RT = B0_DIR / '.runtime'
CAM_DIR = ROOT / 'experiment/experiment43_4_physical_cam_only_validation'
CAM_RT = CAM_DIR / '.runtime'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/'data/ABIDE'
    frame=pd.read_csv(base/'processed_standard_data.csv')
    mapping=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert [len(mapping[k]) for k in mapping]==DIMS
    x=frame.iloc[:,:-1].to_numpy(np.float32)
    y=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert x.shape==(871,320) and np.isfinite(x).all()
    assert np.bincount(y).tolist()==[403,468]
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
    # The 26 age bins, 20 site one-hot columns and sex are the actual PHENO
    # fields.  No unobserved QC field is invented here.
    pheno=list(mapping['PHENO'])
    age_cols=[i for i,n in enumerate(pheno) if str(n).startswith('AGE_AT_SCAN_')]
    site_cols=[i for i,n in enumerate(pheno) if str(n).startswith('SITE_ID_')]
    sex_cols=[i for i,n in enumerate(pheno) if str(n).startswith('SEX_')]
    site=np.argmax(x[:,np.asarray(site_cols)],axis=1).astype(np.int64)
    age=np.argmax(x[:,np.asarray(age_cols)],axis=1).astype(np.int64)
    sex=np.argmax(x[:,np.asarray(sex_cols)],axis=1).astype(np.int64)
    meta=dict(mapping=mapping,age=age,site=site,sex=sex,age_cols=age_cols,site_cols=site_cols,sex_cols=sex_cols)
    return x,y,folds,meta

def hp():
    return SimpleNamespace(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
        nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,
        n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',
        MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,
        nclass=2,mode='simple-2',seed=0)

_CTX=None
def modules():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        src=str(RT/'sources')
        if src not in sys.path: sys.path.insert(0,src)
        import layers, network, utils
        _CTX=dict(torch=torch,layers=layers,network=network,utils=utils)
        assert not hasattr(network,'PLN') and not any('CLIP' in x for x in sys.modules)
    return _CTX

def seed_all(seed=0):
    c=modules(); t=c['torch']
    random.seed(seed); np.random.seed(seed); t.manual_seed(seed); t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark=False; t.backends.cudnn.deterministic=True

def make_pln():
    c=modules(); t=c['torch']; nn=t.nn; h=hp()
    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__(); self.hyperpm=h; self.input_data_dims=DIMS
            self.d_q=self.d_k=self.d_v=h.n_hidden; self.n_head=h.n_head; self.dropout=h.dropout
            self.n_layer=h.nlayer; self.modal_num=h.nmodal; self.n_class=h.nclass
            self.d_out=self.d_v*self.n_head*self.modal_num
            self.fusion_weight=nn.Parameter(t.ones([6,1]))
            self.InputLayer=c['layers'].VariLengthInputLayer(DIMS,self.d_k,self.d_v,self.n_head,self.dropout)
            self.Encoder=[]; self.FeedForward=[]
            for i in range(self.n_layer):
                e=c['layers'].EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout)
                f=c['layers'].FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout)
                self.add_module('encode_%d'%i,e); self.add_module('feed_%d'%i,f)
                self.Encoder.append(e); self.FeedForward.append(f)
            self.Outputlayer=c['layers'].OutputLayer(self.d_v*self.n_head*self.modal_num,
                                                     self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
        def forward(self,x):
            bs=x.size(0); x,a0=self.InputLayer(x); a=a0.mean(dim=1)
            for i in range(self.n_layer):
                x,al=self.Encoder[i](q=x,k=x,v=x,modal_num=self.modal_num)
                a=al.mean(dim=1); x=self.FeedForward[i](x)
            return self.Outputlayer(x.view(bs,-1),a.view(bs,-1))
    return PLNNoPEN()

def load_fold(fold, device='cuda:0', cam=False):
    c=modules(); t=c['torch']; x,y,folds,meta=load_data(); tr,te=folds[fold-1]
    dev=t.device(device); xt=t.from_numpy(x).to(dev); yt=t.from_numpy(y).long().to(dev)
    if cam:
        model=make_pln().to(dev)
        ck=B0_RT/'checkpoints'/f'g0_f{fold:02}.best.pt'
        # CAM-only checkpoints are separately stored under Experiment 43.4.
        ck=CAM_RT/'checkpoints'/f'g0_f{fold:02}.best.pt'
        state=t.load(ck,map_location='cpu',weights_only=False)['model']['cam']
        model.load_state_dict(state); model.eval()
        return dict(fold=fold,x=x,y=y,train=np.asarray(tr),test=np.asarray(te),meta=meta,
                    xt=xt,yt=yt,gfr=None,cam=model,ckpt=str(ck),ckpt_sha=sha(ck))
    gfr=c['network'].GFR(DIMS,DIMS,th=.9,mode='weighted-cosine').to(dev)
    pln=make_pln().to(dev)
    ck=B0_RT/'checkpoints'/f'g0_f{fold:02}.best.pt'; state=t.load(ck,map_location='cpu',weights_only=False)
    gfr.load_state_dict(state['model']['gfr']); pln.load_state_dict(state['model']['pln'])
    gfr.eval(); pln.eval()
    return dict(fold=fold,x=x,y=y,train=np.asarray(tr),test=np.asarray(te),meta=meta,
                xt=xt,yt=yt,gfr=gfr,cam=pln,ckpt=str(ck),ckpt_sha=sha(ck),checkpoint=state)

def split_features(tensor):
    return [tensor[:,lo:hi] for lo,hi in SLICES]

def norm_adj(a):
    c=modules(); t=c['torch']; eye=t.eye(a.size(0),device=a.device,dtype=a.dtype)
    b=a+eye
    d=b.sum(1).clamp_min(1e-12).pow(-.5)
    return d[:,None]*b*d[None,:]

def gfr_extract(ctx, adjs=None, x_override=None):
    """Exact original GFR fusion, with optional fixed adjacency intervention."""
    c=modules(); t=c['torch']; x=ctx['xt'] if x_override is None else x_override
    parts=split_features(x); raw=t.cat(parts,1); g=ctx['gfr']; msgs=[]; outs=[]
    if adjs is None:
        # The caller controls autograd.  Normal fixed-checkpoint calls wrap
        # this function in no_grad; gradient diagnostics deliberately do not.
        _,adjs,_=g(parts,4,ctx['yt'],ctx['train'])
    for i,a in enumerate(adjs):
        na=norm_adj(a); z=g.__getattr__(f'MessagePassing{i}')(raw,na)
        fw=t.sigmoid(g.fusion_weight[i]); msgs.append(fw*z); outs.append(z)
    # This intentionally reproduces the historical i==1 overwrite.
    combined=msgs[0]+raw
    combined=msgs[1]+raw
    combined=msgs[2]+combined
    combined=msgs[3]+combined
    return combined,adjs,msgs,outs

def original_forward(ctx, adjs=None, x_override=None):
    c=modules(); t=c['torch']
    feat,adjs,msgs,outs=gfr_extract(ctx,adjs,x_override)
    logp,hidden=ctx['cam'](feat)
    return dict(logp=logp,hidden=hidden,feat=feat,adjs=adjs,msgs=msgs,outs=outs)

def cam_forward(ctx,x_override=None):
    out,hidden=ctx['cam'](ctx['xt'] if x_override is None else x_override)
    return dict(logp=out,hidden=hidden)

def baseline_adjs(ctx):
    c=modules(); t=c['torch']; parts=split_features(ctx['xt']); g=ctx['gfr']
    with t.no_grad():
        _,adjs,_=g(parts,4,ctx['yt'],ctx['train'])
    return [a.detach().clone() for a in adjs]

def metric_np(logp,y,test):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    p=np.asarray(logp)[np.asarray(test)].argmax(1); tar=np.asarray(y)[np.asarray(test)]
    tp,fn,fp,tn=map(int,confusion_matrix(tar,p,labels=[0,1]).ravel())
    return dict(acc=float((p==tar).mean()),hard_auc=float(roc_auc_score(np.eye(2)[tar],np.eye(2)[p])),
                spe=float(tn/(tn+fp)) if tn+fp else 0.,sen=float(tp/(tp+fn)) if tp+fn else 0.,
                asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn,pred=p)

def adjacency_stats(a, labels=None, site=None):
    a=np.asarray(a); n=a.shape[0]; off=a.copy(); np.fill_diagonal(off,0)
    mask=off>0; deg=mask.sum(1); w=off[mask]
    vals={'n':n,'edges':int(mask.sum()),'density':float(mask.mean()),'degree_mean':float(deg.mean()),
          'degree_sd':float(deg.std()),'degree_min':int(deg.min()),'degree_max':int(deg.max()),
          'isolated':int((deg==0).sum()),'weight_mean':float(w.mean()) if w.size else 0.,
          'weight_sd':float(w.std()) if w.size else 0.,'weight_p50':float(np.quantile(w,.5)) if w.size else 0.,
          'weight_p90':float(np.quantile(w,.9)) if w.size else 0.,
          'symmetry_mae':float(np.abs(off-off.T).mean()),'self_weight_mean':float(np.diag(a).mean())}
    if labels is not None:
        lab=np.asarray(labels); vals['homophily_unweighted']=float((lab[:,None]==lab[None,:])[mask].mean()) if mask.any() else np.nan
        vals['homophily_weighted']=float(np.sum(off*(lab[:,None]==lab[None,:]))/(off.sum()+1e-12))
    if site is not None:
        s=np.asarray(site); vals['same_site_edge_rate']=float((s[:,None]==s[None,:])[mask].mean()) if mask.any() else np.nan
        vals['same_site_weighted']=float(np.sum(off*(s[:,None]==s[None,:]))/(off.sum()+1e-12))
    return vals

def row_random_adjacency(a, allowed, rng, same_site=None, site=None):
    """Preserve each row's off-diagonal support count and weight multiset."""
    a=np.asarray(a); n=a.shape[0]; out=np.zeros_like(a); np.fill_diagonal(out,np.diag(a)); off=a.copy(); np.fill_diagonal(off,0)
    for i in range(n):
        old=np.flatnonzero(off[i]>0); k=len(old)
        if not k: continue
        pool=np.asarray(allowed[i] if allowed is not None else [j for j in range(n) if j!=i],dtype=int)
        pool=pool[pool!=i]
        if same_site and site is not None:
            p2=pool[np.asarray(site)[pool]==np.asarray(site)[i]]
            if len(p2)>=k: pool=p2
        if len(pool)<k: pool=np.asarray([j for j in range(n) if j!=i],dtype=int)
        new=rng.choice(pool,size=k,replace=False); weights=off[i,old]
        out[i,new]=weights[rng.permutation(k)]
    return out

def permute_neighbor_features(x, a, rng):
    n=x.shape[0]; out=x.copy(); perm=rng.permutation(n)
    # Keep the receiver's own raw feature intact, replacing only non-self
    # sender features used by the fixed graph message.
    return out[perm],perm


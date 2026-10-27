"""Read-only integration for Experiment 44.1.

All mutable artifacts are below this experiment directory. Candidate graphs are
fitted with outer-train statistics and the same explicit modality slices are
used by graph construction, teachers, students and audits.
"""
from __future__ import annotations
import hashlib, json, os, shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main'); EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'
DIMS=(48,6,10,256); MODALITIES=('PHENO','ANAT','FUNC','CORR'); SLICES=((0,48),(48,54),(54,64),(64,320))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(tmp,path)

def slice_for_modality(x, modality):
    m=MODALITIES.index(modality) if isinstance(modality,str) else int(modality); lo,hi=SLICES[m]; return x[...,lo:hi]

def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/'data/ABIDE'; frame=pd.read_csv(base/'processed_standard_data.csv'); mapping=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']; assert [len(v) for v in mapping.values()]==list(DIMS); assert frame.shape==(871,321)
    assert list(frame.columns[:-1])==[str(v) for k in mapping for v in mapping[k]]
    x=frame.iloc[:,:-1].to_numpy(np.float32); y=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); return x,y,folds,mapping

def fold_sha(train,test): return hashlib.sha256(np.asarray(train,np.int64).tobytes()+np.asarray(test,np.int64).tobytes()).hexdigest()

def fit_block_normalizer(x,fit_idx):
    z=np.asarray(x,np.float64); f=np.asarray(fit_idx,np.int64); mean=np.zeros(320,np.float64); scale=np.ones(320,np.float64)
    for lo,hi in SLICES:
        mean[lo:hi]=z[f,lo:hi].mean(0); scale[lo:hi]=z[f,lo:hi].std(0); scale[lo:hi][scale[lo:hi]<1e-8]=1.
    return mean,scale

def apply_block_normalizer(x,stats):
    z=np.asarray(x,np.float64).copy(); mean,scale=stats
    for lo,hi in SLICES:
        z[:,lo:hi]=(z[:,lo:hi]-mean[lo:hi])/scale[lo:hi]; n=np.linalg.norm(z[:,lo:hi],axis=1,keepdims=True); n[n<1e-8]=1.; z[:,lo:hi]/=n
    return z.astype(np.float32)

def candidate_pool(x,receivers,senders,k=16,fit_idx=None,stats=None):
    x=np.asarray(x,np.float32); receivers=np.asarray(receivers,np.int64); senders=np.asarray(senders,np.int64)
    if stats is None:
        if fit_idx is None: raise ValueError('candidate normalization requires fit_idx')
        stats=fit_block_normalizer(x,fit_idx)
    z=apply_block_normalizer(x,stats); out=np.full((len(x),4,k),-1,np.int64)
    for m,(lo,hi) in enumerate(SLICES):
        score=z[receivers,lo:hi]@z[senders,lo:hi].T
        for r,row in enumerate(receivers):
            keep=senders!=row; allowed=senders[keep]; vals=score[r,keep]; order=np.lexsort((allowed,-vals))[:k]; out[row,m,:len(order)]=allowed[order]
    return out

def outer_candidates(x,train,test,k=16): return candidate_pool(x,np.concatenate([np.asarray(train),np.asarray(test)]),train,k=k,fit_idx=train)
def inner_candidates(x,outer_train,inner_train,k=16): return candidate_pool(x,np.asarray(outer_train),inner_train,k=k,fit_idx=inner_train)

def source_copy():
    src=RT/'sources'; src.mkdir(parents=True,exist_ok=True); e43=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/sources'
    for name in ('network.py','layers.py','utils.py'): shutil.copy2(e43/name,src/name)
    return {name:sha(src/name) for name in ('network.py','layers.py','utils.py')}

def make_hp():
    return SimpleNamespace(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2',seed=0)

def make_pln(layers,hp=None):
    import torch
    import torch.nn as nn
    hp=hp or make_hp()
    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__(); self.hyperpm=hp; self.input_data_dims=list(DIMS); self.d_q=self.d_k=self.d_v=hp.n_hidden; self.n_head=hp.n_head; self.dropout=hp.dropout; self.n_layer=hp.nlayer; self.modal_num=hp.nmodal; self.n_class=hp.nclass; self.fusion_weight=nn.Parameter(torch.ones(6,1)); self.InputLayer=layers.VariLengthInputLayer(list(DIMS),self.d_k,self.d_v,self.n_head,self.dropout); self.Encoder=[]; self.FeedForward=[]
            for i in range(self.n_layer):
                e=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout); f=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout); self.add_module('encode_%d'%i,e); self.add_module('feed_%d'%i,f); self.Encoder.append(e); self.FeedForward.append(f)
            self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*self.modal_num,self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
        def forward(self,x):
            bs=x.size(0); x,input_attention=self.InputLayer(x); attn=input_attention.mean(dim=1)
            for index in range(self.n_layer): x,layer_attention=self.Encoder[index](q=x,k=x,v=x,modal_num=self.modal_num); attn=layer_attention.mean(dim=1); x=self.FeedForward[index](x)
            return self.Outputlayer(x.view(bs,-1),attn.view(bs,-1))
    return PLNNoPEN()

def build(layers,hp=None): return make_pln(layers,hp or make_hp())
def model_state(module): return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in module.items()}
def state_hash(state):
    h=hashlib.sha256()
    for key in sorted(state):
        for name in sorted(state[key]): h.update((key+'.'+name).encode()+b'\0'+state[key][name].numpy().tobytes())
    return h.hexdigest()

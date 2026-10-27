"""Data, fold, source and candidate-pool integration for Experiment 44.

This file only reads the repository inputs.  The PEN-free source copy and all
generated artifacts live below this experiment directory.
"""
from __future__ import annotations
import hashlib, json, os, random, shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path('/data/adodas/SPromptGL-main')
EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'
DIMS = (48, 6, 10, 256)
SLICES = ((0,48),(48,54),(54,64),(64,320))
MODALITIES = ('PHENO','ANAT','FUNC','CORR')

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/'data/ABIDE'
    frame=pd.read_csv(base/'processed_standard_data.csv')
    mapping=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert [len(v) for v in mapping.values()]==list(DIMS)
    assert frame.shape==(871,321)
    assert list(frame.columns[:-1])==[str(v) for k in mapping for v in mapping[k]]
    x=frame.iloc[:,:-1].to_numpy(np.float32); y=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
    return x,y,folds,mapping

def fold_sha(train,test):
    return hashlib.sha256(np.asarray(train,np.int64).tobytes()+np.asarray(test,np.int64).tobytes()).hexdigest()

def norm_block(x):
    z=x.astype(np.float64); z=z-z.mean(0,keepdims=True); scale=z.std(0,keepdims=True); scale[scale<1e-8]=1
    z=z/scale; n=np.linalg.norm(z,axis=1,keepdims=True); n[n<1e-8]=1
    return (z/n).astype(np.float32)

def candidate_pool(x, receivers, senders, k=16):
    """Fixed row receiver/column sender candidate indices, no labels."""
    out=np.full((len(x),4,k),-1,np.int64)
    senders=np.asarray(senders,np.int64); receivers=np.asarray(receivers,np.int64)
    for m,(lo,hi) in enumerate(SLICES):
        z=norm_block(x[:,lo:hi])
        score=z[receivers]@z[senders].T
        for r,row in enumerate(receivers):
            keep = senders != row
            allowed = senders[keep]
            # score columns were computed before self exclusion; deterministic sort
            vals = score[r, keep]
            order=np.lexsort((allowed,-vals))[:k]
            out[row,m,:len(order)]=allowed[order]
    return out

def all_candidates(x,train,test,k=16):
    return candidate_pool(x,np.concatenate([train,test]),train,k)

def source_copy():
    """Record the B0-B private source; it is never imported from root."""
    src=RT/'sources'; src.mkdir(parents=True,exist_ok=True)
    e43=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/sources'
    for name in ('network.py','layers.py','utils.py'):
        shutil.copy2(e43/name,src/name)
    return {name:sha(src/name) for name in ('network.py','layers.py','utils.py')}

def make_hp():
    return SimpleNamespace(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
      nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,
      n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',
      alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2',seed=0)

def make_pln(layers, hp):
    import torch.nn as nn
    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__(); self.hyperpm=hp; self.input_data_dims=list(DIMS)
            self.d_q=self.d_k=self.d_v=hp.n_hidden; self.n_head=hp.n_head; self.dropout=hp.dropout
            self.n_layer=hp.nlayer; self.modal_num=hp.nmodal; self.n_class=hp.nclass
            import torch
            self.fusion_weight=nn.Parameter(torch.ones(6,1))
            self.InputLayer=layers.VariLengthInputLayer(list(DIMS),self.d_k,self.d_v,self.n_head,self.dropout)
            self.Encoder=[]; self.FeedForward=[]
            for i in range(self.n_layer):
                e=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout)
                f=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout)
                self.add_module('encode_%d'%i,e); self.add_module('feed_%d'%i,f)
                self.Encoder.append(e); self.FeedForward.append(f)
            self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*self.modal_num,self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
        def forward(self, x):
            bs=x.size(0); x, input_attention=self.InputLayer(x); attn=input_attention.mean(dim=1)
            for index in range(self.n_layer):
                x, layer_attention=self.Encoder[index](q=x,k=x,v=x,modal_num=self.modal_num)
                attn=layer_attention.mean(dim=1); x=self.FeedForward[index](x)
            return self.Outputlayer(x.view(bs,-1),attn.view(bs,-1))
    return PLNNoPEN()

def build(layers, hp=None):
    """Construct the private PEN-free CAM used by all CU-GL folds."""
    return make_pln(layers, hp or make_hp())

def metric_row(logp,y,test):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    pred=logp.detach().cpu().numpy()[test].argmax(1); target=y[test]
    tp,fn,fp,tn=map(int,confusion_matrix(target,pred,labels=[0,1]).ravel())
    return dict(acc=float((pred==target).mean()),hard_auc=float(roc_auc_score(np.eye(2)[target],np.eye(2)[pred])),
      spe=float(tn/(tn+fp)),sen=float(tp/(tp+fn)),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)

def model_state(module):
    import torch
    return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in module.items()}

def state_hash(state):
    h=hashlib.sha256()
    for key in sorted(state):
        for name in sorted(state[key]): h.update((key+'.'+name).encode()+b'\0'+state[key][name].numpy().tobytes())
    return h.hexdigest()

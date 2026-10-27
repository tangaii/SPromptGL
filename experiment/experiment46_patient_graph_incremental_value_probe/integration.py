"""Read-only data, frozen CAM and candidate graph integration for PGIV."""
from __future__ import annotations
import hashlib, json, os, random, shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path('/data/adodas/SPromptGL-main'); EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'
DIMS=(48,6,10,256); MODALITIES=('PHENO','ANAT','FUNC','CORR'); SLICES=((0,48),(48,54),(54,64),(64,320))

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def write_json(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(tmp,path)

def read_data():
 import pandas as pd
 from sklearn.model_selection import StratifiedKFold
 base=ROOT/'data/ABIDE'; frame=pd.read_csv(base/'processed_standard_data.csv'); mapping=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
 assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']; assert [len(v) for v in mapping.values()]==list(DIMS); assert frame.shape==(871,321)
 x=frame.iloc[:,:-1].to_numpy(np.float32); y=frame.iloc[:,-1].to_numpy(np.int64)-1
 assert np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
 folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); return x,y,folds,mapping

def slice_for_modality(x,m):
 lo,hi=SLICES[MODALITIES.index(m) if isinstance(m,str) else int(m)]; return x[...,lo:hi]

def fold_sha(train,test): return hashlib.sha256(np.asarray(train,np.int64).tobytes()+np.asarray(test,np.int64).tobytes()).hexdigest()

def fit_norm(x,idx):
 z=np.asarray(x,np.float64); idx=np.asarray(idx,np.int64); mean=np.zeros(320); scale=np.ones(320)
 for lo,hi in SLICES:
  mean[lo:hi]=z[idx,lo:hi].mean(0); scale[lo:hi]=z[idx,lo:hi].std(0); scale[lo:hi][scale[lo:hi]<1e-8]=1.
 return mean,scale

def apply_norm(x,stats):
 z=np.asarray(x,np.float64).copy(); mean,scale=stats
 for lo,hi in SLICES:
  z[:,lo:hi]=(z[:,lo:hi]-mean[lo:hi])/scale[lo:hi]; n=np.linalg.norm(z[:,lo:hi],axis=1,keepdims=True); n[n<1e-8]=1.; z[:,lo:hi]/=n
 return z.astype(np.float32)

def candidate_pool(x,train,test,k=16):
 """A[receiver,sender], sender restricted to outer-train, train-fitted stats."""
 train=np.asarray(train,np.int64); receivers=np.concatenate([train,np.asarray(test,np.int64)]); stats=fit_norm(x,train); z=apply_norm(x,stats); out=np.full((len(x),4,k),-1,np.int64)
 for m,(lo,hi) in enumerate(SLICES):
  sim=z[receivers,lo:hi]@z[train,lo:hi].T
  for r,i in enumerate(receivers):
   allowed=train[train!=i]; vals=sim[r,train!=i]; order=np.lexsort((allowed,-vals))[:k]; out[i,m,:len(order)]=allowed[order]
 return out,stats

def random_candidates(train,receivers,fold,k=16):
 train=np.asarray(train,np.int64); receivers=np.asarray(receivers,np.int64); out=np.full((len(receivers),4,k),-1,np.int64)
 for r,i in enumerate(receivers):
  for m in range(4):
   pool=train[train!=i]; rng=np.random.default_rng(460000+fold*100000+int(i)*4+m); out[i,m]=rng.choice(pool,size=k,replace=False)
 return out

def permuted_candidates(true_cand,train,fold):
 train=np.asarray(train,np.int64); order=np.random.default_rng(460111+fold).permutation(train); mapping={int(order[i]):int(order[(i+1)%len(order)]) for i in range(len(order))}; out=true_cand.copy()
 for i in range(len(out)):
  for m in range(4):
   for s,j in enumerate(out[i,m]):
    if j>=0: out[i,m,s]=mapping[int(j)]
 return out

def make_hp():
 return SimpleNamespace(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2',seed=0)

def make_cam(layers,hp=None):
 import torch.nn as nn
 hp=hp or make_hp()
 class PhysicalCAMOnly(nn.Module):
  def __init__(self):
   super().__init__(); self.hyperpm=hp; self.input_data_dims=list(DIMS); self.d_q=self.d_k=self.d_v=hp.n_hidden; self.n_head=hp.n_head; self.dropout=hp.dropout; self.n_layer=hp.nlayer; self.modal_num=hp.nmodal; self.n_class=hp.nclass; self.d_out=self.d_v*self.n_head*self.modal_num; self.fusion_weight=nn.Parameter(torch.ones(6,1)); self.InputLayer=layers.VariLengthInputLayer(list(DIMS),self.d_k,self.d_v,self.n_head,self.dropout); self.Encoder=[]; self.FeedForward=[]
   for i in range(self.n_layer):
    e=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout); f=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout); self.add_module('encode_%d'%i,e); self.add_module('feed_%d'%i,f); self.Encoder.append(e); self.FeedForward.append(f)
   self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*self.modal_num,self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
  def forward(self,x):
   bs=x.size(0); x,a0=self.InputLayer(x); a=a0.mean(dim=1)
   for i in range(self.n_layer): x,al=self.Encoder[i](q=x,k=x,v=x,modal_num=self.modal_num); a=al.mean(dim=1); x=self.FeedForward[i](x)
   return self.Outputlayer(x.view(bs,-1),a.view(bs,-1))
 return PhysicalCAMOnly()

def model_state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def state_hash(s):
 h=hashlib.sha256()
 for k in sorted(s): h.update(k.encode()+b'\0'+s[k].numpy().tobytes())
 return h.hexdigest()
def rng_capture():
 import torch
 return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':[z.cpu() for z in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None}
def rng_restore(r):
 import torch
 random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['cpu']);
 if torch.cuda.is_available() and r.get('cuda') is not None: torch.cuda.set_rng_state_all(r['cuda'])
def seed_all(seed):
 import torch
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def metric(logp,y,test):
 from sklearn.metrics import confusion_matrix,roc_auc_score
 arr=logp.detach().cpu().numpy() if hasattr(logp,'detach') else np.asarray(logp)
 pred=arr[np.asarray(test)].argmax(1); tar=y[np.asarray(test)]; tp,fn,fp,tn=map(int,confusion_matrix(tar,pred,labels=[0,1]).ravel()); return {'acc':float((pred==tar).mean()),'hard_auc':float(roc_auc_score(np.eye(2)[tar],np.eye(2)[pred])),'spe':float(tn/(tn+fp)),'sen':float(tp/(tp+fn)),'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,'prediction':pred.astype(int).tolist(),'test_indices':np.asarray(test).astype(int).tolist()}

def site_labels(mapping,x):
 names=list(mapping['PHENO']); idx=[i for i,n in enumerate(names) if str(n).startswith('SITE_ID_')]
 return np.argmax(x[:,idx],axis=1) if idx else np.full(len(x),-1),[names[i] for i in idx]

"""Independent E42 mechanism operators. No historical Python runtime imports."""
from __future__ import annotations
import hashlib,json,os,pickle,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_auc_score

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hjson(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def rng(): return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(v):
 random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in v['cuda']])
def rngsha(v):
 h=hashlib.sha256(pickle.dumps((v['python'],v['numpy']),protocol=4))
 for t in [v['cpu']]+v['cuda']:h.update(t.cpu().numpy().tobytes())
 return h.hexdigest()
def trunksha(agent):
 h=hashlib.sha256()
 for k,m in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion)]:
  for n,p in sorted(m.named_parameters()):h.update(k.encode()+b'/'+n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def reference(x,y,train,test,edges,fold,assignment=None):
 x=np.asarray(x,np.float64);train=np.asarray(train);test=np.asarray(test);cov=x[:,:48];fc=x[:,64+np.asarray(edges)]
 if assignment is None:
  assignment=np.full(len(x),-1,dtype=np.int64)
  for k,(_,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=40000+fold).split(cov[train],y[train])):assignment[train[val]]=k
 mu=np.zeros_like(fc);fits=[];models=[]
 for k in range(5):
  val=train[assignment[train]==k];fit=train[(assignment[train]!=k)&(y[train]==1)]
  assert len(fit)>15 and not np.intersect1d(val,fit).size
  r=Ridge(alpha=10,solver='svd').fit(cov[fit],fc[fit]);mu[val]=r.predict(cov[val]);fits.append(fit);models.append((r.coef_,r.intercept_))
 td=train[y[train]==1];r=Ridge(alpha=10,solver='svd').fit(cov[td],fc[td]);mu[test]=r.predict(cov[test]);res=fc[td]-mu[td]
 scale=np.maximum(1.4826*np.median(abs(res-np.median(res,axis=0)),axis=0),1e-4)
 z=np.clip((fc-mu)/scale,-3,3)
 assert np.isfinite(z).all()
 return dict(mu=mu.astype('float32'),sigma=scale.astype('float32'),z=z.astype('float32'),assignment=assignment,fits=fits,final_fit=td,models=models+[(r.coef_,r.intercept_)])
def metrics(y,p):
 tp,fn,fp,tn=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
 return dict(acc=float(np.mean(y==p)),hard_auc=float(roc_auc_score(np.eye(2)[y],np.eye(2)[p])),spe_td=tn/(tn+fp),sen_asd=tp/(tp+fn),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
def smd(a,b):return (np.mean(a,axis=0)-np.mean(b,axis=0))/np.maximum(np.sqrt((np.var(a,axis=0)+np.var(b,axis=0))/2),1e-6)
def partial(values,cov,train):
 r=Ridge(alpha=10,solver='svd').fit(cov[train],values[train]);return values-r.predict(cov)
def wrong_groups(hyp):
 ids=list(range(13))
 for size in (2,3,4):
  ii=[i for i,h in enumerate(hyp) if len(h['edges'])==size]
  for a,b in zip(ii,ii[1:]+ii[:1]):ids[a]=b
 assert all(i!=j for i,j in enumerate(ids));return ids

def text_weights(texts,anatomy,hyp,edges,variant='REAL'):
 """Query-to-edge attention; canonical numeric columns never move."""
 t=norm(np.asarray(texts));a=norm(np.asarray(anatomy));groups=wrong_groups(hyp) if variant=='DERANGED_HYPOTHESIS' else list(range(13))
 if variant=='SWAP_ASD_TD':t=t[:,::-1].copy()
 if variant=='GRAM_MATCHED_ANONYMOUS':
  g=np.random.default_rng(420008);t=t[...,g.permutation(t.shape[-1])]*g.choice([-1.,1.],t.shape[-1])
 weights=np.zeros((13,2,39),np.float64)
 for h,g in enumerate(groups):
  jj=[edges.index(e) for e in hyp[g]['edges']];sim=t[h]@a[jj].T/.1
  if variant in ('NUMERIC_ONLY','STRUCTURED_DIAGNOSTIC'):sim=np.zeros_like(sim)
  att=np.exp(sim-sim.max(1,keepdims=True));att/=att.sum(1,keepdims=True);weights[h][:,jj]=att
 return weights.astype('float32')
def probe_features(z,weights,hyp,edges,structured=True):
 # Same shape/31 trainable parameters for every family and control.
 d=np.zeros(39)
 for hi,h in enumerate(hyp):
  for j in np.flatnonzero(weights[hi].sum(0)>0):d[j]={'hyper_candidate':1,'hypo_candidate':-1}.get(h['direction'],0)
 candidate=np.where(d[None]!=0,z*d,np.abs(z)) if structured else z
 asd=np.einsum('ne,he->nh',candidate,weights[:,0]);td=np.einsum('ne,he->nh',-abs(z),weights[:,1])
 common=np.stack([z.mean(1),abs(z).mean(1),(z*z).mean(1),np.mean(candidate,1)],1)
 return np.concatenate([asd,td,common],1).astype('float32')
def probe_fit(a,y,b,seed,steps=150):
 center=a.mean(0);scale=np.maximum(a.std(0),.1);aa=torch.tensor((a-center)/scale);bb=torch.tensor((b-center)/scale);yy=torch.tensor(y,dtype=torch.float32)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);model=nn.Linear(30,1);nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
  opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.01)
  for step in range(steps):
   opt.zero_grad();F.binary_cross_entropy_with_logits(model(aa).squeeze(-1),1-yy).backward();opt.step()
  pred=(model(bb).detach().squeeze(-1).numpy()<0).astype(int)
  w=model.weight.detach().numpy()[0]/scale;bias=float(model.bias.detach().item()-w@center)
 return pred,w,bias

def edge_features(z,x,edges,direction,discrimination):
 # 7 numeric/context entries plus 39 fixed identity slots = 46 features.
 n=len(z);eye=np.broadcast_to(np.eye(39,dtype='float32'),(n,39,39))
 num=np.stack([z,abs(z),z*z,z*direction[None],abs(z)*abs(discrimination)[None],x[:,64+np.asarray(edges)],np.ones_like(z)],-1)
 return np.concatenate([num,eye],-1).astype('float32')
class Selector(nn.Module):
 def __init__(self,dim=46):
  super().__init__();self.linear=nn.Linear(dim,1)
  nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
 def forward(self,x):return self.linear(x).squeeze(-1)
def topmask(score,k=8,ste=False,temperature=.5):
 hard=torch.zeros_like(score).scatter_(-1,score.topk(k,dim=-1).indices,1.)
 if not ste:return hard
 cut=score.topk(k,dim=-1).values[...,-1:].detach();soft=torch.sigmoid((score-cut)/temperature)
 return hard+(soft-soft.detach())
def intervene(raw,person,cols,mu,mask,keep=False,strength=1.):
 """Only one patient's canonical raw FC; straight-through derivative, exact hard values."""
 out=raw.clone();factor=mask if keep else 1-mask
 smooth=raw[person,cols]+strength*(mu[person]-raw[person,cols])*(1-factor)
 hard=torch.where(factor.detach()>.5,raw[person,cols],raw[person,cols]+strength*(mu[person]-raw[person,cols]))
 out[person,cols]=hard+(smooth-smooth.detach());return out

def prepare_sources(root,out):
 """Irreplaceable tiny patch inside mechanism.py: bypass PEN, optionally freeze graphs."""
 import ast,difflib
 out=Path(out);out.mkdir(parents=True,exist_ok=True);patch=[]
 for name in ('network.py','model.py','layers.py','utils.py'):
  old=(Path(root)/name).read_text();s=old
  if name=='network.py':
   a=s.index('        text_features = self.prompt_text_features\n');b=s.index('        x = x.view(bs, -1)\n',a);s=s[:a]+s[b:]
   for k in range(4):
    marker=f'                adj = self.GraphConstruct{k}(x[i])\n';assert s.count(marker)==1
    s=s.replace(marker,marker+f"                if getattr(self, 'fixed_graph42', None) is not None:\n                    adj = self.fixed_graph42[{k}]\n",1)
  ast.parse(s);p=out/name
  if p.exists():assert p.read_text()==s
  else:p.write_text(s)
  patch.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='root/'+name,tofile='runtime/'+name))
 link=out/'CLIP'
 if not link.exists():link.symlink_to(Path(root)/'CLIP',target_is_directory=True)
 (out/'integration.patch').write_text(''.join(patch))

from __future__ import annotations
from types import SimpleNamespace
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
DIMS=[48,6,10,256]; SLICES=[(0,48),(48,54),(54,64),(64,320)]
class HP(SimpleNamespace):
 def __init__(self): super().__init__(n_hidden=18,n_head=2,nlayer=1,nmodal=4,nclass=2,dropout=.35,th=.9,GC_mode='Sigmoid-like',MP_mode='GCN',MF_mode='CAM',alpha=1)
class PLN(nn.Module):
 def __init__(self,layers,h):
  super().__init__();self.hyperpm=h;self.fusion_weight=nn.Parameter(torch.ones([6,1]));self.InputLayer=layers.VariLengthInputLayer(DIMS,h.n_hidden,h.n_hidden,h.n_head,h.dropout);self.Encoder=[];self.FeedForward=[]
  for i in range(h.nlayer):
   e=layers.EncodeLayer(h.n_hidden*h.n_head,h.n_hidden,h.n_hidden,h.n_head,h.dropout);f=layers.FeedForwardLayer(h.n_hidden*h.n_head,h.n_hidden*h.n_head,dropout=h.dropout);self.add_module('encode_'+str(i),e);self.add_module('feed_'+str(i),f);self.Encoder.append(e);self.FeedForward.append(f)
  self.Outputlayer=layers.OutputLayer(h.n_hidden*h.n_head*4,h.n_hidden*h.n_head,2,4,h.dropout)
 def forward(self,x):
  bs=x.size(0);x,att=self.InputLayer(x);att=att.mean(1)
  for i in range(self.hyperpm.nlayer):x,_=self.Encoder[i](q=x,k=x,v=x,modal_num=4);x=self.FeedForward[i](x)
  out,hid=self.Outputlayer(x.view(bs,-1),att.view(bs,-1));return out,hid
class SymScorer(nn.Module):
 def __init__(self,d=32):super().__init__();self.net=nn.Sequential(nn.Linear(2*d+1,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,zi,zj,s):return self.net(torch.cat([(zi-zj).abs(),zi*zj,s.unsqueeze(-1)],-1)).squeeze(-1)
class APCR(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.ModuleList([nn.Linear(d,32,bias=False) for d in DIMS]);self.scorer=nn.ModuleList([SymScorer() for _ in DIMS]);self.msg=nn.ModuleList([nn.Sequential(nn.Linear(320,64,bias=False),nn.GELU(),nn.Linear(64,320,bias=False)) for _ in DIMS]);self.gate=nn.ModuleList([nn.Sequential(nn.Linear(2,8),nn.Sigmoid(),nn.Linear(8,1)) for _ in DIMS]);self.alpha_raw=nn.Parameter(torch.tensor(-4.595));self.last={}
 def alpha(self):return .2*torch.sigmoid(self.alpha_raw)
 def representations(self,parts):return [F.normalize(self.proj[m](p),p=2,dim=-1,eps=1e-8) for m,p in enumerate(parts)]
 def candidates(self,z):
  edges=[];sim=[]
  for zz in z:
   s=zz@zz.T;s.fill_diagonal_(-1e9);v,ix=torch.topk(s,24,1);edges.append(ix);sim.append(v)
  return edges,sim
 def q_edges(self,z,edges,sim):return [torch.sigmoid(self.scorer[m](z[m].unsqueeze(1),z[m][edges[m]],sim[m])) for m in range(4)]
 def forward(self,parts,group=3,wrong=None,attack_edges=None,feature_only=False):
  x=torch.cat(parts,1);z=self.representations(parts);edges,sim=self.candidates(z);q=self.q_edges(z,edges,sim);chosen=[];deltas=[];gates=[]
  for m in range(4):
   order=q[m].topk(16,1).indices;sel=edges[m].gather(1,order);w=q[m].gather(1,order)
   if wrong is not None:sel=wrong[sel]
   if attack_edges is not None: sel=attack_edges[m]
   chosen.append(sel)
   if feature_only: diff=torch.zeros_like(x[:,None,:]);
   else: diff=x[sel]-x[:,None,:]
   msg=(diff*(w/(w.sum(1,keepdim=True)+1e-8)).unsqueeze(-1)).sum(1);delta=self.msg[m](msg);rel=q[m].mean(1);conf=q[m].std(1);g=torch.sigmoid(self.gate[m](torch.stack([rel,conf],1)).squeeze(-1));deltas.append(delta);gates.append(g)
  delta=sum(g.unsqueeze(-1)*d for g,d in zip(gates,deltas));fused=x+self.alpha()*delta if group>=3 and not feature_only else (x+.01*torch.tanh(x) if group==2 else x)
  self.last={'edges':edges,'chosen':chosen,'q':q,'deltas':deltas,'gates':gates,'alpha':float(self.alpha().detach())};return fused,z,q,chosen
 def srd_loss(self,z,parts,labels,tr,shuffle=False):
  ids=torch.as_tensor(tr,device=labels.device);n=len(ids);g=torch.Generator(device=labels.device);g.manual_seed(9001+n)
  # balanced same ASD, same TD, cross pairs; no truncation bias
  groups=[ids[labels[ids]==c] for c in (0,1)];pairs=[]
  for c in (0,1):
   a=groups[c];perm=torch.randperm(len(a),generator=g,device=a.device);pairs.append(torch.stack([a,a[perm]],1))
  a,b=groups;na=min(len(a),len(b));pa=torch.randperm(len(a),generator=g,device=a.device)[:na];pb=torch.randperm(len(b),generator=g,device=b.device)[:na];pairs.append(torch.stack([a[pa],b[pb]],1));pair=torch.cat(pairs);yy=torch.cat([torch.ones(len(pairs[0])+len(pairs[1]),device=labels.device),torch.zeros(len(pairs[2]),device=labels.device)])
  if shuffle:yy=yy.roll(17)
  losses=[];aucs=[]
  for m in range(4):
   zi=z[m][pair[:,0]];zj=z[m][pair[:,1]];s=(zi*zj).sum(1);log=self.scorer[m](zi,zj,s);losses.append(F.binary_cross_entropy_with_logits(log,yy));aucs.append(float(roc_auc_score(yy.detach().cpu(),torch.sigmoid(log).detach().cpu())))
  return torch.stack(losses).mean(),aucs,len(pair),len(torch.unique(pair))
 def sca_loss(self,q):
  vals=[];weights=[]
  for m in range(4):
   for n in range(m+1,4):
    a,b=q[m],q[n];w=(a.detach()*b.detach()*(1-(a.detach()-b.detach()).abs())).clamp(0,1);vals.append(((a-b).pow(2)*w).sum());weights.append(w.sum())
  den=torch.stack(weights).sum()+1e-8;return torch.stack(vals).sum()/den,float(torch.stack(weights).sum().detach())
def build(torch_mod,layers,network,utils,device,x,y,tr,te):
 h=HP();g=network.GFR(DIMS,DIMS,th=.9).to(device);p=PLN(layers,h).to(device);ap=APCR().to(device);feat=torch_mod.from_numpy(x).float().to(device);targ=torch_mod.from_numpy(y).long().to(device);ids=torch_mod.as_tensor(tr,device=device);cnt=torch.bincount(targ[ids],minlength=2).float();weight=(len(tr)/cnt/2).to(device);return SimpleNamespace(dev=device,hyperpm=h,feat=feat,targ=targ,trn_idx=tr,tst_idx=te,val_idx=te,weight=weight,dataLayer=layers.InitialDataLayer(DIMS),MessagePassing0=g,ModalFusion=p,APCR=ap)
def make_agent(torch_mod,layers,network,utils,device,x,y,tr,te,fold,gid,anchor,seed):
 a=build(torch_mod,layers,network,utils,device,x,y,tr,te);a.MessagePassing0.load_state_dict(anchor['model']['gfr']);a.ModalFusion.load_state_dict(anchor['model']['pln']);a.APCR.apply(utils.my_weight_init);a.base_state={k:v.detach().clone() for k,v in a.MessagePassing0.state_dict().items()};a.optimizer_MP0=torch_mod.optim.Adam(a.MessagePassing0.parameters(),lr=.0038,weight_decay=.11);a.optimizer_MF=torch_mod.optim.Adam(a.ModalFusion.parameters(),lr=.0038,weight_decay=.11);a.optimizer_MP0.load_state_dict(anchor['mp']);a.optimizer_MF.load_state_dict(anchor['mf']);a.optimizer_new=torch_mod.optim.Adam(a.APCR.parameters(),lr=.00038,weight_decay=.0001);return a
def forward(a,group,wrong=None,attack_edges=None):
 _,parts=a.dataLayer(a.feat);base,adjs,aloss=a.MessagePassing0(parts,4,a.targ,a.trn_idx);tmp,z,q,ch=a.APCR(parts,group,wrong,attack_edges,feature_only=(group==2));x=torch.cat(parts,1);fused=base+(tmp-x) if group>=2 else base;lp,h=a.ModalFusion(fused);return parts,base,fused,z,q,ch,lp,aloss,adjs
def train_step(a,group,epoch,utils,wrong=None,shuffle=False,random_attack=False):
 a.MessagePassing0.train();a.ModalFusion.train();a.APCR.train();a.optimizer_MP0.zero_grad(set_to_none=True);a.optimizer_MF.zero_grad(set_to_none=True);a.optimizer_new.zero_grad(set_to_none=True);parts,base,fused,z,q,ch,lp,al,adjs=forward(a,group,wrong);tr=torch.as_tensor(a.trn_idx,device=lp.device);main=utils.ClsLoss(lp,a.targ,a.trn_idx,a.weight);srd=torch.zeros((),device=lp.device);sca=torch.zeros((),device=lp.device);rob=torch.zeros((),device=lp.device);auc=[.5]*4;npair=0;cover=0
 if group>=4:srd,auc,npair,cover=a.APCR.srd_loss(z,parts,a.targ,a.trn_idx,shuffle)
 if group>=5:sca,w=a.APCR.sca_loss(q)
 if group>=6:
  # Two legal alternatives per training node; choose the larger CE among clean and two candidate replacements.
  attack=[]
  for m in range(4):
   sel=ch[m].clone();pool=a.APCR.last['edges'][m];alt=pool[:,0].clone();alt2=pool[:,1].clone();sel[:, -1]=alt;sel[:, -2]=alt2;attack.append(sel)
  _,_,adv,_,_,_,adv_lp,_,_=forward(a,group,wrong,attack)
  rob=utils.ClsLoss(adv_lp,a.targ,a.trn_idx,a.weight)+.05*F.kl_div(adv_lp[tr],lp.detach()[tr],log_target=True,reduction='batchmean')
 graph=sum(utils.GraphConstructLoss(parts[i],adjs[i],1.,.5,0.) for i in range(4))/4;loss=main+graph+.05*srd+.02*sca+.05*rob;loss.backward();torch.nn.utils.clip_grad_norm_(list(a.ModalFusion.parameters())+list(a.APCR.parameters())+list(a.MessagePassing0.parameters()),5.);a.optimizer_MP0.step();a.optimizer_MF.step();a.optimizer_new.step();return {'loss':float(loss.detach()),'main':float(main.detach()),'srd':float(srd.detach()),'sca':float(sca.detach()),'robust':float(rob.detach()),'pair_auc':auc,'pair_count':npair,'pair_coverage':cover,'alpha':float(a.APCR.alpha().detach()),'message_norms':[float(v.detach().norm()) for v in a.APCR.last['deltas']],'gate_mean':[float(v.detach().mean()) for v in a.APCR.last['gates']]}
@torch.no_grad()
def evaluate(a,group,utils,wrong=None):
 a.MessagePassing0.eval();a.ModalFusion.eval();a.APCR.eval();parts,base,fused,z,q,ch,lp,al,adjs=forward(a,group,wrong);return {'logp':lp,'base':base,'fused':fused,'parts':parts,'q':q,'chosen':ch,'deltas':a.APCR.last['deltas'],'gates':a.APCR.last['gates'],'alpha':a.APCR.alpha()}
def model_bundle(a):return {'gfr':{k:v.detach().cpu().clone() for k,v in a.MessagePassing0.state_dict().items()},'pln':{k:v.detach().cpu().clone() for k,v in a.ModalFusion.state_dict().items()},'apcr':{k:v.detach().cpu().clone() for k,v in a.APCR.state_dict().items()}}
def load_bundle(a,s):a.MessagePassing0.load_state_dict(s['gfr']);a.ModalFusion.load_state_dict(s['pln']);a.APCR.load_state_dict(s['apcr'])
def metrics(o,y,te):
 from sklearn.metrics import confusion_matrix
 p=o['logp'].detach().cpu().numpy()[te].argmax(1);tar=y[te];tp,fn,fp,tn=map(int,confusion_matrix(tar,p,labels=[0,1]).ravel());one=np.eye(2);return {'acc':float((p==tar).mean()),'hard_auc':float(roc_auc_score(one[tar],one[p])),'spe':float(tn/(tn+fp)),'sen':float(tp/(tp+fn)),'probability_auc':float(roc_auc_score(tar,o['logp'].detach().cpu().numpy()[te,1])),'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn}

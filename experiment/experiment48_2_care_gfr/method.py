from __future__ import annotations
import copy, math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
INPUT_DIMS=[48,6,10,256]; MODALITIES=['PHENO','ANAT','FUNC','CORR']

def hp(): return SimpleNamespace(n_hidden=18,n_head=2,dropout=.35,nlayer=1,lr=.0038,reg=.11,th=.9)
class PLNNoPEN(nn.Module):
 def __init__(self,layers,h):
  super().__init__();self.hyperpm=h;self.fusion_weight=nn.Parameter(torch.ones([6,1]));self.InputLayer=layers.VariLengthInputLayer(INPUT_DIMS,h.n_hidden,h.n_hidden,h.n_head,h.dropout);self.Encoder=[];self.FeedForward=[]
  for i in range(h.nlayer):
   e=layers.EncodeLayer(h.n_hidden*h.n_head,h.n_hidden,h.n_hidden,h.n_head,h.dropout);f=layers.FeedForwardLayer(h.n_hidden*h.n_head,h.n_hidden*h.n_head,dropout=h.dropout);self.add_module(f'encode_{i}',e);self.add_module(f'feed_{i}',f);self.Encoder.append(e);self.FeedForward.append(f)
  self.Outputlayer=layers.OutputLayer(h.n_hidden*h.n_head*4,h.n_hidden*h.n_head,2,4,h.dropout)
 def forward(self,x):
  bs=x.size(0);x,_=self.InputLayer(x)
  for i in range(self.hyperpm.nlayer):x,_=self.Encoder[i](q=x,k=x,v=x,modal_num=4);x=self.FeedForward[i](x)
  hid=self.Outputlayer.mlp_head(x.view(bs,-1));raw=self.Outputlayer.classifier(hid);return F.log_softmax(raw,1),raw,hid
class GRL(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,c):ctx.c=float(c);return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.c*g,None
def grl(x,c):return GRL.apply(x,float(c))
class RelationScorer(nn.Module):
 def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(3*d+1,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,xi,xj,s):return self.net(torch.cat([xi,xj,xi-xj,s.unsqueeze(-1)],-1)).squeeze(-1)
class CareGraph(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.ModuleList([nn.Linear(d,32) for d in INPUT_DIMS]);self.scorers=nn.ModuleList([RelationScorer(d) for d in INPUT_DIMS]);self.msg_in=nn.ModuleList([nn.Linear(320,64) for _ in range(4)]);self.msg_out=nn.ModuleList([nn.Linear(64,320) for _ in range(4)]);self.gate=nn.ModuleList([nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,1)) for _ in range(4)]);self.rel_head=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,2));self.domain=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,2));self.last={}
 def raw_candidates(self,parts):
  edges=[];raw=[]
  for m,x in enumerate(parts):
   z=F.normalize(self.proj[m](x),-1);sim=z@z.T;sim.fill_diagonal_(-1e9);v,ix=torch.topk(sim,24,1);edges.append(ix);raw.append(v)
  return edges,raw
 def edge_q(self,x,m,ii,jj,s):return self.scorers[m](x[ii],x[jj],s)
 def forward(self,parts,labels,tr,group,wrong_perm=None,attack=False,random_attack=False):
  N=parts[0].size(0);x=torch.cat(parts,1);edges,raw=self.raw_candidates(parts); qs=[];chosen=[];msgs=[];gates=[]
  trmask=torch.zeros(N,dtype=torch.bool,device=x.device);trmask[torch.as_tensor(tr,device=x.device)] = True
  for m in range(4):
   ii=torch.arange(N,device=x.device).unsqueeze(1).expand(-1,24).reshape(-1);jj=edges[m].reshape(-1);qlog=self.edge_q(parts[m],m,ii,jj,raw[m].reshape(-1)).view(N,24);q=torch.sigmoid(qlog);qs.append(q)
   order=torch.topk(q,16,1).indices;sel=torch.gather(edges[m],1,order);w=torch.gather(q,1,order)
   if wrong_perm is not None:sel=wrong_perm[sel]
   if attack or random_attack:
    # Constrained inner edge replacement. The former implementation
    # synchronised the GPU once per candidate through ``float(tensor)``;
    # this equivalent index-only operation computes all candidate scores in
    # one tensor pass and keeps the same candidate/degree/budget semantics.
    pool=edges[m]
    used=(pool.unsqueeze(2)==sel.unsqueeze(1)).any(-1)
    self_edge=pool.eq(torch.arange(N,device=x.device).unsqueeze(1))
    valid=(~used)&(~self_edge)
    if random_attack:
     # Deterministic row-wise choice matching the former candidate order.
     vcpu=valid[trmask].detach().cpu().numpy(); ids=trmask.nonzero(as_tuple=False).flatten().tolist();pickpos=[]
     for row,i0 in zip(vcpu,ids):
      pos=np.flatnonzero(row);pickpos.append(int(pos[(i0+m*17)%len(pos)]) if len(pos) else 0)
     if ids:
      ii=torch.as_tensor(ids,device=x.device,dtype=torch.long);pp=torch.as_tensor(pickpos,device=x.device,dtype=torch.long);picks=pool[ii,pp];sel[ii,-2:]=picks.unsqueeze(1)
    else:
     with torch.no_grad():
      vals=(parts[m][pool]-parts[m].unsqueeze(1)).pow(2).mean(-1).masked_fill(~valid,-float('inf'));picks=pool.gather(1,vals.argmax(1,keepdim=True)).squeeze(1)
     ii=trmask.nonzero(as_tuple=False).flatten();
     if len(ii):sel[ii,-2:]=picks[ii].unsqueeze(1)
   diff=x[sel]-x.unsqueeze(1);wm=w/(w.sum(1,keepdim=True)+1e-6);msg=(diff*wm.unsqueeze(-1)).sum(1);delta=self.msg_out[m](F.gelu(self.msg_in[m](msg)));relq=q.mean(1);conf=q.std(1);gg=torch.sigmoid(self.gate[m](torch.stack([relq,conf],1))).squeeze(-1);msgs.append(delta);gates.append(gg)
   chosen.append(sel)
  fused=x+sum(g.unsqueeze(-1)*d for g,d in zip(gates,msgs));self.last={'edges':edges,'chosen':chosen,'raw':raw,'q':qs,'msgs':msgs,'gates':gates,'fused':fused};return fused,qs,chosen
 def pair_loss(self,parts,labels,tr,shuffle=False):
  # balanced deterministic edge labels; only outer-train nodes are supervised
  idx=torch.as_tensor(tr,device=labels.device);ii,jj=torch.meshgrid(idx,idx,indexing='ij');mask=ii<jj;ii=ii[mask];jj=jj[mask]
  if len(ii)>2048:ii=ii[:2048];jj=jj[:2048]
  y=(labels[ii]==labels[jj]).float();
  if shuffle:y=y[torch.arange(len(y),device=y.device).roll(137)]
  losses=[];aucs=[]
  for m,x in enumerate(parts):
   z=F.normalize(self.proj[m](x),-1);s=(z[ii]*z[jj]).sum(1);log=self.edge_q(x,m,ii,jj,s);losses.append(F.binary_cross_entropy_with_logits(log,y));
   with torch.no_grad():
    try:aucs.append(float(roc_auc_score(y.cpu(),torch.sigmoid(log).cpu())))
    except:aucs.append(.5)
  return torch.stack(losses).mean(),aucs
 def consensus(self,parts,labels,tr,enabled=True):
  if not enabled:return parts[0].new_zeros(())
  idx=torch.as_tensor(tr[:min(len(tr),256)],device=labels.device);ii,jj=torch.meshgrid(idx,idx,indexing='ij');mask=ii<jj;ii=ii[mask];jj=jj[mask];vals=[]
  for m,x in enumerate(parts):
   z=F.normalize(self.proj[m](x),-1);s=(z[ii]*z[jj]).sum(1);vals.append(self.edge_q(x,m,ii,jj,s))
  q=torch.stack(vals);w=torch.sigmoid(q.detach()).mean(0);return ((q-q.mean(0,keepdim=True)).pow(2).mean(0)*w).sum()/(w.sum()+1e-6)

def build_base(torch_mod,layers,network,utils,device,x,y,tr,te):
 h=hp();g=CareGraph().to(device);p=PLNNoPEN(layers,h).to(device);p.apply(utils.my_weight_init);feat=torch_mod.from_numpy(x).float().to(device);targ=torch_mod.from_numpy(y).long().to(device);idx=torch_mod.as_tensor(tr,device=device);cnt=torch.bincount(targ[idx],minlength=2).float();weight=(len(tr)/cnt/2).to(device);return SimpleNamespace(dev=device,hyperpm=h,feat=feat,targ=targ,trn_idx=tr,tst_idx=te,val_idx=te,weight=weight,dataLayer=layers.InitialDataLayer(INPUT_DIMS),MessagePassing0=g,ModalFusion=p,fold=0)
def base_state(a):return {'g':{k:v.detach().cpu().clone() for k,v in a.MessagePassing0.state_dict().items()},'p':{k:v.detach().cpu().clone() for k,v in a.ModalFusion.state_dict().items()}}
def load_base(a,s):
 if 'gfr' in s:a.ModalFusion.load_state_dict(s['pln']); return
 a.MessagePassing0.load_state_dict(s['g']);a.ModalFusion.load_state_dict(s['p'])
def make_agent(torch_mod,layers,network,utils,device,x,y,tr,te,fold,gid,anchor,seed):
 a=build_base(torch_mod,layers,network,utils,device,x,y,tr,te);a.ModalFusion.load_state_dict(anchor['model']['pln']);cr=torch_mod.get_rng_state();cg=torch_mod.cuda.get_rng_state_all();torch_mod.manual_seed(seed+gid*1000);torch_mod.cuda.manual_seed_all(seed+gid*1000);a.MessagePassing0.apply(utils.my_weight_init);torch_mod.set_rng_state(cr);torch_mod.cuda.set_rng_state_all(cg);a.opt=torch_mod.optim.Adam([{'params':a.ModalFusion.parameters(),'lr':.0038,'weight_decay':.11},{'params':a.MessagePassing0.parameters(),'lr':.00038,'weight_decay':.0001}],lr=.00038);return a
def forward(a,group,attack=False,random_attack=False,wrong_perm=None):
 _,parts=a.dataLayer(a.feat);fused,qs,chosen=a.MessagePassing0(parts,a.targ,a.trn_idx,group,wrong_perm,attack,random_attack);lp,raw,hid=a.ModalFusion(fused);return parts,fused,qs,chosen,lp,raw,hid
def train_step(a,group,epoch,utils,cond_labels=None,wrong_perm=None):
 a.MessagePassing0.train();a.ModalFusion.train();a.opt.zero_grad(set_to_none=True);parts,fused,qs,ch,lp,raw,hid=forward(a,group,attack=group in (4,9),random_attack=group==9,wrong_perm=wrong_perm);tr=torch.as_tensor(a.trn_idx,device=lp.device);main=utils.ClsLoss(lp,a.targ,a.trn_idx,a.weight);gl=a.feat.new_zeros(())
 pair,aucs=a.MessagePassing0.pair_loss(parts,a.targ if cond_labels is None else cond_labels,a.trn_idx,shuffle=group==8) if group>=2 else (a.feat.new_zeros(()),[.5]*4)
 cons=a.MessagePassing0.consensus(parts,a.targ,a.trn_idx,enabled=group in (3,4,6,7,8,9));clean_lp=lp.detach();rob=a.feat.new_zeros(())
 if group in (4,9):
  _,_,_,_,adv_lp,_,_=forward(a,group,attack=True,random_attack=group==9,wrong_perm=wrong_perm)
  rob=utils.ClsLoss(adv_lp,a.targ,a.trn_idx,a.weight)+.1*F.kl_div(adv_lp[tr],clean_lp[tr],log_target=True,reduction='batchmean')
 domain=a.feat.new_zeros(())
 if group==6:
  z=[qs[m].mean(1,keepdim=True).expand(-1,64) for m in range(4)];domain=F.cross_entropy(a.MessagePassing0.domain(torch.cat([z[3],z[2]],0)),torch.cat([torch.zeros(len(z[3]),dtype=torch.long,device=lp.device),torch.ones(len(z[2]),dtype=torch.long,device=lp.device)]))*.03
 loss=main+gl+.1*pair+.02*cons+.1*rob+domain;loss.backward();torch.nn.utils.clip_grad_norm_(list(a.MessagePassing0.parameters())+list(a.ModalFusion.parameters()),10.);a.opt.step();return {'loss':float(loss.detach()),'main':float(main.detach()),'pair':float(pair.detach()),'cons':float(cons.detach()),'robust':float(rob.detach()),'pair_auc':aucs,'gates':[float(g.detach().mean()) for g in a.MessagePassing0.last['gates']],'msg_norms':[float(d.detach().norm()) for d in a.MessagePassing0.last['msgs']]}
@torch.no_grad()
def evaluate(a,group,utils,wrong_perm=None):
 a.MessagePassing0.eval();a.ModalFusion.eval();parts,fused,qs,ch,lp,raw,hid=forward(a,group,wrong_perm=wrong_perm);return {'logp':lp,'parts':parts,'fused':fused,'q':qs,'chosen':ch,'msgs':a.MessagePassing0.last['msgs'],'gates':a.MessagePassing0.last['gates']}
def model_bundle(a):return {'g':{k:v.detach().cpu().clone() for k,v in a.MessagePassing0.state_dict().items()},'p':{k:v.detach().cpu().clone() for k,v in a.ModalFusion.state_dict().items()}}
def load_bundle(a,s):a.MessagePassing0.load_state_dict(s['g']);a.ModalFusion.load_state_dict(s['p'])
def metrics(o,y,te):
 p=o['logp'].detach().cpu().numpy()[te].argmax(1);tar=y[te];from sklearn.metrics import confusion_matrix
 tp,fn,fp,tn=map(int,confusion_matrix(tar,p,labels=[0,1]).ravel());one=np.eye(2);return {'acc':float((p==tar).mean()),'hard_auc':float(roc_auc_score(one[tar],one[p])),'spe':float(tn/(tn+fp)),'sen':float(tp/(tp+fn)),'probability_auc':float(roc_auc_score(tar,o['logp'].detach().cpu().numpy()[te,1])),'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn}

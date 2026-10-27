"""Reliability-calibrated CU-GL-R replacement GFR.

The branch is intentionally independent of the historical GFR graph loss. It
keeps the 320-dimensional patient representation and feeds a controlled,
actual cross-patient residual to the original CAM. Labels and identifiers never
enter the scoring networks at inference time.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F

SLICES=((0,48),(48,54),(54,64),(64,320))
DIMS=(48,6,10,256)

def inv_sigmoid(p): return math.log(p/(1-p))

@dataclass(frozen=True)
class RConfig:
    k:int=16; use_cf:bool=False; symmetric:bool=False; use_null:bool=True
    utility_weight:float=0.05; gate_weight:float=0.02; rank_weight:float=0.02

class ModalityRelation(nn.Module):
    def __init__(self, modality:int, full_dim=320, hidden=32):
        super().__init__(); self.modality=modality; self.lo,self.hi=SLICES[modality]; self.dim=self.hi-self.lo
        self.rx=nn.Linear(self.dim,hidden,bias=False); self.sx=nn.Linear(self.dim,hidden,bias=False)
        self.score=nn.Sequential(nn.Linear(hidden*4+1,hidden),nn.Tanh(),nn.Linear(hidden,1,bias=False))
        # Independent acceptance gate: g is the mass available to neighbours.
        self.gate=nn.Sequential(nn.Linear(self.dim*3,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.down1=nn.Linear(full_dim,hidden,bias=False); self.down2=nn.Linear(hidden,hidden,bias=False); self.up=nn.Linear(hidden,full_dim,bias=False)
        self.alpha_logit=nn.Parameter(torch.tensor(inv_sigmoid(.4)))
    @property
    def alpha(self): return .25*torch.sigmoid(self.alpha_logit)
    def _blocks(self,x,cand):
        safe=cand.clamp_min(0); r=x[:,None,self.lo:self.hi].expand(x.shape[0],cand.shape[1],self.dim); s=x[safe][:,:,self.lo:self.hi]
        return r,s,safe
    def pair_score(self,r,s,sim,symmetric=False):
        rr=self.rx(r); ss=self.sx(s); z=torch.cat((rr,ss,rr-ss,rr*ss,sim.unsqueeze(-1)),dim=-1); q=self.score(z).squeeze(-1)
        if symmetric:
            # The reverse evaluation uses the same modality slice but swaps roles.
            r2=self.rx(s); s2=self.sx(r); z2=torch.cat((r2,s2,r2-s2,r2*s2,sim.unsqueeze(-1)),dim=-1); q=.5*(q+self.score(z2).squeeze(-1))
        return q
    def weights(self,x,cand,symmetric=False,force_no_null=False,gate_override=None):
        r,s,safe=self._blocks(x,cand); valid=cand.ge(0); sim=F.cosine_similarity(r,s,dim=-1,eps=1e-6)
        logits=self.pair_score(r.reshape(-1,self.dim),s.reshape(-1,self.dim),sim.reshape(-1),symmetric).reshape(x.shape[0],cand.shape[1]); logits=logits.masked_fill(~valid,-1e9)
        summary=(s*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        rv=x[:,self.lo:self.hi]; gl=self.gate(torch.cat((rv,summary,rv-summary),dim=-1)).squeeze(-1)
        if gate_override is not None: gl=gate_override
        if force_no_null: g=torch.ones_like(gl); null=torch.zeros_like(gl)
        else: g=torch.sigmoid(gl); null=1-g
        probs=torch.softmax(logits,dim=1)*valid.to(x.dtype); w=g[:,None]*probs
        return w,null,logits,gl,g
    def delta_from_weights(self,x,cand,w):
        safe=cand.clamp_min(0); d1=self.down1(x); h1=F.relu(d1+(w.unsqueeze(-1)*d1[safe]).sum(1)); d2=self.down2(h1); h2=F.relu(d2+(w.unsqueeze(-1)*d2[safe]).sum(1)); base=F.relu(d2); return self.alpha*torch.tanh(self.up(h2-base))
    def message(self,x,cand,symmetric=False,use_null=True,gate_override=None):
        w,null,logits,gl,g=self.weights(x,cand,symmetric,force_no_null=not use_null,gate_override=gate_override); d=self.delta_from_weights(x,cand,w)
        return d,w,null,logits,gl,g
    def disabled_message(self,x):
        return torch.zeros_like(x)

class CUGLRBranch(nn.Module):
    dims=DIMS
    def __init__(self,cfg:RConfig):
        super().__init__(); self.cfg=cfg; self.relations=nn.ModuleList([ModalityRelation(m) for m in range(4)])
    def forward(self,x,candidates,mode:Optional[RConfig]=None,enabled=None):
        cfg=mode or self.cfg; enabled=[True]*4 if enabled is None else list(enabled); ds=[]; ws=[]; nulls=[]; logits=[]; gls=[]; gates=[]
        for m,rel in enumerate(self.relations):
            if enabled[m]: d,w,q,l,gl,g=rel.message(x,candidates[:,m,:],cfg.symmetric,cfg.use_null)
            else: d=torch.zeros_like(x); w=torch.zeros((x.shape[0],candidates.shape[2]),device=x.device,dtype=x.dtype); q=torch.ones(x.shape[0],device=x.device,dtype=x.dtype); l=torch.full_like(w,-1e9); gl=torch.full((x.shape[0],),-30.,device=x.device); g=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
            ds.append(d); ws.append(w); nulls.append(q); logits.append(l); gls.append(gl); gates.append(g)
        delta=torch.stack(ds,1); return x+delta.sum(1),{'delta':delta,'weights':ws,'candidate_mass':torch.stack([z.sum(1) for z in ws],1),'null_mass':torch.stack(nulls,1),'logits':logits,'gate_logit':torch.stack(gls,1),'gate':torch.stack(gates,1)}
    def utility_loss(self,info,targets,mask,scales,gate_targets=None,gate_mask=None,cfg=None):
        cfg=cfg or self.cfg; reg=[]; ranks=[]; valid_count=0
        for m,lg in enumerate(info['logits']):
            t=targets[:,m,:].to(lg.dtype); mk=mask[:,m,:].bool()&torch.isfinite(t); valid_count+=int(mk.sum())
            if mk.any():
                reg.append(F.smooth_l1_loss(lg[mk],t[mk],reduction='mean')); enough=mk.sum(1)>=2
                if enough.any():
                    hi=t.masked_fill(~mk,-float('inf')).argmax(1); lo=t.masked_fill(~mk,float('inf')).argmin(1); rows=torch.where(enough)[0]; margin=(t[rows,hi[rows]]-t[rows,lo[rows]]).abs().clamp(.1,1.5).detach(); ranks.append(F.softplus(-(lg[rows,hi[rows]]-lg[rows,lo[rows]])*margin).mean())
        zero=sum((p.sum()*0 for p in self.parameters()))
        ureg=torch.stack(reg).mean() if reg else zero; urank=torch.stack(ranks).mean() if ranks else zero
        gate_loss=zero; gate_valid=0
        if gate_targets is not None and gate_mask is not None:
            gt=gate_targets.to(info['gate_logit'].dtype); gm=gate_mask.bool()&torch.isfinite(gt); gate_valid=int(gm.sum())
            if gm.any(): gate_loss=F.binary_cross_entropy_with_logits(info['gate_logit'][gm],gt[gm],reduction='mean')
        total=cfg.utility_weight*ureg+cfg.rank_weight*urank+cfg.gate_weight*gate_loss
        return total,{'utility_reg':float(ureg.detach()),'utility_rank':float(urank.detach()),'gate_loss':float(gate_loss.detach()),'utility_valid':valid_count,'gate_valid':gate_valid}

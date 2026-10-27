"""Cross-fitted teachers and faithful edge/relationship interventions."""
from __future__ import annotations
import hashlib, random
from typing import Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from integration import make_hp, make_pln
from cugl_r import CUGLRBranch, RConfig

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

class TeacherNet(nn.Module):
    def __init__(self,layers):
        super().__init__(); self.branch=CUGLRBranch(RConfig(use_null=True)); self.cam=make_pln(layers,make_hp())
    def forward(self,x,candidates,enabled=None):
        xn,info=self.branch(x,candidates,enabled=enabled); return self.cam(xn)[0],xn,info
    @torch.no_grad()
    def precompute(self,x,candidates):
        ds=[]; caches=[]
        for m,rel in enumerate(self.branch.relations):
            c=candidates[:,m,:]; d,w,q,l,gl,g=rel.message(x,c,False,True); ds.append(d); caches.append((c,w,q,l,gl,g))
        delta=torch.stack(ds,1); xn=x+delta.sum(1); return caches,delta,xn
    @torch.no_grad()
    def target_without_edge(self,x,candidates,caches,deltas,xn,target,m,slot):
        rel=self.branch.relations[m]; c,w,q,l,gl,g=caches[m]; wt=w[target].clone(); wt[slot]=0.; full_w=torch.zeros_like(w); full_w[target]=wt; dt=rel.delta_from_weights(x,c,full_w)[target]
        return xn[target]-deltas[target,m]+dt

def train_teacher(layers,x,y,train_idx,candidates,seed,epochs=60):
    seed_all(seed); net=TeacherNet(layers).cuda(); idx=torch.as_tensor(train_idx,dtype=torch.long,device=x.device); cnt=torch.bincount(y[idx],minlength=2).float().clamp_min(1); weight=(len(idx)/(2*cnt)).to(x.device); opt=torch.optim.Adam(net.parameters(),lr=make_hp().lr,weight_decay=make_hp().reg); trace=[]; net.train()
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True); out,_,_=net(x,candidates); loss=F.nll_loss(out[idx],y[idx],weight=weight)
        if not torch.isfinite(loss): raise FloatingPointError('teacher loss is not finite')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        if ep in (0,epochs-1) or ep%10==9: trace.append({'epoch':ep,'loss':float(loss.detach())})
    net.eval(); return net,trace

@torch.no_grad()
def compute_interventions(net,x,y,candidates,heldout_idx):
    net.eval(); caches,deltas,xn=net.precompute(x,candidates); base_lp=net.cam(xn)[0]; repeat_lp=net.cam(xn)[0]; repeat_max=float((repeat_lp-base_lp).abs().max().item())
    n,_,k=candidates.shape; U=np.full((n,4,k),np.nan,np.float32); M=np.zeros_like(U,np.uint8); Ug=np.full((n,4),np.nan,np.float32); GM=np.zeros_like(Ug,np.uint8)
    edge_abs=[]; edge_locations=[]; gate_abs=[]; gate_locations=[]
    # Only heldout labels select the true-class log probability after freezing.
    for i in np.asarray(heldout_idx,np.int64):
        yi=int(y[i].item()); base=float(base_lp[i,yi].item()); variants=[]; loc=[]
        for m in range(4):
            c=candidates[i,m]
            for s in range(k):
                if int(c[s])<0: continue
                variants.append(net.target_without_edge(x,candidates,caches,deltas,xn,int(i),m,s)); loc.append((m,s)); M[i,m,s]=1
        if variants:
            alt=net.cam(torch.stack(variants))[0][:,yi].detach().cpu().numpy()
            for v,(m,s) in zip(alt,loc): U[i,m,s]=np.float32(base-float(v)); edge_abs.append(abs(float(U[i,m,s]))); edge_locations.append((i,m,s))
        # Relationship-level intervention: remove all messages from one relation.
        relvars=[]; rels=[]
        for m in range(4):
            relvars.append(xn[i]-deltas[i,m]); rels.append(m); GM[i,m]=1
        if relvars:
            altg=net.cam(torch.stack(relvars))[0][:,yi].detach().cpu().numpy()
            for v,m in zip(altg,rels): Ug[i,m]=np.float32(base-float(v)); gate_abs.append(abs(float(Ug[i,m]))); gate_locations.append((i,m))
    # no-op intervention uses the exact same input and is repeated above; the
    # FP32 repeat difference is recorded rather than amplified.
    st={'heldout_count':int(len(heldout_idx)),'valid_edges':int(M.sum()),'valid_gates':int(GM.sum()),'mean_abs_edge':float(np.nanmean(np.abs(U))) if np.isfinite(U).any() else 0.,'mean_abs_gate':float(np.nanmean(np.abs(Ug))) if np.isfinite(Ug).any() else 0.,'edge_positive':float(np.nanmean(U>0)) if np.isfinite(U).any() else 0.,'edge_negative':float(np.nanmean(U<0)) if np.isfinite(U).any() else 0.,'gate_positive':float(np.nanmean(Ug>0)) if np.isfinite(Ug).any() else 0.,'gate_negative':float(np.nanmean(Ug<0)) if np.isfinite(Ug).any() else 0.,'repeat_noop_max_abs':repeat_max,'edge_abs_values':edge_abs,'gate_abs_values':gate_abs,'baseline_sha256':hashlib.sha256(xn.detach().cpu().numpy().tobytes()).hexdigest()}
    return U,M,Ug,GM,st

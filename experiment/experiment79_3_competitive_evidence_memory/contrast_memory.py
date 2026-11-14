"""G1: Competitive Local Disease-Contrast Memory (CLDCM)."""
from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from memory_common import MemoryFusionBase, normalize_rows, stable_softmax

def _auc(a, b):
    if a.numel() == 0 or b.numel() == 0: return 0.0
    return float(((a[:,None] > b[None,:]).float() + .5*(a[:,None] == b[None,:]).float()).mean())

def build_contrast_memory(cam, labels, ids, sites, max_pairs=64, patient_max_pairs=2,
                          local_support_k=12, min_support_per_class=2,
                          min_auc=.55, min_separation=0., version=1, refresh_epoch=0):
    ids = sorted(set(map(int, ids))); y = labels.detach().long(); device = cam.device
    hn = normalize_rows(cam.detach()); asd=[i for i in ids if int(y[i])==0]; td=[i for i in ids if int(y[i])==1]
    if not asd or not td:
        return {"version":version,"refresh_epoch":refresh_epoch,"ids":[],"h_asd":torch.empty(0,144),"h_td":torch.empty(0,144),"center":torch.empty(0,144),"direction":torch.empty(0,144),"reliability":torch.empty(0),"similarity":torch.empty(0),"support_ids":[],"source_ids":ids,"kind":"CLDCM"}
    aa=torch.tensor(asd,device=device); tt=torch.tensor(td,device=device); sim=hn[aa]@hn[tt].T; cand=[]
    for r,a in enumerate(asd):
        for c in torch.argsort(sim[r],descending=True,stable=True)[:min(12,len(td))].cpu().tolist(): cand.append((float(sim[r,c]),a,td[c]))
    for r,b in enumerate(td):
        for c in torch.argsort(sim[:,r],descending=True,stable=True)[:min(12,len(asd))].cpu().tolist(): cand.append((float(sim[c,r]),asd[c],b))
    cand.sort(key=lambda z:(-z[0],z[1],z[2])); used={i:0 for i in ids}; seen=set(); picked=[]
    for simv,a,b in cand:
        if (a,b) in seen or used[a]>=patient_max_pairs or used[b]>=patient_max_pairs: continue
        center_raw=(cam[a].detach()+cam[b].detach())/2; center=normalize_rows(center_raw[None])[0]
        direction=F.normalize((cam[a].detach()-cam[b].detach()),dim=0,eps=1e-8)
        others=[i for i in ids if i not in (a,b)]
        if not others: continue
        oi=torch.tensor(others,device=device); dist=((hn[oi]-center[None])**2).sum(1)
        supp=oi[torch.argsort(dist,stable=True)[:min(local_support_k,len(others))]]
        ss=((hn[supp]-center[None])*direction[None]).sum(1); sy=y[supp]
        sa=ss[sy==0]; st=ss[sy==1]
        if sa.numel()<min_support_per_class or st.numel()<min_support_per_class: continue
        sep=float((sa.mean()-st.mean()).item()); auc=_auc(sa,st)
        if sep <= min_separation or auc < min_auc: continue
        support_score=0.5*(min(1.,sa.numel()/min(local_support_k,12))+min(1.,st.numel()/min(local_support_k,12)))
        rel=max(.01,min(1.,(auc-.5)*2*.65 + max(0.,math.tanh(sep))* .25 + support_score*.10))
        picked.append((a,b,center_raw,direction,simv,rel,supp.cpu().tolist(),sep,auc)); seen.add((a,b)); used[a]+=1; used[b]+=1
        if len(picked)>=max_pairs: break
    if not picked:
        return {"version":version,"refresh_epoch":refresh_epoch,"ids":[],"h_asd":torch.empty(0,144),"h_td":torch.empty(0,144),"center":torch.empty(0,144),"direction":torch.empty(0,144),"reliability":torch.empty(0),"similarity":torch.empty(0),"support_ids":[],"source_ids":ids,"kind":"CLDCM"}
    return {"version":version,"refresh_epoch":refresh_epoch,"ids":[(a,b) for a,b,*_ in picked],
      "h_asd":torch.stack([cam[a].detach().cpu() for a,b,*_ in picked]),"h_td":torch.stack([cam[b].detach().cpu() for a,b,*_ in picked]),
      "center":torch.stack([x[2].detach().cpu() for x in picked]),"direction":torch.stack([x[3].detach().cpu() for x in picked]),
      "reliability":torch.tensor([x[5] for x in picked]),"similarity":torch.tensor([x[4] for x in picked]),
      "support_ids":[x[6] for x in picked],"local_separation":torch.tensor([x[7] for x in picked]),
      "local_auc":torch.tensor([x[8] for x in picked]),"source_ids":ids,"kind":"CLDCM"}

class CLDCM(MemoryFusionBase):
    def __init__(self, **kw): super().__init__(**kw)
    def forward(self, query_ids, h, base_logp, memory, sites=None, *, content_permuted=False, pair_association_break=False):
        n=h.shape[0]; device,dtype=h.device,h.dtype
        if not memory or not memory.get("ids"):
            out,d=self.empty_forward(h,base_logp); d.update({"memory_count":0,"pair_reliability":h.new_zeros(n)}); return out,d
        ids=[tuple(map(int,p)) for p in memory["ids"]]; m=len(ids); ha=memory["h_asd"].to(device=device,dtype=dtype); ht=memory["h_td"].to(device=device,dtype=dtype); cen=memory["center"].to(device=device,dtype=dtype); rel=memory["reliability"].to(device=device,dtype=dtype).clamp_min(1e-4)
        if pair_association_break and m>1:
            # Preserve anchor pools but break which ASD/TD anchors form a pair.
            ht=ht.roll(1,0); ids=[(p[0],ids[(j+1)%m][1]) for j,p in enumerate(ids)]
        q=self.query(h); ka=self.key(ha); kt=self.key(ht)
        va=self.v_proj(ha-cen); vt=self.v_proj(ht-cen)
        qids=torch.as_tensor(list(map(int,query_ids)),device=device)
        pair_ids=torch.as_tensor(ids,device=device,dtype=torch.long)
        valid=~(qids[:,None,None].eq(pair_ids[None,:,:]).any(2))
        pair_sim=torch.maximum(q@ka.T,q@kt.T); k=min(8,m); top=pair_sim.masked_fill(~valid,-1e9).topk(k,1).indices; row=torch.arange(n,device=device)[:,None]
        la=(q[:,None,:]*ka[top]).sum(-1)/self.temperature + rel[top].log(); lt=(q[:,None,:]*kt[top]).sum(-1)/self.temperature + rel[top].log()
        both=torch.cat([la,lt],1); alpha=torch.softmax(both,1); wa,wt=alpha[:,:k],alpha[:,k:]
        if content_permuted and k>1: va=va.roll(1,0); vt=vt.roll(1,0)
        ra=(wa[...,None]*va[top]).sum(1); rt=(wt[...,None]*vt[top]).sum(1)
        sa=torch.logsumexp(la,1)-math.log(max(k,1)); st=torch.logsumexp(lt,1)-math.log(max(k,1)); enta=-(wa*wa.clamp_min(1e-12).log()).sum(1)/math.log(max(k,2)); entt=-(wt*wt.clamp_min(1e-12).log()).sum(1)/math.log(max(k,2)); av=valid.gather(1,top).any(1).to(dtype); rr=rel[top].mean(1)
        ret=[] if torch.is_grad_enabled() else [{"pair_ids":[ids[int(j)] for j in top[i].cpu().tolist()],"attention_asd":wa[i].detach().cpu().tolist(),"attention_td":wt[i].detach().cpu().tolist()} for i in range(n)]
        return self.finish(h,base_logp,ra,rt,sa,st,rr,enta,entt,av,av,ret,{"memory_count":m,"pair_reliability":rr})

# Backward-compatible aliases are intentionally scoped to this experiment.
LDCM=CLDCM

"""Learnable mask-conditioned, cross-fitted retrieval and CAM enhancement."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class CFCPDEM(nn.Module):
    def __init__(self, temperature=0.2, cap=0.075, gate_bias=-2.0, preselect=16, top_k=8):
        super().__init__()
        self.temperature=float(temperature); self.cap=float(cap)
        self.preselect=int(preselect); self.top_k=int(top_k)
        self.query_norm=nn.LayerNorm(144)
        self.key_norm=nn.LayerNorm(144)
        self.Wq=nn.Linear(144,64,bias=False)
        self.Wk=nn.Linear(144,64,bias=False)
        self.Wv=nn.Linear(144,64,bias=False)
        self.read_projection=nn.Sequential(nn.Linear(192,64,bias=False),nn.GELU(),nn.Linear(64,144,bias=False))
        # base p(2), memory p(2), score difference(1), class reliability(2),
        # attention entropy(2), availability(2), cross-class uncertainty(1).
        self.gate=nn.Linear(12,1,bias=True)
        nn.init.constant_(self.gate.bias,float(gate_bias))
        nn.init.normal_(self.read_projection[-1].weight,mean=0.0,std=1e-2)

    def _read(self, query_ids, h, memory, label, *, content_permuted=False,
              learnable_matching_off=False, reliability_off=False, collect_retrieval=False):
        n=h.shape[0]; device=h.device; dtype=h.dtype
        z=h.new_zeros((n,64)); zero=h.new_zeros(n)
        if not memory or not memory.get("ids"):
            return z,zero,zero,zero,zero,torch.zeros(n,dtype=torch.bool,device=device),[[] for _ in range(n)],zero
        mem_labels=memory["label"].to(device)
        keep=mem_labels.eq(int(label))
        raw_indices=torch.where(keep)[0]
        if raw_indices.numel()==0:
            return z,zero,zero,zero,zero,torch.zeros(n,dtype=torch.bool,device=device),[[] for _ in range(n)],zero
        ids_all=[int(memory["ids"][i]) for i in raw_indices.cpu().tolist()]
        ref=memory["reference"].to(device=device,dtype=dtype)[raw_indices]
        masks=memory["mask"].to(device=device,dtype=dtype)[raw_indices]
        dirs=memory["direction"].to(device=device,dtype=dtype)[raw_indices]
        rel=memory["reliability"].to(device=device,dtype=dtype)[raw_indices].clamp_min(1e-8)
        mu=memory["neutral_mean"].to(device=device,dtype=dtype)
        qids=torch.as_tensor(list(map(int,query_ids)),dtype=torch.long,device=device)
        mids=torch.tensor(ids_all,dtype=torch.long,device=device)
        valid=~qids[:,None].eq(mids[None,:])

        # Fixed masked cosine is only a low-cost candidate proposal.  A and B
        # training queries are already outside the source bank they consume.
        qmask=(h[:,None,:]-mu[None,None,:])*masks[None,:,:]
        kmask=(ref[None,:,:]-mu[None,None,:])*masks[None,:,:]
        fixed=(qmask*kmask).sum(-1)/(qmask.norm(dim=-1)*kmask.norm(dim=-1)).clamp_min(1e-8)
        fixed=fixed.masked_fill(~valid,-1e6)
        kpre=min(self.preselect,ref.shape[0])
        proposed=fixed.topk(kpre,dim=1,largest=True,sorted=True).indices
        row_valid=valid.gather(1,proposed)
        row_masks=masks[proposed]
        row_ref=ref[proposed]
        row_rel=rel[proposed]
        row_fixed=fixed.gather(1,proposed)
        qx=(h[:,None,:]-mu[None,None,:])*row_masks
        kx=(row_ref-mu[None,None,:])*row_masks
        q=self.Wq(self.query_norm(qx.reshape(-1,144))).reshape(n,kpre,64)
        k=self.Wk(self.key_norm(kx.reshape(-1,144))).reshape(n,kpre,64)
        q=F.normalize(q,dim=-1,eps=1e-8); k=F.normalize(k,dim=-1,eps=1e-8)
        learned=(q*k).sum(-1)
        matched=row_fixed if learnable_matching_off else learned
        rel_bias=torch.zeros_like(row_rel) if reliability_off else row_rel.log()
        scores=matched/self.temperature+rel_bias
        scores=scores.masked_fill(~row_valid,-1e6)
        final_k=min(self.top_k,kpre)
        top=scores.topk(final_k,dim=1,largest=True,sorted=True).indices
        top_valid=row_valid.gather(1,top)
        top_scores=scores.gather(1,top)
        attn=torch.softmax(top_scores.masked_fill(~top_valid,-1e6),dim=1)*top_valid.to(dtype)
        attn=attn/attn.sum(1,keepdim=True).clamp_min(1e-8)
        chosen_global=proposed.gather(1,top)
        values=(ref-mu[None])*masks+0.5*dirs
        value_indices=chosen_global
        if content_permuted and final_k>1:
            value_indices=chosen_global.roll(1,dims=1)
        v=self.Wv(values[value_indices])
        read=(attn[:,:,None]*v).sum(1)
        count=top_valid.sum(1)
        avail=count.gt(0)
        logmean=torch.logsumexp(top_scores.masked_fill(~top_valid,-1e6),dim=1)-count.clamp_min(1).to(dtype).log()
        logmean=torch.where(avail,logmean,torch.zeros_like(logmean))
        entropy=-(attn*attn.clamp_min(1e-12).log()).sum(1)/math.log(max(final_k,2))
        rtop=row_rel.gather(1,top)
        reliability=(attn*rtop*top_valid.to(dtype)).sum(1)
        if reliability_off: reliability=torch.full_like(reliability,0.5)
        retrieval=[[] for _ in range(n)]
        if collect_retrieval:
            for i in range(n):
                selected=[]
                for j in range(final_k):
                    if bool(top_valid[i,j]):
                        mi=int(chosen_global[i,j].detach().cpu())
                        selected.append({"patient_id":ids_all[mi],"attention":float(attn[i,j].detach().cpu()),
                                         "value_patient_id":ids_all[int(value_indices[i,j].detach().cpu())],
                                         "reliability":float(rtop[i,j].detach().cpu()),
                                         "fixed_score":float(row_fixed[i,top[i,j]].detach().cpu()),
                                         "fixed_rank":1+int((row_fixed[i]>row_fixed[i,top[i,j]]).sum().detach().cpu()),
                                         "learned_score":float(learned[i,top[i,j]].detach().cpu())})
                retrieval[i]=selected
        learned_delta=(learned-row_fixed).abs().masked_fill(~row_valid,0).sum(1)/row_valid.sum(1).clamp_min(1)
        return read,logmean,entropy,reliability,avail.to(dtype),avail,retrieval,learned_delta

    def forward(self, query_ids, h, base_logp, memory, *, intervention=None, collect_retrieval=False):
        ra,sa,ea,r_a,a_a,va,pa,learned_a=self._read(
            query_ids,h,memory,0,content_permuted=intervention=="Memory-Content-Permuted",
            learnable_matching_off=intervention=="Learnable-Matching-Off",
            reliability_off=intervention=="Cross-Patient-Reliability-Off",collect_retrieval=collect_retrieval)
        rt,st,et,r_t,a_t,vt,pt,learned_t=self._read(
            query_ids,h,memory,1,content_permuted=intervention=="Memory-Content-Permuted",
            learnable_matching_off=intervention=="Learnable-Matching-Off",
            reliability_off=intervention=="Cross-Patient-Reliability-Off",collect_retrieval=collect_retrieval)
        available=va&vt
        evidence_logits=torch.stack([sa,st],1)
        evidence_logits=torch.where(available[:,None],evidence_logits,torch.zeros_like(evidence_logits))
        mem_prob=torch.softmax(evidence_logits,1)
        uncertainty=-(mem_prob*mem_prob.clamp_min(1e-12).log()).sum(1)/math.log(2.0)
        # A missing class can yield an exact zero weighted reliability.  Clamp
        # before sqrt so its derivative cannot become infinite at zero.
        rel_product=(r_a.clamp_min(1e-6)*r_t.clamp_min(1e-6)).clamp_min(1e-12)
        rel_raw=torch.sqrt(rel_product)
        reliability=rel_raw/(rel_raw+0.1)
        diff=(mem_prob[:,0]-mem_prob[:,1])
        gate_x=torch.cat([base_logp.detach(),mem_prob.detach(),(sa-st)[:,None].detach(),
                          r_a[:,None].detach(),r_t[:,None].detach(),ea[:,None],et[:,None],
                          a_a[:,None],a_t[:,None],uncertainty[:,None]],1)
        gate=torch.sigmoid(self.gate(gate_x)).squeeze(1)*reliability*available.to(h.dtype)
        reads=torch.cat([ra,rt,ra-rt],1)
        raw=self.read_projection(reads)
        u=gate[:,None]*raw
        hnorm=h.norm(dim=1,keepdim=True).clamp_min(1e-8)
        cap=self.cap*hnorm
        unorm=u.norm(dim=1,keepdim=True)
        delta=cap*torch.tanh(unorm/cap.clamp_min(1e-8))*u/(unorm+1e-8)
        delta=torch.where(available[:,None],delta,torch.zeros_like(delta))
        enhanced=h+delta
        delta_ratio=delta.norm(dim=1)/hnorm.squeeze(1)
        diag={"delta_h":delta,"raw_delta":raw,"gate":gate,"delta_ratio":delta_ratio,
              "cap_fraction":(delta_ratio>0.99*self.cap).to(h.dtype),"evidence_logits":evidence_logits,
              "evidence_valid":available,"memory_probability":mem_prob,"memory_reliability":reliability,
              "reliability_asd":r_a,"reliability_td":r_t,"score_asd":sa,"score_td":st,
              "entropy_asd":ea,"entropy_td":et,"availability_asd":a_a,"availability_td":a_t,
              "learned_fixed_gap":(learned_a+learned_t)/2,
              "retrieval":[{"asd":pa[i],"td":pt[i]} for i in range(len(query_ids))]}
        return enhanced,diag


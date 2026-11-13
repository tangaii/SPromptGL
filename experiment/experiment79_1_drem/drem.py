"""Memory-dependent attention and recovery-aware feature enhancement for E79.1."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from memory_bank import TYPE_ASD_TO_TD, TYPE_TD_TO_ASD

class DREM(nn.Module):
    def __init__(self, cam_dim=144, memory_dim=64, temperature=.2):
        super().__init__(); self.cam_dim=cam_dim; self.memory_dim=memory_dim; self.temperature=temperature
        self.cam_norm=nn.LayerNorm(cam_dim)
        self.q_proj=nn.Linear(cam_dim,memory_dim,bias=False)
        self.v_proj=nn.Linear(cam_dim,memory_dim,bias=False)
        self.r_proj=nn.Linear(cam_dim,memory_dim,bias=False)
        self.read_to_delta=nn.Sequential(nn.Linear(2*memory_dim,128,bias=False),nn.GELU(),nn.Linear(128,cam_dim,bias=False))
        nn.init.zeros_(self.read_to_delta[-1].weight)
        # base logp (2), abs margin (1), max sim/entropy/availability per branch (6)
        self.gate=nn.Sequential(nn.Linear(9,16),nn.GELU(),nn.Linear(16,1))
        nn.init.constant_(self.gate[-1].bias,-2.2)
        self.gate_input_dim=9
    def qkey(self,h):
        return F.normalize(self.q_proj(self.cam_norm(h)),dim=-1)
    def _bucket_center_ln(self,cand,bucket):
        bf=cand.get('bucket_features',{}).get(bucket)
        if bf is None or bf.numel()==0: return cand['key_features'].new_zeros(self.cam_dim)
        return self.cam_norm(bf).mean(0)
    def _read(self,qrow,cand,use_recovery=True,association_break=False,direction_permuted=False,cache=None):
        if cand is None or cand['key_features'].numel()==0:
            z=qrow.new_zeros(self.memory_dim); return z,False,torch.tensor(-20.,device=qrow.device),torch.tensor(0.,device=qrow.device),[],torch.empty(0,device=qrow.device)
        # Project the selected candidates as one small batch.  This is
        # algebraically identical to the per-memory computation, but avoids
        # thousands of one-row autograd graphs when a fold has many recovery
        # events.
        if cache is None:
            k=F.normalize(self.q_proj(self.cam_norm(cand['key_features'])),dim=-1)
            v_base=None; r_base=None
        else:
            ix=torch.as_tensor([cache['index'][int(pid)] for pid in cand['patient_ids']],device=qrow.device,dtype=torch.long)
            k=cache['key'][ix]; v_base=cache['value'][ix]; r_base=cache['direction'][ix]
        score=(k@qrow)/self.temperature + torch.log(cand['priority'].clamp_min(1e-6))
        alpha=torch.softmax(score,dim=0)
        centers=torch.stack([self._bucket_center_ln(cand,b) for b in cand['buckets']])
        v=(self.v_proj(self.cam_norm(cand['value_features'])) if v_base is None else v_base)-self.v_proj(centers)
        if use_recovery:
            if r_base is None:
                dirs=cand['directions']
                if direction_permuted and dirs.shape[0]>1: dirs=dirs.roll(1,0)
                v=v+self.r_proj(dirs)
            else:
                rb=r_base.roll(1,0) if direction_permuted and r_base.shape[0]>1 else r_base
                v=v+rb
        if association_break and v.shape[0]>1: v=v.roll(1,0)
        read=alpha@v
        sim=(k@qrow).detach()
        simmax=sim.max() if sim.numel() else qrow.new_tensor(-1.)
        ent=(-(alpha*alpha.clamp_min(1e-12).log()).sum()/math.log(max(int(alpha.numel()),2))).detach() if alpha.numel()>1 else qrow.new_zeros(())
        return read,True,simmax,ent,cand['patient_ids'],alpha.detach()
    def _read_batched(self,q,cands,use_recovery=True,association_break=False,direction_permuted=False):
        """Batch the two small cross-attention reads over all query patients.

        The candidate lists are still selected deterministically by the memory
        bank.  Padding is only an implementation detail; masked rows produce
        exactly the same softmax, value centering and rolling operations as the
        per-query reference path.
        """
        n=len(cands); device=q.device; dtype=q.dtype
        lengths=torch.tensor([0 if c is None else len(c['patient_ids']) for c in cands],device=device,dtype=torch.long)
        kmax=int(lengths.max().item()) if n else 0
        if kmax==0:
            z=torch.zeros(n,self.memory_dim,device=device,dtype=dtype); zero=torch.zeros(n,device=device,dtype=dtype); neg=torch.full((n,),-20.,device=device,dtype=dtype)
            return z,torch.zeros(n,2,device=device,dtype=dtype),zero,zero,[[] for _ in range(n)],[torch.empty(0,device=device) for _ in range(n)]
        keys=torch.zeros(n,kmax,self.cam_dim,device=device,dtype=dtype); vals=keys.clone(); dirs=keys.clone(); pri=torch.zeros(n,kmax,device=device,dtype=dtype); mask=torch.zeros(n,kmax,device=device,dtype=torch.bool); bidx=torch.zeros(n,kmax,device=device,dtype=torch.long); patients=[]; buckets=[]
        for i,c in enumerate(cands):
            if c is None:
                patients.append([]); buckets.append([]); continue
            m=len(c['patient_ids']); patients.append(list(c['patient_ids'])); buckets.append(list(c['buckets'])); mask[i,:m]=True
            keys[i,:m]=c['key_features'].to(device=device,dtype=dtype); vals[i,:m]=c['value_features'].to(device=device,dtype=dtype); dirs[i,:m]=c['directions'].to(device=device,dtype=dtype); pri[i,:m]=c['priority'].to(device=device,dtype=dtype)
            bidx[i,:m]=torch.tensor([{'stable':0,'recent':1,'recovered':2}[b] for b in c['buckets']],device=device,dtype=torch.long)
        # Compute each query's bucket center from all candidates in that bucket,
        # including candidates outside its selected top-k set.
        bmax=0
        for c in cands:
            if c is not None: bmax=max(bmax,max((int(x.shape[0]) for x in c.get('bucket_features',{}).values()),default=0))
        if bmax==0: bmax=1
        bf=torch.zeros(n,3,bmax,self.cam_dim,device=device,dtype=dtype); bm=torch.zeros(n,3,bmax,device=device,dtype=dtype)
        for i,c in enumerate(cands):
            if c is None: continue
            for b,j in (('stable',0),('recent',1),('recovered',2)):
                x=c.get('bucket_features',{}).get(b)
                if x is not None and x.numel():
                    m=min(int(x.shape[0]),bmax); bf[i,j,:m]=x[:m].to(device=device,dtype=dtype); bm[i,j,:m]=1.
        ln_bf=self.cam_norm(bf.reshape(-1,self.cam_dim)).reshape(n,3,bmax,self.cam_dim); denom=bm.sum(2,keepdim=True).clamp_min(1.); centers=(ln_bf*bm.unsqueeze(-1)).sum(2)/denom
        selected_centers=centers.gather(1,bidx.unsqueeze(-1).expand(-1,-1,self.cam_dim))
        ln_keys=self.cam_norm(keys.reshape(-1,self.cam_dim)).reshape(n,kmax,self.cam_dim); k=F.normalize(self.q_proj(ln_keys),dim=-1)
        ln_vals=self.cam_norm(vals.reshape(-1,self.cam_dim)).reshape(n,kmax,self.cam_dim); v=self.v_proj(ln_vals-selected_centers)
        if use_recovery:
            if direction_permuted:
                base=torch.arange(kmax,device=device).view(1,-1).expand(n,-1); den=lengths.clamp_min(1).view(-1,1); src=(base-1)%den; src=torch.where(base<lengths.view(-1,1),src,base); dirs=dirs.gather(1,src.unsqueeze(-1).expand(-1,-1,self.cam_dim))
            v=v+self.r_proj(dirs.reshape(-1,self.cam_dim)).reshape(n,kmax,self.memory_dim)
        if association_break:
            base=torch.arange(kmax,device=device).view(1,-1).expand(n,-1); den=lengths.clamp_min(1).view(-1,1); src=(base-1)%den; src=torch.where(base<lengths.view(-1,1),src,base); v=v.gather(1,src.unsqueeze(-1).expand(-1,-1,self.memory_dim))
        score=(k*q.unsqueeze(1)).sum(-1)/self.temperature + torch.log(pri.clamp_min(1e-6)); score=score.masked_fill(~mask,-1e9); valid=lengths>0; score[:,0]=torch.where(valid,score[:,0],torch.zeros_like(score[:,0])); alpha=torch.softmax(score,dim=1)*mask.to(dtype)
        read=(alpha.unsqueeze(1)@v).squeeze(1); sim=(k*q.unsqueeze(1)).sum(-1); simmax=torch.where(valid,sim.masked_fill(~mask,-1e9).max(1).values,torch.full((n,),-20.,device=device,dtype=dtype)); ent=-(alpha*alpha.clamp_min(1e-12).log()).sum(1)/lengths.clamp_min(2).to(dtype).log(); ent=torch.where(valid,ent,torch.zeros_like(ent)); availability=torch.stack([valid.to(dtype),valid.to(dtype)],dim=1); att=[alpha[i,:int(lengths[i])].detach() for i in range(n)]
        return read,availability,simmax,ent,patients,att
    def forward(self,query_ids,h,base_logp,bank,sites,*,use_recovery=True,type_swap=False,association_break=False,direction_permuted=False,recency_off=False):
        ids=[int(x) for x in query_ids]; q=self.qkey(h.detach())
        candidates=bank.retrieval(ids,q,sites,h.device,lambda x:self.qkey(x),type_swap=type_swap,recency_off=recency_off)
        ra,ava,sa,ea,pa,aa=self._read_batched(q,[row[0] for row in candidates],use_recovery,association_break,direction_permuted)
        rb,avb,sb,eb,pb,ab=self._read_batched(q,[row[1] for row in candidates],use_recovery,association_break,direction_permuted)
        avail=torch.stack([ava[:,0],avb[:,0]],dim=1); gs=torch.stack([sa,ea,sb,eb,avail[:,0],avail[:,1]],dim=1)
        retrieval=[{'asd_to_td_patient_ids':pa[i],'td_to_asd_patient_ids':pb[i],'asd_to_td_attention':aa[i],'td_to_asd_attention':ab[i],'asd_to_td_count':len(pa[i]),'td_to_asd_count':len(pb[i])} for i in range(len(candidates))]
        margin=(base_logp[:,0]-base_logp[:,1]).abs().unsqueeze(1)
        gate_input=torch.cat([base_logp.detach(),margin.detach(),gs],dim=1)
        gate=torch.sigmoid(self.gate(gate_input)).squeeze(1)
        raw=self.read_to_delta(torch.cat([ra,rb],dim=1))
        delta=gate.unsqueeze(1)*raw
        hnorm=h.norm(dim=1,keepdim=True); dnorm=delta.norm(dim=1,keepdim=True); cap=.1*hnorm
        scale=torch.minimum(torch.ones_like(dnorm),cap/dnorm.clamp_min(1e-12)); delta=delta*scale
        empty=avail.sum(1).eq(0); delta=torch.where(empty[:,None],torch.zeros_like(delta),delta)
        return h+delta,{'delta_h':delta,'read_asd_to_td':ra,'read_td_to_asd':rb,'availability':avail,'gate':gate,'gate_input':gate_input,'similarity_stats':gs,'retrieval':retrieval,'delta_ratio':(delta.norm(dim=1)/(hnorm.squeeze(1).clamp_min(1e-12))).detach(),'cap_fraction':(scale.squeeze(1)<.999999).float().detach()}

def init_pair(seed=7911,device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        a=DREM().to(device); b=DREM().to(device); b.load_state_dict(a.state_dict())
    return a,b

def output_layer_logp(output_layer,h):
    from torch.func import functional_call
    output_layer.eval(); params={n:v.detach() for n,v in output_layer.named_parameters()}; buffers={n:v.detach() for n,v in output_layer.named_buffers()}
    return functional_call(output_layer,(params,buffers),(h,None))

def memory_loss(logp_mem,base_logp,labels,query_ids,class_weights,delta,h,gate):
    ids=torch.as_tensor(query_ids,dtype=torch.long,device=logp_mem.device); y=labels[ids].long(); base=base_logp.detach(); pred=base.argmax(1); hard=pred.ne(y); sample=torch.where(hard,torch.full_like(y,3.,dtype=torch.float32),torch.ones_like(y,dtype=torch.float32))
    per=F.nll_loss(logp_mem,y,weight=class_weights.to(logp_mem.device).float(),reduction='none'); cls=(per*sample).sum()/sample.sum().clamp_min(1.)
    if hard.any():
        ts=logp_mem.gather(1,y[:,None]).squeeze(1); ws=logp_mem.gather(1,pred[:,None]).squeeze(1); hard_loss=F.softplus(.2-(ts-ws))[hard].mean()
    else: hard_loss=logp_mem.new_zeros(())
    correct=~hard; preserve=((logp_mem-base).square().sum(1)[correct].mean() if correct.any() else logp_mem.new_zeros(()))
    ratio=delta.norm(dim=1)/(h.norm(dim=1).clamp_min(1e-8)); sparse=gate.square().mean()+ratio.square().mean()
    total=cls+.1*hard_loss+.15*preserve+.02*sparse
    return total,{'mem_cls':cls,'hard':hard_loss,'preserve':preserve,'sparse':sparse,'hard_count':int(hard.sum()),'gate_mean':float(gate.detach().mean()),'delta_ratio_mean':float(ratio.detach().mean())}

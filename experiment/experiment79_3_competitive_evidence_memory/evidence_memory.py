"""G2: Cross-Patient Discriminative Evidence Memory (CPDEM)."""
from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from memory_common import MemoryFusionBase, output_layer_logp, stable_softmax

def _select(ids, labels, margins, sites, max_per):
    y=labels.detach().cpu().long(); st=sites.detach().cpu().long(); out=[]
    for cls in (0,1):
        pool=[int(i) for i in ids if int(y[i])==cls]; pool.sort(key=lambda i:(-float(margins[i]),int(st[i]),i)); picks=[]; by={}
        for i in pool: by.setdefault(int(st[i]),[]).append(i)
        while by and len(picks)<max_per:
            for s in sorted(list(by)):
                if not by[s]: by.pop(s,None); continue
                picks.append(by[s].pop(0))
                if not by[s]: by.pop(s,None)
                if len(picks)>=max_per: break
        out.extend(picks)
        out.extend([i for i in pool if i not in picks][:max(0,max_per-len(picks))])
    return sorted(set(out))

def _auc(a,b):
    if a.numel()==0 or b.numel()==0:return 0.
    return float(((a[:,None]>b[None,:]).float()+.5*(a[:,None]==b[None,:]).float()).mean())

def _empty(ids, version, refresh, candidate=0, rejected=0):
    return {"version":version,"refresh_epoch":refresh,"ids":[],"label":torch.empty(0,dtype=torch.long),"reference":torch.empty(0,144),"values":torch.empty(0,144),"mask":torch.empty(0,144),"contrast_direction":torch.empty(0,144),"reliability":torch.empty(0),"joint_drop":torch.empty(0),"support_ids":[],"support_opposite_ids":[],"source_ids":ids,"kind":"CPDEM","candidate_count":candidate,"rejected_count":rejected}

def build_evidence_memory(cam, labels, base_logp, ids, sites, output_layer, max_per_class=48,
                          evidence_dims=12, alphas=(.5,1.), min_same_support=4,
                          min_opposite_support=4, min_auc=.60, min_same_minus_opp=0., version=1, refresh_epoch=0):
    ids=sorted(set(map(int,ids))); y=labels.detach().long(); pred=base_logp.detach().argmax(1); device=cam.device
    correct=[i for i in ids if int(pred[i])==int(y[i])]
    if not correct:return _empty(ids,version,refresh_epoch)
    idt=torch.tensor(ids,device=device); mu=cam[idt].detach().mean(0)
    margins=torch.zeros(cam.shape[0],device=device)
    margins[correct]=base_logp[torch.tensor(correct,device=device),0].where(y[torch.tensor(correct,device=device)]==0,base_logp[torch.tensor(correct,device=device),1])-base_logp[torch.tensor(correct,device=device),1].where(y[torch.tensor(correct,device=device)]==0,base_logp[torch.tensor(correct,device=device),0])
    selected=_select(correct,y,margins.cpu().tolist(),sites,max_per_class); n=len(selected); d=cam.shape[1]
    if not selected:return _empty(ids,version,refresh_epoch)
    ix=torch.tensor(selected,device=device); h=cam[ix].detach(); yy=y[ix]
    true=(base_logp[ix].gather(1,yy[:,None]).squeeze(1)-base_logp[ix].gather(1,(1-yy)[:,None]).squeeze(1))
    rows=[]
    for a in alphas:
        for j in range(d):
            z=h.clone(); z[:,j]=(1-float(a))*z[:,j]+float(a)*mu[j]; rows.append(z)
    with torch.no_grad(): ilp=output_layer_logp(output_layer,torch.cat(rows,0))
    blocks=ilp.reshape(len(alphas),d,n,2).permute(2,0,1,3); imp=[]
    for ai in range(len(alphas)):
        mm=blocks[:,ai].gather(2,yy[:,None,None].expand(-1,d,1)).squeeze(-1)-blocks[:,ai].gather(2,(1-yy)[:,None,None].expand(-1,d,1)).squeeze(-1); imp.append(true[:,None]-mm)
    imp=torch.stack(imp,1); positive=(imp>0).all(1); score=torch.where(positive,imp.min(1).values,torch.zeros_like(imp[:,0])); masks=torch.zeros(n,d,device=device); joint=torch.zeros(n,device=device); prelim=[]
    for r,pid in enumerate(selected):
        order=torch.argsort(score[r],descending=True,stable=True); good=order[score[r,order]>0][:evidence_dims]
        if good.numel()==0:continue
        masks[r,good]=1.; hm=h[r].clone(); hm[good]=mu[good]
        with torch.no_grad(): lm=output_layer_logp(output_layer,torch.stack([h[r],hm]))
        joint[r]=(lm[0,int(yy[r])]-lm[0,1-int(yy[r])])-(lm[1,int(yy[r])]-lm[1,1-int(yy[r])])
        if float(joint[r])>0: prelim.append(r)
    if not prelim:return _empty(ids,version,refresh_epoch,n,n)
    # Vectorized cross-patient support calculation.  The old implementation
    # synchronized once per candidate/support pair and made refreshes scale
    # poorly with ten workers.  This is the same deterministic background
    # neighborhood and evidence-score definition evaluated as matrices.
    all_cam=cam.detach(); D=all_cam[idt]-mu
    pr=torch.tensor(prelim,device=device); pids=torch.tensor([selected[r] for r in prelim],device=device)
    mp=masks[pr]; yp=yy[pr]; bg=1.-mp; qbg=(all_cam[pids]-mu)*bg
    bgD=D[None,:,:]*bg[:,None,:]
    bg_num=torch.einsum('pd,pnd->pn',qbg,bgD)
    bg_den=qbg.norm(dim=1)[:,None]*bgD.norm(dim=2)
    bg_sim=bg_num/bg_den.clamp_min(1e-8)
    same_valid=(y[idt][None,:]==yp[:,None]) & (idt[None,:]!=pids[:,None]); opp_valid=~same_valid & (idt[None,:]!=pids[:,None])
    ks=min(max(min_same_support,16), len(ids)); ko=min(max(min_opposite_support,16), len(ids))
    same_idx=bg_sim.masked_fill(~same_valid,-1e9).topk(ks,dim=1).indices; opp_idx=bg_sim.masked_fill(~opp_valid,-1e9).topk(ko,dim=1).indices
    same_ok=same_valid.gather(1,same_idx).sum(1)>=min_same_support; opp_ok=opp_valid.gather(1,opp_idx).sum(1)>=min_opposite_support
    qev=(all_cam[pids]-mu)*mp; same_vec=D[same_idx]*mp[:,None,:]; opp_vec=D[opp_idx]*mp[:,None,:]
    ss=(qev[:,None,:]*same_vec).sum(2)/(qev.norm(dim=1)[:,None]*same_vec.norm(dim=2)).clamp_min(1e-8)
    oo=(qev[:,None,:]*opp_vec).sum(2)/(qev.norm(dim=1)[:,None]*opp_vec.norm(dim=2)).clamp_min(1e-8)
    auc=((ss[:,:,None]>oo[:,None,:]).float()+.5*(ss[:,:,None]==oo[:,None,:]).float()).mean((1,2)); diff=ss.mean(1)-oo.mean(1)
    accepted=same_ok & opp_ok & (auc>=min_auc) & (diff>min_same_minus_opp)
    records=[]
    for j in torch.where(accepted)[0].detach().cpu().tolist():
        r=prelim[j]; same=[int(x) for x in idt[same_idx[j]].detach().cpu().tolist() if int(x)!=selected[r]][:max(min_same_support,16)]; opp=[int(x) for x in idt[opp_idx[j]].detach().cpu().tolist() if int(x)!=selected[r]][:max(min_opposite_support,16)]
        auc_j=float(auc[j].item()); diff_j=float(diff[j].item()); direction=F.normalize(mp[j]*(D[same_idx[j]].mean(0)-D[opp_idx[j]].mean(0)),dim=0,eps=1e-8)
        rself=float(torch.sigmoid(joint[r]).item()); rwithin=max(0.,min(1.,float((ss[j].mean()+1)/2))); rbetween=max(0.,min(1.,2*(auc_j-.5))); rel=max(.001,(rself*rwithin*rbetween)**(1/3))
        records.append((r,rel,same,opp,diff_j,auc_j,rself,rwithin,rbetween,direction))
    if not records:return _empty(ids,version,refresh_epoch,n,len(prelim))
    records=sorted(records,key=lambda z:(-z[1],selected[z[0]]))[:2*max_per_class]; rr=torch.tensor([z[1] for z in records]); keep=torch.tensor([z[0] for z in records],device=device)
    return {"version":version,"refresh_epoch":refresh_epoch,"ids":[selected[z[0]] for z in records],"label":yy[keep].cpu(),"reference":h[keep].cpu(),"values":h[keep].cpu(),"mask":masks[keep].cpu(),"contrast_direction":torch.stack([z[9] for z in records]).cpu(),"reliability":rr,"joint_drop":joint[keep].cpu(),"support_ids":[z[2] for z in records],"support_opposite_ids":[z[3] for z in records],"support_auc":torch.tensor([z[5] for z in records]),"within_reliability":torch.tensor([z[7] for z in records]),"between_reliability":torch.tensor([z[8] for z in records]),"self_reliability":torch.tensor([z[6] for z in records]),"source_ids":ids,"neutral_mean":mu.cpu(),"kind":"CPDEM","candidate_count":n,"rejected_count":n-len(records)}

class CPDEM(MemoryFusionBase):
    def __init__(self, **kw): super().__init__(**kw)
    def _read(self, query_ids, h, memory, label, content_permuted=False, mask_break=False, reliability_off=False):
        n=h.shape[0]; device,dtype=h.device,h.dtype; z=h.new_zeros((n,self.memory_dim)); zero=h.new_zeros(n)
        if not memory or not memory.get("ids"):return z,zero,zero,zero,[[] for _ in range(n)]
        keep=memory["label"].eq(label); ids=[p for p,k in zip(memory["ids"],keep.tolist()) if k]
        if not ids:return z,zero,zero,zero,[[] for _ in range(n)]
        ref=memory["reference"][keep].to(device=device,dtype=dtype); vals=memory["values"][keep].to(device=device,dtype=dtype); masks=memory["mask"][keep].to(device=device,dtype=dtype); dirs=memory["contrast_direction"][keep].to(device=device,dtype=dtype); rel=memory["reliability"][keep].to(device=device,dtype=dtype).clamp_min(1e-4); mu=memory.get("neutral_mean",h.new_zeros(144)).to(device=device,dtype=dtype)
        if mask_break and len(ids)>1:masks=masks.roll(1,0); dirs=dirs.roll(1,0)
        q=(h[:,None,:]-mu[None,None,:]); k=(ref[None,:,:]-mu[None,None,:]); num=(q*k*masks[None]).sum(-1); den=(q.square()*masks[None]).sum(-1).sqrt()*(k.square()*masks[None]).sum(-1).sqrt(); sim=num/den.clamp_min(1e-8)
        qids=torch.as_tensor(list(map(int,query_ids)),device=device)
        memids=torch.as_tensor(ids,device=device,dtype=torch.long)
        valid=~qids[:,None].eq(memids[None,:])
        kk=min(8,len(ids)); top=sim.masked_fill(~valid,-1e9).topk(kk,1).indices; ss=sim.gather(1,top); rr=torch.ones_like(rel) if reliability_off else rel; logits=ss/self.temperature+rr.log()[top]; w=stable_softmax(logits,torch.gather(valid,1,top)); vv=((vals-mu[None])*masks+dirs*rr[:,None])
        if content_permuted and kk>1:vv=vv.roll(1,0)
        val=self.v_proj(vv); read=(w[...,None]*val[top]).sum(1); score=torch.logsumexp(logits,1)-math.log(max(kk,1)); ent=-(w*w.clamp_min(1e-12).log()).sum(1)/math.log(max(kk,2)); av=valid.gather(1,top).any(1).to(dtype); ret=[] if torch.is_grad_enabled() else [{"patient_ids":[ids[int(j)] for j in top[r].cpu().tolist()],"attention":w[r].detach().cpu().tolist()} for r in range(n)]
        return read,score,ent,av,ret
    def forward(self,query_ids,h,base_logp,memory,sites=None,*,content_permuted=False,evidence_mask_break=False,cross_patient_reliability_off=False):
        ra,sa,ea,aa,pa=self._read(query_ids,h,memory,0,content_permuted,evidence_mask_break,cross_patient_reliability_off); rb,sb,eb,ab,pb=self._read(query_ids,h,memory,1,content_permuted,evidence_mask_break,cross_patient_reliability_off); rel=torch.stack([aa,ab],1).mean(1)
        ret=[] if (not pa and not pb) else [{"asd":pa[i] if i < len(pa) else [], "td":pb[i] if i < len(pb) else []} for i in range(h.shape[0])]
        out,d=self.finish(h,base_logp,ra,rb,sa,sb,rel,ea,eb,aa,ab,ret,{"memory_count":0 if not memory else len(memory.get("ids",[])),"evidence_mask_break":evidence_mask_break,"cross_patient_reliability_off":cross_patient_reliability_off}); return out,d

DDEM=CPDEM

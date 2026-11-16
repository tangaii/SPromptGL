"""Cross-fitted CAM144 evidence construction for Experiment 79.4."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from memory_common import output_layer_logp


def _digest(values):
    return hashlib.sha256(",".join(map(str,sorted(map(int,values)))).encode()).hexdigest()


def _split(ids, labels, sites, modulo, selected_remainder):
    groups=defaultdict(list)
    for pid in sorted(map(int,ids)):
        groups[(int(labels[pid]),int(sites[pid]))].append(pid)
    selected=[]; other=[]
    for key in sorted(groups):
        for rank,pid in enumerate(groups[key]):
            (selected if rank%modulo==selected_remainder else other).append(pid)
    return sorted(selected),sorted(other)


def make_partitions(eligible,labels,sites):
    a,b=_split(eligible,labels,sites,2,0)
    if set(a)&set(b) or set(a)|set(b)!=set(map(int,eligible)):
        raise RuntimeError("A/B cross-fit partition is not a disjoint cover")
    parts={}
    for name,ids in (("A",a),("B",b),("full",sorted(map(int,eligible)))):
        # Largest-remainder stratification gives exactly the nearest integer
        # 25% Support count without swapping Source and Support by accident.
        groups=defaultdict(list)
        for pid in sorted(ids):groups[(int(labels[pid]),int(sites[pid]))].append(pid)
        quotas={key:len(values)//4 for key,values in groups.items()}
        target=int(round(len(ids)*0.25))
        extra=target-sum(quotas.values())
        for key in sorted(groups,key=lambda key:(-(len(groups[key])%4),key))[:extra]:quotas[key]+=1
        support=[]
        for key,values in sorted(groups.items()):
            count=quotas[key]
            support.extend(values[min((4*j),len(values)-1)] for j in range(count))
        support=sorted(support);source=sorted(set(ids)-set(support))
        if set(source)&set(support) or set(source)|set(support)!=set(ids):
            raise RuntimeError(f"Source/Support partition invalid: {name}")
        if len(support)!=target or len(source)!=len(ids)-target:
            raise RuntimeError(f"Source/Support 75/25 size mismatch: {name}")
        parts[name]={"bank_ids":ids,"source_ids":source,"support_ids":support,
                     "bank_digest":_digest(ids),"source_digest":_digest(source),"support_digest":_digest(support)}
    parts["A"]["query_ids"]=a; parts["B"]["query_ids"]=b
    return parts


def _round_robin_candidates(ids,labels,sites,margins,limit):
    chosen=[]
    for cls in (0,1):
        by_site=defaultdict(list)
        for pid in ids:
            if int(labels[pid])==cls: by_site[int(sites[pid])].append(int(pid))
        # Interleave margin levels within each site instead of selecting only
        # the highest-confidence extreme patients.
        for s in by_site:
            ordered=sorted(by_site[s],key=lambda p:(float(margins[p]),p))
            median=len(ordered)//2
            by_site[s]=[ordered[j] for j in sorted(range(len(ordered)),key=lambda j:(abs(j-median),j))]
        out=[]
        while len(out)<limit and by_site:
            for s in sorted(list(by_site)):
                out.append(by_site[s].pop(0))
                if not by_site[s]: del by_site[s]
                if len(out)>=limit: break
        chosen.extend(out)
    return chosen


def _cosine_to_rows(q,rows):
    return rows@q/(rows.norm(dim=1)*q.norm()).clamp_min(1e-8)


def _empty(parts,version,epoch,candidate_count=0,evidence_candidates=0):
    return {"kind":"CF-CPDEM","version":int(version),"refresh_epoch":int(epoch),"ids":[],
            "label":torch.empty(0,dtype=torch.long),"reference":torch.empty((0,144)),
            "mask":torch.empty((0,144)),"direction":torch.empty((0,144)),
            "individual_margin_effects":torch.empty((0,144)),"source_site":torch.empty(0,dtype=torch.long),
            "reliability":torch.empty(0),"r_self":torch.empty(0),"r_within":torch.empty(0),
            "r_between":torch.empty(0),"r_coverage":torch.empty(0),"support_auc":torch.empty(0),
            "support_separation":torch.empty(0),"joint_drop":torch.empty(0),
            "support_same_ids":[],"support_opposite_ids":[],"source_ids":parts["source_ids"],
            "support_ids":parts["support_ids"],"bank_ids":parts["bank_ids"],
            "source_ids_digest":parts["source_digest"],"support_ids_digest":parts["support_digest"],
            "neutral_mean":torch.zeros(144),"candidate_count":int(candidate_count),
            "evidence_candidate_count":int(evidence_candidates),"accepted_count":0,
            "rejected_count":int(candidate_count)}


def _build_bank(cam,labels,base_logp,parts,sites,head,*,version,refresh_epoch,config):
    device=cam.device; sites=sites.detach().cpu(); bank_ids=parts["bank_ids"]; source_ids=parts["source_ids"]; support_ids=parts["support_ids"]
    if not bank_ids or not source_ids or not support_ids: return _empty(parts,version,refresh_epoch)
    bank=torch.tensor(bank_ids,dtype=torch.long,device=device)
    src=torch.tensor(source_ids,dtype=torch.long,device=device)
    sup=torch.tensor(support_ids,dtype=torch.long,device=device)
    y=labels.detach().long(); pred=base_logp.detach().argmax(1); mu=cam[bank].detach().mean(0)
    correct=[p for p in source_ids if int(pred[p])==int(y[p])]
    margin=torch.full((cam.shape[0],),-math.inf,device=device)
    if correct:
        c=torch.tensor(correct,dtype=torch.long,device=device)
        margin[c]=base_logp[c].gather(1,y[c,None]).squeeze(1)-base_logp[c].gather(1,(1-y[c])[:,None]).squeeze(1)
    selected=_round_robin_candidates(correct,y,sites,margin.cpu().tolist(),config["candidate_per_class"])
    if not selected: return _empty(parts,version,refresh_epoch,len(correct))
    cand=torch.tensor(selected,dtype=torch.long,device=device); h=cam[cand].detach(); yy=y[cand]
    original=(base_logp[cand].gather(1,yy[:,None]).squeeze(1)-base_logp[cand].gather(1,(1-yy)[:,None]).squeeze(1))
    importance=torch.zeros((len(selected),144),device=device)
    # Candidate chunks bound the temporary [candidate, dimension, feature] tensor.
    with torch.no_grad():
        for start in range(0,len(selected),6):
            stop=min(start+6,len(selected)); hh=h[start:stop]; variants=[]
            d=torch.arange(144,device=device)
            for alpha in config["mask_alphas"]:
                v=hh[:,None,:].expand(-1,144,-1).clone()
                v[:,d,d]=(1-float(alpha))*hh[:,d]+float(alpha)*mu[d]
                variants.append(v.reshape(-1,144))
            logits=output_layer_logp(head,torch.cat(variants)).reshape(len(config["mask_alphas"]),stop-start,144,2)
            drops=[]
            for ai in range(len(config["mask_alphas"])):
                lp=logits[ai]; labs=yy[start:stop]
                m=lp.gather(2,labs[:,None,None].expand(-1,144,1)).squeeze(2)-lp.gather(2,(1-labs)[:,None,None].expand(-1,144,1)).squeeze(2)
                drops.append(original[start:stop,None]-m)
            importance[start:stop]=torch.stack(drops).amin(0)
    masks=torch.zeros_like(importance); joint=torch.zeros((len(selected),),device=device); prelim=[]
    for r in range(len(selected)):
        good=torch.where(importance[r]>0)[0]
        if not good.numel(): continue
        order=torch.argsort(importance[r,good],descending=True,stable=True); dims=good[order[:config["evidence_dims"]]]
        masks[r,dims]=1.; masked=h[r].clone(); masked[dims]=mu[dims]
        with torch.no_grad(): lp=output_layer_logp(head,torch.stack([h[r],masked]))
        joint[r]=(lp[0,int(yy[r])]-lp[0,1-int(yy[r])])-(lp[1,int(yy[r])]-lp[1,1-int(yy[r])])
        if float(joint[r])>0: prelim.append(r)
    support_cam=cam[sup].detach(); support_y=y[sup]; support_sites=sites[sup.cpu()]
    records=[]
    for r in prelim:
        pid=selected[r]; cls=int(yy[r]); mask=masks[r]; bg=1-mask
        qbg=(h[r]-mu)*bg; bg_rows=(support_cam-mu[None])*bg[None]
        bg_score=_cosine_to_rows(qbg,bg_rows)
        same=torch.where(support_y.eq(cls))[0]; opposite=torch.where(support_y.ne(cls))[0]
        if len(same)<config["min_same_support"] or len(opposite)<config["min_opposite_support"]: continue
        ks=min(config["support_k_per_class"],len(same)); ko=min(config["support_k_per_class"],len(opposite))
        si=same[torch.topk(bg_score[same],ks,sorted=True).indices]
        oi=opposite[torch.topk(bg_score[opposite],ko,sorted=True).indices]
        qev=(h[r]-mu)*mask; sm=(support_cam[si]-mu[None])*mask[None]; om=(support_cam[oi]-mu[None])*mask[None]
        same_match=_cosine_to_rows(qev,sm); opp_match=_cosine_to_rows(qev,om)
        auc=((same_match[:,None]>opp_match[None,:]).float()+.5*(same_match[:,None]==opp_match[None,:]).float()).mean()
        sep=same_match.mean()-opp_match.mean()
        if float(auc)<config["min_support_auc"] or float(sep)<=0: continue
        direction=F.normalize((sm.mean(0)-om.mean(0)),dim=0,eps=1e-8)
        scale=((h[r]-mu)*mask).norm().clamp_min(1e-6); direction_scaled=direction*scale
        r_self=float((1-torch.exp(-joint[r].clamp_min(0))).clamp(0,1))
        r_within=float(((same_match.mean()+1)/2).clamp(0,1))
        r_between=float(((auc-.5)*2).clamp(0,1))
        sites_n=len(set(int(x) for x in support_sites[torch.cat([si,oi]).cpu()].tolist()))
        r_coverage=min(1.,(len(si)+len(oi))/(2*config["support_k_per_class"])) * min(1.,sites_n/3.)
        reliability=r_self*r_within*r_between*r_coverage
        if not math.isfinite(reliability) or reliability<=0: continue
        records.append({"row":r,"pid":pid,"label":cls,"rel":reliability,"self":r_self,"within":r_within,
                        "between":r_between,"coverage":r_coverage,"auc":float(auc),"sep":float(sep),
                        "joint":float(joint[r]),"same_ids":[support_ids[int(i)] for i in si.cpu().tolist()],
                        "opp_ids":[support_ids[int(i)] for i in oi.cpu().tolist()],"direction":direction_scaled.cpu()})
    kept=[]
    for cls in (0,1):
        rr=[r for r in records if r["label"]==cls]; rr.sort(key=lambda r:(-r["rel"],-r["joint"],r["pid"]))
        kept.extend(rr[:config["candidate_per_class"]])
    rows=torch.tensor([r["row"] for r in kept],dtype=torch.long,device=device)
    if not kept: return _empty(parts,version,refresh_epoch,len(selected),len(prelim))
    return {"kind":"CF-CPDEM","version":int(version),"refresh_epoch":int(refresh_epoch),
            "ids":[r["pid"] for r in kept],"label":yy[rows].cpu(),"reference":h[rows].cpu(),
            "mask":masks[rows].cpu(),"direction":torch.stack([r["direction"] for r in kept]),
            "individual_margin_effects":importance[rows].cpu(),"source_site":sites[cand[rows].cpu()],
            "reliability":torch.tensor([r["rel"] for r in kept]),"r_self":torch.tensor([r["self"] for r in kept]),
            "r_within":torch.tensor([r["within"] for r in kept]),"r_between":torch.tensor([r["between"] for r in kept]),
            "r_coverage":torch.tensor([r["coverage"] for r in kept]),"support_auc":torch.tensor([r["auc"] for r in kept]),
            "support_separation":torch.tensor([r["sep"] for r in kept]),"joint_drop":torch.tensor([r["joint"] for r in kept]),
            "support_same_ids":[r["same_ids"] for r in kept],"support_opposite_ids":[r["opp_ids"] for r in kept],
            "source_ids":source_ids,"support_ids":support_ids,"bank_ids":bank_ids,
            "source_ids_digest":parts["source_digest"],"support_ids_digest":parts["support_digest"],
            "neutral_mean":mu.cpu(),"candidate_count":len(selected),"correct_source_count":len(correct),
            "evidence_candidate_count":len(prelim),"accepted_count":len(kept),"rejected_count":len(selected)-len(kept),
            "source_site_coverage":len(set(int(sites[p]) for p in source_ids))}


def build_crossfit_memories(cam,labels,base_logp,partitions,sites,head,*,version,refresh_epoch,config):
    memories={name:_build_bank(cam,labels,base_logp,parts,sites,head,version=version,refresh_epoch=refresh_epoch,config=config)
              for name,parts in partitions.items() if name in ("A","B","full")}
    if set(memories["A"]["source_ids"]) & set(memories["B"]["source_ids"]):
        raise RuntimeError("A/B memory sources overlap")
    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,mem in memories.items():
        ix=torch.tensor(mem["bank_ids"],dtype=torch.long,device=cam.device)
        features=cam[ix].detach().cpu().clone()
        mem["source_cam_digest"]=hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest()
        mem["eligible_patient_ids_digest"]=partitions[name]["bank_digest"]
        mem["source_code_sha256"]=code_sha
        # Only the current memory version is retained. These minimal Bank-local
        # inputs let independent audit rebuild the actual refresh-time evidence.
        mem["build_inputs"]={"cam":features,"logp":base_logp[ix].detach().cpu().clone(),
            "labels":labels[ix].detach().cpu().clone(),"sites":sites[ix].detach().cpu().clone(),
            "head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
    return memories


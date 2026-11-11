"""E76.7 DPC/BAC. Discrete EMA retrieval, live within-step prototype values."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from path_adapter import legacy_module


class CJEBCSPC(nn.Module):
    _site_diverse_indices = staticmethod(legacy_module.BCSPC._site_diverse_indices)
    _confusion_weights = legacy_module.BCSPC._confusion_weights
    semantic_slots = legacy_module.BCSPC.semantic_slots

    def __init__(self, brainprompt, spromptgl):
        super().__init__()
        assert brainprompt.shape == (2,1024) and spromptgl.shape == (8,1024)
        self.register_buffer('brainprompt_vectors', F.normalize(brainprompt.detach().float(),dim=-1))
        self.register_buffer('spromptgl_vectors', F.normalize(spromptgl.detach().float(),dim=-1))
        self.semantic_projection = nn.Sequential(nn.Linear(2048,16),nn.GELU(),nn.Linear(16,144))
        self.query_projection = nn.Linear(36,36,bias=False)
        self.prototype_projection = nn.Linear(36,36,bias=False)
        self.reliable_text_projection = nn.Linear(36,36,bias=False)
        self.boundary_text_projection = nn.Linear(36,36,bias=False)
        self.modality_gate = nn.Sequential(nn.Linear(12,16),nn.GELU(),nn.Linear(16,4))
        self.residual_network = nn.Sequential(nn.Linear(87,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    def _pool(self, q, current, memory, ids, weights, sites, slot, query_ids):
        batch = q.shape[0]
        selected = torch.empty((batch,0),dtype=torch.long,device=q.device)
        pooling = q.new_zeros((batch,0))
        if ids.numel()==0:
            return (q.new_zeros((batch,36)),q.new_zeros(batch),
                    torch.full((batch,8),-1,dtype=torch.long,device=q.device),
                    q.new_zeros((batch,8)),0,0)
        # The previous-epoch EMA selects candidates; live features form the pool.
        with torch.no_grad():
            mem_slots = F.normalize(memory['features'][ids,slot],dim=-1)
            index_query = q.detach().clone()
            if query_ids is not None:
                query_ids = torch.as_tensor(query_ids,device=q.device,dtype=torch.long)
                observed = memory['observed'][query_ids]
                if observed.any():
                    index_query[observed] = F.normalize(memory['features'][query_ids[observed],slot],dim=-1)
            sim_index = index_query @ mem_slots.T
            positions, fallback = self._site_diverse_indices(sim_index,ids,sites,8)
            chosen = ids[positions]
            if query_ids is not None and (chosen==query_ids[:,None]).any():
                raise RuntimeError('illegal prototype self retrieval')
            chosen_sites = sites[chosen]
            counts = (chosen_sites[:,:,None]==chosen_sites[:,None,:]).sum(-1).max(-1).values
            violations = int(((counts>2)&(fallback==0)).sum())
        # Supports are live features from the SAME original CAM forward.
        support_values = current[chosen,slot]
        similarities = (q[:,None,:]*support_values).sum(-1)
        reliabilities = weights[chosen].clamp_min(1e-8)
        attention = torch.softmax(similarities/0.20+reliabilities.log(),dim=-1)
        prototype = F.normalize((attention[...,None]*support_values).sum(1),dim=-1)
        selected = torch.full((batch,8),-1,dtype=torch.long,device=q.device)
        pooling = q.new_zeros((batch,8))
        selected[:,:chosen.shape[1]] = chosen
        pooling[:,:attention.shape[1]] = attention
        return prototype,(attention*reliabilities).sum(-1),selected,pooling,int(fallback.sum()),violations

    def forward(self, cam144, query_ids, memory, support_ids, sites, base_logp,
                *, residual_off=False, boundary_off=False, semantic_swap=False, reliable_only=False):
        if cam144.shape != (871,144):
            raise ValueError('requires original Outputlayer input [871,144]')
        query_ids = torch.as_tensor(query_ids,device=cam144.device,dtype=torch.long)
        current = F.normalize(cam144.reshape(871,4,36),dim=-1)
        query = current[query_ids]
        support = torch.as_tensor(support_ids,device=cam144.device,dtype=torch.long)
        support = support[memory['observed'][support] & (memory['labels'][support]>=0)]
        semantic = self.semantic_slots()
        if semantic_swap: semantic=semantic.flip(0)
        proto, sim, counts, reliability, selected, pooling, candidates = [],[],[],[],[],[],[]
        fallbacks=violations=invalid=0
        for cls in (0,1):
            ids = support[memory['labels'][support]==cls]
            predicted = memory['probabilities'][ids].argmax(-1)
            weight_all = query.new_zeros(871)
            weight_all[ids] = (predicted==cls).to(query.dtype)*((memory['probabilities'][ids,cls]-0.5)/0.5).clamp(0,1)
            rel_ids = ids[weight_all[ids]>0]
            class_proto=[]; class_sim=[]; class_count=[]; class_rel=[]; class_ids=[]; class_pool=[]; class_candidates=[]
            for slot in range(4):
                if boundary_off or reliable_only:
                    bnd_ids,bnd_all=rel_ids,weight_all
                else:
                    bnd_ids,bnd_weights,bad = self._confusion_weights(memory,support,ids,cls,slot)
                    invalid+=bad
                    good=bnd_weights>0
                    bnd_ids,bnd_weights=bnd_ids[good],bnd_weights[good]
                    bnd_all=query.new_zeros(871);bnd_all[bnd_ids]=bnd_weights
                role_proto=[];role_sim=[];role_count=[];role_rel=[];role_ids=[];role_pool=[];role_candidate=[]
                for ids_role,weights in ((rel_ids,weight_all),(bnd_ids,bnd_all)):
                    p,r,chosen,pw,fb,vi=self._pool(query[:,slot],current,memory,ids_role,weights,sites,slot,query_ids)
                    fallbacks+=fb;violations+=vi
                    role_proto.append(p)
                    role_sim.append((query[:,slot]*p).sum(-1))
                    role_count.append((chosen>=0).sum(-1).to(query.dtype))
                    role_rel.append(r);role_ids.append(chosen);role_pool.append(pw)
                    role_candidate.append(query.new_full((query.shape[0],),len(ids_role)))
                class_proto.append(torch.stack(role_proto,1));class_sim.append(torch.stack(role_sim,-1))
                class_count.append(torch.stack(role_count,-1));class_rel.append(torch.stack(role_rel,-1))
                class_ids.append(torch.stack(role_ids,1));class_pool.append(torch.stack(role_pool,1))
                class_candidates.append(torch.stack(role_candidate,-1))
            proto.append(torch.stack(class_proto,1));sim.append(torch.stack(class_sim,1))
            counts.append(torch.stack(class_count,1));reliability.append(torch.stack(class_rel,1))
            selected.append(torch.stack(class_ids,1));pooling.append(torch.stack(class_pool,1))
            candidates.append(torch.stack(class_candidates,1))
        # [query,class,slot,role,36], [query,class,slot,role]
        prototypes=torch.stack(proto,1); similarities=torch.stack(sim,1)
        counts=torch.stack(counts,1);reliability=torch.stack(reliability,1)
        coverage=(counts>0).to(query.dtype)
        qk=self.query_projection(query)[:,None,:,None,:]
        role_text=torch.stack((self.reliable_text_projection(semantic),self.boundary_text_projection(semantic)),2)
        keys=self.prototype_projection(prototypes)+role_text[None]
        logits=(qk*keys).sum(-1)/math.sqrt(36)
        attention=torch.softmax(logits.masked_fill(coverage==0,-1e4),dim=-1)
        attention=attention*coverage
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        if reliable_only:
            attention=torch.stack((coverage[...,0],torch.zeros_like(coverage[...,0])),-1)
        evidence=(attention*similarities).sum(-1)
        difference=evidence[:,0]-evidence[:,1]
        reliable_difference=similarities[:,0,:,0]-similarities[:,1,:,0]
        boundary_difference=similarities[:,0,:,1]-similarities[:,1,:,1]
        modality_weights=torch.softmax(self.modality_gate(torch.cat((reliable_difference,boundary_difference,difference),-1)),-1)
        d=(modality_weights*difference).sum(-1)
        base=base_logp[query_ids,0]-base_logp[query_ids,1]
        # No raw-CAM-to-classifier bypass. Base margin is detached in z AND scale.
        z=torch.cat((base.detach()[:,None],base.detach().abs()[:,None],reliable_difference,boundary_difference,
                     reliable_difference-boundary_difference,difference,attention.flatten(1),coverage.flatten(1),
                     (counts/8).flatten(1),reliability.flatten(1),d[:,None],modality_weights),-1)
        assert z.shape[-1]==87
        activation=self.residual_network(z).squeeze(-1)
        delta=(base.detach().abs()+0.5)*torch.tanh(activation)
        if residual_off: delta=torch.zeros_like(delta)
        logp=correct_logp(base_logp[query_ids],delta)
        return logp,{
            'delta':delta,'activation':activation,'base_score':base,'final_score':base+delta,
            'evidence':evidence,'reliable_similarity':similarities[...,0],'boundary_similarity':similarities[...,1],
            'reliable_difference':reliable_difference,'boundary_difference':boundary_difference,'slot_difference':difference,
            'modality_weights':modality_weights,'attention':attention,'support_topk':counts,'support_coverage':coverage,
            'support_reliability':reliability,'candidate_count':torch.stack(candidates,1),
            'selected_ids':torch.stack(selected,1),'pool_weights':torch.stack(pooling,1),
            'prototype_role_cosine':(prototypes[:,:,:,0]*prototypes[:,:,:,1]).sum(-1),
            'final_boundary_evidence':d,'retrieval_fallbacks':fallbacks,'site_cap_violations':violations,
            'invalid_confusion_candidates':invalid,
        }


def correct_logp(base_logp,delta):
    # Increment log odds by delta while retaining the original Outputlayer path.
    shifted=base_logp+torch.stack((delta/2,-delta/2),-1)
    normalized=F.log_softmax(shifted,dim=-1)
    # Exact zero-residual values, without suppressing the derivative of delta.
    if bool((delta.detach()==0).all()):
        return normalized+(base_logp-normalized).detach()
    return normalized


def init_isolated(brainprompt,spromptgl,fold):
    devices=list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(767101+fold)
        torch.cuda.manual_seed_all(767101+fold)
        return CJEBCSPC(brainprompt,spromptgl)


def branch_losses(final_logp,diagnostics,labels,weights):
    cls=F.nll_loss(final_logp,labels,weight=weights)
    t=torch.where(labels==0,1.0,-1.0)
    boundary=F.softplus(0.15-t*diagnostics['final_boundary_evidence']).mean()
    keep=(diagnostics['base_score'].detach()>=0).long()==(labels==0).long()
    protect=(keep.to(final_logp.dtype)*F.relu(0.2-t*diagnostics['final_score']).square()).mean()
    residual=diagnostics['delta'].square().mean()
    return cls+0.03*boundary+0.10*protect+0.02*residual,{'cls_final':cls,'boundary':boundary,'protect':protect,'residual':residual}

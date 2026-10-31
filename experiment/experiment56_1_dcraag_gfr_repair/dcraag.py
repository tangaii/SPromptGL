"""Repaired DCRAG-GFR: differentiable candidate edge scores and bounded propagation."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale=float(scale); return value.view_as(value)
    @staticmethod
    def backward(ctx, grad): return -ctx.scale*grad, None

def gradient_reverse(value, scale): return _GradientReverse.apply(value, scale)

def disease_condition(prob, pairs):
    pi,pj=prob[pairs[:,0]].detach(),prob[pairs[:,1]].detach()
    out=torch.stack((pi[:,0]*pj[:,0],pi[:,1]*pj[:,1],pi[:,0]*pj[:,1]+pi[:,1]*pj[:,0]),1)
    return out/out.sum(1,keepdim=True).clamp_min(1e-8)

class PairRelationEncoder(nn.Module):
    def __init__(self,in_dim,hidden=32):
        super().__init__(); self.patient=nn.Sequential(nn.Linear(in_dim,hidden),nn.GELU(),nn.LayerNorm(hidden)); self.pair=nn.Sequential(nn.Linear(hidden*2+1,64),nn.GELU(),nn.Linear(64,hidden),nn.GELU()); self.shared_classifier=nn.Linear(16,3); self.private_classifier=nn.Linear(16,3); self.classifier=self.shared_classifier; self.edge=nn.Sequential(nn.Linear(hidden+4,32),nn.GELU(),nn.Linear(32,2))
    def encode_nodes(self,x): return self.patient(x)
    def encode_pairs(self,h,pairs):
        a,b=h[pairs[:,0]],h[pairs[:,1]]; sym=torch.cat((torch.abs(a-b),a*b,F.cosine_similarity(a,b,dim=1).unsqueeze(1)),1); code=self.pair(sym); return code,torch.stack((self.shared_classifier(code[:,:16]),self.private_classifier(code[:,16:])),1)

def _row_normalize(a,self_loop=False):
    if self_loop: a=a+torch.eye(a.shape[0],dtype=a.dtype,device=a.device)
    a=a.clamp_min(0); return a/a.sum(1,keepdim=True).clamp_min(1e-8)

def _row_scale(msg,ref):
    return msg/msg.norm(2,dim=1,keepdim=True).clamp_min(1e-6)*ref.norm(2,dim=1,keepdim=True).detach().clamp_min(1e-6)

class DCRAGGFR(nn.Module):
    def __init__(self,e1_gfr,dims,utils_module):
        super().__init__(); self.graphs=e1_gfr.graphs; self.gcn=e1_gfr.gcn; self.fusion_weight=e1_gfr.fusion_weight; self.utils=utils_module; self.dims=tuple(dims); self.rho=.25
        self.relations=nn.ModuleList([PairRelationEncoder(d) for d in dims]); self.difference=nn.ModuleList([nn.Sequential(nn.Linear(sum(dims),32,bias=False),nn.Tanh(),nn.Linear(32,sum(dims),bias=False)) for _ in dims]); self.eta_raw=nn.Parameter(torch.full((4,),math.log(.1/.9)))
        gate_in=sum(dims)*5+12; self.gate=nn.Sequential(nn.Linear(gate_in,64),nn.GELU(),nn.Linear(64,4)); nn.init.zeros_(self.gate[0].weight); nn.init.zeros_(self.gate[0].bias); nn.init.zeros_(self.gate[2].weight); nn.init.zeros_(self.gate[2].bias)
        self.beta_gate=nn.Linear(gate_in,1); nn.init.zeros_(self.beta_gate.weight); nn.init.constant_(self.beta_gate.bias,math.log(.4/.6)); self.adversary=nn.Sequential(nn.Linear(51,48),nn.GELU(),nn.Linear(48,4)); self.last={}
    def reference_bypass(self,parts):
        raw=torch.cat(parts,1); fused,adjs,msgs=raw,[],[]
        for m,x in enumerate(parts):
            adj=self.graphs[m](x); adjs.append(adj); eye=torch.eye(adj.shape[0],device=adj.device,dtype=adj.dtype); msg=self.gcn[m](raw,self.utils.normalize_adj(adj+eye)); msgs.append(msg); fused=fused+torch.sigmoid(self.fusion_weight[m])*msg
        return fused,adjs,{"adjs":adjs,"msgs":msgs,"fused":fused,"bypass":True}
    def forward(self,parts,candidate_edges,edge_prior,patient_prob,relation_pairs,relation_targets,relation_condition,reliability,grl_scale=0.,intervention=None,reference_bypass=False):
        if reference_bypass:return self.reference_bypass(parts)
        intervention=intervention or {}; raw=torch.cat(parts,1); n=raw.shape[0]; node_codes=[r.encode_nodes(x) for r,x in zip(self.relations,parts)]; rel_codes=[]; rel_logits=[]; private_logits=[]; adv_codes=[]; edge_same=[]; edge_diff=[]; edge_logits=[]; selected_edges=[]; A_same=[]; A_diff=[]; same_msgs=[]; diff_msgs=[]; msgs=[]; raw_adjs=[]; stability=[]; confs=[]
        used_prob=intervention.get('patient_prob',patient_prob).detach()
        for m,(rel,h,edges,prior) in enumerate(zip(self.relations,node_codes,candidate_edges,edge_prior)):
            edge_code,_=rel.encode_pairs(h,edges); pair_code,pair_heads=rel.encode_pairs(h,relation_pairs); shared_logit,private_logit=pair_heads[:,0],pair_heads[:,1]; rel_codes.append(pair_code); rel_logits.append(shared_logit); private_logits.append(private_logit); adv_codes.append(pair_code)
            proj=F.normalize(self.graphs[m].p(parts[m]),dim=1); sim=(proj[edges[:,0]]*proj[edges[:,1]]).sum(1); cond=disease_condition(used_prob,edges); eprob=rel.shared_classifier(edge_code[:,:16]).softmax(-1); score=.45*((sim+1)*.5)+.35*eprob.max(1).values+.20*prior.clamp(0,1)
            chosen=[]
            for i in range(n):
                idx=torch.nonzero(edges[:,0]==i,as_tuple=False).flatten(); k=min(16,idx.numel()); _,order=torch.topk(score[idx],k=k,largest=True,sorted=True); chosen.append(idx[order])
            chosen=torch.cat(chosen); ce,cp,cc=edges[chosen],prior[chosen],edge_code[chosen]; ccond,cprob=cond[chosen],eprob[chosen]; csim=sim[chosen]; confidence=(1-(cprob.clamp_min(1e-8).log()*cprob).sum(1)/math.log(3)).clamp(.05,1.); edge_score=rel.edge(torch.cat((cc,ccond,((csim+1)*.5).unsqueeze(1)),1)).sigmoid(); base=(.15+.85*((csim+1)*.5).clamp(0,1))*confidence*cp.clamp_min(1e-4); same=base*(.1+.9*(cprob[:,0]+cprob[:,1]))*edge_score[:,0]; diff=base*(.1+.9*cprob[:,2])*edge_score[:,1]
            if intervention.get('neutral_edges'):same=cp*.5;diff=cp*.5
            if intervention.get('disable_edges'):same=torch.zeros_like(same);diff=torch.zeros_like(diff)
            ii,jj=ce[:,0],ce[:,1]; smat=raw.new_zeros((n,n)).index_put((ii,jj),same,accumulate=True); dmat=raw.new_zeros((n,n)).index_put((ii,jj),diff,accumulate=True)
            if intervention.get('self_only') or intervention.get('disable_messages') or m in intervention.get('disabled_modalities',()): snorm=torch.eye(n,device=raw.device,dtype=raw.dtype); dnorm=raw.new_zeros((n,n))
            else: snorm=_row_normalize(smat,True); dnorm=_row_normalize(dmat,False)
            raw_adj=self.graphs[m](parts[m]); raw_adjs.append(raw_adj); msg_s=_row_scale(self.gcn[m](raw,snorm),raw); delta=self.difference[m](raw[ii]-raw[jj]); msg_d=raw.new_zeros(raw.shape).index_add(0,ii,delta*dnorm[ii,jj].unsqueeze(1)); msg_d=_row_scale(msg_d,raw); eta=.3*torch.sigmoid(self.eta_raw[m]); dm=eta*msg_d; mm=msg_s+dm
            A_same.append(snorm);A_diff.append(dnorm);edge_same.append(same);edge_diff.append(diff);edge_logits.append(eprob[chosen]);selected_edges.append(ce);same_msgs.append(msg_s);diff_msgs.append(dm);msgs.append(mm);confs.append(confidence); row_p=cp.new_zeros(n).index_add(0,ii,cp.clamp_min(1e-6)); pprior=cp.clamp_min(1e-6)/row_p[ii].clamp_min(1e-8); q=(same+diff)/(same+diff).new_zeros(n).index_add(0,ii,same+diff)[ii].clamp_min(1e-8); stability.append((q*(q.clamp_min(1e-8).log()-pprior.log())).mean())
        rel_logits_t=torch.stack(rel_logits); private_t=torch.stack(private_logits); rel_loss_each=F.cross_entropy(rel_logits_t.reshape(-1,3),relation_targets.repeat(4),reduction='none').view(4,-1).mean(1); rel_loss=(rel_loss_each*reliability).sum()/reliability.sum().clamp_min(1e-8)+.2*F.cross_entropy(private_t.reshape(-1,3),relation_targets.repeat(4)); rel_probs=rel_logits_t.softmax(-1); consensus=(rel_probs*reliability[:,None,None]).sum(0)/reliability.sum().clamp_min(1e-8); consistency_each=(rel_probs.clamp_min(1e-8)*(rel_probs.clamp_min(1e-8).log()-consensus.clamp_min(1e-8).log())).sum(-1).mean(1); consistency=(consistency_each*reliability).sum()/reliability.sum().clamp_min(1e-8)
        proto_loss=raw.new_zeros(())
        for cls in range(3):
            means=[]
            for m in range(4):
                z=F.normalize(rel_codes[m][:,:16],dim=1); mask=relation_targets==cls; means.append(z[mask].mean(0) if mask.any() else z.mean(0))
            target=sum(means)/4; proto_loss+=sum(reliability[m]*(means[m]-target.detach()).pow(2).mean() for m in range(4))/reliability.sum().clamp_min(1e-8)
        source_rows=[]
        for code in adv_codes:
            shared=code[:,:16]; c=relation_condition.detach(); source_rows.append(torch.cat(((gradient_reverse(shared,grl_scale).unsqueeze(2)*c.unsqueeze(1)).reshape(-1,48),c),1))
        source_logits=self.adversary(torch.stack(source_rows).reshape(-1,51)); source_target=torch.arange(4,device=raw.device).view(4,1).expand(4,relation_pairs.shape[0]).reshape(-1); adv_each=F.cross_entropy(source_logits,source_target,reduction='none').view(4,-1).mean(1); adv_loss=(adv_each*reliability).sum()/reliability.sum().clamp_min(1e-8)
        gate_input=torch.cat([F.normalize(raw,dim=1),*[F.normalize(x,dim=1) for x in msgs]],1); summary=torch.cat([torch.stack([x.norm(2,dim=1) for x in msgs],1),torch.stack([x.mean().expand(n) for x in confs],1),reliability.view(1,4).expand(n,4)],1); gf=torch.cat((gate_input,summary),1); gate=self.gate(gf).softmax(-1); 
        if intervention.get('freeze_gate') is not None: gate=intervention['freeze_gate']
        weights=gate*torch.sigmoid(self.fusion_weight).view(1,4); mraw=sum(weights[:,m:m+1]*msgs[m] for m in range(4)); rn=raw.norm(2,dim=1,keepdim=True).clamp_min(1e-6); mn=mraw.norm(2,dim=1,keepdim=True); bound=torch.minimum(torch.ones_like(mn),.25*rn/(mn+1e-8)); mbound=mraw*bound; beta=torch.sigmoid(self.beta_gate(gf)); fused=raw+beta*mbound
        self.last={"adjs":raw_adjs,"msgs":msgs,"same_msgs":same_msgs,"diff_msgs":diff_msgs,"fused":fused,"A_same":A_same,"A_diff":A_diff,"relation_logits":rel_logits_t,"private_logits":private_t,"relation_probs":rel_probs,"edge_same":edge_same,"edge_diff":edge_diff,"edge_logits":edge_logits,"selected_edges":selected_edges,"gate":gate,"beta":beta,"beta_bound":bound,"mraw":mraw,"mbound":mbound,"message_norm_before_bound":mn,"node_codes":node_codes,"relation_codes":rel_codes,"adv_logits":source_logits,"adversary_codes":adv_codes,"adv_targets":source_target,"loss_relation":rel_loss,"loss_adversarial":adv_loss,"loss_prototype":proto_loss,"loss_consistency":consistency,"loss_edge":torch.stack(stability).mean(),"loss_relation_each":rel_loss_each,"loss_adversarial_each":adv_each,"grl_scale":grl_scale,"patient_prob":used_prob,"bound_active":(bound<.999999).float()}
        return fused,raw_adjs,self.last

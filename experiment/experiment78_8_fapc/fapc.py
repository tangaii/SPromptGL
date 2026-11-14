from __future__ import annotations
import hashlib, math
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

def unit(x):
    return F.normalize(x, dim=-1, eps=1e-8)

def _logit(p):
    return math.log(p/(1-p))

class FAPC(nn.Module):
    """Frozen-anchor prototype calibration with signed ASD/TD auxiliary logits.

    The original DARG/CAM logits remain the formal prediction.  This module
    only supplies an auxiliary disease supervision path through CAM144.
    """
    def __init__(self, anchors: torch.Tensor|None, classes: torch.Tensor|None,
                 mode: str, prompt_source: str):
        super().__init__()
        if mode not in ('visual','text'): raise ValueError(mode)
        self.mode=str(mode); self.prompt_source=str(prompt_source)
        self.dim=32; self.prompt_count=5; self.temperature=0.25; self.scale=10.0
        self.beta_max=0.2
        self.patient_projection=nn.Linear(144,32,bias=False)
        self.prototype_direction=nn.Linear(32,32,bias=False)
        if self.mode=='text':
            if anchors is None or tuple(anchors.shape)!=(10,32): raise ValueError('ten [10,32] anchors required')
            if classes is None or tuple(classes.shape)!=(10,): raise ValueError('ten classes required')
            self.register_buffer('anchors',anchors.detach().float().clone())
            self.register_buffer('prompt_classes',classes.detach().long().clone())
            self.calibration_raw=nn.Parameter(torch.tensor([_logit(.5)]*10,dtype=torch.float32))
        else:
            self.register_buffer('anchors',torch.empty(0,32))
            self.register_buffer('prompt_classes',torch.empty(0,dtype=torch.long))
    @property
    def calibration_strength(self):
        if self.mode!='text': return self.patient_projection.weight.new_zeros((0,))
        return self.beta_max*torch.sigmoid(self.calibration_raw)
    def anchor_sha(self):
        return hashlib.sha256(self.anchors.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def _support_prototypes(self, a, labels, support):
        ids=[]; dirs=[]
        for c in (0,1):
            cur=support[labels[support]==c]; opp=support[labels[support]!=(c)]
            if cur.numel()==0 or opp.numel()==0: raise RuntimeError('both support classes are required')
            mu=a[cur].mean(0); muo=a[opp].mean(0)
            d=unit(mu-muo); v=unit(self.prototype_direction(d))
            ids.append(mu); dirs.append(v)
        return torch.stack(ids), torch.stack(dirs)
    def _signed_semantic(self,z,dirs,query,labels,class_weights,training,intervention):
        anchors=self.anchors
        classes=self.prompt_classes
        if intervention=='class_swap':
            classes=1-classes
        if intervention=='no_calibration':
            q=anchors
            beta=self.calibration_strength.detach()*0
        else:
            beta=self.calibration_strength
            qrows=[]
            for i in range(10):
                qrows.append(unit(anchors[i]+beta[i]*dirs[int(classes[i])]))
            q=torch.stack(qrows)
        if intervention=='anchor_only': q=anchors
        scores=torch.stack([z[query]@q[classes==c].T for c in (0,1)],dim=1)
        pooled=self.temperature*torch.logsumexp(scores/self.temperature,dim=-1)-self.temperature*math.log(self.prompt_count)
        logits=self.scale*pooled
        logp=F.log_softmax(logits,dim=-1)
        if training:
            loss=F.nll_loss(logp,labels[query],weight=class_weights)
        else: loss=z.new_zeros(())
        keep=(1-(q*anchors).sum(-1)).mean() if intervention not in ('no_calibration','anchor_only') else z.new_zeros(())
        return q,beta, scores, pooled, logits, logp, loss, keep
    def forward(self,h,labels,support_ids,query_ids,class_weights,base_logp,*,training,intervention='full') -> Dict[str,torch.Tensor]:
        if tuple(h.shape)!=(871,144): raise RuntimeError(f'FAPC requires CAM144 [871,144], got {tuple(h.shape)}')
        device=h.device
        support=torch.as_tensor(support_ids,dtype=torch.long,device=device).sort().values
        query=torch.as_tensor(query_ids,dtype=torch.long,device=device)
        labels=labels.to(device=device,dtype=torch.long)
        if support.numel()==0 or torch.unique(support).numel()!=support.numel(): raise RuntimeError('invalid support')
        u=self.patient_projection(h); z=unit(u); a=F.layer_norm(u,(self.dim,))
        means, dirs=self._support_prototypes(a,labels,support)
        visual_scores=torch.stack([z[query]@dirs[c] for c in (0,1)],dim=-1)
        visual_logits=self.scale*visual_scores; visual_logp=F.log_softmax(visual_logits,dim=-1)
        if training:
            visual_loss=F.nll_loss(visual_logp,labels[query],weight=class_weights)
        else: visual_loss=z.new_zeros(())
        if self.mode=='text':
            q,beta,pscores,pooled,sem_logits,sem_logp,aux,keep=self._signed_semantic(
                z,dirs,query,labels,class_weights,training,intervention)
            semantic_logits=sem_logits; semantic_logp=sem_logp; aux_loss=aux
            semantic_scores=pooled; prompt_scores=pscores
        else:
            q=z.new_zeros((10,32)); beta=z.new_zeros((0,)); semantic_logits=visual_logits; semantic_logp=visual_logp
            semantic_scores=visual_scores; prompt_scores=z.new_zeros((len(query),2,5)); aux_loss=visual_loss; keep=z.new_zeros(())
        # Full formal prediction remains original DARG output.
        final_logp=base_logp[query]
        # compact support diagnostics; no attention bank is used in FAPC.
        n=h.shape[0]
        return {
          'final_logp':final_logp,'final_logits':base_logp[query],'base_logp':base_logp,
          'semantic_logits':semantic_logits,'semantic_logp':semantic_logp,'semantic_loss':aux_loss,
          'classification_loss':aux_loss,'aux_loss':aux_loss,'keep_loss':keep,
          'prototype_means':means,'prototypes':dirs,'prototype_similarity':dirs@dirs.T,
          'patient_projection':z,'attention_features':a,'prompt_base':self.anchors,
          'prompt_calibrated':q,'prompt_final':q,'calibration_strength':beta,
          'semantic_scores':semantic_scores,'semantic_margin':semantic_logits[:,0]-semantic_logits[:,1],
          'prompt_scores':prompt_scores,'support_ids':support,'support_labels':labels[support],
          'query_ids':query,'prompt_classes':self.prompt_classes,'prompt_source':self.prompt_source,
          'enable_a':True,'enable_b':False,'mode':self.mode,
          'gradient_path_note':'signed auxiliary path through live CAM144; no patient attention',
          'attention_weights':z.new_zeros((2,5,1,max(int(support.numel()),1))),
          'attention_logits':z.new_zeros((2,5,1,max(int(support.numel()),1))),
          'attention_entropy':z.new_zeros((2,5,1)), 'patient_teacher':z.new_zeros((n,2)),
          'gamma':z.new_zeros(()), 'attention_tau':z.new_tensor(0.0),
          'anchor_sha':self.anchor_sha(),
        }

def initialize(anchors,classes,mode,prompt_source,fold,device):
    devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(788000+int(fold))
        module=FAPC(anchors,classes,mode,prompt_source)
        # deterministic common initialization for patient/prototype maps
        for p in module.parameters():
            if p.ndim>1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
        if hasattr(module,'calibration_raw'): nn.init.constant_(module.calibration_raw,_logit(.5))
    return module.to(device)

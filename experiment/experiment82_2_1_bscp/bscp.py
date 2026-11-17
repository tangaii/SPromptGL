"""BSCP. Deterministic signed actions, legal-row teacher and isolated auxiliaries.

Read-only E82.2 helpers retain the exact Original-O eigenspace and query geometry.
Their actual file SHA is locked in manifest before formal training.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

REFERENCE = Path('/model/spromptgl/experiment/experiment82_2_psce/psce.py')
_spec = importlib.util.spec_from_file_location('bscp_locked_psce_helpers',REFERENCE)
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
_spec.loader.exec_module(legacy)
EPS, PROMPTS, GROUPS = legacy.EPS, legacy.PROMPTS, legacy.GROUPS
build_spectra, text_geometry, smooth_limit = legacy.build_spectra, legacy.text_geometry, legacy.smooth_limit
teacher_snapshot = legacy.teacher_snapshot

class BSCP(legacy.PSCE):
    def __init__(self,spectrum,text_cache,group,fold):
        super().__init__(spectrum,text_cache,group,fold)
        self.policy = nn.Sequential(nn.Linear(71,32),nn.GELU(),nn.Linear(32,2))
        nn.init.zeros_(self.policy[2].weight)
        nn.init.zeros_(self.policy[2].bias)

    def forward(self,z,intervention='full'):
        assert z.shape == (871,144)
        assert intervention in ('full','semantic_off','correction_off','residual_off','action_neutral','noop')
        h=(z-self.mu)@self.U
        hh=h/torch.sqrt(self.lam+EPS)
        ll,nu=self.loglam.expand_as(h),self.nu.expand_as(h)
        tokens=self.norm(self.direction(self.U.T)[None]+self.evidence(torch.stack([hh,hh.abs(),ll,nu],-1)))
        scores=torch.einsum('kd,njd->nkj',self.q0,self.key(tokens))/math.sqrt(32)
        null=(self.q0@self.null/math.sqrt(32))[None,:,None].expand(len(z),-1,-1)
        a=torch.cat([scores,null],-1).div(.5).softmax(-1)
        coverage=1-(1-a[:,:,:16]).prod(1)
        ws=coverage/(coverage.sum(-1,keepdim=True)+EPS)
        entropy=-(ws*(ws+EPS).log()).sum(-1)/math.log(16)
        n=a[:,:,16].mean(-1)
        wp=self.spectral(torch.stack([hh.abs(),ll,nu],-1)).squeeze(-1).softmax(-1)
        beta=((1-n)*(1-entropy).clamp(.2,.8)).detach()
        w=beta[:,None]*ws+(1-beta[:,None])*wp
        context=torch.einsum('nkj,kd->njd',a[:,:,:16],self.q0)/(a[:,:,:16].sum(1)[:,:,None]+EPS)
        if intervention=='semantic_off':
            ws=wp;w=wp;beta=torch.zeros_like(beta);context=torch.zeros_like(context)
        scalars=torch.stack([hh,hh.abs(),ws,wp,w,ll,nu],-1)
        policy_input=torch.cat([tokens,context,scalars],-1)
        logits=self.policy(policy_input)
        p=torch.cat([logits,torch.zeros_like(logits[:,:,:1])],-1).softmax(-1)
        if intervention=='action_neutral': p=torch.full_like(p,1/3)
        s=p[:,:,0]-p[:,:,1]
        if intervention in ('correction_off','action_neutral'): s=torch.zeros_like(s)
        base=((w-1/16)*h)@self.U.T
        corr=((w*s)*h)@self.U.T
        total=base+.5*corr
        delta=smooth_limit(total,z)
        if intervention in ('noop','residual_off'): delta=torch.zeros_like(z)
        if self.fold==2:
            mask=torch.ones((871,1),device=z.device,dtype=z.dtype)
            mask[[376,406]]=0
            delta=delta*mask
        return z+delta,dict(z=z,h=h,hhat=hh,tokens=tokens,a=a,coverage=coverage,
            w_sem=ws,w_sp=wp,entropy=entropy,null=n,beta=beta,w=w,context=context,
            policy_input=policy_input,P=p,s=s,raw_base=base,raw_corr=corr,raw_total=total,
            delta_raw=total,delta=delta)


def teacher_logits(teacher,z):
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    # Intentionally never invoke even the eval Dropout module.
    return teacher.classifier(teacher.mlp_head[2](teacher.mlp_head[1](teacher.mlp_head[0](z))))


def true_margin(logits,y):
    shape=[len(y)]+[1]*(logits.ndim-2)+[1]
    yy=y.reshape(shape).expand(*logits.shape[:-1],1)
    return logits.gather(-1,yy).squeeze(-1)-logits.gather(-1,1-yy).squeeze(-1)


@torch.no_grad()
def counterfactual(teacher,z,U,mu,legal,legal_labels):
    z=z.detach()[legal];y=legal_labels.detach().long()
    assert y.shape==(len(z),) and set(y.cpu().tolist()).issubset({0,1})
    h=(z-mu)@U
    component=h[:,:,None]*U.T[None]
    d=smooth_limit(component,z[:,None,:])
    inputs=torch.cat([z[:,None,:],z[:,None,:]+d,z[:,None,:]-d],1)
    logits=teacher_logits(teacher,inputs.reshape(-1,144)).reshape(len(z),33,2)
    margins=true_margin(logits,y)
    gp=margins[:,1:17]-margins[:,:1]
    gm=margins[:,17:33]-margins[:,:1]
    gains=torch.stack([gp,gm,torch.zeros_like(gp)],-1)
    target=(gains/.10).softmax(-1).detach()
    margin=margins[:,0]
    weights=torch.where(margin<=0,torch.full_like(margin,1.5),torch.ones_like(margin)).detach()
    return dict(target=target,gains=gains,margins=margins,correct=logits[:,0].argmax(-1)==y,
        weights=weights,mask_relative=d.norm(dim=-1)/z.norm(dim=-1,keepdim=True).clamp_min(EPS))


def balanced_mean(values,y,weights):
    assert values.shape==y.shape==weights.shape
    means=[]
    for label in (0,1):
        mask=y==label
        assert bool(mask.any())
        means.append((values[mask]*weights[mask]).sum()/weights[mask].sum())
    return .5*(means[0]+means[1])


def auxiliary(module,teacher,z,legal,y):
    """z.detach blocks every auxiliary gradient to DARG/CAM/original head."""
    detached=z.detach()
    cf=counterfactual(teacher,detached,module.U,module.mu,legal,y)
    changed,d=module(detached)
    per_action=-(cf['target']*d['P'][legal].clamp_min(EPS).log()).sum(-1).mean(-1)
    # Frozen teacher remains differentiable with respect to BSCP's delta.
    current=teacher_logits(teacher,changed[legal])
    original=teacher_logits(teacher,detached[legal]).detach()
    gain=true_margin(current,y)-true_margin(original,y)
    action=balanced_mean(per_action,y,cf['weights'])
    harm=balanced_mean((-gain).relu(),y,cf['weights'])
    return dict(action=action,harm=harm,cf=cf,aux_details=d,gain=gain,
        per_action=per_action,per_harm=(-gain).relu())

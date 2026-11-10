"""Low-capacity, baseline-preserving probes for frozen E74.4 features."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss,
                             roc_auc_score)


def norm_rows(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def projections(seed):
    rng=np.random.default_rng(int(seed))
    wt=(rng.standard_normal((9,128))/math.sqrt(128)).astype(np.float32)
    wp=(rng.standard_normal((1024,128))/math.sqrt(128)).astype(np.float32)
    return wt,wp


def b1_features(tokens,text128,tags,token_projection):
    t=np.asarray(tokens,dtype=np.float32)@np.asarray(token_projection,dtype=np.float32)
    k=norm_rows(t); q=norm_rows(text128); tags=np.asarray(tags)
    ai=np.flatnonzero(tags=="ASD"); ti=np.flatnonzero(tags=="TD")
    if not len(ai) or not len(ti): raise ValueError("prompt bank must have both classes")
    sim=np.einsum("ntd,jd->ntj",k,q,optimize=True)
    return (sim[:,:,ai].mean(-1)-sim[:,:,ti].mean(-1)).astype(np.float32),sim.astype(np.float32)


def metrics(y,p):
    y=np.asarray(y,dtype=np.int64); p=np.clip(np.asarray(p,dtype=np.float64),1e-8,1-1e-8)
    pred=(p>=.5).astype(np.int64)
    return {"acc":float(accuracy_score(y,pred)),"hard_auc":float(balanced_accuracy_score(y,pred)),
            "prob_auc":float(roc_auc_score(y,p)),"sen":float(((pred==1)&(y==1)).sum()/max(1,(y==1).sum())),
            "spe":float(((pred==0)&(y==0)).sum()/max(1,(y==0).sum())),
            "asd_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),"brier":float(brier_score_loss(y,p)),"prediction":pred}


def fit(tokens,base,y,text128,tags,mode,seed,c,steps,token_projection=None,random_anchor=None,lr=.03):
    """Fit only the residual in BCE(base + residual, y); base is a fixed offset."""
    torch.set_num_threads(1)
    x=torch.as_tensor(np.asarray(tokens),dtype=torch.float32)
    b=torch.as_tensor(np.asarray(base),dtype=torch.float32).reshape(-1).detach()
    yt=torch.as_tensor(np.asarray(y),dtype=torch.float32).reshape(-1)
    q=np.asarray(text128,dtype=np.float32); tags=np.asarray(tags)
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    W=None; u=None; w_item=None; anchor=random_anchor
    if mode=="B1":
        if token_projection is None: raise ValueError("B1 needs its fixed token projection")
        if anchor is None: raw=torch.as_tensor(b1_features(x.numpy(),q,tags,token_projection)[0],dtype=torch.float32)
        else:
            key=F.normalize(x@torch.as_tensor(token_projection,dtype=torch.float32),dim=-1,eps=1e-8)
            aa=F.normalize(torch.as_tensor(anchor,dtype=torch.float32).reshape(1,-1),dim=-1)
            raw=torch.einsum("ntd,bd->nt",key,aa)
        w=torch.zeros(16,requires_grad=True); bias=torch.zeros((),requires_grad=True)
        params=[w,bias]
    elif mode in ("B2","ITEM"):
        W=(torch.randn((9,128),generator=gen)*.04).requires_grad_()
        if mode=="B2":
            w=torch.zeros(16,requires_grad=True); bias=torch.zeros((),requires_grad=True); params=[W,w,bias]
        else:
            # Nonzero factors avoid a zero-gradient bilinear initialization.
            u_item=(torch.randn((16,),generator=gen)*.03).requires_grad_()
            v_item=(torch.randn((len(tags),),generator=gen)*.03).requires_grad_()
            beta=torch.tensor(.1,requires_grad=True); bias=torch.zeros((),requires_grad=True)
            params=[W,u_item,v_item,beta,bias]
            w=beta
    else: raise ValueError(mode)
    opt=torch.optim.Adam(params,lr=float(lr)); n=max(1,len(y)); trace=[]
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        if mode=="B1":
            if anchor is None: raw=torch.as_tensor(b1_features(x.numpy(),q,tags,token_projection)[0],dtype=torch.float32)
            else:
                key=F.normalize(x@torch.as_tensor(token_projection,dtype=torch.float32),dim=-1,eps=1e-8)
                aa=F.normalize(torch.as_tensor(anchor,dtype=torch.float32).reshape(1,-1),dim=-1)
                raw=torch.einsum("ntd,bd->nt",key,aa)
        elif mode=="ITEM":
            key=F.normalize(x@W,dim=-1,eps=1e-8); qq=F.normalize(torch.as_tensor(q,dtype=torch.float32),dim=-1)
            sim=torch.einsum("ntd,jd->ntj",key,qq)
            sg=torch.as_tensor([1. if z=="ASD" else -1. for z in tags],dtype=torch.float32)
            raw=torch.einsum("t,j,ntj,j->n",torch.tanh(u_item),torch.tanh(v_item),sim,sg)[:,None]
        else:
            key=F.normalize(x@W,dim=-1,eps=1e-8)
            if anchor is not None:
                aa=F.normalize(torch.as_tensor(anchor,dtype=torch.float32).reshape(1,-1),dim=-1)
                raw=torch.einsum("ntd,bd->nt",key,aa)
            else:
                qq=F.normalize(torch.as_tensor(q,dtype=torch.float32),dim=-1)
                sim=torch.einsum("ntd,jd->ntj",key,qq); tag=np.asarray(tags)
                ai=torch.as_tensor(np.flatnonzero(tag=="ASD"),dtype=torch.long)
                ti=torch.as_tensor(np.flatnonzero(tag=="TD"),dtype=torch.long)
                raw=sim[:,:,ai].mean(-1)-sim[:,:,ti].mean(-1)
        mu=raw.mean(0,keepdim=True).detach(); sd=raw.std(0,unbiased=False,keepdim=True).clamp_min(1e-4).detach()
        z=(raw-mu)/sd
        if mode=="ITEM": r=(z*w+bias).reshape(-1)
        else: r=z@w+bias
        bce=F.binary_cross_entropy_with_logits(b+r,yt)
        norm=w.square().sum()
        if W is not None: norm=norm+W.square().sum()
        if mode=="ITEM": norm=norm+u_item.square().sum()+v_item.square().sum()
        penalty=norm/(2*float(c)*n)
        loss=bce+penalty; loss.backward()
        grad=float(torch.sqrt(sum((p.grad.square().sum() for p in params if p.grad is not None))).detach())
        opt.step()
        if step+1 in (1,10,40,120): trace.append({"step":step+1,"bce":float(bce.detach()),"objective":float(loss.detach()),"grad_norm":grad})
    # Store normalization from the final fitted representation, after the last update.
    with torch.no_grad():
        if mode=="B1":
            if anchor is None: final_raw=torch.as_tensor(b1_features(x.numpy(),q,tags,token_projection)[0],dtype=torch.float32)
            else:
                key=F.normalize(x@torch.as_tensor(token_projection,dtype=torch.float32),dim=-1,eps=1e-8)
                aa=F.normalize(torch.as_tensor(anchor,dtype=torch.float32).reshape(1,-1),dim=-1)
                final_raw=torch.einsum("ntd,bd->nt",key,aa)
        elif mode=="ITEM":
            key=F.normalize(x@W,dim=-1,eps=1e-8);qq=F.normalize(torch.as_tensor(q,dtype=torch.float32),dim=-1)
            sim=torch.einsum("ntd,jd->ntj",key,qq);sg=torch.as_tensor([1. if z=="ASD" else -1. for z in tags],dtype=torch.float32)
            final_raw=torch.einsum("t,j,ntj,j->n",torch.tanh(u_item),torch.tanh(v_item),sim,sg)[:,None]
        else:
            key=F.normalize(x@W,dim=-1,eps=1e-8)
            if anchor is not None:
                aa=F.normalize(torch.as_tensor(anchor,dtype=torch.float32).reshape(1,-1),dim=-1);final_raw=torch.einsum("ntd,bd->nt",key,aa)
            else:
                qq=F.normalize(torch.as_tensor(q,dtype=torch.float32),dim=-1);sim=torch.einsum("ntd,jd->ntj",key,qq)
                tag=np.asarray(tags);ai=torch.as_tensor(np.flatnonzero(tag=="ASD"),dtype=torch.long);ti=torch.as_tensor(np.flatnonzero(tag=="TD"),dtype=torch.long)
                final_raw=sim[:,:,ai].mean(-1)-sim[:,:,ti].mean(-1)
        mu=final_raw.mean(0,keepdim=True);sd=final_raw.std(0,unbiased=False,keepdim=True).clamp_min(1e-4)
    state={"mode":mode,"W":None if W is None else W.detach(),"w":w.detach(),
           "u":u_item.detach() if mode=="ITEM" else None,"v":v_item.detach() if mode=="ITEM" else None,
           "bias":bias.detach(),"mean":mu.detach(),"std":sd.detach(),"c":float(c),"trace":trace,
           "token_projection":None if token_projection is None else torch.as_tensor(token_projection,dtype=torch.float32)}
    return state


def predict(model,tokens,base,base_probability,text128,tags,lam,cap=2.,random_anchor=None):
    x=torch.as_tensor(np.asarray(tokens),dtype=torch.float32); b=np.asarray(base,dtype=np.float64).reshape(-1)
    q=np.asarray(text128,dtype=np.float32); tags=np.asarray(tags); mode=model["mode"]
    with torch.no_grad():
        if mode=="B1":
            if random_anchor is not None:
                key=F.normalize(x@model["token_projection"],dim=-1,eps=1e-8)
                aa=F.normalize(torch.as_tensor(random_anchor,dtype=torch.float32).reshape(1,-1),dim=-1)
                raw=torch.einsum("ntd,bd->nt",key,aa)
            else: raw=torch.as_tensor(b1_features(x.numpy(),q,tags,model["token_projection"].cpu().numpy())[0],dtype=torch.float32)
        else:
            key=F.normalize(x@model["W"],dim=-1,eps=1e-8); qq=F.normalize(torch.as_tensor(q,dtype=torch.float32),dim=-1,eps=1e-8)
            sim=torch.einsum("ntd,jd->ntj",key,qq)
            if mode=="ITEM":
                sg=torch.as_tensor([1. if z=="ASD" else -1. for z in tags],dtype=torch.float32)
                raw=torch.einsum("t,j,ntj,j->n",torch.tanh(model["u"]),torch.tanh(model["v"]),sim,sg)[:,None]
            elif random_anchor is not None:
                aa=F.normalize(torch.as_tensor(random_anchor,dtype=torch.float32).reshape(1,-1),dim=-1)
                raw=torch.einsum("ntd,bd->nt",key,aa)
            else:
                ai=torch.as_tensor(np.flatnonzero(tags=="ASD"),dtype=torch.long); ti=torch.as_tensor(np.flatnonzero(tags=="TD"),dtype=torch.long)
                raw=sim[:,:,ai].mean(-1)-sim[:,:,ti].mean(-1)
        z=(raw-model["mean"])/model["std"]
        if mode=="ITEM": residual=(z*model["w"]+model["bias"]).cpu().numpy().reshape(-1)
        else: residual=(z@model["w"]+model["bias"]).cpu().numpy()
    active=float(np.mean(np.abs(residual)>cap))
    if float(lam)==0.: p=np.asarray(base_probability,dtype=np.float64).copy()
    else:
        score=b+float(lam)*np.clip(residual,-cap,cap)
        p=1/(1+np.exp(-np.clip(score,-40,40)))
    return p,residual,active

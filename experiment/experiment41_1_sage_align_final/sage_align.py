"""Fixed evidence path for Experiment 41.1. No labels enter semantic scoring."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import KFold


def robust_scale(x, axis=0, floor=.05):
    x=np.asarray(x,dtype=np.float64)
    return np.maximum(1.4826*np.median(np.abs(x-np.median(x,axis=axis,keepdims=True)),axis=axis),floor)


def fit_ridge(d, y, alpha=10.):
    penalty=np.eye(d.shape[1])*alpha; penalty[0,0]=0
    inv=np.linalg.pinv(d.T@d+penalty,hermitian=True)
    return inv@d.T@y,inv


def build_normative(x, train_idx, y, endpoints, mode='conditional'):
    """Five-fold OOF for every outer-training subject; test uses all train TD.

    The robust scale is an OOF residual scale, not a calibrated Gaussian
    predictive interval. Empirical TD score CDFs provide the later calibration.
    """
    x=np.asarray(x,dtype=np.float64); train_idx=np.asarray(train_idx,dtype=np.int64)
    n=len(x); test_idx=np.setdiff1d(np.arange(n),train_idx)
    td=train_idx[np.asarray(y)[train_idx]==1]
    age=np.argmax(x[:,:26],axis=1); site=np.argmax(x[:,26:46],axis=1); sex=np.argmax(x[:,46:48],axis=1)
    fd_center=float(np.median(x[td,60])); fd_scale=float(robust_scale(x[td,60]))
    fd=np.clip((x[:,60]-fd_center)/fd_scale,-4,4)
    design=(np.ones((n,1)) if mode=='global' else np.column_stack([np.ones(n),x[:,:48],fd]))
    pred=np.full((n,256),np.nan); leverage=np.full(n,np.nan); support=np.zeros(n)
    counts=np.zeros((n,3),dtype=np.int64); inner_fold=np.full(n,-1,dtype=np.int64)
    fit_records=[]
    def assign(ref,query):
        coef,inv=fit_ridge(design[ref],x[ref,64:320])
        pred[query]=design[query]@coef
        leverage[query]=np.maximum(np.einsum('ij,jk,ik->i',design[query],inv,design[query]),0)
        for j,values in enumerate([age,site,sex]):
            counts[query,j]=np.bincount(values[ref],minlength=[26,20,2][j])[values[query]]
        if mode=='global': support[query]=1
        else: support[query]=np.cbrt(np.minimum(counts[query]/np.array([20.,20.,50.]),1).prod(1))
        return coef,inv
    for f,(fit,hold) in enumerate(KFold(5,shuffle=True,random_state=41041).split(train_idx)):
        fit_ids=train_idx[fit]; ref=fit_ids[np.asarray(y)[fit_ids]==1]; query=train_idx[hold]
        if len(ref)<80: raise ValueError('insufficient TD reference')
        assign(ref,query); inner_fold[query]=f
        fit_records.append({'inner_fold':f,'reference_ids':ref.tolist(),'query_ids':query.tolist()})
    coef,inv=assign(td,test_idx)
    residual=x[td,64:320]-pred[td]
    sigma=robust_scale(residual)
    raw_z=(x[:,64:320]-pred)/sigma
    z=np.clip(raw_z,-4,4)
    reference_support=support/(1+np.sqrt(leverage))
    good_reference=(reference_support>=.1)
    if mode=='conditional': good_reference &= (counts>0).all(1)
    # Processed fields are monotone affine transforms of verified raw fields.
    # Higher DVARS, outlier fraction, mean FD, percent FD are worse; low is not penalized.
    qc_cols=np.array([57,58,60,62],dtype=np.int64)
    qc_center=np.median(x[td][:,qc_cols],axis=0); qc_scale=robust_scale(x[td][:,qc_cols])
    excess=np.maximum((x[:,qc_cols]-qc_center)/qc_scale,0)
    qc_burden=excess.mean(1)
    qc_cut=float(np.quantile(qc_burden[td],.995))
    severe_qc=qc_burden>qc_cut
    qc_support=np.exp(-.20*np.minimum(qc_burden,30))
    if not np.isfinite(z).all(): raise FloatingPointError('normative z')
    return dict(z=z.astype('float32'),raw_z=raw_z.astype('float32'),prediction=pred.astype('float32'),
                sigma=sigma.astype('float32'),leverage=leverage.astype('float32'),
                reference_support=reference_support.astype('float32'),reference_counts=counts,
                good_reference=good_reference,qc_support=qc_support.astype('float32'),
                qc_burden=qc_burden.astype('float32'),severe_qc=severe_qc,qc_cols=qc_cols,
                qc_center=qc_center,qc_scale=qc_scale,qc_cut=np.asarray(qc_cut),
                known_edge=(np.asarray(endpoints)>=0).all(1),train_idx=train_idx,test_idx=test_idx,td_idx=td,
                inner_fold=inner_fold,full_coef=coef,full_inverse=inv,
                stats={'mode':mode,'inner_fits':fit_records,'full_reference_ids':td.tolist(),
                       'td_oof_mse':float(np.mean(residual**2)),'td_oof_mae':float(np.abs(residual).mean()),
                       'z_clip_fraction':float((np.abs(raw_z)>=4).mean()),
                       'fd_center':fd_center,'fd_scale':fd_scale})


def normalize_weights(w):
    w=np.asarray(w,dtype=np.float64)
    return np.divide(w,w.sum(-1,keepdims=True),out=np.zeros_like(w),where=w.sum(-1,keepdims=True)>0)


def concept_features(ev,weights):
    """Identical 4 x (3 scales x 6 channels) interface for E2--E9.

    Channels contain FC evidence only. They all vanish if z is zero. Reliability
    and CAM cannot create an evidence value on their own.
    """
    z=np.asarray(ev['z'],dtype=np.float64)
    edge=np.stack([np.maximum(z,0),np.maximum(-z,0),(z>1).astype(float),
                   (z< -1).astype(float),z*z/4,np.abs(z)],axis=-1)
    weights=normalize_weights(weights)
    if np.any(weights[:,:,~ev['known_edge']]): raise AssertionError('unknown anatomical edge included')
    return np.einsum('kse,ned->nksd',weights,edge,optimize=True).reshape(len(z),4,18).astype('float32')


def deviation_score(features):
    f=np.asarray(features,dtype=np.float64).reshape(len(features),4,3,6)
    return (.5*(f[:,:,:,0]+f[:,:,:,1])+.5*(f[:,:,:,2]+f[:,:,:,3])).mean(2)


def empirical_cdf(score,sorted_td):
    result=np.zeros_like(score,dtype=np.float64)
    for k in range(4):
        reference=sorted_td[:,k]
        left=np.searchsorted(reference,score[:,k],side='left')
        right=np.searchsorted(reference,score[:,k],side='right')
        result[:,k]=(0.5*(left+right)+.5)/(len(reference)+1.)
    return result


def evidence_for_mapping(ev,weights,coverage,eligible=None,use_qc=True,forced=False,calibration=None):
    features=concept_features(ev,weights); score=deviation_score(features)
    calibration=(np.sort(score[ev['td_idx']],axis=0) if calibration is None else np.asarray(calibration))
    strength=empirical_cdf(score,calibration)
    eligible=np.ones(4,bool) if eligible is None else np.asarray(eligible,bool)
    eligible &= np.asarray(weights).sum((1,2))>0
    ref=np.asarray(ev['reference_support'],float)
    qc=np.asarray(ev['qc_support'],float) if use_qc else np.ones(len(ref))
    accepted=np.asarray(ev['good_reference'],bool).copy()
    if use_qc: accepted &= ~ev['severe_qc']
    reliability=ref[:,None]*qc[:,None]*np.asarray(coverage)[None,:]
    quality=np.clip(reliability*strength,0,1)*eligible[None,:]*accepted[:,None]
    if forced:
        # Only rejection is removed. Same features, slots, parameters and CDF.
        q=quality.copy(); empty=q.sum(1)<=0
        q[empty]=strength[empty]*eligible[None,:]
        total=q.sum(1)
        mass=np.divide(q,total[:,None],out=np.zeros_like(q),where=total[:,None]>0)
        unmatched=(total==0).astype(float)
    else:
        unmatched=np.prod(1-quality,axis=1)
        total=quality.sum(1)
        mass=np.divide((1-unmatched)[:,None]*quality,total[:,None],out=np.zeros_like(quality),where=total[:,None]>0)
    if not np.allclose(mass.sum(1)+unmatched,1,atol=1e-7): raise AssertionError('mass conservation')
    effective=(features*mass[:,:,None]).astype('float32')
    return dict(effective=effective,features=features,score=score,strength=strength,
                matched=mass,unmatched=unmatched,quality=quality,accepted=accepted,
                calibration=calibration,coverage=np.asarray(coverage),eligible=eligible)


class SAGEAlign(nn.Module):
    """Shared, bias-free 5,760-parameter transform with query-only CAM gating."""
    def __init__(self,evidence_dim=18,concepts=4,rank=8,output_dim=144,cam_width=36,cap=.08):
        super().__init__(); self.evidence_dim=evidence_dim; self.concepts=concepts; self.cap=cap
        self.evidence_proj=nn.Linear(evidence_dim,rank,bias=False)
        self.cam_query=nn.Linear(cam_width,concepts,bias=False)
        self.output_proj=nn.Linear(rank,cam_width,bias=False)
        self.residual_proj=nn.Linear(cam_width,output_dim,bias=False)
        nn.init.zeros_(self.residual_proj.weight)
    def forward(self,evidence,cam,details=False):
        if evidence.shape[1:]!=(self.concepts,self.evidence_dim): raise ValueError('evidence shape')
        slots=F.gelu(self.evidence_proj(evidence))
        query=torch.sigmoid(self.cam_query(cam.mean(1)))
        pooled=(query.unsqueeze(-1)*slots).sum(1)
        raw=self.residual_proj(self.output_proj(pooled))
        base=cam.reshape(len(cam),-1); norm=raw.norm(dim=1,keepdim=True)
        cap=self.cap*base.norm(dim=1,keepdim=True).clamp_min(1e-8)
        scale=torch.where(norm>1e-12,cap*torch.tanh(norm/cap)/norm.clamp_min(1e-12),torch.ones_like(norm))
        residual=raw*scale; augmented=(base+residual).reshape_as(cam)
        if details: return augmented,residual,dict(raw=raw,scale=scale,cap=cap,query=query)
        return augmented,residual


def zero_evidence_test(branch,cam):
    z=torch.zeros((len(cam),4,18),device=cam.device,dtype=cam.dtype)
    return bool(torch.count_nonzero(branch(z,cam)[1]).item()==0)

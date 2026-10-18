"""Graph-native conditional alignment regularizer for original GFR.

No new classifier, logits, token residual, patient-level gain teacher or text.
Only outer-train labels are read. Original four GFR adjacency and their
message-passing functions remain unchanged. All auxiliary graph alignment
compares *projected* training-graph cosine similarities; FC is a detached
anchor for alignment (not for classification or FC-supervised loss).
"""
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from e24_contract import DESIGNS,LAMBDA_SUP,LAMBDA_ALIGN


def pairs_from_train(train, labels, site, seed=24001, per_class=1024):
    """Deterministic balanced, across-site pair audit using TRAIN only.
    Target 2*per_class pairs (same/different label). Never references TEST y.
    """
    tr=np.asarray(train,dtype=np.int64)
    yy=np.asarray(labels,dtype=np.int64)[tr]
    ss=np.asarray(site,dtype=np.int64)[tr]
    rng=np.random.default_rng(seed)
    pairs=[]
    # vectorized upper triangular, exact no self pair
    a,b=np.triu_indices(len(tr),1)
    cross=ss[a]!=ss[b]
    for positive in (True,False):
        allowed=np.flatnonzero(cross & ((yy[a]==yy[b])==positive))
        assert len(allowed)>=per_class,(positive,len(allowed))
        chosen=rng.choice(allowed,per_class,replace=False)
        pairs.append((tr[a[chosen]],tr[b[chosen]],np.full(per_class,float(positive))))
    return tuple(np.concatenate([p[i] for p in pairs]).astype(np.int64 if i<2 else np.float32) for i in range(3))


class CARA:
    def __init__(self,config,train,labels,site,fold,per_class=1024):
        assert config in DESIGNS
        self.cfg=config;self.design=DESIGNS[config]
        self.train=np.asarray(train,dtype=np.int64)
        self.ii,self.jj,self.same=pairs_from_train(self.train,labels,site,24000+fold,per_class=per_class)
        gen=np.random.default_rng(24600+fold)
        self.shuffled=gen.permutation(len(self.train)).astype(np.int64)
        assert not set(self.train)==set() and len(set(self.train))==len(self.train)
        self.stats={}
        self.call_count=0

    @staticmethod
    def _cos(g,parts):
        # Similarity *before* original hard >0.9 edge cutoff.
        z=F.normalize(g.p(parts),dim=-1,eps=1e-8)
        return z

    @staticmethod
    def _cos_pairs(z,i,j):
        return (z[i]*z[j]).sum(-1)

    @staticmethod
    def _safe_lambda(cls,regularizer,params):
        # Don't force cross-modal alignment when auxiliary projector gradient
        # conflicts with supervised classification. Only sample the actual
        # original GFR projection parameters; no Hessian or extra optimizer.
        gc=torch.autograd.grad(cls,params,retain_graph=True,allow_unused=True)
        ga=torch.autograd.grad(regularizer,params,retain_graph=True,allow_unused=True)
        dot=cls.new_zeros(());nc=dot.clone();na=dot.clone()
        for x,y in zip(gc,ga):
            if x is None or y is None:continue
            dot=dot+(x.detach()*y.detach()).sum()
            nc=nc+x.detach().square().sum()
            na=na+y.detach().square().sum()
        if float(nc)<=1e-16 or float(na)<=1e-16:return 0.,0.
        cos=float((dot/(nc.sqrt()*na.sqrt()+1e-12)).detach())
        return float(cos>=0),cos

    def loss(self,gfr,parts,labels,train_index,classification):
        assert len(parts)==4 and len(self.train)==len(train_index)
        assert np.array_equal(np.asarray(train_index),self.train)
        design=self.design
        if self.cfg=='A0':return classification.new_zeros(())
        dev=classification.device
        i=torch.as_tensor(self.ii,device=dev);j=torch.as_tensor(self.jj,device=dev)
        y=torch.as_tensor(self.same,device=dev,dtype=classification.dtype)
        zs=[];gcs=[]
        for m in range(4):
            g=getattr(gfr,f'GraphConstruct{m}')
            gcs.append(g)
            zs.append(self._cos(g,parts[m]))
        fc=self._cos_pairs(zs[3],i,j)
        # Site-balanced train-only label graph supervision: independently
        # accounted for by A1. Do NOT call the original unused ALoss.
        sup=F.binary_cross_entropy_with_logits((fc-0.2)*4.0,y)
        total=(LAMBDA_SUP*sup) if design.sup else classification.new_zeros(())
        num_active=0;cov=[];similarities=[];conflicts=[]
        if design.alignment!='none':
            for m in (0,1,2):
                zz=zs[m]
                if design.shuffle:
                    # Shuffle identities ONLY for alignment loss, never in GFR
                    # forward or supervised FC loss; train identities remain
                    # in the same partition. Pair endpoints use shuffled train.
                    perm=self.train[self.shuffled]
                    full=np.arange(len(parts[m]),dtype=np.int64)
                    full[self.train]=perm
                    qq=torch.as_tensor(full,device=dev,dtype=torch.long)
                    zz=zz.index_select(0,qq)
                aux=self._cos_pairs(zz,i,j)
                if design.alignment=='selective':
                    # Soft joint-support: when either graph does not support
                    # an edge, align almost nothing. No division by tiny
                    # overlap mass and no forced nonzero alignment.
                    support=(torch.sigmoid((fc.detach()-0.65)/0.1)*
                             torch.sigmoid((aux.detach()-0.65)/0.1))
                else:
                    support=torch.ones_like(fc)
                align=(support*(aux-fc.detach()).square()).mean()
                if design.safety:
                    active,similarity=self._safe_lambda(classification,align,list(gcs[m].p.parameters()))
                else:active,similarity=1.,float('nan')
                total=total+LAMBDA_ALIGN*active*align
                num_active+=int(active>0)
                cov.append(float(support.detach().mean()))
                similarities.append(float(align.detach()))
                conflicts.append(similarity)
        self.call_count+=1
        self.stats=dict(class_graph=float(sup.detach()),active=num_active,
                        support_mean=float(np.mean(cov)) if cov else 0.,
                        alignment_loss_mean=float(np.mean(similarities)) if similarities else 0.,
                        grad_cos=conflicts,lambda_sup=LAMBDA_SUP if design.sup else 0.,
                        lambda_align=LAMBDA_ALIGN if design.alignment!='none' else 0.)
        assert torch.isfinite(total)
        return total


def self_test():
    import torch.nn as nn
    from types import SimpleNamespace
    torch.manual_seed(24)
    X=[torch.randn(80,d) for d in (48,6,10,256)]
    y=np.array([0,1]*40)
    # paired two sites; cross-site both labels supported
    site=np.array([0,1,2,3]*20)
    train=np.arange(64)
    for cfg in DESIGNS:
        gs=nn.ModuleList([nn.Module() for _ in range(4)])
        for m,g in enumerate(gs):g.p=nn.Linear(X[m].shape[1],min(X[m].shape[1],32))
        root=nn.Module()
        for m,g in enumerate(gs):setattr(root,f'GraphConstruct{m}',g)
        model=nn.Linear(4,2)
        cls=model(torch.randn(80,4))[train].square().mean()
        # test classification gradients linked to GraphConstruct 1,2,3
        z=gs[1].p(X[1]);cls=cls+0.01*z[train].square().mean()
        evaluator=CARA(cfg,train,y,site,fold=1,per_class=128)
        value=evaluator.loss(root,X,y,train,cls)
        if cfg!='A0':value.backward()
        assert torch.isfinite(value)
        assert evaluator.call_count==(0 if cfg=='A0' else 1)
    # Poisoning test labels cannot affect any label pair target
    x=CARA('A4',train,y,site,1,per_class=128); y2=y.copy();y2[64:]=1-y2[64:]
    xx=CARA('A4',train,y2,site,1,per_class=128)
    assert np.array_equal(x.same,xx.same)
    print('e24 alignment self-test PASS')

if __name__=='__main__':self_test()


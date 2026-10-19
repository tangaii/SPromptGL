"""Evidence-Preserving Asymmetric Graph Calibration (EPAGC).

ANAT/FUNC quality-feature consistency weakly downweights EXISTING FC
patient graph edges, never creates/removes edges or modifies diagonal.
Site-adjusted statistics use outer TRAIN inputs only, no labels.
No trainable parameters, no additional supervised auxiliary loss.
"""
import numpy as np
import torch
from e25_contract import DESIGNS,RHO

_EPS=1e-6

def _standardize(x,train,sites,site_adjust):
    x=np.asarray(x,dtype=np.float32)
    tr=np.asarray(train,dtype=np.int64)
    site=np.asarray(sites,dtype=np.int64)
    mu=x[tr].mean(axis=0)
    sd=x[tr].std(axis=0)
    sd=np.maximum(sd,0.10)  # avoid exploding already-standardized QC features
    center=np.broadcast_to(mu,x.shape).copy()
    if site_adjust:
        for s in np.unique(site):
            idx=tr[site[tr]==s]
            if len(idx)>=8:
                center[site==s]=x[idx].mean(axis=0)
    z=np.clip((x-center)/sd,-3.0,3.0).astype(np.float32)
    assert np.isfinite(z).all()
    return z

def _rbf(z):
    # Both the training fold and held-out patients are represented as nodes,
    # consistent with the original FULL-871 transductive GFR.
    z=np.asarray(z,dtype=np.float64)
    q=(z*z).sum(1)
    dist=(q[:,None]+q[None,:]-2*z@z.T)/z.shape[1]
    dist=np.maximum(dist,0.0)
    c=np.exp(-0.5*dist).astype(np.float32)
    np.fill_diagonal(c,1.0)
    assert np.isfinite(c).all() and np.min(c)>=0 and np.max(c)<=1.00001
    return c

def _within_site_shuffle(z,train,sites,seed):
    """Permute only WITHIN site and separately in train/test partitions."""
    tr=np.asarray(train,dtype=np.int64)
    n=len(z)
    is_tr=np.zeros(n,dtype=bool);is_tr[tr]=True
    sites=np.asarray(sites,dtype=np.int64)
    rng=np.random.default_rng(seed) # does not consume global NumPy RNG
    perm=np.arange(n,dtype=np.int64)
    for group in (is_tr,~is_tr):
        for s in np.unique(sites[group]):
            ix=np.flatnonzero(group & (sites==s))
            if len(ix)>1:perm[ix]=rng.permutation(ix)
    assert np.array_equal(np.sort(perm),np.arange(n))
    assert np.all(sites[perm]==sites) and np.all(is_tr[perm]==is_tr)
    return z[perm],int((perm!=np.arange(n)).sum())

def prepare(config,fold,all_x,train,sites):
    assert config in DESIGNS and fold in range(1,11)
    cfg=DESIGNS[config]
    all_x=np.asarray(all_x,dtype=np.float32)
    train=np.asarray(train,dtype=np.int64)
    sites=np.asarray(sites,dtype=np.int64)
    assert all_x.shape==(871,320)
    assert len(sites)==871 and len(train)>=700
    # Original four features: PHENO 48, ANAT 6, FUNC 10, CORR 256.
    za=_standardize(all_x[:,48:54],train,sites,cfg.site_adjust)
    zf=_standardize(all_x[:,54:64],train,sites,cfg.site_adjust)
    moved_a=moved_f=0
    if cfg.shuffle:
        za,moved_a=_within_site_shuffle(za,train,sites,25100+fold)
        zf,moved_f=_within_site_shuffle(zf,train,sites,26100+fold)
    A=_rbf(za);F=_rbf(zf)
    if cfg.kind=='anat':C=A
    elif cfg.kind=='func':C=F
    else:C=(A+F)*.5
    C=np.clip(C,0,1).astype(np.float32)
    assert np.allclose(C,C.T,atol=1e-6)
    assert np.allclose(np.diag(C),1.)
    return dict(consensus=C,reference=(A+F)*.5,
        metadata=dict(config=config,fold=fold,shuffle_anat=moved_a,
        shuffle_func=moved_f,site_adjust=cfg.site_adjust,
        mean_qc_agreement=float(C[np.triu_indices(871,1)].mean())))

class GraphCalibrator:
    def __init__(self,config,fold,all_x,train,sites):
        self.config=config
        self.data=prepare(config,fold,all_x,train,sites)
        self.cache={}
        self.calls=0
        self.last={}
        self.capture_stats=False
        self.enabled=True

    def _matrix(self,arr,device,dtype):
        key=(id(arr),str(device),str(dtype))
        if key not in self.cache:
            self.cache[key]=torch.from_numpy(arr).to(device=device,dtype=dtype)
        return self.cache[key]

    def __call__(self,adj):
        assert adj.ndim==2 and adj.shape==(871,871)
        if not self.enabled:return adj  # exact P0 parity oracle
        if self.calls==0:
            assert torch.isfinite(adj).all()
            assert (adj>=0).all()  # original GraphLearn >0.9 cutoff
        original=adj
        C=self._matrix(self.data['consensus'],adj.device,adj.dtype)
        support=(adj>0).detach()
        eye=torch.eye(len(adj),dtype=torch.bool,device=adj.device)
        off=support & ~eye
        if self.config=='B0':
            real=self._matrix(self.data['reference'],adj.device,adj.dtype)
            m=(real*off.to(adj.dtype)).sum()/off.sum().clamp_min(1)
            c=torch.ones_like(adj)*m.detach()
        else:c=C
        # Strong invariant: no FC support creation/deletion, no self-loop change.
        weight=(1.0-RHO)+RHO*c
        weight=torch.where(eye,torch.ones_like(weight),weight)
        result=adj*weight
        assert torch.isfinite(result).all()
        if self.calls==0:
            assert torch.equal(support,result>0)
            assert torch.equal(torch.diagonal(original),torch.diagonal(result))
            assert torch.equal(result, result.T) or torch.allclose(result,result.T,atol=1e-6)
            assert bool((result[off] <= original[off]+1e-7).all())
        self.calls+=1
        if self.capture_stats or self.calls==1:
            self.last=dict(call_count=self.calls,edge_count=int(off.sum().item()/2),
                mean_edge_multiplier=float(weight[off].mean().detach()) if off.any() else 1.,
                edge_weight_min=float(weight[off].min().detach()) if off.any() else 1.,
                qc_mean=float(C[off].mean().detach()) if off.any() else 1.)
        return result


def self_test():
    rng=np.random.default_rng(25)
    X=rng.normal(size=(871,320)).astype(np.float32)
    sites=np.repeat(np.arange(20),44)[:871] if False else np.arange(871)%20
    tr=np.arange(760)
    S=rng.uniform(.90,.995,size=(871,871)).astype(np.float32)
    S=(S+S.T)/2;S[S<.93]=0;np.fill_diagonal(S,1.)
    for key in DESIGNS:
        gate=GraphCalibrator(key,1,X,tr,sites)
        a=torch.tensor(S,requires_grad=True)
        b=gate(a)
        assert torch.equal(a>0,b>0) and torch.equal(a.diag(),b.diag())
        assert float((b[a>0]/a[a>0]).min().detach())>=1-RHO-1e-6
        b.sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        again=GraphCalibrator(key,1,X,tr,sites)
        assert np.array_equal(gate.data['consensus'],again.data['consensus'])
    print('e25_gate self_test PASS')

if __name__=='__main__':self_test()


"""IA-FDSA: frozen FC-derived semantic teachers; bounded CORR-only residual."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONFIGS=('E0_P0','E1_NO_SPGS','E2_OLD_FC','E3_FINE_FC','E4_FINE_FULL',
         'E5_NUMERIC_FULL','E6_WRONG_ROI','E7_WRONG_DIRECTION',
         'E8_FINE_NO_ALIGN','E9_WRONG_AUX_PAIR')
SEEDS=(0,1,2)
CONTRASTS=(('E3_FINE_FC','E2_OLD_FC'),)+tuple(('E4_FINE_FULL',CONFIGS[i]) for i in (3,5,6,7,8,9,1,0))
INTERVENTIONS=('REAL','SEMANTIC_TEACHER_OFF','SEMANTIC_RESIDUAL_OFF','AUX_OFF',
               'PHENO_OFF','ANAT_QC_OFF','FUNC_QC_OFF','AUX_PAIR_SHUFFLED','FC_ANCHOR_CONTEXT_OFF')
SEMANTIC_WEIGHT=.05
RULES=dict(semantic_dimension=64,td_std_floor=.001,pooled_std_floor=.001,
           site_min_per_class=2,site_shrink_pseudocount=20.,difference_clip=2.,
           r_clip=3.,evidence_threshold=.5,context_weight=.25,aux_weight=.25,
           residual_coordinate_bound=.10,new_weight_decay=0.,new_init_seed=280000,
           auxiliary_shuffle_seed=28009,
           qa_off='replace specified sigmoid gate by ones; remove modulation only',
           pheno_off='zero demographic conditional correction; original PHENO remains',
           coarse_teacher='normalized mean of original abnormal four, common to both directions')

def tasks():return [(c,s,(i*3+s)%2) for i,c in enumerate(CONFIGS) for s in SEEDS]

def partition_permutation(site,train,test,fold):
    site=np.asarray(site);n=len(site);perm=np.arange(n)
    rng=np.random.default_rng(28009+int(fold))
    for part in (np.asarray(train),np.asarray(test)):
        for sid in sorted(np.unique(site[part])):
            ids=part[site[part]==sid]
            if len(ids)>1:
                order=rng.permutation(ids);perm[order]=np.roll(order,-1)
    assert np.array_equal(site[perm],site)
    for p in (train,test): assert set(perm[p])==set(p)
    return perm

def training_statistics(raw_corr,train,y_train,site):
    """Only train labels are accepted. TD reference pooled; effects stratified by site."""
    x=np.asarray(raw_corr,np.float64);train=np.asarray(train,int)
    y=np.asarray(y_train,int);st=np.asarray(site)[train];z=x[train]
    assert x.shape==(871,256) and y.shape==(len(train),) and set(y)=={0,1}
    a=z[y==0];b=z[y==1]
    mean=b.mean(0);sd=np.maximum(b.std(0,ddof=0),.001)
    pooled=np.sqrt((a.var(0,ddof=0)+b.var(0,ddof=0))*.5)
    pooled_effect=(a.mean(0)-b.mean(0))/np.maximum(pooled,.001)
    effects=[];weights=[];sites=[]
    for sid in sorted(np.unique(st)):
        aa=z[(y==0)&(st==sid)];bb=z[(y==1)&(st==sid)]
        rec=dict(site=int(sid),asd_n=len(aa),td_n=len(bb),eligible=min(len(aa),len(bb))>=2)
        if rec['eligible']:
            eff=(aa.mean(0)-bb.mean(0))/np.maximum(np.sqrt((aa.var(0)+bb.var(0))*.5),.001)
            neff=2*len(aa)*len(bb)/(len(aa)+len(bb))
            eff=np.clip(eff,-2.,2.)*(neff/(neff+20.))
            effects.append(eff);weights.append(neff)
            rec['weight']=float(neff)
        sites.append(rec)
    if effects:
        e=np.stack(effects);w=np.asarray(weights);w/=w.sum()
        d=np.clip((e*w[:,None]).sum(0),-2.,2.)
        hetero=np.sqrt(((e-d)**2*w[:,None]).sum(0))
        agreement=(np.sign(e)==np.sign(d)).mean(0)
    else:
        neff=2*len(a)*len(b)/(len(a)+len(b))
        d=np.clip(pooled_effect,-2.,2.)*neff/(neff+20.)
        hetero=np.zeros(256);agreement=np.zeros(256)
    def digest(t):return hashlib.sha256(np.asarray(t,np.float32).tobytes()).hexdigest()
    report=dict(td_mean_range=[float(mean.min()),float(mean.max())],
                td_std_range=[float(sd.min()),float(sd.max())],
                d_range=[float(d.min()),float(d.max())],d_abs_mean=float(abs(d).mean()),
                positive_direction_fraction=float((d>0).mean()),
                site_effect_heterogeneity_mean=float(hetero.mean()),
                site_direction_agreement_mean=float(agreement.mean()),
                pooled_stratified_sign_agreement=float((np.sign(d)==np.sign(pooled_effect)).mean()),
                eligible_sites=len(effects),site_counts=sites,
                td_mean_sha256=digest(mean),td_std_sha256=digest(sd),d_sha256=digest(d),
                labels_scope='outer_train only; no test labels accepted')
    return dict(mean=mean.astype(np.float32),std=sd.astype(np.float32),
                difference=d.astype(np.float32),report=report)

def teacher(raw_corr,statistics,bank,valid_indices):
    """Detach teacher: use the 252 verified edges only; unknown weights are exactly zero."""
    raw=np.asarray(raw_corr,np.float32);valid=np.asarray(valid_indices,int)
    r=np.clip((raw-statistics['mean'])/statistics['std'],-3.,3.)
    d=statistics['difference'];a=np.maximum(np.sign(d)[None,:]*r-.5,0)*np.clip(abs(d),0,2)[None,:]
    weights=np.zeros_like(a);weights[:,valid]=a[:,valid]
    direction=(d[valid]<0).astype(int)
    selected=np.asarray(bank,np.float32)[np.arange(252),direction]
    target=weights[:,valid]@selected
    norm=np.linalg.norm(target,axis=1,keepdims=True)
    active=weights.sum(1)>0
    target=target/np.maximum(norm,1e-12)
    target[~active]=0
    assert np.count_nonzero(weights[:,[195,225,232,236]])==0
    assert np.isfinite(target).all() and target.shape==(871,64)
    return target.astype(np.float32),active,weights

class ImagingAnchoredAlignment(nn.Module):
    def __init__(self,config,raw,train,test,y_train,site,pheno_indices,cache,fold):
        super().__init__()
        assert config in CONFIGS[2:] and raw.shape==(871,320)
        self.config=config;self.condition='REAL';self.full_aux=config not in CONFIGS[2:4]
        self.statistics=training_statistics(raw[:,64:],train,y_train,site)
        bank=cache['fine']
        if config=='E2_OLD_FC':bank=cache['coarse']
        elif config=='E5_NUMERIC_FULL':bank=cache['numeric']
        elif config=='E6_WRONG_ROI':bank=bank[cache['roi_permutation']]
        elif config=='E7_WRONG_DIRECTION':bank=bank[:,::-1].copy()
        target,active,weights=teacher(raw[:,64:],self.statistics,bank,cache['valid_indices'])
        self.register_buffer('teacher_target',torch.as_tensor(target))
        self.register_buffer('teacher_active',torch.as_tensor(active))
        self.register_buffer('semantic_edge_weights',torch.as_tensor(weights))
        self.register_buffer('train_indices',torch.as_tensor(train,dtype=torch.long))
        self.register_buffer('test_indices',torch.as_tensor(test,dtype=torch.long))
        self.register_buffer('pheno_indices',torch.as_tensor(pheno_indices,dtype=torch.long))
        self.register_buffer('aux_permutation',torch.as_tensor(partition_permutation(site,train,test,fold),dtype=torch.long))
        self.fc_encoder=nn.Sequential(nn.Linear(256,64),nn.LayerNorm(64),nn.GELU())
        self.context_encoder=nn.Sequential(nn.Linear(36,64),nn.LayerNorm(64),nn.GELU())
        self.demo_encoder=nn.Linear(92,64)
        self.anat_gate=nn.Linear(6,64)
        self.func_gate=nn.Linear(10,64)
        self.decode=nn.Linear(64,36)
        self.alpha=nn.Parameter(torch.zeros(()))
        self.last={}

    def forward(self,cam,raw):
        assert cam.shape==(871,4,36) and raw.shape==(871,320)
        aux=raw
        if self.config=='E9_WRONG_AUX_PAIR' or self.condition=='AUX_PAIR_SHUFFLED':
            aux=raw[self.aux_permutation]
        fc=self.fc_encoder(raw[:,64:])
        context=self.context_encoder(cam[:,3,:])
        if self.condition=='FC_ANCHOR_CONTEXT_OFF':context=torch.zeros_like(context)
        anchor=fc+.25*context
        demo=torch.tanh(self.demo_encoder(torch.cat((aux[:,self.pheno_indices],anchor),dim=1)))
        qa=torch.sigmoid(self.anat_gate(aux[:,48:54]))
        qf=torch.sigmoid(self.func_gate(aux[:,54:64]))
        if self.condition=='ANAT_QC_OFF':qa=torch.ones_like(qa)
        if self.condition=='FUNC_QC_OFF':qf=torch.ones_like(qf)
        conditional=.25*qa*qf*demo
        if not self.full_aux or self.condition in ('AUX_OFF','PHENO_OFF'):
            conditional=torch.zeros_like(conditional)
        joint=anchor+conditional
        delta=.10*self.alpha.tanh()*torch.tanh(self.decode(joint))
        if self.condition=='SEMANTIC_RESIDUAL_OFF':delta=torch.zeros_like(delta)
        zeros=torch.zeros_like(delta)
        all_delta=torch.stack((zeros,zeros,zeros,delta),dim=1)
        out=cam+all_delta
        self.last=dict(student=anchor,fc=fc,context=context,conditional=conditional,
                       joint=joint,qa=qa,qf=qf,delta=delta,cam=cam)
        assert torch.isfinite(out).all()
        return out

    def semantic_loss(self):
        student=F.normalize(self.last['student'],dim=-1,eps=1e-8)
        idx=self.train_indices[self.teacher_active[self.train_indices]]
        if self.config=='E8_FINE_NO_ALIGN' or self.condition=='SEMANTIC_TEACHER_OFF' or not len(idx):
            return student.sum()*0.
        return .05*F.smooth_l1_loss(student[idx],self.teacher_target[idx].detach())

def self_test():
    rng=np.random.default_rng(28);x=rng.normal(size=(871,320)).astype(np.float32)
    tr=np.arange(780);te=np.arange(780,871);y=np.arange(780)%2;site=np.arange(871)%20
    bank=rng.normal(size=(252,2,64)).astype(np.float32);bank/=np.linalg.norm(bank,axis=-1,keepdims=True)
    cache=dict(fine=bank,coarse=bank,numeric=bank,valid_indices=np.array([i for i in range(256) if i not in (195,225,232,236)]),roi_permutation=np.roll(np.arange(252),1))
    torch.manual_seed(28);base=torch.randn(871,4,36);raw=torch.from_numpy(x)
    counts=[]
    for cfg in CONFIGS[2:]:
        m=ImagingAnchoredAlignment(cfg,x,tr,te,y,site,np.r_[0:26,46:48],cache,1)
        counts.append(sum(p.numel() for p in m.parameters()))
        assert torch.equal(m(base,raw),base)
        with torch.no_grad():m.alpha.fill_(1.)
        out=m(base,raw);assert torch.equal(out[:,:3],base[:,:3])
        assert float(m.last['delta'].detach().abs().max())<=.1
        (out.square().mean()+m.semantic_loss()).backward()
        assert m.alpha.grad.isfinite() and m.decode.weight.grad.norm()>0
        if cfg not in CONFIGS[2:4]:assert m.demo_encoder.weight.grad.norm()>0
    assert len(set(counts))==1
    a=training_statistics(x[:,64:],tr,y,site);b=training_statistics(x[:,64:],tr,1-y,site)
    assert not np.array_equal(a['difference'],b['difference'])
    assert sum(g==0 for _,_,g in tasks())==sum(g==1 for _,_,g in tasks())==15
    print('E28 CPU method contract PASS; new parameters',counts[0])

if __name__=='__main__':self_test()

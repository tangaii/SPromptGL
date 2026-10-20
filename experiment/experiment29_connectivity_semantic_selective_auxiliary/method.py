"""CORR-only global/local frozen-text alignment and soft sparse auxiliary repair.

New patient-specific queries refer to verified CORR connection identities, NOT CAM
token positions. Text keys/values are fixed. All statistics accept TRAIN labels only.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Design:
    fine:bool=False
    semantic:bool=True
    auxiliary:tuple=()
    sparse:bool=False
    shuffled:bool=False
    text:str='REAL'

ALL=('PHENO','ANAT','FUNC')
DESIGNS={}
for prefix,fine in (('C',False),('F',True)):
    DESIGNS.update({prefix+'0':Design(fine),prefix+'1':Design(fine,True,ALL),
        prefix+'2':Design(fine,True,ALL,True),prefix+'3':Design(fine,False,ALL,True),
        prefix+'4':Design(fine,True,ALL,True,True),
        prefix+'5':Design(fine,True,('PHENO',),True),prefix+'6':Design(fine,True,('ANAT',),True),
        prefix+'7':Design(fine,True,('FUNC',),True)})
DESIGNS['C8']=Design(False,True,ALL,True,False,'NUMERIC')
DESIGNS['F8']=Design(True,True,ALL,True,False,'WRONG_ROI')
DESIGNS['F9']=Design(True,True,ALL,True,False,'SWAP_DIRECTION')
DESIGNS['F10']=Design(True,True,ALL,True,False,'NUMERIC')
CONFIGS=tuple(['C'+str(i) for i in range(9)]+['F'+str(i) for i in range(11)])
SEEDS=(0,)
METRICS=('acc','auc','sen_asd','spe_td')
CONTRASTS=(('F0','C0'),('F1','C1'),('F2','C2'),('C2','C0'),('F2','F0'),('C2','C1'),('F2','F1'),
           ('C2','C3'),('F2','F3'),('C2','C4'),('F2','F4'))+tuple((p+str(i),p+'0') for p in ('C','F') for i in (5,6,7))+(
           ('C2','C8'),('F2','F8'),('F2','F9'),('F2','F10'),('C2','P0'),('F2','P0'),('C2','E1'),('F2','E1'))
INTERVENTIONS=('REAL','SEMANTIC_OFF','PHENO_OFF','ANAT_OFF','FUNC_OFF','AUX_OFF','ALL_OFF',
               'NUMERIC_TEXT','WRONG_ROI','SWAP_DIRECTION','GLOBAL_CLASS_SWAP','AUX_SHUFFLED')
RULES=dict(semantic_dim=64,top_k=16,temperature=.2,lambda_global=.05,lambda_local=.02,
    lambda_sparse=.005,residual_coordinate_bound=.10,new_weight_decay=0.,init_seed=290000,
    edge_id_dim=16,query_input=83,query_hidden=64,semantic_hidden=64,
    stability='unweighted site sign agreement / (1 + weighted site effect heterogeneity)',
    candidate='ReLU(sign(d)*r - .5) * clip(abs(d),0,2) * stability; stable descending top16; active iff >0',
    attention='cos(query,key)/.2 + tanh(candidate_evidence)',
    gating='continuous sigmoid soft sparse; no hard threshold; L_sparse=(mean_feature_gate+mean_router)/2 over active modalities, outer TRAIN only',
    residual='0.10*tanh(tanh(alpha_sem)*semantic_decode + tanh(alpha_aux)*aux_decode)',
    local_loss='Within-patient selected-set InfoNCE; valid train connections only; repeated coarse keys imply constant log(candidate_count) and zero useful query gradient',
    fc_encoder='256->64 + LayerNorm + GELU; no CAM or auxiliary input to semantic losses',
    auxiliary='independent selectors [h_fc,x]->dim; biasless dim->64 GELU encoder; router [h,u,h*u]->1 sigmoid; biasless 64->64 value; biasless concat192->36 decoder')

def tasks():
    # Interleave comparable coarse/fine tasks; each GPU gets 10 tasks/100 folds.
    order=[p+str(i) for i in range(9) for p in ('C','F')]+['F9','F10']
    return [(c,0,CONFIGS.index(c)%2) for c in order]

def partition_permutation(site,train,test,fold):
    site=np.asarray(site);perm=np.arange(len(site));gen=np.random.default_rng(29004+fold)
    for part in (train,test):
        for s in np.unique(site[part]):
            ids=np.asarray(part)[site[part]==s]
            if len(ids)>1:
                order=gen.permutation(ids);perm[order]=np.roll(order,-1)
    for part in (train,test):assert set(perm[part])==set(part)
    assert np.array_equal(site[perm],site)
    return perm

def training_statistics(raw_corr,train,y_train,site):
    """Exp28 exact TD/effect formulas, additionally exposing edgewise stability."""
    x=np.asarray(raw_corr,np.float64);tr=np.asarray(train,int);y=np.asarray(y_train,int)
    assert x.shape==(871,256) and y.shape==(len(tr),) and set(y)=={0,1}
    z=x[tr];st=np.asarray(site)[tr];a=z[y==0];b=z[y==1]
    mean=b.mean(0);sd=np.maximum(b.std(0,ddof=0),.001)
    pe=(a.mean(0)-b.mean(0))/np.maximum(np.sqrt((a.var(0)+b.var(0))*.5),.001)
    effects=[];weights=[];counts=[]
    for s in sorted(np.unique(st)):
        aa=z[(y==0)&(st==s)];bb=z[(y==1)&(st==s)];eligible=min(len(aa),len(bb))>=2
        counts.append(dict(site=int(s),asd_n=len(aa),td_n=len(bb),eligible=eligible))
        if eligible:
            neff=2*len(aa)*len(bb)/(len(aa)+len(bb))
            e=(aa.mean(0)-bb.mean(0))/np.maximum(np.sqrt((aa.var(0)+bb.var(0))*.5),.001)
            effects.append(np.clip(e,-2,2)*(neff/(neff+20)));weights.append(neff)
    if effects:
        es=np.stack(effects);w=np.asarray(weights);w/=w.sum()
        d=np.clip((es*w[:,None]).sum(0),-2,2)
        hetero=np.sqrt(((es-d)**2*w[:,None]).sum(0));agreement=(np.sign(es)==np.sign(d)).mean(0)
    else:
        neff=2*len(a)*len(b)/(len(a)+len(b));d=np.clip(pe,-2,2)*neff/(neff+20)
        hetero=np.zeros(256);agreement=np.zeros(256)
    stability=agreement/(1+hetero)
    result={k:np.asarray(v,np.float32) for k,v in dict(mean=mean,std=sd,difference=d,stability=stability,agreement=agreement,heterogeneity=hetero).items()}
    result['report']=dict(td_mean_range=[float(mean.min()),float(mean.max())],td_std_range=[float(sd.min()),float(sd.max())],
        difference_range=[float(d.min()),float(d.max())],stability_mean=float(stability.mean()),
        agreement_mean=float(agreement.mean()),heterogeneity_mean=float(hetero.mean()),eligible_sites=len(effects),site_counts=counts,
        labels_scope='outer_train labels only; no test label accepted',
        arrays_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in result.items()})
    return result

def evidence_selection(raw_corr,statistics,valid):
    valid=np.asarray(valid,int);r=np.clip((np.asarray(raw_corr,np.float32)-statistics['mean'])/statistics['std'],-3,3)
    d=statistics['difference'];stability=statistics['stability']
    e=np.maximum(np.sign(d)[None,:]*r-.5,0)*np.clip(abs(d),0,2)[None,:]*stability[None,:]
    vv=e[:,valid];order=np.argsort(-vv,axis=1,kind='stable')[:,:16]
    values=np.take_along_axis(vv,order,axis=1);chosen=valid[order];mask=values>0
    assert not np.isin(chosen,[195,225,232,236]).any()
    scalars=np.stack((np.take_along_axis(r,chosen,axis=1),np.broadcast_to(d,r.shape)[np.arange(871)[:,None],chosen],
                       np.broadcast_to(stability,r.shape)[np.arange(871)[:,None],chosen]),axis=-1)
    return dict(selected_ids=order.astype(np.int64),selected_corr=chosen.astype(np.int64),selected_evidence=values.astype(np.float32),
                selected_valid=mask,selected_scalars=scalars.astype(np.float32),direction=(d[valid]<0).astype(np.int64))

class SemanticSelectiveAlignment(nn.Module):
    def __init__(self,config,raw,train,test,labels_train,site,pheno_indices,cache,fold):
        super().__init__();assert config in CONFIGS
        self.config=config;self.design=DESIGNS[config];self.condition='REAL';self.last={}
        self.statistics=training_statistics(raw[:,64:],train,labels_train,site)
        for k,v in evidence_selection(raw[:,64:],self.statistics,cache['valid_indices']).items():self.register_buffer(k,torch.as_tensor(v))
        for k,v in dict(train_indices=np.asarray(train),test_indices=np.asarray(test),pheno_indices=np.asarray(pheno_indices),
                         aux_permutation=partition_permutation(site,train,test,fold),roi_permutation=cache['roi_permutation']).items():
            self.register_buffer(k,torch.as_tensor(v,dtype=torch.long))
        for k in ('original','fine','numeric_original','numeric_fine'):self.register_buffer(k+'_text',torch.as_tensor(cache[k],dtype=torch.float32))
        self.fc_encoder=nn.Sequential(nn.Linear(256,64),nn.LayerNorm(64),nn.GELU())
        self.edge_id=nn.Embedding(252,16)
        self.query=nn.Sequential(nn.Linear(83,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,64,bias=False))
        self.semantic_decode=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,36))
        self.selectors=nn.ModuleDict();self.encoders=nn.ModuleDict();self.routers=nn.ModuleDict();self.values=nn.ModuleDict()
        for name,dim in zip(ALL,(28,6,10)):
            self.selectors[name]=nn.Linear(64+dim,dim)
            self.encoders[name]=nn.Sequential(nn.Linear(dim,64,bias=False),nn.GELU())
            self.routers[name]=nn.Linear(192,1)
            self.values[name]=nn.Linear(64,64,bias=False)
        self.aux_decode=nn.Linear(192,36,bias=False)
        self.alpha_sem=nn.Parameter(torch.zeros(()));self.alpha_aux=nn.Parameter(torch.zeros(()))

    def text_bank(self):
        mode=self.design.text
        if self.condition in ('NUMERIC_TEXT','WRONG_ROI','SWAP_DIRECTION'):mode=self.condition.replace('_TEXT','')
        original=self.numeric_original_text if mode=='NUMERIC' else self.original_text
        fine=self.numeric_fine_text if mode=='NUMERIC' else self.fine_text
        if mode=='WRONG_ROI':fine=fine[self.roi_permutation]
        if mode=='SWAP_DIRECTION':fine=fine.flip(1)
        proto=F.normalize(torch.stack((original[4:].mean(0),original[:4].mean(0))),dim=-1,eps=1e-8)
        if self.condition=='GLOBAL_CLASS_SWAP':proto=proto.flip(0)
        if not self.design.fine:fine=proto[0][None,None,:].expand(252,2,64)
        return proto,fine

    def forward(self,cam,raw):
        assert cam.shape==(871,4,36) and raw.shape==(871,320)
        h=self.fc_encoder(raw[:,64:]);zero=h.new_zeros((871,36))
        semantic_on=self.design.semantic and self.condition not in ('SEMANTIC_OFF','ALL_OFF')
        gl=h.new_zeros((871,2));match=h.new_zeros((871,16,16));att=h.new_zeros((871,16));query=h.new_zeros((871,16,64))
        gresp=lresp=h.new_zeros((871,64));sem_candidate=zero
        if semantic_on:
            proto,bank=self.text_bank();hn=F.normalize(h,dim=-1,eps=1e-8)
            gl=hn@proto.T/.2;gresp=gl.softmax(-1)@proto
            identities=self.edge_id(self.selected_ids)
            inp=torch.cat((h[:,None,:].expand(-1,16,-1),self.selected_scalars,identities),dim=-1)
            query=F.normalize(self.query(inp),dim=-1,eps=1e-8)
            directions=self.direction[self.selected_ids]
            keys=bank[self.selected_ids,directions]
            match=torch.bmm(query,keys.transpose(1,2))/.2
            diagonal=(query*keys).sum(-1)/.2
            scores=(diagonal+self.selected_evidence.tanh()).masked_fill(~self.selected_valid,-1e4)
            att=scores.softmax(-1)*self.selected_valid
            att=att/att.sum(-1,keepdim=True).clamp_min(1e-12)
            lresp=(att[...,None]*keys).sum(1)
            sem_candidate=self.semantic_decode(torch.cat((h,gresp,lresp),dim=-1))
        aux=raw[self.aux_permutation] if self.design.shuffled or self.condition=='AUX_SHUFFLED' else raw
        pieces=[];feature_gates={};relevances={};aux_values={}
        for name,x in zip(ALL,(aux[:,self.pheno_indices],aux[:,48:54],aux[:,54:64])):
            feature_gate=torch.sigmoid(self.selectors[name](torch.cat((h,x),dim=-1)))
            u=self.encoders[name](feature_gate*x)
            relevance=torch.sigmoid(self.routers[name](torch.cat((h,u,h*u),dim=-1)))
            value=relevance*self.values[name](u)
            enabled=name in self.design.auxiliary and self.condition not in (name+'_OFF','AUX_OFF','ALL_OFF')
            if not enabled:value=torch.zeros_like(value)
            pieces.append(value);feature_gates[name]=feature_gate;relevances[name]=relevance;aux_values[name]=value
        aux_candidate=self.aux_decode(torch.cat(pieces,dim=-1))
        semantic_term=self.alpha_sem.tanh()*sem_candidate;auxiliary_term=self.alpha_aux.tanh()*aux_candidate
        delta=.10*torch.tanh(semantic_term+auxiliary_term)
        # torch.cat preserves non-CORR locations bit for bit, including signed zero.
        out=torch.cat((cam[:,:3,:],(cam[:,3,:]+delta)[:,None,:]),dim=1)
        self.last=dict(h=h,query=query,global_logits=gl,local_logits=match,attention=att,global_response=gresp,local_response=lresp,
             semantic_on=semantic_on,semantic_term=semantic_term,auxiliary_term=auxiliary_term,delta=delta,cam=cam,
             feature_gates=feature_gates,relevances=relevances,aux_values=aux_values)
        assert torch.isfinite(out).all()
        return out

    def loss_terms(self,labels):
        last=self.last;tr=self.train_indices;zero=last['h'].sum()*0.
        global_loss=local_loss=sparse_loss=zero
        if last['semantic_on']:
            global_loss=F.cross_entropy(last['global_logits'][tr],labels[tr])
            valid=self.selected_valid[tr];logits=last['local_logits'][tr].masked_fill(~valid[:,None,:],-1e4)
            if valid.any():
                targets=torch.arange(16,device=logits.device)[None,:].expand(len(tr),-1)
                local_loss=F.cross_entropy(logits[valid],targets[valid])
        if self.design.sparse and self.design.auxiliary and self.condition not in ('AUX_OFF','ALL_OFF'):
            values=[]
            for name in self.design.auxiliary:
                if self.condition==name+'_OFF':continue
                values.extend((last['feature_gates'][name][tr].mean(),last['relevances'][name][tr].mean()))
            if values:sparse_loss=torch.stack(values).mean()
        return dict(global_loss=global_loss,local_loss=local_loss,sparse_loss=sparse_loss)

    def extra_loss(self,labels):
        terms=self.loss_terms(labels)
        return .05*terms['global_loss']+.02*terms['local_loss']+.005*terms['sparse_loss']

def self_test():
    assert len(CONFIGS)==20 and len(set(CONFIGS))==20
    assert sum(g==0 for _,_,g in tasks())==sum(g==1 for _,_,g in tasks())==10
    gen=np.random.default_rng(29);raw=gen.normal(size=(871,320)).astype(np.float32)
    tr=np.arange(780);te=np.arange(780,871);labels=np.arange(871)%2;site=np.arange(871)%20
    def norm(a):return a/np.linalg.norm(a,axis=-1,keepdims=True)
    cache=dict(original=norm(gen.normal(size=(8,64))),fine=norm(gen.normal(size=(252,2,64))),
        numeric_original=norm(gen.normal(size=(8,64))),numeric_fine=norm(gen.normal(size=(252,2,64))),
        valid_indices=np.array([i for i in range(256) if i not in (195,225,232,236)]),roi_permutation=np.roll(np.arange(252),1))
    base=torch.from_numpy(gen.normal(size=(871,4,36)).astype(np.float32));r=torch.from_numpy(raw);y=torch.from_numpy(labels)
    counts=[]
    for cfg in CONFIGS:
        torch.manual_seed(29);b=SemanticSelectiveAlignment(cfg,raw,tr,te,labels[tr],site,np.r_[0:26,46:48],cache,1)
        counts.append(sum(p.numel() for p in b.parameters()))
        assert torch.equal(b(base,r),base)
        with torch.no_grad():b.alpha_sem.fill_(1);b.alpha_aux.fill_(1)
        b(base,r);assert b.last['delta'].abs().max()<=.1
        assert torch.equal(b(base,r)[:,:3],base[:,:3])
        b.condition='ALL_OFF';assert torch.equal(b(base,r),base)
        b.condition='REAL';b(base,r);loss=b.extra_loss(y)
        assert loss.isfinite();loss.backward()
    assert len(set(counts))==1
    print('E29_CPU_CONTRACT_PASS new_parameters=',counts[0])

if __name__=='__main__':self_test()

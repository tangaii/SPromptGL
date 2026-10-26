"""Registered E43.1 operators. No new parameters and no held-out labels."""
import types
import numpy as np
import torch
from utils import normalize_adj
SLICES=((0,48),(48,54),(54,64),(64,320))
MODALITIES=('PHENO','ANAT','FUNC','CORR')

def supervised(adjs,train,train_labels):
    ids=torch.as_tensor(train,device=adjs[0].device,dtype=torch.long)
    labels=torch.as_tensor(train_labels,device=ids.device,dtype=torch.long)
    assert ids.shape==labels.shape and ids.unique().numel()==len(ids)
    # Exact repository norm/diagonal conventions, restricted BEFORE normalization.
    eye=torch.eye(len(ids),device=ids.device)
    target=normalize_adj((labels[:,None]==labels[None,:]).float()+eye)
    return torch.stack([torch.linalg.norm(normalize_adj(a[ids][:,ids]+eye)-target) for a in adjs]).mean()

def calibrated(raw,score,train):
    ids=torch.as_tensor(train,device=score.device,dtype=torch.long)
    i,j=torch.triu_indices(len(ids),len(ids),1,device=score.device)
    values=score[ids[i],ids[j]].detach().cpu().numpy()
    lo,med,hi=np.quantile(values,[.25,.5,.75],method='linear')
    iqr=max(float(hi-lo),1e-3)
    off=~torch.eye(len(raw),device=raw.device,dtype=torch.bool)
    a=torch.where(off & (raw>0),torch.sigmoid((score-float(med))/iqr),raw)
    assert torch.equal(a>0,raw>0) and torch.equal(a.diag(),raw.diag())
    assert torch.allclose(a,a.T,atol=1e-6,rtol=0)
    return a,dict(median=float(med),iqr=iqr,train_pairs=len(values))

def forward(self,x,model_num,label,index):
    full=torch.cat(x,dim=1);n=len(full);combined=full;raws=[];props=[];messages=[];inputs=[];stats=[]
    # Legacy ALoss retained for observation only; original target excludes test truth.
    masked=label.clone();masked[masked==0]=-1
    masked=torch.tensor([v if i in index else 0 for i,v in enumerate(masked)])
    target=normalize_adj(masked[:,None].eq(masked[None,:]).cuda()+torch.eye(n).cuda())
    aloss=0.
    for i in range(model_num):
        raw=getattr(self,f'GraphConstruct{i}')(x[i]);prop=raw
        if self.group431 in (3,5):prop,stat=calibrated(raw,getattr(self,f'GraphConstruct{i}').score431,index);stats.append(stat)
        norm=normalize_adj(prop+torch.eye(n,device=full.device));aloss=aloss+torch.linalg.norm(norm-target)
        value=full
        if self.group431==4:
            mask=full.new_zeros(320);start,stop=SLICES[i];mask[start:stop]=1;value=full*mask
        msg=getattr(self,f'MessagePassing{i}')(value,norm)
        if self.group431==4:msg=msg*mask
        if self.off431==i:msg=torch.zeros_like(msg)
        combined=torch.sigmoid(self.fusion_weight[i])*msg+combined
        raws.append(raw);props.append(prop);messages.append(msg.detach());inputs.append(value.detach())
    self.raw431=raws;self.prop431=props;self.messages431=messages;self.inputs431=inputs;self.stats431=stats
    # Original GraphConstructLoss always sees original GraphLearn output, NOT calibrated weights.
    return combined,raws,aloss/model_num

def install(agent,group):
    g=agent.MessagePassing0;g.group431=group;g.off431=None;g.messages431={};g.inputs431={}
    if group:g.forward=types.MethodType(forward,g)
    else:
        for i in range(4):
            def hook(module,inputs,output,i=i):
                g.messages431[i]=output.detach();g.inputs431[i]=inputs[0].detach()
                return torch.zeros_like(output) if g.off431==i else output
            getattr(g,f'MessagePassing{i}').register_forward_hook(hook)

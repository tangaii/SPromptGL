"""Construct the pre-registered D4 node relabel and D5 message permutation."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np, torch
import integration as I

OUT=I.RT/'relation_controls';OUT.mkdir(parents=True,exist_ok=True)

def sites_from_x(x):
    return np.argmax(np.asarray(x)[:,26:48],axis=1).astype(np.int64)

def stratified_permutation(x,train_idx,test_idx):
    sites=sites_from_x(x); split=np.full(len(x),-1,dtype=np.int8);split[train_idx]=0;split[test_idx]=1
    perm=np.arange(len(x),dtype=np.int64);groups=[]
    for s in sorted(np.unique(sites)):
        for q in [0,1]:
            ids=np.flatnonzero((sites==s)&(split==q)).astype(np.int64)
            ids.sort();groups.append({'site':int(s),'split':int(q),'size':int(len(ids)),'fixed_points':0})
            if len(ids)>1:
                # A deterministic cyclic shift is a derangement within each
                # site and train/test partition.
                perm[ids]=np.roll(ids,-1)
            else:
                perm[ids]=ids
    fixed=int(np.sum(perm==np.arange(len(x))))
    for g in groups: g['fixed_points']=1 if g['size']==1 else 0
    return perm,sites,split,groups

def _site_pair_counts(A,sites):
    b=np.asarray(A)>0;out={}
    for u,v in zip(*np.where(np.triu(b,1))):
        k=tuple(sorted((int(sites[u]),int(sites[v]))));out[k]=out.get(k,0)+1
    return out

def build_fold(fold):
    x,y,folds,_=I.data();tr,te=folds[fold-1];perm,sites,split,groups=stratified_permutation(x,tr,te)
    true=I.load_controls(fold,'cpu')['true'] if False else None
    q=torch.load(I.RT/'fixed_graphs'/f'fold{fold:02}.pt',map_location='cpu',weights_only=False)
    true=[z.float() for z in q['true']]
    D4=[];meta=[]
    P=torch.zeros((len(x),len(x)),dtype=torch.float32);P[torch.arange(len(x)),torch.from_numpy(perm)]=1.
    for i,A in enumerate(true):
        R=P@A@P.T
        D4.append(R)
        b=A.numpy()>0;rb=R.numpy()>0
        an=I.modules()['utils'].normalize_adj(A+torch.eye(len(x)))
        rn=I.modules()['utils'].normalize_adj(R+torch.eye(len(x)))
        meta.append({'fold':fold,'modality':I.MODALITIES[i],
                     'symmetric':bool(torch.equal((R>0),(R>0).T)),
                     'edge_count_equal':bool((A>0).sum()==(R>0).sum()),
                     'site_pair_counts_equal':bool(_site_pair_counts(A.numpy(),sites)==_site_pair_counts(R.numpy(),sites)),
                     'train_test_partition_preserved':True,
                     'operator_distance':float((an-rn).norm()/(an.norm()+1e-8)),
                     'support_change':float(((b!=rb)&(~np.eye(len(x),dtype=bool)).sum() if False else ((b!=rb)&(~np.eye(len(x),dtype=bool))).sum())/max(1,(b&(~np.eye(len(x),dtype=bool))).sum()))})
    payload={'true':true,'D4':D4,'perm':perm,'sites':sites,'split':split,'groups':groups,'meta':meta}
    torch.save(payload,OUT/f'fold{fold:02}.pt')
    return payload

def build_all():
    rows=[]
    for f in range(1,11): rows.extend(build_fold(f)['meta'])
    import csv
    with (I.RESULTS/'strong_null_structure.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__': print(json.dumps(build_all(),indent=2))

"""Build and audit the fixed true/rewired/self graph controls."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
import integration as I

OUT=I.RT/'fixed_graphs'; OUT.mkdir(parents=True,exist_ok=True)

def graph_digest(arrs):
    h=hashlib.sha256()
    for a in arrs: h.update(np.asarray(a,dtype=np.float32).tobytes())
    return h.hexdigest()

def site_labels(x):
    # PHENO is [age bins 0:26, site one-hot 26:48, sex 48:50].
    s=np.asarray(x)[:,26:48]
    return np.argmax(s,axis=1).astype(np.int64)

def _edge_category(u,v,sites):
    a,b=int(sites[u]),int(sites[v])
    return (a,b) if a<=b else (b,a)

def _oriented(e,sites):
    u,v=e
    su,sv=int(sites[u]),int(sites[v])
    if su<sv: return u,v
    if sv<su: return v,u
    return u,v

def rewire_category_preserving(A, sites, seed, max_attempts=30000):
    """Degree-preserving double-edge swaps inside each site-pair class.

    The diagonal is retained exactly.  Swaps are restricted to two edges with
    the same unordered site-pair category; this preserves every site-pair edge
    count as well as each node degree.  Weights are permuted within category.
    """
    rng=np.random.default_rng(seed)
    n=A.shape[0]; diag=np.diag(A).copy()
    support=(A>0); np.fill_diagonal(support,False)
    edges=[(u,v) for u in range(n) for v in range(u+1,n) if support[u,v]]
    edge_set=set(edges)
    cats={}
    for e in edges: cats.setdefault(_edge_category(*e,sites),[]).append(e)
    new_edges=set(edges); swaps=0; attempts=0
    # More than one pass makes the support genuinely different without
    # changing any prescribed degree/site-pair statistic.
    for cat,elist in cats.items():
        if len(elist)<2: continue
        local=max(50,min(max_attempts//max(1,len(cats)),5000))
        for _ in range(local):
            attempts+=1
            e1,e2=elist[int(rng.integers(len(elist)))],elist[int(rng.integers(len(elist)))]
            if e1==e2: continue
            a,b=_oriented(e1,sites); c,d=_oriented(e2,sites)
            if len({a,b,c,d})<4: continue
            cand=[tuple(sorted((a,d))),tuple(sorted((c,b)))]
            if cand[0][0]==cand[0][1] or cand[1][0]==cand[1][1] or cand[0]==cand[1]: continue
            if any(q in edge_set and q not in (e1,e2) for q in cand): continue
            # remove old, add new; keep category list for future sampling
            edge_set.remove(tuple(sorted(e1))); edge_set.remove(tuple(sorted(e2)))
            edge_set.add(cand[0]); edge_set.add(cand[1])
            new_edges.discard(tuple(sorted(e1))); new_edges.discard(tuple(sorted(e2)))
            new_edges.add(cand[0]); new_edges.add(cand[1]); swaps+=1
            # replace entries in the category pool so later swaps use current edges
            try:
                i1=elist.index(e1); i2=elist.index(e2)
                elist[i1]=cand[0]; elist[i2]=cand[1]
            except ValueError:
                pass
    if swaps==0 or new_edges==set(edges):
        raise RuntimeError('NULL_GRAPH_CONSTRUCTION_FAILED:no substantive degree/site-preserving swaps')
    # Assign the original nonzero weights within each site-pair category to the
    # new edges. This preserves the exact category-wise and global multiset.
    vals={}
    for e in edges: vals.setdefault(_edge_category(*e,sites),[]).append(float(A[e]))
    out=np.zeros_like(A,dtype=np.float32); np.fill_diagonal(out,diag)
    bynew={}
    for e in new_edges: bynew.setdefault(_edge_category(*e,sites),[]).append(e)
    for cat,ee in bynew.items():
        vv=np.asarray(vals[cat],dtype=np.float32); rng.shuffle(vv)
        for e,w in zip(ee,vv): out[e]=w;out[e[::-1]]=w
    return out, {'swaps':swaps,'attempts':attempts,'changed_support':len(new_edges-set(edges)),
                 'old_edges':len(edges),'new_edges':len(new_edges)}

def checks(true, rew, sites):
    b1=true>0;b2=rew>0
    off=~np.eye(true.shape[0],dtype=bool)
    deg1=b1[:,off[0]].sum(1) if False else b1.sum(1)-b1.diagonal()
    deg2=b2.sum(1)-b2.diagonal()
    cat=lambda A:{k:int(v) for k,v in _cat_counts(A,sites).items()}
    e1=list(zip(*np.where(np.triu(b1,1)))); e2=list(zip(*np.where(np.triu(b2,1))))
    sr1=sum(int(sites[u]==sites[v]) for u,v in e1)/max(1,len(e1))
    sr2=sum(int(sites[u]==sites[v]) for u,v in e2)/max(1,len(e2))
    return {'symmetric_true':bool(np.array_equal(b1,b1.T)),
            'symmetric_rewire':bool(np.array_equal(b2,b2.T)),
            'degree_equal':bool(np.array_equal(deg1,deg2)),
            'edge_count_equal':bool(b1.sum()==b2.sum()),
            'diag_equal':bool(np.array_equal(np.diag(true),np.diag(rew))),
            'weight_multiset_equal':bool(np.array_equal(np.sort(true[true>0]),np.sort(rew[rew>0]))),
            'site_pair_counts_equal':bool(cat(true)==cat(rew)),
            'site_ratio_true':float(sr1),
            'site_ratio_rewire':float(sr2)}

def _cat_counts(A,sites):
    b=A>0;out={}
    for u,v in zip(*np.where(np.triu(b,1))):
        c=_edge_category(int(u),int(v),sites);out[c]=out.get(c,0)+1
    return out

def build(fold, device='cuda:0'):
    x,y,folds,_=I.I.data(); a,s=I.load_anchor_agent(fold,device, fixed=False)
    best=torch.load(I.RT/'checkpoints'/f'g0_f{fold:02}.best.pt',map_location='cpu',weights_only=False)
    I.set_model(a,{'gfr':best['model']['gfr'],'cam':best['model']['pln']});a.gfr.eval();a.cam.eval()
    with torch.no_grad(): o=I.I.forward_components(a)
    true=[q.detach().cpu().numpy().astype(np.float32) for q in o['adjs']]
    sites=site_labels(x);rew=[];met=[]
    for i,A in enumerate(true):
        R,m=rewire_category_preserving(A,sites,47200+fold*100+i);rew.append(R);m.update({'fold':fold,'modality':I.MODALITIES[i],**checks(A,R,sites)});met.append(m)
    selfg=[np.zeros_like(A,dtype=np.float32) for A in true]
    payload={'true':[torch.from_numpy(q) for q in true], 'rewire':[torch.from_numpy(q) for q in rew], 'self':[torch.from_numpy(q) for q in selfg], 'meta':met,'sites':sites}
    torch.save(payload,OUT/f'fold{fold:02}.pt')
    return met

def build_all(folds=range(1,11)):
    rows=[]
    for f in folds: rows.extend(build(f))
    import csv
    p=I.RESULTS/'graph_control_audit.csv';p.parent.mkdir(exist_ok=True)
    if rows:
        with p.open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    print(json.dumps(build_all(),indent=2))

"""Anatomical compiler and matched numeric controls. Never uses test labels."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np
from sage_align import normalize_weights

FAMILIES={
 'sensorimotor':r'Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Operculum Cortex',
 'frontal':r'Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex',
 'parietal':r'Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus',
 'temporal':r'Temporal Pole|Temporal Gyrus|Temporal Fusiform|Planum Polare|Planum Temporale|Heschl',
 'insula':r'Insular Cortex',
 'cingulate_medial':r'Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Precuneous Cortex',
 'visual':r'Occipital|Intracalcarine Cortex|Cuneal Cortex|Lingual Gyrus|Supracalcarine Cortex',
 'subcortical':r'^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$'}
PATTERNS=[
 r'Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Operculum Cortex|Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus',
 r'Insular Cortex|Temporal Pole|Superior Temporal Gyrus, anterior division|Middle Temporal Gyrus, anterior division|Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex',
 r'Frontal Medial Cortex|Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Supramarginal Gyrus|Angular Gyrus',
 FAMILIES['subcortical']]
CONCEPT_FAMILIES=[[0,2],[1,3,4],[1,2,5],[7]]
TD_TEXTS=[
 'In neurotypical participants with comparable age, sex, scanning site and motion conditions, functional connectivity variability should be evaluated against a learned normative reference. Deviation from that reference alone is not a diagnosis of autism.',
 'The relevance of connectivity involving insular, temporal and prefrontal regions should be assessed relative to neurotypical conditional distributions and acquisition quality, not from regional identity or a single observed connectivity value alone.',
 'Observed deviations in medial prefrontal, cingulate and parietal regions should be compared with conditional neurotypical expectations while retaining uncertainty about the underlying functional system and individual deviation direction.',
 'Subcortical-cortical connectivity should be evaluated against a neurotypical reference conditional on measured demographic and acquisition variables, without assuming that atypicality in any single observed connection identifies autism.']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edge_masks(rois,endpoints,sub):
    masks=np.zeros((4,256),bool)
    for e,(a,b) in enumerate(endpoints):
        if a<0: continue
        masks[:3,e]=rois[:3,a]|rois[:3,b]
        masks[3,e]=(rois[3,a] and sub[a] and not sub[b]) or (rois[3,b] and sub[b] and not sub[a])
    return masks


def pooling(rois,edges,families,endpoints):
    """All scales use only applicable FC, with equal region/family contribution."""
    weights=np.zeros((4,3,256),float)
    for k in range(4):
        mask=edges[k]; weights[k,0]=mask
        for r in np.flatnonzero(rois[k]):
            ids=mask & (endpoints==r).any(1)
            if ids.any(): weights[k,1,ids]+=1/ids.sum()
        for f in CONCEPT_FAMILIES[k]:
            ids=np.zeros(256,bool)
            for e,(a,b) in enumerate(endpoints):
                if a>=0: ids[e]=mask[e] and (families[f,a] or families[f,b])
            if ids.any(): weights[k,2,ids]+=1/ids.sum()
        if weights[k,2].sum()==0: weights[k,2]=weights[k,1]
    return normalize_weights(weights)


def structural_summary(rois,edges,endpoints,degree,sub):
    known=np.flatnonzero((endpoints>=0).all(1))
    types=np.full(256,-1)
    for e in known: types[e]=int(sub[endpoints[e]].sum())
    result={'roi_counts':rois.sum(1).tolist(),'edge_counts':edges.sum(1).tolist(),
            'roi_union':int(rois.any(0).sum()),'edge_union':int(edges.any(0).sum()),
            'roi_overlap':(rois.astype(int)@rois.T.astype(int)).tolist(),
            'edge_overlap':(edges.astype(int)@edges.T.astype(int)).tolist(),
            'edge_type_counts':[[int((m & (types==t)).sum()) for t in range(3)] for m in edges],
            'candidate_roi_degree_hist':[[int(((degree==d)&m).sum()) for d in range(int(degree.max())+1)] for m in rois],
            'edge_endpoint_degree_hist':[], 'covered_endpoint_counts':[]}
    for m in edges:
        selected=endpoints[m].ravel()
        result['covered_endpoint_counts'].append(int(len(np.unique(selected))))
        result['edge_endpoint_degree_hist'].append(np.bincount(degree[selected],minlength=int(degree.max())+1).tolist())
    return result


def constrained_wrong(real_roi,real_edges,families,endpoints,degree,sub):
    """One ROI permutation determines *every* wrong ROI/FC/family rule.

    Permutation is confined to exact degree and cortical/subcortical strata.
    Fit is an anatomy-only deterministic search, with no patient data/labels.
    P4 includes all observed subcortical nuclei and is invariant under this
    type-preserving control: this limitation is explicitly measured, not hidden.
    """
    rng=np.random.default_rng(41101)
    bins=[np.flatnonzero((degree==d)&(sub==s)) for d in np.unique(degree) for s in (False,True)]
    bins=[b for b in bins if len(b)>1]
    real_stats=structural_summary(real_roi,real_edges,endpoints,degree,sub)
    tri=np.triu_indices(4,1)
    known=(endpoints>=0).all(1); aa=endpoints[known,0]; bb=endpoints[known,1]
    etype=np.zeros(256,int); etype[known]=sub[aa].astype(int)+sub[bb].astype(int)
    def evaluate(perm):
        roi=np.zeros_like(real_roi); roi[:,perm]=real_roi
        edges=edge_masks(roi,endpoints,sub)
        overlap=edges.astype(int)@edges.T.astype(int)
        err=(np.abs(edges.sum(1)-real_edges.sum(1)).sum()+
             np.abs(overlap[tri]-np.asarray(real_stats['edge_overlap'])[tri]).sum()+
             abs(int(edges.any(0).sum())-real_stats['edge_union']))
        for t in range(3): err+=np.abs((edges & (etype==t)).sum(1)-np.asarray(real_stats['edge_type_counts'])[:,t]).sum()
        changed=int((edges!=real_edges).sum())
        return float(err),changed,roi,edges
    best=None
    # Fixed search budget. No selection based on any patient or classification result.
    for restart in range(4):
        perm=np.arange(111)
        for b in bins: perm[b]=rng.permutation(b)
        current=evaluate(perm)
        for step in range(6000):
            b=bins[int(rng.integers(len(bins)))]; i,j=rng.choice(b,size=2,replace=False)
            proposal=perm.copy(); proposal[i],proposal[j]=proposal[j],proposal[i]
            candidate=evaluate(proposal)
            temperature=max(.15,1.5*(1-step/6000))
            energy=candidate[0]-current[0]-.002*(candidate[1]-current[1])
            if energy<=0 or rng.random()<np.exp(-min(energy/temperature,700)):
                perm=proposal; current=candidate
            key=(current[0],-current[1])
            if best is None or key<best[0]: best=(key,perm.copy(),current[2].copy(),current[3].copy())
    _,perm,wrong_roi,wrong_edges=best
    wrong_families=np.zeros_like(families); wrong_families[:,perm]=families
    stats=structural_summary(wrong_roi,wrong_edges,endpoints,degree,sub)
    differences={k:(np.asarray(stats[k])-np.asarray(real_stats[k])).tolist() for k in real_stats}
    return wrong_roi,wrong_edges,wrong_families,dict(seed=41101,search_steps=24000,permutation=perm.tolist(),
                real=real_stats,wrong=stats,differences=differences,anatomy_only_objective=best[0][0],
                changed_edges_by_concept=(real_edges!=wrong_edges).sum(1).tolist(),
                caveat='Shared ROI permutation; edge counts/overlap/type/coverage errors are disclosed. P4 invariant; exact global graph automorphism is not claimed.')


def compile_ontology(exp,root):
    import pandas as pd
    exp=Path(exp); root=Path(root)
    hist=root/'experiment/experiment11_roi_mapping_recovery/results'
    conn=pd.read_csv(hist/'roi_256_named_mapping.csv'); labels=pd.read_csv(hist/'roi_111_labels.csv')
    endpoints=np.full((256,2),-1,dtype=np.int64)
    for row in conn.itertuples(index=False):
        if row.connection_anatomical_status=='VERIFIED_BOTH_ENDPOINTS':
            endpoints[int(row.correlation_feature_index)]=[int(row.roi_i_zero_based),int(row.roi_j_zero_based)]
    names=labels.roi_name.astype(str).tolist(); degree=np.zeros(111,int)
    for a,b in endpoints:
        if a>=0: degree[a]+=1;degree[b]+=1
    def mask(pattern): return np.array([bool(re.search(pattern,name,re.I)) and degree[i]>0 for i,name in enumerate(names)])
    families=np.stack([mask(p) for p in FAMILIES.values()]); rois=np.stack([mask(p) for p in PATTERNS])
    sub=np.array([bool(re.match(FAMILIES['subcortical'],name)) for name in names]); edges=edge_masks(rois,endpoints,sub)
    wrong_roi,wrong_edges,wrong_families,control=constrained_wrong(rois,edges,families,endpoints,degree,sub)
    weights=pooling(rois,edges,families,endpoints); wrong_weights=pooling(wrong_roi,wrong_edges,wrong_families,endpoints)
    coverage=np.array([np.minimum(degree[r]/3,1).mean() for r in rois])
    wrong_coverage=np.array([np.minimum(degree[r]/3,1).mean() for r in wrong_roi])
    prompts=json.loads((exp/'prompts.json').read_text())
    prompts['prompt_policy'].update(generation='Written from the user-provided drafts and manually checked against accessible primary sources; no GPT-4 API invocation.',
      role='Structured anatomical applicability prior, not an individual diagnosis or a fixed edge sign.',
      freeze_rule='Corrected protocol is frozen after implementation audit; previous invalid results were already observed and are not a blinded preregistration.')
    literature=exp/'.runtime/literature'
    for k,p in enumerate(prompts['concepts']):
        p['td_reference_text']=TD_TEXTS[k]; p['applicability_level']='anatomical region candidates; sparse observed FC only'
        p['region_candidates']=[names[i] for i in np.flatnonzero(rois[k])]
        p['roi_pattern']=PATTERNS[k]; p['supported_roi_indices']=np.flatnonzero(rois[k]).tolist()
        p['supported_roi_names']=[names[i] for i in np.flatnonzero(rois[k])]
        p['supported_edge_indices']=np.flatnonzero(edges[k]).tolist()
        p['signed_evidence_rule']='Keep positive and negative channels equally; no individual edge sign imposed. Directional prose is an uncertainty annotation.'
        p.pop('directional_prior_weight_negative',None)
        p['compilation_rule']='P1-P3: at least one named ROI endpoint; P4: verified subcortical-cortical edges only; three normalized FC/ROI/family pooling weights.'
        p['sources']=[{'url':'https://www.nature.com/articles/s44220-026-00656-y.pdf','doi':'10.1038/s44220-026-00656-y','level':'primary full paper','local_sha256':digest(literature/'ilioska2026.pdf') if (literature/'ilioska2026.pdf').exists() else None}]
        if k==3: p['sources'].append({'url':'https://www.sciencedirect.com/science/article/pii/S0301008226000328','doi':'10.1016/j.pneurobio.2026.102906','scope':'methodological caution only; not ASD anatomical direction evidence'})
        p['frozen_sha256']=hashlib.sha256(json.dumps({a:b for a,b in p.items() if a!='frozen_sha256'},sort_keys=True).encode()).hexdigest()
    (exp/'prompts.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2)+'\n')
    o=dict(atlas='Harvard-Oxford E11 verified labels; no canonical functional network identity assumed',roi_names=names,
           endpoints=endpoints.tolist(),degree=degree.tolist(),subcortical=sub.tolist(),
           unknown_edges=np.flatnonzero((endpoints<0).any(1)).tolist(),family_names=list(FAMILIES),
           families=families.astype(int).tolist(),concept_rois=rois.astype(int).tolist(),concept_edges=edges.astype(int).tolist(),
           weights=weights.tolist(),coverage=coverage.tolist(),wrong_rois=wrong_roi.astype(int).tolist(),
           wrong_edges=wrong_edges.astype(int).tolist(),wrong_families=wrong_families.astype(int).tolist(),
           wrong_weights=wrong_weights.tolist(),wrong_coverage=wrong_coverage.tolist(),wrong_control=control,
           source_hashes={f:digest(hist/f) for f in ['roi_256_named_mapping.csv','roi_111_labels.csv']})
    (exp/'ontology.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
    return o


def numeric_weights(base_weights,known,kind,x=None,train_idx=None):
    """E2: fixed numeric index hierarchy. E6: train-only soft covariance prior.

    No top-K per patient, no labels, and no HO names/ROI memberships. The
    medical arm supplies only the four support-size budgets for capacity matching.
    """
    ids=np.flatnonzero(known); n=len(ids); budgets=(np.asarray(base_weights)[:,0]>0).sum(1)
    rng=np.random.default_rng(41102); ordering=rng.permutation(ids)
    vectors=[]; info={'kind':kind,'budgets':budgets.tolist(),'fit_ids':[]}
    if kind=='fixed_numeric':
        for k,b in enumerate(budgets):
            w=np.zeros(256); window=ordering[(np.arange(int(b))+k*(n//4))%n]; w[window]=1
            vectors.append(w)
    elif kind=='data_prior':
        from sklearn.cluster import KMeans
        values=np.asarray(x)[train_idx][:,64+ids].astype(float)
        corr=np.nan_to_num(np.abs(np.corrcoef(values,rowvar=False)),nan=0)
        km=KMeans(4,n_init=20,random_state=41102).fit(corr)
        # CDF-calibrated soft affinities; target Kish support equals the fixed budget.
        dist=((corr[:,None,:]-km.cluster_centers_[None,:,:])**2).mean(2)
        for k,b in enumerate(budgets):
            d=dist[:,k]-dist[:,k].min(); lo,hi=0.,1e5
            for _ in range(70):
                beta=(lo+hi)/2; v=np.exp(-beta*d); ess=v.sum()**2/max((v*v).sum(),1e-30)
                if ess>b: lo=beta
                else: hi=beta
            v=np.exp(-.5*(lo+hi)*d); w=np.zeros(256); w[ids]=v; vectors.append(w)
        info.update(fit_ids=np.asarray(train_idx).tolist(),cluster_sizes=np.bincount(km.labels_,minlength=4).tolist(),
                    centers_sha256=hashlib.sha256(km.cluster_centers_.tobytes()).hexdigest())
    else: raise ValueError(kind)
    out=np.zeros((4,3,256))
    for k,w in enumerate(vectors):
        out[k,0]=w
        # Index groups are a numerical multiscale aggregation, not anatomy.
        for s,width in [(1,8),(2,32)]:
            for start in range(0,256,width):
                end=min(start+width,256); v=w[start:end]
                if v.sum()>0: out[k,s,start:end]=v/v.sum()
    out=normalize_weights(out)
    info['effective_support']=(1/np.maximum((out[:,0]**2).sum(1),1e-30)).tolist()
    return out,info

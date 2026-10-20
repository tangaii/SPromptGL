"""Experiment29 frozen text preparation. No CLIP encoding, patient data or labels.

The existing Exp28 RN50/SVD cache and verified Exp11 HO mapping are authoritative.
All 512 normalized 64D vectors share ONE mean subtraction, then L2 normalization.
"""
from __future__ import annotations
import ast,csv,hashlib,json
from pathlib import Path
import numpy as np

ORIGINAL=('Nromal brain volume','Uniform cortical thickness','Symmetrical brain structure',
          'Normal ventricular morphology','Altered brain volume','Abnormal functional connectivity',
          'White matter integrity issues','Different activation patterns')
EXCLUDED=(195,225,232,236)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(x):
    x=np.asarray(x,np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def geometry(x):
    x=np.asarray(x,np.float64).reshape(-1,np.asarray(x).shape[-1]);u=unit(x)
    centered=u-u.mean(0);s=np.linalg.svd(centered,compute_uv=False)
    p=s*s/max(float((s*s).sum()),1e-30);p=p[p>0]
    pair=u@u.T;pair=pair[np.triu_indices(len(u),1)]
    return dict(effective_rank=float(np.exp(-(p*np.log(p)).sum())),
                numerical_rank=int((s>max(s[0],1e-20)*1e-6).sum()),
                variance=float((centered*centered).sum()/len(u)),
                mean_vector_norm=float(np.linalg.norm(u.mean(0))),
                pair_cos_quantiles=np.quantile(pair,[0,.1,.5,.9,1]).tolist())

def verify_mapping(root,payload,cache):
    root=Path(root);folder=root/'experiment/experiment11_roi_mapping_recovery/results'
    tree=ast.parse((root/'network.py').read_text())
    words=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='attr_words' for t in n.targets))
    assert tuple(words)==tuple(payload['original'])==ORIGINAL
    def read(n):
        with (folder/n).open(newline='') as f:return list(csv.DictReader(f))
    labels={int(r['roi_index_zero_based']):r for r in read('roi_111_labels.csv')}
    rows=read('roi_256_named_mapping.csv');assert len(labels)==111 and len(rows)==256
    ti,tj=np.triu_indices(111,1);verified=[]
    for e,r in enumerate(rows):
        assert int(r['correlation_feature_index'])==e and r['verification_status']=='VERIFIED_NUMERIC_UNIQUE'
        i,j,old=(int(r[k]) for k in ('roi_i_zero_based','roi_j_zero_based','original_connection_index'))
        assert (i,j)==(ti[old],tj[old])
        if r['usable_for_roi_semantic_prompt']!='True':assert e in EXCLUDED;continue
        a,b=labels[i],labels[j]
        assert all(v[k]=='True' for v in (a,b) for k in ('name_verified','column_order_verified'))
        assert r['connection_name']==a['roi_name']+' — '+b['roi_name']
        assert 'UNKNOWN' not in r['connection_name']
        assert int(r['roi_i_atlas_value'])==int(a['atlas_integer_value'])
        assert int(r['roi_j_atlas_value'])==int(b['atlas_integer_value'])
        record=payload['edges'][len(verified)]
        assert (record['corr_index'],record['roi_i_zero_based'],record['roi_j_zero_based'])==(e,i,j)
        assert record['left']==a['roi_name'] and record['right']==b['roi_name']
        for direction,text,h in zip(('increased','decreased'),record['prompts'],record['text_sha256']):
            expected=payload['templates'][payload['selected_template']].format(direction=direction,left=record['left'],right=record['right'])
            assert text==expected and hashlib.sha256(text.encode()).hexdigest()==h
        verified.append(e)
    assert np.array_equal(verified,cache['valid_indices']) and len(verified)==252
    assert sorted(set(range(256))-set(verified))==list(EXCLUDED)
    return verified

def build(root,exp):
    root,exp=Path(root),Path(exp);hist=root/'experiment/experiment28_imaging_anchored_finegrained_semantic_alignment'
    payload=json.loads((hist/'prompt_banks.json').read_text())
    hm=json.loads((hist/'manifest.json').read_text())
    assert sha(hist/'prompt_banks.npz')==hm['prompt_cache_sha256']
    with np.load(hist/'prompt_banks.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    verify_mapping(root,payload,old)
    before=np.concatenate((old['original64'],old['fine'].reshape(504,64))).astype(np.float64)
    assert before.shape==(512,64) and np.allclose(np.linalg.norm(before,axis=1),1,atol=2e-7)
    common=before.mean(0);real=unit(before-common)
    mean=real.mean(0);centered=real-mean
    _,s,v=np.linalg.svd(centered,full_matrices=False)
    noise=np.random.default_rng(29008).normal(size=(512,64));noise-=noise.mean(0)
    q,_=np.linalg.qr(noise,mode='reduced');numeric0=(q*s)@v+mean
    cov_error=float(np.max(np.abs((numeric0-mean).T@(numeric0-mean)-centered.T@centered)))
    assert np.allclose(numeric0.mean(0),mean,atol=1e-12) and cov_error<1e-9
    numeric=unit(numeric0)
    def cos(a,b):return float(unit(a)@unit(b))
    def direction(x):return float(np.sum(unit(x).reshape(252,2,-1)[:,0]*unit(x).reshape(252,2,-1)[:,1],axis=-1).mean())
    metadata=dict(source_npz_sha256=sha(hist/'prompt_banks.npz'),source_json_sha256=sha(hist/'prompt_banks.json'),
        clip_weights_sha256=payload['clip_weights_sha256'],clip_encoding='REUSED_NO_ENCODING',
        original=list(ORIGINAL),edges=payload['edges'],fine_text_sha256=payload['fine_text_sha256'],
        raw1024_geometry=geometry(np.concatenate((old['p0'],old['fine_raw']))),
        old64_geometry=geometry(before),centered64_geometry=geometry(real),numeric64_geometry=geometry(numeric),
        global_class_cos_old=cos(before[4:8].mean(0),before[:4].mean(0)),
        global_class_cos_centered=cos(real[4:8].mean(0),real[:4].mean(0)),
        global_class_cos_numeric=cos(numeric[4:8].mean(0),numeric[:4].mean(0)),
        direction_cos_raw1024=direction(old['fine_raw']),direction_cos_old64=direction(before[8:]),
        direction_cos_centered64=direction(real[8:]),
        roi_identity_distance_centered=float(np.linalg.norm(real[8:].reshape(252,2,64)-real[8:].reshape(252,2,64)[old['roi_permutation']],axis=-1).mean()),
        numeric_seed=29008,numeric_pre_normalization_covariance_error=cov_error,
        numeric_post_normalization_mean_error=float(np.max(np.abs(numeric.mean(0)-real.mean(0)))),
        numeric_post_normalization_covariance_error=float(np.max(np.abs(np.cov(numeric,rowvar=False,bias=True)-np.cov(real,rowvar=False,bias=True)))),
        numeric_gram_mean_abs_difference=float(np.abs(numeric@numeric.T-real@real.T).mean()),
        numeric_boundary='Dimension and unit norms exact; mean/covariance exact BEFORE final normalization only; individual Gram, global class geometry and anatomical identity not preserved.',
        mean_rule='Subtract one mean of all 8+504 historical UNIT 64D text vectors; normalize rows; no patient data, labels or tuning.')
    tmp=exp/'prompt_banks.tmp.npz'
    np.savez_compressed(tmp,p0=old['p0'],original=real[:8].astype(np.float32),fine=real[8:].reshape(252,2,64).astype(np.float32),
                        numeric_original=numeric[:8].astype(np.float32),numeric_fine=numeric[8:].reshape(252,2,64).astype(np.float32),
                        common_mean=common.astype(np.float32),valid_indices=old['valid_indices'],roi_permutation=old['roi_permutation'])
    tmp.replace(exp/'prompt_banks.npz')
    return metadata

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

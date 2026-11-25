"""Frozen original/fine FC text banks; no patient values or labels are used here."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ORIGINAL = ('Nromal brain volume', 'Uniform cortical thickness',
            'Symmetrical brain structure', 'Normal ventricular morphology',
            'Altered brain volume', 'Abnormal functional connectivity',
            'White matter integrity issues', 'Different activation patterns')
EXCLUDED = (195, 225, 232, 236)
TEMPLATES = (
    'Potential autism-associated {direction} resting-state fMRI connectivity between the {left} and the {right}.',
    'Potential autism-associated {direction} FC: {left}; {right}.',
)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalized(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def geometry(x):
    x = np.asarray(x, np.float64).reshape(-1, np.asarray(x).shape[-1])
    u = normalized(x); c = u-u.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    p = s*s / max(float((s*s).sum()), 1e-30)
    gram = u @ u.T
    values = gram[np.triu_indices(len(u), 1)]
    return dict(norm_min=float(np.linalg.norm(x, axis=1).min()),
                norm_mean=float(np.linalg.norm(x, axis=1).mean()),
                norm_max=float(np.linalg.norm(x, axis=1).max()),
                centered_variance=float((c*c).sum()/len(c)),
                effective_rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),
                numerical_rank=int((s>max(s[0], 1e-20)*1e-6).sum()),
                pair_cos_quantiles=np.quantile(values, [0,.1,.5,.9,1]).tolist())

def mapping(root):
    root = Path(root)
    tree = ast.parse((root/'network.py').read_text())
    words = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id=='attr_words' for t in n.targets))
    assert tuple(words)==ORIGINAL
    folder = root/'experiment/experiment11_roi_mapping_recovery/results'
    def read(name):
        with (folder/name).open(newline='') as f: return list(csv.DictReader(f))
    atlas = {int(r['roi_index_zero_based']):r for r in read('roi_111_labels.csv')}
    rows = read('roi_256_named_mapping.csv')
    assert len(atlas)==111 and len(rows)==256
    ti,tj = np.triu_indices(111,1)
    out=[]
    for e,r in enumerate(rows):
        assert int(r['correlation_feature_index'])==e
        assert r['verification_status']=='VERIFIED_NUMERIC_UNIQUE'
        i,j,o = (int(r[k]) for k in ('roi_i_zero_based','roi_j_zero_based','original_connection_index'))
        assert (i,j)==(ti[o],tj[o])
        if r['usable_for_roi_semantic_prompt']!='True':
            assert e in EXCLUDED
            continue
        a,b=atlas[i],atlas[j]
        assert all(v[k]=='True' for v in (a,b) for k in ('name_verified','column_order_verified'))
        assert int(r['roi_i_atlas_value'])==int(a['atlas_integer_value'])
        assert int(r['roi_j_atlas_value'])==int(b['atlas_integer_value'])
        assert r['connection_name']==a['roi_name']+' — '+b['roi_name']
        assert 'UNKNOWN' not in r['connection_name']
        out.append(dict(corr_index=e, original_connection_index=o,
                        roi_i_zero_based=i, roi_j_zero_based=j,
                        roi_i_atlas_value=int(a['atlas_integer_value']),
                        roi_j_atlas_value=int(b['atlas_integer_value']),
                        left=a['roi_name'], right=b['roi_name'],
                        connection_name=r['connection_name'],
                        verification_status=r['verification_status'],
                        evidence_status=r['evidence_status']))
    assert len(out)==252
    assert sorted(set(range(256))-{r['corr_index'] for r in out})==list(EXCLUDED)
    return words,out

def build(root, exp):
    """Encode once on GPU0; preserve separate 8-sentence P0 encoding exactly."""
    import sys
    import torch
    root,exp=Path(root),Path(exp)
    sys.path.insert(0,str(root))
    from CLIP.clip import clip
    words,edges=mapping(root)
    fallback=[]
    for index,template in enumerate(TEMPLATES):
        texts=[template.format(direction=d,left=r['left'],right=r['right'])
               for r in edges for d in ('increased','decreased')]
        try:
            tokens=clip.tokenize(texts,truncate=False)
            selected=index
            break
        except RuntimeError as exc:
            fallback.append(dict(template_index=index, reason=str(exc)))
    else: raise RuntimeError('Neither uniform template fits CLIP77; no text truncation allowed')
    assert tokens.shape==(504,77)
    for k,r in enumerate(edges):
        r['prompts']=texts[2*k:2*k+2]
        r['text_sha256']=[hashlib.sha256(t.encode()).hexdigest() for t in r['prompts']]
        r['cache_indices']=[8+2*k,9+2*k]
    model,_=clip.load('RN50',device='cuda:0',jit=False)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        p0=model.encode_text(clip.tokenize(words,truncate=False).to('cuda:0')).float().cpu().numpy()
        blocks=[]
        for start in range(0,504,64):
            blocks.append(model.encode_text(tokens[start:start+64].to('cuda:0')).float().cpu().numpy())
    fine=np.concatenate(blocks)
    assert p0.shape==(8,1024) and fine.shape==(504,1024)
    # The exact historical P0 cache is authoritative for baseline parity.
    old=root/'experiment/experiment12_roi_semantic_alignment_feasibility/cache/roi_text_embeddings.pt'
    historical=torch.load(old,map_location='cpu',weights_only=False)['p0_features'].numpy()
    assert np.array_equal(p0,historical),'RN50 P0 output differs from historical cache'
    allraw=np.concatenate([p0,fine]);unit=normalized(allraw)
    _,singular,vt=np.linalg.svd(unit,full_matrices=False)
    basis=vt[:64].T.copy()
    for col in range(64):
        if basis[np.abs(basis[:,col]).argmax(),col]<0: basis[:,col]*=-1
    projected=normalized(unit@basis)
    fine64=projected[8:]
    # Numeric control: randomized row coordinates with EXACT pre-normalization
    # mean/covariance, then unit norms. It does not preserve individual pair Gram.
    mean=fine64.mean(0);centered=fine64-mean
    _,cs,cv=np.linalg.svd(centered,full_matrices=False)
    noise=np.random.default_rng(28005).normal(size=(504,64));noise-=noise.mean(0)
    q,_=np.linalg.qr(noise,mode='reduced')
    numeric0=(q*cs)@cv+mean
    assert np.allclose(numeric0.mean(0),mean,atol=1e-12)
    assert np.allclose((numeric0-mean).T@(numeric0-mean),centered.T@centered,atol=1e-10)
    numeric=normalized(numeric0)
    order=np.random.default_rng(28006).permutation(252)
    perm=np.arange(252);perm[order]=np.roll(order,-1)
    assert (perm!=np.arange(252)).all()
    fine64=fine64.reshape(252,2,64)
    coarse=normalized(projected[4:8].mean(0))
    coarse=np.broadcast_to(coarse,(252,2,64)).copy()
    def dircos(x): return np.sum(x[:,0]*x[:,1],-1)
    diagnostics=dict(raw=geometry(allraw), projected=geometry(projected),
                     fine_raw=geometry(fine), fine_projected=geometry(fine64),
                     numeric=geometry(numeric),
                     fine_direction_cos_raw_mean=float(dircos(normalized(fine).reshape(252,2,1024)).mean()),
                     fine_direction_cos_projected_mean=float(dircos(fine64).mean()),
                     coarse_normal_abnormal_cos=float(normalized(projected[:4].mean(0))@normalized(projected[4:8].mean(0))),
                     wrong_roi_distance=float(np.linalg.norm(fine64-fine64[perm],axis=-1).mean()),
                     projected_energy_fraction=float((singular[:64]**2).sum()/(singular**2).sum()),
                     numeric_pre_normalization_covariance_error=float(np.max(np.abs((numeric0-mean).T@(numeric0-mean)-centered.T@centered))),
                     numeric_pair_gram_abs_difference=float(np.abs(numeric@numeric.T-fine64.reshape(504,64)@fine64.reshape(504,64).T).mean()))
    informative=(diagnostics['fine_projected']['centered_variance']>1e-8
                 and diagnostics['wrong_roi_distance']>1e-5
                 and float(np.linalg.norm(fine64[:,0]-fine64[:,1],axis=-1).mean())>1e-5)
    payload=dict(original=words,edges=edges,dimensions=dict(raw=1024,semantic=64),
                 templates=list(TEMPLATES),selected_template=selected,fallbacks=fallback,
                 token_count_max=int((tokens!=0).sum(1).max()),truncation=False,
                 excluded_indices=list(EXCLUDED),projection='uncentered SVD on normalized 512 text vectors; deterministic sign; no whitening',
                 coarse_mapping='Both ASD-associated directions use normalized mean of the four original abnormal sentences; normal sentences participate in the shared basis.',
                 numeric_control='Seed28005 centered orthonormal random row coordinates, colored to exact text mean/covariance before final unit normalization; individual Gram/identity not preserved.',
                 diagnostics=diagnostics,informative=bool(informative),
                 clip_weights_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),
                 original_text_sha256=hashlib.sha256(json.dumps(words,ensure_ascii=False).encode()).hexdigest(),
                 fine_text_sha256=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest())
    exp.mkdir(parents=True,exist_ok=True)
    temp=exp/'prompt_banks.tmp.npz'
    np.savez_compressed(temp,p0=p0,fine_raw=fine,shared_basis=basis.astype(np.float32),
                        original64=projected[:8].astype(np.float32),fine=fine64.astype(np.float32),
                        coarse=coarse.astype(np.float32),numeric=numeric.reshape(252,2,64).astype(np.float32),
                        valid_indices=np.array([r['corr_index'] for r in edges],np.int64),
                        roi_permutation=perm.astype(np.int64))
    temp.replace(exp/'prompt_banks.npz')
    (exp/'prompt_banks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    del model
    torch.cuda.empty_cache()
    return payload

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

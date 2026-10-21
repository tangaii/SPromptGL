"""Frozen 764-sentence RN50 bank for Experiment 30.

The bank is built once during prepare.  No patient feature, label, site or
evaluation result enters the text projection or its permutations.
"""
from __future__ import annotations
import ast, csv, hashlib, json
from pathlib import Path
import numpy as np

ORIGINAL = ('Nromal brain volume','Uniform cortical thickness','Symmetrical brain structure',
            'Normal ventricular morphology','Altered brain volume','Abnormal functional connectivity',
            'White matter integrity issues','Different activation patterns')
EXCLUDED = (195,225,232,236)
TD_TEMPLATE = 'Typical resting-state fMRI connectivity between the {left} and {right} in neurotypical controls.'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(x):
    x=np.asarray(x,np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def _read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def verify_mapping(root, payload):
    root=Path(root); folder=root/'experiment/experiment11_roi_mapping_recovery/results'
    tree=ast.parse((root/'network.py').read_text())
    words=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='attr_words' for t in n.targets))
    assert tuple(words)==ORIGINAL==tuple(payload['original'])
    labels={int(r['roi_index_zero_based']):r for r in _read(folder/'roi_111_labels.csv')}
    rows=_read(folder/'roi_256_named_mapping.csv'); assert len(labels)==111 and len(rows)==256
    ti,tj=np.triu_indices(111,1);valid=[]
    for e,r in enumerate(rows):
        assert int(r['correlation_feature_index'])==e and r['verification_status']=='VERIFIED_NUMERIC_UNIQUE'
        i,j,old=(int(r[k]) for k in ('roi_i_zero_based','roi_j_zero_based','original_connection_index'))
        assert (i,j)==(ti[old],tj[old])
        if r['usable_for_roi_semantic_prompt']!='True': assert e in EXCLUDED; continue
        a,b=labels[i],labels[j]
        assert all(v[k]=='True' for v in (a,b) for k in ('name_verified','column_order_verified'))
        assert r['connection_name']==a['roi_name']+' — '+b['roi_name']
        assert 'UNKNOWN' not in r['connection_name']
        assert int(r['roi_i_atlas_value'])==int(a['atlas_integer_value'])
        assert int(r['roi_j_atlas_value'])==int(b['atlas_integer_value'])
        valid.append(e)
    assert valid==list(payload['valid_indices']) and len(valid)==252
    assert sorted(set(range(256))-set(valid))==list(EXCLUDED)
    return labels,rows


def _derangement(n, seed):
    rng=np.random.default_rng(seed); p=rng.permutation(n)
    while np.any(p==np.arange(n)): p=rng.permutation(n)
    return p.astype(np.int64)


def _encode(sentences):
    import sys, torch
    root=Path(__file__).resolve().parents[2]
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    from CLIP.clip import clip
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_=clip.load('RN50',device=device,jit=False); model.eval()
    vals=[]
    with torch.no_grad():
        for start in range(0,len(sentences),48):
            tok=clip.tokenize(sentences[start:start+48],truncate=False).to(device)
            assert tok.shape[1]==77
            vals.append(model.encode_text(tok).float().cpu().numpy())
    del model
    out=np.concatenate(vals,axis=0)
    assert out.shape==(764,1024) and np.isfinite(out).all()
    return out


def build(root, exp):
    root,exp=Path(root),Path(exp)
    source=root/'experiment/experiment28_imaging_anchored_finegrained_semantic_alignment'
    payload=json.loads((source/'prompt_banks.json').read_text())
    old=np.load(source/'prompt_banks.npz',allow_pickle=False)
    valid=old['valid_indices'].astype(np.int64)
    verify_mapping(root,dict(payload,valid_indices=valid.tolist()))
    edges=payload['edges']; assert len(edges)==252
    asd=[x for e in edges for x in e['prompts']]
    td=[TD_TEMPLATE.format(left=e['left'],right=e['right']) for e in edges]
    sentences=list(ORIGINAL)+asd+td; assert len(sentences)==764
    raw=_encode(sentences); normalized=unit(raw); common=normalized.mean(0)
    centered=normalized-common
    _,s,vt=np.linalg.svd(centered,full_matrices=False)
    projected=unit(centered@vt[:64,:].T).astype(np.float32)
    assert projected.shape==(764,64) and np.isfinite(projected).all()
    # Independent frozen controls.  They are permutations of local rows only.
    roi_perm=_derangement(252,30030)
    scramble_asd=_derangement(252,30031); scramble_td=_derangement(252,30032)
    metadata=dict(source_prompt_json_sha256=sha(source/'prompt_banks.json'),
        source_mapping_sha256=sha(root/'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv'),
        original_text=list(ORIGINAL),td_template=TD_TEMPLATE,valid_indices=valid.tolist(),
        excluded_indices=list(EXCLUDED),roi_permutation=roi_perm.tolist(),
        scramble_asd=scramble_asd.tolist(),scramble_td=scramble_td.tolist(),
        sentences_sha256=[hashlib.sha256(x.encode()).hexdigest() for x in sentences],
        text_raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        pca_energy=float((s[:64]**2).sum()/max((s**2).sum(),1e-30)),
        original_geometry={'class_cos':float(projected[4:].mean(0)@projected[:4].mean(0))},
        asd_td_same_edge_cos=float(np.mean(np.sum(projected[8:512:2]*projected[512:],axis=1))),
        asd_inc_dec_cos=float(np.mean(np.sum(projected[8::2]*projected[9::2],axis=1))),
        encoding='single frozen RN50 pass, normalized -> common mean -> text-only PCA64 -> row L2',
        no_patient_data=True)
    np.savez_compressed(exp/'prompt_banks.npz',original=projected[:8],
        fine_asd=projected[8:512].reshape(252,2,64),fine_td=projected[512:764],
        numeric_original=np.zeros((8,64),np.float32),numeric_fine=np.zeros((252,2,64),np.float32),
        valid_indices=valid,roi_permutation=roi_perm,scramble_asd=scramble_asd,scramble_td=scramble_td,
        p0=raw[:8].astype(np.float32))
    return metadata


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


if __name__=='__main__': print('prompt_bank module ready')

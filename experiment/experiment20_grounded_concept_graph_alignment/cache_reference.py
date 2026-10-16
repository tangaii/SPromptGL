"""Frozen Experiment 20 prior; mathematical operations match the user reference."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def prepare_static_cache(root, encode_text, real_pairs, neutral_pairs):
    root = Path(root)
    folder = root/'experiment/experiment11_roi_mapping_recovery/results'
    atlas = {int(r['roi_index_zero_based']):r for r in read_csv(folder/'roi_111_labels.csv')}
    original = read_csv(folder/'roi_256_named_mapping.csv')
    assert len(atlas)==111 and len(original)==256
    original_pairs = np.triu_indices(111, 1)
    valid, ends, descriptions = [], [], []
    for index,r in enumerate(original):
        assert int(r['correlation_feature_index']) == index
        assert r['verification_status'] == 'VERIFIED_NUMERIC_UNIQUE'
        if r['usable_for_roi_semantic_prompt']!='True':
            assert index in {195,225,232,236}
            continue
        i,j = int(r['roi_i_zero_based']),int(r['roi_j_zero_based'])
        original_index = int(r['original_connection_index'])
        assert (i, j) == (int(original_pairs[0][original_index]), int(original_pairs[1][original_index]))
        assert atlas[i]['name_verified']=='True' and atlas[j]['name_verified']=='True'
        assert atlas[i]['column_order_verified']=='True' and atlas[j]['column_order_verified']=='True'
        assert int(r['roi_i_atlas_value'])==int(atlas[i]['atlas_integer_value'])
        assert int(r['roi_j_atlas_value'])==int(atlas[j]['atlas_integer_value'])
        left,right=atlas[i]['roi_name'],atlas[j]['roi_name']
        valid.append(index); ends.append((i,j))
        descriptions.append(f'Functional connectivity between {left} and {right}.')
    assert len(valid)==252 and sorted(set(range(256))-set(valid))==[195,225,232,236]
    ends=np.asarray(ends,dtype=np.int64)
    anchor_sets = [
     ['Superior Temporal Gyrus','Middle Temporal Gyrus','Angular Gyrus', 'Supramarginal Gyrus','Temporal Pole'],
     ['Superior Frontal Gyrus','Middle Frontal Gyrus','Inferior Frontal Gyrus','Frontal Pole','Superior Parietal Lobule'],
     ['Thalamus','Precentral Gyrus','Postcentral Gyrus',"Heschl's Gyrus",'Central Opercular Cortex'],
     ['Cingulate Gyrus','Insular Cortex','Frontal Operculum Cortex']]
    def clean(idx):
        return atlas[int(idx)]['roi_name'].removeprefix('Left ').removeprefix('Right ').casefold()
    masks=np.zeros((4,252), dtype=bool)
    for e,(i,j) in enumerate(ends):
        names=(clean(i),clean(j))
        for k,anchors in enumerate(anchor_sets):
            masks[k,e]=any(n.startswith(a.casefold()) for n in names for a in anchors)
    assert masks.sum(axis=1).tolist()==[72,56,45,32]
    line=((ends[:,None,:,None]==ends[None,:,None,:]).any(axis=-1).any(axis=-1))
    assert np.all(line==line.T) and np.diag(line).all()
    perm=np.random.default_rng(201020).permutation(252)
    while np.any(perm==np.arange(252)):perm=np.random.default_rng(201020+int(perm[0])).permutation(252)
    wrong=masks[:,perm]
    assert np.array_equal(masks.astype(int)@masks.T.astype(int),wrong.astype(int)@wrong.T.astype(int))
    assert len(real_pairs)==len(neutral_pairs)==4
    def q_for(pairs):
        text=[t for pair in pairs for t in pair]
        v=F.normalize(encode_text(text).float(),dim=-1).reshape(4,2,-1)
        q=F.normalize(v[:,0]-v[:,1],dim=-1)
        assert q.shape==(4,1024) and torch.isfinite(q).all()
        return q
    qr,qn=q_for(real_pairs),q_for(neutral_pairs)
    qr_np=qr.cpu().numpy().astype('float64')
    b=np.linalg.qr(qr_np.T,mode='reduced')[0]
    rng=np.random.default_rng(202020)
    rand=rng.normal(size=(1024,4));rand-=b@(b.T@rand)
    rb=np.linalg.qr(rand,mode='reduced')[0]
    qiso=torch.tensor(((qr_np@b)@rb.T).astype('float32'))
    assert (qiso@qiso.T-qr@qr.T).abs().max()<1e-5
    edge_name=F.normalize(encode_text(descriptions).float(),dim=-1)
    edge_anon=F.normalize(encode_text([f'Functional connectivity feature {i}.' for i in valid]).float(),dim=-1)
    assert edge_name.shape==edge_anon.shape==(252,1024)
    def prior(query,emb):
        cos=query@emb.T
        out=[]
        for k in range(4):
            v=cos[k];chosen=v[torch.tensor(masks[k])]
            sd=chosen.std(unbiased=False).clamp_min(1e-4)
            row=torch.tanh(0.5*(v-chosen.mean())/sd)
            out.append(row)
        return torch.stack(out).cpu().numpy().astype('float32')
    return {
      'valid_idx':np.asarray(valid,dtype='int64'),
      'masks':masks, 'masks_wrong':wrong, 'line_graph':line,
      'prior_real':prior(qr,edge_name), 'prior_neutral':prior(qn,edge_name),
      'prior_random':prior(qiso,edge_name), 'prior_anonymous':prior(qr,edge_anon),
      'roi_endpoints':ends, 'edge_permutation':perm.astype('int64'),
      'real_query_gram':(qr@qr.T).cpu().numpy(), 'neutral_query_gram':(qn@qn.T).cpu().numpy(),
    }

"""Verify Experiment 11 indices, freeze edge texts, and encode RN50 once."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
REF = ROOT / 'experiment/experiment11_roi_mapping_recovery/results'
EXTRA = [195, 225, 232, 236]
SHUFFLE_SEED = 1200


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_inputs():
    frame = pd.read_csv(ROOT / 'data/ABIDE/processed_standard_data.csv')
    modal = np.load(ROOT / 'data/ABIDE/modal_feat_dict.npy', allow_pickle=True).item()
    assert list(modal) == ['PHENO', 'ANAT', 'FUNC', 'Correlation']
    assert [len(x) for x in modal.values()] == [48, 6, 10, 256]
    columns = [str(c) for v in modal.values() for c in v]
    assert list(frame.columns[:-1]) == columns
    assert columns[64:] == [str(i) for i in range(256)]
    assert frame.shape == (871, 321) and np.isfinite(frame.values).all()
    labels = frame.iloc[:, -1].values.astype(int) - 1
    assert set(labels) == {0, 1} and list(np.bincount(labels)) == [403, 468]
    return frame, modal, labels


def build():
    validate_inputs()
    numeric = pd.read_csv(REF / 'roi_256_mapping.csv')
    named = pd.read_csv(REF / 'roi_256_named_mapping.csv')
    rois = pd.read_csv(REF / 'roi_111_labels.csv').set_index('roi_index_zero_based')
    assert len(rois) == 111 and set(rois.index) == set(range(111))
    assert int(rois.loc[82, 'atlas_integer_value']) == 3455
    assert rois.is_extra_roi.sum() == 1 and rois.name_verified.sum() == 110
    assert rois.loc[82, 'roi_name'] == 'UNKNOWN'
    assert len(numeric) == len(named) == 256
    assert list(named.correlation_feature_index) == list(range(256))
    assert list(numeric.correlation_feature_index) == list(range(256))
    assert named.original_connection_index.nunique() == 256
    assert numeric.original_connection_index.nunique() == 256
    for col in ['correlation_feature_index', 'original_connection_index',
                'roi_i_zero_based', 'roi_j_zero_based']:
        assert np.array_equal(numeric[col], named[col]), col
    triu_i, triu_j = np.triu_indices(111, k=1)
    hashes = {name: sha256(REF / name) for name in
              ['roi_111_labels.csv', 'roi_256_named_mapping.csv', 'roi_256_mapping.csv']}
    rows, valid = [], []
    for record in named.to_dict('records'):
        e, ci = int(record['correlation_feature_index']), int(record['original_connection_index'])
        i, j = int(record['roi_i_zero_based']), int(record['roi_j_zero_based'])
        assert 0 <= i < j <= 110
        assert (int(triu_i[ci]), int(triu_j[ci])) == (i, j)
        a, b = rois.loc[i], rois.loc[j]
        extra = bool(a.is_extra_roi or b.is_extra_roi)
        good = bool(a.name_verified and b.name_verified and a.column_order_verified
                    and b.column_order_verified and not extra)
        assert good == bool(record['usable_for_roi_semantic_prompt'])
        assert int(record['roi_i_atlas_value']) == int(a.atlas_integer_value)
        assert int(record['roi_j_atlas_value']) == int(b.atlas_integer_value)
        prompt = ''
        if good:
            assert a.roi_name != 'UNKNOWN' and b.roi_name != 'UNKNOWN'
            assert record['connection_name'] == f'{a.roi_name} — {b.roi_name}'
            prompt = f'Resting-state functional connectivity between the {a.roi_name} and the {b.roi_name}.'
            valid.append(e)
        rows.append(dict(correlation_feature_index=e, original_connection_index=ci,
                         roi_i_zero_based=i, roi_j_zero_based=j,
                         roi_i_atlas_value=int(a.atlas_integer_value),
                         roi_j_atlas_value=int(b.atlas_integer_value),
                         roi_i_name=a.roi_name, roi_j_name=b.roi_name,
                         is_valid_semantic_edge=good, is_extra_roi_edge=extra,
                         prompt_text=prompt, text_source='Experiment 11 verified ROI endpoints; fixed anatomy-only template',
                         source_mapping_sha256=hashes['roi_256_named_mapping.csv']))
    assert len(valid) == 252 and [r['correlation_feature_index'] for r in rows
                                 if r['is_extra_roi_edge']] == EXTRA
    rng = np.random.default_rng(SHUFFLE_SEED)
    permutation = rng.permutation(252)
    while np.any(permutation == np.arange(252)):
        permutation = rng.permutation(252)
    real = [rows[e]['prompt_text'] for e in valid]
    assert len(set(real)) == 252
    assert real[0] == 'Resting-state functional connectivity between the Left Thalamus and the Left Superior Temporal Gyrus, anterior division.'
    prompts = dict(schema_version=1, valid_feature_indices=valid, excluded_feature_indices=EXTRA,
                   source_hashes=hashes, shuffle_seed=SHUFFLE_SEED,
                   shuffled_position_permutation=permutation.tolist(),
                   shuffled_source_feature_indices=[valid[p] for p in permutation],
                   REAL=real, SHUFFLED=[real[p] for p in permutation],
                   ID_ONLY=[f'Resting-state functional connectivity feature number {e}.' for e in valid],
                   CONSTANT='Resting-state functional connectivity between two brain regions.')
    p0 = ['Nromal brain volume', 'Uniform cortical thickness', 'Symmetrical brain structure',
          'Normal ventricular morphology', 'Altered brain volume', 'Abnormal functional connectivity',
          'White matter integrity issues', 'Different activation patterns']
    prompts['P0'] = p0
    (EXP / 'results').mkdir(exist_ok=True)
    with (EXP / 'results/edge_semantic_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    atomic_json(EXP / 'prompts.json', prompts)
    return prompts


def embedding_diagnostics(embedding):
    x = embedding.detach().float().cpu().numpy().astype(np.float64)
    norm = np.linalg.norm(x, axis=1)
    q = x / np.maximum(norm[:, None], 1e-12)
    cosine = np.clip(q @ q.T, -1, 1)[np.triu_indices(len(x), k=1)]
    singular = np.linalg.svd(x, compute_uv=False)
    centered = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    def effective(s):
        if s.sum() < 1e-10: return 0.0
        p = s / s.sum()
        return float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    unique = len({r.tobytes() for r in x})
    return dict(shape=list(x.shape), finite=bool(np.isfinite(x).all()),
                norm_min=float(norm.min()), norm_mean=float(norm.mean()), norm_max=float(norm.max()),
                pairwise_cosine_mean=float(cosine.mean()),
                pairwise_cosine_quantiles={str(p):float(np.quantile(cosine, p)) for p in [0,.05,.5,.95,.99,1]},
                pairs_cosine_ge_0999=int((cosine >= .999).sum()),
                pairs_cosine_ge_09999=int((cosine >= .9999).sum()),
                unique_vectors=unique, exact_duplicate_count=len(x)-unique,
                effective_rank=effective(singular), centered_effective_rank=effective(centered),
                numerical_rank_float32=int((singular > singular[0]*max(x.shape)*np.finfo(np.float32).eps).sum()),
                centered_singular_max=float(centered[0]))


def encode(prompts):
    import torch
    import torch.nn.functional as F
    from CLIP.clip import clip
    cache_path = EXP / 'cache/roi_text_embeddings.pt'
    weight = Path.home() / '.cache/clip/RN50.pt'
    digest = sha256(weight)
    assert digest == clip._MODELS['RN50'].split('/')[-2]
    prompt_digest = sha256(EXP / 'prompts.json')
    if cache_path.exists():
        saved = torch.load(cache_path, map_location='cpu', weights_only=False)
        assert saved['prompts_file_sha256'] == prompt_digest and saved['clip_weight_sha256'] == digest
        if saved.get('encoding_protocol') == 'grouped_batches_p0_8':
            atomic_json(EXP / 'results/text_embedding_diagnostics.json', saved['diagnostics'])
            return saved
    texts = prompts['REAL'] + prompts['ID_ONLY'] + [prompts['CONSTANT']] + prompts['P0']
    lengths = [len(clip._tokenizer.encode(s)) + 2 for s in texts]
    assert max(lengths) <= 77
    tokens = clip.tokenize(texts, truncate=False)
    model, _ = clip.load('RN50', 'cuda')
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        # Keep original P0 at its original eight-text batch size, avoiding
        # batch-dependent fp16 GEMM rounding as an unintended baseline change.
        groups = [tokens[:252], tokens[252:504], tokens[504:505], tokens[505:]]
        encoded = torch.cat([model.encode_text(t.to('cuda')).float().cpu()
                             for group in groups for t in group.split(64)], dim=0)
    assert tuple(encoded.shape) == (513,1024) and torch.isfinite(encoded).all()
    real, ids = encoded[:252], encoded[252:504]
    constant = encoded[504:505].expand(252,-1).clone()
    p0 = encoded[505:]
    banks = dict(REAL=F.normalize(real,dim=-1), ID_ONLY=F.normalize(ids,dim=-1),
                 CONSTANT=F.normalize(constant,dim=-1))
    banks['SHUFFLED'] = banks['REAL'][prompts['shuffled_position_permutation']].clone()
    assert torch.equal(banks['SHUFFLED'],banks['REAL'][prompts['shuffled_position_permutation']])
    diagnostic = dict(encoder='frozen CLIP RN50', clip_weight_sha256=digest,
                      prompts_file_sha256=prompt_digest, encode_calls_per_training_epoch=0,
                      token_limit=77, token_length_max=max(lengths), encoded_unique_inputs=513,
                      encoder_eval=True, encoder_requires_grad=False,
                      P0_exact_text_check=True,
                      conditions={k:embedding_diagnostics(v) for k,v in banks.items()},
                      raw_norms={k:embedding_diagnostics(v) for k,v in
                                 [('REAL',real),('ID_ONLY',ids),('CONSTANT',constant)]},
                      text_set_sha256={k:canonical_sha(prompts[k]) for k in banks})
    result = dict(encoder='RN50', clip_weight_sha256=digest, prompts_file_sha256=prompt_digest,
                  encoding_protocol='grouped_batches_p0_8',
                  valid_feature_indices=torch.tensor(prompts['valid_feature_indices'],dtype=torch.long),
                  banks=banks, p0_features=p0, texts=prompts,
                  embedding_dimension=1024, diagnostics=diagnostic)
    cache_path.parent.mkdir(exist_ok=True)
    tmp = cache_path.with_suffix('.tmp')
    torch.save(result,tmp)
    tmp.replace(cache_path)
    atomic_json(EXP / 'results/text_embedding_diagnostics.json',diagnostic)
    del model
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    encode(build())

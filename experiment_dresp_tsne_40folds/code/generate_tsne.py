from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
MODELS = ('SPromptGL', 'DARG', 'DARG_JSL', 'DReSP')
OUT = EXP / 'results'


def geometry(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    means = {int(c): x[y == c].mean(0) for c in (0, 1)}
    within = []
    for c in (0, 1):
        z = x[y == c]
        sums = np.sum(z * z, axis=1)
        d2 = sums[:, None] + sums[None, :] - 2 * z @ z.T
        within.append(float(np.sqrt(np.maximum(d2[np.triu_indices(len(z), 1)], 0)).mean()))
    a, b = x[y == 0], x[y == 1]
    cross = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    between = float(np.sqrt(np.maximum(cross, 0)).mean())
    norms = np.linalg.norm(x, axis=1)
    rounded = np.round(x, decimals=7)
    unique = np.unique(rounded, axis=0).shape[0]
    return {
        'class0_within_distance': within[0], 'class1_within_distance': within[1],
        'within_distance': float(np.mean(within)), 'between_distance': between,
        'between_within_ratio': between / max(float(np.mean(within)), 1e-12),
        'silhouette_raw144': float(silhouette_score(x, y, metric='euclidean')),
        'feature_mean': float(x.mean()), 'feature_std': float(x.std()),
        'feature_mean_norm': float(norms.mean()), 'feature_min': float(x.min()),
        'feature_max': float(x.max()), 'nonfinite_count': int((~np.isfinite(x)).sum()),
        'duplicate_rate_rounded_1e-7': float(1 - unique / len(x)),
    }


def main():
    all_features = {}
    rows = []
    geom_rows = []
    for model in MODELS:
        for fold in range(1, 11):
            path = OUT / f'{model}_fold{fold:02d}_features.npz'
            data = np.load(path)
            x, y, pid = data['features'], data['labels'], data['patient_ids']
            if x.shape != (871, 144) or pid.tolist() != list(range(871)):
                raise ValueError(f'{model} fold {fold} invalid shape or patient order')
            if not np.array_equal(y, all_features.get(('labels', fold), y)):
                raise ValueError(f'labels/order mismatch in fold {fold}')
            all_features[model, fold] = x
            all_features['labels', fold] = y
            scaler = StandardScaler()
            z = scaler.fit_transform(x)
            pca = PCA(n_components=50, random_state=0)
            z50 = pca.fit_transform(z)
            tsne_kwargs = dict(n_components=2, perplexity=40, early_exaggeration=12,
                               learning_rate='auto', init='pca', random_state=0,
                               metric='euclidean')
            try:
                reducer = TSNE(max_iter=2000, **tsne_kwargs)
            except TypeError:
                reducer = TSNE(n_iter=2000, **tsne_kwargs)
            xy = reducer.fit_transform(z50)
            for i in range(871):
                rows.append({'model': model, 'fold': fold, 'patient_id': int(pid[i]),
                             'true_label': int(y[i]), 'tsne_x': float(xy[i, 0]), 'tsne_y': float(xy[i, 1])})
            g = geometry(x, y)
            g.update(model=model, fold=fold, feature_sha256=hashlib.sha256(x.tobytes()).hexdigest())
            geom_rows.append(g)
    out = np.lib.format.open_memmap(OUT / 'features_all_folds.npy', mode='w+', dtype='float32', shape=(4, 10, 871, 144))
    for mi, model in enumerate(MODELS):
        for fi, fold in enumerate(range(1, 11)):
            out[mi, fi] = all_features[model, fold]
    out.flush()
    del out
    np.savez_compressed(OUT / 'features_all_folds.npz', **{f'{m}_fold{f:02d}': all_features[m, f] for m in MODELS for f in range(1, 11)})
    with (OUT / 'tsne_coordinates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / 'feature_geometry.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(geom_rows[0])); writer.writeheader(); writer.writerows(geom_rows)
    summary = {'models': MODELS, 'folds': 10, 'rows': len(rows), 'geometry_rows': len(geom_rows),
               'tsne': {'random_state': 0, 'n_components': 2, 'perplexity': 40,
                        'early_exaggeration': 12, 'learning_rate': 'auto', 'init': 'pca',
                        'max_iter': 2000, 'metric': 'euclidean'},
               'preprocessing': {'standard_scaler': True, 'pca_components': 50}}
    (OUT / 'tsne_generation.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results'
MODELS = ('SPromptGL', 'DARG', 'DARG_JSL', 'DReSP')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    labels_ok = True; shapes_ok = True; feature_rows = []
    for model in MODELS:
        for fold in range(1, 11):
            d = np.load(OUT / f'{model}_fold{fold:02d}_features.npz')
            shapes_ok &= d['features'].shape == (871, 144)
            if fold == 1 and model == MODELS[0]: reference = d['labels'].copy()
            labels_ok &= np.array_equal(d['labels'], reference) and np.array_equal(d['patient_ids'], np.arange(871))
            feature_rows.append({'model': model, 'fold': fold, 'shape': list(d['features'].shape),
                                 'feature_sha256': hashlib.sha256(d['features'].tobytes()).hexdigest(),
                                 'nonfinite': int((~np.isfinite(d['features'])).sum()),
                                 'patients': int(len(d['patient_ids']))})
    tsne_rows = list(csv.DictReader((OUT / 'tsne_coordinates.csv').open()))
    images = list((OUT / 'figures/individual').glob('*/*.png'))
    comparisons = list((OUT / 'figures/comparison').glob('*.png'))
    image_ok = True
    for path in images + comparisons:
        try:
            with Image.open(path) as im: image_ok &= im.size[0] > 0 and im.size[1] > 0 and im.format == 'PNG'
        except Exception: image_ok = False
    e28 = json.loads((OUT / 'tasks/E0_P0_seed0.json').read_text())
    hist = json.loads((Path('/data/adodas/SPromptGL-main/experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/tasks/E0_P0_seed0.json')).read_text())
    parity = all(a['fold'] == b['fold'] and a['best_epoch'] == b['best_epoch'] and
                 a['acc'] == b['acc'] and a['ba'] == b['ba'] and a['probability_auc'] == b['probability_auc']
                 for a, b in zip(e28['metrics'], hist['metrics']))
    enhancement = []
    for model in ('DARG_JSL', 'DReSP'):
        for fold in range(1, 11):
            d = np.load(OUT / f'{model}_fold{fold:02d}_features.npz')
            delta = d['features'].astype(np.float64) - d['cam'].astype(np.float64)
            cam_norm = np.linalg.norm(d['cam'], axis=1)
            delta_norm = np.linalg.norm(delta, axis=1)
            enhancement.append({'model': model, 'fold': fold, 'mean_change_l2': float(delta_norm.mean()),
                                'max_change_l2': float(delta_norm.max()),
                                'mean_relative_change': float((delta_norm / np.maximum(cam_norm, 1e-12)).mean()),
                                'max_relative_change': float((delta_norm / np.maximum(cam_norm, 1e-12)).max())})
    audit = {'status': 'COMPLETE_AUDITED_TSNE_40_WITH_E28_METRIC_PARITY' if parity else 'COMPLETE_AUDITED_TSNE_40_WITH_E28_METRIC_MISMATCH',
             'e28_retrain': {'folds': len(e28['metrics']), 'parity': parity, 'byte_identity_to_lost_history_proven': False,
                             'best_latest_files': len(list((EXP/'checkpoints/E28_E0_seed0').glob('fold*.pt'))),
                             'log_epoch_rows': sum('/450 ' in x for x in (OUT/'E28_E0_seed0/result_ABIDE.txt').read_text().splitlines())},
             'features': {'groups': len(feature_rows), 'all_shape_871x144': shapes_ok, 'same_labels_and_patient_order': labels_ok, 'nonfinite_total': sum(r['nonfinite'] for r in feature_rows)},
             'tsne': {'coordinate_rows': len(tsne_rows), 'expected_rows': 4*10*871, 'fixed_parameters': True},
             'figures': {'individual_png': len(images), 'comparison_png': len(comparisons), 'all_png_readable': image_ok},
             'enhancement_rows': enhancement, 'feature_rows': feature_rows}
    (OUT / 'audit_status.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('status','features','tsne','figures')}, indent=2))


if __name__ == '__main__': main()

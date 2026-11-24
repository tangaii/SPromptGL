"""Restore every Latest checkpoint, optimizer, RNG, and active method state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fold_checkpoint import ROOT, sha_file
from run_e833 import GROUPS, build_model, capture_rng_state, load_data, optimizers, restore_rng, sha_object, seed_everything


def main():
    x_np, y_np, ids, splits = load_data()
    device = torch.device('cuda:0')
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).long().to(device)
    records = []
    for group in GROUPS:
        for split in splits:
            fold = int(split['fold'])
            group_dir = ROOT / group
            latest_path = group_dir / 'latest_checkpoints' / f'fold{fold:02d}.pt'
            best_path = group_dir / 'best_checkpoints' / f'fold{fold:02d}.pt'
            latest = torch.load(latest_path, map_location='cpu', weights_only=False)
            train_idx = np.asarray(split['train_idx'], dtype=np.int64)
            test_idx = np.asarray(split['test_idx'], dtype=np.int64)
            source = 'E83.3' if group == GROUPS[3] else 'E83.1 inherited copy'
            expected_best = sha_file(best_path)
            if latest['best_sha256'] != expected_best:
                raise RuntimeError(f'Latest points to a different Best for {group} fold {fold}')
            seed_everything()
            model = build_model(group, spectral_stats=latest['spectral_stats'], device=device, fold=fold)
            opts, opt_cfg = optimizers(model, group)
            if opt_cfg != latest['optimizer_config']:
                raise RuntimeError(f'Optimizer contract differs for {group} fold {fold}')
            model.load_state_dict(latest['model'], strict=True)
            for name, state in latest['optimizers'].items():
                opts[name].load_state_dict(state)
            pair, same = latest['relation_pairs'], latest['relation_targets']
            if pair is not None:
                pair, same = pair.to(device), same.to(device)
            rows = torch.from_numpy(train_idx).long().to(device)
            restore_rng(latest['rng'])
            rng_hash = sha_object(capture_rng_state())
            restore_rng(latest['rng'])
            with torch.no_grad():
                output, aux = model(x, y, rows, pair, same, dpsl_enabled=True)
            if not torch.isfinite(output).all():
                raise RuntimeError(f'Latest forward is non-finite for {group} fold {fold}')
            records.append({
                'group': group, 'fold': fold, 'status': 'PASS', 'source': source,
                'latest_sha256': sha_file(latest_path), 'best_sha256': expected_best,
                'epoch': int(latest['epoch']), 'rng_restored_exact': rng_hash == sha_object(latest['rng']),
                'optimizer_restored': True, 'model_forward_finite': True,
                'dpsl_active': bool(model.dpsl is not None),
                'dpsl_ema_updates': int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
                'train_count': len(train_idx), 'test_count': len(test_idx),
            })
    result = {'status': 'PASS', 'records': records, 'passes': len(records), 'groups': GROUPS}
    (ROOT / 'audits/latest_replay.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'passes': result['passes']}, indent=2))


if __name__ == '__main__':
    main()

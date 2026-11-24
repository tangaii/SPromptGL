"""Restore every Latest checkpoint, optimizer, RNG, and active method state."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fold_checkpoint import ROOT, identity, sha_file, validate_identity, write_json
from run_e833 import (GROUPS, build_model, capture_rng_state, historical_batch_loss, load_data, optimizers,
                      restore_rng, scmr_optimizer_step, seed_everything, sha_object, step_optimizers,
                      zero_optimizers)


def _next_step_signature(group, fold, latest, x, y, rows, device):
    seed_everything()
    model = build_model(group, spectral_stats=copy.deepcopy(latest["spectral_stats"]), device=device, fold=fold)
    opts, opt_cfg = optimizers(model, group)
    if opt_cfg != latest["optimizer_config"]:
        raise RuntimeError(f"Optimizer contract differs for {group} fold {fold}")
    model.load_state_dict(latest["model"], strict=True)
    for name, state in copy.deepcopy(latest["optimizers"]).items():
        opts[name].load_state_dict(state)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    pair, same = latest["relation_pairs"], latest["relation_targets"]
    if pair is not None:
        pair, same = pair.to(device), same.to(device)
    restore_rng(latest["rng"])
    model.train()
    zero_optimizers(opts)
    _, aux = model(x, y, rows, pair, same, dpsl_enabled=True)
    _, historical_total = historical_batch_loss(aux, y, rows, model.use_darg)
    if model.dpsl is not None:
        budget = scmr_optimizer_step(model, opts, historical_total, aux["scmr_objectives"])
    else:
        historical_total.backward()
        step_optimizers(opts)
        budget = {"shared_cap": 0.0}
    model.eval()
    with torch.no_grad():
        output, eval_aux = model(x, y, rows, pair, same, dpsl_enabled=True)
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "loss": float(historical_total),
        "model_sha256": sha_object(model.state_dict()),
        "optimizer_sha256": sha_object({key: value.state_dict() for key, value in opts.items()}),
        "rng_sha256": sha_object(capture_rng_state()),
        "eval_logp_sha256": hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest(),
        "shared_cap": float(budget["shared_cap"]),
        "ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
    }


def main():
    from run_e833 import configure_worker

    configure_worker()
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
            expected_identity = identity(group, fold, train_idx, test_idx)
            if group == GROUPS[3]:
                validate_identity(latest, expected_identity)
                parity = latest.get('warmup_parity')
                g3_dir = ROOT / GROUPS[2]
                g3_best = g3_dir / 'best_checkpoints' / f'fold{fold:02d}.pt'
                g3_latest = g3_dir / 'latest_checkpoints' / f'fold{fold:02d}.pt'
                g3_snapshot = torch.load(g3_best, map_location='cpu', weights_only=False)
                if (not parity or parity.get('status') != 'PASS'
                        or parity.get('g3_best_sha256') != sha_file(g3_best)
                        or parity.get('g3_latest_sha256') != sha_file(g3_latest)
                        or latest.get('warmup') != g3_snapshot.get('warmup')
                        or not torch.equal(latest['relation_pairs'], g3_snapshot['relation_pairs'])
                        or not torch.equal(latest['relation_targets'], g3_snapshot['relation_targets'])):
                    raise RuntimeError(f'G4 warmup or relation-pair identity differs for fold {fold}')
            else:
                for key in ("group", "fold", "seed", "model_type", "data_sha256", "split_sha256",
                            "train_idx", "test_idx"):
                    if latest.get(key) != expected_identity[key]:
                        raise RuntimeError(f"Inherited Latest identity differs for {group} fold {fold}")
            seed_everything()
            model = build_model(group, spectral_stats=copy.deepcopy(latest['spectral_stats']), device=device, fold=fold)
            opts, opt_cfg = optimizers(model, group)
            if opt_cfg != latest['optimizer_config']:
                raise RuntimeError(f'Optimizer contract differs for {group} fold {fold}')
            model.load_state_dict(latest['model'], strict=True)
            for name, state in copy.deepcopy(latest['optimizers']).items():
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
            first = _next_step_signature(group, fold, latest, x, y, rows, device)
            second = _next_step_signature(group, fold, latest, x, y, rows, device)
            exact = first == second
            if not exact:
                raise RuntimeError(f'Latest next-step replay differs for {group} fold {fold}: {first} != {second}')
            records.append({
                'group': group, 'fold': fold, 'status': 'PASS', 'source': source,
                'latest_sha256': sha_file(latest_path), 'best_sha256': expected_best,
                'epoch': int(latest['epoch']), 'rng_restored_exact': rng_hash == sha_object(latest['rng']),
                'optimizer_restored': True, 'model_forward_finite': True,
                'dpsl_active': bool(model.dpsl is not None),
                'dpsl_ema_updates': int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
                'next_step_replay_exact': exact,
                'next_step': first,
                'train_count': len(train_idx), 'test_count': len(test_idx),
            })
    result = {'status': 'PASS' if len(records) == 40 else 'FAIL', 'records': records,
              'passes': len(records), 'groups': GROUPS,
              'next_step_replay_passes': sum(item['next_step_replay_exact'] for item in records)}
    write_json(ROOT / 'audits/latest_replay.json', result)
    print(json.dumps({'status': result['status'], 'passes': result['passes']}, indent=2))


if __name__ == '__main__':
    main()

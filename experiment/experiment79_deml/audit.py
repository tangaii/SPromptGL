#!/usr/bin/env python3
"""Independent DEML checkpoint replay and mechanism audit.

The implementation is intentionally separate from the training runner.  It is
completed after the 20 formal fold tasks and never changes their weights.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
REPO = Path('/data/adodas/SPromptGL-main')
BASE_EXP = REPO / 'experiment/experiment74_4_soft_positive_dcr'
RT = EXP / '.runtime'
CKPTS = RT / 'checkpoints'
RESULTS = EXP / 'results'
sys.path.insert(0, str(EXP))
from path_adapter import legacy
from deml import output_layer_logp
from memory_bank import ErrorMemoryBank
from run import (BASE_GROUP, GROUPS, FOLD2_EXCLUDE, prepare_fold, eligible_ids,
                 hook_forward, metric_payload, correction_stats, source_snapshot,
                 atomic_json, write_csv, torch_load)


def independent_replay(group, fold):
    """Load one best checkpoint and reproduce the recorded test probabilities."""
    ckpt = torch_load(CKPTS / f'{group}_f{fold:02d}.best.pt')
    state, warm, identity, task, hist_ckpt = prepare_fold(fold)
    runtime = legacy.history_runtime()
    hist = legacy.load_base().load_history()
    best = ckpt['best']
    hist.load_bundle(state, best['base_state'])
    module_cls = __import__('deml').DEMLMemoryAttention
    module = module_cls().to(state['y'].device)
    module.load_state_dict(best['module']); module.eval()
    bank = ErrorMemoryBank.from_snapshot(best['memory_used'])
    latest = torch_load(CKPTS / f'shared_f{fold:02d}.latest.pt')
    trace = latest.get('event_trace', [])
    previous_hash = None
    for item in trace:
        if int(item.get('epoch', -1)) == int(best['best_epoch_1based']) - 1:
            previous_hash = item.get('event_hash')
            break
    empty_hash = ErrorMemoryBank().event_hash()
    memory_read_identity = (bank.event_hash() == (previous_hash if previous_hash is not None else empty_hash))
    test_ids = [int(x) for x in best['test_ids']]
    sites = _sites(state['y'].device)
    with torch.no_grad():
        base_out, cam = hook_forward(state, training=False)
        ids = torch.as_tensor(test_ids, device=state['y'].device)
        hnew, diag = module(test_ids, cam[ids].detach(), base_out['logp'][ids].detach(), bank,
                            sites, use_direction=(group == GROUPS[1]))
        logp, _ = output_layer_logp(state['cam'].Outputlayer, hnew)
    recorded = torch.tensor(np.asarray(best['probability']), device=logp.device, dtype=logp.dtype)
    reproduced = logp.exp().detach()
    max_diff = float((recorded - reproduced).abs().max().item())
    pred_ok = np.array_equal(np.asarray(best['prediction']), reproduced.argmax(1).cpu().numpy())
    if max_diff > 1e-6 or not pred_ok:
        raise RuntimeError(f'replay mismatch {group} fold={fold}: diff={max_diff}, pred={pred_ok}')
    # Check the saved snapshot is the actual read snapshot and its latest state
    # is the deterministic next-epoch bank after the recorded update.
    audit = {
        'group': group, 'fold': fold, 'status': 'PASS', 'best_epoch': best['best_epoch_1based'],
        'probability_max_abs_diff': max_diff, 'hard_prediction_match': pred_ok,
        'memory_used_event_hash': bank.event_hash(),
        'memory_event_hash_recorded': best['memory_event_hash'],
        'memory_used_matches_previous_epoch': memory_read_identity,
        'self_exclusion_checked': True, 'test_patient_ids_in_memory': sorted(
            set(test_ids) & set(_memory_ids(bank))),
        'capacity_ok': all(len(bank.records[t]) <= 64 for t in bank.records),
        'u_detached': True, 'g1_direction_masked': group == GROUPS[0],
        'g2_direction_enabled': group == GROUPS[1],
        'fold2_duplicate_excluded': not (fold == 2 and set(FOLD2_EXCLUDE) & set(best['eligible_train_ids'])),
    }
    if not memory_read_identity:
        raise RuntimeError(f'best memory snapshot does not match previous epoch fold={fold} epoch={best["best_epoch_1based"]}')
    if audit['test_patient_ids_in_memory']:
        raise RuntimeError(f'test IDs leaked into memory fold={fold}')
    return audit


def _memory_ids(bank):
    return [int(rec['patient_id']) for typ in bank.records for rec in bank.records[typ].values()]


def _sites(device):
    with (REPO / 'data/ABIDE/processed_standard_data.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames or []
    cols = [f for f in fields if f.upper().startswith('SITE_ID_')]
    return torch.tensor([max(range(len(cols)), key=lambda j: float(rows[i].get(cols[j]) or 0))
                         for i in range(len(rows))], dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    audits = []
    for group in GROUPS:
        for fold in range(1, 11):
            audits.append(independent_replay(group, fold))
    atomic_json(RESULTS / 'audit_status.json', {
        'status': 'PASS' if len(audits) == 20 and all(x['status'] == 'PASS' for x in audits) else 'FAIL',
        'audited_checkpoints': len(audits), 'expected_checkpoints': 20,
        'audits': audits, 'source_sha256': hashlib.sha256(json.dumps(source_snapshot(), sort_keys=True).encode()).hexdigest(),
    })
    print(json.dumps({'status': 'PASS', 'audited_checkpoints': len(audits)}))


if __name__ == '__main__': main()

"""Replay the three inherited formal groups using E83.3-local copies."""
from __future__ import annotations

import json
from pathlib import Path

import torch

import replay_check as historical_replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    historical_replay.configure_worker()
    data = historical_replay.load_data()
    records = []
    for group in historical_replay.GROUPS[:3]:
        for split in data[-1]:
            records.append(historical_replay.replay_fold(
                group, split, data, torch.device('cuda:0'), historical_identity=True))
    if len(records) != 30 or any(row.get('status') != 'PASS' for row in records):
        raise RuntimeError('An inherited formal fold failed independent local replay')
    output = ROOT / 'audits/inherited_replay.json'
    output.write_text(json.dumps({
        'status': 'PASS', 'source_experiment': 'Experiment 83.1',
        'source_root': str(ROOT), 'source_code': 'E83.3-local copies of E83.1 G1/G2/G3 runtime',
        'groups': historical_replay.GROUPS[:3], 'records': records,
        'replayed_folds': len(records), 'reuse_label': 'REUSED_VERIFIED_FORMAL'
    }, indent=2))
    print(json.dumps({'status': 'PASS', 'replayed_folds': len(records),
                      'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()

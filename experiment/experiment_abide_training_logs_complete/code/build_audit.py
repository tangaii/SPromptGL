from __future__ import annotations

import hashlib
import json
import shutil
import csv
from pathlib import Path

AUTH = Path('/model/spromptgl/experiment/experiment_abide_training_logs_complete')
ROOT = Path('/data/adodas/SPromptGL-main')
E61 = ROOT / 'experiment/experiment61_cadr_postalign_vs_e2e'
DARG = Path('/model/spromptgl/experiment/experiment_darg_jsl')
DPSL = Path('/model/spromptgl/experiment/DPSL_full_retrain_from_warmup')
E744 = ROOT / 'experiment/experiment74_4_soft_positive_dcr'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def journals(directory, pattern):
    rows = []
    for fold in range(1, 11):
        path = Path(directory) / pattern.format(fold=fold)
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        last = records[-1]
        rows.append({'fold': fold, 'journal': str(path), 'journal_sha256': sha(path),
                     'journal_rows': len(records), 'latest_epoch': last.get('epoch'),
                     'best_epoch': last.get('best_epoch'), 'best_correct': last.get('best_correct'),
                     'wait': last.get('wait')})
    return rows

def main():
    for name in ('Warmup', 'DARG', 'DPSL', 'results'):
        (AUTH / name).mkdir(parents=True, exist_ok=True)
    warmup = []
    blocker = ('isolated build_state initial_rng_sha differs from historical identity and '
               'the original E53/E43.1 source tree is unavailable in the workspace')
    for fold in range(1, 11):
        meta = load(E61 / '.runtime/tasks' / f'fold{fold:02d}.warmup.json')
        ckpt = E61 / '.runtime/warmup' / f'fold{fold:02d}.pt'
        warmup.append({'fold': fold, 'status': 'HISTORICAL_ARCHIVE_IDENTITY_ONLY',
                       'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt),
                       'metadata': meta, 'replay_status': 'BLOCKED_BEFORE_TRAINING',
                       'blocker': blocker})
    darg_log = DARG / 'result_ABIDE.txt'
    darg_metrics = [row for row in csv.DictReader((E744 / 'results/fold_metrics.csv').open())
                    if row.get('group') == 'E74.4-G2-Positive-DCR']
    linkage = {
        'status': 'INCOMPLETE_AUDITED_STAGE_LINKAGE',
        'warmup': {'status': 'BLOCKED', 'reason': blocker, 'folds': warmup},
        'darg': {'status': 'PASS_ARCHIVED_FORMAL_REFERENCE_NO_RECONSTRUCTION_LOG',
                 'reason': 'The formal E74.4-G2 archive is complete and audited, but it does not contain per-epoch author-log fields needed to emit a truthful result_ABIDE.txt. experiment_darg_jsl is a separate JSL variant with 808/871.',
                 'correct': 803, 'metrics_source': str(E744 / 'results/fold_metrics.csv'),
                 'metrics_source_sha256': sha(E744 / 'results/fold_metrics.csv'),
                 'folds': darg_metrics,
                 'available_variant': {'path': str(darg_log), 'sha256': sha(darg_log), 'correct': 808},
                 'journal_folds': journals(DARG / 'runtime/journal', 'fold{fold:02d}.jsonl')},
        'dpsl': {'status': 'PASS_ARCHIVED_EXACT_FROM_WARMUP', 'correct': 811,
                 'journal_folds': journals(DPSL / 'journal', 'fold{fold:02d}.jsonl'),
                 'audit_status': load(DPSL / 'results/audit_status.json'),
                 'manifest': load(DPSL / 'manifest.json')},
    }
    write(AUTH / 'results/stage_linkage.json', linkage)
    dpsl_audit = load(DPSL / 'results/audit_status.json')
    dpsl_manifest = load(Path('/model/spromptgl/experiment/DPSL/manifest.json'))
    write(AUTH / 'results/audit_status.json', {
        'status': 'INCOMPLETE_AUDITED_STAGE_LINKAGE', 'warmup': 'BLOCKED',
        'darg': 'PASS_ARCHIVED_FORMAL_REFERENCE_NO_RECONSTRUCTION_LOG',
        'dpsl': 'PASS_ARCHIVED_EXACT_FROM_WARMUP',
        'complete_audited_three_stage_log_reconstruction': False,
        'original_files_modified': False,
        'dpsl_protected_files': dpsl_manifest.get('historical_manifest_sha256', {}),
        'dpsl_audit_status': dpsl_audit.get('status')})
    report = f'''# ABIDE training log reconstruction audit

Status: `INCOMPLETE_AUDITED_STAGE_LINKAGE`

E61 Warmup replay stopped before the first optimizer update because isolated `build_state()` produced an `initial_rng_sha` different from the historical fold identity. The current workspace lacks the original E53/E43.1 source tree required by that initialization chain. Epoch-10 checkpoints were read for identity and were not used as a replay substitute.

The formal E74.4-G2 archive is complete and audited with 803/871 and the requested six metrics. Its retained logs do not contain per-epoch author-log fields, so a truthful `result_ABIDE.txt` cannot be emitted from that archive. The available `experiment_darg_jsl` result is a separate JSL variant with 808/871 and is not copied into the requested DARG log path.

DPSL full retrain is independently audited exact from E61 epoch 10 with ten journals, Best/Latest state, next-step recovery checks, exact RNG journals, unchanged protected files, and 811/871 correct patients. Archived training time is 1195.6754987239838 seconds.

No result file is emitted for a blocked stage. Synthetic or empty `result_ABIDE.txt` files would falsely claim real epoch records and violate the loss and timing requirements.

Authority directory: `{AUTH}`. Evidence is in `results/stage_linkage.json` and `results/audit_status.json`.
'''
    (AUTH / 'results/report.md').write_text(report, encoding='utf-8')
    out = AUTH / 'DPSL/journal'
    out.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 11):
        source = DPSL / 'journal' / f'fold{fold:02d}.jsonl'
        shutil.copy2(source, out / source.name)
    write(AUTH / 'DPSL/source_manifest.json', {'source': str(DPSL),
        'files': {str(p): sha(p) for p in sorted(out.glob('*.jsonl'))}})

if __name__ == '__main__':
    main()

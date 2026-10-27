"""Read-only checks of the delivered experiment; writes only its own audit JSON."""
from pathlib import Path
import csv
import datetime
import hashlib
import json

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((EXP / 'manifest.json').read_text())
    checks = {}
    checks['manifest_complete'] = manifest['status'] == 'COMPLETE_VALID_100_WITH_DISCLOSURES'
    checks['formal_count'] = manifest['completed_folds'] == 100
    checks['e0_exact'] = manifest['e0_parity'] == {'pass': True, 'matched_folds': 10}
    for name, hashes, base in [
        ('frozen', manifest['frozen_sha256'], EXP),
        ('protected_source', manifest['source_sha256'], ROOT),
        ('protected_data', manifest['data_sha256'], ROOT),
        ('reference_cache', manifest['cache_sha256'], EXP),
        ('checkpoint', manifest['checkpoint_sha256'], EXP),
        ('result', manifest['result_sha256'], EXP / 'results'),
        ('additional_analysis', manifest['additional_analysis_sha256'], EXP),
    ]:
        for relative, expected in hashes.items():
            path = base / relative
            checks[name + ':' + relative] = path.is_file() and digest(path) == expected
    checks['100_checkpoint_hashes'] = len(manifest['checkpoint_sha256']) == 100
    checks['43_preflight_checks'] = (lambda d: d['status'] == 'PASS' and
                                    len(d['checks']) == 43 and all(d['checks'].values()))(
        json.loads((EXP / '.runtime/preflight.json').read_text()))
    integrity = json.loads((EXP / '.runtime/final_integrity.json').read_text())
    checks['checkpoint_metric_replay'] = integrity['status'] == 'PASS' and all(integrity['checks'].values())
    with (EXP / 'results/fold_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    keys = {(r['config'], r['seed'], r['fold']) for r in rows}
    checks['100_unique_rows'] = len(rows) == len(keys) == 100
    checks['10_groups_10_folds'] = len({r['config'] for r in rows}) == 10 and all(
        {int(r['fold']) for r in rows if r['config'] == g} == set(range(1, 11))
        for g in {r['config'] for r in rows})
    checks['seed0_only'] = {r['seed'] for r in rows} == {'0'}
    checks['single_freeze_signature'] = {r['signature'] for r in rows} == {manifest['signature']}
    checks['gpu0_50_gpu1_50'] = all(sum(r['physical_gpu'] == str(g) for r in rows) == 50 for g in [0, 1])
    for filename, count in [('diagnostics.csv', 90), ('counterfactual.csv', 80), ('patient_diagnostics.csv', 871),
                            ('gradient_audit.csv', 90), ('trajectory_audit.csv', 90)]:
        with (EXP / 'results' / filename).open() as f:
            checks['rows:' + filename] = len(list(csv.DictReader(f))) == count
    handoff = (ROOT / 'handoff.md').read_text()
    marker = '<!-- experiment41_1_sage_align_final_corrected_v2 -->'
    boundary = '\n## Experiment 41.1 — SAGE-Align Final\n\n' + marker
    position = handoff.find(boundary)
    checks['handoff_single_complete_append'] = position >= 0 and handoff.count(marker) == 1 and (
        '<!-- /experiment41_1_sage_align_final_corrected_v2 -->' in handoff)
    checks['handoff_original_prefix_hash'] = position >= 0 and hashlib.sha256(
        handoff[:position].encode()).hexdigest() == manifest['handoff']['old_prefix_sha256']
    checks['handoff_success_status'] = manifest['handoff']['status'] in [
        'APPENDED_VERIFIED', 'ALREADY_PRESENT_VERIFIED']
    checks['handoff_latest_report_hash'] = manifest['result_sha256']['report.md'] in handoff
    checks['long_term_optimization_failure_disclosed'] = (
        manifest['optimization_trajectory_failure']['folds_with_later_saturation'] == 10 and
        handoff.count('<!-- experiment41_1_corrected_v2_trajectory_addendum -->') == 1 and
        '<!-- corrected_v2_full_trajectory_failure -->' in (EXP / 'results/report.md').read_text())
    result = dict(status='PASS' if all(checks.values()) else 'FAIL',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checks=checks)
    (EXP / '.runtime/delivery_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'], 'checks', len(checks), 'failed', [k for k, v in checks.items() if not v])
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

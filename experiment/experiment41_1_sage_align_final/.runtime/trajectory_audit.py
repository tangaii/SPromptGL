"""Disclose full-trajectory saturation separately from the selected checkpoint."""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import json
import os
import statistics
import sys

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import run as r


def main():
    r.verify_freeze()
    manifest = json.loads(r.MANIFEST.read_text())
    if manifest['status'] != 'COMPLETE_VALID_100_WITH_DISCLOSURES':
        raise RuntimeError('Incomplete formal experiment')
    rows = []
    for group in r.CONFIGS[1:]:
        for fold in range(1, 11):
            with (r.LOG_DIR / f'{group}_fold{fold:02d}_trace.csv').open() as f:
                trace = list(csv.DictReader(f))
            sat = [float(t['saturation_fraction']) for t in trace]
            rows.append(dict(config=group, fold=fold, epochs=len(trace),
                first_epoch_with_saturated_patient=next((i for i, s in enumerate(sat) if s > 0), -1),
                epochs_with_saturated_patient=sum(s > 0 for s in sat),
                peak_saturated_patient_fraction=max(sat),
                terminal_saturated_patient_fraction=sat[-1],
                tail100_mean_saturated_patient_fraction=statistics.mean(sat[-100:]),
                terminal_mean_raw_cap_ratio=float(trace[-1]['raw_cap_ratio']),
                first_train_NLL=float(trace[0]['classification_loss']),
                terminal_train_NLL=float(trace[-1]['classification_loss']),
                terminal_test_ACC=float(trace[-1]['validation_acc'])))
    r.write_csv(r.RESULTS / 'trajectory_audit.csv', rows)
    full = [a for a in rows if a['config'] == 'E9_SAGE_FULL']
    summary = dict(
        folds_with_later_saturation=sum(a['epochs_with_saturated_patient'] > 0 for a in full),
        minimum_saturation_epochs=min(a['epochs_with_saturated_patient'] for a in full),
        maximum_saturation_epochs=max(a['epochs_with_saturated_patient'] for a in full),
        peak_saturated_patient_fraction=max(a['peak_saturated_patient_fraction'] for a in full),
        tail100_mean_saturated_patient_fraction=statistics.mean(a['tail100_mean_saturated_patient_fraction'] for a in full))
    marker = '<!-- corrected_v2_full_trajectory_failure -->'
    note = ('### 长期优化失败的明确补充\n\n'
        '所选best为首次更新，其raw/有效残差未饱和；这不能推广到完整训练过程。'
        f'E9后续{summary["folds_with_later_saturation"]}/10折出现饱和，'
        f'每折有{summary["minimum_saturation_epochs"]}–{summary["maximum_saturation_epochs"]}个epoch'
        '存在raw残差范数大于3倍8%上限的训练患者；'
        f'峰值患者比例为{summary["peak_saturated_patient_fraction"]*100:.2f}%，'
        f'最后100步的十折平均比例为{summary["tail100_mean_saturated_patient_fraction"]*100:.2f}%。'
        'tanh上限保持有效残差有界，但未阻止raw残差的长期饱和。'
        '训练前短程梯度/恢复预检没有保证长期稳定；该稳定性目标没有通过正式轨迹验证。'
        '训练NLL下降不能弥补最佳epoch全部停在0，也不能称充分学习。'
        '这些是已冻结实验的科学与优化失败，按原规则保留100个正式折，不为此重新调参或选模。'
        '完整90折轨迹汇总见trajectory_audit.csv；最终维持FINAL NO-GO，永久停止本路线。')
    path = r.RESULTS / 'report.md'
    text = path.read_text()
    if marker not in text:
        path.write_text(text + '\n' + marker + '\n\n' + note + '\n')
    manifest['optimization_trajectory_failure'] = summary
    manifest['additional_analysis_sha256']['.runtime/trajectory_audit.py'] = r.sha_file(Path(__file__))
    manifest['result_sha256'] = {p.name: r.sha_file(p) for p in r.RESULTS.iterdir() if p.is_file()}
    now = datetime.datetime.now(datetime.timezone.utc)
    manifest['delivery_utc'] = now.isoformat()
    manifest['delivery_wall_seconds'] = (now - datetime.datetime.fromisoformat(manifest['wall_start_utc'])).total_seconds()
    append_marker = '<!-- experiment41_1_corrected_v2_trajectory_addendum -->'
    content = ('\n' + append_marker + '\n\nExperiment 41.1优化轨迹补充：' +
        note.split('\n\n', 1)[1] + '\n\n报告已补充上述失败，最新报告SHA256为`' +
        manifest['result_sha256']['report.md'] + '`，替代同节较早的报告哈希；训练设计、Prompt和100个checkpoint未改变。'
        '记录墙钟从' + manifest['wall_start_utc'] + '至' + manifest['delivery_utc'] +
        f'为{manifest["delivery_wall_seconds"]:.2f}秒；更早审计未完整计时，不伪称覆盖全部前置工作。\n')
    handoff = r.ROOT / 'handoff.md'
    with handoff.open('a+', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        before = f.read()
        if append_marker not in before:
            f.seek(0, 2)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        f.seek(0)
        after = f.read()
    if not after.startswith(before) or after.count(append_marker) != 1:
        raise RuntimeError('Trajectory handoff append verification')
    manifest['handoff']['trajectory_addendum'] = dict(status='APPENDED_VERIFIED',
        preserved_prefix=True, old_prefix_sha256=hashlib.sha256(before.encode()).hexdigest())
    r.write_json(r.MANIFEST, manifest)
    print('TRAJECTORY_FAILURE_DISCLOSED', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

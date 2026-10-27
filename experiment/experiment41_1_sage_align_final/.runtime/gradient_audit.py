"""Post-training parameter accounting; never trains or selects a checkpoint."""
from pathlib import Path
import csv
import json
import statistics
import sys
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import run as r


def main():
    r.verify_freeze()
    manifest = json.loads(r.MANIFEST.read_text())
    if manifest['status'] != 'COMPLETE_VALID_100_WITH_DISCLOSURES':
        raise RuntimeError('Run after the completed 100-fold report')
    rows = []
    for group in r.CONFIGS[1:]:
        state = json.loads((r.STATE_DIR / (group + '.json')).read_text())
        for row in state['rows']:
            fold = row['fold']
            ck = r.checked_load(torch, r.CKPT_DIR / f'{group}_fold{fold:02d}_best.pt')
            latest = r.checked_load(torch, r.CKPT_DIR / f'{group}_fold{fold:02d}_latest.pt')
            with (r.LOG_DIR / f'{group}_fold{fold:02d}_trace.csv').open() as f:
                trace = list(csv.DictReader(f))
            counts = {}
            for prefix, weights in [('branch', ck['branch']), ('output', ck['output'])]:
                counts[prefix] = sum(v.numel() for name, v in (weights or {}).items()
                                     if ck['gradients'].get(prefix + '/' + name, 0.) > 0.)
            rows.append(dict(config=group, fold=fold, best_epoch=row['best_epoch'],
                input_values=72 if ck['branch'] is not None else 0,
                active_channels_per_slot=row['active_feature_channels'] if ck['branch'] is not None else 0,
                train_effective_evidence_matrix_rank=row['evidence_matrix_rank'] if ck['branch'] is not None else 0,
                declared_trainable_parameters=row['trainable_parameters'],
                best_nonzero_gradient_tensor_parameter_count_branch=counts['branch'],
                best_nonzero_gradient_tensor_parameter_count_output=counts['output'],
                best_Adam_nonzero_first_moment_elements=sum(int(torch.count_nonzero(v['exp_avg']))
                    for v in ck['optimizer']['state'].values()),
                terminal_Adam_nonzero_first_moment_elements=sum(int(torch.count_nonzero(v['exp_avg']))
                    for v in latest['optimizer']['state'].values()),
                first_train_NLL=float(trace[0]['classification_loss']),
                terminal_train_NLL=float(trace[-1]['classification_loss']),
                maximum_epoch_mean_raw_cap_ratio=max(float(t['raw_cap_ratio']) for t in trace),
                maximum_epoch_saturation_fraction=max(float(t['saturation_fraction']) for t in trace),
                terminal_test_ACC=float(trace[-1]['validation_acc']),
                terminal_prediction_flips_vs_anchor=int(trace[-1]['prediction_flips_vs_anchor'])))
    r.write_csv(r.RESULTS / 'gradient_audit.csv', rows)
    marker = '<!-- corrected_v2_gradient_accounting -->'
    path = r.RESULTS / 'report.md'
    text = path.read_text()
    if marker not in text:
        text += ('\n' + marker + '\n\n### 实际梯度参数与输入维数补充\n\n'
                 '`gradient_audit.csv`逐折记录72个证据输入值、18个有效通道、训练有效证据矩阵秩、'
                 '所选best步骤中非零梯度张量包含的参数量，以及best和终止步骤的Adam一阶矩非零元素数。'
                 '非零梯度张量的参数量不是逐元素非零梯度计数。5760是完整分支可训练容量；'
                 '零初始化使首次更新通常仅5184个末层分支参数有非零梯度，不能称全部5760参数已完成学习。'
                 '后续梯度路径已在训练前验证；终止状态有更新不能替代正式选中的best checkpoint。'
                 '本补充只读取已选checkpoint，不训练、不更改选模或冻结设计。\n')
        full = [row for row in rows if row['config'] == 'E9_SAGE_FULL']
        text += ('\nE9训练NLL十折均值从首步的'
                 f'{statistics.mean(a["first_train_NLL"] for a in full):.6f}降至终止步骤的'
                 f'{statistics.mean(a["terminal_train_NLL"] for a in full):.6f}；'
                 '完整轨迹的最大患者饱和比例为'
                 f'{max(a["maximum_epoch_saturation_fraction"] for a in full):.6f}。'
                 '终止步骤只用于优化诊断，正式指标始终取首次严格最佳ACC checkpoint。\n')
        path.write_text(text)
    manifest['result_sha256'] = {p.name: r.sha_file(p) for p in r.RESULTS.iterdir() if p.is_file()}
    manifest['additional_analysis_sha256'] = {'.runtime/gradient_audit.py': r.sha_file(Path(__file__))}
    r.write_json(r.MANIFEST, manifest)
    print('GRADIENT_ACCOUNTING_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()

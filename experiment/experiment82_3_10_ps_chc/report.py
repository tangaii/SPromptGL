"""Build and verify the final results and append-only SABER handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import os
from datetime import datetime, timezone

import run as r
import history as h
import numpy as np

NAMES = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')


def md_table(cols, rows):
    cells = lambda items: '| ' + ' | '.join(map(str, items)) + ' |'
    return '\n'.join([cells(cols), cells(['---']*len(cols)), *(cells(row) for row in rows)])


def six_metrics(aggregate, scope='full'):
    return md_table(('组别', *NAMES),
                    [[key, *(f'{value["mean_pct"]:.4f} ± {value["sd_pct"]:.4f}' for value in gl[scope].values())]
                     for key, gl in aggregate['group_metrics'].items()])


def delta_six(delta):
    return [f'{delta[key]:+.4f}' for key in r.METRICS]


def count_label(counts):
    return '/'.join(str(counts[key]) for key in ('corrected', 'harmed', 'net'))


def wtl_label(item):
    return '/'.join(str(item[key]) for key in ('win','tie','loss'))


def paired_section(aggregate, scope):
    rows = []
    for name, pairs in aggregate['paired'].items():
        value = pairs[scope]
        rows.append([name, *delta_six(value['delta_pp']), wtl_label(value['wtl']['acc']),
                     wtl_label(value['wtl']['hard_auc']), *[count_label(value['changes'][c]) for c in ('asd','td','all')]])
    return md_table(('比较', *NAMES, 'ACC 胜/平/负', 'BA 胜/平/负', 'ASD 纠/误/净', 'TD 纠/误/净', '整体 纠/误/净'), rows)


def stages_section(aggregate):
    return md_table(('组别', '阶段', 'ASD 纠/误/净', 'TD 纠/误/净', '总体 纠/误/净'),
                    [[group, stage, *[count_label(ch[c]) for c in ('asd','td','all')]]
                     for group, g in aggregate['mechanisms'].items() for stage, ch in g['stages'].items()])


def statuses(status):
    def gate(group, kind):
        return 'GO' if status['science']['groups'][group][kind] else '未通过'

    checks = []
    for group, value in status['science']['groups'].items():
        for label, names in (('Performance', 'performance_checks'), ('Mechanism', 'mechanism_checks')):
            checks.append([group,label,gate(group,label.lower()+'_go'),
                           '、'.join(k for k,v in value[names].items() if not v) or '全部通过'])
    return md_table(('组别','门槛','结论','未满足条件'), checks)


def pair_row(name, scope, raw, removed=None, fold=None):
    ch = raw['changes']
    delta = raw['delta_pp']
    row = dict(comparison=name, scope=scope, removed_fold=removed, fold=fold, n_patients=raw.get('n'),
               **{f'{key}_delta_pp':delta[key] for key in r.METRICS})
    for cls in ('asd','td','all'):
        row.update({f'{cls}_{key}': val for key,val in ch[cls].items()})
    if 'wtl' in raw:
        for key, item in raw['wtl'].items():
            row.update({f'{key}_{score}': value for score,value in item.items()})
    return row


def metric_rows(audits):
    rows = [a['baseline'] for a in audits if a['group']=='G1'] + audits
    for row in sorted(rows, key=lambda x:(x['group'],x['fold'])):
        yield dict(group=row['group'], fold=row['fold'], best_epoch=row['best_epoch'], end_epoch=row['end_epoch'],
                   checkpoint_sha256=row['checkpoint_sha'],
                   **{metric:row['metrics'][metric]*100 for metric in r.METRICS})


def write_and_verify_tables(status, audits):
    from audit import changes, metrics

    agg = status['aggregate']
    folds = list(metric_rows(audits))
    h.write_csv(r.RESULTS/'fold_metrics.csv', folds)
    pairs = []
    for name, entry in agg['paired'].items():
        pairs.extend(pair_row(name, scope, entry[scope]) for scope in ('full','no_fold2'))
        for key, reduced in entry['max_contribution_removed'].items():
            pairs.append(pair_row(name, 'exclude_max_'+key, reduced, removed=reduced['removed']))
        for row in entry['full']['fold_deltas']:
            pairs.append(pair_row(name, 'fold', dict(delta_pp=row, changes=row['changes']), fold=row['fold']))
    for group, mech in agg['mechanisms'].items():
        for stage, value in mech['stages'].items():
            pair = pair_row(group+'-B0','mechanism_'+stage, dict(delta_pp=dict.fromkeys(r.METRICS),changes=value,n=869))
            pairs.append(pair)
    h.write_csv(r.RESULTS/'paired_comparisons.csv', pairs)
    patients = [dict(group=a['group'], **x) for a in audits for x in a['records']]
    h.write_csv(r.RESULTS/'patient_diagnostics.csv', patients)

    def csv_read(name):
        with (r.RESULTS/name).open() as handle:
            return list(csv.DictReader(handle))

    f = csv_read('fold_metrics.csv')
    assert len(f)==30 and len(patients)==1742
    f_map = {(x['group'],int(x['fold'])): x for x in f}
    assert len(f_map)==len(f)
    p = csv_read('patient_diagnostics.csv')
    assert len(p)==len(patients)
    report_rows = {}
    for a in audits:
        group, fold = a['group'], a['fold']
        selected = [item for item in p if item['group']==group and int(item['fold'])==fold]
        paired_rows = [dict(x, **{key:int(x[key]) for key in ('b0_label','off_label','full_label','row_1based','y')})
                       for x in selected]
        for pt in paired_rows:
            pt[group+'_label'] = pt['full_label']
        report_rows[(group,fold)] = paired_rows
        raw_y = np.array([int(item['y']) for item in selected])
        for display, prefix in ((group, 'full'), ('B0', 'b0')):
            probs = np.array([(float(item[prefix+'_asd']), float(item[prefix+'_td'])) for item in selected])
            assert np.array_equal(probs.argmax(1), [x[prefix+'_label'] for x in paired_rows])
            exact = metrics(probs, raw_y)
            assert all(abs(float(f_map[display,fold][key])-100*exact[key])<1e-10 for key in r.METRICS)
    for x, expected in zip(f, folds):
        assert (x['group'],int(x['fold']))==(expected['group'],expected['fold'])
        assert all(float(x[key])==expected[key] for key in r.METRICS)
    pair_entries = csv_read('paired_comparisons.csv')
    assert len(pair_entries) == len(pairs)
    for entry, expected in zip(pair_entries, pairs):
        assert entry['scope']==expected['scope'] and entry['comparison']==expected['comparison']
        for cls in ('asd','td','all'):
            for value in ('corrected','harmed','net'):
                key = cls+'_'+value
                assert int(entry[key])==expected[key]
        for metric in r.METRICS:
            key = metric+'_delta_pp'
            if expected[key] is not None:
                assert float(entry[key])==expected[key]
    for group in r.GROUPS:
        rows = [row for fold in r.FOLDS for row in report_rows[(group,fold)]]
        actual = changes(rows,'b0',group)['all']
        assert actual==agg['paired'][f'{group}-B0']['full']['changes']['all']
        allowed = [row for row in rows if row['protected']=='False']
        for stage, left, right in (('A','b0','off'),('B','off','full'),('C','b0','full')):
            assert changes(allowed,left,right)['all']==agg['mechanisms'][group]['stages'][stage]['all']
    return True




def stage_attribution(aggregate):
    rows = []
    for group, info in aggregate['mechanisms'].items():
        attrs = info['attribution']
        wrong = info['strata']['darg_wrong']
        rows.append([group,len(attrs['new_residual_correction']),len(attrs['residual_rescue']),
                     len(attrs['backbone_harm']),len(attrs['new_residual_harm']),
                     len(attrs['remaining_errors']),wrong['n'],f'{wrong["margin"]["B"]["mean"]:+.5f}',
                     f'{100*wrong["positive_margin_share"]:.2f}%'])
    return md_table(('组别','真正新增纠错','救回本组主路径错分','主路径新增误伤','残差新增误伤',
                     '最终错分人数','历史错分人数','历史错分者 B 阶段 ΔMargin','Margin 改善比例'),rows)


def alignment_table(aggregate):
    rows = []
    for group, info in aggregate['mechanisms'].items():
        for name, label in (('same_epoch_best','本组 Best'),('same_epoch_b0','历史 Best')):
            a = info['alignment'][name]
            pairs = [' → '.join(f'{a[variant][k]:.4f}' for variant in ('reference_pct','residual_off_pct'))
                     if a['folds'] else '—' for k in ('acc','hard_auc')]
            rows.append([group,label,a['folds'],*pairs,*[count_label(a['off_change'][c]) for c in ('asd','td','all')]])
    return md_table(('组别','对齐轮次','折数','Reference→Off ACC (%)','Reference→Off BA (%)',
                     'ASD 纠/误/净','TD 纠/误/净','总体 纠/误/净'),rows)


def spectral_table(aggregate):
    rows = []
    for group,g in aggregate['mechanisms'].items():
        a,b,o = g['strata']['all'],g['basis'],g['objectives']
        origin, learned = o['U0'],o['learned']
        rows.append([group,f'{b["projector_change_frobenius"]["mean"]:.4f}',
                     f'{b["max_angle_deg"]["mean"]:.2f}°',f'{b["gram_max_abs"]["max"]:.2e}',
                     f'{origin["variance_effective_rank"]["mean"]:.3f} → {learned["variance_effective_rank"]["mean"]:.3f}',
                     f'{learned["condition"]["max"]:.2f}',
                     f'{100*a["j_cover_u0"]["mean"]:.2f}% → {100*a["j_cover_u"]["mean"]:.2f}%',
                     f'{origin["bhattacharyya"]["mean"]:.3f} → {learned["bhattacharyya"]["mean"]:.3f}',
                     f'{100*a["conflicting_direction_fraction"]:.2f}%',a['llr_overrides']])
    return md_table(('组别','投影距','折最大主角均值','最大 Gram 误差','投影特征有效秩','最大类协方差条件数',
                     'Jacobian 覆盖','Bhattacharyya','语义统计冲突','LLR 反转方向数'),rows)


def optimizer_table(audits):
    rows = []
    for group in r.GROUPS:
        group_rows = [item for item in audits if item['group']==group]
        for name in group_rows[0]['training']['protection']:
            parts = [item['training']['protection'][name] for item in group_rows]
            fraction = max(x['ratio_max'] for x in parts)
            first = max(x['max_moment_budget_fraction']['exp_avg'] for x in parts)
            second = max(x['max_moment_budget_fraction']['exp_avg_sq'] for x in parts)
            increments, steps = (sum(x[k] for x in parts) for k in ('nonzero_step_count','steps'))
            best = np.mean([x['best']['accepted_reference_distance'] for x in parts])
            end = np.mean([x['last']['accepted_reference_distance'] for x in parts])
            mom = [np.mean([x['best']['moments'][m]['drift'] for x in parts]) for m in ('exp_avg','exp_avg_sq')]
            capped, capped_mom, active, fallback = (sum(x[k] for x in parts) for k in
                                                   ('capped_proposals','capped_moments','different_from_reference',
                                                    'fp32_fallbacks'))
            rows.append([group,name,f'{100*fraction:.2f}%',f'{100*first:.2f}% / {100*second:.2f}%',
                         f'{increments}/{steps}',active,f'{capped}/{capped_mom}/{fallback}',f'{best:.4f} / {end:.4f}',
                         f'{mom[0]:.3e} / {mom[1]:.3e}'])
    return md_table(('组别','历史参数组','累计参数预算最大占用','一阶/二阶预算最大占用','实际非零步',
                     '异于参考步','参数/动量限幅/舍入回退','Best/终轮参数 L2 距','Best 一阶/二阶矩 L2 距'),rows)


def optimizer_summary(audits):
    lines = []
    for group in r.GROUPS:
        blocks = [b for a in audits if a['group']==group for b in a['training']['protection'].values()]
        before = sum(b['capped_proposals'] for b in blocks)
        moments = sum(b['capped_moments'] for b in blocks)
        affected = sum(b['different_from_reference'] for b in blocks)
        exact = sum(b['steps'] for b in blocks)
        worst = max(b['ratio_max'] for b in blocks)
        last = max(b['last']['accepted_reference_distance']/b['last']['radius'] for b in blocks)
        lines.append(f'- {group} 的历史参数块在 {affected}/{exact} 次更新后与同步参考实际不同。'
                     f'参数和动量投影分别触发 {before} 和 {moments} 次，'
                     f'全程参数预算最大占用 {worst:.2%}，终轮最大占用 {last:.2%}。'
                     '轨迹距离受约束，临床分类和纠错效果由同轮及固定 Best 配对单独检验。')
    return '\n'.join(lines)


def gradient_table(audits):
    rows = []
    for group in r.GROUPS:
        group_audits = [value for value in audits if value['group'] == group]
        n = sum(value['end_epoch']-r.START for value in group_audits)
        for block in group_audits[0]['training']['gradients']:
            gradients = [value['training']['gradients'][block] for value in group_audits]
            steps = [value['training']['updates'][block] for value in group_audits]
            rows.append([group,block,n,sum(g['nonzero'] for g in gradients),
                         sum(v['nonzero'] for v in steps),f'{max(g["maximum"] for g in gradients):.3e}'])
    return md_table(('组别','新参数组','训练步数','真实分类梯度非零步','实际参数更新非零步','分类梯度范数最大值'),rows)


def subgroup_table(aggregate):
    rows = []
    for group,info in aggregate['mechanisms'].items():
        for name in ('darg_correct','darg_wrong','darg_wrong_asd','darg_wrong_td','high_confidence_b0_wrong',
                     'low_b0_margin','conflicting_evidence'):
            a = info['strata'][name]
            def fmt(value):
                return f'{value:+.5f}' if value is not None else '—'
            rows.append([group,name,a['n'],*[fmt(a['margin'][p]['mean']) for p in ('A','B','C')],
                         f'{100*a["positive_margin_share"]:.2f}%' if a['n'] else '—'])
    return md_table(('组别','患者层','人数','ΔMargin A','ΔMargin B','ΔMargin C','B 改善比例'),rows)


def mechanism_text(aggregate):
    lines = []
    for group, mech in aggregate['mechanisms'].items():
        a = mech['strata']['all']
        res = a['relative_residual']
        semantic = mech['ablations']['semantic_evidence_off']
        fixed = mech['ablations']['spectral_guidance_off']
        ids = '、'.join(map(str,mech['attribution']['new_residual_correction'])) or '无'
        lines.append(f'- {group} 有 {a["residual_nonzero"]}/869 人产生非零 CAM144 残差，相对范数均值/P95/最大为 '
                     f'{100*res["mean"]:.3f}%/{100*res["p95"]:.3f}%/{100*res["max"]:.3f}%，'
                     f'ASD 概率绝对变化均值 {a["probability_change"]["mean"]:.6f}，Gap 绝对变化均值 {a["gap_change"]["mean"]:.6f}。'
                     f'关闭语义引起 {semantic["flips"]} 例硬预测变化，恢复 U₀ 引起 {fixed["flips"]} 例。'
                     f'真正由残差新增纠正的历史行号为 {ids}。')
        lines.append(f'- {group} 的两个类别平均 Null 为 {a["null_asd"]["mean"]:.4f} / {a["null_td"]["mean"]:.4f}，'
                     f'方向级语义和统计加权证据绝对均值为 {a["semantic_contribution"]["mean"]:.4f} / '
                     f'{a["statistical_contribution"]["mean"]:.4f}。')
        if 'routing' in mech:
            value = mech['routing']
            lines.append(f'- {group} 固定 Best 实际 Keep/ASD/TD 选择为 {value["test_argmax"]}，'
                         f'平均概率为 {[round(x,4) for x in value["test_mean_probability"]]}。'
                         f'有 {value["test_reversals"]} 例对本组未增强预测选择反向候选，选择真类 {value["test_reversal_correct"]} 例。'
                         f'执行单一动作的候选间抵消为 0，概率混合的反事实抵消指数均值为 '
                         f'{value["soft_cancellation"]["mean"]:.4f}。'
                         f'所选动作的校准 Gap 与实际 Gap 平均绝对差 '
                         f'{value["selected_gap_calibration_error"]["mean"]:.6f}。'
                         f'训练期确定性分数与校准分数相对同掩码 Gap 的平均绝对误差分别为 '
                         f'{value["training_calibration_mae"][0]:.6f} / {value["training_calibration_mae"][1]:.6f}。'
                         f'训练期间反向选择 {value["training_reversals"]} 次，'
                         f'其中 {value["training_trusted_reversals"]} 次达到预注册真类候选收益条件。')
    return '\n'.join(lines)


def science_text(status):
    lines = []
    for g, flags in status['science']['groups'].items():
        stage = status['aggregate']['mechanisms'][g]['stages']
        previous = {'G1':-7,'G2':-2}[g]
        a,b,c = (stage[x]['all']['net'] for x in 'ABC')
        shift = status['aggregate']['mechanisms'][g]['best_epoch_shift']
        lines.append(f'- {g} 的 Best 相对历史轮次平均偏移 {shift["mean"]:+.1f} 轮，{shift["ties"]}/10 折同轮。'
                     f'A 阶段净变化 {a:+d} 人，较 E82.3.9 同名组变化 {a-previous:+d} 人。'
                     f'B 阶段 {b:+d}，C 阶段 {c:+d}。Performance {"GO" if flags["performance_go"] else "未达标"}，'
                     f'Mechanism {"GO" if flags["mechanism_go"] else "未达标"}，'
                     f'独立残差硬纠错门槛{"达标" if flags["new_residual_correction"] else "未达标"}。')
    return '\n'.join(lines)


GROUP_DESCRIPTION = ('G1 PS-HSC 在 16 维可学习谱空间融合条件化医学语义和完整协方差 LLR，沿原分类头的 Jacobian 产生有符号残差。'
                     'G2 PS-ICC 构造 ASD/TD 候选，按可靠证据与经同掩码监督校准的真实分类响应执行一个保持、ASD 或 TD 动作。'
                     '两组独立恢复 E74.4-G2 并共享同折原生 DARG Reference，原生优化器实际提案受累计参数和两级动量信赖域约束。'
                     '增强 CAM144 进入同一个原始随机分类头。')


def append_handoff(payload, manifest):
    initial = manifest['handoff_before']
    path = r.Path(initial['path'])
    text = payload.encode()
    sha = lambda raw:hashlib.sha256(raw).hexdigest()
    marker = b'## Experiment 82.3.10'
    try:
        with path.open('r+b') as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            original = file.read()
            prefix = original[:initial['size']]
            assert sha(prefix)==initial['sha256'], 'Historical handoff prefix changed'
            if marker in original[initial['size']:]:
                assert original.endswith(text), 'E82.3.10 handoff already contains different text'
                appended = original
            else:
                file.seek(0,os.SEEK_END)
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
                appended = original+text
            file.seek(0)
            assert file.read()==appended
        assert path.read_bytes()==appended and sha(appended[:initial['size']])==initial['sha256']
        return dict(status='APPENDED_READBACK_VERIFIED', path=str(path), prefix_sha256=initial['sha256'],
                    appended_sha256=sha(text), new_size=len(appended), final_sha256=sha(appended))
    except OSError as exc:
        staged = r.RESULTS/'handoff_pending.md'
        with h.atomic_writer(staged,'w') as stream:
            stream.write(payload)
        return dict(status='APPEND_FAILED', path=str(path), pending=str(staged), exception=str(exc))


def publish(status,audits):
    manifest = h.read_json(r.HERE/'manifest.json')
    a = status['aggregate']
    assert write_and_verify_tables(status,audits)
    status['csv_verified'] = True
    status['training'] = manifest['training']
    status['preflight_seconds'] = h.read_json(r.RT/'preflight.json')['elapsed']
    status['outcome_sha256'] = {name:h.sha_file(r.RESULTS/name) for name in
                               ('fold_metrics.csv','paired_comparisons.csv','patient_diagnostics.csv')}
    status['audit_implementation_sha256'] = {name:h.sha_file(r.HERE/name) for name in ('audit.py','report.py')}
    verdict = status['science']['decision']
    followup = ('保留通过门槛的候选，下一阶段在独立选模和预处理协议下复验。' if verdict=='GO' else
                '停止沿本轮两个候选调参。保留同步参考和真实优化器步审计，将后续研究聚焦到可独立纠正历史错分的患者证据。')
    training = manifest['training']
    early = status['audit_paralleled_with_training']
    elapsed = (f'预检 {status["preflight_seconds"]:.1f} 秒，训练 {training["elapsed"]:.1f} 秒，'
               f'训练后审计墙钟 {status["audit_seconds"]:.1f} 秒。'
               f'训练期间并行审计 {len(early["folds"])} 折，累计审计 worker 墙钟 '
               f'{status["cumulative_audit_worker_seconds"]:.1f} 秒。')
    integrity = (f'Preflight、Best、Latest 各 20/20。在线参考历史 Best 模型、测试概率、历史总损失、'
                 f'可用梯度档案、三优化器及末轮和 RNG 均 {status["reference_fidelity"]}/10 通过。'
                 f'独立复算完整 LLR、谱基、候选 Margin、患者指标、Adam/AdamW 更新以及累计参数和动量投影。'
                 f'四类 RNG 逐轮对齐，Latest 的下一训练、评估和 EMA 精确重放。'
                 f'受保护历史文件 {status["protection"]["count"]} 份，SHA-256 核验状态 '
                 f'{"PASS" if status["protection"]["pass_"] else "FAIL"}。')
    limitations = ('Fold2 两行保留原历史训练及测试身份，其新增残差归零，排除于新增带标签计算。'
                   '历史 val=test、CV 前 RFE/scaling、转导图与单 seed 支持同协议比较。'
                   '固定 Best 的语义干预衡量即时作用，医学文本的独立价值仍待重训对照。')
    header = (f'{status["status"]} · {verdict}。G1 10/10，G2 10/10。物理 GPU0，'
              f'{training["workers"]} 个 fold workers，采样峰值 {training["peak_sampled_gpu_mib"]} MiB。'
              f'{elapsed}首选目录返回 errno 5，权威目录为 {r.HERE}。')
    recovery = manifest.get('throughput_recoveries',[])
    if recovery:
        folds = '、'.join(str(item['fold']) for item in recovery)
        header += f'因 GPU 上下文吞吐异常，第 {folds} 折在本地 CPU 核绑定后从各自已提交 Latest 恢复。' 
    body = f'''{header}

{GROUP_DESCRIPTION} [源码定位和公式](../design_review.md)

### 正式结果

十折均值 ± 总体标准差，单位为 %。Hard-AUC 为 BA，Prob-AUC 取连续 ASD 概率。

{six_metrics(a)}

配对差单位为百分点，患者计数为纠错/误伤/净纠错。

{paired_section(a,'full')}

### 患者级归因

A 是历史 B0 → residual-off。B 是同一 Best residual-off → full。C 是历史 B0 → full。机制表剔除 Fold2 两名保护患者。

{stages_section(a)}

{stage_attribution(a)}

{alignment_table(a)}

### 谱空间与疾病证据

{spectral_table(a)}

{mechanism_text(a)}

{optimizer_summary(audits)}

### 审计与科研结论

{integrity}

审计对二进制 checkpoint 和 JSON 日志使用同一序列化比较规则。确定性分类头独立按 FP32 原生算子重算并逐值比较，FP64 解析 Jacobian、谱基及 Gaussian 数值审计保留原误差要求。首次独立复算的 FP64 跨设备范数归约使两项动量的 FP32 写回差一个 ULP。审计增加原设备缩放系数逐值核验，再用独立向量公式重建写回。正式训练数学与选模条件保持冻结。超出历史终轮的同步参考继续执行原更新方程，附录的归档核对区分延伸轮次。

{statuses(status)}

{science_text(status)}

{limitations} {followup}
'''
    extra = f'''
## 去 Fold2 配对

{six_metrics(a,'no_fold2')}

{paired_section(a,'no_fold2')}

## 历史错分与其他分层

{subgroup_table(a)}

## 真正的优化器步与轨迹漂移

{optimizer_table(audits)}

{gradient_table(audits)}

逐轮日志保留合法训练保留损失、原分类梯度、残差、参与秩和实际参数及动量更新预算。

[逐折指标](fold_metrics.csv) · [差值](paired_comparisons.csv) · [患者归因](patient_diagnostics.csv) · [审计](audit_status.json)
'''
    p = r.RESULTS/'report.md'
    status['result_timing_seconds'] = elapsed
    with h.atomic_writer(p,'w') as stream:
        stream.write('# Experiment 82.3.10 · PS-CHC\n\n'+body+extra)
    status['outcome_sha256']['report.md'] = h.sha_file(p)
    handoff = body.replace('(../design_review.md)',f'({r.HERE/"design_review.md"})')
    links = f'\n[完整报告]({p}) · [逐折指标]({r.RESULTS/"fold_metrics.csv"}) · [配对]({r.RESULTS/"paired_comparisons.csv"}) · [审计]({r.RESULTS/"audit_status.json"})\n'
    status['handoff'] = append_handoff('\n\n## Experiment 82.3.10 · PS-CHC\n\n'+handoff+links,manifest)
    h.atomic_json(r.RESULTS/'audit_status.json',status)
    h.update_manifest(status=status['status'], verdict=verdict, handoff=status['handoff'],
                      audit_source=status['audit_implementation_sha256'],
                      results={**status['outcome_sha256'],'audit_status.json':h.sha_file(r.RESULTS/'audit_status.json')})
    print(status['status'],verdict,status['handoff']['status'],flush=True)

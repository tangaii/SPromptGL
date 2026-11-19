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


def mechanism_text(group, m, audited):
    all_ = m['strata']['all']
    basis = m['basis']
    obj = m['objectives']
    delta = all_['relative_residual']
    j0 = all_['j_cover_u0']['mean']
    j = all_['j_cover_u']['mean']
    out = [f'### {group}', '',
           f'谱基每折秩为 16，最大正交误差 {basis["gram_max_abs"]["max"]:.3g}。平均投影矩阵 Frobenius 位移 '
           f'{basis["projector_change_frobenius"]["mean"]:.4f}，各折最大主角度的平均值 '
           f'{basis["max_angle_deg"]["mean"]:.2f}°，最大 Gram 条件数 {basis["gram_condition"]["max"]:.4f}。'
           f'同一 Best 原 CAM 的合法训练投影方差有效秩 {obj["learned"]["variance_effective_rank"]["mean"]:.2f}。', '',
           f'固定 Best 合法训练特征的平均 Bhattacharyya 分离值从 {obj["U0"]["bhattacharyya"]["mean"]:.4f} '
           f'变为 {obj["learned"]["bhattacharyya"]["mean"]:.4f}，其中协方差项从 '
           f'{obj["U0"]["bhattacharyya_covariance"]["mean"]:.4f} 变为 {obj["learned"]["bhattacharyya_covariance"]["mean"]:.4f}。'
           f'同头确定性困难梯度覆盖率从 {100*obj["U0"]["error_gradient_coverage"]["mean"]:.2f}% 变为 '
           f'{100*obj["learned"]["error_gradient_coverage"]["mean"]:.2f}%。以上均在相同 Best 下对 U0 和学习谱基评估。', '',
           f'测试 CAM144 的分类头 Jacobian 覆盖率从 {100*j0:.2f}% 变为 {100*j:.2f}%。实际残差非零 '
           f'{all_["residual_nonzero"]}/{all_["n"]} 人，相对范数均值/P95/最大为 '
           f'{100*delta["mean"]:.3f}%/{100*delta["p95"]:.3f}%/{100*delta["max"]:.3f}%。'
           f'未限幅相对范数超过 8% 的人数为 {all_["raw_limit_count"]}（{100*all_["raw_limit_count"]/all_["n"]:.2f}%），'
           f'ASD 概率平均绝对移动 {100*all_["probability_change"]["mean"]:.4f} pp，'
           f'Gap 平均绝对移动 {all_["gap_change"]["mean"]:.6f}。'
           f'语义与 LLR 在 {100*all_["conflicting_direction_fraction"]:.2f}% 患者方向上冲突，'
           f'LLR 胜过语义方向 {all_["llr_overrides"]} 次，负门控 {all_["reversed_gate_directions"]} 次，'
           f'一阶方向抵消比例均值 {100*all_["cancellation"]["mean"]:.2f}%。', '']
    cohort_names = {'all':'全部','darg_correct':'DARG 判对','darg_wrong':'DARG 判错','low_b0_margin':'低 B0 Margin',
                    'high_confidence_b0_wrong':'高置信 B0 错误','conflicting_evidence':'语义与 LLR 冲突方向过半',
                    'asd':'ASD','td':'TD'}
    def val(x, precision=5):
        return '—' if x is None else f'{x:+.{precision}f}'
    cohorts = []
    for k, v in m['strata'].items():
        cohorts.append([cohort_names[k],v['n'],*[val(v['margin'][s]['mean']) for s in ('A','B','C')],
                        '—' if v['positive_margin_share'] is None else f'{100*v["positive_margin_share"]:.2f}%',
                        val(v['alignment']['mean'])])
    out.extend([md_table(('分层','人数','A ΔMargin','B ΔMargin','C ΔMargin','B Margin 改善率','δ 对齐负损失梯度'), cohorts), ''])
    attribution = m['attribution']
    out.extend([f'患者归因按历史行号复核。残差真正新增纠错 {len(attribution["new_residual_correction"])} 人，'
                f'原本正确患者转为 off 错分 {len(attribution["backbone_harm"])} 人，残差救回其中 '
                f'{len(attribution["residual_rescue"])} 人，对历史正确且 off 仍正确的患者新误伤 '
                f'{len(attribution["new_residual_harm"])} 人。患者行号见审计 JSON。', ''])
    modes = []
    for mode, v in m['ablations'].items():
        modes.append([mode, v['flips'], count_label(v['change']['all']), f'{100*v["probability_absolute_change"]["mean"]:.5f}',
                      *[f'{v["metrics"][key]["mean_pct"]:.4f}' for key in r.METRICS]])
    out.extend([md_table(('固定权重干预','硬翻转','到 Full 纠/误/净','|Δ ASD p| pp',*NAMES), modes), ''])
    gradient_rows = []
    for key in audited[0]['timeline']['shared_gradients']:
        rr = [a['timeline']['shared_gradients'][key] for a in audited]
        steps = sum(a['timeline']['epochs'] for a in audited)
        gradient_rows.append([key, f'{100*sum(x["conflicts"] for x in rr)/steps:.2f}%',
                              f'{100*sum(x["capped"] for x in rr)/steps:.2f}%',
                              f'{100*max(x["maximum_ratio"] for x in rr):.4f}%',
                              f'{100*sum(x["best_relative_drift"] for x in rr)/len(rr):.3f}%'])
    out.extend([md_table(('共享组','冲突轮次率','限幅轮次率','实际梯度增量峰值','Best 相对初始权重漂移'), gradient_rows), ''])
    added = []
    for key in audited[0]['timeline']['added_gradients']:
        rr = [a['timeline']['added_gradients'][key] for a in audited]
        added.append([key, f'{max(x["maximum_class_norm"] for x in rr):.5g}',
                      sum(x['positive_class'] for x in rr), f'{100*max(x["maximum_ratio"] for x in rr):.4f}%'])
    out.extend([md_table(('新增组','真实分类梯度最大范数','正范数轮次','辅助梯度实际占比峰值'), added), ''])
    return '\n'.join(out)


def make_report(status, audits, manifest):
    agg = status['aggregate']
    result = status['science']['verdict']
    ex = status['execution']
    gpu = manifest['environment']['gpu']
    rows = list(metric_rows(audits))
    fold_table = md_table(('组别','折','Best epoch','终止 epoch',*NAMES),
                          [[x['group'],x['fold'],x['best_epoch'],x['end_epoch'],*[f'{x[k]:.4f}' for k in r.METRICS]] for x in rows])
    score_removed = md_table(('比较','移除标准','折','ΔACC pp','ΔBA pp','患者净纠错'),
                             [[name,k,val['removed'],f'{val["delta_pp"]["acc"]:+.4f}',f'{val["delta_pp"]["hard_auc"]:+.4f}',val['changes']['all']['net']]
                              for name,p in agg['paired'].items() for k,val in p['max_contribution_removed'].items()])
    archived = sum(x['timeline']['archived_rng_matches'] for x in audits)
    later = sum(x['timeline']['beyond_archive_matches'] for x in audits)
    compared = sum(x['timeline']['g1_g2_common_rng_matches'] for x in audits)
    if result == 'GO':
        trial = '本轮至少一组通过双重预注册门槛。下一步应独立验证跨数据性能和医学文本的贡献。'
    else:
        trial = ('本轮的双重预注册门槛未达。现有证据不足以将失效归因于 16 维容量，'
                 '同方案继续调整谱方向缺少分类收益依据。')
    explanations = []
    for group, mechanism in agg['mechanisms'].items():
        stage_nets = {key: value['all']['net'] for key,value in mechanism['stages'].items()}
        if stage_nets['A'] < 0 and stage_nets['B'] > 0 and stage_nets['C'] <= 0:
            explanations.append(f'{group} 在同一 Best 下的直接净纠错为 {stage_nets["B"]:+d} 人，'
                                f'A 阶段净 {stage_nets["A"]:+d} 人，抵消了该增益。'
                                '5% 约束控制单次同点梯度，长期 Adam 更新和选中权重仍可远离 DARG Best。')
        elif stage_nets['B'] <= 0:
            explanations.append(f'{group} 的固定 Best 直接净纠错为 {stage_nets["B"]:+d} 人，'
                                '谱空间的数值变化未形成净硬分类收益。')
        else:
            explanations.append(f'{group} 的固定 Best 直接净纠错为 {stage_nets["B"]:+d} 人，'
                                f'完整模型净纠错为 {stage_nets["C"]:+d} 人。')
    text = [
        '# Experiment 82.3.7 · PS-DSO', '',
        f'**{status["status"]} · {result}**。B0 历史十折已核验。G1、G2 各完成十折。', '',
        f'权威目录 `{r.HERE}`。物理 GPU{gpu["index"]} {gpu["name"]}，UUID `{gpu["uuid"]}`，'
        f'训练 G1/G2 分别最多 {ex["G1"]["actual_max_workers"]}/{ex["G2"]["actual_max_workers"]} worker，'
        f'采样显存峰值 {status["gpu0_peak_mib"]} MiB。训练 {status["training_wall_seconds"]:.1f} 秒，'
        f'独立审计 {status["audit_wall_seconds"]:.1f} 秒，总耗时 {status["elapsed_seconds"]:.1f} 秒。'
        f'恢复事件 {sum(len(a["resumes"]) for a in audits)} 次。', '',
        '首选 OSSFS 路径创建时返回 errno 5，正式产物统一位于上述目录。', '',
        '## 设计与历史核对', '',
        '原论文端到端训练图学习和分类，原作者分类头为 LayerNorm、Linear、GELU、Dropout、Linear、log_softmax。'
        '历史 DARG 沿用 PLNNoPEN、CAM144，与 E61 epoch10 的三历史优化器及四类 RNG 恢复链。'
        '逐折记录的原生训练前向有六次 Dropout 模块调用，原分类头的 p 为 0.35。'
        '两组在原随机分类头内对增强 CAM144 计算唯一主分类损失。G1 用正则化高斯距离学习移动谱基，'
        'G2 用合法困难训练患者同掩码分类梯度协方差学习谱基。两组同用 FP64 Stiefel 图坐标、原医学 Prompt、'
        '有符号语义和 LLR、8% 平滑限幅，以及 CAM144 坐标的类别 EMA。历史共享增量按组去冲突并限制在 5%，'
        '新增参数辅助预算 25%。E82.3.5 和 E82.3.6 的结论、数学式、数据边界及超参数详见冻结文档。 '
        f'[设计冻结]({r.HERE}/design_review.md) · [SPromptGL 论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)', '',
        '## 十折正式结果（%）', '',
        six_metrics(agg), '',
        'Hard-AUC 对应硬标签 BA。Prob-AUC 使用连续 ASD 概率。各折等权，标准差使用总体定义。所有判定使用完整精度。', '',
        '## 同折同患者配对（pp）', '', paired_section(agg,'full'), '',
        '## 去 Fold2（%）', '', six_metrics(agg,'no_fold2'), '', paired_section(agg,'no_fold2'), '',
        '## 去最大单折贡献', '', score_removed, '',
        '## 患者级机制', '',
        'A 为 B0 → residual-off，包含历史参数轨迹和选中 epoch 的共同作用。B 为同一模型同一 Best 的 '
        'residual-off → Full。C 为 B0 → Full。以下带标签机制均扣除 Fold2 的 377、407 行。正式性能保留历史 871 人。', '',
        stages_section(agg), '',
    ]
    for group,m in agg['mechanisms'].items():
        text.append(mechanism_text(group,m,[a for a in audits if a['group']==group]))
    table = []
    for group,m in agg['mechanisms'].items():
        wrong = m['strata']['darg_wrong']
        table.append([group,f'{100*wrong["positive_margin_share"]:.2f}%',
                      f'{wrong["margin"]["B"]["mean"]:+.6f}', len(m['attribution']['new_residual_correction']),
                      m['stages']['B']['all']['net']])
    text.extend(['### 对 E82.3.6 错分患者的解释', '',
                 'E82.3.6 的固定 U 只覆盖约 11.91% 的头 Jacobian 能量，非负门控延续了现有证据的符号。'
                 '历史判对者真类 Margin 平均 +0.01042，历史判错者 −0.01080，只有 4.41% 判错者改善。'
                 '两类变化方向支持其残差偏向强化已有决策的解释。该实验无法在子空间、门控与训练目标之间单独归因。', '',
                 md_table(('组别','原错分者直接 Margin 改善率','原错分者 B ΔMargin','真实新增纠错','固定 Best 净纠错'),table), '',
                 '固定 Best 上屏蔽医学证据可量化即时作用。本轮未包含独立语义对照，医学文本的独立诊断价值未建立。', '',
                 '## 独立审计与科研判断', '',
                 f'Preflight {status["preflight_pass"]}/20，Best {status["best_pass"]}/20，Latest {status["latest_pass"]}/20。'
                 f'原生首步与历史逐折匹配。{archived} 个有历史档案的 epoch 共享 RNG 完全匹配，'
                 f'{later} 个超历史停点 epoch 验证链式连续与调用顺序。G1/G2 共同的 {compared} 个 epoch 严格配对。'
                 f'核验 {status["protection"]["count"]} 个保护文件的 SHA-256。Fold2 保护行新增残差均精确为零。', '',
                 'Best 概率由原头独立再算，六指标由单独实现复算。QR 与闭式头 Jacobian 对照训练公式，原始坐标矩的动态投影和 '
                 'Gaussian LLR 经独立复算。Latest 恢复后精确重现已提交评估和 EMA，再重放下一训练步、评估与更新后的完整状态。', '',
                 statuses(status), '', *explanations, '', trial, '',
                 '局限包括单 seed、历史 val=test 选模、CV 前 RFE/scaling、转导图与保留 Fold2 历史身份。'
                 '结果用于历史同协议配对分析，独立泛化诊断性能仍待验证。', '',
                 '## 各折正式结果（%）', '', fold_table, '',
                 f'交接文件追加和前缀核验状态 **{status["handoff"]["status"]}**。', '',
                 f'[逐折指标]({r.RESULTS}/fold_metrics.csv) · [配对]({r.RESULTS}/paired_comparisons.csv) · '
                 f'[患者机制]({r.RESULTS}/patient_diagnostics.csv) · [独立审计]({r.RESULTS}/audit_status.json)', ''])
    return '\n'.join(text)


def handoff_payload(status):
    agg = status['aggregate']
    rows = []
    for group,m in agg['mechanisms'].items():
        p = agg['paired'][f'{group}-B0']['full']
        all_ = m['strata']['all']
        gate = status['science']['groups'][group]
        rows.append(f'- {group} Performance {"GO" if gate["performance_go"] else "未通过"}，'
                    f'Mechanism {"GO" if gate["mechanism_go"] else "未通过"}。相对 B0 ΔACC {p["delta_pp"]["acc"]:+.4f} pp，ΔBA {p["delta_pp"]["hard_auc"]:+.4f} pp，'
                    f'总体纠错/误伤/净 {count_label(p["changes"]["all"])}，固定 Best 直接纠错/误伤/净 '
                    f'{count_label(m["stages"]["B"]["all"])}。学习后 16 维 Jacobian 覆盖平均 '
                    f'{100*all_["j_cover_u"]["mean"]:.2f}%，平均投影位移 {m["basis"]["projector_change_frobenius"]["mean"]:.4f}。')
    return '\n'.join(['', '## Experiment 82.3.7 · PS-DSO', '',
                      f'{status["status"]}，科研判定 **{status["science"]["verdict"]}**。B0 审核 10 折，G1 和 G2 各训练和审核 10 折。'
                      f'正式目录 `{r.HERE}`。物理 GPU0。', '',
                      'G1 用正则化 Bhattacharyya 分离学习活动 16 维子空间。G2 用类均衡困难患者梯度协方差覆盖学习同维子空间。'
                      '双组通过增强 CAM144 和原 Dropout 分类头端到端优化，原坐标类别 EMA 随 U 投影，'
                      '共享分类梯度增量执行去冲突和 5% 范数预算。', '', six_metrics(agg), '', *rows, '',
                      f'Preflight {status["preflight_pass"]}/20，Best {status["best_pass"]}/20，Latest {status["latest_pass"]}/20。'
                      '历史分折、Fold2 保护、原生首步、四类随机轨迹、历史文件保护和下一步重放均完成独立审计。'
                      '医学文本的独立收益未建立。后续是否保留 16 维须看固定权重硬纠错和端到端净收益。', '',
                      f'[正式报告]({r.RESULTS}/report.md) · [逐折]({r.RESULTS}/fold_metrics.csv) · '
                      f'[配对]({r.RESULTS}/paired_comparisons.csv) · [审计]({r.RESULTS}/audit_status.json)', ''])


def append_handoff(status, manifest):
    path = h.HISTORY_ROOT/'handoff.md'
    payload = handoff_payload(status).encode('utf-8')
    marker = b'\n## Experiment 82.3.7 \xc2\xb7 PS-DSO\n'
    pre_check = manifest['handoff_before']
    try:
        with path.open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            first = handle.read()
            assert len(first)>=pre_check['size']
            assert hashlib.sha256(first[:pre_check['size']]).hexdigest()==pre_check['sha256']
            if marker in first:
                index = first.index(marker)
                assert first[index:].startswith(payload)
                original = first[:index]
            else:
                original = first
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            reread = path.read_bytes()
            assert reread[:len(original)]==original and reread[len(original):].startswith(payload)
            assert reread.count(marker)==1
        return dict(status='APPENDED_VERIFIED', path=str(path), original_bytes=len(original),
                    original_sha256=hashlib.sha256(original).hexdigest(), appended_sha256=hashlib.sha256(payload).hexdigest(),
                    verified_file_sha256=hashlib.sha256(reread).hexdigest())
    except (OSError, AssertionError) as exc:
        pending = r.RESULTS/'handoff_pending.md'
        with h.atomic_writer(pending,'wb') as out:
            out.write(payload)
        return dict(status='APPEND_UNVERIFIED', path=str(path), pending=str(pending), exception=repr(exc))


def publish(status, audits):
    manifest = h.read_json(r.HERE/'manifest.json')
    ex = {group:h.read_json(r.RT/f'execution_{group}.json') for group in r.GROUPS}
    status['execution'] = ex
    status['training_wall_seconds'] = sum(x['wall_seconds'] for x in ex.values())
    status['audit_wall_seconds'] = sum(status['audit_wall_seconds_per_group'].values())
    status['gpu0_peak_mib'] = max(x['sampled_gpu0_peak_mib'] for x in ex.values())
    status['elapsed_seconds'] = (datetime.now(timezone.utc)-datetime.fromisoformat(manifest['started_utc'])).total_seconds()
    assert status['status']=='COMPLETE_AUDITED'
    status['tables_verified'] = write_and_verify_tables(status, audits)
    status['handoff'] = append_handoff(status, manifest)
    if status['handoff']['status']!='APPENDED_VERIFIED':
        status['status'] = 'PARTIAL_HANDOFF'
    with h.atomic_writer(r.RESULTS/'report.md','w') as out:
        out.write(make_report(status, audits, manifest))
    h.atomic_json(r.RESULTS/'audit_status.json', status)
    artifacts = ('fold_metrics.csv','paired_comparisons.csv','patient_diagnostics.csv','audit_status.json','report.md')
    h.update_manifest(status=status['status'], scientific_verdict=status['science']['verdict'],
                      result_sha256={x:h.sha_file(r.RESULTS/x) for x in artifacts},
                      protection_after=status['protection'], handoff=status['handoff'],
                      execution=ex, elapsed_seconds=status['elapsed_seconds'],
                      audit_source_sha256=h.sha_file(r.HERE/'audit.py'),
                      report_source_sha256=h.sha_file(r.HERE/'report.py'))
    print(status['status'], status['science']['verdict'], status['handoff']['status'], flush=True)

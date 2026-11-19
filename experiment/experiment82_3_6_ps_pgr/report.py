"""Tables, scientific report and append-only handoff for PS-PGR."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import os

import run as r
import history as h

LABELS = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')
HANDOFF = h.HISTORY_ROOT / 'handoff.md'


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|',
                      *['|'+'|'.join(map(str, row))+'|' for row in rows]])


def metric_table(data):
    return table(('组别', *LABELS), [(g, *(f'{v[k][0]:.4f} ± {v[k][1]:.4f}' for k in r.METRICS))
                                    for g, v in data.items()])


def count_changes(c, prefix=''):
    return '/'.join(str(c[f'{prefix}{key}']) for key in ('fixed', 'harmed', 'net'))


def paired_table(pairs, scope):
    return table(('配对', *LABELS, 'ACC 胜/平/负', 'BA 胜/平/负', 'ASD 纠错/误伤/净', 'TD 纠错/误伤/净', '全部'),
                 [(name, *(f"{v[scope]['delta_pp'][k]:+.4f}" for k in r.METRICS),
                   *('/'.join(map(str,v[scope]['win_tie_loss'][k])) for k in ('acc','hard_auc')),
                   *(count_changes(v[scope]['patients'], pre) for pre in ('asd_','td_','')))
                  for name,v in pairs.items() if scope in v])


def execution_line(status):
    saved = status['group_execution']
    if not saved:
        return '未进入正式训练。'
    return '。'.join(f"{g} {v['completed_folds']}/10 折，训练墙钟 {v['wall_seconds']:.1f} 秒"
                    for g,v in saved.items()) + '。'


def report_tables(status):
    official = [(v['group'], v['fold'], v['best_epoch'], v['end_epoch'],
                 *(f"{100*v['metrics'][k]:.4f}" for k in r.METRICS))
                for v in status['references']+status['tasks']]
    aggregate = status['aggregate']
    return [
        '### 十折指标（%）',
        metric_table(aggregate['metrics_10']),
        'Hard-AUC 为历史 Balanced Accuracy。Prob-AUC 用连续 ASD 概率，ASD=0、TD=1。标准差使用总体定义。'
        '正式性能保留 Fold2 的历史成员。',
        '### 逐折正式结果（%）',
        table(('组别','折','Best epoch','终止 epoch',*LABELS), official),
        '### 同折同患者配对（百分点）',
        paired_table(aggregate['pairs'], 'all'),
        '### 去 Fold2 的九折敏感性分析',
        metric_table(aggregate['metrics_without_fold2']),
        paired_table(aggregate['pairs'], 'excluding_fold2'),
    ]


def audit_line(status):
    hashes, rng, native = (status[k] for k in ('protection','rng','native_protection'))
    total = sum(v['completed_folds'] for v in status['group_execution'].values())
    match = '全部一致' if hashes['pass'] else f"发现 {len(hashes['mismatches'])} 项异常"
    return (f"Preflight {status['preflight_pass']}/10，Best {status['best_pass']}/{total}，Latest {status['latest_pass']}/{total}。"
            f"原 DARG 逐轮精确保护 {native['exact_native_epochs']} 轮，归档日志标记 {native['archive_journal_milestones']} 个，"
            f"归档 Best/Latest 权重、优化器及对应概率/RNG 各核验 "
            f"{native['archive_best_parameters_optimizers_probability_exact']}/{native['archive_latest_parameters_optimizers_rng_exact']} 折。"
            f"共有历史 RNG {rng['common_darg_rng_epochs']} 轮，延伸连续链 {rng['extended_rng_epochs']} 轮，"
            f"与 G1 或医学母组的共同 RNG {rng['paired_g1_rng_epochs']} 轮。"
            f"EMA 独立重放 {rng['ema_updates']} 次，历史保护 SHA-256 {hashes['count']} 项{match}。")


def science_text(status):
    names = {'acc':'ACC','ba':'BA','prob_auc':'Prob-AUC','sen':'SEN-ASD','patient_net':'总体净纠错',
             'direct_hard_net':'直接净纠错','truly_new':'历史错分中真正新增纠错','beyond_fold2':'去 Fold2 敏感性',
             'beyond_one_fold':'单折贡献敏感性','audited_protocol':'训练协议','nonzero_bounded_feature':'有限有界特征',
             'actual_gap_change':'Gap 变化','residual_gradients':'残差梯度'}
    rows = []
    for group,arm in status['science']['arms'].items():
        if arm['audited']:
            p = '通过' if arm['performance_go'] else '未通过'
            m = '通过' if arm['mechanism_go'] else '未通过'
            failures = [names[k] for k,v in {**arm['performance_checks'],**arm['mechanism_checks']}.items() if not v]
            rows.append(f"{group} Performance GO {p}，Mechanism GO {m}。"+
                        (f"未满足 {'、'.join(failures)}。" if failures else ''))
    if status['g2_status']=='PRE_REGISTERED_SKIP_G2':
        g2_reasons = {'net_vs_b0':'相对 B0 总体净纠错','fixed_direct_net':'固定 Best 直接净纠错',
                      'true_new_net_vs_b0':'历史错分真正纠错减直接新误伤'}
        failed = '、'.join(s for k,s in g2_reasons.items() if not status['g2_checks'][k])
        rows.append(f"G2 按预注册跳过，未通过的执行条件为 {failed}。")
    if status['anonymous_status']=='PRE_REGISTERED_SKIP_ANONYMOUS':
        rows.append('未出现医学组双 GO，匿名 Query 独立重训按预注册跳过。即时 Query 换入提供机制诊断，医学文本的独立增益仍待验证。')
    elif status['science']['medical_advantage'] is True:
        rows.append('同折重训的匿名 Query 对照支持医学文本在本历史协议下有正向配对增益。')
    elif status['science']['medical_advantage'] is False:
        rows.append('同折重训的匿名 Query 对照未达到医学文本独立增益判据。')
    return rows


CAUSES = {
    'true_new_from_historical_wrong':'历史错分由残差真正新增纠正',
    'historical_correct_to_off_wrong':'历史正确转为 residual-off 错分',
    'off_error_recovered_from_historical_correct':'先错分后由残差救回的历史正确患者',
    'new_residual_harm_to_historical_correct':'残差对历史正确患者的新误伤',
}


def mechanisms(group, data, tasks):
    a, b, c = (data[k] for k in ('residual_off_vs_darg','direct_hard_contribution','full_vs_darg'))
    resid, rho, fusion = (data[k] for k in ('residual_relative_pct','classifier_spectral_coverage','fusion_counts'))
    t1,t4 = (data['treatments'][k] for k in ('T1','T4'))
    opening = [
        f"### {group} 固定 Best",
        'A 为历史 DARG → residual-off。B 为 residual-off → full。C 为历史 DARG → full。'
        '本节带标签诊断使用 869 人。正式配对 C 使用 871 人。',
        table(('阶段','ASD纠错/误伤/净','TD纠错/误伤/净','全部纠错/误伤/净'),
              [(name,count_changes(row,'asd_'),count_changes(row,'td_'),count_changes(row))
               for name,row in [('A',a),('B',b),('C',c)]]),
        table(('患者归因','人数','ASD','TD'),
              [(label,len(row['rows_1based']),row['asd'],row['td'])
               for key,label in CAUSES.items() for row in (data['patient_attribution'][key],)]),
        f"实际 CAM 残差非零 {data['nonzero_residual_rows']} 人，平均/P95/最大 "
        f"{resid['mean']:.5f}%/{resid['p95']:.5f}%/{resid['max']:.5f}%。"
        f"关闭残差引起 ASD 概率平均绝对变化 {100*t1['probability_change']['mean']:.6f} 个百分点，"
        f"确定性 Gap 平均绝对变化 {data['logit_gap_abs_change']['mean']:.7f}。",
        f"16 维谱的分类雅可比能量覆盖率平均 {100*rho['mean']:.3f}%，P95 {100*rho['p95']:.3f}%。"
        f"有符号残差方向与 Jacobian 余弦均值 {data['classifier_cosine']['mean']:+.6f}。",
        table(('分层','人数','B 真类 Margin 均值','B 正向占比','C 真类 Margin 均值','C 正向占比'),
              [(label,row['n'],f"{row['change']['mean']:+.7f}",f"{100*row['fraction_positive']:.2f}%",
                f"{row['versus_b0_margin']['mean']:+.7f}",f"{100*row['versus_b0_fraction_positive']:.2f}%")
               for key,label in [('all','全部'),('darg_correct','B0 正确'),('darg_wrong','B0 错误'),('asd','ASD'),
                                 ('td','TD'),('low_margin','B0 Margin < 2'),('high_margin','B0 Margin ≥ 2')]
               for row in (data['margin'][key],) if row['n']]),
        f"语义与 LLR 有 {fusion['agree']} 个同号方向和 {fusion['conflict']} 个冲突方向。"
        f"LLR 在 {fusion['llr_overrides']} 个方向改变融合后的符号。"
        f"实际非零方向效应中的 {fusion['semantic_opposed_active']}/{fusion['active_effects']} 个与语义符号相反。"
        f"语义与统计分量对一阶有符号 Gap 均值分别为 {data['semantic_first_order']['mean']:+.7f} 和 "
        f"{data['statistical_first_order']['mean']:+.7f}。",
        f"固定 Best 替换 Query 后 {data['query_swap_direction_flips']} 个方向变号，硬翻转 {t4['versus_full']['flips']} 人。"
        f"残差、门控和差异证据关闭后三种输出逐元素等价。",
        table(('折','A纠错/误伤/净','B纠错/误伤/净','C纠错/误伤/净','真新增','救回','直接新误伤','ASD/TD Null','融合 α','谱覆盖率'),
              [(task['fold'],*(count_changes(m[k]) for k in ('residual_off_vs_darg','direct_hard_contribution','full_vs_darg')),
                *(len(m['patient_attribution'][k]['rows_1based']) for k in ('true_new_from_historical_wrong','off_error_recovered_from_historical_correct','new_residual_harm_to_historical_correct')),
                f"{m['null_asd']['mean']:.4f}/{m['null_td']['mean']:.4f}",f"{m['alpha_distribution']['mean']:.4f}",
                f"{100*np_mean(m['classifier_spectral_coverage']):.3f}%")
               for task in tasks for m in (task['interventions'],)]),
        f"合法训练 L_inc 首轮及 Best 轮均值 {data['first_objective']['inc']:+.7f}、{data['best_objective']['inc']:+.7f}。"
        "方向投影、语义、谱可靠性、融合和门控梯度记录在训练日志及 Preflight。",
    ]
    if align:=data.get('b0_best_alignment'):
        opening += [
            f"B0 Best 权重对齐中，{align['exact_folds']} 折为新增模块与历史基座同 epoch，"
            f"{align['frozen_terminal_folds']} 折使用停止时模块。"
            f"同 epoch 子集的纠错/误伤/净为 {count_changes(align['exact_only'])}，"
            f"全部固定权重探针为 {count_changes(align['all_b0_weight_probes'])}。",
            '原 DARG 与本组 residual-off 在相同 epoch 的全部主干参数和三历史优化器逐字节一致。'
            'A 阶段的变化对应选取不同 checkpoint epoch。',
        ]
    if stats:=data.get('shared_projection'):
        opening += [
            f"共享组×轮数 {stats['epochs']}，冲突投影 {stats['conflicts']}，限幅 {stats['caps']}，"
            f"非零注入 {stats['nonzero_injections']}，实际施加上限 {100*stats['max_injected_ratio']:.5f}%。"
            f"舍入修复 {stats['rounding_repairs']}，数值检查后置零 {stats['numerical_no_injection']}。"
            "各折各组的夹角和参数漂移见审计 JSON。",
        ]
    return opening


def np_mean(values):
    return sum(values)/len(values)


def write_report(status, manifest):
    group_data = status['group_execution']
    workers = max((v['max_workers'] for v in group_data.values()),default=0)
    peak = max((v['peak_sampled_gpu0_mib'] for v in group_data.values()),default=0)
    notes = [
        '# Experiment 82.3.6 · PS-PGR',
        '**Prompt-guided Protected Gradient Residual Learning（疾病 Prompt 引导的受保护梯度残差学习）**',
        f"状态 **{status['status']}**，科研判定 **{status['science']['verdict']}**。"
        f"物理 GPU0，训练 {workers} worker，审计最多 {status['audit_workers']} worker，训练显存采样峰值 {peak} MiB。"
        f"运行总耗时 {status['elapsed_seconds']:.1f} 秒。{execution_line(status)}",
        f"目录 `{r.HERE}`。G2 状态 `{status['g2_status']}`，Anonymous 状态 `{status['anonymous_status']}`。",
        '## 核查与方法',
        'SPromptGL 使用 GFR 多关系患者图、CAM 与 PEN，并通过分类和图学习联合端到端训练。'
        '本实验唯一基座 E74.4-G2 使用历史 PLNNoPEN、CAM144 及带 Dropout 的原分类头。'
        'E82.3.5 的端到端主干变化抵消了其固定 Best 残差直接纠错。'
        '其旧 evidence 辅助语义目标和真实残差 difference 也分属不同路径。'
        '指定提交缺失的 history.py 已纳入新版本并校验输入身份。'
        f" [冻结设计与复现]({r.HERE/'design_review.md'}) · [论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)",
        '两组使用固定 16 维 U、ASD/TD Prompt、Null、有符号语义–LLR 融合和停止梯度的 J@U。'
        '增强特征为 z+δ。均使用历史平滑 8% 限幅和原分类头。'
        'L_inc 对两类等权的实际 CE(on)−CE(off)，以 0.25+0.75·sigmoid(1−m_off) 加权。'
        'L_protect 惩罚高 Margin 样本的 Margin 下降，系数 1。相对残差平方惩罚系数 0.001。'
        '语义单独 NLL 权重为零。G1 旁路共享梯度为零。G2 仅通过 δ(z) 向指定上游注入去冲突且受 5% 约束的增量梯度。',
        'seed=0，E61 epoch 10 原权重、三历史优化器和四类 RNG 逐折恢复。'
        '新增 AdamW lr=0.0005，weight decay=0.01。每折最多 450 epoch，wait>200 停止。'
        'Best 严格 ACC 提高且平局保留最早，Latest 每 10 轮及终止保存。先评估和选模，再用合法 CAM 行更新 EMA。'
        '第 377、407 行保留 Fold2 历史身份，排除新增标签监督、拟合和 EMA，残差为零。',
    ]
    if manifest.get('preferred_path_failure'):
        notes.append(f"首选 OSSFS 路径 I/O 错误 errno {manifest['preferred_path_failure']['errno']}，统一使用上述目录。")
    if status['aggregate']['metrics_10']:
        notes += ['## 正式结果',*report_tables(status)]
        notes += [line for g,m in status['aggregate']['mechanism'].items()
                  for line in mechanisms(g,m,[v for v in status['tasks'] if v['group']==g])]
    notes += ['## 审计与科研判定',audit_line(status),*science_text(status)]
    if status.get('handoff'):
        notes.append(f"Handoff 状态 `{status['handoff']['status']}`。")
    if status['failures']:
        notes.append(f"记录 {len(status['failures'])} 项失败。 [失败记录]({r.RESULTS/'audit_status.json'})")
    notes += [
        '性能和机制解释限于历史 val=test、CV 前 RFE/scaling、转导图、Fold2 重叠及单 seed。'
        '独立泛化结论需要独立选模及验证协议。',
        '下一轮是否评估更宽谱空间，应依据同权重直接纠错、困难病例 Margin 和谱覆盖共同判断。'
        '只凭低谱覆盖率无法确定增加维度可带来净纠错。',
    ]
    with h.atomic_writer(r.RESULTS/'report.md','w') as stream:
        stream.write('\n\n'.join(notes)+'\n')


def handoff_text(status):
    data = status['aggregate']
    sections = [
        '## Experiment 82.3.6 · PS-PGR',
        '**Prompt-guided Protected Gradient Residual Learning（疾病 Prompt 引导的受保护梯度残差学习）**',
        f"状态 **{status['status']}**，科研判定 **{status['science']['verdict']}**。{execution_line(status)}"
        f"物理 GPU0，seed 0。G2 `{status['g2_status']}`，匿名对照 `{status['anonymous_status']}`。",
        'G1 用原历史随机 DARG 更新和隔离的确定性 CAM144 残差训练。'
        'G2 按 G1 真正净纠错条件授权，共享梯度经过冲突投影并限幅为原梯度的 5%。'
        '训练目标使用真实开关残差的类均衡增量 CE 和保护项。沿用固定 16 维 U 与分类头雅可比。',
    ]
    if data['metrics_10']:
        sections += [metric_table(data['metrics_10']),paired_table(data['pairs'],'all')]
    for group,m in data['mechanism'].items():
        a,b,c = (count_changes(m[key]) for key in ('residual_off_vs_darg','direct_hard_contribution','full_vs_darg'))
        rows = m['patient_attribution']
        hard = len(rows['true_new_from_historical_wrong']['rows_1based'])
        rescue = len(rows['off_error_recovered_from_historical_correct']['rows_1based'])
        harmed = len(rows['new_residual_harm_to_historical_correct']['rows_1based'])
        wrong_margin = m['margin']['darg_wrong']['change']['mean']
        sections.append(f"{group} A/B/C 纠错/误伤/净分别为 {a}、{b}、{c}。"
                        f"真正新增 {hard}，救回 {rescue}，直接新误伤 {harmed}。历史错分患者的残差 Margin 均值变化 {wrong_margin:+.7f}。"
                        f"残差均值 {m['residual_relative_pct']['mean']:.5f}%，谱覆盖均值 {100*m['classifier_spectral_coverage']['mean']:.3f}%。")
    sections += [audit_line(status),*science_text(status),
                 'Fold2 第 377、407 行保留历史身份，排除新增有标签处理，残差为零。',
                 '后续谱扩展需优先依据同权重确定性直接收益和困难病例表现。使用独立选模协议验证泛化。',
                 f"实验目录 `{r.HERE}`。 [冻结设计]({r.HERE/'design_review.md'}) · "
                 f"[结果]({r.RESULTS/'report.md'}) · [审计]({r.RESULTS/'audit_status.json'})"]
    if status['failures']:
        sections.append('首项阻塞为 '+status['failures'][0].get('detail','').strip().splitlines()[-1])
    return '\n\n'.join(sections)+'\n'


def write_tables(status):
    from audit import independent_metrics, paired, same_metrics

    records = status['references'] + status['tasks']
    if not records:
        return
    individual = [dict(group=v['group'],fold=v['fold'],best_epoch=v['best_epoch'],end_epoch=v['end_epoch'],
                       **{k:100*v['metrics'][k] for k in r.METRICS}) for v in records]
    path = r.RESULTS/'fold_metrics.csv'
    h.write_csv(path, individual)
    from_csv = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            from_csv[row['group'],int(row['fold'])] = row
    assert len(from_csv) == len(records)

    import numpy as np
    index = {(v['group'],v['fold']):v for v in records}
    for v in records:
        row = from_csv[v['group'],v['fold']]
        calc = independent_metrics(np.asarray(v['probability'],dtype=np.float32), np.asarray(v['labels']))
        same_metrics(calc, {k:float(row[k])/100 for k in r.METRICS})
        assert (int(row['best_epoch']),int(row['end_epoch'])) == (v['best_epoch'],v['end_epoch'])
    rows = []
    for pair, panel in status['aggregate']['pairs'].items():
        for item in status['aggregate']['per_fold_pairs'][pair]:
            rows.append(dict(comparison=pair,scope='fold',fold=item['fold'],**item['delta_pp'],**item['patients']))
        for scope,data in panel.items():
            wins = {f"{metric}_win_tie_loss": "/".join(map(str,result))
                    for metric,result in data['win_tie_loss'].items()}
            rows.append(dict(comparison=pair,scope=scope,fold='',
                             **data['delta_pp'],**data['patients'],**wins))
    if rows:
        path = r.RESULTS/'paired_comparisons.csv'
        h.write_csv(path, rows)
        with path.open(newline='') as f:
            replay = list(csv.DictReader(f))
        assert len(replay) == len(rows)
        for row, expected in zip(replay, rows):
            assert (row['comparison'], row['scope']) == (expected['comparison'], expected['scope'])
            assert all(float(row[k]) == expected[k] for k in r.METRICS)
            if row['scope'] == 'fold':
                left,right = row['comparison'].split('-')
                fld = int(row['fold'])
                check = paired(index[left,fld],index[right,fld])
                assert all(int(row[k]) == check['patients'][k] for k in ('fixed','harmed','net','asd_fixed','td_fixed'))
    status['tables_recomputed_and_verified'] = True


def append_handoff(status, manifest):
    marker = '## Experiment 82.3.6 · PS-PGR'.encode()
    section = handoff_text(status).encode()
    before = manifest['handoff_before']
    try:
        with HANDOFF.open('r+b') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            old = f.read()
            assert hashlib.sha256(old[:before['size']]).hexdigest() == before['sha256'], 'Existing handoff prefix changed'
            if marker in old:
                assert section in old and old.count(marker) == 1, 'Existing section differs'
            else:
                f.seek(0, os.SEEK_END)
                f.write(b'\n\n'+section)
                f.flush()
                os.fsync(f.fileno())
            f.seek(0)
            reread = f.read()
            assert reread[:len(old)] == old and section in reread and reread.count(marker) == 1
        assert HANDOFF.read_bytes() == reread
        return dict(status='APPENDED_VERIFIED',path=str(HANDOFF),prefix_unchanged=True,
                    sha256=hashlib.sha256(reread).hexdigest(),section_sha256=hashlib.sha256(section).hexdigest())
    except (OSError, AssertionError) as exc:
        path = r.RESULTS/'handoff_pending.md'
        with h.atomic_writer(path,'wb') as f:
            f.write(section)
        return dict(status='BLOCKED',path=str(HANDOFF),pending=str(path),error=repr(exc))

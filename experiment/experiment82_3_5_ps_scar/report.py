"""Tables, scientific report and append-only handoff for PS-SCAR."""
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
    p = status['protection']
    rng = status['rng']
    completed = sum(v['completed_folds'] for v in status['group_execution'].values())
    match = '全部一致' if p['pass'] else f"有 {len(p['mismatches'])} 项不一致或无法读取"
    return (f"Preflight {status['preflight_pass']}/10，Best {status['best_pass']}/{completed}，"
            f"Latest {status['latest_pass']}/{completed}。历史共同 RNG {rng['common_darg_rng_epochs']} 轮，"
            f"延伸连续链 {rng['extended_rng_epochs']} 轮，组间共同 RNG {rng['paired_g1_rng_epochs']} 轮，"
            f"逐项匹配历史末轮 RNG 的折数 {rng['historical_terminal_stream_matches']}。"
            f"EMA 独立重放 {rng['ema_updates']} 次，保护 SHA-256 {p['count']} 项{match}。")


def science_text(status):
    s = status['science']
    check_names = {'acc':'ACC','ba':'BA','prob_auc':'Prob-AUC','sen':'SEN-ASD','patient_net':'总体净纠错',
                   'direct_hard_net':'直接净硬纠错','nonzero_finite_residual':'有效残差',
                   'endpoint_gradients':'新增模块梯度','llr_sign_override':'LLR 符号仲裁','logit_gap_change':'Logit gap 变化'}
    text = []
    for g,v in s['arms'].items():
        if not v['audited']:
            text.append(f'{g} 审计未完成。')
            continue
        p = '通过' if v['performance_go'] else '未通过'
        m = '通过' if v['mechanism_go'] else '未通过'
        missing = [check_names[k] for k,ok in {**v['performance_checks'],**v['mechanism_checks']}.items() if not ok]
        text.append(f"{g} Performance GO {p}，Mechanism GO {m}。" +
                    (f"未达到的条件包括{'、'.join(missing)}。" if missing else ''))
    if s['medical_advantage'] is True:
        text.append('配对匿名 Query 重训练支持该被选组在本历史协议下的医学 Query 增量。')
    elif s['medical_advantage'] is False:
        text.append('配对匿名 Query 重训练未支持独立医学 Query 优势。')
    elif status['anonymous_status']=='SKIPPED_PREREGISTERED':
        text.append('医学组均未同时达到双门槛，匿名 Query 正式对照按预注册跳过。')
    else:
        text.append(f"匿名对照状态为 {status['anonymous_status']}。")
    return text


def mechanism_notes(name, m, tasks):
    d, rho = m['residual_relative_pct'],m['classifier_spectral_coverage']
    margins = m['margin']
    c = m['fusion_counts']
    direct = m['direct_hard_contribution']
    backbone = m['residual_off_vs_darg']
    t1,t4 = (m['treatments'][k] for k in ('T1','T4'))
    def cosine(value):
        return f'{value:+.5f}' if value is not None else '—'
    results = [
        f'### {name} 固定 Best',
        f"**A 历史 DARG → residual-off** 纠错/误伤/净 {count_changes(backbone)}。"
        f"**B residual-off → full** 为 {count_changes(direct)}，硬翻转 {direct['flips']}。"
        'A、B 的带标签诊断均使用 869 人，C 的历史十折总体配对使用 871 人。',
        f"实际 CAM 残差在 {m['nonzero_residual_rows']} 人非零，平均/P95/最大为 "
        f"{d['mean']:.5f}%/{d['p95']:.5f}%/{d['max']:.5f}%。"
        f"关闭残差后 ASD 概率绝对变化均值 {t1['probability_change']['mean']*100:.5f} 个百分点，"
        f"Logit gap 平均绝对变化 {m['logit_gap_abs_change']['mean']:.7f}。",
        f"固定谱空间平均敏感度覆盖率 {rho['mean']*100:.3f}%，P95 {rho['p95']*100:.3f}%。"
        f"J 与实际残差的平均有符号余弦 {m['classifier_cosine']['mean']:+.6f}。"
        f"一阶预测与实际 gap 差的 Pearson 为 {cosine(m['first_order_actual_pearson'])}，"
        f"相对残差与绝对 gap 差的 Spearman 为 {cosine(m['residual_gap_abs_spearman'])}。",
        table(('诊断患者分层','人数','真类 Margin 平均变化','正向比例'),
              [(label,margins[k]['n'],
                f"{margins[k]['change']['mean']:+.7f}" if margins[k]['n'] else '—',
                f"{100*margins[k]['fraction_positive']:.2f}%" if margins[k]['n'] else '—')
               for k,label in [('all','全部'),('darg_correct','历史 DARG 正确'),('darg_wrong','历史 DARG 错误'),
                               ('low_margin','Residual-off Margin < 2'),('high_margin','Residual-off Margin ≥ 2'),
                               ('asd','ASD'),('td','TD')]]),
        f"ASD/TD 原始语义差异平均 L2 {m['raw_semantic_l2']['mean']:.6f}。"
        f"{c['elements']} 个患者方向中，语义与 LLR 同号 {c['agree']} 个、冲突 {c['conflict']} 个。"
        f"统计支路覆盖语义符号 {c['llr_overrides']} 个，占冲突方向 {100*m['llr_override_rate_among_conflicts']:.2f}%。"
        f"固定 Best 替换 Query 使 {m['query_swap_direction_flips']} 个融合方向变号，"
        f"融合证据逐方向平均绝对变化 {m['query_swap_d_abs_mean']:.6f}，硬预测翻转 {t4['versus_full']['flips']} 人。",
        table(('折','原始语义 L2','LLR 均值 ± SD','LLR 正/负','一致/冲突/LLR翻转','α均值','ASD / TD Null 均值','可靠性均值','Query替换证据变化','Query替换硬翻转'),
              [(v['fold'],f"{sum(q['raw_semantic_l2'])/len(q['raw_semantic_l2']):.5f}",
                f"{q['llr_distribution']['mean']:+.4f} ± {q['llr_distribution']['sd']:.4f}",
                f"{q['fusion_counts']['llr_positive']}/{q['fusion_counts']['llr_negative']}",
                '/'.join(str(q['fusion_counts'][key]) for key in ('agree','conflict','llr_overrides')),
                f"{q['alpha_distribution']['mean']:.4f}",
                f"{q['null_asd']['mean']:.4f} / {q['null_td']['mean']:.4f}",f"{q['reliability']['mean']:.4f}",
                f"{sum(q['query_swap_d_abs'])/len(q['query_swap_d_abs']):.5f}",q['treatments']['T4']['versus_full']['flips'])
               for v in tasks for q in (v['interventions'],)]),
        table(('干预',*LABELS,'ASD 概率变化 pp','硬翻转','ASD纠错/误伤/净','TD纠错/误伤/净'),
              [(k,*(f"{v['metrics'][metric][0]:.4f}" for metric in r.METRICS),
                f"{100*v['probability_change']['mean']:.6f}",v['versus_full']['flips'],
                count_changes(v['versus_full'],'asd_'),count_changes(v['versus_full'],'td_'))
               for k,v in m['treatments'].items() if k in ('T0','T1','T4')]),
        'T0 为 full，T1 为 residual-off，T4 为固定 Best 即时替换 Query。T1、gate-off 和 evidence-off 逐元素结构性等价。'
        '干预表以 T0 为参照，使用去除 Fold2 两条保护行后的带标签范围。T4 是即时干预。',
        f"实际合计辅助梯度比例最大为 {m['max_actual_aux_ratio']:.8f}。"
        f"语义损失首轮、Best 轮、末轮均值为 {m['semantic_loss_first']:.6f}、"
        f"{m['semantic_loss_at_best']:.6f}、{m['semantic_loss_latest']:.6f}。"
        f"反事实梯度最大范数为 {m['max_counterfactual_gradient']:.6f}，"
        f"反事实损失首轮、Best 轮、末轮均值为 {m['cf_loss_first']:+.7f}、"
        f"{m['cf_loss_at_best']:+.7f}、{m['cf_loss_latest']:+.7f}。",
    ]
    return results


def write_report(status,manifest):
    a = status['aggregate']
    science = status['science']
    workers = max((v['max_workers'] for v in status['group_execution'].values()),default=0)
    peak = max((v['peak_sampled_gpu0_mib'] for v in status['group_execution'].values()),default=0)
    notes = [
        '# Experiment 82.3.5 · PS-SCAR',
        '**Prompt-guided Signed and Classifier-Aware Residual Learning（疾病 Prompt 引导的有符号分类感知特征残差学习）**',
        f"状态 **{status['status']}**，科研判定 **{science['verdict']}**。"
        f"目录 `{r.HERE}`，物理 GPU0，最多 {workers} 个 worker，显存采样峰值 {peak} MiB。"
        f"耗时 {status['elapsed_seconds']:.1f} 秒。{execution_line(status)}",
        '## 历史边界',
        '原 SPromptGL 用 GFR 关系图和 GCN 汇聚患者信息，CAM 在患者内做跨模态交互，PEN 用固定 CLIP 医学语义重加权局部 token 并融合全局特征。'
        '本实验使用 E74.4-G2 的 GFR、DARG 关系机制及 PLNNoPEN。CAM 展平为 871×144 后进入原 LayerNorm、Linear、GELU、Dropout、Linear 分类头。'
        'PS-SCAR 只在两者之间注入特征残差。 [原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · '
        '[原作者源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a)',
        'E82.3.2 的统计权重可调但残差约 0.064%。E82.3.3 学到有符号标量门控和非零残差，两组关闭残差仍零硬翻转。'
        'E82.3.4 的逐方向残差平均 0.782%，DARG 已错患者真类 Margin 平均下降 0.00351，且 LLR 非负调制无法改变单方向语义符号。'
        '现场 E82.3.4 的实现、运行器、配置、审计和报告与所给提交的 Git blob 一致。 '
        '[E82.3.4 提交](https://github.com/tangaii/SPromptGL/commit/1e964eb0ee028c11afc2071edd79231778f35068)',
        '## 冻结设计',
        '保留 E82.2 固定 16 维正交 U、方向规范符号和历史谱白化，使用原八条文本、谱可靠性及 Gaussian EMA。'
        '缓存前四条为 TD，后四条为 ASD。标签轴为 ASD=0、TD=1，Gaussian LLR 正值偏 ASD。',
        '两类独立谱覆盖经过各自 Null 置信度修正，得到原始差异 S。s = tanh(S/τ_s)，l = tanh(LLR/τ_l)。'
        '按各折合法 E61 epoch10 CAM 的患者方向绝对值 P75 拟合两个技术尺度，下限为 0.005 和 0.25。'
        '有界患者方向融合 α 在 [0.2,0.8] 内，D = R·(1−平均 Null)·[α·s+(1−α)·l]。'
        '因此 LLR 与语义冲突时可改变最终符号，α 的固定下限保留显式语义通路。',
        '原分类头的确定性辅助路径复用真实参数并绕过 Dropout，以输入叶子求 ASD−TD gap 的 J，不累积头参数梯度。'
        '原训练前向保留历史随机 Dropout。p=J@U，p̂=p/sqrt(∥p∥²+1e−12)，J 停止梯度。'
        'g=tanh(clamp_min(MLP([LN(z),D,R,Null]),0))。末层精确零初始化，clamp_min 在零点的一侧梯度允许首步学习。'
        'δ=smooth_limit((g·D·p̂)@Uᵀ,z)，保持 8% 上限。每个谱方向的一阶 gap 贡献与 g·D·p² 同号，多个方向可相互抵消。',
        '两组均使用 DARG 全损失 cls+graph+0.05·pair+0.01·adv+0.01·preserve，以主分类梯度端到端更新。'
        '保留 E82.3.4 类均衡语义 NLL（0.01）和残差安全项（0.001）。G2 另加 0.005 的选择性反事实项。'
        '在固定当前确定性头上测量 Δ=真类 Margin_on−Margin_off，off 停止梯度。'
        '困难权重为 0.05+0.9·sigmoid(1−off)。目标 0.1、温度 0.1 的 softplus 收益缺口逐渐饱和正收益，'
        '另加 sigmoid(off−2)·ReLU(−Δ)² 的保护项。该目标直接使用实际开关差值。'
        '辅助梯度只注入新增参数，逐参数组以 0.5 预算限制合计梯度。主损失更新保持原路径。',
        '两医学组同折恢复 E61 epoch10 对应 DARG 主干、CAM、头、三个历史优化器和四类 RNG。'
        '继承谱–语义参数隔离种子 822000+fold，融合与门控分别用 823600+fold 和 823700+fold。'
        '新增 AdamW 的 lr=0.0005，weight decay=0.01。固定十折 seed 0，最多 450 epoch，wait>200 停止。'
        '每十轮及终止保存 Latest，严格测试 ACC 提高时冻结 Best，平局保留更早。以更新前 EMA 评估，随后更新 EMA α=0.1。'
        'Fold2 行 377、407 保留历史训练和测试身份，排除新增监督、统计、EMA 和带标签诊断，残差严格为零。',
    ]
    if manifest.get('preferred_path_failure'):
        notes.append(f"首选目录发生 OSSFS errno {manifest['preferred_path_failure']['errno']}，本轮统一使用回退目录。")
    probe = status.get('preflight_head_probe',{})
    if probe:
        notes += ['## 独立 Preflight 分类敏感性',
                  '使用每折只读历史 DARG Best 和固定 0.01 的局部谱方向探针。有限差分面板取合法训练中每类前四行。'
                  '原生 FP32 导数独立逐元素复算，FP64 闭式链式导数与 FP64 autograd 核对。历史探针不会更新模型。',
                  table(('折','合法训练覆盖率','OOF覆盖率','探针相对范数','平均绝对Gap变化','一阶误差最大'),
                        [(fold,f"{v['rho_training']['mean']:.5f}",f"{v['rho_oof']['mean']:.5f}",
                          f"{v['probe_cam_ratio_pct']['mean']:.5f}%",f"{v['probe_gap_abs_change']['mean']:.7f}",
                          f"{v['probe_linearization_max_error']:.2e}") for fold,v in probe.items()])]
    if a['metrics_10']:
        notes += ['## 正式结果',*report_tables(status)]
    if a['mechanism']:
        notes += ['## 机制对照',*(line for g,m in a['mechanism'].items()
                                  for line in mechanism_notes(g,m,[v for v in status['tasks'] if v['group']==g]))]
    notes += ['## 审计与科研判定',audit_line(status),*science_text(status)]
    if status.get('handoff'):
        notes.append(f"handoff 状态 `{status['handoff']['status']}`。")
    if status['failures']:
        notes.append(f"记录 {len(status['failures'])} 项执行或审计失败。 [失败记录]({r.RESULTS/'audit_status.json'})")
    notes += [
        '正式结果使用同一历史转导患者图、CV 前 RFE/scaling、单 seed 和 val=test 选模。此协议下的即时 Query 效果不能确定医学文本的独立增量。',
        '如面向独立验证继续该方向，应把历史错分患者的直接净硬纠错和对正确患者的保护作为筛选条件，并报告严格外部选模协议。',
    ]
    content = '\n\n'.join(notes)+'\n'
    path = r.RESULTS/'report.md'
    with h.atomic_writer(path,'w') as f:
        f.write(content)
    assert path.read_text()==content


def handoff_text(status):
    a = status['aggregate']
    sections = [
        '## Experiment 82.3.5 · PS-SCAR',
        '**Prompt-guided Signed and Classifier-Aware Residual Learning（疾病 Prompt 引导的有符号分类感知特征残差学习）**',
        f"状态 **{status['status']}**，科研判定 **{status['science']['verdict']}**。"
        f"{execution_line(status)}物理 GPU0，seed 0。匿名组状态 `{status['anonymous_status']}`。",
        'G1 以两类医学 Query、Gaussian LLR 有符号融合和分类头雅可比构造 CAM144 残差。'
        'G2 增加困难患者选择性真实 Margin 反事实监督。两组完整恢复 DARG 独立端到端训练，使用同一原分类头。',
    ]
    if a['metrics_10']:
        sections += [metric_table(a['metrics_10']),paired_table(a['pairs'],'all')]
    for g,m in a['mechanism'].items():
        wrong = m['margin']['darg_wrong']['change']
        margin = f"{wrong['mean']:+.7f}" if wrong else '—'
        sections.append(f"{g} 残差均值 {m['residual_relative_pct']['mean']:.5f}%，"
                        f"LLR 改变语义符号 {m['fusion_counts']['llr_overrides']} 个方向。"
                        f"直接硬纠错/误伤/净 {count_changes(m['direct_hard_contribution'])}，"
                        f"历史错误患者 Margin 平均变化 {margin}。"
                        f"即时匿名 Query 替换硬翻转 {m['treatments']['T4']['versus_full']['flips']} 人。")
    sections += [audit_line(status),*science_text(status),
                 'Fold2 行 377 与 407 保留历史身份，排除新增带标签机制，残差为零。',
                 '后续价值取决于困难患者能否产生直接硬纠错及独立选模重现。',
                 f"实验目录 `{r.HERE}`。 [模型]({r.HERE/'ps_scar.py'}) · [报告]({r.RESULTS/'report.md'}) · [审计]({r.RESULTS/'audit_status.json'})"]
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
    marker = '## Experiment 82.3.5 · PS-SCAR'.encode()
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

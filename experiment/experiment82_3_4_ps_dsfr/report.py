"""Tables, scientific report and append-only handoff for PS-DSFR."""
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
    return '；'.join(f"{g} {v['completed_folds']}/10 折，训练墙钟 {v['wall_seconds']:.1f} 秒"
                    for g,v in saved.items()) + '。'


def mechanism_notes(name, d):
    gate, res = d['gates'], d['residual_relative_pct']
    m = d['margin']
    t1, t4 = (d['treatments'][key] for key in ('T1','T4'))
    p = d['direct_hard_contribution']
    s = d['residual_off_vs_darg']
    subset = m['all']
    spearman = '—' if m['spearman_vs_off'] is None else f"{m['spearman_vs_off']:+.4f}"
    return [
        f"### {name} 固定 Best 机制",
        f"谱覆盖 ASD−TD 的 OOF 平均 L2 为 {d['attention_difference_l2']['mean']:.6f}，最终差异证据平均 L2 "
        f"{d['evidence_difference_l2']['mean']:.6f}，P95 {d['evidence_difference_l2']['p95']:.6f}。"
        f"方向门控正/负/零 {gate['positive']}/{gate['negative']}/{gate['zero']}。"
        f"实际残差有 {d['nonzero_residual_rows']} 名患者非零，相对范数均值 {res['mean']:.5f}%，"
        f"P95 {res['p95']:.5f}%，最大 {res['max']:.5f}%。",
        f"完整模型相对 residual-off 的直接硬分类贡献为 {p['fixed']} 例纠错、{p['harmed']} 例误伤，净 {p['net']:+d}。"
        f"关闭残差的 ASD 概率绝对变化均值 {t1['probability_change']['mean']:.7f}，"
        f"最大 {t1['probability_change']['max']:.7f}，硬翻转 {t1['versus_full']['flips']} 例。"
        f"训练后主干在 residual-off 下相对 DARG 纠错 {s['fixed']} 例、误伤 {s['harmed']} 例，净 {s['net']:+d}。",
        f"真类 Margin 在 {subset['n']} 名诊断患者上平均变化 {subset['change']['mean']:+.7f}，"
        f"正向比例 {100*subset['fraction_positive']:.2f}%。它与关闭残差的真类 Margin 的 Spearman 为 {spearman}。",
        table(('Margin 分层','患者数','均值变化','正向比例'),
              [(label,m[k]['n'],f"{m[k]['change']['mean']:+.7f}" if m[k]['n'] else '—',
                f"{100*m[k]['fraction_positive']:.2f}%" if m[k]['n'] else '—')
               for k,label in [('asd','ASD'),('td','TD'),('darg_correct','历史 DARG 正确'),
                               ('darg_wrong','历史 DARG 错误'),('low_margin','Residual-off Margin < 2'),
                               ('high_margin','Residual-off Margin ≥ 2')]]),
        table(('固定 Best 干预',*LABELS,'概率绝对变化均值','最大','硬翻转','ASD 纠错/误伤/净','TD 纠错/误伤/净'),
              [(tag, *(f"{v['metrics'][k][0]:.4f}" for k in r.METRICS),
                f"{v['probability_change']['mean']:.7f}", f"{v['probability_change']['max']:.7f}",
                v['versus_full']['flips'], count_changes(v['versus_full'],'asd_'), count_changes(v['versus_full'],'td_'))
               for tag,v in d['treatments'].items() if tag in ('T0','T1','T4')]),
        'T1 关闭残差、T2 关闭方向门控和 T3 关闭医学差异逐元素结构性等价。上表干预计数以 T0 为参照，'
        '两名 Fold2 历史重叠身份从全部带标签机制统计中排除。T4 的分数来自同一个 Best 上的即时 Query 替换。',
        f"固定 Best 替换 Query 后有 {t4['versus_full']['flips']} 例硬翻转。"
        f"语义辅助损失首轮、Best 轮和末轮十折均值分别为 {d['mean_sem_loss_first']:.6f}、"
        f"{d['mean_sem_loss_at_best']:.6f}、{d['mean_sem_loss_latest']:.6f}。"
        f"实际辅助梯度/主分类梯度最大比为 {d['max_actual_aux_ratio']:.6f}。",
    ]


def science_text(s):
    if 'performance_checks' not in s:
        return s['evidence_reason']
    p = '通过' if s['performance_go'] else '未通过'
    m = '通过' if s['mechanism_go'] else '未通过'
    bad_p = {'best_and_latest_10':'十折审计','acc':'ACC','ba':'BA','prob_auc':'Prob-AUC','sen':'SEN-ASD','patient_net':'患者净纠错'}
    bad_m = {'endpoint_gradients':'端到端梯度','nonzero_evidence':'类别差异证据','nonzero_residual':'有效残差',
             'direct_hard_net':'固定 Best 净硬纠错'}
    pf = '、'.join(bad_p[k] for k,v in s['performance_checks'].items() if not v)
    mf = '、'.join(bad_m[k] for k,v in s['mechanism_checks'].items() if not v)
    if s['medical_advantage'] is None:
        sem = '医学 Query 的独立贡献尚未建立。'
    elif s['medical_advantage']:
        sem = '在本历史协议下，医学 Query 达到了预注册的匿名重训练配对优势标准。'
    else:
        sem = '匿名重训练对照未支持医学 Query 的独立性能优势。'
    return (f'Performance GO {p}。' + (f'缺失条件为 {pf}。' if pf else '') +
            f'Mechanism GO {m}。' + (f'缺失条件为 {mf}。' if mf else '') + sem)


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


def write_report(status, manifest):
    a = status['aggregate']
    s = status['science']
    workers = max((v['max_workers'] for v in status['group_execution'].values()), default=0)
    peak = max((v['peak_sampled_gpu0_mib'] for v in status['group_execution'].values()), default=0)
    storage = manifest['preferred_path_failure']
    notes = [
        '# Experiment 82.3.4 · PS-DSFR',
        '**Prompt-guided Discriminative Spectral Feature Residual Learning（疾病 Prompt 引导的判别性谱特征残差学习）**',
        f"状态 **{status['status']}**，科研判定 **{s['verdict']}**。G2 状态 `{status['g2_status']}`。",
        f"实际目录 `{r.HERE}`。物理 GPU0 单卡，最多 {workers} 个 worker，采样峰值 {peak} MiB。"
        f"从创建实验目录至审计完成 {status['elapsed_seconds']:.1f} 秒。{execution_line(status)}",
        '## 研究问题与历史核对',
        'E82.3.3 在固定 Best 下已观察到 2.63%–3.32% 的相对残差和平均真类 Margin 上升，两组关闭残差均零硬翻转，'
        '十折 ACC 均低于 DARG。本轮检验同一患者两类医学 Query 的竞争性谱证据，能否在原 CAM144 上形成有分类价值的方向级修正。',
        'SPromptGL 的 GFR 用模态关系图和 GCN 表示患者。CAM 在每个患者内融合模态。原 PEN 用冻结 CLIP、SPGS 局部 token '
        '加权和 RES 局部到全局融合。DARG 路径延续 PLNNoPEN，CAM 由 FeedForward 后展平得到 871×144，送入原 '
        'LayerNorm→Linear→GELU→Dropout→Linear 分类头。 [原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · '
        '[原作者代码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a)',
        '服务器现存 E82.3.3 的模型、运行器、审计、配置、报告与所给提交的 Git blob 逐项一致。'
        'E74.4-G2 通过 E74.1 和 E69.3 恢复 E61 epoch 10 的 CAM、主干、校准和三个优化器；'
        'E82.2 的基包含 8 个全局方差方向和其正交补的 8 个类协方差差异方向，最大绝对载荷取正。'
        'E82.3.1–E82.3.3 的合法训练统计、历史方差稳定化和 EMA 时序均保留。'
        '固定文本前四条属于 TD，后四条属于 ASD。两个 Query 均作用于每位患者。'
        '本目录内固定训练所需的历史缓存适配与数值包。 [历史提交](https://github.com/tangaii/SPromptGL/tree/784f567de5d85bed452748a5015b44e59928968b/experiment/experiment82_3_3_pa_dgpl)',
        '## 方法和冻结协议',
        'E82.3.3 的标量 SignedGate、beta 混合与一阶 DecisionGain 已移除。PS-DSFR 将残差加在 CAM144 上，'
        '保留原始分类头和主分类损失的端到端梯度。',
        r'- $h=zU$ 承载有符号残差，$\hat h=(z-\mu)U/\sqrt{\lambda+\epsilon}$ 供历史 Gaussian 和谱 token 使用。'
        r'固定 $U$ 的列、符号和 dtype 沿用历史身份。' '\n'
        r'- $A_c=1-\prod_{j\in c}(1-a_{j,1:16})$，$N_c$ 为各类 Null 均值，'
        r'$C_c=(1-N_c)A_c/(\sum A_c+\epsilon)$。$S=C_{ASD}-C_{TD}$。'
        r'用历史有界 ASD/TD Gaussian LLR $\ell$ 和可靠性 $R=(1+\tanh(16w_{sp}))/2$ 构造'
        r' $E_c=C_cR\sigma(\ell\tanh(16S))$，$D=E_{ASD}-E_{TD}$。' '\n'
        r'- $g=\tanh(\mathrm{MLP}_{178\to16\to16}([\mathrm{LN}(z),D,R,N_{ASD},N_{TD}]))$，末层权重和偏置精确置零。'
        r'$\delta=\mathrm{smooth\_limit}((g\odot D\odot h)U^T,z)$，使用原 8% 径向 tanh 限幅，并对极小范数保证真实相对限幅。'
        r'原分类头使用 $z+\delta$。' '\n'
        '- $L_{DARG}=L_{cls}+L_{graph}+0.05L_{pair}+0.01L_{adv}+0.01L_{preserve}$。'
        '$L_{sem}$ 是两类证据总量归一化后的类均衡 NLL；$L_{safe}$ 按 '
        r'$\sigma(\mathrm{stopgrad}(m_y)-2)$ 连续加权合法训练行的相对残差平方。'
        '$L=L_{DARG}+s(0.01L_{sem}+0.001L_{safe})$，$s$ 使用冻结的逐组梯度范数预算 0.5。'
        '这两个辅助项只向新增参数注入梯度。',
        'G1 用固定医学 Query，G2 用几何匹配的历史匿名 Query。历史 6081 个参数使用 822000+fold 的原初始化。'
        '新门控使用 823500+fold 的隔离初始化。固定文本向量和谱基；学习 Query 条件化的 direction、evidence、'
        'LayerNorm、key、Null 与谱可靠性网络。新增 AdamW 的 lr=0.0005、weight decay=0.01；三个历史优化器完整恢复。',
        'seed=0，历史十折，val=test，严格 ACC 提高才更新 Best，平局保留较早 epoch。'
        '从 epoch 10 恢复，最多 450 epoch，wait>200 停止，每十轮及终止保存 Latest，先使用旧 EMA 评估与冻结 Best，'
        '再更新 EMA（α=0.1）并保存 Latest。沿用 CV 前 RFE/scaling 和转导图。'
        'Fold2 的行 377 和 407 保留历史训练和测试身份，排除新增 EMA/监督和带标签机制诊断，新增残差严格为零。',
    ]
    if storage:
        notes.append(f"首选 OSSFS 路径创建失败（errno {storage['errno']}），所有正式产物写入同一回退目录。")
    if a['metrics_10']:
        notes += ['## 正式结果', *report_tables(status)]
    if a['mechanism']:
        notes += ['## 机制审计', *(item for g,m in a['mechanism'].items() for item in mechanism_notes(g,m))]
    notes += ['## 独立审计与科研决定', audit_line(status), science_text(s)]
    if status['g2_status'] == 'SKIPPED_PREREGISTERED':
        notes.append('G1 未同时达到两项预注册门槛，G2 正式训练按协议跳过。')
    if status['failures']:
        notes.append(f"有 {len(status['failures'])} 项执行或独立审计失败。 [审计详情]({r.RESULTS/'audit_status.json'})")
    if 'handoff' in status:
        notes.append(f"handoff 状态为 `{status['handoff']['status']}`。")
    notes += ['本轮沿用单 seed、历史测试集选模、CV 前处理、转导图及 Fold2 重叠。结果限于该历史协议。'
              '注意力差异表明 Query 对谱证据有影响，医学病理解释需要额外验证。',
              '如需推进 MICCAI，建议把患者级净硬纠错和 ASD 灵敏度作为优先指标，并在独立选模协议中复验。'
              '匿名 Query 的独立重训练仍是检验医学语义贡献所需的配对证据。']
    content = '\n\n'.join(notes) + '\n'
    path = r.RESULTS / 'report.md'
    with h.atomic_writer(path, 'w') as f:
        f.write(content)
    assert path.read_text() == content


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


def handoff_text(status):
    a = status['aggregate']
    paragraphs = [
        '## Experiment 82.3.4 · PS-DSFR',
        '**Prompt-guided Discriminative Spectral Feature Residual Learning（疾病 Prompt 引导的判别性谱特征残差学习）**',
        f"状态 **{status['status']}**，科研判定 **{status['science']['verdict']}**。"
        f"{execution_line(status)}G2 状态 `{status['g2_status']}`。物理 GPU0 单卡，seed 0。",
        '动机来自 E82.3.3 的零残差硬翻转与低于 DARG 的十折准确率。本轮用独立 ASD/TD 谱注意力、Null、'
        '历史 Gaussian LLR 和可靠性构造 16 维差异证据，替代单标量门控、beta 和 DecisionGain。'
        '方向级有符号门控将有界残差加到 CAM144，然后经原 DARG 分类头。G1 使用医学 Query；'
        'G2 使用几何匹配的历史匿名 Query，只有 G1 双门槛通过才启动。两组保留 DARG 端到端训练与原选模协议。',
    ]
    if a['metrics_10']:
        paragraphs.extend([metric_table(a['metrics_10']),paired_table(a['pairs'],'all')])
    for name,d in a['mechanism'].items():
        direct = d['direct_hard_contribution']
        paragraphs.append(f"{name} OOF 差异证据平均 L2 {d['evidence_difference_l2']['mean']:.6f}，"
                          f"实际相对残差 {d['residual_relative_pct']['mean']:.5f}%。"
                          f"直接硬分类纠错/误伤/净 {count_changes(direct)}，"
                          f"平均真类 Margin 变化 {d['margin']['all']['change']['mean']:+.7f}。"
                          f"T1/T2/T3 结构性等价，固定 Best Query 替换翻转 {d['treatments']['T4']['versus_full']['flips']} 例。")
    paragraphs += [audit_line(status),
                   'Fold2 行 377 与 407 保留历史 DARG 身份，排除新增统计、监督与带标签机制诊断，残差为零。',
                   science_text(status['science']),
                   '后续若复验，建议优先检验独立选模下的净硬纠错和 ASD 灵敏度，并为医学语义归因保留匿名 Query 配对重训练。',
                   f"实验目录 `{r.HERE}`。 [正式报告]({r.RESULTS/'report.md'}) · [独立审计]({r.RESULTS/'audit_status.json'})"]
    if status['failures']:
        concise = ' '.join(status['failures'][0].get('detail','').strip().splitlines()[-2:])
        paragraphs.append(f"首项执行阻塞为 `{concise}`。")
    return '\n\n'.join(paragraphs)+'\n'


def append_handoff(status, manifest):
    marker = '## Experiment 82.3.4 · PS-DSFR'.encode()
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

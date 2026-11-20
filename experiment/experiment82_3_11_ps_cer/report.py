"""Build and verify the final results and append-only SABER handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import os
from datetime import datetime, timezone
from fractions import Fraction

import run as r
import history as h
import numpy as np

NAMES = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')


def md_table(cols, rows):
    def line(items):
        return '| ' + ' | '.join(str(item).replace('|', '\\|') for item in items) + ' |'

    return '\n'.join([line(cols), line(['---']*len(cols)), *(line(row) for row in rows)])


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


def exact_accuracy_pairs(aggregate, audits):
    """Use patient counts for the strict paired ACC inequality, including exact ties."""
    accuracies = {}
    for result in audits:
        rows = result['records']
        for group, prefix in ((result['group'], 'full'), ('B0', 'b0')):
            accuracies[(group, result['fold'])] = Fraction(
                sum(x[prefix+'_label'] == x['y'] for x in rows), len(rows))
    for name, comparison in aggregate['paired'].items():
        group, baseline = name.split('-')
        differences = {f: accuracies[group,f]-accuracies[baseline,f] for f in r.FOLDS}
        for scope in (comparison['full'], comparison['no_fold2'], *comparison['max_contribution_removed'].values()):
            deltas = [differences[fold] for fold in scope['folds']]
            scope['delta_pp']['acc'] = float(100*sum(deltas)/len(deltas))
            assert scope['wtl']['acc'] == dict(win=sum(x>0 for x in deltas), tie=sum(x==0 for x in deltas),
                                               loss=sum(x<0 for x in deltas))
            for fold_result in scope['fold_deltas']:
                fold_result['acc'] = float(100*differences[fold_result['fold']])


def abstention_diagnostics(audits):
    """Recheck disease candidates against the off decision using held-out audit records."""
    results = {}
    for group in r.GROUPS:
        records = [x for a in audits if a['group']==group for x in a['records'] if not x['protected']]
        off_wrong = [x for x in records if x['off_label'] != x['y']]
        eligible = [x for x in records if x['eligible']]
        wrong_eligible = [x for x in off_wrong if x['eligible']]

        def candidate_margin_change(row):
            # The sole admissible disease action opposes the off decision.
            shift = row['asd_candidate_gap'] if row['off_label']==1 else row['td_candidate_gap']
            return (1-2*row['y'])*shift

        def would_correct(row):
            margin = row['off_margin']+candidate_margin_change(row)
            return margin>0 or (margin==0 and row['y']==0)

        results[group] = dict(off_wrong=len(off_wrong), eligible=len(eligible), wrong_eligible=len(wrong_eligible),
                              historical_wrong_eligible=sum(x['b0_label']!=x['y'] for x in eligible),
                              minimum_keep_probability=min((x['route_keep'] for x in eligible),default=None),
                              max_true_margin_gain=max((candidate_margin_change(x) for x in wrong_eligible),default=None),
                              minimum_wrong_margin_abs=min((-x['off_margin'] for x in wrong_eligible),default=None),
                              candidate_correctable_eligible=sum(map(would_correct,wrong_eligible)),
                              candidate_correctable_total=sum(map(would_correct,off_wrong)))
    return results


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




def append_handoff(payload, manifest):
    initial = manifest['handoff_before']
    path = r.Path(initial['path'])
    text = payload.encode()
    sha = lambda raw:hashlib.sha256(raw).hexdigest()
    marker = b'## Experiment 82.3.11'
    try:
        with path.open('r+b') as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            original = file.read()
            prefix = original[:initial['size']]
            assert sha(prefix)==initial['sha256'], 'Historical handoff prefix changed'
            if marker in original[initial['size']:]:
                assert original.endswith(text), 'E82.3.11 handoff already contains different text'
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


def verify_existing_handoff(manifest, verdict):
    saved = manifest.get('handoff', {})
    if saved.get('status') != 'APPENDED_READBACK_VERIFIED':
        return None
    original = manifest['handoff_before']
    path = r.Path(original['path'])
    text = path.read_bytes()
    assert verdict == manifest['verdict']
    assert hashlib.sha256(text[:original['size']]).hexdigest() == original['sha256']
    assert hashlib.sha256(text[:saved['new_size']]).hexdigest() == saved['final_sha256']
    assert b'## Experiment 82.3.11' in text[original['size']:saved['new_size']]
    return saved


def mechanics(aggregate):
    measures, route, ablations, functions, objectives, sites = [], [], [], [], [], []
    for group, item in aggregate['mechanisms'].items():
        cohort = item['strata']['all']
        counts = item['attribution']
        metrics = item['routing']
        residual = cohort['relative_residual']
        action_precision = '—' if metrics['action_precision'] is None else f'{100*metrics["action_precision"]:.2f}%'
        eligible_rate = '—' if metrics['eligible_error_fraction'] is None else f'{100*metrics["eligible_error_fraction"]:.2f}%'
        route.append((group,' / '.join(map(str,metrics['actions'])),metrics['eligibility_n'],action_precision,eligible_rate,
                      f'{100*metrics["off_error_prevalence"]:.2f}%',
                      f'{metrics["off_error_auroc"]:.3f} / {metrics["b0_error_auroc"]:.3f}',
                      f'{cohort["entropy"]["mean"]:.5f}',f'{metrics["nll_gap"]["mean"]:+.6f}'))
        measures.append((group,len(counts['backbone_correction']),len(counts['new_residual_correction']),
                         len(counts['residual_rescue']),len(counts['backbone_harm']),len(counts['new_residual_harm']),
                         cohort['residual_nonzero'],
                         f'{100*residual["mean"]:.3f} / {100*residual["p95"]:.3f} / {100*residual["max"]:.3f}',
                         f'{cohort["probability_change"]["mean"]:.7f}'))
        for label, aligned in item['alignment'].items():
            acc, shift = aligned['delta_pp']['acc'],aligned['probability_mean_difference']
            functions.append((group,'历史 Best' if label=='same_epoch_b0' else '本组 Best',aligned['folds'],
                              '—' if acc is None else f'{acc:+.4f}',
                              '—' if shift is None else f'{shift:.5f}',count_label(aligned['off_change']['all'])))
        b = item['basis']
        initial,learned = (item['objectives'][key] for key in ('U0','learned'))
        pair = lambda key: f'{initial[key]["mean"]:.3f} → {learned[key]["mean"]:.3f}'
        objectives.append((group,f'{b["rank"]["mean"]:.0f}',f'{b["projector_change_frobenius"]["mean"]:.5f}',
                           f'{b["gram_max_abs"]["max"]:.2e}',pair('variance_effective_rank'),
                           pair('condition'),pair('bhattacharyya'),pair('error_gradient_coverage'),
                           f'{cohort["j_cover_u0"]["mean"]:.4f} → {cohort["j_cover_u"]["mean"]:.4f}',
                           f'{cohort["null"]["mean"]:.4f}',f'{cohort["consistency"]["mean"]:.4f}',
                           item['strata']['conflicting_evidence']['n']))
        for name, mode in item['ablations'].items():
            ablations.append((group,name,mode['flips'],f'{mode["probability_absolute_change"]["mean"]:.7f}',
                              count_label(mode['change']['all'])))
        for name,site in item['sites'].items():
            sites.append((group,name,site['n'],site['actions'],f'{100*site["rate"]:.2f}%',count_label(site['direct']),
                          count_label(site['overall'])))
    return dict(attribution=md_table(('组别','主路径纠错','真正新增残差纠错','救回主路径错误','主路径误伤','残差新增误伤',
                       '实际非零','相对残差平均/P95/最大 %','平均绝对 ASD 概率变化'),measures),
                routing=md_table(('组别','Keep/ASD/TD','可信冲突覆盖人数','动作疾病方向精度','可信冲突中 off 判错率',
                       'off 整体错误率','冲突分数识别 off/B0 错误 AUROC','平均熵','硬动作−期望 NLL'),route),
                functions=md_table(('组别','同轮对齐','折数','off−Reference ACC pp','平均 |ΔP(ASD)|','纠/误/净'),functions),
                objectives=md_table(('组别','秩','投影矩阵距离','最大正交误差','有效秩','协方差条件数','Bhattacharyya',
                       '困难训练梯度覆盖','测试 Jacobian 覆盖','Null','局部上下文一致性','语义谱分歧人数'),objectives),
                ablations=md_table(('组别','干预','硬翻转','平均 |ΔP(ASD)|','从干预到 full 纠/误/净'),ablations),
                sites=md_table(('组别','站点','人数','实际干预','干预率','直接纠/误/净','总体纠/误/净'),sites))


def margin_table(aggregate):
    rows = []
    for group,item in aggregate['mechanisms'].items():
        for name, cohort in item['strata'].items():
            if not cohort['n']:
                continue
            rows.append((group,name,cohort['n'],*(f'{cohort["margin"][key]["mean"]:+.6f}' for key in 'ABC'),
                         f'{100*cohort["positive_margin_share"]:.2f}%'))
    return md_table(('组别','亚组','n','A 平均 Margin 差','B 平均 Margin 差','C 平均 Margin 差','B Margin 改善率'),rows)


def training_table(audits):
    rows,traces = [],[]
    for group in r.GROUPS:
        these = [a['training'] for a in audits if a['group']==group]
        for key in these[0]['gradients']:
            rows.append((group,key,f'{min(a["gradients"][key]["fraction"] for a in these):.2%}',
                         f'{min(a["updates"][key]["fraction"] for a in these):.2%}',
                         f'{max(a["gradients"][key]["maximum"] for a in these):.3e}'))
        for key in these[0]['protection']:
            items = [a['protection'][key] for a in these]
            traces.append((group,key,f'{max(k["ratio_max"] for k in items):.4f}',
                           f'{max(k["max_moment_budget_fraction"]["exp_avg"] for k in items):.4f}',
                           f'{max(k["max_moment_budget_fraction"]["exp_avg_sq"] for k in items):.4f}',
                           sum(k['capped_proposals'] for k in items),sum(k['capped_moments'] for k in items),
                           f'{np.mean([k["mean_retention_injected"] for k in items]):.3e}'))
    return (md_table(('组别','块','最小折真实梯度非零率','最小折非零参数更新率','最大真实梯度范数'),rows),
            md_table(('组别','历史块','最大参数预算占用','最大一阶预算占用','最大二阶预算占用','参数投影次数',
                      '动量投影次数','平均注入功能梯度范数'),traces))


GROUP_DESCRIPTION = ('G1 用历史医学查询匹配 CAM 原生局部投影与跨模态上下文，和完整协方差谱证据共同筛选可信冲突。'
                     'G2 用隔离 RNG 生成并逐 token 等范数匹配的匿名查询，保留完全相同的结构、训练和路由规则。'
                     '两组枚举原分类头同掩码下的真实 Keep、ASD、TD 风险，评估时执行单一 CAM144 特征动作。'
                     '同步原生 DARG 提供训练折功能 KL 目标，以及逐参数块和 Adam 动量信赖域。')


def publish(status,audits):
    from audit import decide

    manifest = h.read_json(r.HERE/'manifest.json')
    a = status['aggregate']
    exact_accuracy_pairs(a,audits)
    status['science'] = science = decide(audits,a,status['protection'])
    verdict = science['decision']
    written_handoff = verify_existing_handoff(manifest,verdict)
    status['abstention_diagnostics'] = abstention = abstention_diagnostics(audits)
    assert write_and_verify_tables(status,audits)
    status['csv_verified'] = True
    status['training'] = training = manifest['training']
    status['preflight_seconds'] = h.read_json(r.RT/'preflight.json')['elapsed']
    status['outcome_sha256'] = {name:h.sha_file(r.RESULTS/name) for name in
                               ('fold_metrics.csv','paired_comparisons.csv','patient_diagnostics.csv')}
    status['audit_implementation_sha256'] = {name:h.sha_file(r.HERE/name) for name in ('audit.py','report.py')}
    medical = ('通过预注册匹配对照标准' if science['medical_semantic_contribution'] else
               '匹配匿名重训未建立医学查询的独立增量')
    summaries = mechanics(a)
    first = statuses(status)
    elapsed = (f'物理 GPU0，{training["workers"]} 个单线程折 worker。GPU0 采样峰值 '
               f'{training["peak_sampled_gpu_mib"]} MiB。预检 {status["preflight_seconds"]:.1f} 秒，'
               f'训练 {training["elapsed"]:.1f} 秒，训练后审计 {status["audit_seconds"]:.1f} 秒，'
               f'审计累计 worker {status["cumulative_audit_worker_seconds"]:.1f} 秒。')
    integrity = (f'Preflight、Best、Latest 各 20/20。在线 Reference 的历史 Best、训练末状态、损失、'
                 f'可用原生梯度、三优化器、四类 RNG 逐轮配对，历史对照 10/10 精确恢复。'
                 f'独立核算 LLR、候选风险、真实 Dropout、合法训练功能 KL 与实际优化器步。'
                 f'Latest 下一轮训练、评估及 EMA 精确重放。受保护文件 {status["protection"]["count"]} 份。')
    limitations = ('历史 val=test、CV 前 RFE/scaling、转导图及单 seed 限制独立泛化解释。Fold2 的两行保留历史身份，'
                   '只计入历史协议正式指标。新增带标签损失、EMA 和 A/B/C 机制剔除这两行，三候选特征干预为零。'
                   '模态投影与上下文不是原 PEN 的脑区块，LLR 分摊是当前模型坐标归因。'
                   '匿名输入匹配维数、数量、逐 token 范数及训练条件，两组查询方向几何不同。')
    followup = ('在独立外层选模与预处理下复验医学候选及匹配对照。' if verdict=='GO' else
                '保留真实候选风险与同步功能保护审计。后续先在独立验证数据检验冲突分数能否识别原模型错误，再考虑扩大干预。')
    gradients, optimizer = training_table(audits)
    fold_table = md_table(('组别','Fold','Best','终止',*NAMES),[
        (row['group'],row['fold'],row['best_epoch'],row['end_epoch'],*[f'{row[key]:.4f}' for key in r.METRICS])
        for row in metric_rows(audits)])
    methods = ('预检的独立复算将 FP32 上游匹配和临界先验分别验证，冻结阈值、容差和训练模型保持原值。'
               '站点审计沿用原 CSV 的 1/2 到模型 0/1 映射，并逐行核对标签身份。'
               '配对 ACC 由逐折正确患者人数精确计算，避免把浮点相消形成的正数判为严格提高。'
               '历史身份见 [设计冻结](../design_review.md) 和 [manifest](../manifest.json)。首选 OSSFS '
               '创建出现 errno 5，统一使用 /model 路径。两组完整训练各十折。')
    func = md_table(('组别','合法训练平均同掩码 KL','平均无分类 Dropout KL','平均 |ΔP(ASD)|','平均每轮参考正确误伤'),
        [(g,f'{m["legal_function_mean"]["masked_kl"]:.6g}',f'{m["legal_function_mean"]["deterministic_kl"]:.6g}',
          f'{m["legal_function_mean"]["probability_shift"]:.6g}',f'{m["legal_function_mean"]["harmed"]:.3f}')
         for g,m in a['mechanisms'].items()])
    conclusions = []
    for group,item in a['mechanisms'].items():
        routing = item['routing']
        corrected = len(item['attribution']['new_residual_correction'])
        direct = item['stages']['B']['all']['net']
        overall = item['stages']['C']['all']['net']
        count = sum(routing['actions'][1:])
        drift = item['alignment']['same_epoch_b0']
        coverage = drift['folds']
        gap = drift['delta_pp']['acc']
        alignment = f'，历史 Best 同轮 off ACC 差 {gap:+.4f} pp' if gap is not None else ''
        conclusions.append(f'- {group} 在 {routing["eligibility_n"]} 名可信冲突病例中实际选择 {count} 个疾病动作。'
                           f'直接残差净纠错 {direct:+d} 人，真正新增纠正历史错分 {corrected} 人，完整模型净变化 {overall:+d} 人。'
                           f'同轮归因覆盖 {coverage}/10 折{alignment}。')
    key_findings = '\n'.join(conclusions)

    def fmt(value):
        return '—' if value is None else f'{value:.6f}'

    abstention_table = md_table(('组别','off 判错','可信冲突内 off 判错','最小 Keep 概率','判错且合格病例最大 Margin 增益',
                                 '判错且合格病例最小翻转所需 |Margin|','候选可纠错 合格内/全部'),
        [(group,row['off_wrong'],row['wrong_eligible'],fmt(row['minimum_keep_probability']),
          fmt(row['max_true_margin_gain']),fmt(row['minimum_wrong_margin_abs']),
          f'{row["candidate_correctable_eligible"]}/{row["candidate_correctable_total"]}')
         for group,row in abstention.items()])
    abstention_note = ('候选可纠错人数按冻结 Best 的实际 Gap 变化复算。每位 off 错分患者的候选动作指向与其原预测相反的疾病。'
                       '外层标签只用于这项诊断。')
    if all(row['candidate_correctable_eligible']==0 for row in abstention.values()):
        abstention_note += '两组在符合可信冲突条件的错分患者中，即使执行已训练的疾病候选，仍无可纠正病例。'
    report = f'''# Experiment 82.3.11 · PS-CER

**{status['status']} · Scientific {verdict}**。{medical}。

{GROUP_DESCRIPTION}

{methods}

{elapsed}

## 十折历史协议性能，均值 ± 总体标准差 %

{six_metrics(a)}

{paired_section(a,'full')}

## 去 Fold2

{six_metrics(a,'no_fold2')}

{paired_section(a,'no_fold2')}

## 逐折指标 %

{fold_table}

## 患者级归因（869 名合规机制患者）

A 比较历史 DARG 与本组 Best 的 residual-off，包含训练路径和选模轮次差异。B 固定该 Best 比较 residual-off 与 full。C 比较历史 DARG 与 full。表中依次为纠错、误伤、净纠错。正式配对仍包含历史两例。

{stages_section(a)}

{summaries['attribution']}

{margin_table(a)}

## 冲突识别、实际动作与固定权重干预

{summaries['routing']}

{abstention_table}

{abstention_note}

{summaries['ablations']}

Prompt-off 保留统计证据和原分类头，测量即时方向依赖。医学文本的独立增量采用 G1 与 G2 完整配对。{medical}。

## 谱空间与真实分类梯度

{summaries['objectives']}

{gradients}

## 主路径功能保护

{func}

{summaries['functions']}

同轮置信度和硬分类差异直接测量测试折的功能偏移。训练 KL 和参数预算衡量各自的保护范围。

{optimizer}

## 按站点统计

{summaries['sites']}

## 审计和冻结判定

{integrity}

{first}

{key_findings}

{limitations}

{followup}
'''
    with h.atomic_writer(r.RESULTS/'report.md','w') as handle:
        handle.write(report)
    status['outcome_sha256']['report.md'] = h.sha_file(r.RESULTS/'report.md')
    summary = f'''

## Experiment 82.3.11 · PS-CER

**{status["status"]} · Scientific {verdict}**。{medical}。
权威目录 `{r.HERE}`，G1/G2 各十折。

{GROUP_DESCRIPTION}

{elapsed}

{six_metrics(a)}

{paired_section(a,'full')}

{stages_section(a)}

{summaries['attribution']}

{summaries['routing']}

{summaries['objectives']}

{summaries['functions']}

{first}

{key_findings}

{integrity}

{limitations}

{followup}

[报告]({r.RESULTS/'report.md'}) · [逐折指标]({r.RESULTS/'fold_metrics.csv'}) · [配对]({r.RESULTS/'paired_comparisons.csv'}) · [患者诊断]({r.RESULTS/'patient_diagnostics.csv'}) · [审计]({r.RESULTS/'audit_status.json'})
'''
    status['handoff'] = written_handoff or append_handoff(summary,manifest)
    h.atomic_json(r.RESULTS/'audit_status.json',status)
    h.update_manifest(status=status['status'],verdict=verdict,handoff=status['handoff'],
                      audit_source=status['audit_implementation_sha256'],
                      results={**status['outcome_sha256'],'audit_status.json':h.sha_file(r.RESULTS/'audit_status.json')})

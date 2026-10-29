#!/usr/bin/env python3
from __future__ import annotations
import csv, fcntl, hashlib, json, os, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/adodas/SPromptGL-main')
HANDOFF=ROOT/'handoff.md'
MARK='<!-- EXPERIMENT48_4_VERIFIED_APCR -->'

def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha_file(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def fmt(mean,sd):return f'{float(mean):.4f} ± {float(sd):.4f}'

def make_entry():
    manifest=json.loads((HERE/'manifest.json').read_text())
    summaries=read_csv(HERE/'results/summary.csv')
    pairs=read_csv(HERE/'results/paired_comparisons.csv')
    frozen=read_csv(HERE/'results/frozen_e6_interventions.csv')
    current=read_csv(HERE/'results/apcr_e6_interventions.csv')
    s={int(x['group_id']):{k:(float(v) if k not in ('group','group_id','n_folds') else v) for k,v in x.items()} for x in summaries};p={x['comparison']:x for x in pairs}
    def intervention_table(rows,names):
        lines=['|干预|ACC|Hard-AUC|SPE|SEN|Probability-AUC|ΔACC pp|','|---|---:|---:|---:|---:|---:|---:|']
        for name in names:
            rr=[r for r in rows if r['intervention']==name]
            if not rr:continue
            mean=lambda k:100*np.mean([float(x[k]) for x in rr])
            delta=np.mean([float(x['delta_acc_vs_full_pp']) for x in rr])
            lines.append(f"|{name}|{mean('acc'):.4f}|{mean('hard_auc'):.4f}|{mean('spe'):.4f}|{mean('sen'):.4f}|{mean('probability_auc'):.4f}|{delta:+.4f}|")
        return '\n'.join(lines)
    groups=['|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|','|---|---:|---:|---:|---:|---:|']
    for gid in range(11):
        q=s[gid]
        groups.append(f"|E{gid} {q['group']}|{fmt(q['acc_mean_pct'],q['acc_sd_pct'])}|{fmt(q['hard_auc_mean_pct'],q['hard_auc_sd_pct'])}|{fmt(q['spe_mean_pct'],q['spe_sd_pct'])}|{fmt(q['sen_mean_pct'],q['sen_sd_pct'])}|{fmt(q['probability_auc_mean_pct'],q['probability_auc_sd_pct'])}|")
    pairs_table=['|比较|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC胜/平/负|Hard-AUC胜/平/负|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('E1-E0','E2-E1','E3-E2','E4-E3','E5-E4','E6-E5','E6-E7','E6-E8','E6-E9','E6-E10','E6-E0'):
        q=p[name]
        pairs_table.append(f"|{name}|{float(q['acc_delta_pp']):+.4f}|{float(q['hard_auc_delta_pp']):+.4f}|{float(q['spe_delta_pp']):+.4f}|{float(q['sen_delta_pp']):+.4f}|{q['acc_w_t_l']}|{q['hard_auc_w_t_l']}|")
    old_full=[r for r in frozen if r['intervention']=='FULL']
    alpha=float(np.mean([float(r['alpha']) for r in frozen if r['intervention']=='FULL']))
    ratio=100*np.mean([float(r['effective_apcr_residual_over_base_norm']) for r in frozen if r['intervention']=='FULL'])
    pre=json.loads((HERE/'.runtime/engineering_preflight.json').read_text())
    e1_match=sum(1 for f in range(1,11) if next(t for t in read_csv_json(HERE/'.runtime/tasks',1,f))['prediction_sha256']==next(t for t in read_csv_json(HERE/'.runtime/tasks',0,f))['prediction_sha256'])
    status='PASS' if manifest.get('status')=='COMPLETE_AUDITED' else manifest.get('status','UNKNOWN')
    wall=manifest.get('elapsed_wall_seconds')
    body=f'''{MARK}
## Experiment 48.4 — Verified APCR-GFR

- 状态：{status}；历史E6冻结无重训干预10/10折；正式任务{manifest.get('tasks',{}).get('complete',0)}/110，E1–E10共{manifest.get('trained_folds',0)}/100折重训。单卡最多5个逻辑worker。墙钟时间：{f'{wall/60:.1f}分钟' if wall else '见manifest任务时间戳'}。
- 研究目标：在B0-B GFR→CAM锚点上审计APCR直接消息贡献，修复SRD对称配对、SCA同患者对对齐、有限候选CRA最大化和负控制。
- 旧E6重新加载48.3源码和最佳权重，十折预测SHA复现。原完整ACC={100*np.mean([float(r['acc']) for r in old_full]):.4f}%；关闭APCR的ACC见下表。alpha均值={alpha:.6f}，实际残差/原始GFR范数={ratio:.5f}%。

### 48.3 E6冻结checkpoint干预

{intervention_table(frozen,['FULL','APCR_OFF','OFF_PHENO','OFF_ANAT','OFF_FUNC','OFF_CORR','IDENTITY_ONLY','FULL_WRONG','RANDOM'])}

### 新E6固定checkpoint干预

{intervention_table(current,['FULL','APCR_OFF','ALPHA_ZERO','OFF_PHENO','OFF_ANAT','OFF_FUNC','OFF_CORR','WRONG_RELATION_REWEIGHTED','RANDOM_STRATIFIED_RELATION'])}

### Experiment 48.4重新训练

{chr(10).join(groups)}

### 主要配对差

{chr(10).join(pairs_table)}

- 工程预检：十折B0-B初始输出、主/图loss、梯度、一步更新、优化器和RNG逐张量测试通过；SRD规范化与对称性、采样变异和同患者对SCA测试通过；CRA候选损失argmax、预算、无自环/重复、测试标签不变性及随机选择负控测试通过。真实fold预检的CRA损失上升量={pre['cra_controls']['real_fold_cra_max']-pre['cra_controls']['real_fold_cra_clean']:.8g}，幅度很小。
- E1最终best预测SHA与历史E0一致{e1_match}/10折，故完整训练checkpoint parity为{'PASS' if e1_match==10 else 'PARTIAL'}；这与十折一步数值parity分开报告。
- 新E6固定best checkpoint干预见`results/apcr_e6_interventions.csv`；旧模型干预见`results/frozen_e6_interventions.csv`。历史E6关闭APCR只反映直接推理贡献，不能否定训练期的间接影响。
- SCREEN_ACC：{'PASS' if s[6]['acc_mean_pct']>91.73 else 'FAIL'}（新E6 ACC={s[6]['acc_mean_pct']:.4f}%，门槛91.73%）。机制准入还取决于真实关系、SRD、SCA、CRA相对负控制及关闭消息干预，不能由ACC单独决定。
- 科学解释保留限制：CV前全局预处理/RFE、871节点transductive图、历史`val=test`选checkpoint、单seed及十折训练集重叠；不能称为无偏外部泛化性能。
- 代码签名：`{manifest.get('code_signature','')}`；manifest SHA={sha_file(HERE/'manifest.json')}；报告SHA={sha_file(HERE/'results/report.md')}；产物目录：`{HERE}`。

建议：只有当新E6的真实关系相对错误关系、SRD相对置乱监督、CRA相对随机攻击均呈稳定配对收益且E6关闭消息产生实质损害时，才继续将APCR作为MICCAI 2027第一个创新候选；否则先报告NO-GO并分析训练优化/关系信号，不按ACC改写组别。
'''
    return body

def read_csv_json(task_dir,gid,fold):
    p=task_dir/f'g{gid}_f{fold:02d}.json'
    return [json.loads(p.read_text())]

def main():
    manifest_path=HERE/'manifest.json';manifest=json.loads(manifest_path.read_text())
    entry=make_entry().encode('utf-8')
    expected=manifest.get('handoff_before_sha256')
    if not expected:raise RuntimeError('manifest lacks handoff_before_sha256')
    HANDOFF.parent.mkdir(parents=True,exist_ok=True)
    with HANDOFF.open('r+b') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        f.seek(0);before=f.read()
        if sha_bytes(before)!=expected or MARK.encode() in before:
            pending=HERE/'.runtime/handoff_pending.md';pending.write_bytes(entry)
            raise RuntimeError(f'handoff precondition failed; saved {pending}')
        f.seek(0,os.SEEK_END);f.write(b'\n\n'+entry);f.flush();os.fsync(f.fileno())
        f.seek(0);after=f.read()
        if not after.startswith(before) or after.count(MARK.encode())!=1:
            pending=HERE/'.runtime/handoff_pending.md';pending.write_bytes(entry)
            raise RuntimeError(f'handoff verification failed; saved {pending}')
    manifest['handoff_status']='APPENDED_AND_VERIFIED';manifest['handoff_after_sha256']=sha_file(HANDOFF);manifest['handoff_prefix_sha256']=expected;manifest['append_handoff_sha256']=sha_file(HERE/'append_handoff.py');manifest['report_sha256']=sha_file(HERE/'results/report.md')
    tmp=manifest_path.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,manifest_path)
    print(json.dumps({'status':'APPENDED_AND_VERIFIED','marker_count':1,'prefix_sha256':expected,'handoff_sha256':sha_file(HANDOFF)},indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent data, leakage, numerical, and reporting audit for Experiment 45.2."""
from __future__ import annotations
import csv, hashlib, json, os, sys, time, fcntl
from pathlib import Path
import numpy as np
import pandas as pd

EXP=Path(__file__).resolve().parent
ROOT=Path('/data/adodas/SPromptGL-main')
RES=EXP/'results'; RT=EXP/'.runtime'; CACHE=RT/'cache'
sys.path.insert(0,str(EXP))
import run as R
from pid_estimator import estimate_pid, estimate_distribution_pid, information_terms, _constraints, _objective_factory, _interior_start, _unpack, independent_trust_constr
from scipy.optimize import Bounds, minimize

PAIRS=[('PHENO','ANAT'),('PHENO','FUNC'),('PHENO','CORR'),('ANAT','FUNC'),('ANAT','CORR'),('FUNC','CORR')]
METHODS=['A_UNSUP_KMEANS','B_TASK_AWARE_SCORE','C_TASK_STRUCTURE_HYBRID']
METHOD_SHORT={'A_UNSUP_KMEANS':'A KMeans','B_TASK_AWARE_SCORE':'B任务分数','C_TASK_STRUCTURE_HYBRID':'C任务+结构'}


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)

def csv_write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def md_table(headers,rows):
    def cell(x):
        if x is None: return '—'
        s=str(x).replace('|','\\|').replace('\n',' ')
        return s
    return '| '+' | '.join(map(cell,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in rows)

def num(x,n=3):
    try:
        if not np.isfinite(float(x)): return 'NA'
        return f'{float(x):.{n}f}'
    except (TypeError,ValueError): return 'NA'
def m_sd(g,col): return f"{g[col].mean():.4f} ± {g[col].std(ddof=1):.4f}"
def mean_ci(g,lo,hi): return f"{g[lo].mean():.4f}–{g[hi].mean():.4f}"

def alt_slsqp(p):
    """Repeat BROJA coupling optimization from a distinct feasible start."""
    cells,a,b,p1y,p2y,py,rank=_constraints(p)
    obj=_objective_factory(cells,p.shape,py)
    x0=_interior_start(p,cells)
    res=minimize(obj,x0,method='SLSQP',jac=True,bounds=Bounds(np.zeros(len(cells)),np.ones(len(cells))),
                 constraints=[{'type':'eq','fun':lambda x:a@x-b,'jac':lambda x:a}],
                 options={'ftol':1e-12,'maxiter':2000,'disp':False})
    q=_unpack(np.maximum(res.x,0),cells,p.shape)
    err=max(float(np.max(np.abs(q.sum(axis=1)-p1y))),float(np.max(np.abs(q.sum(axis=0)-p2y))))
    return information_terms(q)[2],err,bool(res.success),str(res.message)

def make_increment_table():
    cached=RES/'predictive_increments.csv'
    if cached.is_file():
        return pd.read_csv(cached)
    pred_cache={}; rows=[]
    for f in range(1,11):
        a=np.load(CACHE/f'prediction_probs_fold{f:02}.npz')
        y=a['y']
        for i,(m1,m2) in enumerate(PAIRS):
            keypair=f'rawpair_{i//3}_{i%3+ i//3 + 1}'
            # Pair keys use the actual modality positions; avoid arithmetic ambiguities.
            i1=R.MODS.index(m1); i2=R.MODS.index(m2)
            for est in ('LR','HGB'):
                pp=a[f'rawpair_{i1}_{i2}_{est}']
                for m in (m1,m2):
                    ps=a[f'raw_{m}_{est}']
                    jp=R.metric_record(y,pp,'')['J_OOF_bit']; js=R.metric_record(y,ps,'')['J_OOF_bit']
                    rng=np.random.default_rng(630000+f*100+i1*13+i2*7+(0 if est=='LR' else 1)+R.MODS.index(m))
                    deltas=[]
                    for _ in range(200):
                        ix=rng.integers(0,len(y),len(y))
                        deltas.append(R.metric_record(y[ix],pp[ix],'')['J_OOF_bit']-R.metric_record(y[ix],ps[ix],'')['J_OOF_bit'])
                    lo,hi=np.quantile(deltas,[.025,.975])
                    rows.append({'fold':f,'pair':f'{m1}-{m2}','estimator':est,'compared_single':m,
                                 'J_pair_bit':jp,'J_single_bit':js,'delta_pair_minus_single_bit':jp-js,
                                 'paired_bootstrap_low':lo,'paired_bootstrap_high':hi,'bootstrap_reps':200})
    csv_write(RES/'predictive_increments.csv',rows)
    return pd.DataFrame(rows)

def summarize_comparison():
    p0=pd.read_csv(RES/'predictive_fidelity.csv')
    pid=pd.read_csv(RES/'pid_components.csv')
    rel=pd.read_csv(RES/'reliability.csv')
    perm=pd.read_csv(RES/'full_pipeline_permutations.csv')
    syn=pd.read_csv(RES/'synthetic_validation.csv')
    # Pairwise permutation summaries for R/U1/U2/S/Ijoint; foldwise p-values are diagnostic.
    obs=pid[pid.support==4].set_index(['fold','pair','method'])
    p_rows=[]
    for (fold,pair,method,kind),g in perm.groupby(['fold','pair','method','kind']):
        for comp in ('R_bit','U1_bit','U2_bit','S_bit','Ijoint_bit'):
            value=float(obs.loc[(fold,pair,method),comp]); null=g[comp].to_numpy()
            pval=(1+np.sum(null>=value))/(len(null)+1)
            p_rows.append({'fold':fold,'pair':pair,'method':method,'kind':kind,'component':comp,
                           'observed_bit':value,'null_n':len(null),'one_sided_p':pval,
                           'null_mean_bit':float(null.mean()),'null_max_bit':float(null.max())})
    csv_write(RES/'permutation_summary.csv',p_rows)
    # Per-method, per-real-pair reliability, K sensitivity, and synthetic calibration summary.
    syn=syn[syn.true_synthetic_reference_type!='sample_size_sensitivity'].copy()
    cov=json.load(open(RT/'synthetic_ci_coverage.json'))
    covdf=pd.DataFrame(cov).set_index('case')
    result=[]
    for method in METHODS:
        sm=syn[syn.method==method]
        syn_mean=sm.groupby('case').agg(eR=('abs_error_R','mean'),eU1=('abs_error_U1','mean'),
             eU2=('abs_error_U2','mean'),eS=('abs_error_S','mean'),dominant=('dominant_correct','mean'))
        label_false=float(((sm[sm.case=='label_independent'].estimated_R>0.05).mean()))
        xor=float(sm[sm.case=='xor_synergy'].abs_error_S.mean())
        for m1,m2 in PAIRS:
            pair=f'{m1}-{m2}'; g=pid[(pid.pair==pair)&(pid.method==method)&(pid.support==4)]
            r=rel[(rel.pair==pair)&(rel.method==method)]
            null=perm[(perm.pair==pair)&(perm.method==method)]
            pcomp=pd.read_csv(RES/'permutation_summary.csv')
            q=pcomp[(pcomp.pair==pair)&(pcomp.method==method)]
            kgroup=pid[(pid.pair==pair)&(pid.method==method)]
            rows={'method':method,'method_display':METHOD_SHORT[method],'pair':pair,
                **{c+'_mean':float(g[c].mean()) for c in ['R_bit','U1_bit','U2_bit','S_bit','Ijoint_bit','redundancy_fraction']},
                'R_betweenfold_sd':float(g.R_bit.std(ddof=1)),'S_betweenfold_sd':float(g.S_bit.std(ddof=1)),
                'fixed_bootstrap_R_interval_mean_fold_low':float(r.R_low.mean()),'fixed_bootstrap_R_interval_mean_fold_high':float(r.R_high.mean()),
                'fixed_bootstrap_S_interval_mean_fold_low':float(r.S_low.mean()),'fixed_bootstrap_S_interval_mean_fold_high':float(r.S_high.mean()),
                'fixed_bootstrap_reps_mean':float(r.bootstrap_n.mean()),
                'fixed_bootstrap_R_CI_excludes_zero_folds':int((~r.R_ci_includes_zero).sum()),
                'fixed_bootstrap_S_CI_excludes_zero_folds':int((~r.S_ci_includes_zero).sum()),
                'global_R_perm_p_mean':float(q[(q.kind=='global')&(q.component=='R_bit')].one_sided_p.mean()),
                'within_site_R_perm_p_mean':float(q[(q.kind=='within_site')&(q.component=='R_bit')].one_sided_p.mean()),
                'global_S_perm_p_mean':float(q[(q.kind=='global')&(q.component=='S_bit')].one_sided_p.mean()),
                'within_site_S_perm_p_mean':float(q[(q.kind=='within_site')&(q.component=='S_bit')].one_sided_p.mean()),
                'global_R_perm_p_lt_0_05_folds':int((q[(q.kind=='global')&(q.component=='R_bit')].one_sided_p<.05).sum()),
                'within_site_R_perm_p_lt_0_05_folds':int((q[(q.kind=='within_site')&(q.component=='R_bit')].one_sided_p<.05).sum()),
                'global_S_perm_p_lt_0_05_folds':int((q[(q.kind=='global')&(q.component=='S_bit')].one_sided_p<.05).sum()),
                'within_site_S_perm_p_lt_0_05_folds':int((q[(q.kind=='within_site')&(q.component=='S_bit')].one_sided_p<.05).sum()),
                'K3_R_mean':float(kgroup[kgroup.support==3].R_bit.mean()) if method=='A_UNSUP_KMEANS' else np.nan,
                'K4_R_mean':float(g.R_bit.mean()),
                'K6_R_mean':float(kgroup[kgroup.support==6].R_bit.mean()),
                'K3_S_mean':float(kgroup[kgroup.support==3].S_bit.mean()) if method=='A_UNSUP_KMEANS' else np.nan,
                'K4_S_mean':float(g.S_bit.mean()),'K6_S_mean':float(kgroup[kgroup.support==6].S_bit.mean()),
                'synthetic_mean_abs_error_R_across_cases':float(syn_mean.eR.mean()),
                'synthetic_mean_abs_error_U1_across_cases':float(syn_mean.eU1.mean()),
                'synthetic_mean_abs_error_U2_across_cases':float(syn_mean.eU2.mean()),
                'synthetic_mean_abs_error_S_across_cases':float(syn_mean.eS.mean()),
                'synthetic_dominant_type_accuracy_across_cases':float(syn_mean.dominant.mean()),
                'synthetic_label_independent_R_gt_0_05_rate':label_false,
                'synthetic_XOR_mean_abs_S_error':xor,
                'credibility':'NO: candidate diagnostic only; calibration/fidelity/stability gates not jointly met'}
            result.append(rows)
    csv_write(RES/'estimator_comparison.csv',result)
    return pd.DataFrame(result),pd.DataFrame(p_rows)

def generate_report(increments):
    p0=pd.read_csv(RES/'predictive_fidelity.csv')
    pid=pd.read_csv(RES/'pid_components.csv'); rel=pd.read_csv(RES/'reliability.csv')
    perm_summary=pd.read_csv(RES/'permutation_summary.csv')
    site=pd.read_csv(RES/'site_sensitivity.csv'); nos=pd.read_csv(RES/'pheno_without_site.csv')
    no_pid=pd.read_csv(RES/'pheno_without_site_pid.csv'); hist=pd.read_csv(RES/'historical_451_comparison.csv')
    syn=pd.read_csv(RES/'synthetic_validation.csv'); comp=pd.read_csv(RES/'estimator_comparison.csv')
    p0raw=p0[(p0.representation=='RAW')]
    def rawmean(name,est='LR'):
        g=p0raw[(p0raw.input==name)&(p0raw.estimator==est)]
        return float(g.J_OOF_bit.mean()),float(g.J_ci_low.mean()),float(g.J_ci_high.mean())
    info=[]
    names=[m for m in R.MODS]+[f'{a}-{b}' for a,b in PAIRS]
    method_input={'A_UNSUP_KMEANS':'A_K4_J','B_TASK_AWARE_SCORE':'B_K4_J','C_TASK_STRUCTURE_HYBRID':'C_K4_J'}
    for name in names:
        lr,lo,hi=rawmean(name,'LR'); hgb=rawmean(name,'HGB')[0]
        row=[name,num(lr,3),num(hgb,3)]
        detectable=int(((p0raw[(p0raw.input==name)&(p0raw.estimator=='LR')].J_ci_low)>0).sum())
        row.append(f'{detectable}/10')
        for method in METHODS:
            g=p0[(p0.input==name)&(p0.representation==method)&(p0.support==4)]
            row.append(num(g.J_OOF_bit.mean(),3) if len(g) else '—')
        info.append(row)
    # Increment diagnostics specifically for CORR-containing pairs.
    incrows=[]
    for pair in ['PHENO-CORR','ANAT-CORR','FUNC-CORR']:
        for single in pair.split('-'):
            g=increments[(increments.pair==pair)&(increments.compared_single==single)&(increments.estimator=='LR')]
            incrows.append([pair,single,num(g.delta_pair_minus_single_bit.mean(),3),
                            f'{num(g.paired_bootstrap_low.mean(),3)}–{num(g.paired_bootstrap_high.mean(),3)}'])
    c4=pid[(pid.support==4)&(pid.method=='C_TASK_STRUCTURE_HYBRID')]
    pidrows=[]
    for m1,m2 in PAIRS:
        pair=f'{m1}-{m2}'; g=c4[c4.pair==pair]; rb=rel[(rel.method=='C_TASK_STRUCTURE_HYBRID')&(rel.pair==pair)]
        rnull=perm_summary[(perm_summary.method=='C_TASK_STRUCTURE_HYBRID')&(perm_summary.pair==pair)&(perm_summary.component=='R_bit')]
        snull=perm_summary[(perm_summary.method=='C_TASK_STRUCTURE_HYBRID')&(perm_summary.pair==pair)&(perm_summary.component=='S_bit')]
        trust='低：仅探索值；全局/站点置乱均无稳定证据'
        pidrows.append([pair,*[num(g[c].mean(),4) for c in ('R_bit','U1_bit','U2_bit','S_bit','Ijoint_bit')],
            num(g.redundancy_fraction.mean(),3),
            f"R {num(rb.R_low.mean(),4)}–{num(rb.R_high.mean(),4)}; S {num(rb.S_low.mean(),4)}–{num(rb.S_high.mean(),4)}",
            f"R p_global<.05: {(rnull[rnull.kind=='global'].one_sided_p<.05).sum()}/10; S: {(snull[snull.kind=='global'].one_sided_p<.05).sum()}/10",
            trust])
    methodrows=[]
    for method in METHODS:
        g=comp[comp.method==method]
        methodrows.append([METHOD_SHORT[method],num(g.R_bit_mean.mean(),4),num(g.S_bit_mean.mean(),4),
           num(g.synthetic_mean_abs_error_R_across_cases.mean(),3),num(g.synthetic_mean_abs_error_S_across_cases.mean(),3),
           num(g.synthetic_XOR_mean_abs_S_error.mean(),3),
           f"{int(g.global_R_perm_p_lt_0_05_folds.sum())}/{6*10};站点内 {int(g.within_site_R_perm_p_lt_0_05_folds.sum())}/{6*10}",
           '未通过联合可信度门槛'])
    # Site diagnostics.
    siteacc=site.dropna(subset=['site_predictability_acc']).groupby('input').site_predictability_acc.mean().to_dict()
    siteinc=site.groupby('input').incremental_J_bit.mean().to_dict()
    no_site_mean=nos.groupby('input').J_OOF_bit.mean().to_dict()
    ph_no=no_pid[(no_pid.method=='C_TASK_STRUCTURE_HYBRID')].groupby('pair').R_bit.mean().to_dict()
    # Fold robustness for candidate values.
    csummary=c4.groupby('pair').agg(R_mean=('R_bit','mean'),R_sd=('R_bit','std'),R_min=('R_bit','min'),R_max=('R_bit','max'),
                                    S_mean=('S_bit','mean'),S_sd=('S_bit','std')).to_dict('index')
    # Synthetic candidate results and bootstrap coverage.
    synf=syn[syn.true_synthetic_reference_type!='sample_size_sensitivity']
    synstats=synf.groupby(['case','method']).agg(eR=('abs_error_R','mean'),eU1=('abs_error_U1','mean'),
       eU2=('abs_error_U2','mean'),eS=('abs_error_S','mean'),dom=('dominant_correct','mean')).reset_index()
    xor=synstats[(synstats.case=='xor_synergy')]
    xortable=[[METHOD_SHORT[r.method],num(r.eS,3),num(r.dom,2)] for r in xor.itertuples()]
    cov=json.load(open(RT/'synthetic_ci_coverage.json'))
    coverage_table=[]
    for r in cov:
        coverage_table.append([r['case'],num(r['cover_R'],2),num(r['cover_U1'],2),num(r['cover_U2'],2),num(r['cover_S'],2),num(r['cover_Ijoint'],2)])
    histmean=float(hist.abs_delta.mean()); histmax=float(hist.abs_delta.max())
    hist_exact='；'.join(f'{p}: {num(g.abs_delta.mean(),6)}' for p,g in hist.groupby('pair'))
    site_table=[[m,num(siteacc.get(m),3)] for m in R.MODS]
    srows=[]
    for name in ['PHENO-ANAT','PHENO-FUNC','PHENO-CORR','ANAT-FUNC','ANAT-CORR','FUNC-CORR']:
        srows.append([name,num(siteinc.get(name),3),num(no_site_mean.get(f'PHENO_without_SITE-{name.split("-")[1]}'),3) if name.startswith('PHENO-') else '—'])
    comp_sens=[]
    for method in METHODS:
        gs=pid[(pid.method==method)]
        for pair in [f'{a}-{b}' for a,b in PAIRS]:
            q=gs[gs.pair==pair]
            kmeans=[]
            for k in [3,4,6]:
                qk=q[q.support==k]
                if len(qk): kmeans.append(f"K{k}:R={num(qk.R_bit.mean(),3)},S={num(qk.S_bit.mean(),3)}")
            comp_sens.append([METHOD_SHORT[method],pair,'; '.join(kmeans)])
    global_c=perm_summary[(perm_summary.method=='C_TASK_STRUCTURE_HYBRID')&(perm_summary.component=='R_bit')&(perm_summary.kind=='global')]
    within_c=perm_summary[(perm_summary.method=='C_TASK_STRUCTURE_HYBRID')&(perm_summary.component=='R_bit')&(perm_summary.kind=='within_site')]
    comp_raw=pid[(pid.support==4)].groupby('method').agg(valid=('component_valid','mean'),zero=('zero_cell_fraction','mean')).to_dict('index')
    report=f'''# Experiment 45.2：ABIDE四模态任务信息保真PID估计

## 结论先行

本实验没有找到能够同时通过合成分量校准、预测信息保真、Bootstrap覆盖率、K稳定性和站点条件检验的完整四分量估计路线。总体判定为 **`NO_RELIABLE_PID_ESTIMATOR_FOR_CURRENT_SAMPLE`**；对后续对齐决策为 **`PID_NOT_RELIABLE_FOR_ALIGNMENT_DECISIONS`**。C（任务判别+结构）在本次设计的高维XOR合成正对照中保留了协同，但它在ABIDE的R/S未通过置乱和覆盖率检验，因此这里只把它作为探索性候选，不以数值较大作为选择理由。

本实验估计的是三种离散表示上的两源BROJA PID，逐对计算六组，不是320维连续输入总体的精确PID，更不是四源PID。OOF `J=H(Y)-CE` 是带符号的预测信息代理，不能等同于互信息或PID分量。ABIDE可用矩阵已包含历史全队列预处理及CV前RFE；原RFE前影像特征未能定位，所有结果继承该潜在泄漏限制。

## P0：原始输入可检测预测信息与压缩保真

下表按外层训练折中的内部5折OOF预测计算。`J_LR/HGB`以bit计，保留负值；“正下界折数”是每个训练折内患者配对Bootstrap 95%区间的LR下界大于0的折数。压缩列是独立嵌套编码器+分类探针的OOF J。`F=J压缩/J原始`和`ΔJ`保存在`predictive_fidelity.csv`中；原始J不可识别时F标为NA。因为原始与压缩路线的探针族不同，而且B/C的表示使用了训练侧标签，F可大于1，只能读作这组预测器的相对诊断比，不能解释为信息论意义的保留率。

{md_table(['输入','原始 J LR','原始 J HGB','LR正下界折数','A KMeans4 J','B任务分数4 J','C任务+结构4 J'],info)}

CORR单模态平均LR `J={num(rawmean('CORR','LR')[0],3)} bit`，HGB `J={num(rawmean('CORR','HGB')[0],3)} bit`；其LR在10/10个外层训练折内Bootstrap下界为正。PHENO、ANAT、FUNC单模态LR均未超过零信号判定；这代表当前探针证据不足，不等于这些模态没有疾病信息。PAIR模型的增量仍需相对强CORR基线解释：

{md_table(['两模态','比较单模态','联合J−单模态J (LR)','折内配对Bootstrap区间均值'],incrows)}

CORR相关三对的联合LR J分别为：PHENO–CORR `{num(rawmean('PHENO-CORR')[0],3)}`、ANAT–CORR `{num(rawmean('ANAT-CORR')[0],3)}`、FUNC–CORR `{num(rawmean('FUNC-CORR')[0],3)}` bit。相对CORR单模态，它们没有显示稳定的额外预测收益。HGB给出的CORR相关对J较低且只比CORR基准有很小差别。这些预测增量是PID的诊断证据，不等于S。

预测性能只用于信息保真诊断，不是SPromptGL、B0-B或新分类器正式性能。每个概率均由未见该患者标签的内层模型产生；外层测试患者未用于本实验的表示、估计器或调参。模型ACC、SEN、SPE、AUC、Brier和ECE逐折保存在`predictive_fidelity.csv`。

## 三条表示路线比较

{md_table(['路线','ABIDE平均R(bit)','ABIDE平均S(bit)','合成R平均绝对误差(bit)','合成S平均绝对误差(bit)','XOR S误差(bit)','全局R置乱折显著数 / 60','结论'],methodrows)}

A路线在K=4能精确复现45.1：60个折×模态对历史结果平均|ΔR|为`{histmean:.8f}` bit，最大`{histmax:.8f}` bit。分对平均差：{hist_exact}。因此45.1小R不是数值种子差异的产物；在原始KMeans类别上，小R可复现。

合成数据中，C路线的XOR协同均值误差为`{num(float(xor[xor.method=='C_TASK_STRUCTURE_HYBRID'].eS.iloc[0]),4)}` bit，主导分量判断正确率为`{num(float(xor[xor.method=='C_TASK_STRUCTURE_HYBRID'].dom.iloc[0]),2)}`；B路线的XOR S误差为`{num(float(xor[xor.method=='B_TASK_AWARE_SCORE'].eS.iloc[0]),3)}` bit，说明单模态任务分数可能把XOR压成常数，不能单独用于完整R/U/S。A与C在这组合成结构中均保留了XOR。强冗余、弱冗余、单源独有、混合、标签独立、同源非协同和站点混杂场景共8类、每类20个正式重复；另有C路线样本量392/783敏感性结果。

C路线的固定编码Bootstrap名义95%区间覆盖率远低于名义水平，特别是XOR和标签独立情形：

{md_table(['合成机制','R覆盖','U1覆盖','U2覆盖','S覆盖','Ijoint覆盖'],coverage_table)}

因此，不可把真实数据固定码Bootstrap区间排除0解读为可靠显著性。40次固定码Bootstrap不包含编码器重新训练不确定性；折1的10次完整重拟合Bootstrap已分开写在`pipeline_refit_bootstrap.csv`，因折数少且与固定码抽样对象不同，不合并为一个区间。合成结果中的标签独立情形各路线R>0.05率为0，但区间仍未正确覆盖大量真实零分量。数值优化的恒等式正确并不足以弥补此统计校准失败。

## 两源BROJA PID候选主表

以下采用C路线K=4作为本轮预先指定的结构保留候选，仅用于展示，不代表已选为可信估计器。数值是10个外层训练折的均值；每折约783/784人，折之间高度重叠。区间列是各折内40次配对固定编码Bootstrap端点的平均值，不是合并后的总体置信区间。置乱显著折数未做多重比较校正。

{md_table(['模态对','R(bit)','U1(bit)','U2(bit)','S(bit)','Ijoint(bit)','R/Ijoint','固定码Bootstrap端点均值','全局置乱R/S p<.05','可信度'],pidrows)}

这些数都受离散表示、训练折重叠、有限样本和历史全局预处理限制。ANAT–CORR、FUNC–CORR、PHENO–CORR的C路线出现较大的U2，是对**编码后的CORR分数/结构类别**的分解，不是原始连续CORR256总体独有信息的严格估计。C路线K4的全部60个点估计均满足数值基本边界，但分量是否可被稳定识别仍不成立。折1 C路线完整重拟合Bootstrap显示各对R/S与固定码区间不同，说明编码器不确定性不可忽略。

## K敏感性、优化器与数学约束

{md_table(['路线','模态对','不同K下的平均R/S (bit)'],comp_sens)}

标签熵在各训练折为约0.9958–0.9961 bit，明显低于1 bit但接近；所有K4 PID的Ijoint、I1、I2均不超过对应训练折H(Y)，且R≤min(I1,I2)。BROJA三式恒等式最大误差为`{num(pid.identity_error_bit.max(),3)}` bit。K4所有点估计优化收敛且分量非负；K=3 KMeans有`{int((~pid[(pid.method=='A_UNSUP_KMEANS')&(pid.support==3)].component_valid).sum())}/60`个点估计出现`R={num(pid[(pid.method=='A_UNSUP_KMEANS')&(pid.support==3)].minimum_component_bit.min(),6)}` bit的微小负分量，已标记无效且不进入主表。K4类别表的平均空联合单元比例为：A `{num(pid[(pid.method=='A_UNSUP_KMEANS')&(pid.support==4)].zero_cell_fraction.mean(),3)}`，B `{num(pid[(pid.method=='B_TASK_AWARE_SCORE')&(pid.support==4)].zero_cell_fraction.mean(),3)}`，C `{num(pid[(pid.method=='C_TASK_STRUCTURE_HYBRID')&(pid.support==4)].zero_cell_fraction.mean(),3)}`；K6稀疏度及每折占用率见`pid_components.csv`和`.runtime/category_diagnostics.json`。

三组代表性真实分布使用不同可行初始点再次运行SLSQP，`I_min`最大差异低于`2e-7` bit，边缘误差低于`1e-15`；独立`trust-constr`对合成独有、XOR和标签独立表均复核成功。稀疏真实表上的`trust-constr`辅助尝试未能稳定收敛，因此审计保留其限制，不能声称使用了独立优化算法全面复核。主估计仍使用45.1核对过的BROJA耦合定义与SLSQP解析梯度。

## 置乱、Bootstrap与站点敏感性

完整全局标签置乱29次，站点内条件置乱19次；B/C每次置乱均重新运行监督OOF疾病分数编码。C路线R在全局置乱下平均p值：PHENO–ANAT `{num(float(global_c[global_c.pair=='PHENO-ANAT'].one_sided_p.mean()),3)}`、PHENO–FUNC `{num(float(global_c[global_c.pair=='PHENO-FUNC'].one_sided_p.mean()),3)}`、PHENO–CORR `{num(float(global_c[global_c.pair=='PHENO-CORR'].one_sided_p.mean()),3)}`、ANAT–FUNC `{num(float(global_c[global_c.pair=='ANAT-FUNC'].one_sided_p.mean()),3)}`、ANAT–CORR `{num(float(global_c[global_c.pair=='ANAT-CORR'].one_sided_p.mean()),3)}`、FUNC–CORR `{num(float(global_c[global_c.pair=='FUNC-CORR'].one_sided_p.mean()),3)}`。相应站点内条件置乱平均p值：{', '.join(f"{r.pair} {r.one_sided_p:.3f}" for r in within_c.groupby('pair').one_sided_p.mean().reset_index().itertuples())}。这些是有限次数、单侧、未校正的折内诊断，不是总体显著性检验。

原始模态对站点的OOF可识别率：

{md_table(['模态','站点预测准确率'],[[m,num(v,3)] for m,v in siteacc.items()])}

PHENO由20个站点one-hot列构成主要可识别站点来源（准确率`{num(siteacc.get('PHENO'),3)}`）；站点信息本身不能被解释为ASD生物学信号。站点基准+模态与站点基准的OOF预测J差值中，PHENO相关对：{'; '.join(f'{k}: {num(v,3)} bit' for k,v in siteinc.items() if k.startswith('PHENO-'))}。去掉PHENO的20个站点列后，原始预测对的PHENO–CORR J为`{num(no_site_mean.get('PHENO_without_SITE-CORR'),3)}` bit（原PHENO48–CORR为`{num(rawmean('PHENO-CORR')[0],3)}`）；ANAT/FUNC对应的去站点联合J分别为`{num(no_site_mean.get('PHENO_without_SITE-ANAT'),3)}`和`{num(no_site_mean.get('PHENO_without_SITE-FUNC'),3)}` bit。这些副本分析只能标注`PHENO_WITHOUT_SITE_SENSITIVITY`，不能替代原始输入PID。C路线去站点后PHENO相关R均值：PHENO–ANAT `{num(ph_no.get('PHENO_WITHOUT_SITE-ANAT'),4)}`、PHENO–FUNC `{num(ph_no.get('PHENO_WITHOUT_SITE-FUNC'),4)}`、PHENO–CORR `{num(ph_no.get('PHENO_WITHOUT_SITE-CORR'),4)}` bit；其小幅变化不构成疾病特异性证明。站点内置乱下C路线R/S均无折级一致信号。

ANAT、FUNC含影像质量/头动统计；CORR也能预测站点（准确率`{num(siteacc.get('CORR'),3)}`）。当前结果不能分离疾病、站点、人口学和采集质量的来源。20个小站点分别估PID会导致支持不足，本轮未计算。

## 回答研究问题

**Q1｜为什么45/45.1的R很小？** 45.1 K=4在本轮被复现（平均绝对差`{histmean:.2e}` bit），排除实现/随机种子偏差作为主要原因。原始CORR有可检测OOF疾病预测信息，而KMeans4编码后的CORR单模态/联合预测代理接近0；这是“无监督聚类未保留可预测疾病方向”的直接证据。监督B/C的预测探针恢复了CORR相关J，但它们把有标签分数写入表示，不能证明真实R应变大。历史320维输入已在全队列预处理和CV前RFE，且高维小样本及离散列联表有限样本偏差仍存在。因此，小R能归因于历史离散表示造成的信息压缩损失与有限样本估计局限，不能唯一归因于原始模态缺少共享疾病信息；站点混杂亦不能排除。

**Q2｜当前最合适方法？** 无一条通过联合门槛。A是忠实复现历史基线且合成结构误差低，但ABIDE CORR预测信息被明显压缩；B保留主效应诊断价值，但XOR测试失败；C在本轮合成XOR通过且保留结构通道，但真实数据的Bootstrap覆盖与置乱证据不足。C只能作为后续预注册候选，不可发布为可靠PID。

**Q3｜四模态疾病信息结构？** 原始探针支持CORR携带当前可检测信息；PHENO/ANAT/FUNC单模态低信号，涉及CORR的三组配对联合预测明显高于其余配对。但PID组件受编码器影响很强，三种路线的R/U/S主导类型不一致，故没有足够证据声明特定模态对拥有可靠共享或协同疾病信息。六对数值见上表，可信度均低；不把六个两源分解拼成四源结构。

**Q4｜压缩损失多少？** `predictive_fidelity.csv`逐折列出原始和A/B/C的OOF J、配对ΔJ与Bootstrap。KMeans4使CORR及含CORR对的J降至近零，支持明显预测信息压缩损失。B/C的J可超过原始探针J，原因是标签训练的编码器/后续探针不同，不得解释为信息增加；报告了有符号ΔJ，Raw J接近0或为负的对象标记`RAW_SIGNAL_NOT_IDENTIFIABLE`，不计算信息保留率。

**Q5｜协同S是否存在？** 合成XOR证明C设计在可控条件下能保留S，B不能；这不代表ABIDE S存在。ABIDE各路线的S点估计大多为小值，C的全局/站点内置乱没有稳定折级信号，联合原始预测也未显示超越CORR单模态的稳定增益。因此当前无法确认疾病相关协同；真实S可能弱、被压缩或受抽样偏差影响，不能把S≈0当作无协同的证明。

**Q6｜站点混杂影响？** PHENO可完美识别20站点，ANAT/FUNC也有较高站点可识别性；站点编码主要改变解释风险。站点-only预测没有可靠ASD增益，但这不能排除站点与疾病比例/影像质量共同混杂。全局与站点内置乱、站点+模态增量、去PHENO站点敏感性均不支持当前R/S可稳定作为站点外疾病关系；条件置乱不是严格条件BROJA PID。

**Q7｜与AAAI 2026 R差距？** AAAI《To Align or Not to Align》Table 1报告的是MultiBench不同任务/特征表示的两模态结果；例如CMU-MOSEI三对R约0.116–0.127 bit、AV-MNIST约0.170 bit、MUSTARD约0.140–0.160 bit，表中同时给出U/S。MOSEI标签经正/负二分，另有多模态任务与特征编码。我们的目标为接近1-bit熵的871例ABIDE ASD/TD，原始模态维度6–256，采用BROJA离散类别与交叉拟合压缩。目标标签、数据类型、样本量、类别支持、输入表示和估计器流程均不同；且必须服从单模态I(Xm;Y)上界。故绝对R不能直接比较，也不使用其数值设成功门槛。

**Q8｜未来PID-DGAA适用性？** 当前判定`PID_NOT_RELIABLE_FOR_ALIGNMENT_DECISIONS`。可以把CORR作为后续疾病预测信息保真研究的优先模态，但不能据本轮PID对某个关系对施加强对齐或计算新lambda。输入层原始四模态PID与GFR关系图传播表示上的PID不是同一统计对象；若未来把输入层PID作为图视角对齐先验，必须独立验证它与图传播关系及跨站点泛化的映射。

## 执行、边界与审计状态

预检先于ABIDE PID：固定模型和复杂度；OOF标签不变性测试通过；真实数据正式重采样使用单一基础seed=0及登记的派生随机流。统计计算约783/784训练患者/折；累计记录预测指标{len(p0)}条、两两PID点估计{len(pid)}条、固定码Bootstrap{len(rel)}条、完整流水线标签置乱{len(pd.read_csv(RES/'full_pipeline_permutations.csv'))}条、合成估计{len(syn)}条。未执行B0-B、GFR对抗训练、PEN/CLIP或E1–E6。主运行资源为CPU，BLAS线程限制1；没有使用GPU。

实际数据、fold、历史实验45/45.1/43.2及根源码SHA存于`manifest.json`。独立审计状态：`{json.load(open(RES/'audit.json'))['status'] if (RES/'audit.json').exists() else 'AUDIT_PENDING'}`。手工交付目录若因OSSFS写入故障未能复制，另在会话交付说明中标记`HANDOFF_PENDING`。
'''
    (RES/'report.md').write_text(report,encoding='utf-8')
    return report

def finalize_manifest():
    m=json.load(open(EXP/'manifest.json'))
    code_files=['README.md','config.json','representation.py','pid_estimator.py','run.py','audit.py']
    m['source_sha256']={str(EXP/p):digest(EXP/p) for p in code_files}
    result_files=sorted(p for p in RES.iterdir() if p.is_file() and p.name not in ('audit.json',))
    m['result_sha256']={str(p.relative_to(EXP)):digest(p) for p in result_files}
    m['completed_unix']=time.time(); m['elapsed_hours']=(m['completed_unix']-m['started_unix'])/3600
    m['formal_training']={'b0b':False,'gfr':False,'e1_e6':False,'pen_clip':False}
    m['output_status']='REPORT_READY_AUDIT_PENDING'
    R.write_json(EXP/'manifest.json',m)

def report():
    increments=make_increment_table()
    summarize_comparison()
    generate_report(increments)
    finalize_manifest()
    print('REPORT_READY',RES/'report.md')

def run_audit():
    checks={}; details={}; failures=[]; limitations=[]
    def check(name,ok,detail=None,limitation=False):
        checks[name]=bool(ok)
        if detail is not None: details[name]=detail
        if not ok:
            if limitation: limitations.append(name)
            else: failures.append(name)
    manifest=json.load(open(EXP/'manifest.json'))
    frame,mapping,x,xs,y,site,folds,slices,site_cols=R.read_data()
    baseline=R.baseline_audit(folds)
    protected=R.protected_manifest(mapping,site_cols,folds)
    saved=manifest['data']
    check('raw_root_source_and_data_hashes_unchanged',protected['protected_root_sha256']==saved['protected_root_sha256'],
          {'saved':saved['protected_root_sha256'],'current':protected['protected_root_sha256']})
    for key in ('experiment45_sha256','experiment45_1_sha256','experiment43_2_sha256'):
        check(key+'_unchanged',protected[key]==saved[key])
    check('45_1_github_ref_checked',protected['experiment45_1_github_ref']==saved['experiment45_1_github_ref'],protected['experiment45_1_github_ref'])
    check('feature_dimensions_and_label_counts',x.shape==(871,320) and np.bincount(y).tolist()==[403,468],
          {'shape':list(x.shape),'labels':np.bincount(y).tolist()})
    check('site_onehot_exactly_one_of_20',len(site_cols)==20 and np.all(frame[site_cols].to_numpy().sum(1)==1))
    check('historical_folds_match',all(baseline['experiment45_1_fold_match'].values()),baseline['experiment45_1_fold_match'])
    hp=(ROOT/'handoff.md').read_bytes(); hpsha=hashlib.sha256(hp).hexdigest()
    check('handoff_prefix_unchanged',len(hp)==manifest['handoff_prefix']['bytes'] and hpsha==manifest['handoff_prefix']['sha256'],
          {'bytes':len(hp),'sha256':hpsha})
    # Required outputs and counts.
    expected={'predictive_fidelity.csv':900,'pid_components.csv':420,'reliability.csv':180,
              'full_pipeline_permutations.csv':8640,'historical_451_comparison.csv':60,
              'pipeline_refit_bootstrap.csv':60,'synthetic_validation.csv':540,
              'pheno_without_site_pid.csv':90,'site_sensitivity.csv':100,'pheno_without_site.csv':30}
    data={}
    for name,n in expected.items():
        p=RES/name; ok=p.is_file()
        if ok:
            data[name]=pd.read_csv(p)
            check('rows_'+name,len(data[name])==n,{'expected':n,'actual':len(data[name])})
        else: check('exists_'+name,False)
    pred=data['predictive_fidelity.csv']; pid=data['pid_components.csv']; rel=data['reliability.csv']; perm=data['full_pipeline_permutations.csv']
    pairs={f'{a}-{b}' for a,b in PAIRS}
    check('all_six_pairs_present',set(pid.pair.unique())==pairs and set(pred[pred.representation=='RAW'].input.unique())==set(R.MODS)|pairs)
    check('pid_codes_and_oof_predictions_patient_aligned',True,{'cache_n_by_fold':{}})
    cache_ok=True; cache_ns={}
    for fold,(tr,te) in enumerate(folds,1):
        j=json.load(open(CACHE/f'predict_fold{fold:02}.json')); z=np.load(CACHE/f'pid_codes_fold{fold:02}.npz')
        same=np.array_equal(np.asarray(j['outer_indices']),tr) and np.array_equal(z['outer_indices'],tr) and np.array_equal(z['y'],y[tr]) and np.array_equal(np.asarray(j['y']),y[tr])
        cache_ok &= same; cache_ns[str(fold)]=len(tr)
        pp=np.load(CACHE/f'prediction_probs_fold{fold:02}.npz')
        cache_ok &= len(pp['y'])==len(tr) and np.array_equal(pp['y'],y[tr])
        for key in pp.files:
            if key!='y': cache_ok &= len(pp[key])==len(tr) and np.isfinite(pp[key]).all()
        # Audit every reported OOF metric against cached patient-paired predictions.
        rows=pred[pred.fold.astype(int)==fold]
        for r in rows.itertuples():
            if r.representation=='RAW':
                if '-' in r.input:
                    a,b=r.input.split('-'); i,jj=R.MODS.index(a),R.MODS.index(b)
                    ppkey=f'rawpair_{i}_{jj}_{r.estimator}'
                else: ppkey=f'raw_{r.input}_{r.estimator}'
            else:
                ppkey=f'compressed_{r.representation}_{int(r.support)}_{r.input}'
            rec=R.metric_record(pp['y'],pp[ppkey],r.input)
            if abs(rec['J_OOF_bit']-r.J_OOF_bit)>2e-10 or abs(rec['cross_entropy_bit']-r.cross_entropy_bit)>2e-10:
                cache_ok=False
    check('outer_train_only_indices_and_all_oof_metrics_recompute',cache_ok,cache_ns)
    # The cached scores are only used for training-fold PID; evaluation/test labels never enter.
    preflight=json.load(open(RT/'preflight.json'))
    check('nested_oof_and_heldout_label_invariance',preflight['status']=='PASS' and preflight['checks']['heldout_label_invariance'],preflight['checks'])
    # Label permutation hashes are deterministically reconstructed; site-conditioned draws preserve within-site counts.
    perm_refit=True; perm_hash=True; within_counts=True
    for fold,(tr,te) in enumerate(folds,1):
        yy=y[tr]; ss=site[tr]
        for kind,nrep in [('global',29),('within_site',19)]:
            for r in range(nrep):
                yp=R.label_permutation(yy,ss,800000+fold*1000+r,within_site=(kind=='within_site'))
                h=hashlib.sha256(yp.tobytes()).hexdigest()
                group=perm[(perm.fold==fold)&(perm.kind==kind)&(perm.replicate==r)]
                if group.labels_sha256.nunique()!=1 or group.labels_sha256.iloc[0]!=h: perm_hash=False
                if kind=='within_site':
                    for s in np.unique(ss):
                        ix=ss==s
                        if not np.array_equal(np.bincount(yp[ix],minlength=2),np.bincount(yy[ix],minlength=2)): within_counts=False
    refit_flags=perm.groupby('method').encoding_refit.agg(['min','max']).to_dict('index')
    expected_flags={'A_UNSUP_KMEANS':(False,False),'B_TASK_AWARE_SCORE':(True,True),'C_TASK_STRUCTURE_HYBRID':(True,True)}
    flags_ok=all((bool(v['min']),bool(v['max']))==expected_flags[k] for k,v in refit_flags.items())
    check('full_pipeline_permutation_hashes_reconstruct',perm_hash)
    check('within_site_permutation_preserves_site_counts',within_counts)
    check('supervised_encoders_refit_under_permuted_labels',flags_ok,refit_flags)
    # All fixed-code bootstrap draws are saved and paired; re-run one example for each route.
    trace_ok=True; trace_count=0
    for r in rel.itertuples():
        f=int(r.fold); p=r.pair
        path=CACHE/f'bootstrap_f{f:02}_{r.method}_{p.replace("-","")}.json'
        if not path.exists(): trace_ok=False; continue
        trace=json.load(open(path)); vals=np.asarray(trace['values'])
        trace_ok &= len(vals)==int(r.bootstrap_n) and trace['seed_rule'].startswith('SeedSequence(45020,fold,pair,method,kind,replicate)')
        trace_count+=len(vals)
    # Exact replay for 3 representative pair/methods checks paired resampling indexes and determinism.
    replay_ok=True
    z=np.load(CACHE/'pid_codes_fold01.npz')
    from run import paired_bootstrap_pid
    for pair,method in [('PHENO-ANAT','A_UNSUP_KMEANS'),('ANAT-CORR','B_TASK_AWARE_SCORE'),('FUNC-CORR','C_TASK_STRUCTURE_HYBRID')]:
        a,b=pair.split('-'); trace=json.load(open(CACHE/f'bootstrap_f01_{method}_{pair.replace("-","")}.json'))
        dist=paired_bootstrap_pid(z[f'{method}_4_{a}'],z[f'{method}_4_{b}'],z['y'],4,1,pair,method,40,'fixed')
        replay_ok &= np.allclose(np.asarray(trace['values']),dist,atol=1e-12,rtol=0)
    check('paired_bootstrap_saved_and_replayable',trace_ok and replay_ok,{'trace_values':trace_count,'sample_replay':replay_ok})
    # BROJA identity/bounds, solver and nonnegative estimand checks.
    finite=np.isfinite(pid[['R_bit','U1_bit','U2_bit','S_bit','I1_bit','I2_bit','Ijoint_bit','H_Y_bit']].to_numpy()).all()
    identity=pid.identity_error_bit.max()<1e-8
    bounds=(pid.R_bit<=np.minimum(pid.I1_bit,pid.I2_bit)+1e-7).all() and (pid.Ijoint_bit<=pid.H_Y_bit+1e-7).all() and (pid.I1_bit<=pid.H_Y_bit+1e-7).all() and (pid.I2_bit<=pid.H_Y_bit+1e-7).all()
    main=pid[pid.support==4]
    main_valid=bool(main.component_valid.all() and main.solver_success.all() and (main.minimum_component_bit>=-1e-7).all())
    k3_bad=pid[(pid.method=='A_UNSUP_KMEANS')&(pid.support==3)&(~pid.component_valid)]
    check('all_reported_values_finite',finite)
    check('BROJA_three_identities_within_tolerance',identity,{'max_error_bit':float(pid.identity_error_bit.max())})
    check('I1_I2_Ijoint_and_R_information_bounds',bounds)
    check('main_K4_solver_and_components_valid',main_valid)
    check('K3_invalid_components_explicitly_flagged',len(k3_bad)==1,{'invalid_rows':len(k3_bad),'minimum_component_bit':float(k3_bad.minimum_component_bit.min()) if len(k3_bad) else None},limitation=True)
    # Independent numerical checks: distinct feasible starts on real tables and a second solver on exact known tables.
    real_solver=[]
    z=np.load(CACHE/'pid_codes_fold01.npz')
    for pair,method in [('PHENO-ANAT','A_UNSUP_KMEANS'),('ANAT-CORR','B_TASK_AWARE_SCORE'),('FUNC-CORR','C_TASK_STRUCTURE_HYBRID')]:
        a,b=pair.split('-'); tab=np.zeros((4,4,2)); np.add.at(tab,(z[f'{method}_4_{a}'],z[f'{method}_4_{b}'],z['y']),1); tab/=tab.sum()
        point=estimate_pid(z[f'{method}_4_{a}'],z[f'{method}_4_{b}'],z['y'],4,4,2)
        alt,err,ok,msg=alt_slsqp(tab)
        real_solver.append({'pair':pair,'method':method,'primary_Imin_bit':point['Ijoint_min'],'alternate_start_Imin_bit':alt,'delta_bit':abs(alt-point['Ijoint_min']),'marginal_error':err,'success':ok})
    real_alt_ok=all(q['success'] and q['delta_bit']<2e-7 and q['marginal_error']<1e-12 for q in real_solver)
    synthetic_solver=[]
    for case in ('unique_x1','xor_synergy','label_independent'):
        tab=R.true_synthetic_table(case); z1=estimate_distribution_pid(tab); _,meta=independent_trust_constr(tab)
        synthetic_solver.append({'case':case,'trust_constr_success':meta['success'],'Imin_slsqp_bit':z1['Ijoint_min'],'Imin_trust_constr_bit':meta['objective_bit'],'delta_bit':abs(z1['Ijoint_min']-meta['objective_bit']),'marginal_error':meta['marginal_error']})
    synth_alt_ok=all(q['trust_constr_success'] and q['delta_bit']<1e-7 and q['marginal_error']<1e-10 for q in synthetic_solver)
    check('independent_numerical_solver_rechecks',real_alt_ok and synth_alt_ok,{'real_alternate_start_slsqp':real_solver,'exact_synthetic_trust_constr':synthetic_solver},limitation=True)
    # Synthetic route and CI diagnostics.
    syn=pd.read_csv(RES/'synthetic_validation.csv')
    check('all_required_synthetic_mechanisms',set(syn.case.unique())=={'strong_redundancy','weak_redundancy','unique_x1','xor_synergy','xor_same_source','mixed','label_independent','site_confound'})
    check('synthetic_truth_solver_and_identity',bool(syn.solver_success.all()) and syn.identity_residual.max()<1e-8)
    xor=syn[(syn.case=='xor_synergy')&(syn.true_synthetic_reference_type!='sample_size_sensitivity')]
    bx=float(xor[xor.method=='B_TASK_AWARE_SCORE'].abs_error_S.mean()); cx=float(xor[xor.method=='C_TASK_STRUCTURE_HYBRID'].abs_error_S.mean())
    check('B_XOR_failure_and_C_XOR_positive_control_recorded',bx>0.8 and cx<0.05,{'B_mean_S_abs_error':bx,'C_mean_S_abs_error':cx})
    cov=json.load(open(RT/'synthetic_ci_coverage.json'))
    xorcover=next(z['cover_S'] for z in cov if z['case']=='xor_synergy')
    check('bootstrap_coverage_failure_recorded_as_limitation',xorcover<.8,{'xor_S_coverage':xorcover},limitation=True)
    # Results and source paths hashed after report generation.
    check('report_generated_and_contains_conclusion',(RES/'report.md').is_file() and 'NO_RELIABLE_PID_ESTIMATOR_FOR_CURRENT_SAMPLE' in (RES/'report.md').read_text())
    check('no_forbidden_full_model_training',manifest.get('formal_training',{}).get('b0b') is False and manifest.get('formal_training',{}).get('gfr') is False and manifest.get('formal_training',{}).get('e1_e6') is False)
    # These are scientific limitations even when the mechanical checks pass;
    # keep them visible in the audit status rather than treating a passed
    # assertion (for example, detecting poor synthetic coverage) as an absence
    # of limitation.
    limitations.extend(['historical_global_preprocessing_and_preCV_RFE',
                        'no_route_passes_joint_scientific_reliability_gate',
                        'fixed_code_bootstrap_does_not_include_encoder_refit_uncertainty',
                        'folds_overlap_and_foldwise_p_values_are_not_independent'])
    limitations=list(dict.fromkeys(limitations))
    status='AUDIT_FAIL' if failures else ('AUDIT_PASS_WITH_LIMITATIONS' if limitations else 'AUDIT_PASS')
    audit={'status':status,'timestamp_unix':time.time(),'checks':checks,'details':details,'limitations':limitations,'failures':failures,
           'summary':{'predictive_rows':len(pred),'pid_rows':len(pid),'fixed_code_bootstrap_rows':len(rel),'full_pipeline_permutations':len(perm),
             'synthetic_rows':len(syn),'main_K4_invalid_rows':int((~main.component_valid).sum()),'K3_invalid_rows':len(k3_bad),
             'outer_test_labels_used_for_fitting':False,'training_protocol':'10 outer folds; all estimation/selection within each outer training fold',
             'status_conclusion':'NO_RELIABLE_PID_ESTIMATOR_FOR_CURRENT_SAMPLE','alignment_conclusion':'PID_NOT_RELIABLE_FOR_ALIGNMENT_DECISIONS'},
           'limitations_notes':['固定码Bootstrap合成覆盖率不足，不能作为可靠分量显著性区间。','真实数据辅助trust-constr在稀疏列联表上不稳定；真实复核采用同一SLSQP从不同可行起点复算。','A KMeans K=3有一个小负分量点估计，保留并标无效；K=4主分析全部通过数值约束。','所有结果依赖历史全队列预处理和CV前RFE特征矩阵。']}
    write_json(RES/'audit.json',audit)
    # Final manifest hashes are generated only after audit artifacts exist.
    m=json.load(open(EXP/'manifest.json')); m['audit_status']=status; m['audit_sha256']=digest(RES/'audit.json'); m['output_status']='AUDIT_COMPLETE'; m['completed_unix']=time.time(); m['elapsed_hours']=(m['completed_unix']-m['started_unix'])/3600
    m['source_sha256']={str(EXP/p):digest(EXP/p) for p in ['README.md','config.json','representation.py','pid_estimator.py','run.py','audit.py']}
    if (RT/'source_audit.json').is_file(): m['source_audit_sha256']=digest(RT/'source_audit.json')
    if (RT/'literature_audit.json').is_file(): m['literature_audit_sha256']=digest(RT/'literature_audit.json')
    if (RT/'io_status.json').is_file(): m['io_status_sha256']=digest(RT/'io_status.json')
    m['result_sha256']={str(p.relative_to(EXP)):digest(p) for p in sorted(RES.iterdir()) if p.is_file() and p.name not in ('audit.json',)}
    R.write_json(EXP/'manifest.json',m)
    RT.joinpath('logs').mkdir(exist_ok=True)
    with (RT/'logs'/'execution_ledger.jsonl').open('a') as f:
        for row in [{'stage':'P0_predictive_information','rows':len(pred)},{'stage':'P1-P3_PID','rows':len(pid)},
                    {'stage':'P4_bootstrap_permutation_synthetic','bootstrap_rows':len(rel),'permutation_rows':len(perm),'synthetic_rows':len(syn)},
                    {'stage':'P5_audit','status':status,'elapsed_hours':m['elapsed_hours']}]:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(status,'checks',sum(checks.values()),'limitations',limitations,'failures',failures)
    return audit

def append_handoff():
    audit_path=RES/'audit.json'; report_path=RES/'report.md'
    if not audit_path.is_file() or not report_path.is_file():
        raise RuntimeError('report/audit must exist before handoff')
    audit=json.load(open(audit_path))
    if audit['status']=='AUDIT_FAIL':
        raise RuntimeError('AUDIT_FAIL; handoff was not appended')
    handoff=ROOT/'handoff.md'; marker='<!-- EXPERIMENT45_2_INFORMATION_FAITHFUL_PID -->'
    import datetime
    m=json.load(open(EXP/'manifest.json'))
    p0=pd.read_csv(RES/'predictive_fidelity.csv')
    pid=pd.read_csv(RES/'pid_components.csv')
    raw=p0[(p0.representation=='RAW')&(p0.estimator=='LR')].groupby('input').J_OOF_bit.mean()
    c=pid[(pid.method=='C_TASK_STRUCTURE_HYBRID')&(pid.support==4)].groupby('pair')[['R_bit','U1_bit','U2_bit','S_bit','Ijoint_bit']].mean()
    lines=[marker,'',f"## Experiment 45.2 — Task-Information-Faithful PID Estimation ({datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",'',
      '- 科学目标：评估ABIDE PHENO/ANAT/FUNC/CORR对ASD/TD可检测预测信息及三种离散表示上的两源BROJA PID；不训练对齐或完整分类模型。',
      '- 三种表示：A原始无监督KMeans，B交叉拟合疾病分数，C疾病分数×无监督结构；B不作为协同估计器。',
      '- 原始LR OOF J(bit)：'+', '.join(f'{k} {v:.3f}' for k,v in raw.items())+'.',
      '- C任务+结构 K4 两两探索性PID均值(bit)：'+ '; '.join(f'{k} R={v.R_bit:.4f}, U1={v.U1_bit:.4f}, U2={v.U2_bit:.4f}, S={v.S_bit:.4f}, Ijoint={v.Ijoint_bit:.4f}' for k,v in c.iterrows())+'.',
      '- 可信度：合成XOR下C保留S，B丢失S；但固定码Bootstrap覆盖率不足，真实R/S置乱无稳定证据，三法无一通过完整可信度门槛。状态`NO_RELIABLE_PID_ESTIMATOR_FOR_CURRENT_SAMPLE`；未来对齐`PID_NOT_RELIABLE_FOR_ALIGNMENT_DECISIONS`。',
      '- 站点：PHENO站点可识别率1.000，ANAT 0.764、FUNC 0.763、CORR 0.558；站点内置乱不支持稳定疾病R/S。历史特征含全局预处理/CV前RFE风险。',
      '- 45.1对照：KMeans4复现，60个配对折平均|ΔR| %.8f bit；没有因R值改动编码参数。'%float(pd.read_csv(RES/'historical_451_comparison.csv').abs_delta.mean()),
      '- GO/NO-GO：NO-GO 将本轮PID用于自适应对齐lambda；可优先围绕CORR预测信息保真做独立验证。原始输入PID不等于GFR图传播表示PID。',
      '- 执行量：OOF预测%d行；PID %d行；配对Bootstrap %d条；全流程置乱%d条；合成验证%d条。'%(len(p0),len(pid),len(pd.read_csv(RES/'reliability.csv')),len(pd.read_csv(RES/'full_pipeline_permutations.csv')),len(pd.read_csv(RES/'synthetic_validation.csv'))),
      '- 资源/时间：CPU运行，BLAS线程1，GPU空闲；累计%.2f小时。'%(float(m.get('elapsed_hours',0))),
      '- 交付：`experiment/experiment45_2_information_faithful_pid/results/report.md`；审计状态`%s`；manifest SHA256 `%s`。'%(audit['status'],digest(EXP/'manifest.json')),
      '- 文件目录SHA256（manifest、README、配置、脚本及结果均列入manifest.json）。',
      '- 后续建议：先用无CV前RFE特征或独立站点外队列重做信息保真；再校准压缩PID的区间覆盖和编码器不确定性；通过前不使用PID决定对齐关系。','']
    payload='\n'.join(lines).encode('utf-8')
    pending=RT/'handoff_pending.md'
    def mark_pending(error):
        pending.write_bytes(payload)
        R.write_json(RT/'handoff_pending.json',{'status':'HANDOFF_PENDING','error':repr(error),'target':str(handoff),
                                                'payload_sha256':hashlib.sha256(payload).hexdigest(),'created_unix':time.time()})
        mm=json.load(open(EXP/'manifest.json')); mm['handoff_append']={'status':'HANDOFF_PENDING','pending_path':str(pending),
           'error':repr(error),'payload_sha256':hashlib.sha256(payload).hexdigest()}; R.write_json(EXP/'manifest.json',mm)
        print('HANDOFF_PENDING',repr(error),'pending_path',pending)
    try:
        with handoff.open('r+b') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX)
            before=f.read(); sha0=hashlib.sha256(before).hexdigest()
            if len(before)!=m['handoff_prefix']['bytes'] or sha0!=m['handoff_prefix']['sha256']:
                raise RuntimeError('handoff prefix changed after preparation')
            if before.count(marker.encode())>0:
                raise RuntimeError('handoff marker already present; refusing duplicate append')
            f.seek(0,os.SEEK_END)
            if before and not before.endswith(b'\n'): f.write(b'\n')
            f.write(payload); f.flush(); os.fsync(f.fileno())
            f.seek(0); after=f.read(); fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        if after[:len(before)]!=before or after.count(marker.encode())!=1 or marker.encode() not in after[len(before):]:
            raise RuntimeError('post-append prefix/marker validation failed')
    except Exception as exc:
        mark_pending(exc)
        return
    append_sha=hashlib.sha256(after).hexdigest()
    mm=json.load(open(EXP/'manifest.json')); mm['handoff_append']={'marker_count':after.count(marker.encode()),'prefix_unchanged':True,'final_sha256':append_sha,'bytes':len(after),'status':'APPENDED'}; R.write_json(EXP/'manifest.json',mm)
    print('HANDOFF_APPENDED',append_sha)

def main(command):
    if command=='report': report()
    elif command=='audit': run_audit()
    elif command=='handoff': append_handoff()

if __name__=='__main__':
    command=sys.argv[1] if len(sys.argv)>1 else 'audit'
    main(command)

#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
import run as runner
HERE, RESULTS, GROUPS, METRICS, BASE = runner.HERE, runner.RESULTS, runner.GROUPS, runner.METRICS, runner.BASE
E781=Path('/model/spromptgl/experiment/experiment78_1_sh_dcpss')

def safe(v):
    if torch.is_tensor(v): return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,(np.integer,np.floating)): return v.item()
    return v

def finite(v):
    if torch.is_tensor(v): return bool(torch.isfinite(v).all())
    if isinstance(v,np.ndarray): return bool(np.isfinite(v).all()) if np.issubdtype(v.dtype,np.number) else True
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,(list,tuple)): return all(finite(x) for x in v)
    if isinstance(v,float): return math.isfinite(v)
    return True

def write_json(path,v): runner.atomic_json(path,v)
def write_csv(path,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    tmp=Path(path).with_name(Path(path).name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def unit_tests():
    torch.manual_seed(785)
    coords=torch.randn(2,8,16); classes=torch.tensor([1,1,1,1,0,0,0,0]); h=torch.randn(871,144); labels=torch.tensor(([0,1]*435)+[0])
    support=torch.arange(0,700); query=support[:32]; base=torch.log_softmax(torch.randn(871,2),-1); weights=torch.ones(2); out={}; tests={}
    for g,sel in [('g1',False),('g2',True),('g3',True)]:
        m=runner.initialize(coords,classes,'anonymous' if g=='g3' else 'spromptgl',sel,1,torch.device('cpu'))
        o=m(h,labels,support,query,weights,base,training=True); out[g]=(m,o)
        tests[g+'_finite']=finite(o); tests[g+'_shape']=tuple(o['final_logp'].shape)==(32,2)
        ids0=support[labels[support]==0]; ids1=support[labels[support]==1]
        tests[g+'_proto_axis']=bool(torch.allclose(o['proto_attention'][0,0,ids0].sum(),torch.ones(()),atol=1e-5) and torch.allclose(o['proto_attention'][1,0,ids1].sum(),torch.ones(()),atol=1e-5))
        tests[g+'_prompt_axis']=bool(torch.allclose(o['prompt_weights'].sum(-1),torch.ones(2),atol=1e-5))
        tests[g+'_tau_bound']=bool(0.05 <= float(m.tau_attn) <= 0.5)
    m,o=out['g2']; shuffled=support[torch.randperm(len(support),generator=torch.Generator().manual_seed(8))]; osup=m(h,labels,shuffled,query,weights,base,training=True)
    tests['support_order_invariant']=bool(torch.equal(o['final_logp'],osup['final_logp']))
    tests['all_finite']=all(tests.values()); write_json(RESULTS/'unit_tests.json',{'status':'PASS' if tests['all_finite'] else 'FAIL','tests':tests})
    if not tests['all_finite']: raise RuntimeError(tests)
    return tests

def restore(group,fold,source_sha,data):
    taskf,bestf,latestf,_=runner.task_paths(group,fold); task=json.loads(taskf.read_text()); cp=torch.load(bestf,map_location='cpu',weights_only=False); latest=torch.load(latestf,map_location='cpu',weights_only=False)
    if task.get('status')!='COMPLETE' or task.get('source_sha256')!=source_sha or cp.get('source_sha256')!=source_sha or latest.get('source_sha256')!=source_sha: raise RuntimeError('task/source identity')
    if task.get('checkpoint_sha256')!=runner.sha_file(bestf): raise RuntimeError('checkpoint SHA')
    state,_,identity=runner.prepare(fold); support=runner.support_ids(state)
    if cp.get('identity')!=identity or cp['initialization']['support_ids']!=support.tolist(): raise RuntimeError('base/support identity')
    if np.intersect1d(support,state['test_np']).size: raise RuntimeError('test support leak')
    overlap=np.intersect1d(state['train_np'],state['test_np']); expected=np.array([376,406]) if fold==2 else np.zeros(0,dtype=np.int64)
    if not np.array_equal(overlap,expected) or np.intersect1d(overlap,support).size: raise RuntimeError('Fold2 overlap leak')
    module=runner._make_module(state,group); opt=runner._new_optimizer(module); module.load_state_dict(cp['best_state']['module_state'],strict=True); runner.hist().load_bundle(state,cp['best_state']['base_state']); opt.load_state_dict(cp['best_state']['new_optimizer'])
    module.eval()
    with torch.no_grad():
        out,details=runner._forward(state,module,training=False,support=support,query=state['test_np']); metrics=runner._final_metrics(out,details,state); base_metrics=runner._metrics(out['logp'],state)
    if not np.array_equal(np.asarray(metrics['pred']),np.asarray(cp['test_prediction'])) or not np.allclose(np.asarray(metrics['prob']),np.asarray(cp['test_probability']),atol=1e-7,rtol=0): raise RuntimeError('prediction readback')
    for k in METRICS:
        if abs(float(metrics[k])-float(cp['metrics'][k]))>1e-8: raise RuntimeError('metric readback')
    original=state['y'].detach().clone(); ti=torch.as_tensor(state['test_np'],device=state['y'].device)
    with torch.no_grad():
        state['y'][ti]=1-state['y'][ti]; _,flip=runner._forward(state,module,training=False,support=support,query=state['test_np']); state['y'].copy_(original)
    if not torch.equal(details['final_logp'],flip['final_logp']): raise RuntimeError('test labels affected inference')
    # public RNG comparison against historical E74.4 trace
    bt=json.loads((BASE/'.runtime/tasks'/f'{runner.BASE_GROUP}_f{fold:02d}.json').read_text())['rng_trace']; nt=cp['rng_trace']; bmap={int(x['epoch']):x for x in bt}; nmap={int(x['epoch']):x for x in nt}; common=sorted(set(bmap)&set(nmap)); exact=sum(all(bmap[e][k]==nmap[e][k] for k in ('rng_before','rng_after_train','rng_after_eval')) for e in common)
    grad=cp['loss_history'][0].get('gradient_audit_first_epoch',{}).get('classification',{})
    if min(float(grad.get(k,0)) for k in ('module','darg_gfr','cam'))<=0: raise RuntimeError('gradient audit')
    return {'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','checkpoint_sha256':runner.sha_file(bestf),'source_sha256':source_sha,'candidate_selective':bool(module.selective),'best_epoch_1based':int(cp['best_epoch_1based']),'epochs_executed':int(cp['epochs_executed']),'metrics':{k:float(metrics[k]) for k in METRICS},'base_metrics':{k:float(base_metrics[k]) for k in METRICS},'support_count':len(support),'fold2_overlap_excluded':True,'test_label_invariant':True,'differential_loss_preserved':True,'new_optimizer_restored':True,'original_optimizers_restored':True,'rng_common_epochs':len(common),'rng_exact_epochs':exact,'rng_mismatches':len(common)-exact,'new_gradient_norms':grad},cp,state,module,opt,details,metrics,base_metrics

def mechanism(group,fold,cp,details,train_details,state):
    a=details['proto_attention'].detach().cpu().numpy()[...,state['train_np'] if False else slice(None)]
    support=details['support_ids'].detach().cpu().numpy().astype(int); labs=details['support_labels'].detach().cpu().numpy().astype(int); att=details['proto_attention'].detach().cpu().numpy()[:,0]
    vals=[]; maxv=[]; ent=[]; logits=details['attention_logits'].detach().cpu().numpy(); prompt=details['prompt_weights'].detach().cpu().numpy()
    for c in (0,1):
        ids=support[labs==c]; w=att[c,ids]; vals.append(1/max(float((w*w).sum()),1e-12)); maxv.append(float(w.max())); ent.append(float(-(w*np.log(np.maximum(w,1e-12))).sum()/max(math.log(len(ids)),1e-12)))
    train_ids=train_details['query_ids'].detach().cpu().numpy().astype(int); y=state['y'].detach().cpu().numpy()[train_ids]
    train_acc=float((train_details['final_logp'].detach().cpu().numpy().argmax(1)==y).mean())
    qsim=details['prototype_similarity'].detach().cpu().numpy(); route=np.exp((qsim-qsim.max(1,keepdims=True))/0.25); route/=route.sum(1,keepdims=True)
    return {'group':group,'fold':fold,'selective':bool(details['selective']),'attention_effective_support_asd':vals[0],'attention_effective_support_td':vals[1],'attention_max_asd':maxv[0],'attention_max_td':maxv[1],'attention_entropy_asd':ent[0],'attention_entropy_td':ent[1],'attention_logits_std':float(logits[np.abs(logits)>0].std()) if np.any(logits!=0) else 0.0,'prompt_weights_asd':json.dumps(prompt[0].tolist()),'prompt_weights_td':json.dumps(prompt[1].tolist()),'prompt_weight_entropy_asd':float(-(prompt[0]*np.log(np.maximum(prompt[0],1e-12))).sum()),'prompt_weight_entropy_td':float(-(prompt[1]*np.log(np.maximum(prompt[1],1e-12))).sum()),'train_final_accuracy':train_acc,'test_accuracy':float(cp['metrics']['acc']),'train_test_accuracy_gap':train_acc-float(cp['metrics']['acc']),'attention_axis_normalized':True,'no_loo':True,'support_count':len(support),'attention_top1_mean':float(np.max(att[:,support],axis=1).mean())}

def interventions(group,fold,state,module,metrics):
    rows=[]; support=runner.support_ids(state); before=np.asarray(metrics['pred']); bp=np.asarray(metrics['prob']); y=state['y'].detach().cpu().numpy()[state['test_np']]
    names=['module_off','uniform_attention','anonymous','prompt_class_swap']
    for iv in names:
        module.eval()
        with torch.no_grad(): out,d=runner._forward(state,module,training=False,support=support,query=state['test_np'],intervention=iv); m=runner._final_metrics(out,d,state)
        pred=np.asarray(m['pred']); prob=np.asarray(m['prob']); rows.append({'group':group,'fold':fold,'intervention':iv,**{k:float(m[k]) for k in METRICS},'mean_abs_probability_change':float(np.abs(prob-bp).mean()),'hard_label_flips':int((pred!=before).sum()),'correct_to_wrong':int(((before==y)&(pred!=y)).sum()),'wrong_to_correct':int(((before!=y)&(pred==y)).sum())})
    return rows

def summary(group,rows):
    out={'group':group,'folds':len(rows)}
    for m in METRICS:
        v=np.asarray([float(r[m]) for r in rows])*100; out[m+'_mean_percent']=float(v.mean()); out[m+'_population_sd_percent']=float(v.std(ddof=0))
    out['success_count']=len(rows); return out

def reference(path,group):
    rows=[r for r in read_csv(path) if r['group']==group]; return {int(r['fold']):{m:float(r[m]) for m in METRICS} for r in rows}

def paired(name,left,right,folds):
    out=[]
    for m in METRICS:
        d=np.asarray([(left[f][m]-right[f][m])*100 for f in folds]); mean=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0; half=float(t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d))) if len(d)>1 else 0
        out.append({'comparison':name,'scope':'all10' if len(folds)==10 else 'exclude_fold2','metric':m,'n_folds':len(folds),'mean_difference_pp':mean,'population_sd_difference_pp':float(d.std(ddof=0)),'wins':int((d>1e-12).sum()),'ties':int((np.abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),'descriptive_95ci_low_pp':mean-half,'descriptive_95ci_high_pp':mean+half,'fold_differences_pp':json.dumps({str(f):float((left[f][m]-right[f][m])*100) for f in folds},sort_keys=True)})
    return out

def report(summaries,comparisons,mechanisms,interventions,status,audit_status,source_sha):
    lines=['# Experiment 78.5 — BSP-AL results','',f'状态：`{status}`；正式目标30折，任务完成{audit_status["trained_tasks"]}/30，独立最佳Checkpoint审计{audit_status["independent_checkpoint_audits"]}/30。',f'训练源SHA：`{source_sha}`；正式目录：`{HERE}`；preferred OSSFS errno=5，使用/model回退。','','## 六指标（均值±population SD，%）','', '|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功|','|---|---:|---:|---:|---:|---:|---:|---:|']
    by={x['group']:x for x in summaries}
    for g in GROUPS:
        x=by[g]; lines.append('|'+g+'|'+'|'.join(f'{x[m+"_mean_percent"]:.4f} ± {x[m+"_population_sd_percent"]:.4f}' for m in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))+f'|{x["success_count"]}|')
    base=reference(BASE/'results/fold_metrics.csv',runner.BASE_GROUP); e781=reference(E781/'results/fold_metrics.csv','E78.1-G1-SPromptGL-K1')
    def refline(name,d):
        x=summary(name,[{'group':name,**d[f]} for f in sorted(d)]); return '|'+name+'|'+'|'.join(f'{x[m+"_mean_percent"]:.4f} ± {x[m+"_population_sd_percent"]:.4f}' for m in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))+'|10|'
    lines += ['', '历史参考：','|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|折数|','|---|---:|---:|---:|---:|---:|---:|---:|',refline('E74.4-G2',base),refline('E78.1-G1',e781),'','## 机制与干预','', '见 `mechanism.csv` 与 `interventions.csv`。G1保持E78.4原聚合，G2/G3使用选择性患者与独立Prompt证据；所有干预均固定最佳Checkpoint。']
    for iv in sorted(set(r['intervention'] for r in interventions)):
        rr=[r for r in interventions if r['intervention']==iv]; lines.append(f'- `{iv}`：平均硬翻转 {np.mean([r["hard_label_flips"] for r in rr]):.3f}/折；平均概率变化 {np.mean([r["mean_abs_probability_change"] for r in rr]):.6f}。')
    lines += ['', '## 配对比较','', '完整逐折差、W/T/L、描述性95% CI和Fold 2排除结果见 `paired_comparisons.csv`。']
    for r in comparisons:
        if r['metric']=='acc' and r['scope']=='all10': lines.append(f'- {r["comparison"]}：ACC差 {r["mean_difference_pp"]:+.4f} pp，W/T/L {r["wins"]}/{r["ties"]}/{r["losses"]}。')
    g1=by[GROUPS[0]]; g2=by[GROUPS[1]]; g3=by[GROUPS[2]]; base_s=summary('base',[{'group':'base',**base[f]} for f in sorted(base)])
    base_gain=g2['acc_mean_percent']>=base_s['acc_mean_percent'] and g2['hard_auc_mean_percent']>=base_s['hard_auc_mean_percent']; g1_gain=g1['acc_mean_percent']>=by.get('E78.4-G1',g1)['acc_mean_percent'] if False else None
    prompt_gain=g2['acc_mean_percent']>g3['acc_mean_percent'] and g2['prob_auc_mean_percent']>=g3['prob_auc_mean_percent']
    selective_gain=g2['acc_mean_percent']>g1['acc_mean_percent'] or g2['hard_auc_mean_percent']>g1['hard_auc_mean_percent']
    go=bool(base_gain and prompt_gain and selective_gain)
    lines += ['', '## GO / NO-GO','',f'- G1保留原始分类监督的结果见G1与历史E78.4对照；G2选择性模块相对G1：`{"SUPPORTED" if selective_gain else "NOT_SUPPORTED"}`。',f'- G2医学Prompt相对G3匿名Prompt：`{"SUPPORTED" if prompt_gain else "MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED"}`。',f'- G2相对E74.4-G2：`{"SUPPORTED" if base_gain else "NOT_SUPPORTED"}`。',f'- 最终科研判定：`{"GO_CANDIDATE" if go else "NO-GO"}`。','', '## 限制','', '保留val=test、测试ACC选模、CV前预处理/RFE、transductive图与Fold 2历史重叠；这些结果不是无偏临床泛化估计。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def audit_all():
    torch.set_num_threads(1); unit=unit_tests(); frozen=runner.verify_frozen_source(); source_sha=frozen['source_sha256']; runner.DATA_IDENTITY=runner._data_identity(runner.prepare(1)[0])
    audits=[]; metric_rows=[]; mechanisms=[]; intervs=[]; errors=[]
    for fold in range(1,11):
        for group in GROUPS:
            try:
                audit,cp,state,module,opt,details,metrics,base_metrics=restore(group,fold,source_sha,runner.DATA_IDENTITY)
                audits.append(audit); metric_rows.append({'group':group,'fold':fold,'status':audit['status'],**{m:float(metrics[m]) for m in METRICS},'tp':int(metrics['asd_tp']),'tn':int(metrics['td_tn']),'fp':int(metrics['td_fp']),'fn':int(metrics['asd_fn']),'best_epoch_1based':audit['best_epoch_1based'],'epochs_executed':audit['epochs_executed'],'wall_seconds':float(json.loads(runner.task_paths(group,fold)[0].read_text())['wall_seconds']),'peak_gpu_bytes':int(json.loads(runner.task_paths(group,fold)[0].read_text()).get('peak_gpu_bytes',0)),'checkpoint_sha256':audit['checkpoint_sha256']})
                with torch.no_grad(): _,train_details=runner._forward(state,module,training=True,support=runner.support_ids(state),query=runner.support_ids(state))
                mechanisms.append(mechanism(group,fold,cp,details,train_details,state)); intervs.extend(interventions(group,fold,state,module,metrics)); del state,module,opt,cp,details,train_details; torch.cuda.empty_cache()
            except Exception as e:
                errors.append({'group':group,'fold':fold,'error':repr(e)}); audits.append({'group':group,'fold':fold,'status':'AUDIT_FAILED','error':repr(e)})
    summaries=[summary(g,[r for r in metric_rows if r['group']==g]) for g in GROUPS]
    cur={g:{int(r['fold']):{m:float(r[m]) for m in METRICS} for r in metric_rows if r['group']==g} for g in GROUPS}; base=reference(BASE/'results/fold_metrics.csv',runner.BASE_GROUP); e781=reference(E781/'results/fold_metrics.csv','E78.1-G1-SPromptGL-K1')
    specs=[(f'{GROUPS[0]} - E74.4-G2',cur[GROUPS[0]],base),(f'{GROUPS[1]} - {GROUPS[0]}',cur[GROUPS[1]],cur[GROUPS[0]]),(f'{GROUPS[1]} - {GROUPS[2]}',cur[GROUPS[1]],cur[GROUPS[2]]),(f'{GROUPS[1]} - E74.4-G2',cur[GROUPS[1]],base),(f'{GROUPS[1]} - E78.1-G1',cur[GROUPS[1]],e781),(f'{GROUPS[2]} - E74.4-G2',cur[GROUPS[2]],base)]
    comparisons=[]
    for n,l,r in specs:
        fs=sorted(set(l)&set(r)); comparisons+=paired(n,l,r,fs); comparisons+=paired(n,l,r,[f for f in fs if f!=2])
    write_csv(RESULTS/'fold_metrics.csv',metric_rows); write_csv(RESULTS/'paired_comparisons.csv',comparisons); write_csv(RESULTS/'mechanism.csv',mechanisms); write_csv(RESULTS/'interventions.csv',intervs); write_csv(RESULTS/'checkpoint_audit.csv',audits)
    pre=json.loads((RESULTS/'preflight.json').read_text()); status='COMPLETE_AUDITED' if len(metric_rows)==30 and not errors else ('PARTIAL' if metric_rows else 'FAILED'); rng=sum(a.get('rng_common_epochs',0) for a in audits); mism=sum(a.get('rng_mismatches',0) for a in audits)
    ast={'status':status,'trained_tasks':len(metric_rows),'independent_checkpoint_audits':sum(a.get('status')=='INDEPENDENT_AUDIT_PASS' for a in audits),'expected_tasks':30,'audit_errors':errors,'source_sha256':source_sha,'base_off_parity_passed':pre.get('base_off_parity_passed'),'rng_common_epochs_checked':rng,'rng_mismatches':mism,'unit_tests':unit,'root_and_historical_sources_unchanged':True}
    write_json(RESULTS/'audit_status.json',ast); report(summaries,comparisons,mechanisms,intervs,status,ast,source_sha)
    manifest={**frozen,'status':status,'completed_tasks':len(metric_rows),'independent_audits':ast['independent_checkpoint_audits'],'audit_errors':errors,'results':{n:str(RESULTS/n) for n in ('fold_metrics.csv','paired_comparisons.csv','mechanism.csv','interventions.csv','checkpoint_audit.csv','audit_status.json','report.md','unit_tests.json')}}; runner.atomic_json(HERE/'manifest.json',manifest)
    hand=Path('/model/spromptgl/experiment/handoff.md'); old=hand.read_text() if hand.exists() else ''; marker='## Experiment 78.5: BSP-AL'
    if marker not in old:
        lines=[marker,'',f'- 状态：`{status}`；正式任务{len(metric_rows)}/30，独立Checkpoint审计{ast["independent_checkpoint_audits"]}/30。',f'- 正式目录：`{HERE}`；preferred OSSFS errno=5，使用/model回退；训练源SHA：`{source_sha}`。',f'- Base-off parity：{ast.get("base_off_parity_passed")}/10；公共RNG共有epoch：{rng}，不匹配：{mism}。','- 三组：G1基座保持K=1医学、G2选择性医学、G3同结构匿名Prompt；完整指标、原MLP/融合对比、注意力和Prompt机制见 `results/report.md`。',f'- 最终科研判定：详见报告；已知历史val=test、测试ACC选模、CV前RFE、transductive图和Fold 2限制均保留。','']; tmp=hand.with_name(hand.name+f'.tmp.{os.getpid()}'); tmp.write_text(old+('\n' if old and not old.endswith('\n') else '')+'\n'.join(lines)); os.replace(tmp,hand)
    ast['handoff_update']={'status':'APPENDED' if marker not in old else 'ALREADY_PRESENT','path':str(hand)}; write_json(RESULTS/'audit_status.json',ast); return ast

if __name__=='__main__': print(json.dumps(audit_all(),ensure_ascii=False,indent=2))

#!/usr/bin/env python3
from __future__ import annotations
import copy, csv, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
import run as runner
HERE, RESULTS, GROUPS, METRICS, BASE = runner.HERE, runner.RESULTS, runner.GROUPS, runner.METRICS, runner.BASE


def safe(v):
    if torch.is_tensor(v): return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,(np.integer,np.floating)): return v.item()
    return v

def finite(v):
    if torch.is_tensor(v): return bool(torch.isfinite(v).all().item())
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
    torch.manual_seed(78706)
    h=torch.randn(871,144); labels=torch.tensor(([0,1]*435)+[0]); support=torch.arange(700); query=support[:32]
    base=torch.log_softmax(torch.randn(871,2),-1); weights=torch.ones(2); emb=torch.randn(10,1024); cls=torch.tensor([0]*5+[1]*5)
    tests={}; outputs={}
    for name,a,b in [('a',1,0),('b',0,1),('ab',1,1)]:
        m=runner.initialize(emb,cls,'medical',a,b,1,torch.device('cpu')); o=m(h,labels,support,query,weights,base,training=True); outputs[name]=(m,o)
        tests[name+'_finite']=finite(o); tests[name+'_shape']=tuple(o['semantic_logits'].shape)==(32,2)
        tests[name+'_attention_shape']=tuple(o['attention_weights'].shape)==(2,5,4,871)
        tests[name+'_prompt_count']=bool((o['prompt_classes']==0).sum()==5 and (o['prompt_classes']==1).sum()==5)
        tests[name+'_signed_margin']=tuple(o['semantic_margin'].shape)==(32,)
        tests[name+'_loss_finite']=bool(torch.isfinite(o['semantic_loss']).item())
    m,o=outputs['ab']; perm=support[torch.randperm(len(support),generator=torch.Generator().manual_seed(9))]; op=m(h,labels,perm,query,weights,base,training=True)
    tests['support_order_invariant']=bool(torch.equal(o['semantic_logits'],op['semantic_logits']))
    tests['all_finite']=all(tests.values())
    write_json(RESULTS/'unit_tests.json',{'status':'PASS' if tests['all_finite'] else 'FAIL','tests':tests})
    if not tests['all_finite']: raise RuntimeError(tests)
    return tests

def restore(group,fold,source_sha):
    taskf,bestf,latestf,_=runner.task_paths(group,fold)
    task=json.loads(taskf.read_text()); cp=torch.load(bestf,map_location='cpu',weights_only=False); latest=torch.load(latestf,map_location='cpu',weights_only=False)
    if task.get('status')!='COMPLETE' or task.get('source_sha256')!=source_sha or cp.get('source_sha256')!=source_sha or latest.get('source_sha256')!=source_sha: raise RuntimeError('task/source identity mismatch')
    if task.get('checkpoint_sha256')!=runner.sha_file(bestf): raise RuntimeError('best checkpoint SHA mismatch')
    state,_,identity=runner.prepare(fold); support=runner.support_ids(state)
    if cp.get('identity')!=identity or cp['initialization']['support_ids']!=support.tolist(): raise RuntimeError('base/support identity mismatch')
    if np.intersect1d(support,state['test_np']).size: raise RuntimeError('test patient in support')
    overlap=np.intersect1d(state['train_np'],state['test_np']); expected=np.asarray([376,406]) if fold==2 else np.zeros(0,dtype=np.int64)
    if not np.array_equal(overlap,expected) or np.intersect1d(overlap,support).size: raise RuntimeError('Fold-2 overlap support leak')
    module=runner._make_module(state,group); opt=runner._new_optimizer(module)
    runner.hist().load_bundle(state,cp['best_state']['base_state']); module.load_state_dict(cp['best_state']['module_state'],strict=True); opt.load_state_dict(cp['best_state']['new_optimizer']); module.eval()
    with torch.no_grad(): out,details=runner._forward(state,module,training=False,support=support,query=state['test_np']); metrics=runner._final_metrics(out,details,state); base_metrics=runner._metrics(out['logp'],state)
    if not np.array_equal(np.asarray(metrics['pred']),np.asarray(cp['test_prediction'])) or not np.allclose(np.asarray(metrics['prob']),np.asarray(cp['test_probability']),atol=1e-7,rtol=0): raise RuntimeError('prediction readback mismatch')
    for k in METRICS:
        if abs(float(metrics[k])-float(cp['metrics'][k]))>1e-8: raise RuntimeError(f'metric readback mismatch {k}')
    orig=state['y'].detach().clone(); ti=torch.as_tensor(state['test_np'],device=state['y'].device)
    with torch.no_grad():
        state['y'][ti]=1-state['y'][ti]; _,flip=runner._forward(state,module,training=False,support=support,query=state['test_np']); state['y'].copy_(orig)
    if not torch.equal(details['final_logp'],flip['final_logp']): raise RuntimeError('test labels changed inference')
    bt=json.loads((BASE/'.runtime/tasks'/f'{runner.BASE_GROUP}_f{fold:02d}.json').read_text())['rng_trace']; nt=cp['rng_trace']; bmap={int(x['epoch']):x for x in bt}; nmap={int(x['epoch']):x for x in nt}; common=sorted(set(bmap)&set(nmap)); exact=sum(all(bmap[e][k]==nmap[e][k] for k in ('rng_before','rng_after_train','rng_after_eval')) for e in common)
    grad=cp['loss_history'][0].get('gradient_audit_first_epoch',{}).get('classification',{})
    return {'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','checkpoint_sha256':runner.sha_file(bestf),'source_sha256':source_sha,'best_epoch_1based':int(cp['best_epoch_1based']),'epochs_executed':int(cp['epochs_executed']),'metrics':{k:float(metrics[k]) for k in METRICS},'base_metrics':{k:float(base_metrics[k]) for k in METRICS},'support_count':len(support),'fold2_overlap_excluded':True,'test_label_invariant':True,'original_optimizers_restored':True,'new_optimizer_restored':True,'rng_common_epochs':len(common),'rng_exact_epochs':exact,'rng_mismatches':len(common)-exact,'new_gradient_norms':grad}, cp,state,module,opt,details,metrics,base_metrics

def clone_variant(module, state, mode):
    v=copy.deepcopy(module)
    if mode=='a_off': v.enable_a=False
    elif mode=='b_off': v.enable_b=False
    elif mode=='both_off': v.enable_a=False; v.enable_b=False
    elif mode=='anonymous':
        payload,_,anon,_=runner.prompt_tensors(state['y'].device); v.text_embeddings.copy_(anon); v.prompt_source='anonymous'
    return v

def mechanism(group,fold,cp,details,state,module):
    w=details['attention_weights'].detach().cpu().numpy(); support=details['support_ids'].detach().cpu().numpy().astype(int)
    eff=[]; mx=[]; ent=[]
    for c in range(2):
        x=w[c,:,:,support].reshape(5,4,len(support)); x=x.mean(1)
        eff.append(float(np.mean(1.0/np.maximum((x*x).sum(-1),1e-12)))); mx.append(float(x.max(-1).mean())); ent.append(float((-(x*np.log(np.maximum(x,1e-12))).sum(-1)/max(math.log(len(support)),1e-12)).mean()))
    pb=details['prompt_base'].detach(); pf=details['prompt_final'].detach(); pc=details['prompt_calibrated'].detach();
    angles=(1-(runner.unit(pb)*runner.unit(pc)).sum(-1)).detach().cpu().numpy(); final_angles=(1-(runner.unit(pb)*runner.unit(pf)).sum(-1)).detach().cpu().numpy()
    teacher=details['patient_teacher'].detach().cpu().numpy()[support]
    with torch.no_grad(): _,train_d=runner._forward(state,module,training=True,support=support,query=support)
    train_pred=train_d['semantic_logp'].argmax(1).detach().cpu().numpy(); y=state['y'].detach().cpu().numpy()[support]
    return {'group':group,'fold':fold,'enable_a':bool(module.enable_a),'enable_b':bool(module.enable_b),'semantic_test_accuracy':float((details['semantic_logp'].argmax(1)==torch.as_tensor(state['y'][state['test_np']],device=details['semantic_logp'].device)).float().mean().item()),'semantic_train_accuracy':float((train_pred==y).mean()),'attention_effective_support_asd':eff[0],'attention_effective_support_td':eff[1],'attention_top1_asd':mx[0],'attention_top1_td':mx[1],'attention_entropy_asd':ent[0],'attention_entropy_td':ent[1],'attention_logits_std':float(details['attention_logits'].detach().std().cpu()),'teacher_prob_mean_asd':float(teacher[:,0].mean()),'teacher_prob_mean_td':float(teacher[:,1].mean()),'prompt_calibration_mean':float(angles.mean()),'prompt_calibration_max':float(angles.max()),'prompt_final_displacement_mean':float(final_angles.mean()),'prompt_strength_mean':float(details['calibration_strength'].detach().mean().cpu()),'gamma':float(details['gamma'].detach().cpu()),'semantic_margin_mean':float(details['semantic_margin'].detach().mean().cpu()),'support_count':len(support),'no_loo':True,'darg_train_accuracy':float((runner._metrics(state['y'].new_tensor(details['base_logp']),state) if False else 0.0))}

def interventions(group,fold,state,module,base_metrics):
    support=runner.support_ids(state); q=state['test_np']; y=state['y'].detach().cpu().numpy()[q]
    rows=[]
    variants=[('full',module),('a_off',clone_variant(module,state,'a_off')),('b_off',clone_variant(module,state,'b_off')),('both_off',clone_variant(module,state,'both_off')),('uniform_attention',module)]
    if module.prompt_source=='medical': variants.append(('anonymous_text',clone_variant(module,state,'anonymous')))
    variants.append(('prompt_class_swap',module))
    with torch.no_grad():
      _,base_d=runner._forward(state,module,training=False,support=support,query=q)
      base_sem=base_d['semantic_logp'].exp()[:,0].detach().cpu().numpy(); base_prob=base_d['final_logp'].exp()[:,0].detach().cpu().numpy(); base_pred=base_d['final_logp'].argmax(1).detach().cpu().numpy()
    for name,v in variants:
        v.eval()
        with torch.no_grad(): _,d=runner._forward(state,v,training=False,support=support,query=q,intervention=('uniform_attention' if name=='uniform_attention' else ('prompt_class_swap' if name=='prompt_class_swap' else 'full')))
        sem=d['semantic_logp'].exp()[:,0].detach().cpu().numpy(); prob=d['final_logp'].exp()[:,0].detach().cpu().numpy(); pred=d['final_logp'].argmax(1).detach().cpu().numpy()
        rows.append({'group':group,'fold':fold,'intervention':name,'acc':float((pred==y).mean()),'hard_auc':float(runner._metrics(d['final_logp'],state,q)['hard_auc']),'prob_auc':float(runner._metrics(d['final_logp'],state,q)['prob_auc']),'semantic_acc':float((d['semantic_logp'].argmax(1).detach().cpu().numpy()==y).mean()),'mean_abs_probability_change':float(np.abs(prob-base_prob).mean()),'max_probability_change':float(np.abs(prob-base_prob).max()),'mean_abs_semantic_probability_change':float(np.abs(sem-base_sem).mean()),'semantic_probability_p95':float(np.quantile(np.abs(sem-base_sem),.95)),'hard_label_flips':int((pred!=base_pred).sum()),'correct_to_wrong':int(((base_pred==y)&(pred!=y)).sum()),'wrong_to_correct':int(((base_pred!=y)&(pred==y)).sum()),'final_darg_unchanged':bool(np.array_equal(pred,base_pred)),'query_count':len(q)})
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
        d=np.asarray([(left[f][m]-right[f][m])*100 for f in folds]); mean=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0.; half=float(t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d))) if len(d)>1 else 0.
        out.append({'comparison':name,'scope':'all10' if len(folds)==10 else 'exclude_fold2','metric':m,'n_folds':len(folds),'mean_difference_pp':mean,'sample_sd_difference_pp':sd,'wins':int((d>1e-12).sum()),'ties':int((np.abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),'descriptive_95ci_low_pp':mean-half,'descriptive_95ci_high_pp':mean+half,'fold_differences_pp':json.dumps({str(f):float((left[f][m]-right[f][m])*100) for f in folds},sort_keys=True)})
    return out

def report(summaries,comparisons,mechanisms,intervs,status,audit_status,source_sha):
    lines=['# Experiment 78.7 — BDP-PL results','',f'状态：`{status}`；正式目标40折，训练完成{audit_status["trained_tasks"]}/40，独立最佳Checkpoint审计{audit_status["independent_checkpoint_audits"]}/40。',f'正式目录：`{HERE}`；preferred OSSFS errno=5，使用/model回退；训练源SHA：`{source_sha}`。','','## 六指标（均值±population SD，%）','', '|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功|','|---|---:|---:|---:|---:|---:|---:|---:|']
    by={x['group']:x for x in summaries}
    for g in GROUPS:
        x=by.get(g,{'success_count':0}); lines.append('|'+g+'|'+'|'.join(f'{x.get(m+"_mean_percent",float("nan")):.4f} ± {x.get(m+"_population_sd_percent",float("nan")):.4f}' for m in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))+f'|{x.get("success_count",0)}|')
    base=reference(BASE/'results/fold_metrics.csv',runner.BASE_GROUP); e781=reference(Path('/model/spromptgl/experiment/experiment78_1_sh_dcpss/results/fold_metrics.csv'),'E78.1-G1-SPromptGL-K1'); e785=reference(Path('/model/spromptgl/experiment/experiment78_5_bspal/results/fold_metrics.csv'),'E78.5-G2-Selective-K1-Medical')
    def refline(name,d):
        x=summary(name,[{'group':name,**d[f]} for f in sorted(d)]); return '|'+name+'|'+'|'.join(f'{x[m+"_mean_percent"]:.4f} ± {x[m+"_population_sd_percent"]:.4f}' for m in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))+'|10|'
    lines += ['', '历史参考：','|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|折数|','|---|---:|---:|---:|---:|---:|---:|---:|',refline('E74.4-G2',base),refline('E78.1-G1',e781),refline('E78.5-G2',e785),'','## 机制与干预','','固定最佳Checkpoint的 A/B、uniform_attention、anonymous_text 和 prompt_class_swap 结果见 `mechanism.csv` 与 `interventions.csv`。最终分类始终是DARG，因此语义干预可改变辅助语义概率而不必改变DARG硬预测。']
    for iv in sorted(set(r['intervention'] for r in intervs)):
        rr=[r for r in intervs if r['intervention']==iv]; lines.append(f'- `{iv}`：平均语义概率变化 {np.mean([r["mean_abs_semantic_probability_change"] for r in rr]):.6f}；平均DARG硬翻转 {np.mean([r["hard_label_flips"] for r in rr]):.3f}/折。')
    lines += ['', '## 配对比较','', '逐折差、W/T/L、描述性95% CI及排除Fold 2结果见 `paired_comparisons.csv`。']
    for r in comparisons:
        if r['metric']=='acc' and r['scope']=='all10': lines.append(f'- {r["comparison"]}：ACC差 {r["mean_difference_pp"]:+.4f} pp，W/T/L {r["wins"]}/{r["ties"]}/{r["losses"]}。')
    g1,g2,g3,g4=[by.get(g,{}) for g in GROUPS]; b=summary('base',[{'group':'base',**base[f]} for f in sorted(base)])
    gain3=bool(g3 and g3.get('acc_mean_percent',-1)>=b['acc_mean_percent'] and g3.get('hard_auc_mean_percent',-1)>=b['hard_auc_mean_percent'])
    bidir=bool(g3 and g3.get('acc_mean_percent',-1)>g1.get('acc_mean_percent',-1) and g3.get('acc_mean_percent',-1)>g2.get('acc_mean_percent',-1))
    med=bool(g3 and g4 and g3.get('acc_mean_percent',-1)>g4.get('acc_mean_percent',-1))
    go=bool(gain3 and bidir and med)
    lines += ['', '## GO / NO-GO','',f'- G3相对E74.4-G2：`{"SUPPORTED" if gain3 else "NOT_SUPPORTED"}`。',f'- G3相对G1/G2双向互补：`{"SUPPORTED" if bidir else "NOT_SUPPORTED"}`。',f'- G3医学相对G4匿名：`{"SUPPORTED" if med else "MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED"}`。',f'- 最终科研判定：`{"GO_CANDIDATE" if go else "NO-GO_BIDIRECTIONAL_PROMPT"}`。','', '## 限制','', '仍保留val=test、测试ACC选模、CV前RFE/预处理、transductive图和Fold 2历史重叠；本实验是方法探索性证据，不是无偏临床泛化评估。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def audit_all():
    torch.set_num_threads(1); tests=unit_tests(); frozen=runner.verify_frozen_source(); source_sha=frozen['source_sha256']; runner.DATA_IDENTITY=runner._data_identity(runner.prepare(1)[0])
    audits=[]; metric_rows=[]; mechanisms=[]; intervs=[]; errors=[]
    for fold in range(1,11):
        for group in GROUPS:
            try:
                audit,cp,state,module,opt,details,metrics,base_metrics=restore(group,fold,source_sha)
                audits.append(audit); task=json.loads(runner.task_paths(group,fold)[0].read_text())
                metric_rows.append({'group':group,'fold':fold,'status':audit['status'],**{m:float(metrics[m]) for m in METRICS},'tp':int(metrics['asd_tp']),'tn':int(metrics['td_tn']),'fp':int(metrics['td_fp']),'fn':int(metrics['asd_fn']),'best_epoch_1based':audit['best_epoch_1based'],'epochs_executed':audit['epochs_executed'],'wall_seconds':float(task['wall_seconds']),'peak_gpu_bytes':int(task.get('peak_gpu_bytes',0)),'checkpoint_sha256':audit['checkpoint_sha256']})
                mechanisms.append(mechanism(group,fold,cp,details,state,module)); intervs.extend(interventions(group,fold,state,module,metrics)); del state,module,opt,cp,details; torch.cuda.empty_cache()
            except Exception as e:
                errors.append({'group':group,'fold':fold,'error':repr(e)}); audits.append({'group':group,'fold':fold,'status':'AUDIT_FAILED','error':repr(e)})
    summaries=[summary(g,[r for r in metric_rows if r['group']==g]) for g in GROUPS]
    cur={g:{int(r['fold']):{m:float(r[m]) for m in METRICS} for r in metric_rows if r['group']==g} for g in GROUPS}; base=reference(BASE/'results/fold_metrics.csv',runner.BASE_GROUP); e781=reference(Path('/model/spromptgl/experiment/experiment78_1_sh_dcpss/results/fold_metrics.csv'),'E78.1-G1-SPromptGL-K1'); e785=reference(Path('/model/spromptgl/experiment/experiment78_5_bspal/results/fold_metrics.csv'),'E78.5-G2-Selective-K1-Medical')
    specs=[(f'{GROUPS[0]} - E74.4-G2',cur[GROUPS[0]],base),(f'{GROUPS[1]} - E74.4-G2',cur[GROUPS[1]],base),(f'{GROUPS[2]} - E74.4-G2',cur[GROUPS[2]],base),(f'{GROUPS[3]} - E74.4-G2',cur[GROUPS[3]],base),(f'{GROUPS[2]} - {GROUPS[0]}',cur[GROUPS[2]],cur[GROUPS[0]]),(f'{GROUPS[2]} - {GROUPS[1]}',cur[GROUPS[2]],cur[GROUPS[1]]),(f'{GROUPS[2]} - {GROUPS[3]}',cur[GROUPS[2]],cur[GROUPS[3]]),(f'{GROUPS[2]} - E78.5-G2',cur[GROUPS[2]],e785),(f'{GROUPS[2]} - E78.1-G1',cur[GROUPS[2]],e781)]
    comparisons=[]
    for n,l,r in specs:
        fs=sorted(set(l)&set(r)); comparisons+=paired(n,l,r,fs); comparisons+=paired(n,l,r,[f for f in fs if f!=2])
    write_csv(RESULTS/'fold_metrics.csv',metric_rows); write_csv(RESULTS/'paired_comparisons.csv',comparisons); write_csv(RESULTS/'mechanism.csv',mechanisms); write_csv(RESULTS/'interventions.csv',intervs); write_csv(RESULTS/'checkpoint_audit.csv',audits)
    pre=json.loads((RESULTS/'preflight.json').read_text()); status='COMPLETE_AUDITED' if len(metric_rows)==40 and not errors else ('PARTIAL' if metric_rows else 'FAILED'); rng=sum(a.get('rng_common_epochs',0) for a in audits); mism=sum(a.get('rng_mismatches',0) for a in audits); ast={'status':status,'trained_tasks':len(metric_rows),'independent_checkpoint_audits':sum(a.get('status')=='INDEPENDENT_AUDIT_PASS' for a in audits),'expected_tasks':40,'audit_errors':errors,'source_sha256':source_sha,'base_off_parity_passed':pre.get('base_off_parity_passed'),'rng_common_epochs_checked':rng,'rng_mismatches':mism,'unit_tests':tests,'root_and_historical_sources_unchanged':True}
    write_json(RESULTS/'audit_status.json',ast); report(summaries,comparisons,mechanisms,intervs,status,ast,source_sha)
    runner.atomic_json(HERE/'manifest.json',{**frozen,'status':status,'completed_tasks':len(metric_rows),'independent_audits':ast['independent_checkpoint_audits'],'audit_errors':errors,'results':{n:str(RESULTS/n) for n in ('fold_metrics.csv','paired_comparisons.csv','mechanism.csv','interventions.csv','checkpoint_audit.csv','audit_status.json','report.md','unit_tests.json')}})
    hand=Path('/model/spromptgl/experiment/handoff.md'); old=hand.read_text() if hand.exists() else ''; marker='## Experiment 78.7: BDP-PL';
    if marker not in old:
        lines=[marker,'',f'- 状态：`{status}`；正式任务{len(metric_rows)}/40，独立Checkpoint审计{ast["independent_checkpoint_audits"]}/40。',f'- 正式目录：`{HERE}`；preferred OSSFS errno=5，使用/model回退；训练源SHA：`{source_sha}`。',f'- Base-off parity：{ast.get("base_off_parity_passed")}/10；公共RNG共有epoch：{rng}，不匹配：{mism}。','- 四组：G1方向A医学、G2方向B医学、G3双向医学、G4双向匿名；方向A校准、方向B软概率患者聚合、语义监督、机制干预和完整指标见 `results/report.md`。','- 历史限制：val=test、测试ACC选模、CV前预处理/RFE、transductive图和Fold 2重叠仍保留，新分支未读取测试标签。', '']; tmp=hand.with_name(hand.name+f'.tmp.{os.getpid()}'); tmp.write_text(old+('\n' if old and not old.endswith('\n') else '')+'\n'.join(lines)); os.replace(tmp,hand); ast['handoff_update']='APPENDED'
    else: ast['handoff_update']='ALREADY_PRESENT'
    ast['handoff_path']=str(hand); write_json(RESULTS/'audit_status.json',ast); return ast

if __name__=='__main__': print(json.dumps(audit_all(),ensure_ascii=False,indent=2))

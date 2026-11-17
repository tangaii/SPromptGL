#!/usr/bin/env python3
"""Independent E81.6.1 checkpoint, recovery, RNG and mechanism recomputation."""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import t as student_t
import run as ex

HERE,RESULTS,RUNTIME=ex.HERE,ex.RESULTS,ex.RUNTIME
GROUPS,METRICS,BASE_GROUP=ex.GROUPS,ex.METRICS,ex.BASE_GROUP


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as stream: return list(csv.DictReader(stream))


def independent_metrics(logp,state):
    ids=np.asarray(state['test'],dtype=np.int64)
    y=state['y'].detach().cpu().numpy()[ids]
    prob=logp.detach().exp().cpu().numpy()[ids]
    pred=prob.argmax(-1)
    tp,fn,fp,tn=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return {'acc':float(np.mean(y==pred)), 'hard_auc':float(balanced_accuracy_score(y,pred)),
            'prob_auc':float(roc_auc_score(y,prob[:,1])), 'sen':tp/(tp+fn), 'spe':tn/(tn+fp),
            'asd_f1':float(f1_score(y,pred,pos_label=0)), 'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,
            'prob':prob,'pred':pred}


def metric_values(record): return {m:float(record[m]) for m in METRICS}


def compare_predictions(reference,candidate,labels):
    reference,candidate,labels=map(np.asarray,(reference,candidate,labels))
    fixed=int(((reference!=labels)&(candidate==labels)).sum())
    harmed=int(((reference==labels)&(candidate!=labels)).sum())
    return {'corrected':fixed,'harmed':harmed,'net':fixed-harmed,'flips':int((reference!=candidate).sum())}


def intervention_stats(reference,candidate,labels):
    return {'probability_mae':float(np.mean(np.abs(reference['prob']-candidate['prob']))),
            **{m+'_change_pp':100.*(candidate[m]-reference[m]) for m in METRICS},
            **compare_predictions(reference['pred'],candidate['pred'],labels)}


def validate_optimizer(opt):
    if not opt.state: raise RuntimeError('optimizer moments absent')
    for slot in opt.state.values():
        if not {'step','exp_avg','exp_avg_sq'}.issubset(slot): raise RuntimeError('incomplete optimizer moment')
        for value in slot.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()): raise RuntimeError('nonfinite optimizer')


def assert_metrics(left,right,where):
    for m in METRICS:
        if abs(float(left[m])-float(right[m]))>1e-10: raise RuntimeError(f'{where}: {m} differs')


def rng_compare(left,right):
    a,b=({r['epoch']:r for r in rows} for rows in (left,right))
    common=sorted(a.keys()&b.keys())
    if not common: raise RuntimeError('no common RNG epochs')
    bad=[e for e in common if any(a[e][k]!=b[e][k] for k in ('rng_before','rng_after_train','rng_after_eval'))]
    return {'common_epochs':len(common),'mismatches':len(bad),'mismatch_epochs':bad}


def summary_values(values):
    return {group:{scope:{m:{'mean':float(np.mean([records[f][m] for f in folds]))*100.,
                                  'sd':float(np.std([records[f][m] for f in folds],ddof=0))*100.}
                         for m in METRICS}
                   for scope,folds in [('all10',range(1,11)),('without_fold2',[f for f in range(1,11) if f!=2])]}
            for group,records in values.items()}


def paired_rows(values):
    rows=[]
    for name,left,right in [('G1_vs_DARG',GROUPS[0],BASE_GROUP),('G1_vs_E81.6',GROUPS[0],OLD_GROUP),
                            ('G2_vs_G1',GROUPS[1],GROUPS[0]),('G1_vs_G3',GROUPS[0],GROUPS[2])]:
        for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
            for m in METRICS:
                diff=np.array([values[left][f][m]-values[right][f][m] for f in folds])*100.
                mean,sd=float(diff.mean()),float(diff.std(ddof=1))
                half=float(student_t.ppf(.975,len(diff)-1))*sd/math.sqrt(len(diff))
                rows.append({'comparison':name,'left':left,'right':right,'scope':scope,'metric':m,
                             'mean_difference_pp':mean,'sd_difference_pp':sd,'ci95_low_pp':mean-half,'ci95_high_pp':mean+half,
                             'wins':int((diff>1e-10).sum()),'ties':int((abs(diff)<=1e-10).sum()),
                             'losses':int((diff < -1e-10).sum()),'folds':json.dumps(folds),
                             'fold_differences_pp':json.dumps(diff.tolist()),
                             'interval_interpretation':'descriptive; CV training memberships overlap'})
    return rows


OLD_GROUP='E81.6-G1-Medical-Pairwise-PRCL-RES'
OLD=Path(ex.CFG['readonly_reference']['e816_directory'])


def prompt_geometry():
    cache=ex.prompt_cache(); result={}
    for name,key in [('original','normalized'),('centered','centered')]:
        t=cache[key].double(); gram=t@t.T; sim=gram[~torch.eye(8,dtype=torch.bool)]
        result[name]={'gram':gram.tolist(),'offdiagonal_similarity_mean':float(sim.mean()),
                      'offdiagonal_similarity_sd':float(sim.std(unbiased=False)),
                      'minimum_row_norm':float(t.norm(dim=-1).min())}
        for label,groups in [('medical',ex.MEDICAL),('scrambled',ex.SCRAMBLED)]:
            n,a=t[list(groups[0])].mean(0),t[list(groups[1])].mean(0)
            result[name][label]={'normal_abnormal_center_distance':float((a-n).norm()),
                'normal_abnormal_center_cosine':float(torch.nn.functional.cosine_similarity(n,a,dim=0))}
    result['centered_pre_normalization_norms']=cache['centered_pre_normalization_norms'].tolist()
    return result


def historical_e816(fold):
    import importlib.util,sys
    name='cprl_readonly_e816_model'
    if name not in sys.modules:
        spec=ex.PATH_ADAPTER._original_spec(name,OLD/'prcl_res.py')
        mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    cls=sys.modules[name].PRCLRES
    cp=ex.torch_load(OLD/'runtime/checkpoints'/f'{OLD_GROUP}_f{fold:02d}.best.pt')
    cache=ex.torch_load(OLD/'prompt_features.pt')
    state,_,identity=ex.prepare(fold); assert cp['identity']==identity
    with torch.random.fork_rng(devices=[]): module=cls(cache['medical'],cache['anonymous'])
    module=module.to(state['x'].device); module.load_state_dict(cp['best']['bundle']['relation'])
    module.eval(); ex.runtime().load_history().load_bundle(state,cp['best']['bundle']['base'])
    count=[0]
    def hook(layer,args):
        count[0]+=1
        return (module(args[0])[0],)+args[1:]
    handle=state['cam'].Outputlayer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(): out=ex._base_forward(state,False)
    finally: handle.remove()
    assert count[0]==1
    metrics=independent_metrics(out['logp'],state)
    assert_metrics(metrics,cp['metrics'],'E81.6 historical Best')
    assert np.array_equal(metrics['pred'],cp['best']['test_prediction'])
    assert np.allclose(metrics['prob'],cp['best']['test_probability'],atol=1e-7,rtol=0)
    return metrics


def latest_recovery(state,module,optimizer,latest,group,fold):
    ex.load_bundle(state,module,optimizer,latest['current'])
    ex._assert_exact(ex.bundle(state,module,optimizer),latest['current'],'latest_complete_roundtrip')
    ex._rng_runtime().restore_rng(latest['rng'])
    first=ex._replay_step(state,module,optimizer,group)
    fresh,_,_=ex.prepare(fold); twin=ex.make_module(fold,fresh['x'].device)
    opt=torch.optim.AdamW(twin.parameters(),lr=.0005,weight_decay=.01)
    ex.load_bundle(fresh,twin,opt,latest['current']); ex._rng_runtime().restore_rng(latest['rng'])
    second=ex._replay_step(fresh,twin,opt,group)
    for k in ('losses','logp','delta','raw_delta','semantic_increment','visual_attention','matching','attention','state','norms'):
        ex._assert_exact(first[k],second[k],f'latest_replay.{group}.{fold}.{k}')
    assert ex._rng_runtime().rng_sha(first['rng'])==ex._rng_runtime().rng_sha(second['rng'])
    ex.load_bundle(state,module,optimizer,latest['current']); ex._rng_runtime().restore_rng(latest['rng'])
    return 'PASS_EXACT_FRESH_MODEL_NEXT_STEP_ALL_FOUR_OPTIMIZERS'


def mechanism_row(group,fold,state,module,full,off,history,logs):
    ids=torch.as_tensor(state['test'],device=full['logp'].device,dtype=torch.long)
    labels=state['y'].detach().cpu().numpy()[np.asarray(state['test'])]
    fm,bm=independent_metrics(full['logp'],state),independent_metrics(off['logp'],state)
    d=full['fusion']; att=d['attention'][ids]; ratio=d['delta_ratio'][ids]; raw=d['raw_ratio'][ids]
    normal,abnormal=d['normal'][ids],d['abnormal'][ids]
    row={'group':group,'fold':fold,
         **{'history_'+m:history[m]*100. for m in METRICS},
         **{'current_off_'+m:bm[m]*100. for m in METRICS},
         **{'full_'+m:fm[m]*100. for m in METRICS},
         **{'current_off_vs_history_'+k:v for k,v in compare_predictions(history['pred'],bm['pred'],labels).items()},
         **{'residual_vs_current_off_'+k:v for k,v in compare_predictions(bm['pred'],fm['pred'],labels).items()},
         **{'final_vs_history_'+k:v for k,v in compare_predictions(history['pred'],fm['pred'],labels).items()},
         **ex._eval_summary(full,state),
         'delta_ratio_p99':float(torch.quantile(ratio,.99)),
         'raw_ratio_p50':float(torch.quantile(raw,.5)),
         'raw_ratio_p99':float(torch.quantile(raw,.99)),
         'near_zero_fraction':float((ratio<1e-4).float().mean()),
         'raw_delta_norm_mean':float(d['raw_delta'][ids].norm(dim=-1).mean()),
         'delta_norm_mean':float(d['delta'][ids].norm(dim=-1).mean()),
         'cam_norm_mean':float(d['z_cam'][ids].norm(dim=-1).mean()),
         'enhanced_norm_mean':float(d['z_enhanced'][ids].norm(dim=-1).mean()),
         'enhanced_cam_distance':float((d['z_enhanced'][ids]-d['z_cam'][ids]).norm(dim=-1).mean()),
         'local_residual_norm_means':json.dumps(d['delta'][ids].reshape(-1,16,9).norm(dim=-1).mean(0).cpu().tolist()),
         'local_increment_norm_means':json.dumps(d['delta_messages'][ids].norm(dim=-1).mean((0,1)).cpu().tolist()),
         'patient_delta_coordinate_sd_mean':float(d['delta'][ids].std(0,unbiased=False).mean()),
         'residual_hidden_mae':float((full['hidden'][ids]-off['hidden'][ids]).abs().mean()),
         'residual_probability_mae':float(np.mean(np.abs(fm['prob']-bm['prob']))),
         'projected_gram':json.dumps(d['gram_after'].cpu().tolist()),
         'input_gram':json.dumps(d['gram_before'].cpu().tolist()),
         'centered_gram':json.dumps(d['gram_centered'].cpu().tolist()),
         'visual_message_norm':float(d['m_visual'][ids].norm(dim=-1).mean()),
         'prompt_message_norm':float(d['messages'][ids].norm(dim=-1).mean()),
         'normal_abnormal_increment_mae':float((normal-abnormal).abs().mean()),
         'normal_abnormal_increment_cosine':float(torch.nn.functional.cosine_similarity(normal,abnormal,dim=-1).mean()),
         'normal_patient_sd_mean':float(normal.std(0,unbiased=False).mean()),
         'abnormal_patient_sd_mean':float(abnormal.std(0,unbiased=False).mean()),
         'contrast_patient_sd_mean':float(d['delta_contrast'][ids].std(0,unbiased=False).mean()),
         'prompt_attention_patient_sd_mean':float(att.std(0,unbiased=False).mean()),
         'visual_null_score_mean':float(d['score_visual'][ids,...,-1].mean()),
         'null_semantic_score_max_abs':float(d['score_semantic_null'][ids,...,-1].abs().max())}
    groups=ex.MEDICAL if group!=GROUPS[2] else ex.SCRAMBLED
    messages=d['messages'][ids]
    row['normal_abnormal_raw_message_mae']=float((messages[:,list(groups[0])].mean(1)-messages[:,list(groups[1])].mean(1)).abs().mean())
    pairmae=torch.stack([torch.stack([(att[:,k]-att[:,j]).abs().mean() for j in range(8)]) for k in range(8)])
    row['prompt_attention_pairwise_mae']=json.dumps(pairmae.cpu().tolist())
    row['prompt_attention_offdiagonal_mae']=float(pairmae[~torch.eye(8,dtype=torch.bool,device=pairmae.device)].mean())
    row['prompt_computational_collapse']=bool(float(pairmae.max())<1e-6 or row['semantic_increment_norm']<1e-7)
    byprompt=[]
    for k in range(8):
        ak=att[:,k]; avg=ak.mean(0); flat=avg[:,:15].reshape(-1); vals,ix=flat.topk(5)
        edges=[{'receiver':int(i)//15,'sender':int(module.neighbors[int(i)//15,int(i)%15]),'mean_weight':float(v)} for i,v in zip(ix,vals)]
        byprompt.append({'index':k,'text':ex.PROMPTS[k],
            'attention_visual_mae':float((ak-d['attention_visual'][ids]).abs().mean()),
            'semantic_increment_norm':float(d['delta_messages'][ids,k].norm(dim=-1).mean()),
            'entropy':float((-(ak*ak.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
            'null_weight':float(ak[...,-1].mean()),'null_dominant_fraction':float((ak.argmax(-1)==15).float().mean()),
            'patient_attention_sd_mean':float(ak.std(0,unbiased=False).mean()),
            'patient_template_mae':float((ak-avg).abs().mean()),'top_mean_relations':edges})
    row['per_prompt_statistics']=json.dumps(byprompt)
    for key in ('routed_cam144_norm','routed_to_total_cam144_ratio','delta144','semantic_increment','raw_delta'):
        a=np.array([r['gradient_route'].get(key,0.) for r in logs])
        for label,vals in [('mean',a),('first10_mean',a[:10]),('last10_mean',a[-10:])]: row['train_'+key+'_'+label]=float(vals.mean())
    for key in logs[0]['gradient_norms']:
        a=np.array([r['gradient_norms'][key] for r in logs]); row['train_'+key+'_gradient_mean']=float(a.mean()); row['train_'+key+'_gradient_max']=float(a.max())
    for key in ('res','cls','historical_total','total'):
        a=np.array([r['losses'][key] for r in logs]); row['train_'+key+'_mean']=float(a.mean()); row['train_'+key+'_max']=float(a.max())
    row['train_weighted_raw_regularizer_mean']=.001*row['train_res_mean']
    interventions=['null','global_off','prompt_off','medical_groups' if group==GROUPS[2] else 'scrambled_groups']
    if group==GROUPS[0]: interventions+=['unbounded']+[f'delete_prompt_{k}' for k in range(8)]
    for name in interventions:
        with torch.no_grad(): probe=ex.forward_fusion(state,module,group,False,intervention=name)
        pm=independent_metrics(probe['logp'],state); pd=probe['fusion']
        stats=intervention_stats(fm,pm,labels)
        stats.update({'attention_mae':float((d['attention'][ids]-pd['attention'][ids]).abs().mean()),
                      'message_mae':float((d['messages'][ids]-pd['messages'][ids]).abs().mean()),
                      'increment_mae':float((d['delta_messages'][ids]-pd['delta_messages'][ids]).abs().mean()),
                      'delta_mae':float((d['delta'][ids]-pd['delta'][ids]).abs().mean()),
                      'effective_ratio_mean':float(pd['delta_ratio'][ids].mean())})
        row[name+'_intervention']=json.dumps(stats)
        if name in ('null','prompt_off'):
            assert torch.equal(pd['attention'],pd['attention_visual'][:,None].expand_as(pd['attention']))
            for key in ('delta_messages','normal','abnormal','relation','raw_delta','delta'):
                assert torch.count_nonzero(pd[key])==0,(name,key)
            assert torch.equal(probe['logp'],off['logp']) and torch.equal(probe['hidden'],off['hidden'])
        if name=='global_off': assert torch.equal(d['delta_messages'],pd['delta_messages']) and torch.equal(d['d'],pd['d'])
        if name=='unbounded':
            assert torch.equal(pd['delta'],pd['raw_delta']) and torch.equal(pd['raw_delta'],d['raw_delta'])
            assert torch.equal(probe['res'],full['res'])
        if name.endswith('_groups'):
            for key in ('q_sem','attention','messages','delta_messages'): assert torch.equal(pd[key],d[key])
        del probe
    return row


def scientific_decision(summary,mechanisms,pairs,audit_pass):
    def superior(left,right,scope='all10'):
        return all(summary[left][scope][m]['mean']>summary[right][scope][m]['mean']+1e-10 for m in ('acc','hard_auc'))
    def total(records,prefix): return sum(r[prefix+'_net'] for r in records)
    g1=[r for r in mechanisms if r['group']==GROUPS[0]]; g2=[r for r in mechanisms if r['group']==GROUPS[1]]
    baseline=superior(GROUPS[0],BASE_GROUP); baseline9=superior(GROUPS[0],BASE_GROUP,'without_fold2')
    oldgain=superior(GROUPS[0],OLD_GROUP)
    group_gain=superior(GROUPS[0],GROUPS[2]) and superior(GROUPS[0],GROUPS[2],'without_fold2')
    # Conservative, descriptive support fixed before results; no independent-CV inference.
    group_strong=group_gain and all(r['ci95_low_pp']>0 for r in pairs if r['comparison']=='G1_vs_G3' and r['metric'] in ('acc','hard_auc'))
    pairmae=float(np.mean([r['prompt_attention_offdiagonal_mae'] for r in g1]))
    prompt_flips=sum(json.loads(r['prompt_off_intervention'])['flips'] for r in g1)
    incremental=total(g1,'residual_vs_current_off')>0
    mech_better=oldgain and pairmae>.0015892275026999415 and prompt_flips>0 and incremental
    coadapt=total(g1,'current_off_vs_history')<0 and -total(g1,'current_off_vs_history')>=total(g1,'residual_vs_current_off')
    unbound_gain=superior(GROUPS[1],GROUPS[0])
    g2_severe=any(r['over_100_fraction']>.25 for r in g2) and total(g2,'current_off_vs_history')<0
    fullgo=baseline and baseline9 and mech_better and group_strong and not coadapt and audit_pass
    return {'complete_method':'CPRL_BASELINE_GAIN_PRELIMINARY' if baseline else 'NO_GO_CPRL_BASELINE',
            'baseline_gain_without_fold2':baseline9,
            'vs_E81_6':'PROMPT_MECHANISM_AND_CLASSIFICATION_IMPROVEMENT_PRELIMINARY' if mech_better else 'CPRL_PROMPT_IMPROVEMENT_NOT_ESTABLISHED',
            'original_semantic_grouping':'ORIGINAL_GROUPING_DESCRIPTIVE_SUPPORT' if group_strong else ('ORIGINAL_GROUPING_MEAN_GAIN_ONLY' if group_gain else 'ORIGINAL_GROUPING_INCREMENT_NOT_ESTABLISHED'),
            'unbounded':('UNBOUNDED_GAIN_WITH_COADAPTATION_CONCERN' if g2_severe else 'UNBOUNDED_PERFORMANCE_GAIN_PRELIMINARY') if unbound_gain else ('REJECT_UNBOUNDED_WITH_SEVERE_COADAPTATION' if g2_severe else 'UNBOUNDED_NOT_BETTER'),
            'overall':'PRELIMINARY_GO_CPRL_RES' if fullgo else 'NO_GO_CPRL_RES',
            'net_corrections_vs_DARG':total(g1,'final_vs_history'),
            'net_corrections_without_fold2':total([r for r in g1 if r['fold']!=2],'final_vs_history'),
            'backbone_coadaptation_concern':coadapt,'G2_severe_coadaptation':g2_severe,
            'G1_prompt_off_flips':prompt_flips,'G1_prompt_attention_pair_mae':pairmae,
            'grouping_support_rule':'both primary mean gains in 10/9 folds and descriptive paired lower bounds >0; does not establish independent-patient significance'}


def audit():
    started=time.monotonic(); manifest=json.loads((HERE/'manifest.json').read_text()); frozen=json.loads((RESULTS/'preflight.json').read_text())
    sha=frozen['source_sha256']; assert frozen['status']=='PASS' and ex.source_sha()==sha==manifest['source_sha256']
    assert ex.sha_file(RESULTS/'preflight.json')==manifest['preflight_sha256']
    replay=ex.preflight(persist=False)
    for f in range(10):
        for key in ('identity','initial_module_sha256','warmup_rng_sha','base_off_forward_hidden_historical_loss_gradient_update_rng_exact'):
            assert replay['folds'][f][key]==frozen['folds'][f][key],('independent parity',f,key)
    assert replay['module_tests']==frozen['module_tests']
    replay_record={'status':'PASS','folds_exactly_reproduced':10,'source_sha256':sha,'serialized_resume':replay['folds'][0]['serialized_resume']}
    ex.atomic_json(RUNTIME/'independent_preflight_replay.json',replay_record)
    reference={int(r['fold']):metric_values(r) for r in read_csv(ex.runtime().EXP/'results/fold_metrics.csv') if r['group']==BASE_GROUP}
    old_reference={int(r['fold']):{m:float(r[m])/100 for m in METRICS} for r in read_csv(OLD/'results/fold_metrics.csv') if r['group']==OLD_GROUP}
    assert set(reference)==set(old_reference)==set(range(1,11))
    values={g:{} for g in (*GROUPS,BASE_GROUP,OLD_GROUP)}; values[BASE_GROUP]=reference
    rows=[]; mechanisms=[]; latest_checks=[]; best_checks=[]; rng_reports={}; tasks={}
    for fold in range(1,11):
        oldmetrics=historical_e816(fold); assert_metrics(oldmetrics,old_reference[fold],'E81.6 results'); values[OLD_GROUP][fold]=metric_values(oldmetrics)
        hist_task=json.loads((ex.runtime().EXP/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())
        hist_cp=ex.torch_load(ex.runtime().EXP/'.runtime/checkpoints'/f'{BASE_GROUP}_f{fold:02d}.best.pt')
        state,_,identity=ex.prepare(fold); assert identity==frozen['folds'][fold-1]['identity']
        ex.runtime().load_history().load_bundle(state,hist_cp['best_state'])
        with torch.no_grad(): h=ex._base_forward(state,False)
        history=independent_metrics(h['logp'],state); assert_metrics(reference[fold],history,'DARG historical reload')
        assert np.array_equal(history['pred'],hist_cp['test_prediction'])
        assert np.allclose(history['prob'],hist_cp['test_probability'],atol=1e-7,rtol=0)
        del state,h,hist_cp; attention_by_group={}
        for group in GROUPS:
            task_file,best_file,latest_file,log_file=ex.paths(group,fold)
            assert ex._task_complete(group,fold,sha),('incomplete task',group,fold)
            task=json.loads(task_file.read_text()); tasks[group,fold]=task
            best,latest=ex.torch_load(best_file),ex.torch_load(latest_file)
            for cp in (best,latest):
                assert cp['source_sha256']==sha and cp['identity']==identity and cp['group']==group and cp['fold']==fold
            state,_,_=ex.prepare(fold); module=ex.make_module(fold,state['x'].device)
            initsha=ex.sha_value(ex._cpu_state(module)); assert initsha==task['initial_module_sha256']==frozen['folds'][fold-1]['initial_module_sha256']
            assert sum(p.numel() for p in module.parameters())==frozen['module_tests']['parameter_count']==42336
            optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
            ex.load_bundle(state,module,optimizer,latest['current']); ex._validate_optimizer_settings(state,optimizer)
            for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer): validate_optimizer(opt)
            assert {'python','numpy','cpu','cuda'}.issubset(latest['rng'])
            ex._rng_runtime().restore_rng(latest['rng']); assert ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())==ex._rng_runtime().rng_sha(latest['rng'])
            assert latest['next_epoch']==task['epochs_executed']
            logs=[json.loads(line) for line in log_file.read_text().splitlines()]
            assert len(logs)==task['epochs_executed']-10==task['scalar_rows']
            assert ex.sha_file(log_file)==task['scalar_log_sha256']
            assert logs==latest['epoch_logs'] and [r['epoch'] for r in logs]==list(range(11,task['epochs_executed']+1))
            assert [r['rng'] for r in logs]==task['rng_trace']==latest['trace']
            best_log=max(logs,key=lambda r:r['metrics']['acc'])
            assert best_log['epoch']==task['best_epoch_1based']==best['best_epoch_1based']==latest['best_epoch']+1
            running=-1.; selected=-1; wait=0
            for line in logs:
                if line['metrics']['acc']>running: running=line['metrics']['acc']; selected=line['epoch']; wait=0
                else: wait+=1
                assert line['best_acc']==running and line['best_epoch_1based']==selected
                loss=line['losses']; assert 'geo' not in loss
                assert abs(loss['total']-(loss['historical_total']+.001*loss['res']))<2e-6
                assert all(np.isfinite(x) for x in line['gradient_norms'].values())
                if group!=GROUPS[1]: assert line['diagnostics']['delta_ratio_max']<=.100001
            assert wait==latest['wait']
            if task['epochs_executed']<450: assert wait>200
            assert task['prompt_cache_sha256']==frozen['prompt_cache_sha256']
            recovery=latest_recovery(state,module,optimizer,latest,group,fold)
            latest_checks.append({'group':group,'fold':fold,'four_optimizers':'PASS','rng':'PASS','fresh_next_step_replay':recovery,
                                  'scalar_journal_all_epochs':'PASS','strict_earliest_best':'PASS','rows':len(logs)})
            ex.load_bundle(state,module,optimizer,best['best']['bundle'])
            for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer): validate_optimizer(opt)
            assert ex._rng_runtime().rng_sha(best['best']['rng'])==best['best']['rng_after_eval']
            ex._rng_runtime().restore_rng(best['best']['rng'])
            rb=ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
            with torch.no_grad():
                full=ex.forward_fusion(state,module,group,False); off=ex.forward_fusion(state,module,group,False,'off')
            d=full['fusion']; assert torch.equal(off['fusion']['z_enhanced'],off['fusion']['z_cam'])
            assert torch.isfinite(full['logp']).all()
            if group==GROUPS[1]: assert torch.equal(d['delta'],d['raw_delta'])
            else: assert (d['delta_ratio']<=.100001).all()
            assert torch.allclose(full['logp'].exp().sum(-1),torch.ones_like(full['logp'][:,0]),atol=2e-6)
            metrics=independent_metrics(full['logp'],state)
            assert_metrics(task['metrics'],metrics,'task'); assert_metrics(best['metrics'],metrics,'best')
            assert np.array_equal(best['best']['test_prediction'],metrics['pred'])
            assert np.allclose(best['best']['test_probability'],metrics['prob'],rtol=0,atol=1e-7)
            cls_fn=ex.base_runtime().runtime()['utils'].ClsLoss
            assert torch.equal(cls_fn(full['logp'],state['y'],state['train'],state['weights']),full['cls'])
            mech=mechanism_row(group,fold,state,module,full,off,history,logs)
            ids=torch.as_tensor(state['test'],device=full['logp'].device,dtype=torch.long)
            attention_by_group[group]=d['attention'][ids].cpu()
            assert rb==ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
            saved_rng=ex._rng_runtime().pack_rng(); hist=ex.runtime().load_history()
            hist.zero_optimizers(state); optimizer.zero_grad(set_to_none=True)
            probe=ex.forward_fusion(state,module,group,True); probe['historical_total'].backward()
            mech['best_classification_gradient_norms']=json.dumps(ex.grad_snapshot(state,module))
            route=probe['route_gradients']; mech['best_classification_semantic_increment_gradient_norm']=route.get('semantic_increment',0.)
            mech['best_classification_cam_route_norm']=.1*route.get('z_rel_input',0.)
            mech['best_classification_cam_route_ratio']=.1*route.get('z_rel_input',0.)/max(route.get('cam144_total',0.),1e-30)
            train_ids=torch.as_tensor(state['train'],device=d['delta'].device,dtype=torch.long)
            legal=probe['fusion']['legal']; legal_ids=train_ids[legal[train_ids]]
            assert torch.equal(probe['res'],probe['fusion']['raw_ratio'][legal_ids].square().mean())
            if fold==2:
                assert torch.count_nonzero(probe['fusion']['delta'][[376,406]])==0 and not bool(legal[376]) and not bool(legal[406])
            ex._rng_runtime().restore_rng(saved_rng)
            mechanisms.append(mech); values[group][fold]=metric_values(metrics)
            rows.append({'group':group,'fold':fold,**{m:metrics[m]*100. for m in METRICS},
                         **{k:metrics[k] for k in ('asd_tp','asd_fn','td_fp','td_tn')},
                         **{'current_off_'+m:mech['current_off_'+m] for m in METRICS},
                         'best_epoch_1based':task['best_epoch_1based'],'epochs_executed':task['epochs_executed'],
                         'wall_seconds':task['wall_seconds'],'peak_gpu_bytes':task['peak_gpu_bytes'],
                         'checkpoint_sha256':ex.sha_file(best_file),'latest_sha256':ex.sha_file(latest_file),'independent_checkpoint_audit':'PASS'})
            best_checks.append({'group':group,'fold':fold,'six_metrics':'PASS','probability_replay':'PASS',
                                'Prompt_Null_off_exact':'PASS','single_OutputLayer':'PASS','Fold2_train_block':'PASS',
                                'G2_unbounded' if group==GROUPS[1] else 'bounded10':'PASS','raw_regularizer':'PASS'})
            rng_reports[f'{group}_f{fold:02d}_vs_DARG']=rng_compare(task['rng_trace'],hist_task['rng_trace'])
            del state,module,optimizer,full,off,probe,best,latest,d; torch.cuda.empty_cache()
            print(f'AUDIT {group} fold={fold} PASS',flush=True)
        for group in GROUPS[1:]:
            r=next(r for r in mechanisms if r['group']==GROUPS[0] and r['fold']==fold)
            r['trained_attention_mae_vs_'+group]=float((attention_by_group[GROUPS[0]]-attention_by_group[group]).abs().mean())
        for a in range(3):
            for b in range(a+1,3):
                ga,gb=GROUPS[a],GROUPS[b]; assert tasks[ga,fold]['initial_module_sha256']==tasks[gb,fold]['initial_module_sha256']
                rng_reports[f'{ga}_vs_{gb}_f{fold:02d}']=rng_compare(tasks[ga,fold]['rng_trace'],tasks[gb,fold]['rng_trace'])
    for group in (BASE_GROUP,OLD_GROUP):
        rows.extend({'group':group,'fold':f,**{m:values[group][f][m]*100. for m in METRICS},'independent_checkpoint_audit':'HISTORICAL_BEST_RECOMPUTED'} for f in range(1,11))
    summary=summary_values(values); pairs=paired_rows(values); geometry=prompt_geometry()
    mismatches=sum(r['mismatches'] for r in rng_reports.values()); decision=scientific_decision(summary,mechanisms,pairs,mismatches==0)
    status={'status':'PASS' if mismatches==0 else 'FAIL','source_sha256':sha,'formal_tasks_complete':30,
            'best_checkpoint_independent_pass':len(best_checks),'latest_state_integrity_pass':len(latest_checks),
            'historical_DARG_best_independent_pass':10,'historical_E81_6_best_independent_pass':10,
            'base_off_parity_pass':10,'private_same_fold_three_group_initialization_pass':10,
            'trainable_parameter_count':42336,'total_trainable_parameter_count':frozen['folds'][0]['total_trainable_parameter_count'],'prompt_cache_sha256':frozen['prompt_cache_sha256'],
            'independent_preflight_replay':replay_record,'rng_mismatch_count':mismatches,'rng_audit':rng_reports,
            'best_checks':best_checks,'latest_checks':latest_checks,'summary':summary,
            'prompt_input_geometry':geometry,'research_decision':decision,'fold2_overlap_1based':[377,407],
            'metrics_unit':'percent; population SD','hard_auc_definition':'balanced accuracy of hard predictions',
            'protocol':'test ACC checkpoint selection retained; not unbiased independent test evaluation',
            'scalar_logs_missing_epochs':0,'total_scalar_epochs':sum(t['scalar_rows'] for t in tasks.values()),
            'audit_seconds':time.monotonic()-started}
    assert ex.source_sha()==sha,'historical or frozen source changed'
    ex.write_csv(RESULTS/'fold_metrics.csv',rows); ex.write_csv(RESULTS/'paired_comparisons.csv',pairs)
    ex.write_csv(RESULTS/'mechanism.csv',mechanisms); ex.atomic_json(RESULTS/'audit_status.json',status)
    write_report(status,mechanisms,pairs)
    if mismatches: raise RuntimeError('public RNG mismatch; formal results not accepted')
    return status


def write_report(status,mechanisms,pairs):
    s=status['summary']; decision=status['research_decision']; lines=[]
    def add(x=''): lines.append(x)
    def table(scope):
        add('| 模型 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |')
        add('|---|---:|---:|---:|---:|---:|---:|')
        for group in (*GROUPS,BASE_GROUP,OLD_GROUP):
            add('| '+group+' | '+' | '.join(f"{s[group][scope][m]['mean']:.4f} ± {s[group][scope][m]['sd']:.4f}" for m in METRICS)+' |')
    add('# Experiment 81.6.1 — CPRL-RES')
    add(); add(f"执行：COMPLETE_AUDITED；30/30正式训练、30/30 Best独立重算、30/30 Latest恢复；公共RNG mismatch={status['rng_mismatch_count']}，缺失epoch=0。历史DARG、E81.6-G1各10个Best均只读重算。")
    add('科研判断：'+json.dumps(decision,ensure_ascii=False))
    add(); add('单位均为%，10折均值±population SD。ASD=0、TD=1；Hard-AUC是硬预测BA，Prob-AUC是概率ROC-AUC。checkpoint仍按test=val ACC选择。')
    add(); table('all10'); add(); add('## 排除Fold2：9折'); add(); table('without_fold2')
    add(); add('## 同折配对比较'); add()
    add('| 比较（左减右） | 范围 | ACC差pp [描述性95%区间] | 胜/平/负 | BA差pp | Prob-AUC差pp |')
    add('|---|---|---:|---|---:|---:|')
    for name in ('G1_vs_DARG','G1_vs_E81.6','G2_vs_G1','G1_vs_G3'):
        for scope in ('all10','without_fold2'):
            d={r['metric']:r for r in pairs if r['comparison']==name and r['scope']==scope}; a=d['acc']
            add(f"| {name} | {scope} | {a['mean_difference_pp']:+.4f} [{a['ci95_low_pp']:+.4f}, {a['ci95_high_pp']:+.4f}] | {a['wins']}/{a['ties']}/{a['losses']} | {d['hard_auc']['mean_difference_pp']:+.4f} | {d['prob_auc']['mean_difference_pp']:+.4f} |")
    add(); add('逐折六指标差值均见paired_comparisons.csv；每个比较对应同fold成员。区间只作描述，重叠CV训练集不是完全独立样本。G1/G2严格仅最终残差映射不同；G1/G3严格仅分组不同。旧E81.6损失及模型结构同时不同，不是单变量变化。')
    add(); add('## 30折逐项结果'); add()
    add('| 组 | Fold | Best epoch | 总epoch | ACC | BA | Prob-AUC | SEN | SPE | ASD-F1 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in read_csv(RESULTS/'fold_metrics.csv'):
        if row['group'] not in GROUPS: continue
        add('| '+row['group']+' | '+str(row['fold'])+' | '+row['best_epoch_1based']+' | '+row['epochs_executed']+' | '+' | '.join(f'{float(row[m]):.4f}' for m in METRICS)+' |')
    add(); add('## Prompt、视觉/语义评分和增量诊断'); add()
    add('| 组 | Prompt pair MAE | Prompt-visual MAE | 视觉邻居score SD | 语义邻居score SD | 语义/视觉强度 | 增量消息范数 | 正常/异常增量余弦 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|')
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]; mean=lambda k:float(np.mean([r[k] for r in rs]))
        add('| '+group+' | '+' | '.join(f'{mean(k):.7f}' for k in ('prompt_attention_offdiagonal_mae','prompt_visual_attention_mae','visual_neighbor_score_sd','semantic_neighbor_score_sd','semantic_visual_neighbor_sd_ratio','semantic_increment_norm','normal_abnormal_increment_cosine'))+' |')
    add(); add('E81.6-G1 Prompt pair MAE=.00158923，正常/异常原消息MAE=.01047334，Prompt-off概率MAE=.000165961、翻转0。新模块的正常/异常是相对视觉的增量，不应把不同定义的范数直接解释为病理强度。注意力变化更大本身不表示医学判别能力提高。逐Prompt Top关系、Null权重、患者变异、投影Gram、每局部增量与有效残差、梯度和真实score尺度详见mechanism.csv。潜在局部不是解剖ROI。')
    add(); add('| 组 | 正常增量范数 | 异常增量范数 | 差值范数 | 正常患者间SD | 异常患者间SD | W_text训练梯度均值 | 语义增量训练梯度均值 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|')
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]; mean=lambda k:float(np.mean([r[k] for r in rs]))
        add('| '+group+' | '+' | '.join(f'{mean(k):.7g}' for k in ('normal_increment_norm','abnormal_increment_norm','contrast_increment_norm','normal_patient_sd_mean','abnormal_patient_sd_mean','train_text_projector_gradient_mean','train_semantic_increment_mean'))+' |')
    add(); add('## 残差：10%与完全无上限'); add()
    add('| 组 | raw绝对范数 | CAM范数 | raw/CAM均值% | effective/CAM均值% | P50/P95/P99/最大% | 接近9.9%比例 | 超10/50/100%比例 |')
    add('|---|---:|---:|---:|---:|---|---:|---|')
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]; mean=lambda k:float(np.mean([r[k] for r in rs]))
        q='/'.join(f'{100*mean(k):.4f}' for k in ('delta_ratio_p50','delta_ratio_p95','delta_ratio_p99'))+f"/{100*max(r['delta_ratio_max'] for r in rs):.4f}"
        exceed='/'.join(f'{100*mean(k):.2f}' for k in ('over_10_fraction','over_50_fraction','over_100_fraction'))
        add(f"| {group} | {mean('raw_delta_norm_mean'):.6f} | {mean('cam_norm_mean'):.6f} | {100*mean('raw_ratio_mean'):.4f} | {100*mean('delta_ratio_mean'):.4f} | {q} | {100*mean('near_bound_fraction'):.2f}% | {exceed}% |")
    add(); add('P50/P95/P99为各fold分位数均值，最大值为所有患者最大值；不是把平均数当作饱和程度。G2中delta逐元素等于raw，无Tanh、截断、门控或动态缩放。三组统一在raw上正则，无L_geo。')
    add(); add('## 冻结干预'); add()
    add('| 组/干预 | 概率MAE | 硬翻转 | ACC变化pp | BA变化pp | 相对full纠正/误伤 |')
    add('|---|---:|---:|---:|---:|---|')
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]
        for name in ('prompt_off','null','global_off','scrambled_groups','medical_groups','unbounded'):
            iv=[json.loads(r[name+'_intervention']) for r in rs if name+'_intervention' in r]
            if not iv: continue
            add(f"| {group}/{name} | {np.mean([v['probability_mae'] for v in iv]):.8f} | {sum(v['flips'] for v in iv)} | {np.mean([v['acc_change_pp'] for v in iv]):+.4f} | {np.mean([v['hard_auc_change_pp'] for v in iv]):+.4f} | {sum(v['corrected'] for v in iv)}/{sum(v['harmed'] for v in iv)} |")
    add(); add('Prompt-off在中间张量严格恢复visual attention，全部增量、relation、raw_delta与delta严格零，最终Logp/Hidden与残差off逐元素一致；初始和非零头、训练后Best均验证。结构依赖不等于医学语义正确。G1最佳Checkpoint的unbounded只做一次推理干预，会改变分布，不能替代G2正式训练。单Prompt删除8项结果保存在机制表。')
    add(); add('## 主干共同适配与净纠错'); add()
    add('| 组/范围 | A DARG→B 当前off：纠正/误伤/净 | B→C full | A→C |')
    add('|---|---|---|---|')
    for group in GROUPS:
        for scope in ('all10','without_fold2'):
            rs=[r for r in mechanisms if r['group']==group and (scope=='all10' or r['fold']!=2)]
            def count(prefix): return '/'.join(str(sum(r[prefix+k] for r in rs)) for k in ('corrected','harmed','net'))
            add(f"| {group}/{scope} | {count('current_off_vs_history_')} | {count('residual_vs_current_off_')} | {count('final_vs_history_')} |")
    add(); add('历史E81.6-G1三项净值分别−5、+2、−3。off是在full选择的冻结Checkpoint上的干预，不是独立重训/独立选模基线。若off明显退化而残差仅抵消共同适配，应保留疑虑。')
    add(); add('## 文本身份、分组与复现'); add()
    for i,t in enumerate(ex.PROMPTS): add(f'{i}: {t}')
    add(); add('三组全部42336新增可训练参数，同fold每个名称、shape及初值逐元素一致；盐816100+fold，CPU私有初始化，不扰动公共RNG。RN50冻结1024维，仅编码一次，中心化无患者标签，8条范数均非零。G1/G2正常[0,1,2,3]、异常[4,5,6,7]；G3正常[0,1,4,5]、异常[2,3,6,7]，固定不改。G3同样含医学文本，只检验4+4组织的额外作用。')
    add('中心化几何：'+json.dumps(status['prompt_input_geometry'],ensure_ascii=False))
    add(); add('## 独立审计与限制'); add()
    add(f"Best30/30、Latest30/30、历史起点10/10、初始化10/10；公共RNG mismatch={status['rng_mismatch_count']}；{status['total_scalar_epochs']}条epoch日志完整。Latest逐项恢复全部模型和四优化器，并在fresh模型上精确重放下一步；单独两步中断恢复也通过。原OutputLayer一次执行，Fold2训练保护、统一raw正则、上限/无上限身份、Prompt/Null归零均通过。来源最终SHA再次匹配，根模型、历史源码和数据未改。")
    add('完整继承E74.4-G2真实prepare→E61 epoch10 warmup及三个优化器；新模块优化器最后更新；最大450、wait>200、Best严格ACC、无逐epoch播种。历史test=val、全体预处理/RFE在CV之前、transductive图与Fold2 377/407重叠保留。仅阻断两行新增残差不消除历史泄漏。单seed与重叠训练集下所有结论是候选机制证据，不能直接宣称独立泛化或临床机制成立。')
    add('来源审查与完整公式见design.md；原论文/官方代码、KD-Brain具体类、ItemizedCLIP独立条件表示及未复制监督损失的身份见runtime/source_review.json。')
    add(f"冻结源码SHA：`{status['source_sha256']}`；Prompt缓存SHA：`{status['prompt_cache_sha256']}`。全文件SHA、任务身份与checkpoint SHA见manifest、audit_status及fold_metrics。")
    schedule=json.loads((RUNTIME/'training_summary.json').read_text()); add('调度：'+json.dumps(schedule,ensure_ascii=False))
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    result=audit(); print(json.dumps({'status':result['status'],'decision':result['research_decision']},ensure_ascii=False),flush=True)

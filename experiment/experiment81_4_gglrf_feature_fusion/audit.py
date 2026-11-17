#!/usr/bin/env python3
"""Independent best/latest, metric, RNG and mechanism audit for E81.4."""
from __future__ import annotations
import run as ex
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import t as student_t

HERE,RESULTS,RUNTIME=ex.HERE,ex.RESULTS,ex.RUNTIME
GROUPS,METRICS,BASE_GROUP=ex.GROUPS,ex.METRICS,ex.BASE_GROUP
E811='E81.1-G1-Visual-PRCL-v2'
E813='E81.3-G2-PairwiseRelation-E2E'


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as stream: return list(csv.DictReader(stream))


def independent_metrics(logp, state):
    ids=np.asarray(state['test'],dtype=np.int64)
    labels=state['y'].detach().cpu().numpy()[ids]
    probability=logp.detach().exp().cpu().numpy()[ids]
    prediction=probability.argmax(-1)
    tp,fn,fp,tn=map(int,confusion_matrix(labels,prediction,labels=[0,1]).ravel())
    sen,spe=tp/(tp+fn),tn/(tn+fp)
    return {'acc':float((labels==prediction).mean()),'hard_auc':float(balanced_accuracy_score(labels,prediction)),
            'prob_auc':float(roc_auc_score(labels,probability[:,1])), 'sen':sen,'spe':spe,
            'asd_f1':float(f1_score(labels,prediction,pos_label=0)),
            'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,'prob':probability,'pred':prediction}


def metric_values(record): return {m:float(record[m]) for m in METRICS}


def compare_predictions(reference,candidate,labels):
    reference,candidate,labels=map(np.asarray,(reference,candidate,labels))
    fixed=int(((reference!=labels)&(candidate==labels)).sum())
    harmed=int(((reference==labels)&(candidate!=labels)).sum())
    return {'corrected':fixed,'harmed':harmed,'net':fixed-harmed,'flips':int((reference!=candidate).sum())}


def validate_optimizer(optimizer):
    if not optimizer.state: raise RuntimeError('empty optimizer moments')
    for slot in optimizer.state.values():
        if not {'step','exp_avg','exp_avg_sq'}.issubset(slot): raise RuntimeError('missing optimizer moments')
        for value in slot.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()): raise RuntimeError('nonfinite optimizer state')


def paired_rows(values):
    pairs=[('G1_vs_DARG',GROUPS[0],BASE_GROUP),('G2_vs_DARG',GROUPS[1],BASE_GROUP),
           ('G2_vs_G1',GROUPS[1],GROUPS[0]),('G2_vs_E81.1_G1_context',GROUPS[1],E811),
           ('G2_vs_E81.3_G2_context',GROUPS[1],E813)]
    rows=[]
    for name,left,right in pairs:
        for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
            for metric in METRICS:
                diff=np.array([values[left][f][metric]-values[right][f][metric] for f in folds])*100.
                mean,sd=float(diff.mean()),float(diff.std(ddof=1))
                half=float(student_t.ppf(.975,len(diff)-1))*sd/math.sqrt(len(diff))
                rows.append({'comparison':name,'left':left,'right':right,'scope':scope,'metric':metric,
                             'mean_difference_pp':mean,'sd_difference_pp':sd,'ci95_low_pp':mean-half,'ci95_high_pp':mean+half,
                             'wins':int((diff>1e-10).sum()),'ties':int((abs(diff)<=1e-10).sum()),
                             'losses':int((diff < -1e-10).sum()),'fold_differences_pp':json.dumps(diff.tolist()),
                             'interval_interpretation':'descriptive only; CV training folds overlap'})
    return rows


def summary_values(values):
    return {name:{scope:{m:{'mean':float(np.mean([foldvals[f][m] for f in folds]))*100.,
                                 'sd':float(np.std([foldvals[f][m] for f in folds],ddof=0))*100.} for m in METRICS}
                  for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]}
            for name,foldvals in values.items()}


def _assert_metrics(expected,actual,context):
    for m in METRICS:
        if abs(float(expected[m])-float(actual[m]))>1e-10:
            raise RuntimeError(f'{context} metric mismatch: {m} {expected[m]} vs {actual[m]}')


def intervention_stats(reference,candidate,labels):
    return {'probability_mae':float(np.mean(abs(reference['prob']-candidate['prob']))),
            **{m+'_change_pp':(candidate[m]-reference[m])*100. for m in METRICS},
            **compare_predictions(reference['pred'],candidate['pred'],labels)}


def mechanism_row(group,fold,state,module,full,off,null,globaloff,unary,history,logs):
    ids=torch.as_tensor(state['test'],dtype=torch.long,device=full['logp'].device)
    labels=state['y'].detach().cpu().numpy()[np.asarray(state['test'])]
    fm,bm,gm=map(lambda out:independent_metrics(out['logp'],state),(full,off,globaloff))
    d=full['fusion']; beta=d['beta'][ids]; attn=d['attention'][ids]; ratio=d['delta_ratio'][ids]
    localdelta=d['delta'][ids].reshape(-1,16,9).norm(dim=-1)
    meanbeta=beta.mean(0)
    final_change=compare_predictions(history['pred'],fm['pred'],labels)
    trajectory=compare_predictions(history['pred'],bm['pred'],labels)
    residual=compare_predictions(bm['pred'],fm['pred'],labels)
    global_contribution=compare_predictions(gm['pred'],fm['pred'],labels)
    row={'group':group,'fold':fold,
         **{'final_vs_history_'+k:v for k,v in final_change.items()},
         **{'current_off_vs_history_'+k:v for k,v in trajectory.items()},
         **{'residual_vs_current_off_'+k:v for k,v in residual.items()},
         **{'global_guidance_vs_off_'+k:v for k,v in global_contribution.items()},
         **{'current_off_'+m:bm[m]*100. for m in METRICS},
         **{'full_'+m:fm[m]*100. for m in METRICS},
         **{'history_'+m:history[m]*100. for m in METRICS},
         'residual_probability_mae':float(np.mean(abs(fm['prob']-bm['prob']))),
         'residual_hidden_mae':float((full['hidden'][ids]-off['hidden'][ids]).abs().mean()),
         'delta_norm_mean':float(d['delta'][ids].norm(dim=-1).mean()),
         'raw_delta_norm_mean':float(d['raw_delta'][ids].norm(dim=-1).mean()),
         'cam_norm_mean':float(d['z_cam'][ids].norm(dim=-1).mean()),
         'delta_ratio_mean':float(ratio.mean()),'delta_ratio_p50':float(torch.quantile(ratio,.5)),
         'delta_ratio_p95':float(torch.quantile(ratio,.95)),
         'near_bound_fraction':float((ratio>=.099).float().mean()),
         'near_zero_residual_fraction':float((ratio<1e-4).float().mean()),
         'local_delta_norm_means':json.dumps(localdelta.mean(0).cpu().tolist()),
         'patient_delta_coordinate_sd_mean':float(d['delta'][ids].std(0,unbiased=False).mean()),
         'beta_entropy':float((-(beta*beta.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
         'beta_max_mean':float(beta.max(-1).values.mean()),'beta_max_p95':float(torch.quantile(beta.max(-1).values,.95)),
         'beta_patient_sd_mean':float(beta.std(0,unbiased=False).mean()),
         'beta_mean_by_local':json.dumps(meanbeta.cpu().tolist()),
         'beta_top_mean_minus_uniform':float(meanbeta.max()-1/16),
         'beta_near_uniform_fraction':float(((beta-1/16).abs().max(-1).values<.002).float().mean()),
         'beta_single_local_collapse_fraction':float((beta.max(-1).values>.8).float().mean()),
         'beta_most_frequent_top_local_fraction':float(torch.bincount(beta.argmax(-1),minlength=16).max()/len(ids)),
         'null_weight':float(attn[...,-1].mean()),
         'attention_entropy':float((-(attn*attn.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
         'message_norm_mean':float(d['m_local'][ids].norm(dim=-1).mean()),
         'head_message_cancellation_ratio':float((d['m_local'][ids].norm(dim=-1)/
              d['messages'][ids].norm(dim=-1).mean(1).clamp_min(1e-12)).mean()),
         'local_direction_cancellation_ratio':float((d['m_local'][ids].mean(1).norm(dim=-1)/
              d['m_local'][ids].norm(dim=-1).mean(1).clamp_min(1e-12)).mean()),
         'global_query_norm':float(d['query'][ids].norm(dim=-1).mean()),
         'global_read_query_norm':float(d['q_read'][ids].norm(dim=-1).mean()),
         'null_off_logp_exact':bool(torch.equal(null['logp'],off['logp'])),
         'null_off_hidden_exact':bool(torch.equal(null['hidden'],off['hidden'])),
         'global_off_beta_mae':float((d['beta'][ids]-globaloff['fusion']['beta'][ids]).abs().mean()),
         'global_off_attention_mae':float((d['attention'][ids]-globaloff['fusion']['attention'][ids]).abs().mean()),
         'global_off_delta_mae':float((d['delta'][ids]-globaloff['fusion']['delta'][ids]).abs().mean())}
    for name,out in [('residual_off',off),('null_only',null),('global_off',globaloff)]+([('pairwise_to_unary',unary)] if unary is not None else []):
        im=independent_metrics(out['logp'],state)
        row.update({name+'_'+k:v for k,v in intervention_stats(fm,im,labels).items()})
        row[name+'_hidden_mae']=float((full['hidden'][ids]-out['hidden'][ids]).abs().mean())
        if name=='pairwise_to_unary':
            row[name+'_message_mae']=float((d['m_local'][ids]-out['fusion']['m_local'][ids]).abs().mean())
    correct=bm['pred']==labels
    margin=(off['logp'][ids,1]-off['logp'][ids,0]).abs().cpu().numpy()
    change=abs(fm['prob'][:,1]-bm['prob'][:,1])
    # Fixed descriptive partition of base confidence; never used for selection.
    for name,mask in [('base_correct',correct),('base_error',~correct),('base_low_margin',margin<1.),
                      ('base_high_margin',margin>=1.),('base_high_margin_error',(margin>=1.)&(~correct))]:
        row[name+'_n']=int(mask.sum())
        row[name+'_probability_change_mae']=float(change[mask].mean()) if mask.any() else 0.
        row[name+'_corrected']=int(((bm['pred']!=labels)&(fm['pred']==labels)&mask).sum())
        row[name+'_harmed']=int(((bm['pred']==labels)&(fm['pred']!=labels)&mask).sum())
    for key in ('routed_cam144_norm','routed_to_total_cam144_ratio','delta144'):
        vals=np.array([r['gradient_route'].get(key,0.) for r in logs])
        row['train_'+key+'_mean']=float(vals.mean())
        row['train_'+key+'_first10_mean']=float(vals[:10].mean())
        row['train_'+key+'_last10_mean']=float(vals[-10:].mean())
    for key in ('global_encoder','global_readout','darg','cam','gglrf','output_projection'):
        row['train_'+key+'_grad_mean']=float(np.mean([r['gradient_norms'][key] for r in logs]))
    return row


def _rng_compare(left,right):
    left={r['epoch']:r for r in left}; right={r['epoch']:r for r in right}
    common=sorted(set(left)&set(right))
    mismatch=[e for e in common if any(left[e][k]!=right[e][k] for k in ('rng_before','rng_after_train','rng_after_eval'))]
    if not common: raise RuntimeError('no common RNG epochs')
    return {'common_epochs':len(common),'exact_triplets':len(common)-len(mismatch),'mismatches':len(mismatch),
            'mismatch_epochs':mismatch}


def audit():
    started=time.monotonic()
    manifest=json.loads((HERE/'manifest.json').read_text())
    if manifest['status'] not in ('PREFLIGHT_PASS','COMPLETE_AUDITED','TRAINING_COMPLETE_AUDIT_RETRY'): raise RuntimeError('unfrozen manifest')
    frozen=json.loads((RESULTS/'preflight.json').read_text())
    sha=ex.source_sha()
    training_sha=manifest.get('training_source_sha256',manifest['source_sha256'])
    current_map=ex.source_map()
    if sha!=manifest['source_sha256'] or training_sha!=frozen['source_sha256'] or frozen['status']!='PASS':
        raise RuntimeError('source/config/history SHA mismatch')
    if current_map!=manifest['source_map'] or current_map.keys()!=frozen['source_map'].keys():
        raise RuntimeError('source map differs from recorded revision')
    changed=[p for p in current_map if current_map[p]!=frozen['source_map'][p]]
    if changed:
        if changed!=[str(HERE/'audit.py')]:
            raise RuntimeError('formal training source/config/history changed after freeze')
        repair=json.loads((RUNTIME/'audit_repair.json').read_text())
        if ex.sha_file(RUNTIME/'audit_training_frozen.py')!=frozen['source_map'][str(HERE/'audit.py')]:
            raise RuntimeError('original frozen audit snapshot mismatch')
        if repair['corrected_audit_sha256']!=current_map[str(HERE/'audit.py')]:
            raise RuntimeError('audit-only revision SHA not recorded')
        reconstructed=dict(current_map)
        reconstructed[str(HERE/'audit.py')]=frozen['source_map'][str(HERE/'audit.py')]
        if ex.sha_value(reconstructed)!=training_sha:
            raise RuntimeError('training dependency reconstruction mismatch')
    replay=json.loads((RUNTIME/'independent_preflight_replay.json').read_text())
    if replay['status']!='PASS' or replay['folds_exactly_reproduced']!=10 or replay['source_sha256']!=training_sha:
        raise RuntimeError('independent post-training base-off/recovery replay failed')
    if ex.sha_file(RESULTS/'preflight.json')!=manifest['preflight_sha256']: raise RuntimeError('preflight SHA changed')
    assert frozen['base_off_parity']['passed']==10
    # Repeat deterministic module tests independently after training.
    repeated=ex.module_preflight()
    assert repeated==frozen['module_tests']
    values={g:{} for g in GROUPS}; references={}
    for name,path,scale in [(BASE_GROUP,ex.runtime().EXP/'results/fold_metrics.csv',1.),
                            (E811,Path('/model/spromptgl/experiment/experiment81_1_prcl_v2/results/fold_metrics.csv'),.01),
                            (E813,Path('/model/spromptgl/experiment/experiment81_3_e2e_pairwise_relational_learning/results/fold_metrics.csv'),.01)]:
        references[name]={int(r['fold']):{m:float(r[m])*scale for m in METRICS} for r in read_csv(path) if r['group']==name}
        if set(references[name])!=set(range(1,11)): raise RuntimeError(f'reference fold coverage: {name}')
    taskmap={}; rows=[]; mechanisms=[]; rng_reports={}; latest_checks=[]; best_checks=[]
    for fold in range(1,11):
        history_task=json.loads((ex.runtime().EXP/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())
        history_path=ex.runtime().EXP/'.runtime/checkpoints'/f'{BASE_GROUP}_f{fold:02d}.best.pt'
        history_payload=ex.torch_load(history_path)
        state,warmup,identity=ex.prepare(fold)
        assert identity==frozen['folds'][fold-1]['identity']
        ex.runtime().load_history().load_bundle(state,history_payload['best_state'])
        with torch.no_grad():
            historical_out=ex._base_forward(state,training=False)
            history=independent_metrics(historical_out['logp'],state)
        _assert_metrics(references[BASE_GROUP][fold],history,'historical independent reload')
        assert np.array_equal(history['pred'],history_payload['test_prediction'])
        assert np.allclose(history['prob'],history_payload['test_probability'],atol=1e-7,rtol=0)
        del historical_out,state
        for group in GROUPS:
            task_path,best_path,latest_path,log_path=ex.paths(group,fold)
            if not ex._task_complete(group,fold,training_sha): raise RuntimeError(f'incomplete/changed formal task: {group} {fold}')
            task=json.loads(task_path.read_text()); taskmap[(group,fold)]=task
            best=ex.torch_load(best_path); latest=ex.torch_load(latest_path)
            for item in (best,latest):
                if item['source_sha256']!=training_sha or item['identity']!=identity or item['group']!=group or item['fold']!=fold:
                    raise RuntimeError('checkpoint identity mismatch')
            state,_,_=ex.prepare(fold)
            module=ex.make_module(fold,state['x'].device)
            initsha=ex.sha_value(ex._cpu_state(module))
            if initsha!=task['initial_module_sha256'] or initsha!=frozen['folds'][fold-1]['initial_module_sha256']:
                raise RuntimeError('same-fold parameter initialization differs')
            optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
            ex.load_bundle(state,module,optimizer,latest['current'])
            ex._validate_optimizer_settings(state,optimizer)
            for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer): validate_optimizer(opt)
            assert {'python','numpy','cpu','cuda'}.issubset(latest['rng'])
            ex._rng_runtime().restore_rng(latest['rng'])
            assert ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())==ex._rng_runtime().rng_sha(latest['rng'])
            assert latest['next_epoch']==task['epochs_executed']
            assert len(latest['trace'])==latest['next_epoch']-10
            logs=[json.loads(l) for l in log_path.read_text().splitlines()]
            assert len(logs)==task['epochs_executed']-10
            assert [r['rng'] for r in logs]==task['rng_trace']==latest['trace']
            best_log=max(logs,key=lambda r:r['metrics']['acc'])
            assert best_log['epoch']==task['best_epoch_1based']==best['best_epoch_1based']
            if task['epochs_executed']<450: assert latest['wait']>200
            latest_checks.append({'group':group,'fold':fold,'four_optimizers_complete':True,'rng_complete':True,
                                  'strict_earliest_ACC_selection':True,'latest_sha256':ex.sha_file(latest_path)})
            ex.load_bundle(state,module,optimizer,best['best']['bundle'])
            for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer): validate_optimizer(opt)
            rng_before=ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
            with torch.no_grad():
                full=ex.forward_fusion(state,module,group,training=False)
                off=ex.forward_fusion(state,module,group,training=False,intervention='off')
                null=ex.forward_fusion(state,module,group,training=False,intervention='null')
                globaloff=ex.forward_fusion(state,module,group,training=False,intervention='global_off')
                unary=ex.forward_fusion(state,module,group,training=False,intervention='unary') if group==GROUPS[1] else None
            assert rng_before==ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
            assert torch.equal(null['logp'],off['logp']) and torch.equal(null['hidden'],off['hidden'])
            assert torch.count_nonzero(null['fusion']['delta'])==0 and torch.count_nonzero(null['fusion']['messages'])==0
            assert torch.equal(full['fusion']['u'],globaloff['fusion']['u'])
            assert bool(torch.isfinite(full['logp']).all())
            assert bool(torch.allclose(full['logp'].exp().sum(-1),torch.ones_like(full['logp'][:,0]),atol=2e-6))
            assert bool((full['fusion']['delta_ratio']<=.100001).all())
            metrics=independent_metrics(full['logp'],state); base_metrics=independent_metrics(off['logp'],state)
            _assert_metrics(task['metrics'],metrics,'task metric')
            _assert_metrics(best['metrics'],metrics,'checkpoint metric')
            assert np.array_equal(best['best']['test_prediction'],metrics['pred'])
            assert np.allclose(best['best']['test_probability'],metrics['prob'],rtol=0,atol=1e-7)
            # Confirm the pre-hook classifier output is exactly the historical cls input.
            cls_fn=ex.base_runtime().runtime()['utils'].ClsLoss
            cls=cls_fn(full['logp'],state['y'],state['train'],state['weights'])
            assert torch.equal(cls,full['cls'])
            mech=mechanism_row(group,fold,state,module,full,off,null,globaloff,unary,history,logs)
            mechanisms.append(mech)
            values[group][fold]=metric_values(metrics)
            rows.append({'group':group,'fold':fold,**{m:metrics[m]*100. for m in METRICS},
                         **{k:metrics[k] for k in ('asd_tp','asd_fn','td_fp','td_tn')},
                         **{'current_off_'+m:base_metrics[m]*100. for m in METRICS},
                         'best_epoch_1based':task['best_epoch_1based'],'epochs_executed':task['epochs_executed'],
                         'wall_seconds':task['wall_seconds'],'peak_gpu_bytes':task['peak_gpu_bytes'],
                         'checkpoint_sha256':ex.sha_file(best_path),'latest_sha256':ex.sha_file(latest_path),
                         'train_test_overlap_1based':json.dumps(identity['membership']['train_test_overlap_1based']),
                         'independent_checkpoint_audit':'PASS'})
            best_checks.append({'group':group,'fold':fold,'metrics_recomputed':True,'probability_prediction_replay':True,
                                'null_off_exact':True,'single_outputlayer_each_forward':True})
            rng_reports[f'{group}_f{fold:02d}_vs_DARG']=_rng_compare(task['rng_trace'],history_task['rng_trace'])
            del state,module,optimizer,full,off,null,globaloff,unary,latest,best
            torch.cuda.empty_cache()
            print(f'AUDIT {group} fold={fold} PASS',flush=True)
        rng_reports[f'between_groups_f{fold:02d}']=_rng_compare(taskmap[(GROUPS[0],fold)]['rng_trace'],taskmap[(GROUPS[1],fold)]['rng_trace'])
        assert taskmap[(GROUPS[0],fold)]['initial_module_sha256']==taskmap[(GROUPS[1],fold)]['initial_module_sha256']
    values.update(references)
    for name,reference in references.items():
        rows.extend({'group':name,'fold':f,**{m:reference[f][m]*100. for m in METRICS},'independent_checkpoint_audit':
                     'HISTORICAL_CHECKPOINT_RECOMPUTED' if name==BASE_GROUP else 'READ_ONLY_CONTEXT'} for f in range(1,11))
    summary=summary_values(values); pairs=paired_rows(values)
    mismatches=sum(r['mismatches'] for r in rng_reports.values())
    status={'status':'PASS' if not mismatches else 'FAIL','source_sha256':sha,
            'training_source_sha256':training_sha,'audit_only_repair':json.loads((RUNTIME/'audit_repair.json').read_text()),
            'independent_post_training_preflight_replay':replay,
            'formal_tasks_complete':20,'best_checkpoint_independent_pass':len(best_checks),
            'historical_best_checkpoint_independent_pass':10,'latest_state_integrity_pass':len(latest_checks),
            'base_off_parity_pass':10,'private_same_fold_initialization_pass':10,
            'eta_route_and_global_conditioning_repeated':'PASS','serialized_resume':frozen['folds'][0]['serialized_resume'],
            'module_tests_repeated':repeated,'rng_mismatch_count':mismatches,'rng_audit':rng_reports,
            'best_checks':best_checks,'latest_checks':latest_checks,'summary':summary,
            'fold2_overlap_1based':[377,407],'metrics_unit':'percent; population SD',
            'hard_auc_definition':'balanced accuracy of hard predictions','audit_seconds':time.monotonic()-started}
    status['research_assessment']=assess(summary,mechanisms)
    ex.write_csv(RESULTS/'fold_metrics.csv',rows)
    ex.write_csv(RESULTS/'paired_comparisons.csv',pairs)
    ex.write_csv(RESULTS/'mechanism.csv',mechanisms)
    ex.atomic_json(RESULTS/'audit_status.json',status)
    if status['status']!='PASS': raise RuntimeError('public RNG mismatch: investigate before accepting results')
    report=render_report(summary,pairs,mechanisms,status)
    (RESULTS/'report.md').write_text(report,encoding='utf-8')
    assert (RESULTS/'report.md').read_text()==report
    return status


def _sum(mechanisms,group,key,exclude2=False):
    return sum(float(r[key]) for r in mechanisms if r['group']==group and (not exclude2 or r['fold']!=2))


def assess(summary,mechanisms):
    g2,g1,darg=map(lambda name:summary[name]['all10'],(GROUPS[1],GROUPS[0],BASE_GROUP))
    base_met=g2['acc']['mean']>=92.1891 and g2['hard_auc']['mean']>=92.0294
    pair_gain=g2['acc']['mean']>g1['acc']['mean']+1e-10 and g2['hard_auc']['mean']>g1['hard_auc']['mean']+1e-10
    nine_support=all(summary[GROUPS[1]]['without_fold2'][m]['mean']>=summary[BASE_GROUP]['without_fold2'][m]['mean'] for m in ('acc','hard_auc'))
    mechanism=(_sum(mechanisms,GROUPS[1],'final_vs_history_net')>0 and
               _sum(mechanisms,GROUPS[1],'residual_vs_current_off_net')>0 and
               _sum(mechanisms,GROUPS[1],'final_vs_history_net',True)>0 and
               _sum(mechanisms,GROUPS[1],'global_guidance_vs_off_net')>0)
    if not base_met:
        verdict='NO_GO_E81_4'
        secondary='PAIRWISE_GAIN_BELOW_BASELINE' if pair_gain else 'PAIRWISE_GAIN_NOT_ESTABLISHED'
    elif not pair_gain:
        verdict='GLOBAL_FUSION_ONLY_PAIRWISE_NOT_ESTABLISHED'; secondary='BASELINE_FLOOR_MET'
    elif nine_support and mechanism:
        verdict='PRELIMINARY_GGLRF_GAIN'; secondary='SAME_PROTOCOL_PRELIMINARY_EVIDENCE_ONLY'
    else:
        verdict='NO_GO_E81_4'; secondary='PERFORMANCE_SIGNAL_WITHOUT_COMPLETE_MECHANISTIC_SUPPORT'
    return {'decision':verdict,'secondary':secondary,'baseline_floor_met':base_met,'pairwise_gain_over_G1':pair_gain,
            'without_fold2_performance_support':nine_support,'positive_mechanistic_net_support':mechanism}


def render_report(summary,pairs,mechanisms,status):
    lines=['# E81.4 — Global-Guided Local Relational Fusion (GGLRF)','',
           f"Independent audit: **{status['status']}**. Research decision: **{status['research_assessment']['decision']}**.",
           '', '## Source and implementation review','',
           'The [SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) and '
           '[official code](https://github.com/wanxixi11/SPromptGL) were checked against the '
           '[project implementation](https://github.com/tangaii/SPromptGL). RES motivates global-conditioned selection; '
           'this implementation retains 16 latent locals and writes a bounded residual into CAM144. '
           'The inherited PEN-free CAM and original 144→36→2 OutputLayer with dropout 0.35 remain active. '
           'GFR/DARG/CAM/OutputLayer/GGLRF all train end to end. No PEN, CLIP, text prompt or new classifier was added.',
           '', 'G1/G2 each add 16,384 trainable parameters, identical same-fold initialization, 15 real distinct neighbors '
           'and one zero-value Null. Only Unary versus product/signed-difference Key/Value construction differs. '
           'G1 query-key attention itself models implicit interactions; G2 is not compared against a relation-free model.',
           '', 'The pre-hook runs the original classifier once. Training reuses historical_out.total plus '
           '0.001 residual regularization. Eta=0.1 applies only to the new z_rel input path; the norm bound uses '
           'detached CAM and rho=0.10. Zero residual restores baseline gradients exactly. Nonzero fusion changes '
           'the classifier input and optimization trajectory, so this is not full-time historical-gradient preservation.',
           '', '## Six metrics','', 'Percent, ten-fold mean ± population SD. Hard-AUC is hard-prediction balanced accuracy; '
           'Prob-AUC is probability ROC-AUC. ASD=0, TD=1.','',
           '| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in (BASE_GROUP,GROUPS[0],GROUPS[1],E811,E813):
        s=summary[name]['all10']; lines.append('| '+name+' | '+' | '.join(f"{s[m]['mean']:.4f}±{s[m]['sd']:.4f}" for m in METRICS)+' |')
    lines+=['','## Paired changes','', 'Percentage points; W/T/L uses same-fold differences. The 95% paired t intervals '
            'are descriptive: overlapping training folds violate an independent patient-sampling interpretation.','',
            '| Comparison | Metric | Mean Δ (pp) | Descriptive 95% interval | W/T/L |',
            '|---|---|---:|---:|---:|']
    for row in pairs:
        if row['scope']=='all10':
            lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_difference_pp']:+.4f} | "
                         f"[{row['ci95_low_pp']:+.4f}, {row['ci95_high_pp']:+.4f}] | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines+=['','Per-fold differences for all six metrics are preserved in paired_comparisons.csv.',
            '', '## Excluding Fold 2','',
            '| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name in (BASE_GROUP,GROUPS[0],GROUPS[1]):
        s=summary[name]['without_fold2']; lines.append('| '+name+' | '+' | '.join(f"{s[m]['mean']:.4f}±{s[m]['sd']:.4f}" for m in METRICS)+' |')
    lines+=['','## Historical backbone / current backbone / feature contribution','',
            'The historical model is its original best checkpoint, never a training initializer. The current original '
            'branch is the trained E81.4 checkpoint with residual disabled. Counts below are test-fold patient predictions.',
            '', '| Group | Transition | Corrected | Harmed | Net | Net excluding Fold 2 |',
            '|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for prefix,label in [('current_off_vs_history','historical → current residual-off'),
                             ('residual_vs_current_off','current residual-off → full'),('final_vs_history','historical → full'),
                             ('global_guidance_vs_off','global-guidance-off → full')]:
            vals=[int(_sum(mechanisms,group,prefix+'_'+k)) for k in ('corrected','harmed','net')]
            lines.append(f'| {group} | {label} | {vals[0]} | {vals[1]} | {vals[2]:+d} | {int(_sum(mechanisms,group,prefix+"_net",True)):+d} |')
    lines+=['','Current trained residual-off metrics (ten-fold mean ± population SD):','',
            '| Group | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        rr=[r for r in mechanisms if r['group']==group]
        lines.append('| '+group+' | '+' | '.join(f"{np.mean([r['current_off_'+m] for r in rr]):.4f}±{np.std([r['current_off_'+m] for r in rr]):.4f}" for m in METRICS)+' |')
    lines+=['','## Frozen inference interventions','',
            'Each intervention uses the selected trained checkpoint and unchanged learned parameters. Global-off zeros '
            'global inputs to both relation and readout queries while keeping local input fixed. Pairwise→Unary uses G2 '
            'projections on Unary inputs and entails distribution replacement; it does not replace independent G1/G2 training.',
            '', '| Group | Intervention | Prob MAE | ΔACC pp | ΔBA pp | ΔF1 pp | Flips | Corrected / harmed vs full |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        rr=[r for r in mechanisms if r['group']==group]
        for mode in ('residual_off','null_only','global_off')+(('pairwise_to_unary',) if group==GROUPS[1] else ()):
            avg=lambda key:float(np.mean([r[mode+'_'+key] for r in rr]))
            total=lambda key:int(sum(r[mode+'_'+key] for r in rr))
            lines.append(f"| {group} | {mode} | {avg('probability_mae'):.6f} | {avg('acc_change_pp'):+.4f} | "
                         f"{avg('hard_auc_change_pp'):+.4f} | {avg('asd_f1_change_pp'):+.4f} | {total('flips')} | {total('corrected')} / {total('harmed')} |")
    lines+=['','Null-only and residual-off logp/hidden are exactly equal for all 20 checkpoints.',
            '', '## Global readout, residual and gradient diagnostics','']
    keys=['beta_entropy','beta_max_mean','beta_patient_sd_mean','beta_near_uniform_fraction',
          'beta_single_local_collapse_fraction','attention_entropy','null_weight','message_norm_mean',
          'head_message_cancellation_ratio','local_direction_cancellation_ratio','global_query_norm',
          'delta_ratio_mean','delta_ratio_p50','delta_ratio_p95','near_bound_fraction','near_zero_residual_fraction',
          'residual_hidden_mae','global_off_beta_mae','global_off_attention_mae','global_off_delta_mae',
          'train_global_encoder_grad_mean','train_global_readout_grad_mean','train_delta144_mean',
          'train_routed_to_total_cam144_ratio_mean','train_routed_to_total_cam144_ratio_first10_mean',
          'train_routed_to_total_cam144_ratio_last10_mean']
    lines+=['| Diagnostic (fold means) | G1 | G2 |','|---|---:|---:|']
    for key in keys:
        a=[np.mean([r[key] for r in mechanisms if r['group']==group]) for group in GROUPS]
        lines.append(f'| {key} | {a[0]:.7g} | {a[1]:.7g} |')
    lines+=['','Normalized entropies are in [0,1]. The routed/total gradient-norm ratio is a descriptive norm ratio, '
            'not a bound on parameter drift. Eta is exactly 0.1 on the new input path; a new path can still be small, '
            'large, or partly cancel the identity path. Per-local beta and residual vectors, confidence/error partitions '
            'and global-off changes are in mechanism.csv. Head/local cancellation ratios diagnose averaging directions; '
            '16 locals are not averaged by the model. Latent tokens have no validated anatomical meaning.',
            '', '## Training, reproducibility and audit','']
    training=json.loads((RUNTIME/'training_summary.json').read_text()) if (RUNTIME/'training_summary.json').exists() else {}
    lines+=[f"Formal tasks: 20/20; independent best reload: 20/20; latest/four optimizers: 20/20; historical best reload: 10/10. "
            f"Base-off exact checks: 10/10; shared-epoch RNG mismatches: {status['rng_mismatch_count']}.",
            '', f"Scheduler: {training.get('max_workers','see runtime')} workers maximum, "
            f"{training.get('wall_seconds',0):.1f}s dispatch wall time; failures: {len(training.get('failures',[]))}. "
            'One CPU thread per worker; private CPU initialization; no public reseeding or new stochastic layer.',
            '', f"Serialized fresh-model two-step replay: {status['serialized_resume']}. "
            'Preflight checked the first zero-head gradient and second-update global/local/readout gradients, '
            '15 unique neighbors, patient isolation, eta 0/.1/1 and detached norm boundaries.',
            '', f"Frozen training source/config/history SHA256: `{status['training_source_sha256']}`; corrected audit/source SHA256: `{status['source_sha256']}`. All task/checkpoint SHA values, "
            'source dependencies and independent records are available in manifest.json and audit_status.json.',
            '', 'The first independent audit stopped on an audit-only dict-versus-Logp argument error. The corrected audit '
            'selects out[logp]; no training logic/configuration or completed checkpoint changed. The original frozen audit '
            'is preserved in runtime, and every other dependency is checked against the original source map. A fresh '
            'post-training process repeated all ten exact Base-off checks and serialized recovery without replacing '
            'the original frozen preflight.',
            '', '## Scientific interpretation and limits','',
            f"Decision: **{status['research_assessment']['decision']}**; secondary: `{status['research_assessment']['secondary']}`.",
            '', f"Baseline floor met: {status['research_assessment']['baseline_floor_met']}; "
            f"G2 ACC and BA improve over G1: {status['research_assessment']['pairwise_gain_over_G1']}; "
            f"nine-fold support: {status['research_assessment']['without_fold2_performance_support']}; "
            f"positive residual/global/final net support: {status['research_assessment']['positive_mechanistic_net_support']}.",
            '', 'The comparison with E81.1/E81.3 is historical context, not a controlled attribution of the integration '
            'point alone: their complete architectures and objectives differ. Classification changes from current-off '
            'to full isolate the selected checkpoint residual effect; changes from historical to current-off also '
            'include changed training and selected epoch. An active attention or a nonzero residual alone does not '
            'establish diagnostic gain.',
            '', 'The inherited protocol selects checkpoints using the test fold (val=test), uses preprocessing/RFE '
            'before CV and a transductive graph, and preserves Fold 2 training/test overlap IDs 377/407. New training '
            'residuals and regularization exclude those two rows, but the historical and indirect effects remain. '
            'Therefore these are same-protocol exploratory results, not unbiased independent test estimates. '
            'Fold intervals are descriptive and no MICCAI-level validation or medical prompt effectiveness is claimed. '
            'No E81.5 is launched.',
            '', '## Artifacts','',
            'Code/config/design/manifest are in this experiment directory. results/ contains preflight.json, '
            'fold_metrics.csv, paired_comparisons.csv, mechanism.csv, audit_status.json and this report. '
            'runtime/ holds only necessary best/latest checkpoints, task ledgers, compact logs and scheduling provenance.','']
    return '\n'.join(lines)


if __name__=='__main__':
    try:
        result=audit()
        print(json.dumps({'status':result['status'],'assessment':result['research_assessment']},ensure_ascii=False),flush=True)
    except Exception as error:
        ex.atomic_json(RESULTS/'audit_status.json',{'status':'FAIL','error':repr(error),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        raise

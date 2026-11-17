#!/usr/bin/env python3
"""Independent best/latest, metric, RNG and mechanism audit for E81.5."""
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
E814=ex.E814_GROUP
CURRENT_OFF=GROUPS[0]+'_ResidualOff'
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
    pairs=[('E815_vs_E814_capped',GROUPS[0],E814),('E815_vs_DARG',GROUPS[0],BASE_GROUP),
           ('E815_vs_E813_context',GROUPS[0],E813)]
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
         'delta_ratio_max':float(ratio.max()),
         'above10_fraction':float((ratio>.1).float().mean()),
         'above20_fraction':float((ratio>.2).float().mean()),
         'above50_fraction':float((ratio>.5).float().mean()),
         'above100_fraction':float((ratio>1.).float().mean()),
         'enhanced_norm_mean':float(d['z_enhanced'][ids].norm(dim=-1).mean()),
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
    for key in ('res','weighted_res','weighted_res_to_cls'):
        vals=np.asarray([r['losses'][key] for r in logs])
        row['train_'+key+'_mean']=float(vals.mean())
        row['train_'+key+'_max']=float(vals.max())
    for key in ('residual_regularizer_weighted','regularizer_to_gglrf','local_encoder','key','value','query'):
        vals=np.asarray([r['gradient_norms'][key] for r in logs])
        row['train_'+key+'_grad_mean']=float(vals.mean())
        row['train_'+key+'_grad_max']=float(vals.max())
    for key in ('delta_ratio_mean','delta_ratio_max','raw_delta_norm_mean'):
        vals=np.asarray([r['diagnostics'][key] for r in logs])
        row['train_'+key+'_max']=float(vals.max())
        row['train_'+key+'_first10_mean']=float(vals[:10].mean())
        row['train_'+key+'_last10_mean']=float(vals[-10:].mean())
    return row


def _rng_compare(left,right):
    left={r['epoch']:r for r in left}; right={r['epoch']:r for r in right}
    common=sorted(set(left)&set(right))
    mismatch=[e for e in common if any(left[e][k]!=right[e][k] for k in ('rng_before','rng_after_train','rng_after_eval'))]
    if not common: raise RuntimeError('no common RNG epochs')
    return {'common_epochs':len(common),'exact_triplets':len(common)-len(mismatch),'mismatches':len(mismatch),
            'mismatch_epochs':mismatch}


def frozen_e814_diagnostic(fold, state, module):
    saved=ex.torch_load(ex.E814_ROOT/'runtime/checkpoints'/f'{E814}_f{fold:02d}.best.pt')
    optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
    ex.load_bundle(state,module,optimizer,saved['best']['bundle'])
    module.residual_mode='capped'
    with torch.no_grad(): capped=ex.forward_fusion(state,module,GROUPS[0],training=False)
    module.residual_mode='unbounded'
    with torch.no_grad(): unbounded=ex.forward_fusion(state,module,GROUPS[0],training=False)
    for k in ('u','g','attention','beta','raw_delta'):
        assert torch.equal(capped['fusion'][k],unbounded['fusion'][k])
    cm,um=independent_metrics(capped['logp'],state),independent_metrics(unbounded['logp'],state)
    _assert_metrics(saved['metrics'],cm,'frozen E814 capped')
    ids=torch.as_tensor(state['test'],device=state['x'].device)
    labels=state['y'][ids].cpu().numpy()
    stats={'frozen_e814_'+k:v for k,v in intervention_stats(cm,um,labels).items()}
    stats['frozen_e814_hidden_mae']=float((capped['hidden'][ids]-unbounded['hidden'][ids]).abs().mean())
    for name,out in [('capped',capped),('unbounded',unbounded)]:
        d=out['fusion']; ratio=d['delta_ratio'][ids]
        stats.update({f'frozen_e814_{name}_'+m:independent_metrics(out['logp'],state)[m]*100. for m in METRICS})
        stats.update({f'frozen_e814_{name}_ratio_mean':float(ratio.mean()),
                      f'frozen_e814_{name}_ratio_p50':float(torch.quantile(ratio,.5)),
                      f'frozen_e814_{name}_ratio_p95':float(torch.quantile(ratio,.95)),
                      f'frozen_e814_{name}_ratio_max':float(ratio.max()),
                      f'frozen_e814_{name}_delta_norm_mean':float(d['delta'][ids].norm(dim=-1).mean()),
                      f'frozen_e814_{name}_all_finite':bool(torch.isfinite(out['logp']).all())})
    return stats


def read_training_logs(log_path,task):
    rows=[json.loads(line) for line in log_path.read_text().splitlines()]
    expected=set(range(11,task['epochs_executed']+1))
    actual={r['epoch'] for r in rows}
    assert len(rows)==len(actual) and actual.issubset(expected)
    missing=sorted(expected-actual)
    if missing:
        path=RUNTIME/f"log_recovery_f{task['fold']:02d}.json"
        rec=json.loads(path.read_text())
        assert rec['status']=='PASS_EXACT' and rec['missing_epochs']==missing
        assert rec['source_sha256']==task['source_sha256']
        assert rec['original_records_exact']==len(rows) and rec['replayed_epochs']==len(expected)
        assert rec['best_and_latest_four_optimizer_model_rng_exact']
        assert rec['replay_script_sha256']==ex.sha_file(RUNTIME/'recover_missing_logs.py')
        for original,sha in rec['original_artifacts_unchanged'].items():assert ex.sha_file(original)==sha
        recovered=rec['recovered_rows']
        assert {r['epoch'] for r in recovered}==set(missing) and len(recovered)==len(missing)
        trace={r['epoch']:r for r in task['rng_trace']}
        for row in recovered:
            assert row['group']==task['group'] and row['fold']==task['fold']
            assert row['rng']==trace[row['epoch']]
        rows=sorted(rows+recovered,key=lambda row:row['epoch'])
    assert [r['epoch'] for r in rows]==sorted(expected)
    return rows,{'raw_missing_epochs':missing,'exact_replay_recovered':len(missing),
                 'original_log_sha256':ex.sha_file(log_path),
                 'recovery_sha256':ex.sha_file(RUNTIME/f"log_recovery_f{task['fold']:02d}.json") if missing else None}


def audit():
    started=time.monotonic()
    manifest=json.loads((HERE/'manifest.json').read_text())
    frozen=json.loads((RESULTS/'preflight.json').read_text())
    current_map=ex.source_map();current_sha=ex.sha_value(current_map)
    sha=frozen['source_sha256']
    assert sha==manifest['training_source_sha256']
    assert current_sha==manifest['source_sha256'] and current_map==manifest['source_map']
    changed=[path for path in current_map if current_map[path]!=frozen['source_map'][path]]
    assert changed==[str(HERE/'audit.py')]
    snapshot=RUNTIME/'audit_training_frozen.py'
    assert ex.sha_file(snapshot)==frozen['source_map'][str(HERE/'audit.py')]
    reconstructed=dict(current_map);reconstructed[str(HERE/'audit.py')]=ex.sha_file(snapshot)
    assert reconstructed==frozen['source_map'] and ex.sha_value(reconstructed)==sha
    repair=json.loads((RUNTIME/'audit_repair.json').read_text())
    assert repair['corrected_audit_sha256']==ex.sha_file(HERE/'audit.py')
    assert ex.sha_file(RESULTS/'preflight.json')==manifest['preflight_sha256']
    assert frozen['status']=='PASS' and frozen['base_off_parity']['passed']==10
    # The first independent audit completed all ten exact preflight replays
    # before detecting missing raw scalar rows. Training code is unchanged.
    replay=json.loads((RUNTIME/'independent_preflight_replay.json').read_text())
    assert ex.sha_file(RUNTIME/'independent_preflight_replay.json')==repair['independent_preflight_replay_sha256']
    for k in ('module_tests','folds','base_off_parity','source_map','source_sha256'):
        ex._assert_exact(frozen[k],replay[k],'independent_preflight.'+k)
    values={GROUPS[0]:{},CURRENT_OFF:{}}; references={}
    for name,path,scale in [(BASE_GROUP,ex.runtime().EXP/'results/fold_metrics.csv',1.),
                           (E814,ex.E814_ROOT/'results/fold_metrics.csv',.01),
                           (E813,Path('/model/spromptgl/experiment/experiment81_3_e2e_pairwise_relational_learning/results/fold_metrics.csv'),.01)]:
        references[name]={int(r['fold']):{m:float(r[m])*scale for m in METRICS} for r in read_csv(path) if r['group']==name}
        assert set(references[name])==set(range(1,11)),name
    rows=[]; mechanisms=[]; rng_reports={}; latest_checks=[]; best_checks=[];log_integrity={}
    group=GROUPS[0]
    for fold in range(1,11):
        hist_task=json.loads((ex.runtime().EXP/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())
        hist_payload=ex.torch_load(ex.runtime().EXP/'.runtime/checkpoints'/f'{BASE_GROUP}_f{fold:02d}.best.pt')
        old_task=json.loads((ex.E814_ROOT/'runtime/tasks'/f'{E814}_f{fold:02d}.json').read_text())
        state,warmup,identity=ex.prepare(fold)
        assert identity==frozen['folds'][fold-1]['identity']
        ex.runtime().load_history().load_bundle(state,hist_payload['best_state'])
        with torch.no_grad(): history=independent_metrics(ex._base_forward(state,False)['logp'],state)
        _assert_metrics(references[BASE_GROUP][fold],history,'historical DARG reload')
        assert np.array_equal(history['pred'],hist_payload['test_prediction'])
        assert np.allclose(history['prob'],hist_payload['test_probability'],atol=1e-7,rtol=0)
        task_path,best_path,latest_path,log_path=ex.paths(group,fold)
        if not ex._task_complete(group,fold,sha): raise RuntimeError(f'incomplete task fold{fold}')
        task=json.loads(task_path.read_text());best=ex.torch_load(best_path);latest=ex.torch_load(latest_path)
        for item in (best,latest):
            assert item['source_sha256']==sha and item['identity']==identity and item['group']==group and item['fold']==fold
        state,_,_=ex.prepare(fold);module=ex.make_module(fold,state['x'].device)
        initsha=ex.sha_value(ex._cpu_state(module))
        assert initsha==task['initial_module_sha256']==old_task['initial_module_sha256']==frozen['folds'][fold-1]['initial_module_sha256']
        assert module.residual_mode=='unbounded' and sum(p.numel() for p in module.parameters())==16384
        optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
        ex.load_bundle(state,module,optimizer,latest['current']);ex._validate_optimizer_settings(state,optimizer)
        for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer):validate_optimizer(opt)
        assert {'python','numpy','cpu','cuda'}.issubset(latest['rng'])
        ex._rng_runtime().restore_rng(latest['rng'])
        assert ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())==ex._rng_runtime().rng_sha(latest['rng'])
        assert latest['next_epoch']==task['epochs_executed'] and len(latest['trace'])==latest['next_epoch']-10
        logs,log_integrity[str(fold)]=read_training_logs(log_path,task)
        assert len(logs)==task['epochs_executed']-10
        assert [r['rng'] for r in logs]==task['rng_trace']==latest['trace']
        best_log=max(logs,key=lambda r:r['metrics']['acc'])
        assert best_log['epoch']==task['best_epoch_1based']==best['best_epoch_1based']
        wait=0;maximum=-1.;earliest=-1
        for r in logs:
            if r['metrics']['acc']>maximum:maximum=r['metrics']['acc'];earliest=r['epoch'];wait=0
            else:wait+=1
        assert latest['wait']==wait and latest['best_epoch']+1==earliest
        if task['epochs_executed']<450:assert wait>200
        for r in logs:
            assert all(math.isfinite(v) for v in r['losses'].values())
            assert all(math.isfinite(v) for v in r['gradient_norms'].values())
            assert abs(r['losses']['weighted_res']-.001*r['losses']['res'])<1e-12
            assert r['gradient_route']['eta']==.1
        latest_checks.append({'fold':fold,'four_optimizers_complete':True,'all_rng_complete':True,'strict_earliest_ACC_selection':True,'latest_sha256':ex.sha_file(latest_path)})
        ex.load_bundle(state,module,optimizer,best['best']['bundle'])
        before=ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
        with torch.no_grad():
            full=ex.forward_fusion(state,module,group,False)
            off=ex.forward_fusion(state,module,group,False,'off')
            null=ex.forward_fusion(state,module,group,False,'null')
            globaloff=ex.forward_fusion(state,module,group,False,'global_off')
            unary=ex.forward_fusion(state,module,group,False,'unary')
        assert before==ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
        assert torch.equal(null['logp'],off['logp']) and torch.equal(null['hidden'],off['hidden'])
        assert torch.count_nonzero(null['fusion']['delta'])==0 and torch.count_nonzero(null['fusion']['messages'])==0
        assert torch.equal(full['fusion']['delta'],full['fusion']['raw_delta'])
        assert torch.equal(full['fusion']['u'],globaloff['fusion']['u'])
        assert torch.isfinite(full['logp']).all() and torch.isfinite(full['fusion']['delta']).all()
        assert torch.allclose(full['logp'].exp().sum(-1),torch.ones_like(full['logp'][:,0]),atol=2e-6)
        metrics=independent_metrics(full['logp'],state);base_metrics=independent_metrics(off['logp'],state)
        _assert_metrics(task['metrics'],metrics,'task');_assert_metrics(best['metrics'],metrics,'best')
        assert np.array_equal(metrics['pred'],best['best']['test_prediction'])
        assert np.allclose(metrics['prob'],best['best']['test_probability'],atol=1e-7,rtol=0)
        cls_fn=ex.base_runtime().runtime()['utils'].ClsLoss
        assert torch.equal(cls_fn(full['logp'],state['y'],state['train'],state['weights']),full['cls'])
        d=full['fusion'];ids=torch.as_tensor(state['train'],device=state['x'].device);ids=ids[d['legal_rows'][ids]]
        assert torch.equal(full['res'],d['delta_ratio'][ids].square().mean())
        row=mechanism_row(group,fold,state,module,full,off,null,globaloff,unary,history,logs)
        row.update(frozen_e814_diagnostic(fold,state,module));mechanisms.append(row)
        values[group][fold]=metric_values(metrics);values[CURRENT_OFF][fold]=metric_values(base_metrics)
        rows.append({'group':group,'fold':fold,**{m:metrics[m]*100. for m in METRICS},
                     **{k:metrics[k] for k in ('asd_tp','asd_fn','td_fp','td_tn')},
                     **{'current_off_'+m:base_metrics[m]*100. for m in METRICS},
                     'best_epoch_1based':task['best_epoch_1based'],'epochs_executed':task['epochs_executed'],
                     'wall_seconds':task['wall_seconds'],'peak_gpu_bytes':task['peak_gpu_bytes'],
                     'checkpoint_sha256':ex.sha_file(best_path),'latest_sha256':ex.sha_file(latest_path),
                     'train_test_overlap_1based':json.dumps(identity['membership']['train_test_overlap_1based']),
                     'independent_checkpoint_audit':'PASS'})
        best_checks.append({'fold':fold,'six_metrics_recomputed':True,'prediction_probability_replay':True,'raw_equals_delta':True,'null_off_exact':True})
        rng_reports[f'f{fold:02d}_vs_DARG']=_rng_compare(task['rng_trace'],hist_task['rng_trace'])
        rng_reports[f'f{fold:02d}_vs_E814']=_rng_compare(task['rng_trace'],old_task['rng_trace'])
        del state,module,optimizer,full,off,null,globaloff,unary,best,latest,hist_payload
        torch.cuda.empty_cache()
        print(f'AUDIT E81.5 fold={fold} PASS',flush=True)
    values.update(references);summary=summary_values(values);pairs=paired_rows(values)
    for name,ref in references.items():
        rows.extend({'group':name,'fold':f,**{m:ref[f][m]*100. for m in METRICS},'independent_checkpoint_audit':'HISTORICAL_RELOAD' if name==BASE_GROUP else 'READ_ONLY_CONTEXT'} for f in range(1,11))
    mismatch=sum(r['mismatches'] for r in rng_reports.values())
    status={'status':'PASS' if mismatch==0 else 'FAIL','source_sha256':current_sha,'training_source_sha256':sha,
            'audit_revision':repair,'log_integrity':log_integrity,
            'raw_scalar_records_recovered':sum(v['exact_replay_recovered'] for v in log_integrity.values()),
            'formal_tasks_complete':10,'best_checkpoint_independent_pass':len(best_checks),'latest_state_integrity_pass':len(latest_checks),
            'historical_best_checkpoint_independent_pass':10,'base_off_parity_pass':10,'same_e814_initialization_pass':10,
            'capped_checkpoint_exact_replay_pass':10,'serialized_resume':frozen['folds'][0]['serialized_resume'],
            'independent_preflight_replayed':True,'independent_preflight_sha256':ex.sha_file(RUNTIME/'independent_preflight_replay.json'),
            'rng_mismatch_count':mismatch,'rng_audit':rng_reports,'best_checks':best_checks,'latest_checks':latest_checks,
            'summary':summary,'fold2_overlap_1based':[377,407],'audit_seconds':time.monotonic()-started,
            'metrics_unit':'percent; population SD','hard_auc_definition':'hard-prediction balanced accuracy'}
    status['research_assessment']=assess(summary,mechanisms)
    assert ex.source_sha()==current_sha,'historical or current dependency changed during audit'
    ex.write_csv(RESULTS/'fold_metrics.csv',rows);ex.write_csv(RESULTS/'paired_comparisons.csv',pairs)
    ex.write_csv(RESULTS/'mechanism.csv',mechanisms);ex.atomic_json(RESULTS/'audit_status.json',status)
    if mismatch:raise RuntimeError('public RNG mismatch: locate cause; do not accept results')
    report=render_report(summary,pairs,mechanisms,status)
    (RESULTS/'report.md').write_text(report,encoding='utf-8');assert (RESULTS/'report.md').read_text()==report
    return status


def assess(summary,mechanisms):
    new=summary[GROUPS[0]]['all10'];old=summary[E814]['all10'];base=summary[BASE_GROUP]['all10']
    gains={m:new[m]['mean']-old[m]['mean'] for m in METRICS}
    vsbase={m:new[m]['mean']-base[m]['mean'] for m in METRICS}
    nine={m:summary[GROUPS[0]]['without_fold2'][m]['mean']-summary[BASE_GROUP]['without_fold2'][m]['mean'] for m in METRICS}
    net=sum(r['final_vs_history_net'] for r in mechanisms)
    net9=sum(r['final_vs_history_net'] for r in mechanisms if r['fold']!=2)
    both=vsbase['acc']>1e-10 and vsbase['hard_auc']>1e-10
    if gains['acc']<-1e-10 or gains['hard_auc']<-1e-10:decision='NO_GO_UNBOUNDED'
    elif vsbase['acc']>1e-10 and vsbase['hard_auc']<=1e-10:decision='PARTIAL_GAIN_NOT_ESTABLISHED'
    elif not both:decision='UNBOUNDED_GAIN_BELOW_BASELINE' if gains['acc']>1e-10 or gains['hard_auc']>1e-10 else 'NO_GO_UNBOUNDED'
    elif net>0 and net9>0 and nine['acc']>0 and nine['hard_auc']>0:
        decision='PRELIMINARY_UNBOUNDED_GAIN'
    else:decision='PARTIAL_GAIN_NOT_ESTABLISHED'
    return {'decision':decision,'vs_E814_difference_pp':gains,'vs_DARG_difference_pp':vsbase,
            'without_fold2_vs_DARG_pp':nine,'final_net_corrected':net,'final_net_without_fold2':net9,
            'both_primary_metrics_exceed_DARG':both,
            'interpretation':'same historical protocol; no claim of validated medical prompts or MICCAI-level independent innovation'}


def render_report(summary,pairs,mechanisms,status):
    group=GROUPS[0];avg=lambda key:float(np.mean([r[key] for r in mechanisms]))
    total=lambda key:int(sum(r[key] for r in mechanisms))
    lines=['# E81.5 — Unbounded Global-Guided Local Relational Feature Fusion','',
           f"Independent audit: **{status['status']}**. Research decision: **{status['research_assessment']['decision']}**.",'',
           '## Source, single-variable contrast and training identity','',
           'Reviewed the [original paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), '
           '[official source](https://github.com/wanxixi11/SPromptGL) and '
           '[published E81.4 commit](https://github.com/tangaii/SPromptGL/commit/3e599ea9a2cdd3c90b6d4e5ce557e8031922d274). '
           'All 13 local E81.4 published files match that commit byte for byte. GFR learns cosine patient graphs and DGCN propagation; '
           'CAM combines cross-modal messages. PEN SPGS uses frozen CLIP prompt similarities; RES equation6 adds weighted locals to global features. '
           'The official implementation uses fixed local reshaping and feature-coordinate softmax, which differs from the written RES abstraction. '
           'Neither the paper nor its RES defines the E81.4-specific 10% cap. PEN/CLIP/prompts remain excluded.', '',
           'The sole formal architectural change is delta=raw_delta, replacing the smooth 10% norm mapping. No tanh, gate, parameter, '
           'new classifier or new token layout is introduced. The same 16,384 parameters, private CPU salt814000+fold and exact same-fold '
           'initial state_dict are verified. Global/local encoders, 8 heads, 15 nonself real neighbors, zero-value Null, Pairwise products '
           'and signed differences, global beta, selected=16*beta*m_local and shared Linear(32,9) remain identical. '
           'All ten historical trained checkpoints reproduce exactly in local capped mode, including attention/beta/raw/delta/CAM/logp/hidden/loss.', '',
           'Each formal task starts independently from E61 epoch10 through audited E74.4-G2 prepare(fold), restoring full model, '
           'three optimizers, RNG, split and 4096 pairs. The fourth AdamW(.0005,.01) steps last. Classification uses the original '
           'LayerNorm144→Linear144,36→GELU→Dropout.35→Linear36,2→log_softmax once. Loss is historical_out.total + .001*L_res; '
           'L_res=mean_legal_train((norm(delta)/(norm(CAM.detach)+eps))²). There is no coefficient search or adjustment.', '',
           'Eta=.1 scales only the new relation-input path; the direct identity path remains unscaled and the norm denominator is detached. '
           'Zero-head gradients/updates match DARG exactly. After learning a residual, classifier inputs and parameter trajectories change; '
           'common RNG does not imply common optimization. Regularizer-gradient telemetry uses autograd.grad without parameter.grad accumulation '
           'or RNG consumption, followed by one total backward.', '',
           'The E81.4 training aggregate SHA is 7dd23d91341597d6eae1f7ae648c718279aed9f59b0cdd5021982dd4d5ae5b68; '
           'its corrected audit aggregate is 0fa19058f70a75838e62b43599820d1ac763827a001fd33754259887496ee86c. '
           'The historical audit-only dict/Logp fix is preserved separately and no historical files or checkpoints are changed.', '',
           '## Six metrics','',
           'Percent, ten-fold mean ± population SD; Hard-AUC=balanced accuracy of hard labels; Prob-AUC=probability ROC-AUC; ASD=0, TD=1.','',
           '| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in (BASE_GROUP,E814,group,E813,CURRENT_OFF):
        s=summary[name]['all10'];lines.append('| '+name+' | '+' | '.join(f"{s[m]['mean']:.4f}±{s[m]['sd']:.4f}" for m in METRICS)+' |')
    lines+=['','## Paired fold changes','',
            'All differences are percentage points. The paired t95% intervals are descriptive only: cross-validation training sets overlap. '
            'E81.3 is background context, not the single-variable comparator.','',
            '| Comparison | Metric | Mean Δ | Descriptive95% interval | W/T/L |','|---|---|---:|---:|---:|']
    for r in pairs:
        if r['scope']=='all10':lines.append(f"| {r['comparison']} | {r['metric']} | {r['mean_difference_pp']:+.4f} | [{r['ci95_low_pp']:+.4f}, {r['ci95_high_pp']:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','Per-fold differences for all metrics and both scopes are stored in paired_comparisons.csv.','',
            '## Excluding Fold2','', '| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in (BASE_GROUP,E814,group,CURRENT_OFF):
        s=summary[name]['without_fold2'];lines.append('| '+name+' | '+' | '.join(f"{s[m]['mean']:.4f}±{s[m]['sd']:.4f}" for m in METRICS)+' |')
    lines+=['','## Residual size and classification effect','',
            'No theoretical10% bound remains; actual learned size is measured rather than assumed. Ratio statistics below average '
            'fold-specific test-patient summaries; the reported overall maximum is the maximum over folds.','',
            '| Statistic | E81.5 |','|---|---:|']
    for key in ('delta_ratio_mean','delta_ratio_p50','delta_ratio_p95','above10_fraction','above20_fraction','above50_fraction','above100_fraction',
                'raw_delta_norm_mean','delta_norm_mean','cam_norm_mean','enhanced_norm_mean','residual_hidden_mae','residual_probability_mae',
                'beta_entropy','beta_max_mean','beta_patient_sd_mean','attention_entropy','null_weight','message_norm_mean','global_query_norm',
                'global_off_beta_mae','global_off_attention_mae','global_off_delta_mae'):
        lines.append(f'| {key} | {avg(key):.8g} |')
    lines+=[f"| delta_ratio_overall_max | {max(r['delta_ratio_max'] for r in mechanisms):.8g} |",'',
            'Historical capped E81.4-G2: mean9.8224%, median9.8814%, P95 9.9699%, and61.37% at or above9.9%. '
            'The ratio distribution is not an anatomical or medical-confidence measure.','',
            '## Historical backbone → current off → full','',
            '| Transition | Corrected | Harmed | Net | Net without Fold2 |','|---|---:|---:|---:|---:|']
    for prefix,label in [('current_off_vs_history','DARG → current residual-off'),('residual_vs_current_off','current residual-off → full'),('final_vs_history','DARG → full')]:
        lines.append(f"| {label} | {total(prefix+'_corrected')} | {total(prefix+'_harmed')} | {total(prefix+'_net'):+d} | {sum(r[prefix+'_net'] for r in mechanisms if r['fold']!=2):+d} |")
    lines+=['','Counts are held-out fold predictions. Current-off uses the checkpoint selected for full-model ACC; it is not a separately '
            'trained or independently selected best baseline. E81.4-G2 previously showed −15/+15/0 net corrections across these three stages.','',
            '## Frozen historical E81.4 weights: remove cap at inference','',
            'These interventions never train, select checkpoints, or adjust formal hyperparameters. Raw_delta, attention and beta are identical '
            'between capped/unbounded at the same frozen weights. This differs from the formal independent re-training contrast.','',
            '| Frozen mode | ACC | BA | Prob-AUC | SEN | SPE | ASD-F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('capped','unbounded'):
        lines.append('| '+mode+' | '+' | '.join(f"{avg('frozen_e814_'+mode+'_'+m):.4f}±{np.std([r['frozen_e814_'+mode+'_'+m] for r in mechanisms]):.4f}" for m in METRICS)+' |')
    lines+=['',f"Frozen removal: mean ratio {avg('frozen_e814_unbounded_ratio_mean')*100:.4f}%; probability MAE {avg('frozen_e814_probability_mae'):.6f}; "
            f"flips {total('frozen_e814_flips')}; corrected/harmed {total('frozen_e814_corrected')}/{total('frozen_e814_harmed')}; "
            f"ΔACC {avg('frozen_e814_acc_change_pp'):+.4f}pp; ΔBA {avg('frozen_e814_hard_auc_change_pp'):+.4f}pp. "
            f"All frozen outputs finite: {all(r['frozen_e814_unbounded_all_finite'] for r in mechanisms)}.", '',
            '## Regularizer, gradients and stability','', '| Scalar (fold mean) | Value |','|---|---:|']
    for key in ('train_res_mean','train_res_max','train_weighted_res_mean','train_weighted_res_max','train_weighted_res_to_cls_mean','train_weighted_res_to_cls_max',
                'train_residual_regularizer_weighted_grad_mean','train_regularizer_to_gglrf_grad_mean','train_global_encoder_grad_mean','train_local_encoder_grad_mean',
                'train_key_grad_mean','train_value_grad_mean','train_output_projection_grad_mean','train_gglrf_grad_mean','train_darg_grad_mean','train_cam_grad_mean',
                'train_routed_to_total_cam144_ratio_mean','train_routed_to_total_cam144_ratio_first10_mean','train_routed_to_total_cam144_ratio_last10_mean',
                'train_delta_ratio_mean_first10_mean','train_delta_ratio_mean_last10_mean','train_delta_ratio_max_max'):
        lines.append(f'| {key} | {avg(key):.8g} |')
    lines+=['','All formal losses, measured gradients, checkpoints and outputs passed finite-value checks. A larger residual is not automatically '
            'better discriminative evidence. Regularizer/total-gradient norm ratios are descriptive, can be affected by cancellation, and do not bound parameter drift. '
            'Per-fold size, gradient, beta, null, global-off and Pairwise→Unary intervention diagnostics are in mechanism.csv. '
            'The Unary replacement is a frozen distribution substitution, not a second formal training group.','',
            '## Audit, resources and limitations','',
            f"10/10 formal tasks, 10/10 independent best metrics, 10/10 latest/four optimizers, 10/10 exact initial identities and Base-off checks; "
            f"10/10 exact E81.4 capped checkpoint replays; common-epoch public RNG mismatches={status['rng_mismatch_count']} against DARG and E81.4. "
            'A fresh audit process repeated all preflight checks and serialized two-step recovery. Strict earliest maximal ACC selection is verified.', '',
            f"Frozen training source/config/history aggregate SHA256: `{status['training_source_sha256']}`; current audited-source SHA256: `{status['source_sha256']}`. Full dependency hashes, historical versions and source review are retained. "
            'Historical dependency hashes are checked again after the complete audit.', '',
            'Raw JSONL logs lack three scalar records (fold1 epoch75; fold2 epochs137/301); the storage cause is not established. '
            'The first independent audit stopped rather than waiving the check. Both affected folds were deterministically replayed from the original '
            'warmup solely for audit: all 756 retained scalar rows, both best/latest models, all four optimizers and RNG reproduced exactly. '
            'Three recovered records are stored in separate immutable recovery JSON files; original logs, tasks and checkpoints remain byte-identical. '
            'The original audit is snapshotted; only audit.py adds verified-record reconciliation and dual revision SHA checks. '
            'No formal training code, configuration, checkpoint or research threshold changed. A preliminary diagnostic replay comparator was corrected '
            'to compare CPU-normalized tensors after a CPU/CUDA metadata mismatch; no formal artifacts were changed.', '',
            'Inherited limitations: test=validation ACC checkpoint selection is biased rather than independent testing; pre-CV RFE/standardization; '
            'transductive patient graph; Fold2 patients377/407 in original train/test; overlapping CV training sets. Training-only new-residual blocking '
            'does not repair the original overlap or indirect graph effects. All eval patients receive the full model without label-dependent fallback. '
            'The16 latent tokens are not validated brain regions. No medical prompt was tested.', '',
            f"Research decision: **{status['research_assessment']['decision']}**. "
            'No further experiment is started automatically. Causal attribution is limited to this fixed residual-mapping contrast within the inherited protocol.', '',
            '## Requested answers','',
            f"Q1. Actual mean residual ratio={avg('delta_ratio_mean')*100:.4f}%, P50={avg('delta_ratio_p50')*100:.4f}%, P95={avg('delta_ratio_p95')*100:.4f}%; "
            f"fraction above10%={avg('above10_fraction')*100:.2f}%, overall maximum={max(r['delta_ratio_max'] for r in mechanisms)*100:.4f}%.", '',
            f"Q2. Versus capped E81.4: ACC {status['research_assessment']['vs_E814_difference_pp']['acc']:+.4f}pp, BA {status['research_assessment']['vs_E814_difference_pp']['hard_auc']:+.4f}pp.", '',
            f"Q3. Versus DARG: ACC {status['research_assessment']['vs_DARG_difference_pp']['acc']:+.4f}pp, BA {status['research_assessment']['vs_DARG_difference_pp']['hard_auc']:+.4f}pp; "
            f"both exceed baseline={status['research_assessment']['both_primary_metrics_exceed_DARG']}.", '',
            f"Q4. DARG→current-off net {total('current_off_vs_history_net'):+d}; residual contribution {total('residual_vs_current_off_net'):+d}; DARG→full net {total('final_vs_history_net'):+d}.", '',
            f"Q5. Without Fold2, DARG→full net {sum(r['final_vs_history_net'] for r in mechanisms if r['fold']!=2):+d}; "
            f"ACC/BA differences {status['research_assessment']['without_fold2_vs_DARG_pp']['acc']:+.4f}/{status['research_assessment']['without_fold2_vs_DARG_pp']['hard_auc']:+.4f}pp. "
            'No recorded nonfinite loss/gradient/output; large finite residuals and fold heterogeneity must still be considered.','']
    training=json.loads((RUNTIME/'training_summary.json').read_text())
    lines+=['## Resource record','',f"Single GPU; maximum{training['max_workers']} workers, one CPU thread each. Dispatch wall time {training['wall_seconds']:.2f}s; "
            f"failures {len(training['failures'])}; free GPU bytes at dispatch {training['free_gpu_bytes_at_dispatch']}; observed preflight peak {training['peak_preflight_bytes']} bytes.",'']
    return '\n'.join(lines)


if __name__=='__main__':
    try:
        status=audit();print(json.dumps({'status':status['status'],'research_assessment':status['research_assessment']},ensure_ascii=False),flush=True)
    except Exception as exc:
        ex.atomic_json(RUNTIME/'audit_failure.json',{'status':'FAIL','error':repr(exc),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        raise

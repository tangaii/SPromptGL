#!/usr/bin/env python3
"""Independent E81.6 checkpoint, scalar-journal and mechanism recomputation."""
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
    for name,right in [('G1_vs_DARG',BASE_GROUP),('G1_vs_G2',GROUPS[1]),('G1_vs_G3',GROUPS[2])]:
        for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
            for m in METRICS:
                diff=np.array([values[GROUPS[0]][f][m]-values[right][f][m] for f in folds])*100.
                mean,sd=float(diff.mean()),float(diff.std(ddof=1))
                half=float(student_t.ppf(.975,len(diff)-1))*sd/math.sqrt(len(diff))
                rows.append({'comparison':name,'left':GROUPS[0],'right':right,'scope':scope,'metric':m,
                             'mean_difference_pp':mean,'sd_difference_pp':sd,'ci95_low_pp':mean-half,'ci95_high_pp':mean+half,
                             'wins':int((diff>1e-10).sum()),'ties':int((abs(diff)<=1e-10).sum()),
                             'losses':int((diff < -1e-10).sum()),'folds':json.dumps(folds),
                             'fold_differences_pp':json.dumps(diff.tolist()),
                             'interval_interpretation':'descriptive; CV training memberships overlap'})
    return rows


def prompt_geometry():
    cache=ex.prompt_cache(); result={}
    for condition in ('medical','anonymous'):
        t=torch.nn.functional.normalize(cache[condition].double(),dim=-1)
        gram=t@t.T; sim=gram[~torch.eye(8,dtype=torch.bool)]
        result[condition]={'gram':gram.tolist(),'raw_l2_norms':cache[condition].norm(dim=-1).tolist(),
                           'offdiagonal_similarity_mean':float(sim.mean()),
                           'offdiagonal_similarity_sd':float(sim.std(unbiased=False)),
                           'offdiagonal_similarity_min':float(sim.min()),
                           'offdiagonal_similarity_max':float(sim.max())}
    result['gram_mae']=float((torch.tensor(result['medical']['gram'])-torch.tensor(result['anonymous']['gram'])).abs().mean())
    return result


def mechanism_row(group,fold,state,module,full,off,history,logs):
    ids=torch.as_tensor(state['test'],device=full['logp'].device,dtype=torch.long)
    labels=state['y'].detach().cpu().numpy()[np.asarray(state['test'])]
    fm,bm=independent_metrics(full['logp'],state),independent_metrics(off['logp'],state)
    d=full['fusion']; att=d['attention'][ids]; ratio=d['delta_ratio'][ids]; comp=d['competition'][ids]
    row={'group':group,'fold':fold,
         **{'history_'+m:history[m]*100. for m in METRICS},
         **{'current_off_'+m:bm[m]*100. for m in METRICS},
         **{'full_'+m:fm[m]*100. for m in METRICS},
         **{'current_off_vs_history_'+k:v for k,v in compare_predictions(history['pred'],bm['pred'],labels).items()},
         **{'residual_vs_current_off_'+k:v for k,v in compare_predictions(bm['pred'],fm['pred'],labels).items()},
         **{'final_vs_history_'+k:v for k,v in compare_predictions(history['pred'],fm['pred'],labels).items()},
         'delta_ratio_mean':float(ratio.mean()),'delta_ratio_p50':float(torch.quantile(ratio,.5)),
         'delta_ratio_p95':float(torch.quantile(ratio,.95)),'delta_ratio_max':float(ratio.max()),
         'near_bound_fraction':float((ratio>=.099).float().mean()),
         'near_zero_fraction':float((ratio<1e-4).float().mean()),
         'raw_delta_norm_mean':float(d['raw_delta'][ids].norm(dim=-1).mean()),
         'delta_norm_mean':float(d['delta'][ids].norm(dim=-1).mean()),
         'cam_norm_mean':float(d['z_cam'][ids].norm(dim=-1).mean()),
         'enhanced_norm_mean':float(d['z_enhanced'][ids].norm(dim=-1).mean()),
         'local_residual_norm_means':json.dumps(d['delta'][ids].reshape(-1,16,9).norm(dim=-1).mean(0).cpu().tolist()),
         'patient_delta_coordinate_sd_mean':float(d['delta'][ids].std(0,unbiased=False).mean()),
         'residual_hidden_mae':float((full['hidden'][ids]-off['hidden'][ids]).abs().mean()),
         'residual_probability_mae':float(np.mean(np.abs(fm['prob']-bm['prob']))),
         'gamma':float(d['gamma']),'gram_mse':float(d['geo']),
         'projected_gram':json.dumps(d['gram_after'].cpu().tolist()),
         'input_gram':json.dumps(d['gram_before'].cpu().tolist()),
         'normal_weight_mean':float(comp[...,0].mean()),'abnormal_weight_mean':float(comp[...,1].mean()),
         'competition_patient_sd_mean':float(comp.std(0,unbiased=False).mean()),
         'competition_entropy':float((-(comp*comp.clamp_min(1e-12).log()).sum(-1)/math.log(2)).mean()),
         'competition_single_channel_099_fraction':float((comp.max(-1).values>.99).float().mean()),
         'normal_message_norm':float(d['normal'][ids].norm(dim=-1).mean()),
         'abnormal_message_norm':float(d['abnormal'][ids].norm(dim=-1).mean()),
         'normal_abnormal_message_mae':float((d['normal'][ids]-d['abnormal'][ids]).abs().mean()),
         'null_weight':float(att[...,-1].mean()),
         'attention_entropy':float((-(att*att.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
         'prompt_attention_patient_sd_mean':float(att.std(0,unbiased=False).mean()),
         'prompt_message_cancellation_ratio':float((d['messages'][ids].mean(1).norm(dim=-1)/
                  d['messages'][ids].norm(dim=-1).mean(1).clamp_min(1e-12)).mean())}
    pairmae=torch.stack([torch.stack([(att[:,k]-att[:,j]).abs().mean() for j in range(8)]) for k in range(8)])
    row['prompt_attention_pairwise_mae']=json.dumps(pairmae.cpu().tolist())
    row['prompt_attention_offdiagonal_mae']=float(pairmae[~torch.eye(8,dtype=torch.bool,device=pairmae.device)].mean())
    row['prompt_computational_collapse']=bool(float(d['gamma'])<1e-3 or float(pairmae.max())<1e-6)
    byprompt=[]
    for k in range(8):
        ak=att[:,k]; avg=ak.mean(0); flat=avg[:,:15].reshape(-1)
        vals,ix=flat.topk(5)
        edges=[{'receiver':int(i)//15,'sender':int(module.neighbors[int(i)//15,int(i)%15]),'mean_weight':float(v)} for i,v in zip(ix,vals)]
        byprompt.append({'index':k,'text':ex.PROMPTS[k] if group!=GROUPS[2] else f'anonymous_{k}',
                         'entropy':float((-(ak*ak.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
                         'null_weight':float(ak[...,-1].mean()),
                         'null_dominant_fraction':float((ak.argmax(-1)==15).float().mean()),
                         'patient_attention_sd_mean':float(ak.std(0,unbiased=False).mean()),
                         'patient_template_mae':float((ak-avg).abs().mean()),'top_mean_relations':edges})
    row['per_prompt_statistics']=json.dumps(byprompt)
    for key in ('routed_cam144_norm','routed_to_total_cam144_ratio','delta144'):
        a=np.array([r['gradient_route'].get(key,0.) for r in logs])
        row['train_'+key+'_mean']=float(a.mean())
        row['train_'+key+'_first10_mean']=float(a[:10].mean())
        row['train_'+key+'_last10_mean']=float(a[-10:].mean())
    for key in logs[0]['gradient_norms']:
        a=np.array([r['gradient_norms'][key] for r in logs])
        row['train_'+key+'_gradient_mean']=float(a.mean()); row['train_'+key+'_gradient_max']=float(a.max())
    for key in ('res','geo','cls','historical_total','total'):
        a=np.array([r['losses'][key] for r in logs])
        row['train_'+key+'_mean']=float(a.mean()); row['train_'+key+'_max']=float(a.max())
    row['train_weighted_res_mean']=.001*row['train_res_mean']
    row['train_weighted_geo_mean']=.001*row['train_geo_mean']
    row['train_res_to_cls_ratio_mean']=float(np.mean([.001*r['losses']['res']/max(abs(r['losses']['cls']),1e-30) for r in logs]))
    # All interventions keep weights and selected checkpoint fixed. Labels only score.
    interventions=['null','global_off','prompt_off','anonymous','swap_groups']
    if group==GROUPS[2]: interventions.append('medical')
    if group==GROUPS[0]:
        interventions+=['unary']+[f'delete_prompt_{k}' for k in range(8)]+[f'swap_prompt_{k}' for k in range(8)]
    for name in interventions:
        with torch.no_grad(): probe=ex.forward_fusion(state,module,group,False,intervention=name)
        pm=independent_metrics(probe['logp'],state); pd=probe['fusion']
        stats=intervention_stats(fm,pm,labels)
        stats.update({'attention_mae':float((d['attention'][ids]-pd['attention'][ids]).abs().mean()),
                      'message_mae':float((d['messages'][ids]-pd['messages'][ids]).abs().mean()),
                      'weighted_relation_mae':float((d['weighted_relation'][ids]-pd['weighted_relation'][ids]).abs().mean()),
                      'delta_mae':float((d['delta'][ids]-pd['delta'][ids]).abs().mean())})
        row[name+'_intervention']=json.dumps(stats)
        if name=='null':
            assert torch.count_nonzero(pd['messages'])==torch.count_nonzero(pd['relation'])==torch.count_nonzero(pd['delta'])==0
            assert torch.equal(probe['logp'],off['logp']) and torch.equal(probe['hidden'],off['hidden'])
        if name=='global_off':
            assert torch.equal(d['messages'],pd['messages']) and torch.equal(d['d'],pd['d'])
        del probe
    return row


def scientific_decision(summary,mechanisms):
    g1=summary[GROUPS[0]]; base=summary[BASE_GROUP]
    def superior(right,scope='all10'):
        return all(g1[scope][m]['mean']>summary[right][scope][m]['mean']+1e-10 for m in ('acc','hard_auc'))
    baseline=superior(BASE_GROUP); pairwise=superior(GROUPS[1]) and superior(GROUPS[1],'without_fold2')
    collapse=any(r['prompt_computational_collapse'] for r in mechanisms if r['group']==GROUPS[0])
    medical=superior(GROUPS[2]) and superior(GROUPS[2],'without_fold2') and not collapse
    g1rows=[r for r in mechanisms if r['group']==GROUPS[0]]
    net=sum(r['final_vs_history_net'] for r in g1rows)
    net9=sum(r['final_vs_history_net'] for r in g1rows if r['fold']!=2)
    allgo=baseline and superior(BASE_GROUP,'without_fold2') and pairwise and medical and net>0 and net9>0
    # Mechanisms are reported quantitatively. Full GO is withheld for numerical
    # prompt collapse, missing RES effect or a collapsed competition channel.
    res_effect=any(json.loads(r['global_off_intervention'])['probability_mae']>1e-8 for r in g1rows)
    competition_ok=all(r['competition_single_channel_099_fraction']<.95 for r in g1rows)
    allgo=allgo and res_effect and competition_ok
    return {'complete_method':'BASELINE_GAIN_PRELIMINARY' if baseline else 'NO_GO_PRCL_RES_BASELINE',
            'pairwise':'PAIRWISE_INCREMENT_PRELIMINARY' if pairwise else 'PAIRWISE_INCREMENT_NOT_ESTABLISHED',
            'medical_input':'MEDICAL_INPUT_INCREMENT_PRELIMINARY' if medical else 'MEDICAL_PROMPT_INCREMENT_NOT_ESTABLISHED',
            'overall':'PRELIMINARY_GO_PRCL_RES' if allgo else 'NO_GO_PRCL_RES',
            'net_corrections_vs_DARG':net,'net_corrections_without_fold2':net9,
            'prompt_computational_collapse':collapse,'res_inference_effect':res_effect,
            'competition_not_collapsed':competition_ok}


def audit():
    started=time.monotonic()
    manifest=json.loads((HERE/'manifest.json').read_text()); frozen=json.loads((RESULTS/'preflight.json').read_text())
    sha=frozen['source_sha256']
    assert frozen['status']=='PASS' and ex.source_sha()==sha==manifest['source_sha256']
    assert ex.sha_file(RESULTS/'preflight.json')==manifest['preflight_sha256']
    replay=ex.preflight(persist=False)
    for f in range(10):
        for key in ('identity','initial_module_sha256','warmup_rng_sha','base_off_forward_hidden_historical_loss_gradient_update_rng_exact'):
            assert replay['folds'][f][key]==frozen['folds'][f][key],('independent parity',f,key)
    assert replay['module_tests']==frozen['module_tests']
    replay_record={'status':'PASS','folds_exactly_reproduced':10,'source_sha256':sha,
                   'serialized_resume':replay['folds'][0]['serialized_resume']}
    ex.atomic_json(RUNTIME/'independent_preflight_replay.json',replay_record)
    reference={int(r['fold']):metric_values(r) for r in read_csv(ex.runtime().EXP/'results/fold_metrics.csv') if r['group']==BASE_GROUP}
    assert set(reference)==set(range(1,11))
    values={g:{} for g in GROUPS}; values[BASE_GROUP]=reference
    rows=[]; mechanisms=[]; latest_checks=[]; best_checks=[]; rng_reports={}; tasks={}
    for fold in range(1,11):
        hist_task=json.loads((ex.runtime().EXP/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())
        hist_cp=ex.torch_load(ex.runtime().EXP/'.runtime/checkpoints'/f'{BASE_GROUP}_f{fold:02d}.best.pt')
        state,_,identity=ex.prepare(fold); assert identity==frozen['folds'][fold-1]['identity']
        ex.runtime().load_history().load_bundle(state,hist_cp['best_state'])
        with torch.no_grad(): h=ex._base_forward(state,False)
        history=independent_metrics(h['logp'],state); assert_metrics(reference[fold],history,'historical reload')
        assert np.array_equal(history['pred'],hist_cp['test_prediction'])
        assert np.allclose(history['prob'],hist_cp['test_probability'],atol=1e-7,rtol=0)
        del state,h,hist_cp
        attention_by_group={}
        for group in GROUPS:
            task_file,best_file,latest_file,log_file=ex.paths(group,fold)
            assert ex._task_complete(group,fold,sha),('incomplete task',group,fold)
            task=json.loads(task_file.read_text()); tasks[group,fold]=task
            best,latest=ex.torch_load(best_file),ex.torch_load(latest_file)
            for cp in (best,latest):
                assert cp['source_sha256']==sha and cp['identity']==identity
                assert cp['group']==group and cp['fold']==fold
            state,_,_=ex.prepare(fold); module=ex.make_module(fold,state['x'].device)
            initsha=ex.sha_value(ex._cpu_state(module))
            assert initsha==task['initial_module_sha256']==frozen['folds'][fold-1]['initial_module_sha256']
            assert sum(p.numel() for p in module.parameters())==42499
            optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
            ex.load_bundle(state,module,optimizer,latest['current']); ex._validate_optimizer_settings(state,optimizer)
            for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer): validate_optimizer(opt)
            assert {'python','numpy','cpu','cuda'}.issubset(latest['rng'])
            ex._rng_runtime().restore_rng(latest['rng'])
            assert ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())==ex._rng_runtime().rng_sha(latest['rng'])
            assert latest['next_epoch']==task['epochs_executed']
            logs=[json.loads(line) for line in log_file.read_text().splitlines()]
            assert len(logs)==task['epochs_executed']-10==task['scalar_rows']
            assert ex.sha_file(log_file)==task['scalar_log_sha256']
            assert logs==latest['epoch_logs'],('latest vs scalar journal',group,fold)
            assert [r['epoch'] for r in logs]==list(range(11,task['epochs_executed']+1))
            assert [r['rng'] for r in logs]==task['rng_trace']==latest['trace']
            best_log=max(logs,key=lambda r:r['metrics']['acc'])
            assert best_log['epoch']==task['best_epoch_1based']==best['best_epoch_1based']==latest['best_epoch']+1
            running=-1.; selected=-1; wait=0
            for line in logs:
                if line['metrics']['acc']>running: running=line['metrics']['acc']; selected=line['epoch']; wait=0
                else: wait+=1
                assert line['best_acc']==running and line['best_epoch_1based']==selected
                loss=line['losses']
                assert abs(loss['total']-(loss['historical_total']+.001*loss['res']+.001*loss['geo']))<2e-6
                assert all(np.isfinite(x) for x in line['gradient_norms'].values())
                assert line['diagnostics']['delta_ratio_max']<=.100001
            assert wait==latest['wait']
            if task['epochs_executed']<450: assert wait>200
            assert task['prompt_cache_sha256']==frozen['prompt_cache_sha256']
            latest_checks.append({'group':group,'fold':fold,'four_optimizers':'PASS','rng':'PASS',
                                  'scalar_journal_all_epochs':'PASS','strict_earliest_best':'PASS','rows':len(logs)})
            ex.load_bundle(state,module,optimizer,best['best']['bundle'])
            for opt in (state['opt_base'],state['opt_cam'],state['opt_added'],optimizer): validate_optimizer(opt)
            rb=ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
            with torch.no_grad():
                full=ex.forward_fusion(state,module,group,False)
                off=ex.forward_fusion(state,module,group,False,'off')
            assert torch.equal(off['fusion']['z_enhanced'],off['fusion']['z_cam'])
            assert torch.isfinite(full['logp']).all() and (full['fusion']['delta_ratio']<=.100001).all()
            assert torch.allclose(full['logp'].exp().sum(-1),torch.ones_like(full['logp'][:,0]),atol=2e-6)
            metrics=independent_metrics(full['logp'],state)
            assert_metrics(task['metrics'],metrics,'task'); assert_metrics(best['metrics'],metrics,'best')
            assert np.array_equal(best['best']['test_prediction'],metrics['pred'])
            assert np.allclose(best['best']['test_probability'],metrics['prob'],rtol=0,atol=1e-7)
            cls_fn=ex.base_runtime().runtime()['utils'].ClsLoss
            assert torch.equal(cls_fn(full['logp'],state['y'],state['train'],state['weights']),full['cls'])
            mech=mechanism_row(group,fold,state,module,full,off,history,logs)
            ids=torch.as_tensor(state['test'],device=full['logp'].device,dtype=torch.long)
            attention_by_group[group]=full['fusion']['attention'][ids].cpu()
            assert rb==ex._rng_runtime().rng_sha(ex._rng_runtime().pack_rng())
            saved_rng=ex._rng_runtime().pack_rng()
            hist=ex.runtime().load_history(); hist.zero_optimizers(state); optimizer.zero_grad(set_to_none=True)
            probe=ex.forward_fusion(state,module,group,True); probe['historical_total'].backward()
            mech['best_classification_gradient_norms']=json.dumps(ex.grad_snapshot(state,module))
            route=probe['route_gradients']
            mech['best_classification_cam_route_norm']=.1*route.get('z_rel_input',0.)
            mech['best_classification_cam_route_ratio']=.1*route.get('z_rel_input',0.)/max(route.get('cam144_total',0.),1e-30)
            if fold==2:
                assert torch.count_nonzero(probe['fusion']['delta'][[376,406]])==0
                legal=probe['fusion']['legal']; assert not bool(legal[376]) and not bool(legal[406])
                train_ids=torch.as_tensor(state['train'],device=legal.device,dtype=torch.long); legal_ids=train_ids[legal[train_ids]]
                assert torch.equal(probe['res'],probe['fusion']['delta_ratio'][legal_ids].square().mean())
            ex._rng_runtime().restore_rng(saved_rng)
            mechanisms.append(mech); values[group][fold]=metric_values(metrics)
            rows.append({'group':group,'fold':fold,**{m:metrics[m]*100. for m in METRICS},
                         **{k:metrics[k] for k in ('asd_tp','asd_fn','td_fp','td_tn')},
                         **{'current_off_'+m:mech['current_off_'+m] for m in METRICS},
                         'best_epoch_1based':task['best_epoch_1based'],'epochs_executed':task['epochs_executed'],
                         'wall_seconds':task['wall_seconds'],'peak_gpu_bytes':task['peak_gpu_bytes'],
                         'checkpoint_sha256':ex.sha_file(best_file),'latest_sha256':ex.sha_file(latest_file),
                         'independent_checkpoint_audit':'PASS'})
            best_checks.append({'group':group,'fold':fold,'six_metrics':'PASS','probability_replay':'PASS',
                                'Null_off_exact':'PASS','single_OutputLayer':'PASS','Fold2_train_block':'PASS'})
            rng_reports[f'{group}_f{fold:02d}_vs_DARG']=rng_compare(task['rng_trace'],hist_task['rng_trace'])
            del state,module,optimizer,full,off,probe,best,latest
            torch.cuda.empty_cache(); print(f'AUDIT {group} fold={fold} PASS',flush=True)
        for g in GROUPS[1:]:
            r=next(r for r in mechanisms if r['group']==GROUPS[0] and r['fold']==fold)
            r['trained_attention_mae_vs_'+g]=float((attention_by_group[GROUPS[0]]-attention_by_group[g]).abs().mean())
        for a in range(3):
            for b in range(a+1,3):
                ga,gb=GROUPS[a],GROUPS[b]
                assert tasks[ga,fold]['initial_module_sha256']==tasks[gb,fold]['initial_module_sha256']
                rng_reports[f'{ga}_vs_{gb}_f{fold:02d}']=rng_compare(tasks[ga,fold]['rng_trace'],tasks[gb,fold]['rng_trace'])
    rows.extend({'group':BASE_GROUP,'fold':f,**{m:reference[f][m]*100. for m in METRICS},
                 'independent_checkpoint_audit':'HISTORICAL_BEST_RECOMPUTED'} for f in range(1,11))
    summary=summary_values(values); pairs=paired_rows(values); geometry=prompt_geometry()
    decision=scientific_decision(summary,mechanisms); mismatches=sum(r['mismatches'] for r in rng_reports.values())
    status={'status':'PASS' if mismatches==0 else 'FAIL','source_sha256':sha,
            'formal_tasks_complete':30,'best_checkpoint_independent_pass':len(best_checks),
            'latest_state_integrity_pass':len(latest_checks),'historical_best_checkpoint_independent_pass':10,
            'base_off_parity_pass':10,'private_same_fold_three_group_initialization_pass':10,
            'trainable_parameter_count':42499,'prompt_cache_sha256':frozen['prompt_cache_sha256'],
            'independent_preflight_replay':replay_record,'rng_mismatch_count':mismatches,'rng_audit':rng_reports,
            'best_checks':best_checks,'latest_checks':latest_checks,'summary':summary,
            'prompt_input_geometry':geometry,'research_decision':decision,'fold2_overlap_1based':[377,407],
            'metrics_unit':'percent; population SD','hard_auc_definition':'balanced accuracy of hard predictions',
            'protocol':'test ACC checkpoint selection retained; not unbiased independent test evaluation',
            'scalar_logs_missing_epochs':0,'audit_seconds':time.monotonic()-started}
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
        add('| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |')
        add('|---|---:|---:|---:|---:|---:|---:|')
        for group in (*GROUPS,BASE_GROUP):
            add('| '+group+' | '+' | '.join(f"{s[group][scope][m]['mean']:.4f} ± {s[group][scope][m]['sd']:.4f}" for m in METRICS)+' |')
    add('# Experiment 81.6 — PRCL-RES')
    add(); add(f"Execution: COMPLETE_AUDITED; 30/30 formal tasks, 30/30 best and latest checks; public RNG mismatches {status['rng_mismatch_count']}; missing scalar epochs {status['scalar_logs_missing_epochs']}.")
    add(f"Research: `{decision['overall']}`; `{decision['complete_method']}`; `{decision['pairwise']}`; `{decision['medical_input']}`.")
    add(); add('Metrics are ten-fold mean ± population SD, %. ASD=0, TD=1; Hard-AUC is hard-prediction balanced accuracy; Prob-AUC is TD-probability ROC-AUC. Test ACC selects checkpoints under the retained historical protocol.')
    add(); table('all10'); add(); add('## Excluding Fold 2'); add(); table('without_fold2')
    add(); add('## Paired comparisons'); add()
    add('| Comparison | Scope | ACC Δ pp [descriptive 95% interval] | W/T/L | Hard-AUC Δ pp | Prob-AUC Δ pp |')
    add('|---|---|---:|---|---:|---:|')
    for name in ('G1_vs_DARG','G1_vs_G2','G1_vs_G3'):
        for scope in ('all10','without_fold2'):
            d={r['metric']:r for r in pairs if r['comparison']==name and r['scope']==scope}; a=d['acc']
            add(f"| {name} | {scope} | {a['mean_difference_pp']:+.4f} [{a['ci95_low_pp']:+.4f}, {a['ci95_high_pp']:+.4f}] | {a['wins']}/{a['ties']}/{a['losses']} | {d['hard_auc']['mean_difference_pp']:+.4f} | {d['prob_auc']['mean_difference_pp']:+.4f} |")
    add(); add('All six per-fold differences are in paired_comparisons.csv. Intervals are descriptive across folds with overlapping training populations, not independent-patient confidence inference.')
    add(); add('## Backbone and residual decomposition'); add()
    add('| Group/scope | DARG → current off corrected/harmed/net | Current off → full corrected/harmed/net | DARG → full corrected/harmed/net |')
    add('|---|---|---|---|')
    for group in GROUPS:
        for scope in ('all10','without_fold2'):
            records=[r for r in mechanisms if r['group']==group and (scope=='all10' or r['fold']!=2)]
            def counts(prefix): return '/'.join(str(sum(r[prefix+k] for r in records)) for k in ('corrected','harmed','net'))
            add(f"| {group}/{scope} | {counts('current_off_vs_history_')} | {counts('residual_vs_current_off_')} | {counts('final_vs_history_')} |")
    add(); add('Residual-off performance is evaluated at checkpoints selected using the full model. It is a frozen counterfactual with changed classifier inputs, not an independently trained or optimally selected DARG baseline.')
    add(); add('| Current residual-off path | ACC | Hard-AUC | Prob-AUC | SEN | SPE | ASD-F1 |')
    add('|---|---:|---:|---:|---:|---:|---:|')
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]
        add('| '+group+' | '+' | '.join(f"{np.mean([r['current_off_'+m] for r in rs]):.4f} ± {np.std([r['current_off_'+m] for r in rs]):.4f}" for m in METRICS)+' |')
    add(); add('## Prompt, competition and residual diagnostics'); add()
    add('| Group | Gamma | Prompt attention pair MAE | Normal weight | Competition entropy | Residual mean/P50/P95/max % | Near 9.9% cap |')
    add('|---|---:|---:|---:|---:|---|---:|')
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]; mean=lambda k:float(np.mean([r[k] for r in rs]))
        residual='/'.join(f'{100*mean(k):.4f}' for k in ('delta_ratio_mean','delta_ratio_p50','delta_ratio_p95'))+f"/{100*max(r['delta_ratio_max'] for r in rs):.4f}"
        add(f"| {group} | {mean('gamma'):.6f} | {mean('prompt_attention_offdiagonal_mae'):.7f} | {mean('normal_weight_mean'):.6f} | {mean('competition_entropy'):.6f} | {residual} | {100*mean('near_bound_fraction'):.2f}% |")
    add(); add('Residual means are fold patient averages; P50/P95 columns average the fold quantiles, and max is the largest patient value. Per-patient raw norms, token statistics and all gradient summaries are reduced to scalars in mechanism.csv.')
    g1=[r for r in mechanisms if r['group']==GROUPS[0]]
    add(); add('| Frozen G1 intervention | Probability MAE | Prediction flips | ACC Δ pp | BA Δ pp | Corrected/harmed relative to full | Attention MAE |')
    add('|---|---:|---:|---:|---:|---|---:|')
    for name in ['null','global_off','prompt_off','anonymous','swap_groups','unary']+[f'delete_prompt_{k}' for k in range(8)]+[f'swap_prompt_{k}' for k in range(8)]:
        rs=[json.loads(r[name+'_intervention']) for r in g1]
        avg=lambda k:float(np.mean([r[k] for r in rs]))
        add(f"| {name} | {avg('probability_mae'):.8f} | {sum(r['flips'] for r in rs)} | {avg('acc_change_pp'):+.4f} | {avg('hard_auc_change_pp'):+.4f} | {sum(r['corrected'] for r in rs)}/{sum(r['harmed'] for r in rs)} | {avg('attention_mae'):.8f} |")
    add(); add('These use frozen weights without training or reselection. Global-off substitutes uniform 1/9 coordinate matching while keeping relation messages and d identical. Text replacement also changes input geometry; inference sensitivity alone does not establish correct medical semantics.')
    add(); add('| G1 prompt | Attention entropy | Null weight | Patient attention SD |')
    add('|---|---:|---:|---:|')
    promptstats=[json.loads(r['per_prompt_statistics']) for r in g1]
    for k,text in enumerate(ex.PROMPTS):
        add(f"| {k+1}. {text} | {np.mean([r[k]['entropy'] for r in promptstats]):.6f} | {np.mean([r[k]['null_weight'] for r in promptstats]):.6f} | {np.mean([r[k]['patient_attention_sd_mean'] for r in promptstats]):.7f} |")
    add(); add('Top latent relation positions and the complete per-prompt difference matrix are retained in mechanism.csv. These latent positions have no validated anatomical interpretation. Normal/abnormal text groups are not patient truth labels.')
    add(); add('## Fixed input geometry'); add()
    geometry=status['prompt_input_geometry']
    for condition in ('medical','anonymous'):
        g=geometry[condition]
        add(f"{condition}: off-diagonal cosine mean {g['offdiagonal_similarity_mean']:.6f}, SD {g['offdiagonal_similarity_sd']:.6f}, range [{g['offdiagonal_similarity_min']:.6f}, {g['offdiagonal_similarity_max']:.6f}].")
        add(); add('| | '+' | '.join(str(k+1) for k in range(8))+' |'); add('|---|'+'---:|'*8)
        for k,row in enumerate(g['gram']): add('| '+str(k+1)+' | '+' | '.join(f'{v:.5f}' for v in row)+' |')
        add()
    add('The anonymous Gaussian vectors match raw per-item L2 norm, not the medical Gram matrix. Their different geometry, a single anonymous draw and lack of per-prompt patient annotations limit semantic attribution.')
    add(); add('## Training, numerical stability and audit'); add()
    for group in GROUPS:
        rs=[r for r in mechanisms if r['group']==group]
        add(f"{group}: mean routed/new-to-total CAM gradient ratio {np.mean([r['train_routed_to_total_cam144_ratio_mean'] for r in rs]):.6g}; weighted residual regularizer mean {np.mean([r['train_weighted_res_mean'] for r in rs]):.6g}; mean residual-reg/classification ratio {np.mean([r['train_res_to_cls_ratio_mean'] for r in rs]):.6g}; classification-only trained text gradient norms are in mechanism.csv.")
    scheduler=RUNTIME/'training_summary.json'
    if scheduler.exists():
        tr=json.loads(scheduler.read_text()); add(f"Scheduler: {tr['completed_tasks']}/{tr['expected_tasks']} tasks, wall {tr['wall_seconds']:.1f}s, maximum {tr['max_workers']} workers; failures {len(tr['failures'])}. One measured worker is completed before increasing concurrency. Original three optimizer steps precede the PRCL optimizer.")
    add('Thirty independent best reloads recompute six metrics and saved predictions; all latest states retain four optimizer moments, full RNG, next epoch, early stopping and complete scalar journals. Ten-fold post-training zero-residual replay and serialized two-step fresh-model recovery pass. No historical source/result mutation is accepted; source_map is checked again at audit end.')
    add(); add('## Direct answers'); add()
    add(f"Q1: Prompt participation is measured by independent Query/message differences and frozen prompt interventions above; numerical prompt collapse flag={decision['prompt_computational_collapse']}. Computational participation does not itself imply diagnostic value.")
    add(f"Q2: Pairwise vs Unary: {decision['pairwise']}; use the independent G1−G2 ten-/nine-fold comparisons.")
    add(f"Q3: Actual medical input: {decision['medical_input']}; G1−G3 and input geometry both matter. This experiment cannot alone establish clinical semantic understanding.")
    add(f"Q4: RES frozen inference effect={decision['res_inference_effect']}; its probability/accuracy changes are in the global-off table. This is not an independently retrained RES ablation.")
    add(f"Q5: Joint ACC and BA baseline criterion: {decision['complete_method']}; nine-fold results are explicitly shown above.")
    add(f"Q6: Full G1 relative to historical DARG: net {decision['net_corrections_vs_DARG']:+d}, excluding Fold 2 {decision['net_corrections_without_fold2']:+d}; distinguish this from correction of current residual-off errors.")
    add(); add('## Provenance and limitations'); add()
    add('[SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) and [official code](https://github.com/wanxixi11/SPromptGL) were reviewed before implementation. KD-Brain borrows only semantic Query conditioning from [Attention_AggregateLayer](https://github.com/LiuSy0731/KDBrain/blob/main/KDBrain/models/model.py); ItemizedCLIP borrows independent text conditioning from its [model code](https://github.com/MLNeurosurg/ItemizedCLIP/blob/main/Radiology_Tasks/src/open_mri/my_model.py), without patient-item supervision losses. The requested CVF page returned 403; the official repository-linked [ItemizedCLIP arXiv paper](https://arxiv.org/abs/2512.11141) supplied the primary paper fallback. Exact source identities and differences are in runtime/source_review.json and reviewed_sources.json.')
    add('Historical E81.4-G2 is context, not a matched no-Prompt ablation. E81.5 unbounded fusion was NO_GO and is not reused. This study keeps seed0, test=val checkpoint selection, pre-CV preprocessing/RFE, transductive patient graphs and the Fold2 overlap. Blocking only new training residuals does not repair the original leakage or indirect graph effects. Single-seed overlapping CV is preliminary and does not establish independent medical generalization or a MICCAI-level validated innovation.')
    add(f"Frozen source/config/dependency SHA: `{status['source_sha256']}`. Prompt cache SHA: `{status['prompt_cache_sha256']}`. The manifest retains every file identity; no post-result hyperparameter adjustment is permitted.")
    body='\n'.join(lines)+'\n'; target=RESULTS/'report.md'; temp=target.with_suffix('.tmp')
    temp.write_text(body); temp.replace(target); assert target.read_text()==body


def main():
    result=audit(); print(json.dumps({'status':result['status'],'research':result['research_decision']},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

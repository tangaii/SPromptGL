#!/usr/bin/env python3
"""Independent E79 checkpoint, pairing, memory and correction analysis."""
from __future__ import annotations

import csv
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
REPO = Path('/data/adodas/SPromptGL-main')
BASE_EXP = REPO / 'experiment/experiment74_4_soft_positive_dcr'
RESULTS = EXP / 'results'; CKPTS = EXP / '.runtime/checkpoints'
import sys
sys.path.insert(0, str(EXP))
from run import (GROUPS, BASE_GROUP, FOLD2_EXCLUDE, prepare_fold, hook_forward,
                 source_snapshot, torch_load, write_csv, atomic_json)
from deml import DEMLMemoryAttention, output_layer_logp
from memory_bank import ErrorMemoryBank, TYPE_ASD_TO_TD, TYPE_TD_TO_ASD
from audit import independent_replay, _sites

METRICS = ('acc','hard_auc','prob_auc','sen','spe','ba','asd_f1')


def pct(x): return 100.0 * float(x)


def summary(vals):
    a = np.asarray(vals, dtype=float)
    return {'mean': float(a.mean()), 'sd': float(a.std(ddof=0)),
            'mean_percent': pct(a.mean()), 'sd_percent': pct(a.std(ddof=0))}


def ci(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean() - 1.96*a.std(ddof=0)/math.sqrt(len(a))), float(a.mean() + 1.96*a.std(ddof=0)/math.sqrt(len(a)))


def historical_base():
    rows=[]
    with (BASE_EXP/'results/fold_metrics.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('group') == BASE_GROUP:
                rows.append({k: float(row[k]) for k in METRICS} | {'fold': int(row['fold'])})
    if len(rows)!=10: raise RuntimeError('E74.4-G2 historical fold inventory is incomplete')
    return {int(r['fold']):r for r in rows}


def replay_checkpoint(group, fold, interventions=None):
    interventions = interventions or {}
    ckpt = torch_load(CKPTS/f'{group}_f{fold:02d}.best.pt')
    best=ckpt['best']; state, warm, identity, task, hist_ckpt=prepare_fold(fold)
    hist=__import__('path_adapter').legacy.load_base().load_history(); hist.load_bundle(state,best['base_state'])
    module=DEMLMemoryAttention().to(state['y'].device); module.load_state_dict(best['module']); module.eval()
    bank=ErrorMemoryBank.from_snapshot(best['memory_used'])
    sites=_sites(state['y'].device); test_ids=[int(x) for x in best['test_ids']]
    with torch.no_grad():
        out,cam=hook_forward(state,training=False)
        ids=torch.as_tensor(test_ids,device=state['y'].device)
        enhanced,diag=module(test_ids,cam[ids].detach(),out['logp'][ids].detach(),bank,sites,
                             use_direction=(group==GROUPS[1]),**interventions)
        logp,_=output_layer_logp(state['cam'].Outputlayer,enhanced)
    full=out['logp'].detach().clone(); full[ids]=logp
    labels=state['y'].detach();
    # Keep only the test rows for all downstream patient-level analyses.
    base=out['logp'][ids].detach(); method=logp.detach()
    prob_base=base.exp().cpu().numpy(); prob_method=method.exp().cpu().numpy()
    y=labels[ids].cpu().numpy(); bp=prob_base.argmax(1); mp=prob_method.argmax(1)
    base_acc=float((bp==y).mean()); method_acc=float((mp==y).mean())
    base_margin=(base[:,0]-base[:,1]).cpu().numpy(); method_margin=(method[:,0]-method[:,1]).cpu().numpy()
    true_margin=np.where(y==0,method[:,0].cpu().numpy()-method[:,1].cpu().numpy(),method[:,1].cpu().numpy()-method[:,0].cpu().numpy())
    base_true_margin=np.where(y==0,base[:,0].cpu().numpy()-base[:,1].cpu().numpy(),base[:,1].cpu().numpy()-base[:,0].cpu().numpy())
    base_correct=bp==y; method_correct=mp==y
    delta=diag['delta_h'].detach().cpu(); h=cam[ids].detach().cpu()
    # “direction correct” is a positive true-vs-wrong log-probability change on
    # original errors; this is the task-level effect of the feature direction.
    margin_gain=true_margin-base_true_margin
    original_errors=~base_correct
    result={
        'group':group,'fold':fold,'best_epoch':int(best['best_epoch_1based']),
        'best_metrics':best['metrics'],'base_metrics_at_method_checkpoint':{k:float(v) for k,v in best.get('base_metrics',{}).items()},
        'base_acc_replay':base_acc,'method_acc_replay':method_acc,
        'prob_auc':float(best['metrics']['prob_auc']),
        'mean_abs_prob_delta':float(np.abs(prob_method-prob_base).mean()),
        'base_prob':prob_base,'method_prob':prob_method,'y':y,'base_pred':bp,'method_pred':mp,
        'ids':np.asarray(test_ids,dtype=int),'base_margin':base_margin,'method_margin':method_margin,
        'margin_gain':margin_gain,'base_correct':base_correct,'method_correct':method_correct,
        'delta':delta.numpy(),'h':h.numpy(),'diag':diag,'bank':bank,
        'transition':{
            'error_to_correct':int(((~base_correct)&method_correct).sum()),
            'correct_to_error':int((base_correct&(~method_correct)).sum()),
            'net_correction':int(((~base_correct)&method_correct).sum()-(base_correct&(~method_correct)).sum()),
            'hard_prediction_changes':int((bp!=mp).sum()),
            'asd_error_corrected':int(((~base_correct)&method_correct&(y==0)).sum()),
            'td_error_corrected':int(((~base_correct)&method_correct&(y==1)).sum()),
            'correct_retention':float((base_correct&method_correct).sum()/max(int(base_correct.sum()),1)),
            'original_error_count':int(original_errors.sum()),
            'margin_gain_on_original_errors':float(margin_gain[original_errors].mean()) if original_errors.any() else 0.0,
            'direction_correct_on_original_errors':float((margin_gain[original_errors]>0).mean()) if original_errors.any() else 0.0,
            'direction_correct_on_original_correct':float((margin_gain[base_correct]>0).mean()) if base_correct.any() else 0.0,
            'mean_abs_delta_ratio':float((delta.norm(dim=1)/(h.norm(dim=1)+1e-12)).mean()),
            'mean_delta_norm':float(delta.norm(dim=1).mean()),
            'delta_cap_fraction':float(((delta.norm(dim=1)/(h.norm(dim=1)+1e-12))>=0.099999).float().mean()),
        },
        'best':best,
    }
    return result


def intervention_rows(group, fold, replay):
    rows=[]
    interventions=[('full',{}),('Memory-Off',None),('Error-Type-Swap',{'error_type_swap':True}),
                   ('Memory-Feature-Permuted',{'feature_permuted':True}),('Recency-Off',{'recency_off':True}),
                   ('Correction-Direction-Off',{'use_direction':False}),('Correction-Direction-Permuted',{'direction_permuted':True})]
    # Replay the base checkpoint once per intervention to keep each intervention
    # a same-Checkpoint read-only operation.
    for name,kwargs in interventions:
        if name=='full':
            r=replay
        elif name=='Memory-Off':
            # A fresh replay with an empty bank is implemented by replacing the
            # saved snapshot in a shallow checkpoint-independent path below.
            r=replay_checkpoint_with_bank(group,fold,ErrorMemoryBank(),{})
        else:
            r=replay_checkpoint_with_bank(group,fold,None,kwargs)
        tr=r['transition']; rows.append({'group':group,'fold':fold,'intervention':name,
            'acc':r['method_acc_replay'],'error_to_correct':tr['error_to_correct'],
            'correct_to_error':tr['correct_to_error'],'net_correction':tr['net_correction'],
            'hard_prediction_changes':tr['hard_prediction_changes'],
            'prob_auc':r.get('prob_auc'),
            'mean_abs_prob_delta':r.get('mean_abs_prob_delta'),
            'mean_abs_delta_ratio':tr['mean_abs_delta_ratio'],
            'mean_delta_norm':r.get('mean_delta_norm')})
    return rows


def replay_checkpoint_with_bank(group,fold,bank_override,kwargs):
    ckpt=torch_load(CKPTS/f'{group}_f{fold:02d}.best.pt'); best=ckpt['best']
    state,*_=prepare_fold(fold); hist=__import__('path_adapter').legacy.load_base().load_history(); hist.load_bundle(state,best['base_state'])
    module=DEMLMemoryAttention().to(state['y'].device); module.load_state_dict(best['module']); module.eval()
    bank=ErrorMemoryBank.from_snapshot(best['memory_used']) if bank_override is None else bank_override
    sites=_sites(state['y'].device); ids_list=[int(x) for x in best['test_ids']]
    with torch.no_grad():
        out,cam=hook_forward(state,False); ids=torch.as_tensor(ids_list,device=state['y'].device)
        use_direction=(group==GROUPS[1]) if 'use_direction' not in kwargs else bool(kwargs.pop('use_direction'))
        hnew,diag=module(ids_list,cam[ids].detach(),out['logp'][ids].detach(),bank,sites,use_direction=use_direction,**kwargs)
        lp,_=output_layer_logp(state['cam'].Outputlayer,hnew)
    base=out['logp'][ids].detach(); y=state['y'][ids].detach().cpu().numpy(); bp=base.exp().cpu().numpy().argmax(1); mp=lp.exp().cpu().numpy().argmax(1)
    delta=diag['delta_h'].detach().cpu(); h=cam[ids].detach().cpu(); bc=bp==y; mc=mp==y
    ratio=float((delta.norm(dim=1)/(h.norm(dim=1)+1e-12)).mean())
    full=out['logp'].detach().clone(); full[ids]=lp
    metrics=__import__('run').legacy.extended_metrics(full,ids_list,state['y'])
    mean_prob_delta=float((lp.exp()-base.exp()).abs().mean().item())
    mean_delta_norm=float(delta.norm(dim=1).mean().item())
    return {'method_acc_replay':float(mc.mean()),'prob_auc':float(metrics['prob_auc']),
        'mean_abs_prob_delta':mean_prob_delta,'mean_delta_norm':mean_delta_norm,'transition':{
        'error_to_correct':int(((~bc)&mc).sum()),'correct_to_error':int((bc&(~mc)).sum()),
        'net_correction':int(((~bc)&mc).sum()-(bc&(~mc)).sum()),'hard_prediction_changes':int((bp!=mp).sum()),
        'mean_abs_delta_ratio':ratio},
        'delta':delta.numpy(),'h':h.numpy(),'group':group,'fold':fold,
        'transition_extra':{},'method_pred':mp}


def paired(rows_a, rows_b, key):
    vals=[]
    for fold in range(1,11): vals.append(float(rows_a[fold][key])-float(rows_b[fold][key]))
    a=np.asarray(vals); lo,hi=ci(a)
    return {'metric':key,'mean_diff':float(a.mean()),'mean_diff_percent':pct(a.mean()),
            'sd_diff_percent':pct(a.std(ddof=0)),'ci95_low_percent':pct(lo),'ci95_high_percent':pct(hi),
            'W':int((a>0).sum()),'T':int((a==0).sum()),'L':int((a<0).sum()),
            'per_fold':vals,'exclude_fold2_mean_percent':pct(a[1:].mean()),
            'exclude_fold2_sd_percent':pct(a[1:].std(ddof=0))}


def main():
    base=historical_base(); all_results={g:{} for g in GROUPS}; audit=[]
    for g in GROUPS:
        for f in range(1,11):
            audit.append(independent_replay(g,f)); all_results[g][f]=replay_checkpoint(g,f)
    # Historical and method fold metrics.
    rows=[]; fold_metric_rows=[]
    for g in GROUPS:
        for f,r in all_results[g].items():
            for k in METRICS:
                val=float(r['best']['metrics'][k]); fold_metric_rows.append({'group':g,'fold':f,'metric':k,'value':val})
            for k in METRICS:
                rows.append({'method':g,'fold':f,**{k:float(r['best']['metrics'][k]) for k in METRICS},
                             'best_epoch':r['best']['best_epoch_1based']})
    for f in range(1,11): rows.append({'method':BASE_GROUP,'fold':f,**{k:base[f][k] for k in METRICS},'best_epoch':None})
    write_csv(RESULTS/'fold_metrics.csv',rows)
    summary_rows=[]
    for method, mapping in [(BASE_GROUP,base),*[(g,{f:{k:float(all_results[g][f]['best']['metrics'][k]) for k in METRICS} for f in range(1,11)}) for g in GROUPS]]:
        vals=[mapping[f] for f in range(1,11)]
        summary_rows.append({'method':method,**{f'{k}_mean_percent':pct(np.mean([v[k] for v in vals])) for k in METRICS},
                             **{f'{k}_sd_percent':pct(np.std([v[k] for v in vals],ddof=0)) for k in METRICS}})
    write_csv(RESULTS/'summary.csv',summary_rows)
    # Pairings A/B/C.
    pair_rows=[]
    method_maps={g:{f:{k:float(all_results[g][f]['best']['metrics'][k]) for k in METRICS} for f in range(1,11)} for g in GROUPS}
    for label,left,right in [('G1-minus-base',GROUPS[0],BASE_GROUP),('G2-minus-G1',GROUPS[1],GROUPS[0]),('G2-minus-base',GROUPS[1],BASE_GROUP)]:
        lm=method_maps[left] if left in method_maps else base; rm=method_maps[right] if right in method_maps else base
        for k in METRICS:
            p=paired(lm,rm,k); p.update({'comparison':label}); pair_rows.append(p)
    write_csv(RESULTS/'paired_comparisons.csv',pair_rows)
    # Patient transitions and mechanism diagnostics.
    trans_rows=[]; mechanism_rows=[]; inter_rows=[]; memory_rows=[]
    for g in GROUPS:
        for f,r in all_results[g].items():
            t=r['transition']; trans_rows.append({'group':g,'fold':f,**t,'base_acc':r['base_acc_replay'],'method_acc':r['method_acc_replay']})
            b=r['bank']; best=r['best']
            latest_data=torch_load(EXP/'.runtime/checkpoints'/f'shared_f{f:02d}.latest.pt')
            tr=latest_data.get('event_trace', [])
            max_total=max((x['size_asd_to_td']+x['size_td_to_asd'] for x in tr),default=0)
            first_full=next((x['epoch'] for x in tr if x['size_asd_to_td']>=64 and x['size_td_to_asd']>=64),None)
            memory_rows.append({'group':g,'fold':f,'best_epoch':r['best_epoch'],
                'final_asd_to_td':b.size(TYPE_ASD_TO_TD),'final_td_to_asd':b.size(TYPE_TD_TO_ASD),
                'max_total':max_total,'first_both_full_epoch':first_full,
                'writes_total':sum(x['writes'] for x in tr),'updates_total':sum(x['updates'] for x in tr),
                'evictions_total':sum(x['evictions'] for x in tr),'correct_streak_removals_total':sum(x['correct_streak_removals'] for x in tr),
                'base_errors_epoch_sum':sum(x['base_error_count'] for x in tr),
                'final_event_hash':tr[-1]['event_hash'] if tr else None,
                'direction_channel': 'masked' if g==GROUPS[0] else 'enabled'})
            mechanism_rows.append({'group':g,'fold':f,**t})
            inter_rows.extend(intervention_rows(g,f,r))
    write_csv(RESULTS/'correction_analysis.csv',trans_rows)
    write_csv(RESULTS/'memory_analysis.csv',memory_rows)
    write_csv(RESULTS/'mechanism.csv',mechanism_rows)
    write_csv(RESULTS/'interventions.csv',inter_rows)
    # Independent latest restoration check: all required tensors and memory
    # snapshots can be loaded without touching training labels.
    latest=[]
    for f in range(1,11):
        p=EXP/'.runtime/checkpoints'/f'shared_f{f:02d}.latest.pt'; d=torch_load(p)
        latest.append({'fold':f,'next_epoch':int(d['next_epoch']),'memory_total':ErrorMemoryBank.from_snapshot(d['memory_next']).size(),
                       'rng_present':isinstance(d.get('rng'),dict),'optimizer_groups':len(d['optimizers'])})
    # Identify the actual source hash used in training and the current audit
    # source hash separately; audit.py was allowed to be repaired after weight
    # production without altering any model parameter.
    current_source=source_snapshot();
    report={
        'status':'COMPLETE_AUDITED','group_count':2,'folds':10,'best_checkpoint_audits':audit,
        'latest_restore':latest,'source_sha256_current':hashlib.sha256(json.dumps(current_source,sort_keys=True).encode()).hexdigest(),
        'historical_base':base,'summary':summary_rows,
        'transition_totals':trans_rows,'memory_rows':memory_rows,
        'scientific_note':'This is a single-seed, historical-protocol experiment; no test label enters DEML memory.',
    }
    atomic_json(RESULTS/'analysis.json',report)
    print(json.dumps({'status':'COMPLETE_AUDITED','audited':len(audit),'summary':summary_rows},ensure_ascii=False))


if __name__=='__main__': main()

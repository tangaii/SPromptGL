#!/usr/bin/env python3
"""Independent DREM metrics, pairing, recovery and intervention analysis."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
import torch

EXP=Path(__file__).resolve().parent; REPO=Path('/data/adodas/SPromptGL-main')
BASE_EXP=REPO/'experiment/experiment74_4_soft_positive_dcr'; CKPTS=EXP/'.runtime/checkpoints'; RESULTS=EXP/'results'
sys.path.insert(0,str(EXP))
from path_adapter import legacy
from drem import DREM, output_layer_logp
from memory_bank import DREMErrorMemory, TYPES, BUCKETS
from run import GROUPS, BASE_GROUP, FOLD2_EXCLUDE, prepare_fold, hook_forward, torch_load, write_csv, atomic_json, source_snapshot, site_tensor

METRICS=('acc','hard_auc','prob_auc','sen','spe','ba','asd_f1')
PCT_METRICS=('acc','hard_auc','prob_auc','sen','spe','asd_f1')

def pct(x): return 100.*float(x)
def ci95(a):
    a=np.asarray(a,dtype=float); m=float(a.mean()); se=float(a.std(ddof=0))/math.sqrt(max(len(a),1)); return m-1.96*se,m+1.96*se
def _rows_csv(path,rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def historical_base():
    out={}
    with (BASE_EXP/'results/fold_metrics.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('group')==BASE_GROUP: out[int(r['fold'])]={k:float(r[k]) for k in METRICS}|{'best_epoch':int(r['best_epoch_1based'])}
    if len(out)!=10: raise RuntimeError('historical base fold inventory incomplete')
    return out
def transition(base,method,y):
    bp=base.exp().argmax(1).detach().cpu().numpy(); mp=method.exp().argmax(1).detach().cpu().numpy(); yy=y.detach().cpu().numpy(); bc=bp==yy; mc=mp==yy
    return {'base_errors':int((~bc).sum()),'method_errors':int((~mc).sum()),'error_to_correct':int(((~bc)&mc).sum()),'correct_to_error':int((bc&~mc).sum()),'net_correction':int(((~bc)&mc).sum()-((bc&~mc).sum())),'hard_prediction_changes':int((bp!=mp).sum()),'asd_error_corrected':int(((~bc)&mc&(yy==0)).sum()),'td_error_corrected':int(((~bc)&mc&(yy==1)).sum()),'correct_retention':float((bc&mc).sum()/max(int(bc.sum()),1)),'base_wrong_direction_fraction':float((~bc).mean())}
def replay(group,fold,ledger,intervention='Full'):
    ck=torch_load(CKPTS/f'{group}_f{fold:02d}.{ledger}.pt'); best=ck['best']; state,*_=prepare_fold(fold); hist=legacy.load_base().load_history(); hist.load_bundle(state,best['base_state']); mod=DREM().to(state['y'].device); mod.load_state_dict(best['module']); mod.eval(); bank=DREMErrorMemory.from_snapshot(best['memory_used']); ids_list=[int(x) for x in best['test_ids']]; ids=torch.as_tensor(ids_list,device=state['y'].device)
    flags={'use_recovery':group==GROUPS[1]}
    if intervention=='Memory-Off': bank=DREMErrorMemory()
    if intervention=='Key-Value-Association-Break': flags['association_break']=True
    if intervention=='Error-Type-Swap': flags['type_swap']=True
    if intervention=='Recency-Off': flags['recency_off']=True
    if intervention=='Recovery-Direction-Off': flags['use_recovery']=False
    if intervention=='Recovery-Direction-Permuted': flags['direction_permuted']=True
    base_out,cam=hook_forward(state,False)
    with torch.no_grad(): enhanced,diag=mod(ids_list,cam[ids].detach(),base_out['logp'][ids].detach(),bank,site_tensor(cam.device),**flags); method_lp,_=output_layer_logp(state['cam'].Outputlayer,enhanced)
    base_lp=base_out['logp'][ids].detach(); full=base_out['logp'].detach().clone(); full[ids]=method_lp; met=legacy.extended_metrics(full,ids_list,state['y']); y=state['y'][ids].detach(); tr=transition(base_lp,method_lp,y); h=cam[ids].detach(); delta=diag['delta_h'].detach(); ratio=delta.norm(dim=1)/(h.norm(dim=1).clamp_min(1e-12));
    return {'group':group,'fold':fold,'ledger':ledger,'intervention':intervention,'best_epoch':int(best['best_epoch_1based']),'metrics':{k:float(met[k]) for k in METRICS},'base_metrics':{k:float(legacy.extended_metrics(base_out['logp'],ids_list,state['y'])[k]) for k in METRICS},'transition':tr,'mean_abs_prob_delta':float((method_lp.exp()-base_lp.exp()).abs().mean()),'mean_delta_norm':float(delta.norm(dim=1).mean()),'mean_delta_ratio':float(ratio.mean()),'p50_delta_ratio':float(ratio.quantile(.50)),'p90_delta_ratio':float(ratio.quantile(.90)),'p95_delta_ratio':float(ratio.quantile(.95)),'cap_fraction':float(diag['cap_fraction'].mean()),'gate_mean':float(diag['gate'].mean()),'gate_p50':float(diag['gate'].quantile(.50)),'gate_p90':float(diag['gate'].quantile(.90)),'gate_p95':float(diag['gate'].quantile(.95)),'zero_delta_fraction':float((delta.norm(dim=1)<1e-12).float().mean()),'diag':diag,'bank':bank,'best':best,'test_ids':ids_list,'y':y,'base_lp':base_lp,'method_lp':method_lp,'delta':delta,'h':h}
def metric_summary(method,rows):
    out={'method':method,'n_folds':len(rows)}
    for k in METRICS:
        a=np.asarray([r['metrics'][k] for r in rows]); out[f'{k}_mean_percent']=pct(a.mean()); out[f'{k}_sd_percent']=pct(a.std(ddof=0))
    return out
def paired(left,right,key):
    d=np.asarray([left[f][key]-right[f][key] for f in range(1,11)],dtype=float); lo,hi=ci95(d)
    return {'mean_diff':float(d.mean()),'mean_diff_percent':pct(d.mean()),'sd_diff_percent':pct(d.std(ddof=0)),'ci95_low_percent':pct(lo),'ci95_high_percent':pct(hi),'W':int((d>0).sum()),'T':int((d==0).sum()),'L':int((d<0).sum()),'per_fold':json.dumps(d.tolist()),'exclude_fold2_mean_percent':pct(d[1:].mean()),'exclude_fold2_sd_percent':pct(d[1:].std(ddof=0))}

def main():
    base=historical_base(); allres={}; fold_rows=[]; summary_rows=[]; correction_rows=[]; mechanism_rows=[]; intervention_rows=[]; memory_rows=[]; recovery_rows=[]
    # Formal method and base-anchored ledgers are both retained. Every row is
    # regenerated by loading the corresponding checkpoint.
    for ledger in ('baseanchored','method'):
        for g in GROUPS:
            vals=[]
            for f in range(1,11):
                r=replay(g,f,ledger,'Full'); allres[(ledger,g,f,'Full')]=r; vals.append(r)
                fold_rows.append({'method':g,'ledger':ledger,'fold':f,'best_epoch':r['best_epoch'],**r['metrics']})
                correction_rows.append({'method':g,'ledger':ledger,'fold':f,'best_epoch':r['best_epoch'],**r['transition'],'mean_abs_prob_delta':r['mean_abs_prob_delta'],'mean_delta_ratio':r['mean_delta_ratio'],'gate_mean':r['gate_mean']})
                mechanism_rows.append({'method':g,'ledger':ledger,'fold':f,'best_epoch':r['best_epoch'],'gate_mean':r['gate_mean'],'gate_p50':r['gate_p50'],'gate_p90':r['gate_p90'],'gate_p95':r['gate_p95'],'mean_delta_ratio':r['mean_delta_ratio'],'p50_delta_ratio':r['p50_delta_ratio'],'p90_delta_ratio':r['p90_delta_ratio'],'p95_delta_ratio':r['p95_delta_ratio'],'cap_fraction':r['cap_fraction'],'zero_delta_fraction':r['zero_delta_fraction'],'mean_delta_norm':r['mean_delta_norm'],'mean_abs_prob_delta':r['mean_abs_prob_delta']})
                b=r['bank']; latest=torch_load(CKPTS/f'shared_f{f:02d}.latest.pt'); trace=latest.get('event_trace',[]); counts=[e.get('counts',{}) for e in trace]
                recovered=[rec for t in b.records.values() for rec in t['recovered'].values()]
                memory_rows.append({'method':g,'ledger':ledger,'fold':f,'best_epoch':r['best_epoch'],'stable_asd_to_td':len(b.records['ASD_to_TD']['stable']),'recent_asd_to_td':len(b.records['ASD_to_TD']['recent']),'recovered_asd_to_td':len(b.records['ASD_to_TD']['recovered']),'stable_td_to_asd':len(b.records['TD_to_ASD']['stable']),'recent_td_to_asd':len(b.records['TD_to_ASD']['recent']),'recovered_td_to_asd':len(b.records['TD_to_ASD']['recovered']),'max_stable':max((c.get('ASD_to_TD:stable',0)+c.get('TD_to_ASD:stable',0) for c in counts),default=0),'max_recent':max((c.get('ASD_to_TD:recent',0)+c.get('TD_to_ASD:recent',0) for c in counts),default=0),'max_recovered':max((c.get('ASD_to_TD:recovered',0)+c.get('TD_to_ASD:recovered',0) for c in counts),default=0),'writes_total':b.counters['writes'],'updates_total':b.counters['updates'],'moves_total':b.counters['moves'],'evictions_total':b.counters['evictions'],'recovery_candidates':b.counters['recovery_candidates'],'recovery_confirmed':b.counters['recovery_confirmed'],'recovery_rejected':b.counters['recovery_rejected'],'recovery_invalidated':b.counters['recovery_invalidated'],'recovered_records_current':len(recovered),'recovery_margin_gain_mean':float(np.mean([x['same_head_margin_gain'] for x in recovered])) if recovered else 0.0,'recovery_direction_norm_mean':float(np.mean([x['u_recovery'].norm().item() for x in recovered])) if recovered else 0.0,'event_trace_epochs':len(trace)})
                for rec in recovered: recovery_rows.append({'method':g,'ledger':ledger,'fold':f,'patient_id':int(rec['patient_id']),'error_type':rec['error_type'],'wrong_epoch':rec['wrong_epoch'],'confirmed_correct_epoch':rec['confirmed_correct_epoch'],'same_head_margin_gain':rec['same_head_margin_gain'],'raw_change_norm':float(rec['raw_recovery_change'].norm()),'drift_corrected_norm':float(rec['drift_corrected_recovery_change'].norm()),'direction_norm':float(rec['u_recovery'].norm()),'site_id':rec['site_id']})
            summary_rows.append(metric_summary(f'{g} [{ledger}]',vals))
    # Historical row and paired comparisons use the locked E74.4-G2 fold files.
    summary_rows.insert(0,{'method':BASE_GROUP,**{f'{k}_mean_percent':pct(np.mean([base[f][k] for f in range(1,11)])) for k in METRICS},**{f'{k}_sd_percent':pct(np.std([base[f][k] for f in range(1,11)],ddof=0)) for k in METRICS}})
    for ledger in ('baseanchored','method'):
        maps={g:{f:allres[(ledger,g,f,'Full')]['metrics'] for f in range(1,11)} for g in GROUPS}; bmap={f:base[f] for f in range(1,11)}
        for label,lft,rgt in [('G1-minus-base',GROUPS[0],BASE_GROUP),('G2-minus-G1',GROUPS[1],GROUPS[0]),('G2-minus-base',GROUPS[1],BASE_GROUP)]:
            lm=maps[lft] if lft in maps else bmap; rm=maps[rgt] if rgt in maps else bmap
            for k in METRICS: fold_rows_key=key=paired({f:{k:lm[f][k] for k in METRICS} for f in range(1,11)},{f:{k:rm[f][k] for k in METRICS} for f in range(1,11)},k); fold_rows_key.update({'ledger':ledger,'comparison':label,'metric':k});
            # append after constructing all metrics below
    pair_rows=[]
    for ledger in ('baseanchored','method'):
        maps={g:{f:allres[(ledger,g,f,'Full')]['metrics'] for f in range(1,11)} for g in GROUPS}; bmap={f:base[f] for f in range(1,11)}
        for label,lft,rgt in [('G1-minus-base',GROUPS[0],BASE_GROUP),('G2-minus-G1',GROUPS[1],GROUPS[0]),('G2-minus-base',GROUPS[1],BASE_GROUP)]:
            lm=maps[lft] if lft in maps else bmap; rm=maps[rgt] if rgt in maps else bmap
            for k in METRICS: pair_rows.append({'ledger':ledger,'comparison':label,'metric':k,**paired({f:{k:lm[f][k] for k in METRICS} for f in range(1,11)},{f:{k:rm[f][k] for k in METRICS} for f in range(1,11)},k)})
    # Same-checkpoint interventions. These are read-only and are deliberately
    # not promoted to additional training groups.
    interventions=('Full','Memory-Off','Key-Value-Association-Break','Error-Type-Swap','Recency-Off','Recovery-Direction-Off','Recovery-Direction-Permuted')
    for ledger in ('baseanchored','method'):
        for g in GROUPS:
            for f in range(1,11):
                for iv in interventions:
                    if iv=='Recovery-Direction-Off' and g==GROUPS[0]: continue
                    r=replay(g,f,ledger,iv); intervention_rows.append({'method':g,'ledger':ledger,'fold':f,'intervention':iv,'best_epoch':r['best_epoch'],**r['metrics'],**r['transition'],'mean_abs_prob_delta':r['mean_abs_prob_delta'],'mean_delta_ratio':r['mean_delta_ratio'],'cap_fraction':r['cap_fraction'],'gate_mean':r['gate_mean']})
    _rows_csv(RESULTS/'summary.csv',summary_rows); _rows_csv(RESULTS/'fold_metrics.csv',fold_rows); _rows_csv(RESULTS/'paired_comparisons.csv',pair_rows); _rows_csv(RESULTS/'correction_analysis.csv',correction_rows); _rows_csv(RESULTS/'mechanism.csv',mechanism_rows); _rows_csv(RESULTS/'memory_analysis.csv',memory_rows); _rows_csv(RESULTS/'recovery_analysis.csv',recovery_rows); _rows_csv(RESULTS/'intervention_analysis.csv',intervention_rows)
    report={'status':'ANALYZED','summary':summary_rows,'paired_comparisons':pair_rows,'folds':20,'interventions':len(intervention_rows),'memory_rows':memory_rows,'recovery_events':len(recovery_rows),'note':'All intervention rows are same-checkpoint replays; no test label enters training or memory updates.'}
    atomic_json(RESULTS/'analysis.json',report); print(json.dumps({'status':'ANALYZED','summary_rows':len(summary_rows),'pair_rows':len(pair_rows),'interventions':len(intervention_rows),'recovery_events':len(recovery_rows)},ensure_ascii=False))

if __name__=='__main__': main()

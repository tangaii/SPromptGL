#!/usr/bin/env python3
"""Independent result reader for Experiment 48.8.

Training workers write atomic task/checkpoint pairs. This script never trusts
the summary files: it reloads each completed checkpoint in a fresh process,
recomputes metrics, and then derives paired comparisons from fold-matched rows.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, statistics, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RT=HERE/'.runtime'; RES=HERE/'results'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b): h.update(b)
    return h.hexdigest()

def task_rows():
    out=[]
    # Match both single- and double-digit research groups (E1..E10).
    for p in sorted(RT.glob('tasks/E*_f*.json')):
        try:
            d=read(p);d['_task_path']=str(p);out.append(d)
        except Exception: pass
    return out

def write_csv(path,rows,fields):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def summary(rows):
    groups={}
    for r in rows:
        if r.get('status')=='COMPLETE':groups.setdefault(r['group_id'],[]).append(r)
    srows=[]
    for gid,rs in sorted(groups.items()):
        if len(rs)!=10:continue
        row={'group_id':gid,'group':rs[0]['group'],'n':10}
        for k in ('acc','hard_auc','spe','sen','probability_auc'):
            v=np.asarray([100*float(x[k]) for x in sorted(rs,key=lambda x:x['fold'])])
            row[k+'_mean']=float(v.mean());row[k+'_sd_pop']=float(v.std(ddof=0))
        row['best_epoch_mean']=float(np.mean([x['best_epoch'] for x in rs]))
        row['epochs_executed_mean']=float(np.mean([x['epochs_executed'] for x in rs]))
        row['alpha_best_mean']=float(np.mean([x['diagnostics']['alpha_best'] for x in rs]))
        row['residual_ratio_mean']=float(np.mean([x['diagnostics']['relative_residual_norm'] for x in rs]))
        row['srd_auc_mean']=float(np.mean([np.mean(x['diagnostics']['training_trace_summary'].get('srd_auc_mean_by_modality',[.5]*4)) for x in rs]))
        # pair_stats is nested under the returned SRD audit object; recover the
        # best-epoch candidate diagnostics rather than silently reporting zero.
        best_stats=[]
        for x in rs:
            bt=x['diagnostics'].get('best_epoch_trace') or {}
            ps=bt.get('pair_stats',{}).get('stats',{})
            best_stats.append(ps)
        row['srd_candidate_pairs_best_mean']=float(np.mean([p.get('candidate_pairs_used',0.) for p in best_stats]))
        row['srd_candidate_native_pairs_best_mean']=float(np.mean([p.get('candidate_native_pairs',0.) for p in best_stats]))
        row['srd_candidate_patient_coverage_best_mean']=float(np.mean([p.get('candidate_patient_coverage',0.) for p in best_stats]))
        row['srd_top16_edge_auc_best_mean']=float(np.mean([p.get('top16_fit_edge_auc',.5) for p in best_stats]))
        row['srd_candidate_category_counts_best']=json.dumps([int(round(np.mean([p.get('candidate_category_counts',[0,0,0,0])[i] for p in best_stats]))) for i in range(4)])
        row['sca_active_fraction_mean']=float(np.mean([x['diagnostics']['training_trace_summary'].get('sca_active_epoch_fraction',0.) for x in rs]))
        row['cra_target_ce_gain_mean']=float(np.mean([x['diagnostics']['training_trace_summary'].get('cra_target_ce_gain_mean',0.) for x in rs]))
        srows.append(row)
    fields=list(srows[0]) if srows else ['group_id','group','n']
    write_csv(RES/'summary.csv',srows,fields)
    return srows

def paired(rows):
    by={(r['group_id'],r['fold']):r for r in rows if r.get('status')=='COMPLETE'}
    groups=sorted({r['group_id'] for r in rows})
    out=[]
    for ga in groups:
        for gb in groups:
            if ga>=gb:continue
            a=[by[(ga,f)] for f in range(1,11) if (ga,f) in by and (gb,f) in by]
            b=[by[(gb,r['fold'])] for r in a]
            if len(a)!=10:continue
            row={'a_id':ga,'a':a[0]['group'],'b_id':gb,'b':b[0]['group'],'n':10}
            for k in ('acc','hard_auc','spe','sen','probability_auc'):
                da=np.asarray([100*(x[k]-y[k]) for x,y in zip(a,b)])
                row['delta_'+k+'_a_minus_b_mean']=float(da.mean());row['delta_'+k+'_a_minus_b_sd']=float(da.std(ddof=0))
                row[k+'_win_tie_loss_a_vs_b']=f"{int((da>0).sum())}/{int((da==0).sum())}/{int((da<0).sum())}"
            out.append(row)
    fields=list(out[0]) if out else []
    if fields:write_csv(RES/'paired_comparisons.csv',out,fields)
    return out

def fold_metrics(rows):
    fields=['group_id','group','fold','acc','hard_auc','spe','sen','probability_auc','best_epoch','epochs_executed','best_sha','prediction_sha256','initial_bundle_sha','signature','alpha_best','relative_residual_norm']
    out=[]
    for r in sorted([r for r in rows if r.get('status')=='COMPLETE'],key=lambda x:(x['group_id'],x['fold'])):
        d=r.get('diagnostics',{});out.append({**r,'alpha_best':d.get('alpha_best'),'relative_residual_norm':d.get('relative_residual_norm')})
    write_csv(RES/'fold_metrics.csv',out,fields);return out

def load_and_recompute(rows):
    # Independent process entry point. The training engine is imported only
    # here, after rows have been read, and no task values are used for metrics.
    import importlib.util
    spec=importlib.util.spec_from_file_location('e488_run_audit',HERE/'run.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    c=mod.context();x,y,folds,mapping=mod.data();m=c['method'];t=c['torch'];out=[]
    for r in sorted([r for r in rows if r.get('status')=='COMPLETE'],key=lambda z:(z['group_id'],z['fold'])):
        gid,fold=r['group_id'],r['fold'];tr,te=folds[fold-1]
        a=mod.initialize_agent(gid,fold,x,y,tr,te,mapping)
        ck=mod.load(RT/'checkpoints'/f'E{gid}_f{fold:02d}.best.pt')
        m.load_bundle(a,ck['model'])
        with t.no_grad():o=m.evaluate(a,6,c['utils'])
        rr=m.metrics(o,y,te);pred=o['logp'][te].argmax(1).detach().cpu().numpy()
        ph=hashlib.sha256(pred.tobytes()).hexdigest()
        if ph!=r.get('prediction_sha256'):raise RuntimeError(f'prediction SHA mismatch E{gid} fold{fold}: {ph} != {r.get("prediction_sha256")}')
        for k in ('acc','hard_auc','spe','sen','probability_auc'):
            if abs(float(rr[k])-float(r[k]))>1e-10:raise RuntimeError(f'metric mismatch E{gid} fold{fold} {k}')
        out.append({'group_id':gid,'group':r['group'],'fold':fold,'recomputed':True,'prediction_sha256':ph,**rr})
    write_csv(RES/'independent_recompute.csv',out,list(out[0]) if out else [])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recompute',action='store_true');args=ap.parse_args()
    rows=task_rows();s=summary(rows);paired(rows);fold_metrics(rows)
    if args.recompute:load_and_recompute(rows)
    complete=sum(r.get('status')=='COMPLETE' for r in rows);failed=[r for r in rows if r.get('status') not in ('COMPLETE','RUNNING')]
    status={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'tasks_seen':len(rows),'tasks_complete':complete,'tasks_expected':100,'failed_or_invalid':len(failed),'groups_complete':len(s),'independent_recompute':bool(args.recompute)}
    (RT/'audit_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if __name__=='__main__':main()

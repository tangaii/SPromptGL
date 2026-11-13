"""Independent replay and mechanism audit for Experiment 79.2."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run
from memory_common import output_layer_logp

def _jsonable(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x,dict): return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_jsonable(v) for v in x]
    if isinstance(x,np.floating): return float(x)
    if isinstance(x,np.integer): return int(x)
    return x

def _metrics(m): return {k:float(v) for k,v in m.items() if k not in ('prob','pred')}
def _allclose(a,b,atol=1e-6): return bool(np.allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=1e-6))

def _memory_count(mem,idx):
    if not mem:return 0
    return len(mem.get('ids',[])) if idx<2 else len(mem.get('components',[]))

def _memory_ids(mem,idx):
    if not mem:return []
    if idx==0:return [int(pid) for pair in mem.get('ids',[]) for pid in pair]
    if idx==1:return [int(pid) for pid in mem.get('ids',[])]
    return [int(pid) for c in mem.get('components',[]) for pid in c.get('members',[])]

def _retrieval_bad(diag,idx,test_ids):
    bad=[]; testset=set(map(int,test_ids))
    for q,row in zip(test_ids,diag.get('retrieval',[])):
        vals=[]
        if idx==0:
            for pair in row.get('pair_ids',[]): vals.extend(list(pair))
        elif idx==1:
            for z in (row.get('asd',{}),row.get('td',{})): vals.extend(z.get('patient_ids',[]))
        else:
            for z in (row.get('asd',{}),row.get('td',{})):
                for members in z.get('component_members',[]): vals.extend(members if isinstance(members,list) else [members])
        if any(int(v)==int(q) or int(v) in testset for v in vals): bad.append(int(q))
    return bad

def replay_branch(idx,module,state,ids,cam,base_logp,memory,sites,intervention=None):
    module.eval(); kwargs={}
    if idx==0: kwargs={'content_permuted':intervention=='Memory-Content-Permuted','pair_direction_permuted':intervention=='Pair-Direction-Permuted'}
    elif idx==1: kwargs={'content_permuted':intervention=='Memory-Content-Permuted','evidence_mask_permuted':intervention=='Evidence-Mask-Permuted'}
    else: kwargs={'content_permuted':intervention=='Memory-Content-Permuted','variance_off':intervention=='Variance-Off'}
    it=torch.as_tensor(ids,dtype=torch.long,device=cam.device)
    with torch.no_grad():
        enh,diag=module(ids,cam[it].detach(),base_logp[it].detach(),memory,sites,**kwargs)
        lp,_=output_layer_logp(state['cam'].Outputlayer,enh)
    return lp,enh,diag

def audit_checkpoint(fold,ledger,group,ckpt_path,state,hist,hist_best,test_ids,eligible,y,sites):
    rec=torch.load(ckpt_path,map_location='cpu',weights_only=False); idx=run.GROUPS.index(group)
    hist.load_bundle(state,rec['base_state'])
    with torch.no_grad(): evaluated,cam=run.hook_forward(state,False)
    base_lp=evaluated['logp'].detach(); base_m=run.legacy.extended_metrics(base_lp,test_ids,y)
    module=run.init_modules(y.device)[idx]; module.load_state_dict(rec['module']); module.eval(); mem=rec.get('memory_used',[None,None,None])[idx]
    lp,enh,diag=replay_branch(idx,module,state,test_ids,cam,base_lp,mem,sites)
    ti=torch.as_tensor(test_ids,dtype=torch.long,device=y.device); full=base_lp.clone(); full[ti]=lp
    method_m=run.legacy.extended_metrics(full,test_ids,y)
    replay_ok=_allclose(method_m['prob'],rec.get('probability',[])) and _allclose(base_m['prob'],rec.get('base_probability',[]))
    base_pred_ok=np.array_equal(np.asarray(base_m['pred'],int),np.asarray(rec.get('base_prediction',[]),int)); method_pred_ok=np.array_equal(np.asarray(method_m['pred'],int),np.asarray(rec.get('prediction',[]),int))
    anchor_ok=True
    if ledger=='base_anchored': anchor_ok=run.same_state(rec['base_state'],hist_best['best_state'])
    mem_ids=_memory_ids(mem,idx); legal=set(map(int,eligible)); testset=set(map(int,test_ids)); excluded=run.FOLD2_EXCLUDE if fold==2 else set()
    legality=not(set(mem_ids)&testset) and not(set(mem_ids)&excluded) and set(mem_ids).issubset(legal)
    off_lp,off_h,off_diag=replay_branch(idx,module,state,test_ids,cam,base_lp,None,sites)
    off_exact=bool(torch.equal(off_h,cam[ti])); off_full=base_lp.clone(); off_full[ti]=off_lp; off_m=run.legacy.extended_metrics(off_full,test_ids,y)
    interventions=[]
    names=['Memory-Off','Memory-Content-Permuted']
    if idx==0:names.append('Pair-Direction-Permuted')
    if idx==1:names.append('Evidence-Mask-Permuted')
    if idx==2:names.append('Variance-Off')
    for name in names:
        ilp,ih,idg=replay_branch(idx,module,state,test_ids,cam,base_lp,mem,sites,None if name=='Memory-Off' else name)
        if name=='Memory-Off': im=off_m
        else:
            ff=base_lp.clone(); ff[ti]=ilp; im=run.legacy.extended_metrics(ff,test_ids,y)
        interventions.append({'fold':fold,'ledger':ledger,'group':group,'epoch':int(rec['best_epoch_1based']),'intervention':name,'metrics':_metrics(im),'acc_delta_full_minus_intervention':float(method_m['acc']-im['acc']),'hard_auc_delta':float(method_m['hard_auc']-im['hard_auc']),'prob_auc_delta':float(method_m['prob_auc']-im['prob_auc']),'hard_prediction_flips':int(np.sum(np.asarray(method_m['pred'])!=np.asarray(im['pred']))),'prob_abs_delta_mean':float(np.mean(np.abs(np.asarray(method_m['prob'])-np.asarray(im['prob'])))),'off_exact':off_exact if name=='Memory-Off' else None,'delta_ratio_mean':float(idg['delta_ratio'].mean())})
    transition=run.correction_stats(base_lp,full,test_ids,y)
    return {'fold':fold,'ledger':ledger,'group':group,'checkpoint':str(ckpt_path),'epoch':int(rec['best_epoch_1based']),'checkpoint_sha256':hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),'training_source_sha256':rec.get('source_sha256'),'replay_metrics':_metrics(method_m),'stored_metrics':rec.get('metrics',{}),'base_metrics':_metrics(base_m),'replay_probability_match':replay_ok,'base_prediction_match':base_pred_ok,'method_prediction_match':method_pred_ok,'base_anchor_state_match':anchor_ok,'cam_shape':list(cam.shape),'empty_memory_exact':off_exact,'memory_count':_memory_count(mem,idx),'memory_legal':legality,'memory_ids_count':len(mem_ids),'self_or_test_retrieval_bad':_retrieval_bad(diag,idx,test_ids),'transition':transition,'gate_mean':float(diag['gate'].mean()),'delta_ratio_mean':float(diag['delta_ratio'].mean()),'cap_fraction':float(diag['cap_fraction'].mean()),'interventions':interventions}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('replay','summary'),default='replay'); ap.parse_args()
    started=time.time(); best_rows=[]; int_rows=[]; latest_rows=[]
    for fold in range(1,11):
        state,warm,identity,task,hist_ckpt=run.prepare_fold(fold); hist=run.legacy.load_base().load_history(); hist_best=run.torch_load(hist_ckpt); eligible=run.eligible_ids(state,fold); test_ids=[int(x) for x in state['test_np']]; y=state['y'].long(); sites=run.site_tensor(y.device)
        for ledger in ('base_anchored','method'):
            for group in run.GROUPS:
                p=ROOT/f'.runtime/checkpoints/{group}_f{fold:02d}.{ledger}.pt'; row=audit_checkpoint(fold,ledger,group,p,state,hist,hist_best,test_ids,eligible,y,sites); ints=row.pop('interventions'); best_rows.append(row); int_rows.extend(ints)
        latest=ROOT/f'.runtime/checkpoints/shared_f{fold:02d}.latest.pt'; obj=torch.load(latest,map_location='cpu',weights_only=False); trace=obj.get('trace',[]); expected=(int(trace[-1]['epoch'])+1) if trace else None; ok=expected is not None and int(obj.get('next_epoch',-1))==expected and set(obj.get('modules',{}))==set(run.GROUPS) and set(obj.get('optimizers',{}))==set(run.GROUPS) and len(obj.get('memories',[]))==3 and set(obj.get('rng',{}))=={'python','numpy','cpu','cuda'}
        hist.load_bundle(state,obj['base_state']);
        with torch.no_grad(): _,latest_cam=run.hook_forward(state,False)
        ok=ok and tuple(latest_cam.shape)==(871,144); latest_rows.append({'fold':fold,'status':'PASS' if ok else 'FAIL','next_epoch':obj.get('next_epoch'),'expected_next_epoch':expected,'trace_last_epoch':trace[-1]['epoch'] if trace else None,'cam_shape':list(latest_cam.shape),'required_state_present':ok})
    best_ok=all(x['replay_probability_match'] and x['base_prediction_match'] and x['method_prediction_match'] and x['base_anchor_state_match'] and x['empty_memory_exact'] and x['memory_legal'] and not x['self_or_test_retrieval_bad'] for x in best_rows); latest_ok=all(x['status']=='PASS' for x in latest_rows)
    audit={'status':'PASS' if best_ok and latest_ok else 'FAIL','best_checkpoints_expected':60,'best_checkpoints_audited':len(best_rows),'latest_expected':10,'latest_audited':len(latest_rows),'best_pass':best_ok,'latest_pass':latest_ok,'best_rows':best_rows,'interventions':int_rows,'latest_rows':latest_rows,'audit_repairs':[{'scope':'latest metadata only','detail':'Corrected next_epoch to trace_last_epoch+1; no weights, predictions, metrics or base state changed.'}],'elapsed_seconds':time.time()-started}
    (ROOT/'results/audit_status.json').write_text(json.dumps(_jsonable(audit),ensure_ascii=False,indent=2)+'\n')
    fields=('fold','ledger','group','epoch','intervention','acc_delta_full_minus_intervention','hard_auc_delta','prob_auc_delta','hard_prediction_flips','delta_ratio_mean'); lines=[','.join(fields)]
    for r in int_rows: lines.append(','.join(str(r.get(k,'')) for k in fields))
    (ROOT/'results/intervention_analysis.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':audit['status'],'best':len(best_rows),'latest':len(latest_rows),'elapsed_seconds':audit['elapsed_seconds']},indent=2))

if __name__=='__main__': main()

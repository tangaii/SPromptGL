"""Scientific preflight, fixed-checkpoint diagnostics and Chinese reporting."""
from __future__ import annotations
import copy,csv,json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
import run as r
from sage_align import build_normative,evidence_for_mapping,concept_features,empirical_cdf,deviation_score,zero_evidence_test,SAGEAlign
import integration


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def base_snapshot(agent):
    return dict(model=r.model_state(agent),gfr_opt=copy.deepcopy(agent.optimizer_MP0.state_dict()),
        pln_opt=copy.deepcopy(agent.optimizer_MF.state_dict()),rng=r.rng_capture(torch))


def base_restore(agent,s):
    r.load_model_state(agent,s['model']);agent.optimizer_MP0.load_state_dict(s['gfr_opt']);agent.optimizer_MF.load_state_dict(s['pln_opt']);r.rng_restore(torch,s['rng'])


def preflight():
    started=time.monotonic();checked_sources={name:r.sha_file(r.EXP/name) for name in ['run.py','sage_align.py','integration.py','audit.py','config.json','prompts.json','ontology.json']}
    c=r.context(0);o=r.load_ontology();x,y=c['x'],c['y'];train,test=c['folds'][0]
    checks={};details={};names=[str(v) for k in c['mapping'] for v in c['mapping'][k]]
    expected={57:'dvars',58:'outlier',60:'mean_fd',62:'perc_fd',63:'gsr'}
    checks['FUNC_field_names']=all(v in names[k].lower() for k,v in expected.items())
    checks['modal_dimension_and_counts']=x.shape==(871,320) and [len(v) for v in c['mapping'].values()]==[48,6,10,256] and np.bincount(y).tolist()==[403,468]
    checks['phenotype_one_hot']=all(np.allclose(x[:,a:b].sum(1),1) for a,b in [(0,26),(26,46),(46,48)])
    checks['CORR_column_names']=names[64:]==[str(v) for v in c['mapping']['Correlation']]
    checks['endpoint_unknown_four']=o['unknown_edges']==[195,225,232,236] and int((np.asarray(o['endpoints'])>=0).all(1).sum())==252
    # Actual wrong graph is regenerated from the same ROI permutation and rules.
    perm=np.asarray(o['wrong_control']['permutation']);rois=np.asarray(o['concept_rois'],bool);wr=np.zeros_like(rois);wr[:,perm]=rois
    edges=integration.edge_masks(wr,np.asarray(o['endpoints']),np.asarray(o['subcortical'],bool))
    families=np.asarray(o['families'],bool);wf=np.zeros_like(families);wf[:,perm]=families
    weights=integration.pooling(wr,edges,wf,np.asarray(o['endpoints']))
    checks['wrong_ROI_FC_joint_definition']=np.array_equal(wr,np.asarray(o['wrong_rois'],bool)) and np.array_equal(edges,np.asarray(o['wrong_edges'],bool)) and np.array_equal(weights,np.asarray(o['wrong_weights']))
    checks['wrong_preserves_exact_ROI_degree_and_type']=np.array_equal(np.asarray(o['degree']),np.asarray(o['degree'])[perm]) and np.array_equal(np.asarray(o['subcortical']),np.asarray(o['subcortical'])[perm])
    checks['wrong_mapping_nontrivial']=bool(np.any(edges!=np.asarray(o['concept_edges'])))
    ev=r.load_evidence(1,'conditional');real=r.bundle('E9_SAGE_FULL',ev,o,x,train);wrong=r.bundle('E5_WRONG_SEMANTICS',ev,o,x,train)
    num=r.bundle('E2_NUMERIC_ONLY',ev,o,x,train);data=r.bundle('E6_DATA_PRIOR',ev,o,x,train)
    checks['E2_E6_distinct_evidence']=bool(np.mean(np.abs(num['effective'][train]-data['effective'][train]))>1e-7)
    checks['numeric_all_18_channels_present']=all(np.any(b['features'][train]!=0,axis=(0,1)).sum()==18 for b in [num,data,real])
    checks['same_score_CDF_calibration']=np.array_equal(real['score'],deviation_score(real['features'])) and np.array_equal(real['calibration'],np.sort(real['score'][ev['td_idx']],axis=0)) and np.array_equal(real['strength'],empirical_cdf(real['score'],real['calibration']))
    checks['conserved_mass']=bool(np.allclose(real['matched'].sum(1)+real['unmatched'],1,atol=1e-7))
    checks['continuous_not_all_rejected_or_forced']=bool(0<real['unmatched'][train].mean()<1 and (real['matched'][train]>0).any())
    checks['hard_rejection_exact']=bool(np.all(real['matched'][~real['accepted']]==0) and np.all(real['unmatched'][~real['accepted']]==1))
    # No held-out label can change the fitted normative reference or training CDF.
    mutated=y.copy();mutated[test]=1-mutated[test]
    ev_flip=build_normative(x,train,mutated,np.asarray(o['endpoints']),'conditional')
    checks['outer_test_labels_unused_reference']=all(np.array_equal(ev[k],ev_flip[k]) for k in ev_flip if isinstance(ev_flip[k],np.ndarray))
    checks['all_inner_references_TD_disjoint_queries']=all(set(f['reference_ids']).isdisjoint(f['query_ids']) and np.all(y[f['reference_ids']]==1) and set(f['reference_ids']).issubset(set(train)) for f in ev['stats']['inner_fits'])
    checks['outer_test_uses_full_train_TD']=set(ev['stats']['full_reference_ids'])==set(train[y[train]==1])
    # Field perturbation tests evaluate reference/evidence only, never test ACC.
    changed=x.copy();changed[test[0],60]+=2
    fd_ev=build_normative(changed,train,y,np.asarray(o['endpoints']),'conditional')
    checks['actual_mean_FD_changes_normative_prediction']=bool(np.max(np.abs(fd_ev['prediction'][test[0]]-ev['prediction'][test[0]]))>1e-8)
    changed=x.copy();changed[test[0],54]+=20
    efc_ev=build_normative(changed,train,y,np.asarray(o['endpoints']),'conditional')
    checks['FUNC_EFC_not_mistaken_for_FD']=np.array_equal(efc_ev['z'],ev['z']) and np.array_equal(efc_ev['qc_support'],ev['qc_support'])

    checks['QC_one_sided_actual_formula']=bool(np.allclose(ev['qc_burden'],np.maximum((x[:,ev['qc_cols']]-ev['qc_center'])/ev['qc_scale'],0).mean(1),atol=1e-6) and np.array_equal(ev['qc_cols'],np.array([57,58,60,62])) and np.allclose(ev['qc_support'],np.exp(-.2*np.minimum(ev['qc_burden'],30)),atol=1e-7))
    poor=copy.deepcopy(ev);poor['qc_support']=np.zeros(len(x),dtype=np.float32);poor['severe_qc']=np.ones(len(x),bool)
    rejected=r.bundle('E9_SAGE_FULL',poor,o,x,train)
    checks['severe_QC_suppresses_evidence']=bool(np.count_nonzero(rejected['effective'])==0)
    forced=r.bundle('E7_FORCED_ALIGNMENT',poor,o,x,train)
    checks['E7_forces_available_concepts_same_values']=bool(np.allclose(forced['matched'].sum(1),1) and np.array_equal(forced['features'],real['features']))
    # Compare historical simple-2 training to the implementation, including both Adam states.
    r.seed_everything(torch,0);agent=r.make_agent(c,1);start=base_snapshot(agent)
    for _ in range(2):r.e0_step(agent,c)
    implemented=base_snapshot(agent);impl_logp=r.evaluate_base(agent,c)['logp']
    base_restore(agent,start)
    for _ in range(2):agent.run_epoch(mode='simple-2')
    historical=base_snapshot(agent);hist_logp=r.evaluate_base(agent,c)['logp']
    checks['historical_simple2_two_Adam_two_steps_exact']=equal(implemented,historical) and torch.equal(impl_logp,hist_logp)
    checks['terminal_resume_takes_no_extra_update']=list(r.epoch_indices(202,201))==[] and list(r.epoch_indices(450,0))==[] and next(iter(r.epoch_indices(201,200)))==201
    point=base_snapshot(agent);r.e0_step(agent,c);expected_next=base_snapshot(agent)
    disk=r.RUNTIME/'preflight_baseline_restore.pt';r.atomic_torch_save(torch,point,disk)
    disk.with_suffix('.pt.sha256').unlink()
    loaded_point=r.checked_load(torch,disk)
    checks['atomic_payload_recovers_missing_sidecar']=equal(point,loaded_point) and disk.with_suffix('.pt.sha256').read_text().strip()==r.sha_file(disk)
    base_restore(agent,loaded_point);r.e0_step(agent,c)
    checks['baseline_disk_model_two_Adam_RNG_resume_exact']=equal(expected_next,base_snapshot(agent))
    continuation=r.rng_capture(torch);a2=r.make_agent(c,2);a2_hash=r.source_digest(a2);a2_rng=r.rng_hash(r.rng_capture(torch))
    r.rng_restore(torch,continuation);a2b=r.make_agent(c,2)
    checks['baseline_fold_boundary_RNG_exact']=a2_hash==r.source_digest(a2b) and a2_rng==r.rng_hash(r.rng_capture(torch))
    # PEN was constructed, but hooks prove it is never evaluated in the B0 path.
    called={'MLP':0,'avg_pool':0}
    hooks=[getattr(agent.ModalFusion,key).register_forward_hook(lambda _m,_i,_o,key=key:called.__setitem__(key,called[key]+1)) for key in called]
    r.evaluate_base(agent,c)
    for h in hooks:h.remove()
    checks['PEN_constructed_and_forward_bypassed']=all(v==0 for v in called.values()) and agent.ModalFusion.prompt_text_features.shape==(8,1024)
    # A previously parity-matched B0 fold is used for training-only probes, never for checkpoint selection.
    ap=r.RUNTIME/'rejected_protocol_audit_v1/anchors/anchor_fold01.pt'
    anchor=torch.load(ap,map_location='cpu',weights_only=False)
    model,branch,opt,params=r.setup_adaptation(c,1,'E9_SAGE_FULL',anchor)
    cam=anchor['cam'].cuda();att=anchor['attention'].cuda();features=torch.tensor(real['effective'],device='cuda')
    frozen=r.trunk_hash(model);head_before=copy.deepcopy(model.ModalFusion.Outputlayer.state_dict())
    checks['initial_residual_exact_zero']=bool(torch.count_nonzero(branch(features,cam)[1])==0)
    gradient_steps=[]
    for step in range(3):
        cls,energy,g=r.adaptation_step(model,branch,opt,params,features,cam,att,c,train);gradient_steps.append(g)
    checks['first_step_has_branch_gradient']=gradient_steps[0]['branch/residual_proj.weight']>0
    checks['later_all_trainable_paths_have_nonzero_gradients']=all(v>0 for v in gradient_steps[2].values())
    checks['head_adaptation_updates_original_weights']=any(not torch.equal(v,head_before[k]) for k,v in model.ModalFusion.Outputlayer.state_dict().items())
    checks['trunk_weights_frozen_actual']=r.trunk_hash(model)==frozen
    checks['zero_match_strict_zero_after_learning']=zero_evidence_test(branch,cam)
    checks['zero_evidence_changed_CAM_strict_zero']=zero_evidence_test(branch,cam*3+7)
    off=evidence_for_mapping(ev,np.asarray(o['weights']),np.asarray(o['coverage']),eligible=np.zeros(4,bool))
    checks['semantic_off_cuts_actual_tensor']=np.count_nonzero(off['effective'])==0 and zero_evidence_test(branch,cam)
    neutral=copy.deepcopy(ev);neutral['z']=np.zeros_like(ev['z'])
    nb=r.bundle('E9_SAGE_FULL',neutral,o,x,train)
    checks['no_disease_evidence_from_neutral_or_QC_alone']=np.count_nonzero(nb['effective'])==0
    patients=train[:2];patient_rows=[]
    with torch.no_grad():
        rr=branch(features,cam)[1];ww=branch(torch.tensor(wrong['effective'],device='cuda'),cam)[1]
        for idx in patients:
            delta=float(np.mean(np.abs(real['features'][idx]-wrong['features'][idx])))
            delta_res=float((rr[idx]-ww[idx]).norm())
            patient_rows.append(dict(subject_index=int(idx),features_L1_difference=delta,residual_L2_difference=delta_res,
               real_positive_by_concept=real['features'][idx,:,0].tolist(),wrong_positive_by_concept=wrong['features'][idx,:,0].tolist(),
               real_negative_by_concept=real['features'][idx,:,1].tolist(),wrong_negative_by_concept=wrong['features'][idx,:,1].tolist()))
    checks['two_training_patients_mapping_changes_evidence_and_residual']=all(p['features_L1_difference']>0 and p['residual_L2_difference']>0 for p in patient_rows)
    point=dict(branch=copy.deepcopy(branch.state_dict()),head=copy.deepcopy(model.ModalFusion.Outputlayer.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=r.rng_capture(torch))
    disk2=r.RUNTIME/'preflight_adaptation_restore.pt';r.atomic_torch_save(torch,point,disk2)
    r.adaptation_step(model,branch,opt,params,features,cam,att,c,train)
    expected=dict(branch=copy.deepcopy(branch.state_dict()),head=copy.deepcopy(model.ModalFusion.Outputlayer.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=r.rng_capture(torch))
    pt=r.checked_load(torch,disk2);branch.load_state_dict(pt['branch']);model.ModalFusion.Outputlayer.load_state_dict(pt['head']);opt.load_state_dict(pt['optimizer']);r.rng_restore(torch,pt['rng'])
    r.adaptation_step(model,branch,opt,params,features,cam,att,c,train)
    actual=dict(branch=branch.state_dict(),head=model.ModalFusion.Outputlayer.state_dict(),optimizer=opt.state_dict(),rng=r.rng_capture(torch))
    checks['adaptation_disk_Adam_RNG_next_step_exact']=equal(expected,actual)
    init_heads=[];init_branches=[];init_rng=[];e8_check=True
    for group in r.CONFIGS[1:]:
        a,b,opt2,p=r.setup_adaptation(c,1,group,anchor)
        init_heads.append(r.sha_bytes(b''.join(v.cpu().numpy().tobytes() for v in a.ModalFusion.Outputlayer.state_dict().values())))
        init_rng.append(r.rng_hash(r.rng_capture(torch)))
        if b is not None:init_branches.append(r.sha_bytes(b''.join(v.cpu().numpy().tobytes() for v in b.state_dict().values())))
        if group=='E8_FROZEN_HEAD':
            before=copy.deepcopy(a.ModalFusion.Outputlayer.state_dict());r.adaptation_step(a,b,opt2,p,features,cam,att,c,train)
            e8_check=equal(before,a.ModalFusion.Outputlayer.state_dict()) and all(not p.requires_grad for p in a.ModalFusion.Outputlayer.parameters())
        del a,b,opt2,p
    checks['E8_head_weights_strictly_frozen']=e8_check
    checks['all_groups_same_head_branch_initialization_and_training_RNG']=len(set(init_heads))==len(set(init_branches))==len(set(init_rng))==1
    with torch.no_grad():
        _,rr,extra=branch(features,cam,True)
        checks['finite_bounded_smooth_residual']=bool(torch.isfinite(rr).all() and torch.all(rr.norm(dim=1)<=.080001*cam.flatten(1).norm(dim=1)))
    details.update(patient_probes=patient_rows,gradients_by_step=gradient_steps,wrong_mapping=o['wrong_control'],
        fields={str(i):names[i] for i in range(64)},probe_anchor_sha256=r.sha_file(ap),baseline_initial_hash=r.source_digest(agent),
        actual_device=torch.cuda.get_device_name(0),cpu_threads=torch.get_num_threads(),wall_seconds=time.monotonic()-started)
    # All-fold training-only calibration diagnostics, broken down by reference support.
    calibration_rows=[]
    for fold,(tr,te) in enumerate(c['folds'],1):
        evf=r.load_evidence(fold,'conditional');br=r.bundle('E9_SAGE_FULL',evf,o,x,tr);bw=r.bundle('E5_WRONG_SEMANTICS',evf,o,x,tr)
        for population,ids in [('TRAIN_ALL',tr),('TRAIN_TD_OOF',evf['td_idx'])]:
            for low,high in [(0,.25),(.25,.5),(.5,.75),(.75,1.00001)]:
                mask=(evf['reference_support'][ids]>=low)&(evf['reference_support'][ids]<high);q=ids[mask]
                if not len(q):continue
                calibration_rows.append(dict(fold=fold,population=population,support_low=low,support_high=high,n=len(q),
                   matched_mean=float(br['matched'][q].sum(1).mean()),unmatched_mean=float(br['unmatched'][q].mean()),
                   unmatched_q10=float(np.quantile(br['unmatched'][q],.1)),unmatched_q90=float(np.quantile(br['unmatched'][q],.9)),
                   hard_reject_fraction=float((~br['accepted'][q]).mean()),positive_z_mean=float(np.maximum(evf['z'][q],0).mean()),
                   negative_z_mean=float(np.maximum(-evf['z'][q],0).mean()),real_wrong_feature_delta=float(np.abs(br['features'][q]-bw['features'][q]).mean()),
                   td_oof_mae=evf['stats']['td_oof_mae'],td_oof_mse=evf['stats']['td_oof_mse']))
    r.write_csv(r.RESULTS/'train_calibration.csv',calibration_rows)
    checks['source_unchanged_during_preflight']=all(r.sha_file(r.EXP/name)==digest for name,digest in checked_sources.items())
    result=dict(checked_source_sha256=checked_sources,status='PASS' if all(checks.values()) else 'FAIL',utc=r.utc(),checks={k:bool(v) for k,v in checks.items()},details=details)
    r.write_json(r.RUNTIME/'preflight.json',result)
    for p in [disk,disk2]:p.unlink();p.with_suffix('.pt.sha256').unlink()
    print(json.dumps({'status':result['status'],'checks':result['checks']},ensure_ascii=False,indent=2),flush=True)
    if result['status']!='PASS':raise RuntimeError('preflight scientific/engineering failure')


def all_rows():
    rows=[]
    for g in r.CONFIGS:
        s=json.loads((r.STATE_DIR/f'{g}.json').read_text())
        if s['status']!='COMPLETE':raise RuntimeError('incomplete '+g)
        rows+=s['rows']
    rows.sort(key=lambda row:(r.CONFIGS.index(row['config']),int(row['fold'])))
    if len(rows)!=100 or len({(a['config'],int(a['seed']),int(a['fold'])) for a in rows})!=100:raise RuntimeError('100-fold uniqueness')
    return rows


def counterfactuals(c,o):
    rows=[];patients=[]
    for fold,(train,test) in enumerate(c['folds'],1):
        ck=r.checked_load(torch,r.CKPT_DIR/f'E9_SAGE_FULL_fold{fold:02d}_best.pt');anchor=r.checked_load(torch,r.ANCHOR_DIR/f'anchor_fold{fold:02d}.pt')
        agent,branch,opt,params=r.setup_adaptation(c,fold,'E9_SAGE_FULL',anchor)
        branch.load_state_dict(ck['branch']);agent.ModalFusion.Outputlayer.load_state_dict(ck['output'])
        ev=r.load_evidence(fold,'conditional');cal=ck['calibration'];real=r.bundle('E9_SAGE_FULL',ev,o,c['x'],train,cal)
        wrong=r.bundle('E5_WRONG_SEMANTICS',ev,o,c['x'],train,cal);numeric=r.bundle('E6_DATA_PRIOR',ev,o,c['x'],train,cal)
        noqc=r.bundle('E4_NO_QC',ev,o,c['x'],train,cal);forced=r.bundle('E7_FORCED_ALIGNMENT',ev,o,c['x'],train,cal)
        off={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in real.items()}
        off['effective']=np.zeros_like(real['effective']);off['matched']=np.zeros_like(real['matched']);off['unmatched']=np.ones_like(real['unmatched'])
        perm=np.arange(len(c['x']));perm[test]=np.roll(test,1)
        mismatch=dict(real);mismatch['effective']=real['effective'][perm];mismatch['matched']=real['matched'][perm];mismatch['unmatched']=real['unmatched'][perm]
        views={'REAL':real,'WRONG_SEMANTICS':wrong,'NUMERIC_PRIOR':numeric,'SEMANTIC_OFF':off,'QC_OFF':noqc,
               'FORCED_ALIGNMENT':forced,'PATIENT_EVIDENCE_MISMATCH':mismatch,'ALL_RESIDUAL_OFF':off}
        cam=anchor['cam'].cuda();att=anchor['attention'].cuda();baseline=None;real_res=None;store={}
        for name,b in views.items():
            with torch.no_grad():lp,res,extra=r.adapt_forward(branch,agent.ModalFusion.Outputlayer,torch.tensor(b['effective'],device='cuda'),cam,att,False)
            if name=='REAL':
                baseline=lp.detach().clone();real_res=res.detach().clone()
                if not torch.equal(lp.cpu(),ck['logp']):raise RuntimeError('REAL checkpoint replay differs')
            store[name]=lp.detach().clone()
            rows.append(dict(fold=fold,intervention=name,prediction_flips_vs_real=int((lp[test].argmax(1)!=baseline[test].argmax(1)).sum()),
               mean_abs_logp_delta_vs_real=float((lp[test]-baseline[test]).abs().mean()),
               mean_residual_delta_L2=float((res[test]-real_res[test]).norm(dim=1).mean()),
               matched_test_mean=float(b['matched'][test].sum(1).mean()),unmatched_test_mean=float(b['unmatched'][test].mean()),
               **r.branch_diag(res,extra,cam,test)))
        off_delta=(store['REAL']-store['SEMANTIC_OFF']).abs().mean(1).cpu().numpy()
        qc_median=float(np.median(ev['qc_burden'][train]))
        for idx in test:
            patients.append(dict(fold=fold,subject_index=int(idx),site_bin=int(np.argmax(c['x'][idx,26:46])),
              age_bin=int(np.argmax(c['x'][idx,:26])),qc_burden=float(ev['qc_burden'][idx]),qc_stratum='higher' if ev['qc_burden'][idx]>qc_median else 'lower',
              reference_support=float(ev['reference_support'][idx]),matched_mass=float(real['matched'][idx].sum()),unmatched_mass=float(real['unmatched'][idx]),
              semantic_off_logp_delta=float(off_delta[idx]),residual_cam_ratio=float(real_res[idx].norm()/cam[idx].norm()),
              real_margin=float((store['REAL'][idx,0]-store['REAL'][idx,1]).cpu())))
        del agent,branch,opt,params;torch.cuda.empty_cache()
    return rows,patients


def report():
    signature=r.verify_freeze();rows=all_rows();c=r.context(0);o=r.load_ontology();checks={}
    checks['all_rows_signature']=all(a['signature']==signature for a in rows)
    checks['GPU_assignment']=all(a['config'] in r.GPU_GROUPS[int(a['physical_gpu'])] for a in rows)
    checks['E0_parity']=all(json.loads((r.STATE_DIR/'E0_B0A.json').read_text())['parity'])
    for fold,(train,test) in enumerate(c['folds'],1):
        group_rows=[a for a in rows if a['fold']==fold and a['config']!='E0_B0A']
        checks[f'fold{fold}_shared_head_init_RNG']=len({a['initial_head_sha256'] for a in group_rows})==1 and len({a['training_rng_sha256'] for a in group_rows})==1
        checks[f'fold{fold}_shared_branch_init']=len({a['initial_branch_sha256'] for a in group_rows if a['initial_branch_sha256'] is not None})==1
        for row in [a for a in rows if a['fold']==fold]:
            g=row['config'];path=r.ANCHOR_DIR/f'anchor_fold{fold:02d}.pt' if g=='E0_B0A' else r.CKPT_DIR/f'{g}_fold{fold:02d}_best.pt'
            ck=r.checked_load(torch,path);actual=r.metric_row(ck['logp'],c['y'],test)
            checks[f'{g}_fold{fold}_checkpoint_metrics']=all(abs(actual[k]-row[k])<1e-12 for k in ['acc','hard_auc','spe_td','sen_asd'])
            trace=list(csv.DictReader((r.LOG_DIR/f'{g}_fold{fold:02d}_trace.csv').open()))
            maximum=max(float(t['validation_acc']) for t in trace);first=next(int(t['epoch']) for t in trace if float(t['validation_acc'])==maximum)
            checks[f'{g}_fold{fold}_first_best_ACC']=first==row['best_epoch'] and len(trace)==row['epochs_executed'] and [int(t['epoch']) for t in trace]==list(range(len(trace)))
            if g=='E8_FROZEN_HEAD':
                anchor=r.checked_load(torch,r.ANCHOR_DIR/f'anchor_fold{fold:02d}.pt')
                checks[f'E8_fold{fold}_head_frozen']=all(torch.equal(v,anchor['model']['pln']['Outputlayer.'+k]) for k,v in ck['output'].items())
    if not all(checks.values()):raise RuntimeError('final integrity failure '+str([k for k,v in checks.items() if not v]))
    r.write_json(r.RUNTIME/'final_integrity.json',dict(status='PASS',checks=checks,utc=r.utc()))
    r.write_csv(r.RESULTS/'fold_metrics.csv',rows);r.write_csv(r.RESULTS/'diagnostics.csv',[a for a in rows if a['config']!='E0_B0A'])
    cf,patients=counterfactuals(c,o);r.write_csv(r.RESULTS/'counterfactual.csv',cf);r.write_csv(r.RESULTS/'patient_diagnostics.csv',patients)
    assoc=[]
    for field in ['site_bin','age_bin','qc_stratum']:
        for group in sorted({p[field] for p in patients}):
            subset=[p for p in patients if p[field]==group]
            assoc.append(dict(variable=field,stratum=group,n=len(subset),**{k:float(np.mean([p[k] for p in subset])) for k in ['matched_mass','unmatched_mass','semantic_off_logp_delta','residual_cam_ratio']}))
    r.write_csv(r.RESULTS/'associations.csv',assoc)
    metrics=['acc','hard_auc','spe_td','sen_asd']
    summary={g:{k:dict(mean=float(np.mean([a[k] for a in rows if a['config']==g])),population_sd=float(np.std([a[k] for a in rows if a['config']==g],ddof=0))) for k in metrics} for g in r.CONFIGS}
    e9=[a for a in rows if a['config']=='E9_SAGE_FULL'];pairs={}
    for g in r.CONFIGS[:-1]:
        base=[a for a in rows if a['config']==g];pairs[g]={}
        for k in metrics:
            d=np.array([a[k]-b[k] for a,b in zip(e9,base)])
            pairs[g][k]=dict(delta_pp=float(d.mean()*100),wins=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()),fold_delta_pp=(d*100).tolist())
    p=pairs['E0_B0A'];classification=p['acc']['delta_pp']>=.2-1e-12 and p['hard_auc']['delta_pp']>=.2-1e-12 and p['acc']['wins']>=7 and p['hard_auc']['wins']>=7 and p['sen_asd']['delta_pp']>=-.5
    semantic_comparisons=all(pairs[g]['acc']['delta_pp']>0 and pairs[g]['hard_auc']['delta_pp']>0 for g in ['E1_HEAD_ONLY','E2_NUMERIC_ONLY','E5_WRONG_SEMANTICS','E6_DATA_PRIOR'])
    semantic_response=float(np.mean([a['mean_abs_logp_delta_vs_real'] for a in cf if a['intervention']=='WRONG_SEMANTICS']))>1e-9
    semantics=semantic_comparisons and semantic_response
    reliability=all(pairs[g]['acc']['delta_pp']>0 and pairs[g]['hard_auc']['delta_pp']>0 for g in ['E4_NO_QC','E7_FORCED_ALIGNMENT'])
    decisions=dict(GO_CLASSIFICATION=bool(classification),GO_MEDICAL_SEMANTICS=bool(semantics),GO_RELIABILITY=bool(reliability),
       final='GO_INTERNAL_ONLY' if classification and semantics else 'FINAL NO-GO',
       interpretation='Internal descriptive screening; no statistical significance or independent generalization proof.')
    r.write_json(r.RESULTS/'summary.json',dict(summary=summary,paired=pairs,decision=decisions))
    table=['| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |','|---|---:|---:|---:|---:|']
    for g in r.CONFIGS:table.append('| '+g+' | '+' | '.join(f'{summary[g][k]["mean"]*100:.4f} ± {summary[g][k]["population_sd"]*100:.4f}' for k in metrics)+' |')
    text=['# Experiment 41.1 — SAGE-Align 最终验证','','## 执行状态与纠错边界','',
      '修正协议已完成100/100个唯一正式折（seed=0，10组各10折）。GPU0执行E0/E2/E4/E6/E8共50折，GPU1执行E1/E3/E5/E7/E9共50折；每卡一个worker、CPU线程固定为1。未修改原始源码、数据或历史模型，未提交或推送Git。',
      '', '此前实现的100行结果因E5映射不一致、E2/E6重复、QC方向及恢复状态等缺陷被作废，保存在`.runtime/rejected_protocol_audit_v1`；另有早期10折RNG哈希序列化核验失败记录位于`.runtime/e0_rejected`。它们不计入当前100有效折。旧handoff v1的“100有效折”及墙钟声明被本次纠错撤回。纠错前结果已经被观察，当前冻结是工程修正后的冻结，不能声称盲法预注册；未根据这些ACC/AUC调整学习率、8%上限或选模规则。',
      '', '沿用 **val=test**、历史外层CV之前的RFE/标准化、871人transductive图以及单seed。所有性能只表示历史协议下的内部比较，不是无偏独立泛化证明。',
      '', '## 论文、源码及历史审计','', (r.EXP/'design_audit.md').read_text(),
      '', '## 四模态职责与解剖映射','',
      'PHENO48为26维年龄分箱、20维站点和2维性别；没有连续年龄。ANAT6为CNR/EFC/FBER/FWHM/QI1/SNR，新增分支不从其推导病理或通用质量方向。FUNC10包括EFC/FBER/FWHM/DVARS/outlier/quality/mean FD/num FD/percent FD/GSR，GSR为ghost-to-signal ratio。新增可靠性只使用全局零基列57/58/60/62的DVARS、outlier、mean FD、percent FD，较高值才受抑制。CORR位于全局64–319列，是新增疾病候选证据的来源。原四模态仍完整进入GFR。',
      '', 'Experiment11核验252条FC的两个端点；195/225/232/236四条未知端点保留于GFR，全部新增对照均排除。HO111标签有110个被已观察FC覆盖；稀疏RFE连接不能代表完整连接组。当前候选只表达解剖范围，不把岛叶分成未经核验的前后网络，也不把CAM的4×36潜变量解释成脑区或原始四模态特征。',
      '', '## Prompt全文与实际计算角色','',
      '没有调用GPT-4/API，没有新增CLIP文本编码器；原PEN构造中的RN50缓存仅为复演B0-A随机副作用且完全旁路。下列文本由研究设计据任务草案撰写并核对可访问原文。验证对象是 **Prompt编译后的结构化医学先验**，不是CLIP嵌入对齐或GPT文本生成的价值。改变ROI适用规则会改变实际FC池化权重；只改措辞而保持结构不变会得到相同计算，不能据此声称验证自然语言含义。正负偏离均保留，文献方向仅为带不确定性的解释，不构成逐边标签。']
    for p in json.loads((r.EXP/'prompts.json').read_text())['concepts']:
        text += ['', '### '+p['concept_id']+' '+p['concept_name'],'', '**ASD候选：** '+p['prompt_text'],'','**TD参考：** '+p['td_reference_text'],'',
           '候选ROI：'+'；'.join(p['supported_roi_names'])+'。',
           f'适用FC数：{len(p["supported_edge_indices"])}；DOI：{p["doi"]}；允许进入计算：{p["model_eligible"]}；概念SHA256：`{p["frozen_sha256"]}`。',
           '不支持的推断：'+'；'.join(p['not_supported_claims'])+'。']
    text += ['', '## 正确、错误和数值先验的公平性','',
      'E5采用同一个ROI置换，同时决定错误ROI、由两个真实端点诱导的错误FC、错误粗解剖族和全部三尺度池化权重。置换严格限制在相同实际连接度及皮质/皮质下层内；24,000步仅解剖约束搜索不读取任何患者或标签。候选ROI数、ROI重叠、联合ROI及候选ROI度分布严格保持；FC数量、重叠、联合FC、连接类型和端点覆盖的剩余误差如实列在ontology与下表，不声称完全公平。',
      '', 'P4覆盖全部可观察皮质下候选，其皮质下-皮层身份在类型保持置换下不变；因此E5不能单独否定P4宽泛解剖假设，主要检验P1–P3的对应。没有先构造错误ROI再用无关随机FC替代。',
      '', '| 结构量 | 正确 | 错误 | 差 |','|---|---|---|---|']
    ctrl=o['wrong_control']
    for k in ['roi_counts','edge_counts','roi_union','edge_union','covered_endpoint_counts','edge_type_counts']:
        text.append(f'| {k} | {ctrl["real"][k]} | {ctrl["wrong"][k]} | {ctrl["differences"][k]} |')
    text += ['', '逐概念改变的FC数量：'+str(ctrl['changed_edges_by_concept'])+'。完整重叠矩阵、端点连接度分布误差见ontology。',
      '', 'E2固定随机FC索引槽及8/32列索引层次，完全不读取HO身份或训练标签；E6只用outer-train FC协方差KMeans四原型的连续软亲和力，按训练内Kish有效支持量匹配四槽证据规模。不实施患者Top-K选边，不使用outer-test标签。二者共享4×18接口和全部参数，三尺度均有效，数值组解剖覆盖系数固定为1而不读取HO度数；数值索引层次不是解剖层次，因此E9−E2仍含医学解剖组织与数值分组方式的差异，E5是更直接的医学对应对照。',
      '', '## 条件TD参考、连续校准与拒绝','',
      '每外层折仅在训练TD拟合ridge（alpha=10）；全部训练患者使用固定5折交叉拟合，测试使用完整训练TD。条件项为真实26维年龄分箱、站点、性别及mean FD；E3只有全局TD截距。OOF TD残差MAD用于尺度化并在±4处平滑分支之前截断证据输入；这不是经过临床验证的预测置信区间，也不是FUNCOIN完整全脑协方差模型。',
      '', '每尺度六通道为正z、负z、正向z>1占比、负向z<−1占比、z²/4、|z|；零偏离时全部为零。概念分数为三尺度平均的0.5×平均|z|+0.5×|z|>1占比，训练TD OOF及所有推理都使用同一函数。匹配强度取同一训练TD分数的经验CDF。a为参考支持×单向FUNC支持×解剖覆盖；q=a×CDF，拒绝u=∏k(1−qk)，各概念质量=(1−u)qk/Σq。所有质量严格守恒。参考支持<0.1、条件层无TD或QC负担超过训练TD99.5分位时严格拒绝；这也是冻结规则，不按测试ACC调整。E4仅取消QC，E7仅把可用槽质量归一化并取消拒绝。',
      '', '训练分层、正负偏离、OOF残差、匹配分位数和正确/错误映射敏感性见`train_calibration.csv`；硬拒绝与连续拒绝质量是不同量。FUNC支持是工程可靠性条件，不是真实扫描质量金标准。',
      '', '## SAGE-Align与训练协议','',
      '有效证据vik=massik×eik；共享无bias 18→8 GELU、CAM均值36→4 sigmoid查询调节、8→36→144线性变换产生raw残差。所有槽为零时共享变换严格为零；CAM没有独立产生残差的通路。最后线性权重全零初始化；r_eff=raw×[c tanh(||raw||/c)/||raw||]，c=0.08||CAM||。在原OutputLayer之前加入残差，未新增独立分类器。',
      '', 'GFR/InputLayer/Encoder/FeedForward权重冻结并处于eval状态，缓存每折B0的CAM等价于其确定性前向。OutputLayer架构不变，E1–E7/E9从相同B0权重适配；E8仅冻结这些权重，其训练Dropout仍与E9相同。原加权NLL加统一1e−3残差相对能量正则，无其他新损失。新阶段Adam、两组lr均3.8e−4，weight_decay=0；B0原双Adam的lr=0.0038、reg=0.11保持。450上限、wait>200停止；只在ACC严格提高时保存best，平局保留首次，epoch0表示第一次更新之后。',
      '', '所有适配组同折分支初始化种子411000+fold，Dropout训练随机流412000+fold；此协议与B0跨折seed0连续随机流明确区分。best与latest保存Adam及Python/NumPy/CPU/CUDA RNG；B0另存下一折连续RNG，每25epoch覆盖恢复点，完整折跳过。预检已经逐值验证基线双Adam、适配Adam、磁盘恢复和下一折初始化。',
      '', '| 组别 | 唯一改动 | 分支参数 | 可训练总参数 |','|---|---|---:|---:|']
    descriptions=['B0-A原样重训','仅原OutputLayer','固定数值层次证据','全局TD参考','关闭FUNC支持','共同置换的错误解剖对应','训练折软数值先验','强制可用概念对齐','仅分支训练、OutputLayer冻结','完整结构化医学先验+联合适配']
    for i,g in enumerate(r.CONFIGS):
        row=next(a for a in rows if a['config']==g)
        text.append(f'| {g} | {descriptions[i]} | {row["branch_parameters"]} | {row["trainable_parameters"]} |')
    text += ['', 'E2–E9均18维/槽、4槽（共72个输入值）、5,760分支参数；|z|与正负z存在冗余，训练有效证据矩阵的数值秩保存在diagnostics，不将72个值声称为72个独立维度。E2–E9均5,760分支参数，联合适配加OutputLayer5,582参数共11,342；E8仅5,760。零初始化使第一步仅最后残差投影取得非零分支梯度，随后所有分支路径均实测有梯度；不能以第一步部分零梯度伪称参数无效。',
      '', '## B0-A精确复现','', '10/10折四指标、best epoch、执行epoch、初始主干SHA和入口RNG SHA逐项与Experiment32匹配。',
      '', '| 折 | best epoch | 执行epoch | ACC | Hard-AUC | SPE-TD | SEN-ASD |','|---|---:|---:|---:|---:|---:|---:|']
    for a in [a for a in rows if a['config']=='E0_B0A']:
        text.append(f'| {a["fold"]} | {a["best_epoch"]} | {a["epochs_executed"]} | '+' | '.join(f'{a[k]*100:.4f}' for k in metrics)+' |')
    text += ['', '## 四项正式结果','', '单位%；十折均值±population SD。Hard-AUC沿用硬预测one-hot口径，ASD=0、TD=1。','']+table
    text += ['', '## E9核心配对比较','', '单位百分点；同折W/T/L，不将重叠训练折当作独立外部样本。','',
       '| E9减对照 | ΔACC / W-T-L | ΔHard-AUC / W-T-L | ΔSPE | ΔSEN |','|---|---|---|---:|---:|']
    for g,pair in pairs.items():
        aa=pair['acc'];uu=pair['hard_auc']
        text.append(f'| {g} | {aa["delta_pp"]:+.4f} / {aa["wins"]}-{aa["ties"]}-{aa["losses"]} | {uu["delta_pp"]:+.4f} / {uu["wins"]}-{uu["ties"]}-{uu["losses"]} | {pair["spe_td"]["delta_pp"]:+.4f} | {pair["sen_asd"]["delta_pp"]:+.4f} |')
    text += ['', '逐折全部四指标保存在`fold_metrics.csv`，完整逐折配对差见`summary.json`。','', '## 优化、残差和固定checkpoint反事实','',
       '| 组别 | best epoch=0折数 | best epoch中位数 | 未匹配质量 | 有效残差/CAM | raw残差/8%上限 | 饱和比例 |','|---|---:|---:|---:|---:|---:|---:|']
    for g in r.CONFIGS[1:]:
        a=[a for a in rows if a['config']==g]
        text.append(f'| {g} | {sum(t["best_epoch"]==0 for t in a)}/10 | {np.median([t["best_epoch"] for t in a]):.1f} | {np.mean([t["unmatched_test_mean"] for t in a]):.6f} | {np.mean([t["residual_cam_ratio"] for t in a]):.6g} | {np.mean([t["raw_cap_ratio"] for t in a]):.6g} | {np.mean([t["saturation_fraction"] for t in a]):.6g} |')
    n0=sum(a['best_epoch']==0 for a in e9)
    text += ['', f'E9有{n0}/10折best停留在首次更新。若该数字很高，只能判断后续学习未在历史选模口径改善，不能宣称这是充分学习后的最优医学模块。每epoch训练NLL、残差能量、验证ACC、预测翻转和raw/effective残差都保存在`.runtime/logs/*_trace.csv`；未事后改选连续指标。',
       '', '| 固定E9干预 | 翻转数/871 | 平均绝对log-prob变化 | 平均残差变化L2 | 匹配质量 | 拒绝质量 |','|---|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(a['intervention'] for a in cf):
        a=[a for a in cf if a['intervention']==name]
        text.append(f'| {name} | {sum(t["prediction_flips_vs_real"] for t in a)}/871 | {np.mean([t["mean_abs_logp_delta_vs_real"] for t in a]):.8g} | {np.mean([t["mean_residual_delta_L2"] for t in a]):.8g} | {np.mean([t["matched_test_mean"] for t in a]):.6f} | {np.mean([t["unmatched_test_mean"] for t in a]):.6f} |')
    text += ['', 'REAL逐值重放best预测；其余干预固定E9参数及训练校准CDF，不重新拟合或选模。SEMANTIC_OFF与ALL_RESIDUAL_OFF在单分支结构中应相同；它们保留共同适配后的分类头。非零连续响应说明使用了证据通路，不能替代E5/E6独立重训对照，也不证明临床因果。',
       '', '站点20类、年龄26分箱和按各折训练QC中位数定义的高低层机制关联见`associations.csv`；患者级机制记录见`patient_diagnostics.csv`。年龄分箱不冒充连续岁数；站点/QC关联只能排查混杂，不能证明无混杂或外部泛化。',
       '', '## GO/NO-GO与创新性结论','',
       '冻结内部筛选门槛：E9−E0 ACC和Hard-AUC均≥0.2pp、二者至少7/10折严格胜，SEN下降不超过0.5pp；医学必要性要求E9同时优于E1/E2/E5/E6并有真实FC驱动的决策响应。可靠性单列要求优于E4/E7。阈值不是统计显著性或跨中心泛化证据。','',
       '```json',json.dumps(decisions,ensure_ascii=False,indent=2),'```','',
       '**'+decisions['final']+'。** '+('只有内部候选证据，仍需严格独立验证。' if classification and semantics else '没有同时满足分类增益与医学语义必要性，不能作为MICCAI2027第一创新点的已成立证据。结束此SAGE-Align/UCSMA医学语义对齐路线，不继续调参，不提出实验41.2。'),
       '', '分类头微调、数值规范偏离、多尺度聚合、医学适用范围和FUNC可靠性是不同贡献；本矩阵只能按相应对照讨论，不能合并称“多模态对齐有效”。P4不变及E5剩余结构误差限制必要性检验强度；当前负结果也不证明相关脑区没有生物学价值。',
       '', '未来若评价任何其他独立方法，应在外层训练内完成预处理/RFE/规范建模/选模，独立验证或站点留出且测试标签不参与选择，明确inductive/transductive部署条件，并使用独立队列及重复随机种子评估不确定性。此处只说明证据标准，不自动启动新实验。',
       '', '## 文件、恢复和交付','',
       '正式结果、诊断与报告均已写到用户指定目录；所有正式checkpoint含恢复所需Adam和RNG，100折载荷SHA及指标重算通过。原始六文件、数据、Prompt、ROI映射、缓存及checkpoint哈希见manifest与`.sha256`。handoff只幂等追加，明确更正旧v1，保留全部历史内容。']
    (r.RESULTS/'report.md').write_text('\n'.join(text)+'\n')
    m=json.loads(r.MANIFEST.read_text());m.update(status='COMPLETE_VALID_100_WITH_DISCLOSURES',completed_folds=100,summary=summary,paired=pairs,
       decision=decisions,e0_parity={'pass':True,'matched_folds':10},completed_utc=r.utc(),
       result_sha256={p.name:r.sha_file(p) for p in r.RESULTS.iterdir() if p.is_file()},
       checkpoint_sha256={str(p.relative_to(r.EXP)):r.sha_file(p) for p in list(r.ANCHOR_DIR.glob('*.pt'))+list(r.CKPT_DIR.glob('*_best.pt'))},
       corrected_wall_seconds=(__import__('datetime').datetime.now(__import__('datetime').timezone.utc)-__import__('datetime').datetime.fromisoformat(m['corrected_start_utc'])).total_seconds(),
       total_wall_seconds=(__import__('datetime').datetime.now(__import__('datetime').timezone.utc)-__import__('datetime').datetime.fromisoformat(m['wall_start_utc'])).total_seconds(),
       workers={g:{k:v for k,v in json.loads((r.STATE_DIR/f'{g}.json').read_text()).items() if k!='rows'} for g in r.CONFIGS})
    r.write_json(r.MANIFEST,m);print('REPORT_COMPLETE_VALID_100',decisions,flush=True)


def finalize():
    import fcntl
    m=json.loads(r.MANIFEST.read_text())
    if m['status']!='COMPLETE_VALID_100_WITH_DISCLOSURES':raise RuntimeError('not fully complete')
    marker='<!-- experiment41_1_sage_align_final_corrected_v2 -->';end='<!-- /experiment41_1_sage_align_final_corrected_v2 -->'
    summary=m['summary'];pairs=m['paired'];e9=[a for a in all_rows() if a['config']=='E9_SAGE_FULL']
    lines=['## Experiment 41.1 — SAGE-Align Final','',marker,'',
       '本更正段取代此前v1的有效折及墙钟声明；历史内容不删除。v1的100行结果因E5对应不一致、E2/E6重复、QC方向和恢复/预检不足而作废，隔离于`.runtime/rejected_protocol_audit_v1`。当前修正协议100/100有效唯一折完成；纠错前结果已观察，不声称盲法预注册。',
       '',f'完成UTC：{m["completed_utc"]}。GPU0=E0/E2/E4/E6/E8共50折，GPU1=E1/E3/E5/E7/E9共50折，每卡一个worker；E0四指标、best/执行epoch、初始模型和入口RNG逐折10/10精确复现E32 B0-A。',
       '', '| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |','|---|---:|---:|---:|---:|']
    for g in r.CONFIGS:lines.append('| '+g+' | '+' | '.join(f'{summary[g][k]["mean"]*100:.4f} ± {summary[g][k]["population_sd"]*100:.4f}' for k in ['acc','hard_auc','spe_td','sen_asd'])+' |')
    lines+=['','E9核心配对差（pp，ACC/Hard-AUC）：'+'；'.join(f'{g} {pairs[g]["acc"]["delta_pp"]:+.4f}/{pairs[g]["hard_auc"]["delta_pp"]:+.4f}' for g in r.CONFIGS[:-1])+'。',
       '',f'E9未匹配质量={np.mean([a["unmatched_test_mean"] for a in e9]):.6f}，有效残差/CAM={np.mean([a["residual_cam_ratio"] for a in e9]):.8f}，best epoch0={sum(a["best_epoch"]==0 for a in e9)}/10。',
       '', '主要修复：无bias证据必经分支5,760参数；同一TD OOF经验CDF；单向FUNC质量及严格拒绝；E5共同ROI/FC/族置换并公开匹配误差；E2固定数值与E6训练软先验区分；轻量分类头对照；能量正则、零残差预检及Adam/RNG精确恢复。P4因类型保持而不变，不能称完全公平；连续语义响应不等于独立疾病收益。',
       '',f'最终科研结论：{json.dumps(m["decision"],ensure_ascii=False)}。'+('只允许后续严格验证的内部候选。' if m['decision']['final']=='GO_INTERNAL_ONLY' else '永久停止当前SAGE-Align/UCSMA医学语义对齐路线；无足够证据作为MICCAI2027第一创新点，不提出41.2。'),
       '', f'总墙钟={m["total_wall_seconds"]:.2f}s；修正阶段墙钟={m["corrected_wall_seconds"]:.2f}s（并行任务时间未求和冒充墙钟）。结果目录：`{r.EXP}`。',
       '', '哈希：'+'；'.join(f'{p} `{r.sha_file(r.EXP/p)}`' for p in ['run.py','sage_align.py','integration.py','prompts.json','ontology.json','results/report.md'])+'。原始六源码、数据与100个正式best/anchor哈希见manifest。',
       '', '沿用val=test、预CV RFE/标准化及871人transductive图；单seed内部比较，不是无偏独立泛化或临床证明。', '',end,'']
    content='\n'.join(lines);path=r.ROOT/'handoff.md'
    with path.open('a+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);old=f.read()
        if marker not in old:f.seek(0,2);f.write('\n'+content);f.flush();os.fsync(f.fileno())
        elif end not in old:raise RuntimeError('partial handoff marker')
        f.seek(0);after=f.read()
    if not after.startswith(old) or after.count(marker)!=1 or end not in after:raise RuntimeError('handoff append verification')
    m['handoff']={'status':'APPENDED_VERIFIED' if marker not in old else 'ALREADY_PRESENT_VERIFIED','path':str(path),'utc':r.utc(),'old_prefix_sha256':hashlib.sha256(old.encode()).hexdigest(),'preserved_old_prefix':True}
    r.write_json(r.MANIFEST,m);print(json.dumps(m['handoff'],ensure_ascii=False),flush=True)

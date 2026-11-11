"""Independent correctness, provenance, checkpoint readback and mechanism audits."""
from __future__ import annotations
import copy, csv, gc, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import run as R
from memory import Memory, identity_split, audit_bank, cpu_bank, device_bank
from path_adapter import BASE, E773, HERE, REPO, sha_file, verify_relocation
from qcrl_pm import base_forward

def read(path):
    return json.loads(Path(path).read_text())

def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def add_controlled_pairs(memory,h,raw,support):
    """Correctness fixture only; never used by any formal task."""
    a = support[memory.labels[support]==0][:3]
    t = support[memory.labels[support]==1][:3]
    for i,(ai,ti) in enumerate(zip(a,t)):
        ai,ti=int(ai),int(ti)
        ids=(ai,ti);key=f'{memory.fold}:{ai}:{ti}'
        memory.pairs[key]=dict(pair_id=key,ids=ids,hard_id=ai,opposite_id=ti,source=1,
            first_epoch=16,last_epoch=16,frequency=1,cam=h[list(ids)].copy(),raw=raw[list(ids)].copy(),
            delta_cam=h[ai]-h[ti],delta_raw=raw[ai]-raw[ti],quality=.9,similarity=.8,reliability=.8)
    memory.epoch=16

def preflight():
    start=time.time()
    print('PREFLIGHT: source relocation and inherited fold states',flush=True)
    protected=verify_relocation()
    R.save_json(HERE/'results/protected_before.json',protected)
    import pandas as pd
    frame=pd.read_csv(REPO/'data/ABIDE/processed_standard_data.csv')
    mapping=np.load(REPO/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
    assert list(frame.columns[:-1])==[str(v) for k in ('PHENO','ANAT','FUNC','Correlation') for v in mapping[k]]
    raw=frame.iloc[:,64:320].to_numpy(np.float32)
    labels=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert raw.shape==(871,256) and np.bincount(labels).tolist()==[403,468]
    inherited=[]
    for fold in range(1,11):
        s,module,optimizer,memory,initial=R.initialize(fold,'E77.4-G1')
        assert np.array_equal(s['parts'][3].cpu().numpy(),raw)
        assert np.array_equal(s['y'].cpu().numpy(),labels)
        assert set(memory.allowed).isdisjoint(s['test_np'])
        assert np.all(memory.labels[s['test_np']]==-1)
        assert all(np.bincount(labels[memory.split==b],minlength=2).min()>0 for b in range(3))
        ref=read(BASE/'.runtime/tasks'/f'{R.BASE_GROUP}_f{fold:02d}.json')
        assert ref['identity']==initial['identity']
        inherited.append(dict(fold=fold,initial=initial,allowed=len(memory.allowed),
            query_counts=[np.bincount(labels[memory.split==b],minlength=2).tolist() for b in range(3)],
            warmup_exact=True,raw_corr_exact=True))
        del s,module,optimizer,memory
        gc.collect()
    R.save_json(HERE/'results/initial_inheritance.json',inherited)
    print('PREFLIGHT: exact base forward/backward/optimizer and gradient isolation',flush=True)
    s,module,optimizer,memory,initial=R.initialize(1,'E77.4-G1')
    r=R.runtime();initial_cp=R.snapshot(s,module,optimizer,memory)
    # Direct historical execution, then new runner execution from the identical state.
    direct=R.base()._one_update(s,r,R.BASE_GROUP)
    with torch.no_grad():
        direct_eval=R.base().forward(s,R.BASE_GROUP,training=False)['logp'].detach().clone()
    R.restore(s,module,optimizer,memory,initial_cp)
    z=R.step(s,module,optimizer,memory,'E77.4-G1',1)
    assert R.digest(R.public_bundle(s))==R.digest(direct[2])
    assert z['losses']['total']==direct[1]
    assert z['rng']['rng_after_train']==direct[3]
    assert torch.equal(z['base_logp'],direct_eval)
    after_g1=R.snapshot(s,module,optimizer,memory)
    R.restore(s,module,optimizer,memory,initial_cp)
    g2=R.step(s,module,optimizer,memory,'E77.4-G2',1)
    assert R.digest(R.snapshot(s,module,optimizer,memory))==R.digest(after_g1)
    # No test-accuracy gate: all checks below concern implementation and identities.
    print('PREFLIGHT: third-patient relations, signed evidence and semantic gradients',flush=True)
    with torch.no_grad():
        out,h=base_forward(R.base(),s,False)
        logp=out['logp'].detach()
    del out
    query=np.flatnonzero(memory.split==0)
    support=np.setdiff1d(memory.allowed,query)
    hc=h.cpu().numpy();rawc=s['parts'][3].cpu().numpy()
    control=copy.deepcopy(memory)
    add_controlled_pairs(control,hc,rawc,support)
    bank=control.bank(hc,rawc,support,h.device)
    isolation=audit_bank(bank,query,s['test_np'],labels)
    qi=torch.as_tensor(query,device=h.device)
    evidence=module(h[qi],logp[qi],s['parts'][3][qi],qi,bank,hard=True)
    total,losses=module.losses(evidence,s['y'][qi])
    gradients={}
    for name in ('pair_key','query_key','corr_value','cam_relation','pair_difference','text_projection','confusion_gate'):
        params=tuple(getattr(module,name).parameters())
        grad=torch.autograd.grad(losses['qcls'],params,allow_unused=True,retain_graph=True)
        norm=float(torch.sqrt(sum((v.square().sum() for v in grad if v is not None),torch.zeros((),device=h.device))))
        assert math.isfinite(norm) and norm>0,(name,norm)
        gradients[name+'_qcls']=norm
    grad=torch.autograd.grad(losses['qrelation'],tuple(module.corr_value.parameters())+tuple(module.pair_key.parameters()),retain_graph=True)
    assert all(float(v.norm())>0 for v in grad)
    gradients['qrelation_corr']=float(grad[0].norm());gradients['qrelation_key']=float(grad[1].norm())
    public=tuple(p for mm in (s['model'],s['cam']) for p in mm.parameters() if p.requires_grad)
    assert all(g is None for g in torch.autograd.grad(total,public,allow_unused=True,retain_graph=True))
    gradient_before=R.digest([p.grad for p in public])
    optimizer.zero_grad(set_to_none=True);total.backward()
    assert R.digest([p.grad for p in public])==gradient_before
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    # Feature copies at different Query IDs make the orientation check exact
    # without using either endpoint identity as a supervised query.
    one={k:(v[:1] if k.startswith('pair_') or k in ('delta_cam','delta_raw') else v) for k,v in bank.items()}
    a,t=one['pair_ids'][0].tolist();sq=qi[:2]
    e=module(torch.stack((h[a],h[t])),logp[sq],torch.stack((s['parts'][3][a],s['parts'][3][t])),sq,one,hard=True)
    assert e['confusion'][0]>0 and e['confusion'][1]<0
    assert not set(sq.cpu().tolist()) & {a,t}
    # Empty bank and all-invalid retrieval remain finite; no relation loss exists.
    empty=memory.bank(hc,rawc,support,h.device)
    ee=module(h[qi],logp[qi],s['parts'][3][qi],qi,empty,hard=True)
    _,el=module.losses(ee,s['y'][qi])
    assert el['qrelation'].item()==0 and torch.isfinite(ee['logp']).all()
    off=module(h[qi],logp[qi],s['parts'][3][qi],qi,bank,hard=True,correction_off=True)
    assert torch.equal(off['logp'],logp[qi])
    # Labels of Query or test cannot change a Support bank or retrieval.
    changed=labels.copy();changed[query]=1-changed[query];changed[s['test_np']]=1-changed[s['test_np']]
    clean=Memory(R.CFG['method'],memory.allowed,changed,memory.split,1)
    clean.epoch=memory.epoch
    assert R.digest(cpu_bank(clean.bank(hc,rawc,support,h.device)))==R.digest(cpu_bank(empty))
    # Persistent relation identity, support masking, EMA and expiration.
    pm=Memory(R.CFG['method'],memory.allowed,labels,memory.split,1)
    pred=labels.copy();hard_ids=[int(support[labels[support]==0][0]),int(support[labels[support]==1][0])]
    pred[hard_ids]=1-pred[hard_ids]
    for epoch in (14,15,16):
        pm.observe_predictions(pred,epoch)
        pm.update_relations(hc,rawc,support,epoch==16)
    assert len(pm.pairs)>0
    key=next(iter(pm.pairs));previous=copy.deepcopy(pm.pairs[key])
    pm.observe_predictions(pred,17);pm.update_relations(hc+.01,rawc,support,True)
    rec=pm.pairs[key]
    assert rec['first_epoch']==16 and rec['last_epoch']==17 and rec['frequency']==2
    assert np.array_equal(rec['cam'],.8*previous['cam']+.2*(hc+.01)[list(rec['ids'])])
    restricted=np.setdiff1d(support,[rec['ids'][0]])
    filtered=pm.bank(hc,rawc,restricted,h.device)
    assert rec['ids'][0] not in filtered['pair_ids'].cpu().numpy()
    pm.observe_predictions(labels,38);pm.update_relations(hc,rawc,np.array([],dtype=np.int64),True)
    assert not pm.pairs
    isolated=copy.deepcopy(control);isolated.pairs[key if key in isolated.pairs else next(iter(isolated.pairs))]['frequency']+=1
    assert R.digest(isolated.state_dict())!=R.digest(control.state_dict())
    print('PREFLIGHT: full checkpoint restore and exact next-step replay',flush=True)
    R.restore(s,module,optimizer,memory,after_g1)
    path=HERE/'runtime/preflight_resume.pt';R.save_torch(path,R.snapshot(s,module,optimizer,memory))
    before=r.rng_sha(r.pack_rng())
    with torch.no_grad():
        ob,hb=base_forward(R.base(),s,False)
        eb=module(hb[qi],ob['logp'][qi],s['parts'][3][qi],qi,device_bank(memory.eval_bank,h.device),hard=False)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    R.restore(s,module,optimizer,memory,saved)
    with torch.no_grad():
        oa,ha=base_forward(R.base(),s,False)
        ea=module(ha[qi],oa['logp'][qi],s['parts'][3][qi],qi,device_bank(memory.eval_bank,h.device),hard=False)
    assert torch.equal(eb['logp'],ea['logp']) and r.rng_sha(r.pack_rng())==before
    R.restore(s,module,optimizer,memory,saved)
    first=R.step(s,module,optimizer,memory,'E77.4-G2',2)
    after=R.digest(R.snapshot(s,module,optimizer,memory))
    R.restore(s,module,optimizer,memory,torch.load(path,map_location='cpu',weights_only=False))
    second=R.step(s,module,optimizer,memory,'E77.4-G2',2)
    assert R.digest(R.snapshot(s,module,optimizer,memory))==after
    assert first['losses']==second['losses'] and first['rng']==second['rng']
    path.unlink()
    result=dict(status='PASS',source_sha=R.source_sha(),source_map=R.source_map(),
        protected_files=len(protected),initialization_folds=10,public_single_step_exact=True,
        base_output_exact=True,base_optimizer_exact=True,public_rng_exact=True,
        public_new_gradient_strictly_zero=True,query_support_isolation=isolation,
        query_labels_only_losses=True,test_labels_excluded=True,signed_direction=True,
        no_endpoint_self_supervision=True,empty_memory_safe=True,persistent_ema_and_expiry=True,
        warmup_g1_g2_one_step_exact=True,all_folds_warmup_parity_deferred_to_formal=True,
        worker_memory_isolated=True,checkpoint_readback_exact=True,next_step_replay_exact=True,
        classification_gradients=gradients,numerical_losses={k:float(v.detach()) for k,v in losses.items()},
        coefficients_frozen_without_test_tuning=True,wall_seconds=time.time()-start)
    R.save_json(HERE/'results/preflight.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def ratio(a,b):
    return float(a/b) if b else None

def moments(values):
    x=np.asarray(values,dtype=float)
    return dict(mean=float(x.mean()),sd=float(x.std(ddof=0)),n=len(x))

def table(headers,rows):
    cell=lambda v:str(v).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(cell,row))+' |' for row in rows])

def equal_metrics(observed,expected):
    for k,v in expected.items():
        assert observed[k]==v,(k,observed[k],v)

def as_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

def training_logs(path,task):
    """Retain explicit evidence of damaged JSONL; never invent missing records."""
    rows=[];invalid=[]
    for i,line in enumerate(path.read_text().splitlines(),1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            invalid.append(dict(line=i,fragment=line,error=str(e)))
    epochs=[v['epoch'] for v in rows]
    assert len(set(epochs))==len(epochs) and epochs==sorted(epochs)
    expected=set(range(11,task['epochs_executed']+1))
    assert set(epochs)<=expected
    integrity=dict(path=str(path),sha256=sha_file(path),expected_epochs=len(expected),
        observed_epochs=len(rows),missing_epochs=sorted(expected-set(epochs)),invalid_fragments=invalid)
    return rows,integrity

def audit_training(task,logs,ref):
    trace=task['trace']
    assert len(trace)==task['epochs_executed']-10
    assert [v['epoch'] for v in trace]==list(range(11,task['epochs_executed']+1))
    trace_lookup={v['epoch']:v for v in trace}
    reference_rng=0;reference_loss=0
    for tr in trace:
        old=ref['rng'].get(tr['epoch'])
        assert tr['rng_after_memory']==tr['rng_after_eval']
        if old:
            assert [tr[k] for k in ('rng_before_train','rng_after_train','rng_after_eval')]==[old[k] for k in ('rng_before','rng_after_train','rng_after_eval')]
            reference_rng+=1
    for row in logs:
        tr=trace_lookup[row['epoch']]
        assert row['public_sha']==tr['public_sha']
        assert row['gradients']['public_new_gradient']==0
        assert all(row['isolation'][k] for k in ('identity_isolation','test_exclusion','pair_orientation','prototype_members'))
        rr=row['rng']
        assert all(rr[k]==tr[k] for k in rr)
        assert rr['rng_after_memory']==rr['rng_after_eval']
        if row['epoch'] in ref['log']:
            assert row['losses']['total']==ref['log'][row['epoch']]['loss']
            reference_loss+=1
    checks={r['epoch']:r for r in task['reference_checks']}
    assert set(checks)==set(e for e in ref['states'] if e<=task['epochs_executed'])
    assert all(r['base_model_optimizer_exact'] for r in checks.values())
    active=[v for v in logs if v['query']['valid_relation']]
    q=sum(v['query']['count'] for v in logs)
    errors=sum(v['query']['base_errors'] for v in logs)
    valid=sum(v['query']['valid_relation'] for v in logs)
    corrected=sum(v['query']['corrected'] for v in logs)
    harmed=sum(v['query']['harmed'] for v in logs)
    out=dict(query_exposures=q,query_asd_exposures=sum(v['query']['asd'] for v in logs),
        query_td_exposures=sum(v['query']['td'] for v in logs),query_base_error_exposures=errors,
        query_corrected_exposures=corrected,query_harmed_exposures=harmed,
        query_error_correction_rate=ratio(corrected,errors),query_correct_damage_rate=ratio(harmed,q-errors),
        query_relation_exposures=valid,query_relation_direction_correct=sum(v['query']['relation_direction_correct'] for v in logs),
        query_relation_direction_accuracy=ratio(sum(v['query']['relation_direction_correct'] for v in logs),valid),
        identity_isolation_observed_epochs_pass=len(logs),reference_rng_epochs=reference_rng,
        complete_public_trace_epochs=len(trace),observed_mechanism_log_epochs=len(logs),missing_mechanism_log_epochs=len(trace)-len(logs),
        reference_sparse_loss_epochs=reference_loss,reference_state_optimizer_epochs=len(checks),
        epochs_beyond_historical_reference=sum(v['epoch']>max(ref['rng']) for v in trace),
        active_relation_epochs=len(active),empty_relation_epochs=len(logs)-len(active),
        new_pair_events=sum(v['memory']['new_pairs'] for v in logs),
        ema_update_events=sum(v['memory']['updated_pairs'] for v in logs),
        expired_pair_events=sum(v['memory']['expired_pairs'] for v in logs),
        peak_stable_fp=max(v['memory']['stable_fp'] for v in logs),
        peak_stable_fn=max(v['memory']['stable_fn'] for v in logs),
        max_stored_pairs=max(v['memory']['retained_pairs'] for v in logs),
        relation_loss_first20=float(np.mean([v['losses']['qrelation'] for v in active[:20]])) if active else None,
        relation_loss_last20=float(np.mean([v['losses']['qrelation'] for v in active[-20:]])) if active else None,
        relation_loss_near_theoretical_floor_rate=ratio(sum(v['losses']['qrelation']<=float(F.softplus(torch.tensor(-2.)))+.001 for v in active),len(active)),
        protect_loss_mean=float(np.mean([v['losses']['protect'] for v in logs])),
        weighted_protect_over_qcls=float(np.mean([.03*v['losses']['protect']/max(v['losses']['qcls'],1e-12) for v in logs])))
    for name in ('cam_typical','cam_relation','corr_value','query_key','pair_key','pair_difference','text_projection','typical_gate','confusion_gate'):
        rows=active if name in ('cam_relation','query_key','pair_key','pair_difference','confusion_gate') else logs
        out[name+'_zero_gradient_rate']=ratio(sum(v['gradients'][name]==0 for v in rows),len(rows))
        out[name+'_mean_gradient']=float(np.mean([v['gradients'][name] for v in rows])) if rows else None
    return out

def audit_memory(memory,bank,h,raw,test,labels):
    audit_bank(bank,test,test,labels)
    assert np.array_equal(memory.allowed,np.sort(bank['support'].cpu().numpy()))
    assert np.all(memory.labels[test]==-1) and np.all(memory.split[test]==-1)
    assert np.all(memory.history[:,test]==-1) and np.all(memory.error_count[test]==0)
    rebuilt=memory.bank(h.cpu().numpy(),raw.cpu().numpy(),memory.allowed,h.device)
    assert R.digest(cpu_bank(rebuilt))==R.digest(cpu_bank(bank)), 'Saved bank did not reconstruct exactly'
    for m in range(4):
        for cls in range(2):
            groups=bank['members'][(m*2+cls)*4:(m*2+cls+1)*4]
            assert sorted(i for g in groups for i in g)==memory.allowed[memory.labels[memory.allowed]==cls].tolist()
    for cls in range(2):
        groups=bank['raw_members'][cls*4:(cls+1)*4]
        assert sorted(i for g in groups for i in g)==memory.allowed[memory.labels[memory.allowed]==cls].tolist()
    hard_counts={};opposite_counts={}
    for key,p in memory.pairs.items():
        a,t=p['ids']
        assert key==p['pair_id']==f'{memory.fold}:{a}:{t}'
        assert a in memory.allowed and t in memory.allowed and labels[a]==0 and labels[t]==1
        assert memory.epoch-p['last_epoch']<=20 and p['first_epoch']<=p['last_epoch']<=memory.epoch
        assert 1<=p['frequency']<=p['last_epoch']-p['first_epoch']+1
        assert labels[p['hard_id']]==(1 if p['source']==0 else 0)
        assert p['hard_id'] in memory.pool[p['source']]
        assert np.array_equal(p['delta_cam'],p['cam'][0]-p['cam'][1])
        assert np.array_equal(p['delta_raw'],p['raw'][0]-p['raw'][1])
        hard_counts[p['hard_id']]=hard_counts.get(p['hard_id'],0)+1
        opposite_counts[p['opposite_id']]=opposite_counts.get(p['opposite_id'],0)+1
    assert max(hard_counts.values(),default=0)<=2 and max(opposite_counts.values(),default=0)<=8
    assert all(len(set(pool))==len(pool)<=128 for pool in memory.pool.values())
    return rebuilt

def checkpoint_mechanisms(group,fold,memory,bank,z,y,base_pred,final_pred):
    a=lambda key:as_numpy(z[key])
    y=as_numpy(y);et,ec,delta=(a(k) for k in ('typical','confusion','delta'))
    valid=a('has_pairs');selected=a('selected_pairs');chosen=set(selected[selected>=0].tolist())
    counts=as_numpy(bank['counts']);raw_counts=as_numpy(bank['raw_counts'])
    collapse=[]
    for centers in list(as_numpy(bank['cam']).reshape(8,4,36))+list(as_numpy(bank['raw'])):
        norm=centers/np.maximum(np.linalg.norm(centers,axis=-1,keepdims=True),1e-8)
        sim=norm@norm.T
        collapse.extend((sim[np.triu_indices(len(centers),1)]>.98).tolist())
    records=list(memory.pairs.values())
    pairs=as_numpy(bank['pair_ids']);dr=as_numpy(bank['delta_raw'])
    perclass={}
    for cls,name in ((0,'asd'),(1,'td')):
        use=y==cls
        perclass[name+'_delta_mean']=float(delta[use].mean())
        perclass[name+'_confusion_positive']=int(((ec>0)&use).sum())
        perclass[name+'_confusion_negative']=int(((ec<0)&use).sum())
    row=dict(group=group,fold=fold,cam_asd_prototypes=int((counts[:,0]>0).sum()),
        cam_td_prototypes=int((counts[:,1]>0).sum()),raw_asd_prototypes=int((raw_counts[0]>0).sum()),
        raw_td_prototypes=int((raw_counts[1]>0).sum()),prototype_support_coverage=1.,
        prototype_pair_cosine_gt_098_rate=float(np.mean(collapse)),
        prototype_retrieved_unique=len(np.unique(a('prototype_selection'))),
        prototype_total_slots=40,cam_member_consistency=True,raw_member_consistency=True,
        stored_pairs=len(records),effective_pairs=len(pairs),pool_fp=len(memory.pool[0]),pool_fn=len(memory.pool[1]),
        unique_pair_hard_patients=len({p['hard_id'] for p in records}),
        pair_matching_quality_mean=float(np.mean([p['quality'] for p in records])) if records else None,
        pair_frequency_mean=float(np.mean([p['frequency'] for p in records])) if records else None,
        pair_frequency_max=max([p['frequency'] for p in records],default=0),
        pair_age_mean=float(np.mean([memory.epoch-p['last_epoch'] for p in records])) if records else None,
        pair_lifetime_mean=float(np.mean([memory.epoch-p['first_epoch']+1 for p in records])) if records else None,
        pair_delta_corr_norm_mean=float(np.linalg.norm(dr,axis=1).mean()) if len(dr) else None,
        pair_delta_corr_norm_sd=float(np.linalg.norm(dr,axis=1).std()) if len(dr) else None,
        test_relation_query_coverage=float(valid.mean()),test_pair_retrieved_unique=len(chosen),
        test_pair_use_fraction=ratio(len(chosen),len(pairs)),
        test_relation_direction_accuracy=ratio(int((((1-2*y)*ec>0)&valid).sum()),int(valid.sum())),
        test_typical_direction_accuracy=float(((1-2*y)*et>0).mean()),
        test_relation_positive=int((ec>0).sum()),test_relation_negative=int((ec<0).sum()),test_relation_zero=int((ec==0).sum()),
        corrected=int(((base_pred!=y)&(final_pred==y)).sum()),harmed=int(((base_pred==y)&(final_pred!=y)).sum()),
        delta_mean_abs=float(np.abs(delta).mean()),delta_max_abs=float(np.abs(delta).max()),
        delta_q05=float(np.quantile(delta,.05)),delta_q50=float(np.quantile(delta,.5)),delta_q95=float(np.quantile(delta,.95)),
        typical_gate_mean=float(a('typical_gate').mean()),confusion_gate_mean=float(a('confusion_gate').mean()),
        cam_evidence_mean_abs=float(np.abs(a('cam_evidence')).mean()),corr_evidence_mean_abs=float(np.abs(a('corr_evidence')).mean()),
        corr_view_weight_mean=float(a('view_weights')[:,1].mean()),**perclass)
    for m,name in enumerate(('pheno','anat','func','corr')):
        row[name+'_cam_weight_mean']=float(a('modality_weights')[:,m].mean())
    if valid.any():
        att=a('concept_attention')[valid]
        row['semantic_attention_entropy']=float(-(att*np.log(np.maximum(att,1e-12))).sum(1).mean())
        row['semantic_attention_between_query_sd']=float(att.std(0).mean())
    return row

def history_rows():
    rows=[]
    for path,name,mult in ((BASE/'results/fold_metrics.csv',R.BASE_GROUP,100.),
                            (E773/'results/fold_metrics.csv','E77.3-G2',1.)):
        chosen=[v for v in csv_rows(path) if v['group']==name]
        assert len(chosen)==10
        rows += [dict(group=name,fold=int(v['fold']),**{k:float(v[k])*mult for k in R.METRICS}) for v in chosen]
    return rows

def summarize(rows,exclude2=False):
    groups=list(dict.fromkeys(v['group'] for v in rows))
    return {g:{k:moments([v[k] for v in rows if v['group']==g and (not exclude2 or v['fold']!=2)]) for k in R.METRICS} for g in groups}

def summary_table(summary):
    return table(['组','ACC','Hard-AUC/BA','Prob-AUC','SEN-ASD','SPE-TD','ASD-F1'],
        [[g]+[f"{s[k]['mean']:.4f}±{s[k]['sd']:.4f}" for k in R.METRICS] for g,s in summary.items()])

def finalize():
    start=time.time();print('INDEPENDENT AUDIT: loading all 30 best checkpoints',flush=True)
    audit_source_sha=sha_file(HERE/'audit.py')
    manifest=read(HERE/'manifest.json');expected=manifest['source_sha']
    assert R.source_sha()==expected and R.source_map()==manifest['frozen_sources']
    protected_before=read(HERE/'results/protected_before.json')
    assert verify_relocation()==protected_before
    initial_ref={v['fold']:v['initial'] for v in read(HERE/'results/initial_inheritance.json')}
    metrics_rows=[];mechanisms=[];interventions=[];details=[];tasks=[];log_integrity=[]
    references={fold:R.references(fold) for fold in range(1,11)}
    for fold in range(1,11):
        foldtasks=[]
        for group in R.GROUPS:
            taskpath,bestpath,latestpath,logpath=R.paths(group,fold)
            task=read(taskpath);tasks.append(task);foldtasks.append(task)
            assert task['status']=='COMPLETE' and task['source_sha']==expected
            assert sha_file(bestpath)==task['checkpoint_sha256']
            logs,integrity=training_logs(logpath,task);log_integrity.append(integrity)
            train_audit=audit_training(task,logs,references[fold])
            s,module,optimizer,memory,initial=R.initialize(fold,group)
            assert initial==task['initial']==initial_ref[fold]
            cp=torch.load(bestpath,map_location='cpu',weights_only=False)
            assert cp['group']==group and cp['fold']==fold and cp['source_sha']==expected
            assert cp['best_epoch']==task['best_epoch'] and cp['initial']==initial
            assert cp['text_cache_sha256']==R.CFG['text_source_sha256']==sha_file(HERE/'runtime/text_embeddings.pt')
            cache=torch.load(HERE/'runtime/text_embeddings.pt',map_location='cpu',weights_only=False)
            assert torch.equal(cp['module_state']['texts'],cache['texts'])
            R.restore(s,module,optimizer,memory,cp)
            assert R.digest(R.snapshot(s,module,optimizer,memory))==R.digest({k:cp[k] for k in ('base_state','module_state','new_optimizer','memory_state','rng')})
            assert R.digest({k:R.cpu_state(s[k]) for k in ('model','cam')})==cp['base_public_sha']
            assert cp['base_public_sha']==task['trace'][cp['best_epoch']-11]['public_sha']
            before=R.runtime().rng_sha(R.runtime().pack_rng())
            with torch.no_grad():
                out,h=base_forward(R.base(),s,False)
                base_logp=out['logp'].detach();del out
                base_metrics=R.metric_values(base_logp,s['test_np'],s['y'])
                equal_metrics(base_metrics,cp['base_metrics'])
                assert np.array_equal(base_metrics['prob'],np.asarray(cp['base_probability']))
                assert np.array_equal(base_metrics['pred'],np.asarray(cp['base_prediction']))
                bank=device_bank(memory.eval_bank,h.device)
                audit_memory(memory,bank,h,s['parts'][3],s['test_np'],s['labels_np'])
                ti=torch.as_tensor(s['test_np'],device=h.device);y=s['y'][ti]
                args=(h[ti],base_logp[ti],s['parts'][3][ti],ti,bank)
                options=dict(hard=group!='E77.4-G1' and cp['local_epoch']>15,text_index=int(group=='E77.4-G3'))
                module.eval();z=module(*args,**options)
                m=R.metric_values(z['logp'],np.arange(len(ti)),y)
                equal_metrics(m,cp['metrics']);equal_metrics(m,task['metrics'])
                assert np.array_equal(m['pred'],np.asarray(cp['test_prediction']))
                assert np.array_equal(m['prob'],np.asarray(cp['test_probability']))
                assert z['delta'].abs().max()<=.65
                row=dict(group=group,fold=fold,status='INDEPENDENT_AUDIT_PASS',**{k:100.*m[k] for k in R.METRICS},
                    tp=m['asd_tp'],tn=m['td_tn'],fp=m['td_fp'],fn=m['asd_fn'],n_test=m['n_test'],
                    base_acc=100.*base_metrics['acc'],base_hard_auc=100.*base_metrics['hard_auc'],
                    base_prob_auc=100.*base_metrics['prob_auc'],base_sen=100.*base_metrics['sen'],
                    base_spe=100.*base_metrics['spe'],base_asd_f1=100.*base_metrics['asd_f1'],
                    best_epoch_1based=cp['best_epoch'],epochs_executed=task['epochs_executed'],checkpoint_sha256=sha_file(bestpath))
                metrics_rows.append(row)
                mechanism=checkpoint_mechanisms(group,fold,memory,bank,z,y,base_metrics['pred'],m['pred'])
                mechanisms.append(mechanism|train_audit)
                yp=as_numpy(y);bp=base_metrics['pred'];fp=m['pred']
                detail=dict(group=group,fold=fold,status='PASS',checkpoint_sha256=sha_file(bestpath),
                    initial_exact=True,public_base_sha=cp['base_public_sha'],public_optimizer_sha=R.digest({k:cp['base_state'][k] for k in ('opt_base','opt_cam','opt_added')}),
                    new_optimizer_sha=R.digest(cp['new_optimizer']),memory_sha=R.digest(cp['memory_state']),rng_sha=before,
                    text_sha=cp['text_cache_sha256'],pred_probability_metrics_exact=True,bank_reconstruction_exact=True,
                    legal_train_ids=memory.allowed.tolist(),test_ids=s['test_np'].tolist(),split_sha=R.digest(memory.split),
                    corrected_test_ids=s['test_np'][(bp!=yp)&(fp==yp)].tolist(),
                    harmed_test_ids=s['test_np'][(bp==yp)&(fp!=yp)].tolist(),**train_audit)
                if group=='E77.4-G2':
                    true_margin=(z['logp'][:,0]-z['logp'][:,1])*(1-2*y)
                    for name,changes in [('confusion_off',dict(confusion_off=True)),('typical_off',dict(typical_off=True)),
                        ('anonymous_prompt',dict(text_index=1)),('semantic_off',dict(semantic_off=True)),('correction_off',dict(correction_off=True))]:
                        iz=module(*args,**(options|changes));im=R.metric_values(iz['logp'],np.arange(len(ti)),y)
                        imargin=(iz['logp'][:,0]-iz['logp'][:,1])*(1-2*y)
                        interventions.append(dict(group=group,fold=fold,intervention=name,
                            **{k:100.*im[k] for k in R.METRICS},**{'delta_'+k:100.*(im[k]-m[k]) for k in R.METRICS},
                            hard_prediction_flips=int(np.sum(im['pred']!=m['pred'])),
                            true_margin_change=float((imargin-true_margin).mean()),
                            mean_absolute_asd_probability_change=float(np.abs(im['prob'][:,0]-m['prob'][:,0]).mean()),
                            corrected_relative_base=int(((bp!=yp)&(im['pred']==yp)).sum()),
                            harmed_relative_base=int(((bp==yp)&(im['pred']!=yp)).sum())))
                    # Per-concept attention is mechanistic conditioning, not proof of medical truth.
                    detail['mean_concept_attention']=as_numpy(z['concept_attention']).mean(0).tolist()
                assert R.runtime().rng_sha(R.runtime().pack_rng())==before
                assert R.digest(memory.state_dict())==R.digest(cp['memory_state'])
            # Restore latest as well: all optimizer/memory/RNG state is loadable,
            # while exact next-step replay was independently checked in preflight.
            latest=torch.load(latestpath,map_location='cpu',weights_only=False)
            assert latest['source_sha']==expected and latest['initial']==initial
            assert latest['trace']==task['trace'] and latest['next_epoch']==task['epochs_executed']
            assert latest['best_epoch']==task['best_epoch'] and latest['best_acc']==task['metrics']['acc']
            assert latest['wait']>200 or latest['next_epoch']==450
            R.restore(s,module,optimizer,memory,latest)
            assert R.digest(R.snapshot(s,module,optimizer,memory))==R.digest({k:latest[k] for k in ('base_state','module_state','new_optimizer','memory_state','rng')})
            detail['latest_sha256']=sha_file(latestpath);detail['latest_full_state_restore_exact']=True
            details.append(detail)
            print(f"AUDIT PASS {group} fold={fold} ACC={row['acc']:.4f} base={row['base_acc']:.4f}",flush=True)
            del s,module,optimizer,memory,cp,latest,h,base_logp,bank,z,args,cache
            gc.collect()
        assert foldtasks[0]['warmup_signatures']==foldtasks[1]['warmup_signatures']
        assert len(foldtasks[0]['warmup_signatures'])==15
        for t in foldtasks[1:]:
            assert t['initial']==foldtasks[0]['initial']
            assert all(a==b for a,b in zip(t['trace'],foldtasks[0]['trace']))
        assert read(HERE/'results'/f'parity_f{fold:02d}.json')['status']=='PASS'
        R.save_json(HERE/'results/checkpoint_audit_partial.json',dict(status='IN_PROGRESS',completed=len(details),checkpoints=details))
    assert len(metrics_rows)==len(details)==30
    recovery=[]
    for path in sorted((HERE/'runtime/recovery').glob('*_after*.json')):
        groupfold=path.stem.split('_after')[0]
        log_task=next(t for t in tasks if f"{t['group']}_f{t['fold']:02d}"==groupfold)
        valid_rows,_=training_logs(HERE/'runtime/logs'/f'{groupfold}.jsonl',log_task)
        current={r['epoch']:r for r in valid_rows}
        old=read(path)
        matched=[v['epoch'] for v in old if v['epoch'] in current]
        assert all(current[v['epoch']]==v for v in old if v['epoch'] in current),f'Recovery replay differs: {path}'
        recovery.append(dict(path=str(path),replayed_epochs=matched,exact=True,
            missing_log_epochs=[v['epoch'] for v in old if v['epoch'] not in current]))
    protected_after=verify_relocation();assert protected_after==protected_before
    R.save_json(HERE/'results/protected_after.json',protected_after)
    R.write_csv(HERE/'results/fold_metrics.csv',metrics_rows)
    R.write_csv(HERE/'results/mechanism.csv',mechanisms)
    R.write_csv(HERE/'results/interventions.csv',interventions)
    assert sha_file(HERE/'audit.py')==audit_source_sha,'Audit source changed during readback'
    result=dict(status='PASS',best_checkpoints_independently_read=30,latest_checkpoints_restored=30,audit_code_sha256=audit_source_sha,
        preflight_next_step_replay=True,protected_files_unchanged=len(protected_after),
        source_sha=expected,text_cache_sha256=R.CFG['text_source_sha256'],
        warmup_parity_folds=10,public_new_gradients_zero=True,base_common_trajectories_exact=True,
        historical_parameter_reference='full state/optimizer at retained best/latest epochs; sparse losses and RNG at all available recorded epochs',
        recovery_replays=recovery,epoch_log_integrity=log_integrity,
        epoch_log_missing_total=sum(len(v['missing_epochs']) for v in log_integrity),
        epoch_log_invalid_fragments_total=sum(len(v['invalid_fragments']) for v in log_integrity),
        checkpoints=details,audit_seconds=time.time()-start)
    R.save_json(HERE/'results/checkpoint_audit.json',result)
    partial=HERE/'results/checkpoint_audit_partial.json'
    if partial.exists():partial.unlink()
    write_report(metrics_rows,mechanisms,interventions,result,tasks)

def write_report(rows,mechanisms,interventions,audit,tasks):
    import datetime
    history=history_rows();all_rows=history+rows
    summary=summarize(all_rows);nine=summarize(all_rows,True)
    current={g:[r for r in rows if r['group']==g] for g in R.GROUPS}
    g1,g2,g3=R.GROUPS
    delta=lambda a,b:{k:summary[a][k]['mean']-summary[b][k]['mean'] for k in R.METRICS}
    deltas={f'{a} minus {b}':delta(a,b) for a,b in [(g2,R.BASE_GROUP),(g2,g1),(g2,g3),(g2,'E77.3-G2')]}
    paired=[]
    for g in R.GROUPS:
        for ref in (R.BASE_GROUP,'E77.3-G2'):
            for fold in range(1,11):
                a=next(r for r in all_rows if r['group']==g and r['fold']==fold)
                b=next(r for r in all_rows if r['group']==ref and r['fold']==fold)
                paired.append(dict(group=g,reference=ref,fold=fold,**{k:a[k]-b[k] for k in R.METRICS}))
    for other in (g1,g3):
        for fold in range(1,11):
            a=next(r for r in rows if r['group']==g2 and r['fold']==fold)
            b=next(r for r in rows if r['group']==other and r['fold']==fold)
            paired.append(dict(group=g2,reference=other,fold=fold,**{k:a[k]-b[k] for k in R.METRICS}))
    R.write_csv(HERE/'results/paired_comparisons.csv',paired)
    ms={g:[r for r in mechanisms if r['group']==g] for g in R.GROUPS}
    totals={g:{k:sum(r[k] for r in ms[g]) for k in ('corrected','harmed','query_exposures','query_relation_exposures',
        'query_relation_direction_correct','query_base_error_exposures','query_corrected_exposures','query_harmed_exposures',
        'observed_mechanism_log_epochs','missing_mechanism_log_epochs','complete_public_trace_epochs',
        'reference_rng_epochs','reference_sparse_loss_epochs','reference_state_optimizer_epochs',
        'new_pair_events','ema_update_events','expired_pair_events','active_relation_epochs','empty_relation_epochs')} for g in R.GROUPS}
    inter_summary={}
    for name in dict.fromkeys(v['intervention'] for v in interventions):
        ir=[v for v in interventions if v['intervention']==name]
        inter_summary[name]={k:float(np.mean([v[k] for v in ir])) for k in
            ['delta_'+m for m in R.METRICS]+['true_margin_change','mean_absolute_asd_probability_change']}
        inter_summary[name]['hard_prediction_flips']=sum(v['hard_prediction_flips'] for v in ir)
        inter_summary[name]['corrected_relative_base']=sum(v['corrected_relative_base'] for v in ir)
        inter_summary[name]['harmed_relative_base']=sum(v['harmed_relative_base'] for v in ir)
    main=delta(g2,R.BASE_GROUP);relation=delta(g2,g1);sem=delta(g2,g3)
    # No post-hoc tuning or extra training follows either outcome.
    eligible=main['acc']>0 and main['hard_auc']>0 and relation['acc']>0
    if not eligible:
        decision='NO-GO'
        decision_text=(f"完整G2相对G1的ACC增量为{relation['acc']:+.4f}个百分点，但相对历史基座ACC {main['acc']:+.4f}、Hard-AUC {main['hard_auc']:+.4f}个百分点，"
            '未满足完整方法的正向基座增量标准。当前固定协议不支持将其作为已成功的第二创新模块。停止当前原型记忆路线的自动扩展，不启动E77.5/E77.6或温度、数量、Prompt、门控搜索。')
    elif main['sen']<0 or main['spe']<0:
        decision='CONDITIONAL / NO-GO for full claim'
        decision_text='G2的ACC/Hard-AUC与相对G1的ACC增量为正，但存在单侧SEN/SPE下降。记录为有条件候选，不判定完整成功；是否能接受类别代价需要独立临床目标和后续无偏验证，不能由本单seed结果决定。'
    else:
        decision='GO — fixed-protocol candidate only'
        decision_text='G2在本次单seed历史协议下改善ACC与Hard-AUC，且相对G1具有ACC增量，SEN/SPE未出现下降。可保留为候选方法，但不代表无偏泛化、统计稳健性或达到MICCAI 2027发表标准。'
    if sem['acc']>0 and sem['hard_auc']>0:
        semantic_text='医学组G2在本次独立训练对照中ACC与Hard-AUC高于匿名G3；这是当前协议下的描述性支持，须结合干预幅度理解，不能证明通用医学知识或跨seed稳定收益。'
    else:
        semantic_text='医学组未同时优于匿名组的ACC与Hard-AUC，医学语义缺乏独立诊断收益的充分证据；文本交换引起概率变化仅证明模型依赖该条件。'
    # The paired error counts below refer to held-out rows at each method's own
    # selected epoch; direct instantaneous correction uses the same checkpoint.
    correction_text=(f"G2同一最佳checkpoint基础头→最终头：纠正{totals[g2]['corrected']}人次，"
        f"错误翻转{totals[g2]['harmed']}人次，净{totals[g2]['corrected']-totals[g2]['harmed']:+d}。"
        '不同组或历史方法的最佳epoch可能不同，不能把全部最佳值差异归因为即时修正。')
    metrics_names=['ACC','Hard-AUC/BA','Prob-AUC','SEN-ASD','SPE-TD','ASD-F1']
    lines=['# Experiment 77.4 — QCRL-PM 正式研究报告','',
        f"最终状态：30/30正式任务完成，30/30最佳checkpoint独立读回，30/30 latest完整状态恢复；**{decision}**。",'',
        decision_text,'',semantic_text,'',correction_text,'',
        '## 研究动机、论文和代码证据','',
        '目标是将容易混淆的ASD/TD支持关系迁移到第三名查询患者，同时严格阻断新增分支对DARG-CAM基座训练的影响。'
        'E77弱CAM残差、E77.1持续误诊原型与分数修正、E77.2条件增量不足、E77.3端点自比较捷径及共同适应，构成本次结构改变的动机。'
        '本方法是同一正式运行中的基座保护式并行学习，不声称新增分支端到端改变DARG表征。',
        '', '论文全文/公开预览、官方代码、历史真实实现及证据边界见 [source_review.md](source_review.md)。'
        'SPromptGL与BrainPrompt全文已读，SABER完整HTML已读，HP²L仅公开摘要/预览可访问，未声称精读HP²L全文。'
        '这些论文不构成本方法有效性的先验证明。','',
        '## 方法与第三患者监督','',
        '真实基座是DARG + Positive-DCR + CAM + PLNNoPEN，没有活动原始PEN。'
        '数据仍为871名患者、四模态48/6/10/256维，CAM分类前144维分为4×36。'
        'raw CORR严格取原processed_standard_data.csv第64:320零基列，非重新预处理，也不解释为256个有解剖标签ROI。','',
        '合法新增成员为allowed_train=train_indices−explicit_test_indices。用独立NumPy Generator(774300+fold)类别分层、轮转分到B0/B1/B2。'
        'local epoch 1/2/3分别用B0/B1/B2作Query，其他两桶作Support并循环。'
        '所有新损失只作用于Query；原型与活动配对只来自当前Support，Query不属于任一支持成员或配对端点。'
        '基础表征仍是历史transductive训练产物，三路隔离只防新增记忆的直接身份捷径，不等价完整OOF表征。','',
        'CAM四模态每类4个中心，确定性最远点初始化和Lloyd分配。raw CORR另建每类4个全局中心，明确记录其独立成员；'
        '不与不同CAM成员中心虚假拼接，也不把同一个CORR相似度重复加到四模态。'
        'CAM证据按患者预测的归一化有界四模态权重聚合，raw CORR证据只作为一次独立视图，再以患者条件视图权重融合。','',
        '稳定困难患者来自未经修正基础头最近3次eval预测中至少2次错误；前15个新增epoch只收集历史、训练典型分支。'
        'FP真实TD、FN真实ASD；各最多128名独立历史患者，对侧为稳定正确的异病Support近邻，匹配0.6×CAM余弦+0.4×raw CORR余弦。'
        '每困难患者最多2个配对，对侧复用最多8次，20 epoch过期，特征EMA=0.8，确定性淘汰。'
        '稳定pair_id=(fold,ASD_ID,TD_ID)，统一Delta=ASD−TD；历史记录可以保留，但当前Query训练只用两端均在Support的配对。','',
        '共享32维投影：Query Key由CAM144+CORR256产生，Pair Key由双方CAM均值和差异产生；Pair Difference由CAM差异与CORR差异产生。'
        '基于局部上下文、边界接近度、Query与差异的关系、文本兼容性和历史可靠性检索Top8，并用可微分softmax聚合。','',
        r'\[z_x=\mathrm{norm}(W_{CAM}\mathrm{norm}(h_x)+W_{CORR}\mathrm{norm}(r_x)),\quad e(q,a,t)=\tanh((\cos(z_q,z_a)-\cos(z_q,z_t))/0.2).\]', '',
        '语义维度权重同时作用于该Query与候选双方关系向量。正值支持ASD，负值支持TD，来源FP/FN不固定符号。'
        '不存在以配对端点为监督Query的自比较损失。', '',
        r'\[L_{qrel}=\operatorname{softplus}(-(1-2y_q)E_{confusion}(q)/0.5),\quad q\ne a,\;q\ne t.\]', '',
        'CLIP RN50冻结，继承已审计的8条医学文本（TD前4、ASD后4），保留逐概念表示。'
        'Query与候选差异的联合表示对概念做动态注意力，影响每个候选检索、关系维度与可靠性。'
        '两条测量/采集相关描述共享为共同上下文，不当作单一疾病生物标志物。匿名组8条文本、相同缓存编码、网络与参数量。'
        '医学语义是否有效以G2/G3独立训练和固定checkpoint干预判断。','',
        '基础路径严格先执行原forward→L_base.backward→原Adam/Adam/AdamW依次step，再执行一次原eval。'
        '新增分支只读取detach(CAM)、detach(base_logp)、raw CORR、训练记忆和冻结文本；独立AdamW(lr=.001, wd=.01)。', '',
        r'\[L_{base}=L_{cls}+L_{graph}+0.05L_{pair}+0.01L_{adv}+0.01L_{preserve}.\]', '',
        r'\[L_{new}=L_{qcls}+0.05L_{typical}+0.05L_{qrel}+0.03L_{protect}.\]', '',
        'Query权重按训练基础错误和置信度强调困难患者，并在类别内归一化，每类贡献1/2。保护项只用基础已正确且置信度≥.8的训练Query。'
        '没有合法配对时关系损失为0，回退典型分支。','',
        r'\[\delta=0.65\tanh(c_q(g_tE_{typical}+g_cE_{confusion})),\quad correction=[\delta/2,-\delta/2],\quad logp_f=\log\mathrm{softmax}(logp_b+correction).\]', '',
        'c_q=.25+.75×max(基础不确定度,配对可靠性)，两个门控还由患者CAM、基础margin、证据强度、覆盖和一致性决定。'
        '0.65是两类margin的最大修正界，不是每类各0.65。原分类器高置信时仍允许由可信关系驱动修正。','',
        '## 三个正式组及固定训练协议','',
        table(['组','典型原型','第三患者配对','文本','新参数数'],[
            [g1,'四模态CAM + 一次raw CORR','关闭','医学',tasks[0]['initial']['parameter_count']],
            [g2,'相同','持续配对/EMA/第三患者关系监督','医学',tasks[0]['initial']['parameter_count']],
            [g3,'相同','与G2相同','固定匿名8条',tasks[0]['initial']['parameter_count']]]),'',
        '所有组从对应E61 epoch10完整warmup和E74.4初始化链开始；GFR/CAM Adam lr=.0038 wd=.11，DARG原AdamW lr=.001 wd=.01，'
        '固定学习率，无scheduler。seed=0，原10折/4096训练患者对及顺序/full-graph传播，最大450总epoch，wait>200。'
        '使用最终头测试ACC严格提升选最佳checkpoint，val=test。G1/G2前15新增epoch有效计算、参数和优化器轨迹10/10折完全一致。'
        'G3初始公共与非文本新增参数及内部划分相同，允许不同文本下新参数更新不同。','',
        '## 初始化、公共基座保护与公平性','',
        f"基座真实路径：`{BASE}`。E77.3实际路径：`{E773}`。新输出：`{HERE}`。",'',
        '仅在新目录通过path_adapter适配历史失效/model绝对路径；执行原源码原始字节，核对SHA，不改历史源码、配置、数据或建立旧路径软链接。'
        'E51/E43依赖使用已有E72 vendor封存的同SHA文件。','',
        table(['历史文件','SHA256'],[[name,sha_file(BASE/name)] for name in ('run.py','soft_positive_dcr.py','audit.py','config.json','manifest.json')]),'',
        '完整初始化继承见initial_inheritance.json：模型、原优化器、Python/NumPy/Torch CPU/CUDA RNG、患者对和fold身份均10/10通过。'
        '新增参数在独立fork_rng中初始化；无逐epoch重设seed。预检原始单步forward/backward/optimizer完全一致，新增损失对公共参数梯度严格为0，'
        '分类和第三患者关系损失对Key、CORR Value及语义投影存在非零梯度。完整checkpoint下一步恢复重放预检一致。','',
        table(['组','完整公共epoch记录','历史RNG匹配epoch','可读历史稀疏loss匹配','保留历史模型+优化器匹配点','超出历史参考的epoch'],[
            [g,totals[g]['complete_public_trace_epochs'],totals[g]['reference_rng_epochs'],totals[g]['reference_sparse_loss_epochs'],
             totals[g]['reference_state_optimizer_epochs'],sum(r['epochs_beyond_historical_reference'] for r in ms[g])] for g in R.GROUPS]),'',
        '公共RNG在所有有历史参考的epoch逐一匹配；公共模型与优化器在原保留best/latest状态点严格相等，'
        '每组之间同折所有共同epoch公共参数SHA与RNG也相等。原历史没有保存每个epoch的模型/优化器SHA，因此不声称不存在的逐epoch参数参考已逐一比较。'
        '超过原历史early stopping的epoch没有历史参考，只能确认仍执行同一原始算法与随机规则。每折具体比较范围和状态点见checkpoint_audit.json。','',
        '## 正式结果（百分比，mean±population SD）','',summary_table(summary),'',
        'Hard-AUC是对硬预测计算的AUC，与二分类BA数值等价；Prob-AUC为连续概率ROC-AUC，两者分开报告。ASD为正诊断类0，TD=1。','',
        table(['比较']+metrics_names,[[key]+[f'{d[k]:+.4f}' for k in R.METRICS] for key,d in deltas.items()]),'',
        '以下每折数值均来自独立读回；TP/FN对应ASD，TN/FP对应TD。所有均值按折等权，SD用ddof=0。','',
        table(['组','Fold','ACC','Hard-AUC','Prob-AUC','SEN','SPE','F1','TP/TN/FP/FN','最佳epoch','终止epoch','同epoch基础ACC'],[
            [r['group'],r['fold']]+[f'{r[k]:.4f}' for k in R.METRICS]+[
                f"{r['tp']}/{r['tn']}/{r['fp']}/{r['fn']}",r['best_epoch_1based'],r['epochs_executed'],f"{r['base_acc']:.4f}"] for r in rows]),'',
        '相对基座逐折六指标差值和G2相对G1/G3，完整见paired_comparisons.csv。','',
        table(['Fold','G1−基座ACC','G2−基座ACC','G3−基座ACC','G2−G1 ACC','G2−G3 ACC','G2−E77.3 ACC'],[
            [fold]+[f"{next(v for v in paired if v['group']==a and v['reference']==b and v['fold']==fold)['acc']:+.4f}" for a,b in
                [(g1,R.BASE_GROUP),(g2,R.BASE_GROUP),(g3,R.BASE_GROUP),(g2,g1),(g2,g3),(g2,'E77.3-G2')]] for fold in range(1,11)]),'',
        '## 同checkpoint纠错、原型与第三患者关系','',correction_text,'',
        table(['组','同epoch基础ACC均值','最终ACC均值','纠正','错误翻转','净纠正','平均|delta|','最大|delta|'],[
            [g,f"{np.mean([r['base_acc'] for r in current[g]]):.4f}",f"{summary[g]['acc']['mean']:.4f}",
             totals[g]['corrected'],totals[g]['harmed'],totals[g]['corrected']-totals[g]['harmed'],
             f"{np.mean([r['delta_mean_abs'] for r in ms[g]]):.6f}",f"{max(r['delta_max_abs'] for r in ms[g]):.6f}"] for g in R.GROUPS]),'',
        '因此基础头未出现E77.3式由新增梯度导致的联合退化；它仍沿历史公共轨迹训练。各组基础ACC均值可以因选中epoch不同而不同。'
        '基础头正确/错误改变的患者ID（零基）及完整状态SHA见checkpoint_audit.json。','',
        '匿名G3在Fold-3选中epoch294，基础头73.5632%而最终89.6552%；该基础状态仍满足历史公共轨迹核验。'
        '这反映选择到了原基座自然训练曲线中的较弱epoch，并非新增梯度改变基座。也因此不能直接把G3的较大即时纠错量解释为优于历史最佳基座的稳健增量。','',
        table(['组','CAM ASD/TD有效原型均值','raw ASD/TD有效中心均值','中心对cos>.98比例','CORR视图权重','PHENO/ANAT/FUNC/CORR CAM权重'],[
            [g,f"{np.mean([r['cam_asd_prototypes'] for r in ms[g]]):.2f}/{np.mean([r['cam_td_prototypes'] for r in ms[g]]):.2f}",
             f"{np.mean([r['raw_asd_prototypes'] for r in ms[g]]):.2f}/{np.mean([r['raw_td_prototypes'] for r in ms[g]]):.2f}",
             f"{np.mean([r['prototype_pair_cosine_gt_098_rate'] for r in ms[g]]):.4f}",
             f"{np.mean([r['corr_view_weight_mean'] for r in ms[g]]):.4f}",
             '/'.join(f"{np.mean([r[n+'_cam_weight_mean'] for r in ms[g]]):.4f}" for n in ('pheno','anat','func','corr'))] for g in R.GROUPS]),'',
        '所有保存原型都从明确成员重新计算并逐字节匹配，支持成员覆盖100%，测试排除通过。CAM和raw CORR分别聚类，清晰区分成员，未声称二者有同一聚类身份。'
        '中心余弦>.98只是高相似中心的描述性指标；CAM非负/压缩可导致中心相似，不能单凭此判为病理信息完全坍缩。'
        'CORR视图权重及其他CAM模态权重仅表示网络计算贡献，不代表疾病病理重要性。','',
        table(['组','有效Query人次','ASD/TD人次','有配对Query人次','第三Query方向准确率','Query错误纠正率','原正确Query损害率','新建/EMA更新/过期事件'],[
            [g,totals[g]['query_exposures'],
             f"{sum(r['query_asd_exposures'] for r in ms[g])}/{sum(r['query_td_exposures'] for r in ms[g])}",
             totals[g]['query_relation_exposures'],
             f"{100*ratio(totals[g]['query_relation_direction_correct'],totals[g]['query_relation_exposures']):.4f}%" if totals[g]['query_relation_exposures'] else '无配对',
             f"{100*ratio(totals[g]['query_corrected_exposures'],totals[g]['query_base_error_exposures']):.4f}%",
             f"{100*ratio(totals[g]['query_harmed_exposures'],totals[g]['query_exposures']-totals[g]['query_base_error_exposures']):.4f}%",
             f"{totals[g]['new_pair_events']}/{totals[g]['ema_update_events']}/{totals[g]['expired_pair_events']}"] for g in R.GROUPS]),'',
        '上表训练统计是可读日志中的患者×epoch暴露次数，并非独立患者数；统计为新增优化步骤前预测。'
        '其中方向准确率只针对确有合法配对的第三Query，绝不是支持端点自比较准确率。Query/Support及测试隔离由每一步代码断言、保存bank重建和独立预检共同核验。','',
        table(['组','最佳CP配对均值','FP/FN池均值','匹配质量均值','历史出现次数均值','配对检索利用比例均值','测试第三Query方向准确率均值','Econf正/负/零'],[
            [g,f"{np.mean([r['effective_pairs'] for r in ms[g]]):.2f}",
             f"{np.mean([r['pool_fp'] for r in ms[g]]):.2f}/{np.mean([r['pool_fn'] for r in ms[g]]):.2f}",
             f"{np.mean([r['pair_matching_quality_mean'] for r in ms[g] if r['pair_matching_quality_mean'] is not None]):.4f}" if any(r['stored_pairs'] for r in ms[g]) else '空',
             f"{np.mean([r['pair_frequency_mean'] for r in ms[g] if r['pair_frequency_mean'] is not None]):.4f}" if any(r['stored_pairs'] for r in ms[g]) else '空',
             f"{np.mean([r['test_pair_use_fraction'] for r in ms[g] if r['test_pair_use_fraction'] is not None]):.4f}" if any(r['effective_pairs'] for r in ms[g]) else '空',
             f"{100*np.mean([r['test_relation_direction_accuracy'] for r in ms[g] if r['test_relation_direction_accuracy'] is not None]):.4f}%" if any(r['test_relation_query_coverage'] for r in ms[g]) else '空',
             f"{sum(r['test_relation_positive'] for r in ms[g])}/{sum(r['test_relation_negative'] for r in ms[g])}/{sum(r['test_relation_zero'] for r in ms[g])}"] for g in R.GROUPS]),'',
        'pair_id、年龄、首次/最近更新、累计频次、EMA特征、双向差异和可靠性均完整保存在checkpoint。'
        '独立审计核对配对类别方向、端点合法性、容量/复用上限、时间关系和差异精确一致。'
        '没有保存每次历史EMA的全量特征，不能声称重建所有历史EMA数值；预检验证EMA公式，正式保存当前状态和每epoch更新次数。','',
        '错误频次/持续计数在患者处于Support时更新，三次预测历史则对全部合法训练成员收集；Query标签不参与当前Support筛选。'
        '因此这些历史计数描述支持身份下的持续错误，不等同每个自然epoch都更新的全患者错误持续时间。','',
        table(['组','关系loss前20活动epoch均值','后20活动epoch均值','接近理论下限比例','保护loss均值','加权保护/QueryCE','Key零梯度比例','Value零梯度比例'],[
            [g]+[f"{np.mean([r[k] for r in ms[g] if r[k] is not None]):.6f}" if any(r[k] is not None for r in ms[g]) else '无活动关系' for k in
                ('relation_loss_first20','relation_loss_last20','relation_loss_near_theoretical_floor_rate','protect_loss_mean','weighted_protect_over_qcls','pair_key_zero_gradient_rate','corr_value_zero_gradient_rate')] for g in R.GROUPS]),'',
        '有界证据|E|≤1且tau_loss=.5，单项softplus理论下限为softplus(-2)=0.126928；本报告以≤下限+.001定义“接近下限”，用于机制描述，未用于选模或调参。'
        'E77.3的92.78%端点任务饱和与本次第三Query监督不是同一任务。非零梯度不等于有效泛化，须与正式G2−G1和实际纠错一起判断。','',
        '## 医学语义与固定checkpoint干预','',semantic_text,'',
        table(['组','关系概念注意力熵（最大ln8≈2.07944）','概念权重跨Query平均SD'],[
            [g]+[f"{np.mean([r[k] for r in ms[g] if k in r]):.8f}" if any(k in r for r in ms[g]) else '无活动关系' for k in
                ('semantic_attention_entropy','semantic_attention_between_query_sd')] for g in R.GROUPS]),'',
        '若熵接近ln8且患者间权重变化很小，说明实际概念选择接近均匀；即使结构允许Query条件化，也不能因此宣称学到了有选择性的医学知识。','',
        '仅对G2已选定最佳checkpoint执行以下干预，不重训、不选新checkpoint、不改阈值。差值均为干预−完整G2，单位百分点；概率变化为平均|ΔP(ASD)|。','',
        table(['干预','ΔACC','ΔHard','ΔProb','ΔSEN','ΔSPE','硬预测翻转总数','真类margin变化','概率变化'],[
            [name]+[f"{v['delta_'+k]:+.6f}" for k in ('acc','hard_auc','prob_auc','sen','spe')]+[
                v['hard_prediction_flips'],f"{v['true_margin_change']:+.6f}",f"{v['mean_absolute_asd_probability_change']:.8f}"] for name,v in inter_summary.items()]),'',
        '关闭配对测量固定状态下关系分支的即时贡献，G2−G1独立训练测量包含学习路径的差异。'
        '典型证据、语义与门控存在交互，不能把各干预指标简单相加。关闭最终修正严格还原同epoch基础头。'
        '文本注意力的患者间变化可说明条件化计算存在，不能单独证明医学知识有用。','',
        f"G2关闭配对导致{inter_summary['confusion_off']['hard_prediction_flips']}个硬预测变化、ACC {inter_summary['confusion_off']['delta_acc']:+.4f}个百分点；"
        f"其相对G1仅{relation['acc']:+.4f}个百分点。关系证据确实被使用，也解决了旧端点目标过早饱和，但不足以使完整医学方法超过历史最佳基座。",'',
        '## Fold-2、历史协议及九折描述性结果','',
        '原Fold-2两名train/test重叠患者（1基行377、407）在原公共训练/评估中的处理完全保留；'
        '所有新增Query、Support、原型、配对、错误历史和新损失均排除explicit_test_ids，包括该两人。'
        '历史val=test选模、CV前RFE/标准化与transductive全图传播仍在，协议一致不等于无偏泛化。'
        '本轮单seed结果不代表外部验证或MICCAI 2027发表标准。','',
        '排除Fold-2的九折仅作描述，无新划分或重训，也不解决其他偏差：','',summary_table(nine),'',
        '## 独立审计、恢复和日志完整性','',
        f"30/30最佳checkpoint预测类别、连续概率和六指标与保存值严格相同；30/30 latest的模型、原/新优化器、记忆和RNG可完整恢复。"
        f"{audit['protected_files_unchanged']}个受保护历史文件SHA前后不变，训练冻结SHA `{audit['source_sha']}`。",'',
        f"文本缓存SHA `{audit['text_cache_sha256']}`；模型/优化器/记忆/RNG/划分/产物SHA见manifest.json与checkpoint_audit.json。",'',
        '首次预检仅修复历史指标函数CPU NumPy索引接口，发生在正式启动前，影响正式任务0，见preflight_repairs.json。'
        '正式训练方法源码/config/text自预检冻结后未更改；独立审计和报告代码后续完成，不属于训练冻结文件。','',
        f"最终报告仅修复Markdown表格转义并补充注意力/因果解释，随后用固定审计源SHA `{audit['audit_code_sha256']}` 重新独立读回全部30个最佳checkpoint。"
        '修订未改变任何训练算法、配置、checkpoint或正式指标，见reporting_revision.json。','',
        '会话中断终止最初附着进程，已完成任务保持原checkpoint与结果，其他任务从完整latest恢复；未随机重启已运行任务。'
        '恢复后已保留的尾部epoch日志逐项核对，结果见checkpoint_audit.json/recovery_replays。'
        'Fold-2日志存在非法片段，阻止一次恢复解析，保留原始字节后清理非法行，再从同一latest继续；算法源码及checkpoint未改。'
        '其他少量日志也存在缺行/片段，不能仅凭这一现象确定底层原因。未收到OSSFS EIO/errno5，未启用/model回退。','',
        f"完整原子任务记录保存全部公共轨迹，共{sum(totals[g]['complete_public_trace_epochs'] for g in R.GROUPS)}个epoch；"
        f"JSONL中缺失{audit['epoch_log_missing_total']}个机制epoch记录、非法片段{audit['epoch_log_invalid_fragments_total']}条。"
        '未重训以补写或伪造缺失统计，训练机制总数仅按可读记录报告。完整预测/最佳及latest状态、所有公共RNG轨迹与warmup parity不依赖这些缺失行。'
        '因此核心指标与状态审计通过，逐epoch机制日志完整性有上述明确限制。','',
        '## 研究判断与下一步','',decision_text,'',semantic_text,'',
        f"G2相对基座ACC {main['acc']:+.4f}、Hard-AUC {main['hard_auc']:+.4f}、SEN {main['sen']:+.4f}、SPE {main['spe']:+.4f}；"
        f"G2−G1 ACC {relation['acc']:+.4f}，G2−G3 ACC {sem['acc']:+.4f}。",'',
        '本轮不追加第四组或超参数搜索。若保留候选，下一步应先制定与测试选模分离的无偏验证协议和独立seed/外部数据验证，'
        '再讨论医学语义或论文创新主张；如果NO-GO，转向具有不同信息来源或可验证临床机制的研究问题，停止当前普通原型记忆的局部修补。','',
        '复现入口：`PYTHONDONTWRITEBYTECODE=1 python run.py --preflight`、`python run.py --all --workers 10`、`python run.py --audit`。'
        '重新启动正式入口会校验冻结源并跳过已完成任务，不覆盖历史实验。保留必要best/latest，不保存逐epoch全量特征。','']
    report=HERE/'results/report.md';report.write_text('\n'.join(lines))
    summary_payload=dict(status='COMPLETE_AUDITED_WITH_EPOCH_LOG_LIMITATION' if audit['epoch_log_missing_total'] else 'COMPLETE_AUDITED',
        decision=decision,decision_text=decision_text,semantic_conclusion=semantic_text,formal_completed=30,independent_audited=30,
        summary=summary,nine_fold_summary=nine,deltas=deltas,mechanism_totals=totals,intervention_summary=inter_summary)
    R.save_json(HERE/'results/summary.json',summary_payload)
    R.write_csv(HERE/'results/summary.csv',[
        dict(group=g,**{k:v[k]['mean'] for k in R.METRICS},**{k+'_sd':v[k]['sd'] for k in R.METRICS}) for g,v in summary.items()])
    handoff=Path('/model/spromptgl/experiment/handoff.md')
    old=handoff.read_bytes();old_sha=sha_file(handoff)
    previous_handoff=read(HERE/'results/handoff_update.json') if (HERE/'results/handoff_update.json').exists() else None
    marker='## Experiment 77.4 — QCRL-PM：正式训练与独立审计结论'
    if marker.encode() not in old:
        addition=['',marker,'',datetime.datetime.now(datetime.timezone.utc).isoformat(),'',
            f"STATUS：{summary_payload['status']}；30/30正式训练，30/30独立最佳checkpoint读回，30/30 latest完整恢复。最终：**{decision}**。",'',
            '动机：E77–E77.3相似度原型增量不足、端点排序捷径与基座共同适应；本次改为独立第三患者Query监督和公共基座保护。'
            'QCRL-PM：四模态CAM每类每模态4中心 + 一次独立raw CORR全局4中心/类；三路类别分层交替Support/Query，查询不进当前记忆。'
            '稳定FP/FN各128患者、每困难患者最多2配对、对侧复用8、20epoch过期、EMA=.8、统一ASD−TD差异；Top8第三患者条件化关系检索，32维共享映射，无端点自监督。'
            '冻结RN50的8条医学Prompt逐概念参与Query×差异注意力、候选选择和可靠性；G3等结构匿名文本。'
            '基础DARG-CAM独立按原损失和原优化器训练，新增输入detach、独立AdamW(.001,.01)，Lnew=CE+.05典型+.05第三关系+.03保护，margin bound=.65。','',
            f"唯一基座：`{BASE}`；run SHA `{sha_file(BASE/'run.py')}`；soft_positive SHA `{sha_file(BASE/'soft_positive_dcr.py')}`。"
            f"E77.3本次实际路径`{E773}`。635项受保护历史SHA核对前后不变，只读路径适配在新目录内。",'',
            '10/10完整E61 epoch10模型/优化器/Python/NumPy/Torch CPU/CUDA RNG/训练患者对/原配置继承通过。'
            'G1/G2共同15新增epoch在10折参数/优化器/原型/RNG parity通过；三组共同epoch公共轨迹一致，新增公共梯度严格0。'
            '所有有历史参考的RNG和可读稀疏loss匹配，保留best/latest完整公共模型+优化器点匹配。未声称历史未保存的逐epoch参数SHA或超出历史停止范围的比较。','',
            'G1：查询隔离典型原型、医学文本；G2：完整持久FP/FN第三患者混淆关系；G3：等容量匿名文本。seed0，原10折/max450/wait>200/val=test严格最终ACC提升选模。','',
            summary_table(summary),'',
            table(['比较']+metrics_names,[[key]+[f'{d[k]:+.4f}' for k in R.METRICS] for key,d in deltas.items()]),'',
            correction_text,'',semantic_text,'',
            '逐折六指标、G1/G2/G3对基座与E77.3及G2对G1/G3的配对差异见results/paired_comparisons.csv；完整报告含30折指标、九折排除Fold2结果和同epoch基础头。'
            '原型从成员独立重建完全一致；配对身份/方向/年龄/频次/EMA当前状态、测试排除核验通过。'
            '关系证据、梯度、训练Query纠错/损害和固定checkpoint五项干预见mechanism.csv/interventions.csv；不把端点符号正确或文本交换变化当成医学增益证明。','',
            f"训练中断从完整latest恢复，尾部可读记录严格重放核对；Fold2一次JSONL解析失败已保留原始片段并清理非法行，未改算法。"
            f"JSONL缺{audit['epoch_log_missing_total']}个机制epoch、非法片段{audit['epoch_log_invalid_fragments_total']}条，报告按可读记录统计，不伪造缺失值；公共完整trace和checkpoint不受此日志缺行限制。"
            '无OSSFS EIO，无目录回退。未修改历史协议，Fold2重叠377/407只在新增所有监督/记忆中排除；val=test/RFE前置/transductive与单seed偏差仍在。','',
            decision_text,'',
            f"训练冻结SHA `{audit['source_sha']}`；文本缓存SHA `{audit['text_cache_sha256']}`。"
            f"产物：`{HERE}`；报告`{report}`；manifest/checkpoint_audit/protected_after记录全部关键SHA。"
            '不自动启动E77.5/E77.6，不追加原型数量/温度/Prompt/门控搜索。','']
        with handoff.open('ab') as f:
            f.write(('\n'.join(addition)).encode());f.flush()
            import os
            os.fsync(f.fileno())
        assert handoff.read_bytes().startswith(old)
    supplement='### E77.4 最终机制核对补记'
    if supplement.encode() not in handoff.read_bytes():
        entropy=float(np.mean([r['semantic_attention_entropy'] for r in ms[g2] if 'semantic_attention_entropy' in r]))
        variation=float(np.mean([r['semantic_attention_between_query_sd'] for r in ms[g2] if 'semantic_attention_between_query_sd' in r]))
        note=(f'\n{supplement}\n\nG2第三Query关系loss前20活动epoch均值0.573199、后20为0.306605，接近理论下限比例0；'
            f"测试第三患者关系方向准确率均值88.1727%，不再是端点自比较。固定checkpoint关闭配对ACC {inter_summary['confusion_off']['delta_acc']:+.4f}pp、"
            f"硬预测变化{inter_summary['confusion_off']['hard_prediction_flips']}例，说明关系确有即时作用，但独立G2−G1仅{relation['acc']:+.4f}pp，G2仍低于历史基座。"
            f'医学概念注意力熵{entropy:.8f}（ln8≈2.07944）、跨Query平均SD {variation:.8f}，实际选择接近均匀；结合匿名G3更高ACC，医学知识独立收益未成立。'
            '匿名G3 Fold3选择到自然基座曲线较弱的epoch294（基础73.5632%、最终89.6552%），不代表新增分支改变了公共轨迹。'
            f"最终审计代码SHA `{audit['audit_code_sha256']}`；仅报告格式/解释修订后再次完成30/30独立读回，训练冻结源与全部checkpoint SHA未变。\n")
        with handoff.open('ab') as f:
            f.write(note.encode());f.flush()
            import os
            os.fsync(f.fileno())
    handoff_state=dict(path=str(handoff),status='APPENDED_HISTORY_PRESERVED',
        before_sha256=previous_handoff['before_sha256'] if previous_handoff else old_sha,after_sha256=sha_file(handoff))
    R.save_json(HERE/'results/handoff_update.json',handoff_state)
    manifest=read(HERE/'manifest.json')
    manifest.update(status=summary_payload['status'],decision=decision,formal_completed=30,independent_audited=30,
        protected_files_unchanged=audit['protected_files_unchanged'],frozen_training_sources_unchanged=True,
        audit_code_sha256=audit['audit_code_sha256'],text_cache_sha256=audit['text_cache_sha256'],
        epoch_log_missing_total=audit['epoch_log_missing_total'],recovery_events=len(audit['recovery_replays']),
        handoff=handoff_state,artifacts={str(p.relative_to(HERE)):sha_file(p) for p in sorted((HERE/'results').glob('*')) if p.is_file()},
        checkpoints={str(p.relative_to(HERE)):sha_file(p) for p in sorted((HERE/'checkpoints').glob('*.pt'))},
        environment=dict(python=__import__('sys').version,torch=torch.__version__,device=torch.cuda.get_device_name(0),cuda=torch.version.cuda))
    R.save_json(HERE/'manifest.json',manifest)
    terminal=['STATUS: '+summary_payload['status'],'FORMAL: 30/30','INDEPENDENT AUDIT: 30/30',summary_table(summary),
        table(['比较']+metrics_names,[[key]+[f'{d[k]:+.4f}' for k in R.METRICS] for key,d in deltas.items()]),
        'BASE PROTECTION: 10/10 warmup parity; all available historical RNG and retained state/optimizer references exact; public new gradients zero',
        correction_text,semantic_text,
        f"SHA: frozen training unchanged; {audit['protected_files_unchanged']} historical protected files unchanged",
        f"EPOCH LOG LIMITATION: {audit['epoch_log_missing_total']} missing mechanism records; complete public traces retained",
        'DECISION: '+decision,'REPORT: '+str(report),'HANDOFF: '+str(handoff)+' APPENDED']
    text='\n\n'.join(terminal)+'\n'
    (HERE/'results/final_status.txt').write_text(text)
    manifest['artifacts']['results/final_status.txt']=sha_file(HERE/'results/final_status.txt')
    R.save_json(HERE/'manifest.json',manifest)
    print(text,flush=True)

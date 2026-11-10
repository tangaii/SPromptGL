#!/usr/bin/env python3
"""Independent inheritance, mathematical correctness and checkpoint readback audit."""
from __future__ import annotations
import os, csv, json, gc, hashlib, pickle, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import run as R
from ds_cbpm import build, forward, prepare_texts
from path_adapter import BASE, REPO, LEGACY, ANCHORS, resolve_history, sha_file
HERE=R.HERE
E771=REPO/'experiment/experiment77_1_sc_perm'
E772=REPO/'experiment/experiment77_2_prototype_mechanism_audit'
E77=Path('/model/spromptgl/experiment/experiment77_ecpsm')
KEYS=('acc','hard_auc','prob_auc','sen','spe','asd_f1')

def read(p):
    return json.loads(Path(p).read_text())

def csv_rows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))

def protected_inventory():
    paths=set()
    for root in (BASE,E771,E772,E77):
        for directory,dirs,files in os.walk(root):
            dirs[:]=[v for v in dirs if v not in ('__pycache__','.git')]
            for n in files:
                p=Path(directory)/n
                if p.is_file(): paths.add(p)
    for name in read(BASE/'manifest.json')['protected_sources']:
        p=resolve_history(name)
        if p.is_file(): paths.add(p)
    for p in REPO.glob('*.py'): paths.add(p)
    for root in (LEGACY,ANCHORS):
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts: paths.add(p)
    paths.update([REPO/'data/ABIDE/processed_standard_data.csv', REPO/'data/ABIDE/modal_feat_dict.npy'])
    return {str(p):sha_file(p) for p in sorted(paths)}

def migration_audit():
    expected={'run.py':'cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd',
     'soft_positive_dcr.py':'3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36736718',
     'config.json':'a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a',
     'manifest.json':'4bdfb4fbf191d830c9484c9fdc5d584079895ba9157add248f3f354ea2b48840',
     'audit.py':'cba6be99dd4eb418fed230d1b545199ad95ae674b520883680fe7b9d1ab8cafb'}
    # Historical E77 manifest/config supplies the recorded pre-migration bytes.
    e=read(E77/'config.json')['base']
    expected.update({'run.py':e['runner_sha256'],'soft_positive_dcr.py':e['model_sha256'],
                     'config.json':e['config_sha256'],'manifest.json':e['manifest_sha256']})
    checked={n:{'expected':v,'actual':sha_file(BASE/n)} for n,v in expected.items()}
    assert all(v['expected']==v['actual'] for v in checked.values()), checked
    R.save_json(HERE/'results/migration_audit.json',dict(status='PASS',base=str(BASE),checks=checked,
       adapter='Exact source import; E51/E43 path globals target byte-identical E72 archival files',
       legacy=str(LEGACY),anchors=str(ANCHORS),historical_edits=False,links_created=False))
    return checked

def validate_memory(module, s, allowed):
    m=module.memory; ids=allowed; test=s['test']
    assert not m.allowed[test].any()
    assert torch.equal(m.allowed.nonzero().flatten(),allowed)
    assert (m.labels[test]==-1).all() and (m.assignment[:,test]==-1).all()
    assert not m.cam_cache[test].count_nonzero() and not m.raw_cache[test].count_nonzero()
    max_error=0.
    for md in range(4):
        for c in range(2):
            for k in range(m.cfg['k']):
                members=ids[m.assignment[md,ids]==c*m.cfg['k']+k]
                assert float(m.counts[md,c,k])==len(members)
                assert (m.labels[members]==c).all()
                for expected,actual in ((m.cam_cache[members].sum(0),m.cam_sum[md,c,k]),(m.raw_cache[members].sum(0),m.raw_sum[md,c,k])):
                    error=float((expected-actual).abs().max());max_error=max(max_error,error)
                    assert torch.allclose(expected,actual,atol=1e-4,rtol=1e-5)
    for md in range(4):
        for c in range(2):
            cam,raw,counts,valid=m.leave_one_out(ids,md,c)
            for row,i in enumerate(ids[:8]):
                cluster=int(m.assignment[md,i])-c*m.cfg['k']
                if 0<=cluster<m.cfg['k']:
                    members=ids[m.assignment[md,ids]==c*m.cfg['k']+cluster];members=members[members!=i]
                    assert int(counts[row,cluster])==len(members)
                    if len(members):
                        assert torch.allclose(cam[row,cluster],m.cam_cache[members].mean(0),atol=1e-4,rtol=1e-5)
                        assert torch.allclose(raw[row,cluster],m.raw_cache[members].mean(0),atol=1e-4,rtol=1e-5)
    p=int(m.pair_count);pair=m.pair_ids[:p]
    if p:
        assert m.allowed[pair].all() and (m.labels[pair[:,0]]==0).all() and (m.labels[pair[:,1]]==1).all()
        assert len(set(map(tuple,pair.cpu().tolist())))==p
        assert torch.equal(m.pair_raw[:p],module.raw[pair])
        assert torch.equal(m.pair_raw[:p],m.raw_cache[pair])
        assert torch.equal(m.pair_cam[:p],m.cam_cache[pair])
        assert torch.equal(m.pair_difference[:p],m.pair_raw[:p,0]-m.pair_raw[:p,1])
        assert torch.equal(m.pair_key[:p],m.pair_cam[:p].mean(1))
        assert all(int((m.hard_ids[:p]==i).sum())<=2 for i in m.hard_ids[:p].unique())
    return dict(status='PASS',members=int(m.allowed.sum()),pairs=p,dual_view_members_exact=True,
                max_sum_error=max_error,test_member_count=0,self_leave_one_out=True)

def preflight():
    start=time.time();torch.set_num_threads(1);(HERE/'results').mkdir(exist_ok=True)
    migration=migration_audit()
    print('Caching frozen RN50 medical and matched anonymous prompts...',flush=True)
    texts=prepare_texts(HERE,R.CFG)
    assert texts['texts'].shape==(2,8,1024) and torch.isfinite(texts['texts']).all()
    print('Hashing protected sources/data/historical experiments...',flush=True)
    protected=protected_inventory();R.save_json(HERE/'runtime/protected_before.json',protected)
    import pandas as pd
    frame=pd.read_csv(REPO/'data/ABIDE/processed_standard_data.csv')
    mapping=np.load(REPO/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
    assert [len(mapping[k]) for k in ('PHENO','ANAT','FUNC','Correlation')]==[48,6,10,256]
    assert list(frame.columns[:-1])==[str(v) for k in ('PHENO','ANAT','FUNC','Correlation') for v in mapping[k]]
    raw=torch.tensor(frame.iloc[:,64:320].to_numpy(np.float32))
    inheritance=[]
    for fold in range(1,11):
        s,w,identity=R.base().prepare(fold);s['fold']=fold
        assert torch.equal(s['parts'][3].cpu(),raw)
        assert tuple(s['parts'][3].shape)==(871,256)
        assert torch.equal(s['y'].cpu(),torch.tensor(frame.iloc[:,-1].to_numpy(np.int64)-1))
        assert w['completed_epoch']==10 and w['seed']==0
        rng=R.runtime().rng_sha(R.runtime().pack_rng());assert rng==w['rng_sha']==identity['warmup_rng_sha']
        module=build(R.CFG['method'],fold,texts['texts'],s['parts'][3])
        initial=R.digest(dict(module.named_parameters()))
        assert R.runtime().rng_sha(R.runtime().pack_rng())==rng
        for _ in range(2):
            duplicate=build(R.CFG['method'],fold,texts['texts'],s['parts'][3])
            assert initial==R.digest(dict(duplicate.named_parameters()))
            assert R.runtime().rng_sha(R.runtime().pack_rng())==rng
            del duplicate
        cp=torch.load(BASE/'.runtime/checkpoints'/f'{R.BASE_GROUP}_f{fold:02d}.best.pt',map_location='cpu',weights_only=False)
        task=read(BASE/'.runtime/tasks'/f'{R.BASE_GROUP}_f{fold:02d}.json')
        assert identity==cp['identity']==task['identity']
        R.base().load_history().load_bundle(s,cp['best_state'])
        with torch.no_grad(): out,cap=forward(R.base(),s,module,R.GROUPS[0],0,False,enabled=False)
        met=R.runtime().classification_metrics(out['logp'],s['test'],s['y'])
        pred=np.asarray(cp['test_prediction']);prob=np.asarray(cp['test_probability'])
        assert np.array_equal(pred,met['pred'])
        actual=met['prob'] if prob.ndim==2 else met['prob'][:,0]
        assert np.array_equal(prob,actual),float(np.abs(prob-actual).max())
        assert all(abs(met[k]-cp['metrics'][k])<1e-9 for k in KEYS)
        inheritance.append(dict(fold=fold,status='PASS',identity=identity,initial_module_sha=initial,
           historical_forward_exact=True,raw_corr_exact=True,checkpoint_sha=sha_file(BASE/'.runtime/checkpoints'/f'{R.BASE_GROUP}_f{fold:02d}.best.pt')))
        print(f'Exact historical best forward + complete warmup lineage: fold{fold} PASS',flush=True)
        del s,module,w,cp;gc.collect();torch.cuda.empty_cache()
    R.save_json(HERE/'results/initial_inheritance.json',inheritance)
    s,module,allowed,identity,w,oldgroups=R.initialize(1,R.GROUPS[0]);R.start_memory(s,module,allowed,R.GROUPS[0])
    first=R.snapshot(s,module)
    def off_step(wrapped):
        R.base().load_history().zero_optimizers(s)
        if wrapped: out,_=forward(R.base(),s,module,R.GROUPS[0],1,True,enabled=False)
        else: out=R.base().forward(s,R.BASE_GROUP,training=True)
        out['total'].backward()
        for k in ('opt_base','opt_cam','opt_added'): s[k].step()
        return R.digest(R.public_bundle(s)),R.digest(out['logp'].detach()),R.runtime().rng_sha(R.runtime().pack_rng())
    baseline=off_step(False);R.restore(s,module,first);disabled=off_step(True)
    assert baseline==disabled
    R.restore(s,module,first)
    z=R.step(s,module,allowed,R.GROUPS[0],1)
    ev=z['capture']['evidence']
    assert z['gradients']['corr_projection']>0 and z['gradients']['text_projection']>0
    assert ev['margin_correction'].abs().max()<=.8
    assert ev['margin_correction'].abs().mean()>1e-6
    assert torch.allclose(ev['modw'].sum(1),torch.ones(871,device=s['x'].device))
    assert ev['modw'].min()>=.1 and ev['modw'].max()<=.7
    # Direct classification-only derivative must reach both evidence projections.
    R.restore(s,module,first);R.base().load_history().zero_optimizers(s)
    out,cap=forward(R.base(),s,module,R.GROUPS[0],1,True)
    g=torch.autograd.grad(out['cls'],(module.corr_projection.weight,module.text_projection.weight),retain_graph=False)
    cls_norm=[float(a.norm()) for a in g];assert min(cls_norm)>0
    # A mathematical fixture (not a training run) creates known real-label
    # directions to test pair construction, exclusion, and two-sided gradients.
    R.restore(s,module,first)
    with torch.no_grad(): _,cap=forward(R.base(),s,module,R.GROUPS[1],16,False)
    y=s['y'];pred=y.clone();hard=torch.cat([allowed[y[allowed]==c][:2] for c in (0,1)])
    pred[hard]=1-pred[hard]
    synthetic=F.one_hot(pred,2).float()*4
    for epoch in (13,14,15):
        module.memory.update_errors_and_pairs(synthetic,synthetic,cap['h'],s['parts'][3],y,allowed,epoch,True)
    memory_check=validate_memory(module,s,allowed);assert int(module.memory.pair_count)>0
    a=module.anchors(0);tdiff=F.normalize(a[0]-a[1],dim=0);p=int(module.memory.pair_count)
    pair_raw=module.memory.pair_raw[:p]
    score=module.signed_pair_scores(pair_raw.reshape(-1,256),pair_raw,tdiff)
    row=torch.arange(p,device=score.device)
    assert (score[2*row,row]>0).all() and (score[2*row+1,row]<0).all()
    with torch.no_grad(): out,cap=forward(R.base(),s,module,R.GROUPS[1],16,False)
    selection=cap['evidence']['selected_pairs'];valid=selection>=0
    qids=torch.arange(871,device=selection.device)[:,None].expand_as(selection)
    selected=module.memory.pair_ids[selection.clamp_min(0)]
    assert ((qids[valid]!=selected[:,:,0][valid]) & (qids[valid]!=selected[:,:,1][valid])).all()
    aux,rank=module.losses(cap['evidence'],y,allowed,0,True)
    # Re-evaluate losses with live projections for ranking supervision.
    a=module.anchors(0);dummy={'typical':cap['evidence']['typical'].detach()}
    _,rank=module.losses(dummy,y,allowed,0,True)
    rg=torch.autograd.grad(rank,(module.corr_projection.weight,module.semantic_dimensions.weight))
    ranking_grad=[float(v.norm()) for v in rg];assert min(ranking_grad)>0
    # Restart equivalence includes base/new optimizer, memory, CPU/CUDA RNG.
    snap=R.snapshot(s,module);path=HERE/'runtime/preflight_restore.pt';R.save_torch(path,snap)
    t1=R.step(s,module,allowed,R.GROUPS[1],16);R.update_memory(s,module,allowed,t1['capture']|{'final_logp':t1['evaluated']['logp']},R.GROUPS[1],16)
    state1=R.snapshot(s,module);sig1=(R.digest(state1['base_state']),R.digest(state1['module_state']),R.runtime().rng_sha(state1['rng']),R.digest(t1['evaluated']['logp'].detach()))
    R.restore(s,module,torch.load(path,map_location='cpu',weights_only=False))
    t2=R.step(s,module,allowed,R.GROUPS[1],16);R.update_memory(s,module,allowed,t2['capture']|{'final_logp':t2['evaluated']['logp']},R.GROUPS[1],16)
    state2=R.snapshot(s,module);sig2=(R.digest(state2['base_state']),R.digest(state2['module_state']),R.runtime().rng_sha(state2['rng']),R.digest(t2['evaluated']['logp'].detach()))
    assert sig1==sig2, 'resume not exact'
    path.unlink()
    R.restore(s,module,first)
    # Two legal shared steps, including memory update, before any boundary path.
    states=[]
    for group in R.GROUPS[:2]:
        R.restore(s,module,first)
        for local in (1,2):
            v=R.step(s,module,allowed,group,local);R.update_memory(s,module,allowed,v['capture']|{'final_logp':v['evaluated']['logp']},group,local)
        states.append((R.digest(R.public_bundle(s)),R.digest(R.cpu_state(module)),R.runtime().rng_sha(R.runtime().pack_rng())))
    assert states[0]==states[1]
    result=dict(status='PASS',source_sha=R.source_sha(),source_map=R.source_map(),wall_seconds=time.time()-start,
       historical_folds_exact=10,disabled_one_update_exact=True,initial_model_optimizer_rng_exact=True,
       raw_corr_columns='CSV features64:320, original four-modality 256-dimensional Correlation features',
       memory=memory_check,pair_fixture='Mathematical correctness only; no formal task initialized from fixture',
       classification_only_projection_gradient_norms=cls_norm,ranking_projection_gradient_norms=ranking_grad,
       initial_losses=z['losses'],initial_gradient_norms=z['gradients'],
       weighted_aux_over_abs_base=R.CFG['method']['class_loss_weight']*z['losses']['typical_aux']/max(abs(z['losses']['total']),1e-8),
       weighted_fixture_rank_over_abs_base=R.CFG['method']['ranking_loss_weight']*float(rank)/max(abs(z['losses']['total']),1e-8),
       initial_correction_mean_abs=float(ev['margin_correction'].abs().mean()),
       no_relu_on_new_evidence=True,strict_restart_exact=True,two_step_g1_g2_parity=True,
       workers_private_memory='Separate OS processes and newly constructed state per task',
       frozen_configuration='Single predeclared candidate accepted by numerical correctness/scale audit; no test-guided tuning')
    assert protected_inventory()==protected
    R.save_json(HERE/'results/preflight.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)



def independent_metrics(y,pred,prob):
    from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score
    tp,fn,fp,tn=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(acc=float(np.mean(y==pred)),hard_auc=float(roc_auc_score(y,pred)),
       prob_auc=float(roc_auc_score(y,prob[:,1])),sen=float(tp/(tp+fn)),spe=float(tn/(tn+fp)),
       asd_f1=float(f1_score(y,pred,pos_label=0)),tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn))

def evaluation(s,module,group,epoch,**intervention):
    with torch.no_grad():
        out,cap=forward(R.base(),s,module,group,epoch,False,**intervention)
    return out,cap

def compare_predictions(y,before,after):
    return dict(corrected=int(((before!=y)&(after==y)).sum()),
                harmed=int(((before==y)&(after!=y)).sum()),hard_changes=int((before!=after).sum()))

def extend_rng_reference():
    """No optimizer step: replay public forward RNG beyond historical stopping."""
    result=[];torch.set_num_threads(1)
    for fold in range(1,11):
        taskpath=BASE/'.runtime/tasks'/f'{R.BASE_GROUP}_f{fold:02d}.json'
        latestpath=BASE/'.runtime/checkpoints'/f'{R.BASE_GROUP}_f{fold:02d}.latest.pt'
        original=read(taskpath)['rng_trace'];reference={int(v['epoch']):v for v in original}
        s,w,identity=R.base().prepare(fold);s['fold']=fold
        before=R.runtime().rng_sha(R.runtime().pack_rng())
        output=R.base().forward(s,R.BASE_GROUP,training=True)
        after=R.runtime().rng_sha(R.runtime().pack_rng())
        assert before==reference[11]['rng_before'] and after==reference[11]['rng_after_train']
        del output
        latest=torch.load(latestpath,map_location='cpu',weights_only=False)
        R.base().load_history().load_bundle(s,latest['current']);R.runtime().restore_rng(latest['rng'])
        last=int(latest['next_epoch'])
        assert last==max(reference) and R.runtime().rng_sha(R.runtime().pack_rng())==reference[last]['rng_after_eval']
        parameters=R.digest((dict(s['model'].named_parameters()),dict(s['cam'].named_parameters())))
        for epoch in range(last,450):
            rb=R.runtime().rng_sha(R.runtime().pack_rng())
            out=R.base().forward(s,R.BASE_GROUP,training=True)
            rt=R.runtime().rng_sha(R.runtime().pack_rng());del out
            reference[epoch+1]=dict(epoch=epoch+1,rng_before=rb,rng_after_train=rt,rng_after_eval=rt,
                                   kind='Original public forward RNG replay; no backward or optimizer step')
        assert parameters==R.digest((dict(s['model'].named_parameters()),dict(s['cam'].named_parameters())))
        result.append(dict(fold=fold,status='PASS',historical_epochs=len(original),
              replayed_epochs=450-last,epoch11_forward_rng_matches_historical=True,parameters_unchanged=True,
              historical_task_sha256=sha_file(taskpath),historical_latest_sha256=sha_file(latestpath),
              reference=[reference[e] for e in sorted(reference)]))
        print(f'Full RNG reference fold{fold}: {len(original)} historical + {450-last} no-update replay epochs',flush=True)
        del s,latest;gc.collect();torch.cuda.empty_cache()
    R.save_json(HERE/'results/full_rng_reference.json',dict(status='PASS',folds=result,
       scope='Historical digests where available; same public forward operations replay RNG only beyond stopping. No new trained model.'))
    return result

def readback_one(group,fold,expected):
    taskpath,cp_path,latestpath,logpath=R.paths(group,fold)
    task=read(taskpath);assert task['status']=='COMPLETE' and task['source_sha']==expected
    assert sha_file(cp_path)==task['checkpoint_sha256']
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    assert cp['source_sha']==expected and cp['text_cache_sha256']==sha_file(HERE/'runtime/text_embeddings.pt')
    s,module,allowed,identity,w,oldgroups=R.initialize(fold,group)
    assert identity==task['identity']==cp['identity']
    assert task['initial_module_parameters_sha']==cp['initial_module_parameters_sha']==R.digest(dict(module.named_parameters()))
    R.restore(s,module,cp)
    module_sha=R.digest(R.cpu_state(module));base_sha=R.digest(R.public_bundle(s))
    rng=R.runtime().rng_sha(R.runtime().pack_rng())
    out,cap=evaluation(s,module,group,cp['local_epoch'])
    y=s['y'][s['test']].cpu().numpy();prob=out['logp'][s['test']].exp().cpu().numpy();pred=prob.argmax(1)
    assert np.array_equal(prob,np.asarray(cp['test_probability'])) and np.array_equal(pred,np.asarray(cp['test_prediction']))
    calc=independent_metrics(y,pred,prob)
    assert max(abs(calc[k]-cp['metrics'][k]) for k in KEYS)<1e-10
    mem=validate_memory(module,s,allowed)
    before=read(BASE/'.runtime/tasks'/f'{R.BASE_GROUP}_f{fold:02d}.json')
    extended=read(HERE/'results/full_rng_reference.json') if (HERE/'results/full_rng_reference.json').exists() else None
    historical={v['epoch']:v for v in (extended['folds'][fold-1]['reference'] if extended else before['rng_trace'])};compared=0
    for v in task['rng_trace']:
        assert v['rng_after_memory']==v['rng_after_eval']==v['rng_after_train']
        if v['epoch'] in historical:
            h=historical[v['epoch']]
            assert (v['rng_before_train'],v['rng_after_train'],v['rng_after_eval'])==(h['rng_before'],h['rng_after_train'],h['rng_after_eval'])
            compared+=1
    assert task['rng_trace'][0]['rng_before_train']==w['rng_sha']
    for a,b in zip(task['rng_trace'],task['rng_trace'][1:]):
        assert a['rng_after_memory']==b['rng_before_train']
    latest=torch.load(latestpath,map_location='cpu',weights_only=False)
    assert latest['source_sha']==expected and latest['identity']==identity and latest['next_epoch']==task['epochs_executed']
    assert latest['rng_trace']==task['rng_trace'] and latest['warmup_signatures']==task['warmup_signatures']
    # Complete recovery state is saved after the last memory update.
    saved_rng=R.runtime().rng_sha(latest['rng']);assert saved_rng==task['rng_trace'][-1]['rng_after_memory']
    assert all(k in latest for k in ('base_state','module_state','rng','best_acc','best_epoch','wait','next_epoch'))
    m=module.memory;ev=cap['evidence'];n=int(m.pair_count)
    valid_centers=m.counts>0
    collapse=[]
    for md in range(4):
        for c in (0,1):
            centers=m.cam_sum[md,c,valid_centers[md,c]]/m.counts[md,c,valid_centers[md,c],None]
            cosine=F.normalize(centers,dim=-1)@F.normalize(centers,dim=-1).T
            collapse.extend(cosine[torch.triu(torch.ones_like(cosine,dtype=torch.bool),1)].cpu().tolist())
    selections=ev['prototype_selection'][s['test']].reshape(-1)
    unique_proto=int(selections.unique().numel())
    proto_usage=torch.bincount(selections,minlength=32).cpu().tolist()
    selection=ev['selected_pairs'];weight=ev['pair_weights'];valid=(selection>=0)&(weight>0)
    selected_pair_ids=selection[valid]
    unique_pair_used=int(selected_pair_ids.unique().numel())
    q=torch.arange(871,device=selection.device)[:,None].expand_as(selection)
    assert ((q[valid]!=m.pair_ids[selection.clamp_min(0)][:,:,0][valid]) & (q[valid]!=m.pair_ids[selection.clamp_min(0)][:,:,1][valid])).all()
    unique_test_pairs=int(selection[s['test']][valid[s['test']]].unique().numel())
    if n:
        assert ((m.direction[:n]==0)&(m.hard_ids[:n]==m.pair_ids[:n,1]) |
                (m.direction[:n]==1)&(m.hard_ids[:n]==m.pair_ids[:n,0])).all()
        opposite=torch.where(m.direction[:n]==0,m.pair_ids[:n,0],m.pair_ids[:n,1])
        assert m.reliable[opposite].all()
        assert int(torch.bincount(opposite,minlength=871).max())<=8
        anchor=module.anchors(1 if group==R.GROUPS[2] else 0);tdiff=F.normalize(anchor[0]-anchor[1],dim=0)
        own=module.signed_pair_scores(m.pair_raw[:n].reshape(-1,256),m.pair_raw[:n],tdiff)
        pr=torch.arange(n,device=own.device)
        own_sign=float(((own[2*pr,pr]>0)&(own[2*pr+1,pr]<0)).float().mean())
        _,rank=module.losses({'typical':ev['typical'].detach()},s['y'],allowed,1 if group==R.GROUPS[2] else 0,True)
        grad=torch.autograd.grad(rank,(module.corr_projection.weight,module.semantic_dimensions.weight))
        rank_grad=[float(x.norm()) for x in grad]
    else:
        own_sign=0.;rank_grad=[0.,0.]
    # The following interventions are applied only after restoring a formal best
    # checkpoint. No parameter, memory or RNG update is allowed.
    interventions=[];intervention_data={}
    configs={'raw_corr_off':{'corr_off':True}}
    if group==R.GROUPS[1]:
        configs.update({'boundary_off':{'boundary_off':True},'anonymous_substitution':{'text_override':1},
                        'semantic_swap':{'swap':True},'semantic_off':{'semantic_off':True}})
    for name,kw in configs.items():
        changed,changed_cap=evaluation(s,module,group,cp['local_epoch'],**kw)
        chprob=changed['logp'][s['test']].exp().cpu().numpy();chpred=chprob.argmax(1)
        cm=independent_metrics(y,chpred,chprob)
        margin=(out['logp'][:,0]-out['logp'][:,1])
        chmargin=(changed['logp'][:,0]-changed['logp'][:,1])
        comparison=compare_predictions(y,pred,chpred)
        row=dict(group=group,fold=fold,intervention=name,**{k:100*cm[k] for k in KEYS},
            **{f'delta_{k}_pp':100*(cm[k]-calc[k]) for k in KEYS},
            probability_mean_abs_change=float(np.abs(chprob[:,0]-prob[:,0]).mean()),
            margin_mean_abs_change=float((margin[s['test']]-chmargin[s['test']]).abs().mean()),
            true_margin_mean_change=float(((1-2*s['y'][s['test']])*(chmargin[s['test']]-margin[s['test']])).mean()),**comparison)
        ce=changed_cap['evidence']
        row['prototype_selection_changed_fraction']=float((ev['prototype_selection'][s['test']]!=ce['prototype_selection'][s['test']]).any(-1).float().mean())
        row['pair_selection_changed_fraction']=float((ev['selected_pairs'][s['test']]!=ce['selected_pairs'][s['test']]).any(-1).float().mean())
        row['pair_ranked_weight_mean_l1_change']=float((ev['pair_weights'][s['test']]-ce['pair_weights'][s['test']]).abs().sum(1).mean())
        interventions.append(row);intervention_data[name]=(changed,changed_cap)
    base_pred=cap['base_logp'].argmax(1);final_pred=out['logp'].argmax(1)
    if 'boundary_off' in intervention_data:
        no=intervention_data['boundary_off'][0]['logp'].argmax(1)
    else: no=final_pred
    mechanism=dict(group=group,fold=fold,memory_epoch=int(m.epoch),
        prototypes_asd=int(valid_centers[:,0].sum()),prototypes_td=int(valid_centers[:,1].sum()),
        clustering_coverage=float((m.assignment[:,allowed]>=0).float().mean()),
        prototype_collapse_rate=float(np.mean(np.asarray(collapse)>.98)) if collapse else 0.,
        prototype_max_cosine=max(collapse) if collapse else 0.,
        aligned_view_members=True,prototype_retrieved_unique_test=unique_proto,
        prototype_retrieval_fraction_test=unique_proto/int(valid_centers.sum()),
        raw_contrast_mean_test=float(ev['raw_contrast'][s['test']].mean()),
        raw_contrast_mean_abs_test=float(ev['raw_contrast'][s['test']].abs().mean()),
        raw_margin_effect_mean_abs_test=interventions[0]['margin_mean_abs_change'],
        boundary_pairs=n,boundary_unique_pairs=len(set(map(tuple,m.pair_ids[:n].cpu().tolist()))),
        boundary_used_pairs=unique_pair_used,boundary_used_pairs_test=unique_test_pairs,
        boundary_query_coverage_test=float(valid[s['test']].any(1).float().mean()),
        pair_endpoint_orientation_accuracy=own_sign,pair_rank_corr_gradient=rank_grad[0],pair_rank_semantic_dimension_gradient=rank_grad[1],
        **m.summary(base_pred,final_pred,s['y']))
    if n:
        pair_records=[]
        for j in range(n):
            hard=int(m.hard_ids[j]);opposite=int(m.pair_ids[j,0] if int(m.direction[j])==0 else m.pair_ids[j,1])
            pair_records.append(dict(group=group,fold=fold,pair_index=j,
                hard_row_1based=hard+1,hard_true_label=int(m.labels[hard]),
                opposite_row_1based=opposite+1,opposite_true_label=int(m.labels[opposite]),
                asd_row_1based=int(m.pair_ids[j,0])+1,td_row_1based=int(m.pair_ids[j,1])+1,
                direction='FP' if int(m.direction[j])==0 else 'FN',match_similarity=float(m.pair_similarity[j]),
                match_distance=1-float(m.pair_similarity[j]),reliability=float(m.pair_reliability[j]),
                raw_difference_norm=float(m.pair_difference[j].norm()),
                cam_cross_disease_cosine=float(F.cosine_similarity(m.pair_cam[j,0],m.pair_cam[j,1],dim=0)),
                raw_cross_disease_cosine=float(F.cosine_similarity(m.pair_raw[j,0],m.pair_raw[j,1],dim=0)),
                cumulative_base_errors=int(m.errors[hard]),consecutive_base_errors=int(m.error_streak[hard]),
                stable_frequency=int(m.stable_frequency[hard]),error_age=int(m.epoch)-int(m.last_stable[hard]),
                first_error_epoch=int(m.first_error[hard]),last_error_epoch=int(m.last_error[hard]),
                used_by_test_queries=int(((selection[s['test']]==j)&(weight[s['test']]>0)).any(1).sum())))
        R.write_csv(HERE/'results'/f'pair_members_{group}_f{fold:02d}.csv',pair_records)
        for name,values in (('raw_difference_norm',m.pair_difference[:n].norm(dim=-1)),
                            ('match_similarity',m.pair_similarity[:n]),('reliability',m.pair_reliability[:n])):
            quantile=np.quantile(values.cpu().numpy(),[0,.25,.5,.75,1])
            mechanism.update({f'pair_{name}_q{q}':float(v) for q,v in zip((0,25,50,75,100),quantile)})
        mechanism['pair_matching_success_rate']=float(m.hard_ids[:n].unique().numel())/max(int((m.pool_ids>=0).sum()),1)
    for split,idx in (('train',allowed),('test',s['test'])):
        truth=s['y'][idx];bp=base_pred[idx];fp=final_pred[idx];npred=no[idx]
        corr=compare_predictions(truth.cpu().numpy(),bp.cpu().numpy(),fp.cpu().numpy())
        paircorr=compare_predictions(truth.cpu().numpy(),npred.cpu().numpy(),fp.cpu().numpy())
        e=ev['boundary'][idx];et=ev['typical'][idx];mc=ev['margin_correction'][idx]
        mechanism.update({f'{split}_base_to_final_{k}':v for k,v in corr.items()})
        mechanism.update({f'{split}_boundary_{k}':v for k,v in paircorr.items()})
        for name,value in (('typical',et),('boundary',e),('margin_correction',mc)):
            mechanism[f'{split}_{name}_mean']=float(value.mean())
            mechanism[f'{split}_{name}_mean_abs']=float(value.abs().mean())
            mechanism[f'{split}_{name}_positive_fraction']=float((value>1e-8).float().mean())
            mechanism[f'{split}_{name}_negative_fraction']=float((value< -1e-8).float().mean())
            nonzero=value.abs()>1e-8
            mechanism[f'{split}_{name}_label_direction_accuracy']=float(((value[nonzero]>0)==(truth[nonzero]==0)).float().mean()) if nonzero.any() else 0.
        for c,label in ((0,'asd'),(1,'td')):
            mechanism[f'{split}_boundary_mean_{label}']=float(e[truth==c].mean())
        for md,name in enumerate(('pheno','anat','func','corr')):
            mechanism[f'{split}_context_weight_{name}']=float(ev['modw'][idx,md].mean())
            mechanism[f'{split}_typical_weighted_abs_{name}']=float((ev['modw'][idx,md]*ev['typical_per_modality'][idx,md]).abs().mean())
        if n:
            scores=ev['pair_scores'][idx];sel=selection[idx];weights=weight[idx]
            for d,dname in ((0,'fp'),(1,'fn')):
                dmask=(m.direction[sel.clamp_min(0)]==d)&(sel>=0)
                evidence=(weights*scores.gather(1,sel.clamp_min(0))*dmask).sum(1)
                mechanism[f'{split}_{dname}_pair_evidence_mean']=float(evidence.mean())
                mechanism[f'{split}_{dname}_pair_evidence_abs']=float(evidence.abs().mean())
                mechanism[f'{split}_{dname}_pair_weight_mass']=float((weights*dmask).sum(1).mean())
    # Numerical and temporal diagnostics from the immutable formal epoch logs.
    logs=[json.loads(v) for v in logpath.read_text().splitlines()]
    active_logs=[v for v in logs if v['local_epoch']>15 and v['memory']['pair_count']>0]
    def stat(field,subset=logs):
        vals=[float(field(v)) for v in subset]
        return dict(mean=float(np.mean(vals)),minimum=float(np.min(vals)),maximum=float(np.max(vals)),exact_zero_fraction=float(np.mean(np.asarray(vals)==0))) if vals else dict(mean=0.,minimum=0.,maximum=0.,exact_zero_fraction=0.)
    numerical=dict(group=group,fold=fold,epochs=len(logs),active_pair_epochs=len(active_logs),
       typical_loss=stat(lambda v:v['losses']['typical_aux']),
       ranking_loss=stat(lambda v:v['losses']['boundary_rank'],active_logs),
       weighted_typical_over_cls=stat(lambda v:.05*v['losses']['typical_aux']/max(v['losses']['cls'],1e-8)),
       weighted_rank_over_cls=stat(lambda v:.02*v['losses']['boundary_rank']/max(v['losses']['cls'],1e-8),active_logs),
       corr_projection_gradient=stat(lambda v:v['gradients']['corr_projection']),
       text_projection_gradient=stat(lambda v:v['gradients']['text_projection']),
       max_fp_pool=max(v['memory']['pool_fp'] for v in logs),max_fn_pool=max(v['memory']['pool_fn'] for v in logs),
       max_pair_count=max(v['memory']['pair_count'] for v in logs))
    numerical['typical_loss_theoretical_floor']=float(np.logaddexp(0.,-2.))
    floor=float(np.logaddexp(0.,(module.cfg['ranking_margin']-1.)/.2))
    numerical['ranking_loss_theoretical_floor']=floor
    numerical['active_ranking_epochs_within_5percent_of_floor']=float(np.mean([v['losses']['boundary_rank']<=1.05*floor for v in active_logs])) if active_logs else 0.
    assert module_sha==R.digest(R.cpu_state(module)) and base_sha==R.digest(R.public_bundle(s))
    assert rng==R.runtime().rng_sha(R.runtime().pack_rng())
    audit=dict(group=group,fold=fold,status='PASS',checkpoint_sha256=sha_file(cp_path),model_sha=base_sha,
       module_sha=module_sha,memory_sha=R.digest(m.state_dict()),text_cache_sha256=cp['text_cache_sha256'],
       rng_sha=rng,checkpoint_readback_exact=True,manual_metric_max_error=max(abs(calc[k]-cp['metrics'][k]) for k in KEYS),
       memory=mem,shared_rng_reference_epochs_checked=compared,shared_rng_new_epochs=len(task['rng_trace']),
       full_latest_recovery_state_checked=True,initial_module_sha=task['initial_module_parameters_sha'],
       parameter_count=sum(p.numel() for p in module.parameters()),prototype_test_usage_counts=proto_usage,
       train_indices_sha=R.digest(s['train_np']),test_indices_sha=R.digest(s['test_np']),historical_pair_sha=identity['pair_sha'])
    result=dict(group=group,fold=fold,**{k:100*calc[k] for k in KEYS},
                **{k:calc[k] for k in ('tp','tn','fp','fn')},
                best_epoch_1based=cp['best_epoch'],epochs_executed=task['epochs_executed'],
                checkpoint_sha256=audit['checkpoint_sha256'])
    del cp,s,module,latest,cap,ev,intervention_data;gc.collect();torch.cuda.empty_cache()
    return result,mechanism,interventions,audit,numerical


def aggregate(rows,groups=R.GROUPS):
    return {group:{key:dict(mean=float(np.mean([v[key] for v in rows if v['group']==group])),
                           sd=float(np.std([v[key] for v in rows if v['group']==group],ddof=0))) for key in KEYS}
            for group in groups}

def table(summary,groups):
    text=['| 组别 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        text.append('| '+group+' | '+' | '.join(f"{summary[group][k]['mean']:.4f}±{summary[group][k]['sd']:.4f}" for k in KEYS)+' |')
    return '\n'.join(text)


def paired_comparisons(rows):
    lookup={(v['group'],int(v['fold'])):v for v in rows};out=[]
    pairs=[(g,'E74.4-G2') for g in R.GROUPS]+[(R.GROUPS[1],R.GROUPS[0]),(R.GROUPS[1],R.GROUPS[2]),(R.GROUPS[1],'E77.1-G2')]
    for first,second in pairs:
        for fold in range(1,11):
            a,b=lookup[first,fold],lookup[second,fold]
            out.append(dict(comparison=first+' minus '+second,fold=fold,
                            **{f'delta_{k}_pp':a[k]-b[k] for k in KEYS}))
    return out


def cached_readback(group,fold,expected):
    """Reuse only a completed independent readback of immutable identical files."""
    taskpath,bestpath,latestpath,_=R.paths(group,fold)
    identity=dict(audit_source_sha256=sha_file(HERE/'audit.py'),source_sha=expected,
        task_sha256=sha_file(taskpath),checkpoint_sha256=sha_file(bestpath),latest_sha256=sha_file(latestpath),
        full_rng_reference_sha256=sha_file(HERE/'results/full_rng_reference.json'))
    path=HERE/'runtime/audit_cache'/f'{group}_f{fold:02d}.json'
    if path.exists():
        saved=read(path)
        if saved['identity']==identity and saved['status']=='PASS':
            return saved['readback']
    result=readback_one(group,fold,expected)
    R.save_json(path,dict(status='PASS',group=group,fold=fold,identity=identity,readback=result,
                         completed_utc=datetime.now(timezone.utc).isoformat()))
    print(f'INDEPENDENT READBACK PASS {group} fold{fold}',flush=True)
    return result


def watch_readbacks():
    torch.set_num_threads(1);expected=R.source_sha();version=sha_file(HERE/'audit.py')
    completed=set()
    while len(completed)<30:
        assert version==sha_file(HERE/'audit.py'), 'Auditor changed: restart watcher to avoid mixed versions'
        for fold in range(1,11):
            for group in R.GROUPS:
                if (group,fold) in completed: continue
                path=R.paths(group,fold)[0]
                if path.exists() and read(path)['status']=='COMPLETE':
                    cached_readback(group,fold,expected);completed.add((group,fold))
        R.save_json(HERE/'results/readback_progress.json',dict(status='COMPLETE' if len(completed)==30 else 'IN_PROGRESS',
                   completed=len(completed),total=30,audit_source_sha256=version))
        print(f'Independent checkpoint audits {len(completed)}/30',flush=True)
        if len(completed)<30: time.sleep(30)


def finalize():
    start=time.time();torch.set_num_threads(1);expected=R.source_sha()
    pre=read(HERE/'results/preflight.json');assert pre['status']=='PASS' and pre['source_sha']==expected
    if not (HERE/'results/full_rng_reference.json').exists():
        extend_rng_reference()
    rows=[];mech=[];inter=[];audits=[];numerical=[];parity=[]
    for fold in range(1,11):
        tasks=[read(R.paths(g,fold)[0]) for g in R.GROUPS]
        assert all(v['status']=='COMPLETE' for v in tasks)
        assert tasks[0]['warmup_signatures']==tasks[1]['warmup_signatures'] and len(tasks[0]['warmup_signatures'])==15
        assert len(set(v['initial_module_parameters_sha'] for v in tasks))==1
        common=min(len(v['rng_trace']) for v in tasks)
        assert all(tasks[0]['rng_trace'][:common]==v['rng_trace'][:common] for v in tasks[1:])
        parity.append(dict(fold=fold,status='PASS',warmup_parity_epochs=15,common_rng_epochs=common,
                           identical_initial_module_parameters=True))
        for group in R.GROUPS:
            result,m,i,a,n=cached_readback(group,fold,expected)
            rows.append(result);mech.append(m);inter.extend(i);audits.append(a);numerical.append(n)
            R.write_csv(HERE/'results/fold_metrics.csv',rows)
            print(f'Independent checkpoint readback {group} fold{fold}: PASS',flush=True)
    assert len({v['parameter_count'] for v in audits})==1
    refs=[]
    for label,path,selection in (('E74.4-G2',BASE/'results/fold_metrics.csv',R.BASE_GROUP),
                                ('E77.1-G2',E771/'results/fold_metrics.csv','E77.1-G2')):
        # The relocated E74.4 CSV uses proportions; E77.1 uses percentages.
        # Normalize at this explicit source boundary, never in model metrics.
        factor=100. if label=='E74.4-G2' else 1.
        for row in csv_rows(path):
            if row['group']==selection:
                if label=='E74.4-G2':
                    assert all(0.<=float(row[k])<=1. for k in KEYS)
                refs.append(dict(group=label,fold=int(row['fold']),**{k:factor*float(row[k]) for k in KEYS}))
    assert len(refs)==20
    allrows=rows+refs;groups=['E74.4-G2','E77.1-G2',*R.GROUPS]
    summary=aggregate(allrows,groups);nine=aggregate([v for v in allrows if v['fold']!=2],groups)
    gold={'acc':(92.1891,2.7648),'hard_auc':(92.0294,2.7255),'prob_auc':(93.4946,2.9195),
          'sen':(90.0671,3.8862),'spe':(93.9917,4.7320),'asd_f1':(91.4590,2.9013)}
    for metric,(mean,sd) in gold.items():
        assert abs(summary['E74.4-G2'][metric]['mean']-mean)<1e-4
        assert abs(summary['E74.4-G2'][metric]['sd']-sd)<1e-4
    paired=paired_comparisons(allrows)
    R.write_csv(HERE/'results/paired_differences.csv',paired)
    R.write_csv(HERE/'results/mechanism.csv',mech)
    R.write_csv(HERE/'results/interventions.csv',inter)
    combined_pairs=[]
    for p in sorted((HERE/'results').glob('pair_members_E77.3-*_f*.csv')):
        combined_pairs.extend(csv_rows(p))
    if combined_pairs:
        R.write_csv(HERE/'results/pair_members.csv',combined_pairs)
        for p in (HERE/'results').glob('pair_members_E77.3-*_f*.csv'):
            p.unlink()
    R.save_json(HERE/'results/checkpoint_audit.json',dict(status='PASS',readback_count=30,checkpoints=audits,parity=parity))
    R.save_json(HERE/'results/numerical_audit.json',numerical)
    R.save_json(HERE/'results/summary.json',dict(ten_fold=summary,excluding_fold2=nine))
    after=protected_inventory();before=read(HERE/'runtime/protected_before.json');assert after==before
    R.save_json(HERE/'results/protected_after.json',after)
    comparisons={name:{k:float(np.mean([v[f'delta_{k}_pp'] for v in paired if v['comparison']==name])) for k in KEYS}
                 for name in dict.fromkeys(v['comparison'] for v in paired)}
    # GO requires improvements across base classification, independent pair
    # training comparison and matched anonymous condition. No post-hoc retuning.
    g2=summary[R.GROUPS[1]];b=summary['E74.4-G2'];g1=summary[R.GROUPS[0]];g3=summary[R.GROUPS[2]]
    go=all(g2[k]['mean']>b[k]['mean'] for k in ('acc','hard_auc')) and g2['acc']['mean']>g1['acc']['mean'] and g2['acc']['mean']>g3['acc']['mean']
    decision='GO_CANDIDATE_FIXED_PROTOCOL_ONLY' if go else 'NO-GO'
    status=dict(status='COMPLETE_AUDITED',decision=decision,completed_tasks=30,total_tasks=30,
          source_sha=expected,protected_files=len(after),historical_hashes_unchanged=True,
          checkpoint_audit='PASS_30_OF_30',warmup_parity='PASS_10_OF_10',
          comparisons=comparisons,ten_fold=summary,excluding_fold2=nine,audit_seconds=time.time()-start)
    status['paired_acc_wins_ties_losses']={name:dict(wins=sum(v['delta_acc_pp']>1e-9 for v in paired if v['comparison']==name),
        ties=sum(abs(v['delta_acc_pp'])<=1e-9 for v in paired if v['comparison']==name),
        losses=sum(v['delta_acc_pp']< -1e-9 for v in paired if v['comparison']==name)) for name in comparisons}
    R.save_json(HERE/'results/final_status.json',status)
    write_report(status,rows,mech,inter,audits,numerical,parity)
    update_handoff(status,mech,inter)
    manifest=dict(experiment=R.CFG['experiment'],status=status['status'],decision=decision,source_sha=expected,
         frozen_source_map=R.source_map(),audit_source_sha256=sha_file(HERE/'audit.py'),
         historical_base=str(BASE),historical_base_source_sha256=read(HERE/'results/migration_audit.json')['checks'],
         protected_sources=before,checkpoint_sha256={str(R.paths(g,f)[1]):sha_file(R.paths(g,f)[1]) for g in R.GROUPS for f in range(1,11)},
         result_sha256={str(p.relative_to(HERE)):sha_file(p) for p in (HERE/'results').iterdir() if p.is_file()},
         handoff_updated=True,output_fallback=R.CFG['fallback'])
    R.save_json(HERE/'manifest.json',manifest)
    print('\nFINAL STATUS: '+status['status']+'; '+decision+'; 30/30 folds',flush=True)
    print(table(summary,groups),flush=True)
    print('Paired mean differences (pp): '+json.dumps(comparisons,ensure_ascii=False),flush=True)
    print('Independent audit 30/30 PASS; warmup parity 10/10 PASS; historical hashes unchanged.',flush=True)
    print('Experiment: '+str(HERE)+'; handoff appended.',flush=True)


def mechanism_summary(mech,inter,numerical):
    result={}
    fields=('prototypes_asd','prototypes_td','clustering_coverage','prototype_collapse_rate',
      'prototype_retrieval_fraction_test','raw_margin_effect_mean_abs_test','stable_fp','stable_fn',
      'pool_fp','pool_fn','pair_count','unique_pairs','paired_hard_patients','paired_fp','paired_fn',
      'match_similarity_mean','diff_norm_mean','boundary_used_pairs_test','boundary_query_coverage_test',
      'pair_endpoint_orientation_accuracy','train_boundary_label_direction_accuracy','test_boundary_label_direction_accuracy',
      'train_boundary_mean_abs','test_boundary_mean_abs','test_margin_correction_mean_abs',
      'train_context_weight_pheno','train_context_weight_anat','train_context_weight_func','train_context_weight_corr',
      'test_typical_weighted_abs_pheno','test_typical_weighted_abs_anat','test_typical_weighted_abs_func','test_typical_weighted_abs_corr')
    sums=('train_base_to_final_corrected','train_base_to_final_harmed','test_base_to_final_corrected','test_base_to_final_harmed',
          'train_boundary_corrected','train_boundary_harmed','test_boundary_corrected','test_boundary_harmed')
    for group in R.GROUPS:
        selected=[v for v in mech if v['group']==group]
        result[group]={k:float(np.mean([v.get(k,0) for v in selected])) for k in fields}
        result[group].update({k:int(sum(v.get(k,0) for v in selected)) for k in sums})
        ns=[v for v in numerical if v['group']==group]
        result[group]['active_pair_epochs_total']=sum(v['active_pair_epochs'] for v in ns)
        result[group]['corr_gradient_exact_zero_epoch_fraction']=float(np.mean([v['corr_projection_gradient']['exact_zero_fraction'] for v in ns]))
        result[group]['text_gradient_exact_zero_epoch_fraction']=float(np.mean([v['text_projection_gradient']['exact_zero_fraction'] for v in ns]))
    iv={}
    for name in dict.fromkeys(v['intervention'] for v in inter if v['group']==R.GROUPS[1]):
        selected=[v for v in inter if v['group']==R.GROUPS[1] and v['intervention']==name]
        iv[name]={k:float(np.mean([v[k] for v in selected])) for k in
           ['probability_mean_abs_change','margin_mean_abs_change',*[f'delta_{k}_pp' for k in KEYS]]}
        iv[name].update({k:int(sum(v[k] for v in selected)) for k in ('corrected','harmed','hard_changes')})
    return dict(groups=result,g2_fixed_checkpoint_interventions=iv)


def write_report(status,rows,mech,inter,audits,numerical,parity):
    summary=status['ten_fold'];means=mechanism_summary(mech,inter,numerical)
    R.save_json(HERE/'results/mechanism_summary.json',means)
    comp=status['comparisons'];g2=R.GROUPS[1]
    lines=[f'# Experiment 77.3 — DS-CBPM\n\n状态：**{status["status"]}**；研究判定：**{status["decision"]}**。正式训练30/30，最佳检查点独立读回30/30，G1/G2共同15 epoch更新一致性10/10。',
    '\n## 研究动机与真实基座\n\nE77与E77.1将类别/误诊原型相似度加到CAM表示或类别分数，未超过成熟DARG-CAM基座；E77.2未发现稳定的CAM条件增量。DS-CBPM改为记住异病但易混淆的患者双方，并利用原始CORR视图比较双方，学习ASD为正方向的连续局部证据。这里的增量是现有数据中额外可利用的判别证据，不是新的原始信息。',
    f'\n唯一基座：`{BASE}`，E74.4-G2-Positive-DCR。该实际模型为`PLNNoPEN`；本模块附加于DARG-CAM，没有替换一个活动PEN。四模态GFR/DARG/Positive-DCR/CAM及其公共优化器均沿用真实历史实现。路径适配器只在内存中将已断裂的E51/E43路径指向E72封存的相同SHA文件，没有复制仓库、修改历史源码或创建软链接。10个原始锚点、10个完整E61 epoch10 warmup、配置、患者对、模型/优化器和所有RNG继承均复核。基座SHA见`migration_audit.json`、`initial_inheritance.json`和`manifest.json`。',
    '\n## 原论文与历史实现核查\n\n[SPromptGL原文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)及[官方代码](https://github.com/wanxixi11/SPromptGL)：GFR按模态构图并传播，CAM在四模态嵌入之间交互；原PEN由疾病文本引导局部潜在特征权重。原始代码的优化器、分类损失和按测试ACC选模路径已与E74.4链逐层核对。实际实验通过历史`PLNNoPEN`的`Outputlayer`入口捕获CAM144，并在同一输出位置返回最终log-softmax，原分类损失与checkpoint评估均使用修正后输出。历史损失和图构建实现保持原样。',
    '\n[BrainPrompt](https://github.com/AngusMonroe/BrainPrompt)的官方分类器及[原文](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)用于核查归一化疾病条件对齐；[SABER](https://arxiv.org/abs/2607.01901)用于患者条件语义检索和门控思路；[PGP-SAM](https://arxiv.org/abs/2501.06692)用于原型检索思路。均不作为本方法有效性证据，未引入其大模型或图主干。',
    '\n[HP²L](https://doi.org/10.1016/j.media.2026.104063)已核查PubMed摘要及出版商公开的引言、方法/讨论预览；出版商完整正文访问返回403，未取得可独立核查的全文，不能声称完成全文精读。可确认其是多层级疾病分类与原型细化Prompt的框架，与本实验ASD/TD局部混淆边界的双视图记忆不同；其结果不证明DS-CBPM有效。',
    '\nE74.4的`run.py/soft_positive_dcr.py/audit.py/config.json/manifest.json`、结果表、10折任务与checkpoint已核查。E77/E77.1训练与记忆实现、E77.2原型条件增量及ReLU审计已只读检查；未把ReLU当作已确证的主要失效原因。',
    '\n## 固定方法\n\n1. CAM144拆成PHENO/ANAT/FUNC/CORR四个36维槽位。原始CORR256严格来自基座处理后CSV的特征64:320，与模态字典逐列比对；没有重新预处理，也不解释为256个解剖脑区。每模态、每类4个确定性多中心：对当前合法训练成员的归一化CAM槽位做最远点初始化与15轮以内Lloyd。每个CAM中心和CORR中心始终使用同一成员集合，保存完整成员映射、和与计数；训练查询同时从两视图的聚合中减去自身。',
    '\n2. 典型检索在每模态/类选Top2，结合CAM相似度、候选特异的医学语义项和计数可靠性，softmax归一化。相对支持为`E_m=tanh((support_ASD-support_TD)/0.2)`，其中support平均CAM余弦与原始/投影CORR余弦。四模态上下文权重为`0.1+0.6*softmax(W h)`，和为1，范围0.1到0.7，结构性防止单槽位塌缩，不加额外损失。典型证据是有方向的支持，不是疾病概率。',
    '\n3. 原分类头eval-mode预测在最近3次中至少2次错误且窗口齐全，才形成稳定FP/FN候选。前15个新增epoch只收集历史；之后按同一阈值激活配对。每方向最多128名历史困难患者，20 epoch过期，以票数、上限化错误频次、最近稳定时间和ID确定性排序淘汰。历史记录包括累计错误、连续错误长度、首/末错误、稳定频次、最近稳定时间与原/最终头3次预测。模型修正后的正确/错误状态独立记录。',
    '\n4. 每个困难患者最多匹配2名异病、3次中至少2次正确且当前正确的合法训练患者，按`0.75*cos(CAM)+0.25*cos(rawCORR)`最近邻选择；同一对去重，对侧复用最多8次。没有降低稳定错误门槛，也没有强制把普通边界患者当FP/FN。Key为双方CAM均值，Value显式保存`CORR_ASD-CORR_TD`，同时保存双方ID、特征、真实标签、折、方向、相似度、可靠性和完整错误历史。患者ID使用基座数据行索引，导出的一基行号与历史一致。',
    '\n5. 每个查询结合四模态CAM上下文、典型对比、局部边界位置和候选特异的语义相容度检索Top8配对。对当前训练查询屏蔽任何一侧含自身的配对。局部证据为`tanh((cos(q_CORR,ASD_CORR)-cos(q_CORR,TD_CORR))/0.2)`，采用轻量共享32维投影与语义维度权重；负证据保留。FP/FN只表示配对来源，绝不固定指定修正符号。',
    '\n6. 使用E77.1已审计的8条医学文本和冻结CLIP RN50缓存，投影参与每个候选的语义匹配、差异维度和可靠性。匿名组8条固定alpha/beta文本数量、编码和参数完全相同。两组都将第4/8条作为共享语义贡献、其他3条按类别汇总；匿名类身份与医学条件的编码组合规则相同。匿名缓存预先生成，正式任务不加载文本编码器，所有新增初始化在独立CPU/CUDA RNG fork中完成，私有初始化种子固定为773000+fold，三组共享；公共实验seed仍为0。',
    '\n7. 连续患者门控由归一化CAM、基础P(ASD)、两类证据及可靠性组成，分别输出典型/边界门。基础不确定度为`0.2+0.8*4p(1-p)`，矛盾证据降低边界贡献；最终ASD减TD的margin修正有界于±0.8，并以`[+delta/2,-delta/2]`加入原类别得分。困难记忆为空时边界路径与排序损失为0。',
    '\n8. 保留全部原损失`L_cls+L_graph+0.05L_pair+0.01L_adv+0.01L_preserve`；新增仅`0.05*softplus(-label_sign*E_typical/0.5)`与`0.02*L_boundary_rank`。排序在合法异病配对端点上用连续softplus监督正/负方向，margin0.15、温度0.2；这是显式端点监督，端点仍不能从自身配对检索。预检仅用单个预先固定配置检查尺度与梯度，无测试调参。',
    '\n## 正式配置与恢复\n\nG1：医学文本+双视图典型原型，无混淆配对。G2：G1+完整持续FP/FN跨病配对及排序监督。G3：完整G2结构，唯一主要差异为匿名文本。三个模型参数量完全一致（G1保持困难分支参数，但从不执行或产生梯度）；G1/G2前15个新增epoch参数、优化器、预测、类别记忆及RNG逐项一致。',
    '\nseed=0，原10折；每折从对应E61 epoch10完整warmup开始，最大总epoch450，wait>200停止，固定学习率，无scheduler。公共Adam组lr0.0038/wd0.11、DARG的AdamW lr0.001/wd0.01与原状态保持；新增参数追加独立AdamW参数组，旧组不改动。保留4096个原训练患者对及顺序。未从E77/E77.1最佳模型微调，未重训历史基座，没有第四组。',
    '\n每次最佳保存发生在eval之后、下次记忆更新之前，故最佳checkpoint的记忆与其评估精确匹配。latest在每10 epoch/结束时保存完整模型、优化器、记忆/历史、公共CPU/CUDA/Python/NumPy RNG、当前/最佳epoch和wait。恢复从latest严格继续，预检对恢复后单步参数、优化器、记忆、概率及RNG做了完全一致核验。8个worker分别运行独立进程，任务内重新构造每组状态，不共享可变记忆。',
    '\n## 正式结果\n\n所有数值为10折均值±population SD（%），Hard-AUC是历史硬预测AUC/BA，Prob-AUC来自连续概率。ASD=0、TD=1。',
    table(summary,['E74.4-G2','E77.1-G2',*R.GROUPS]),
    '\n逐折预测复算得到TP/TN/FP/FN、最佳epoch及完整指标见`fold_metrics.csv`。六项逐折配对差见`paired_differences.csv`。下表为平均差（pp），顺序ACC/Hard-AUC/Prob-AUC/SEN/SPE/ASD-F1：',
    '\n| 比较 | ACC | Hard | Prob | SEN | SPE | F1 |\n|---|---:|---:|---:|---:|---:|---:|']
    for name,values in comp.items():
        lines.append('| '+name+' | '+' | '.join(f'{values[k]:+.4f}' for k in KEYS)+' |')
    lines.append('\n配对ACC获胜/持平/落后折数（描述性，不当作独立临床检验）：')
    for name,value in status['paired_acc_wins_ties_losses'].items():
        lines.append(f'\n- {name}：{value["wins"]}/{value["ties"]}/{value["losses"]}。')
    lines.append('\n逐折ACC差（pp）：\n\n| Fold | G1−Base | G2−Base | G3−Base | G2−G1 | G2−G3 | G2−E77.1-G2 |\n|---|---:|---:|---:|---:|---:|---:|')
    paired=csv_rows(HERE/'results/paired_differences.csv');names=list(comp)
    for f in range(1,11):
        vals=[next(float(v['delta_acc_pp']) for v in paired if v['comparison']==name and int(v['fold'])==f) for name in names]
        lines.append('| '+str(f)+' | '+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    lines.append('\n排除Fold-2的九折描述性结果，不重新训练，也不消除其余协议偏差：\n\n'+table(status['excluding_fold2'],['E74.4-G2','E77.1-G2',*R.GROUPS]))
    lines.append('\n## 双视图与混淆配对机制\n\n下列记忆计数/平均为每折最佳checkpoint状态；人数、纠错等合计均按fold-context计，跨折可能重复，不是全局唯一临床患者。逐折所有符号、方向与贡献见`mechanism.csv`，训练期容量及损失范围见`numerical_audit.json`。')
    fields=[('prototypes_asd','ASD有效原型'),('prototypes_td','TD有效原型'),('clustering_coverage','聚类覆盖率'),('prototype_collapse_rate','类内中心cos>0.98比例'),
       ('prototype_retrieval_fraction_test','测试检索到的不同中心比例'),('raw_margin_effect_mean_abs_test','关闭CORR差异后平均绝对margin变化'),
       ('stable_fp','稳定FP患者'),('stable_fn','稳定FN患者'),('pool_fp','FP历史池患者'),('pool_fn','FN历史池患者'),
       ('pair_count','有效独立配对'),('paired_hard_patients','成功配对困难患者'),('paired_fp','FP来源配对'),('paired_fn','FN来源配对'),
       ('match_similarity_mean','配对CAM/CORR混合余弦'),('diff_norm_mean','跨疾病原始CORR差异范数'),
       ('boundary_used_pairs_test','测试实际使用不同配对'),('boundary_query_coverage_test','测试配对查询覆盖率'),
       ('test_boundary_label_direction_accuracy','非零边界证据测试类别符号准确率'),('test_boundary_mean_abs','测试边界证据平均绝对值'),
       ('test_margin_correction_mean_abs','测试margin修正平均绝对值')]
    lines.append('\n| 机制量（每折均值） | G1 | G2 | G3 |\n|---|---:|---:|---:|')
    for k,label in fields:
        lines.append('| '+label+' | '+' | '.join(f"{means['groups'][g].get(k,0):.6f}" for g in R.GROUPS)+' |')
    x=means['groups'][g2]
    lines.append(f'\nG2配对端点方向正确率均值为{x["pair_endpoint_orientation_accuracy"]:.6f}；这核验ASD/TD编码和数学方向，并不等价于查询患者诊断正确。G2测试端关闭配对→完整配对：纠正{x["test_boundary_corrected"]}，误翻转{x["test_boundary_harmed"]}；训练端纠正{x["train_boundary_corrected"]}，误翻转{x["train_boundary_harmed"]}。基础头→最终头测试纠正{x["test_base_to_final_corrected"]}，误翻转{x["test_base_to_final_harmed"]}。局部证据有正负两种方向，不用检索来源决定类别。')
    lines.append('\n| 上下文/贡献（测试，每折均值） | PHENO | ANAT | FUNC | CORR |\n|---|---:|---:|---:|---:|')
    lines.append('| G2典型证据加权绝对值 | '+' | '.join(f'{x["test_typical_weighted_abs_"+m]:.6f}' for m in ('pheno','anat','func','corr'))+' |')
    lines.append('\n四槽位代表CAM上下文；所有槽位典型中心还对齐rawCORR，因此这些贡献不能解释为独立的四种临床病理来源。ANAT/FUNC质量特征仅用于上下文，未解释为ASD病理。rawCORR-off仅关闭相对CORR类别/配对比较分数，候选语义和检索上下文中的raw投影仍保留；这是差异证据干预，不是删除全部CORR输入。')
    lines.append('\n## 语义与固定检查点干预\n\nG2与G3的正式独立训练对照是语义贡献的主要依据。下列干预仅解释固定权重的即时响应，不能用语义交换敏感性替代训练对照：\n\n| 干预 | mean abs ΔP(ASD) | mean abs Δmargin | 硬预测变化总数 | ΔACC pp | ΔHard pp | ΔProb pp | ΔSEN pp | ΔSPE pp |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name,v in means['g2_fixed_checkpoint_interventions'].items():
        lines.append('| '+name+' | '+f'{v["probability_mean_abs_change"]:.7f} | {v["margin_mean_abs_change"]:.7f} | {v["hard_changes"]}'+' | '+' | '.join(f'{v["delta_"+k+"_pp"]:+.4f}' for k in KEYS[:5])+' |')
    # Explain the actual independent training comparisons, including tradeoffs.
    basecomp=comp[g2+' minus E74.4-G2'];paircomp=comp[g2+' minus '+R.GROUPS[0]];semcomp=comp[g2+' minus '+R.GROUPS[2]]
    lines.append(f'\nG2−G1的ACC/Hard/SEN差为{paircomp["acc"]:+.4f}/{paircomp["hard_auc"]:+.4f}/{paircomp["sen"]:+.4f}pp；G2−G3为{semcomp["acc"]:+.4f}/{semcomp["hard_auc"]:+.4f}/{semcomp["sen"]:+.4f}pp。只有这些独立训练比较及纠错方向共同支持时才讨论对应增量。若点估计相当或方向不一致，本轮不支持医学知识的独立收益。')
    lines.append('\n## 随机、数据边界和数值审计\n\n30/30最佳checkpoint重新从磁盘读取，预测标签与连续概率逐项精确一致，全部六项指标用独立sklearn公式复算。模型、记忆、配对、文本缓存和RNG摘要列于`checkpoint_audit.json`。30个latest完整恢复状态也已读回。新增参数数目所有组完全一致。10/10 G1/G2前15 epoch共同warmup逐项一致，三组同折公共RNG逐epoch一致；与E74.4有记录的对应epoch全部digest匹配。超出基座历史停止epoch的部分，从其完整latest RNG开始，只读回放原始公共forward随机操作至450 epoch，没有backward或optimizer step，模型参数不变；其epoch11消耗先与真实历史核验。所有新任务的每个epoch再与历史/回放完整参考核验，见full_rng_reference.json。该回放只证明随机消耗，不声称重建了停止后的基座学习轨迹。没有逐epoch重设种子。')
    lines.append(f'\n新增记忆成员严格为effective_train minus explicit_test，Fold-2行377/407也排除，类别标签只在允许训练成员上缓存。双视图成员100%一致，自检索通过双视图leave-one-out与配对双方ID屏蔽；每个记忆属于单个fold和task。保护清单{status["protected_files"]}个历史/根源码/数据/产物SHA前后相同。')
    lines.append('\n预检分类损失对CORR/语义投影的梯度、配对排序对CORR/语义维度投影的梯度均非零；所有正式epoch的总梯度有限。各折辅助损失、相对分类损失比例、投影梯度零比例和激活epoch范围在`numerical_audit.json`。没有配对和warmup阶段的排序损失为零属于设计；活动配对状态最佳checkpoint另核查排序梯度。原图损失可以为负，因此数值比例同时参考分类损失，而非把有正负抵消的总损失误当稳定分母。')
    for group in R.GROUPS:
        xx=means['groups'][group]
        lines.append(f'\n{group}：活动配对epoch合计{xx["active_pair_epochs_total"]}；CORR/语义投影总梯度精确为零的epoch比例均值{xx["corr_gradient_exact_zero_epoch_fraction"]:.6f}/{xx["text_gradient_exact_zero_epoch_fraction"]:.6f}。G1困难参数始终无梯度，不计为该组失效。')
    active=[v for v in numerical if v['active_pair_epochs']]
    if active:
        near=float(np.mean([v['active_ranking_epochs_within_5percent_of_floor'] for v in active]))
        lines.append(f'\n排序数值解释：softplus排序损失的有界理论下限是{active[0]["ranking_loss_theoretical_floor"]:.7f}；活动任务中，距该下限5%以内的epoch比例均值为{near:.4f}。端点与自身的余弦为1，所以同一投影下ASD/TD端点方向天然正确，排序主要施加额外距离间隔；端点100%符号正确不能当作对新患者诊断成功。该损失不是ReLU死梯度，但接近下限时监督压力较弱，必须结合独立G2−G1和真实测试纠错判断。此发现只记录，不据此修改固定方法或追加训练。')
    lines.append('\n## 运行记录、结论与边界\n\n首选/data实验目录mkdir返回OSSFS EIO/errno5，因此在任何正式训练前切换到唯一有效输出/model/spromptgl/experiment/experiment77_3_ds_cbpm。E74.4始终从搬迁后的/data目录只读加载。最初历史依赖位置断裂的问题由相同SHA的原始封存文件解决，没有猜测恢复模型或RNG。训练过程中的worker恢复记录见`execution.json`，正式模型/记忆/配置冻结后没有依据测试指标修改方法或覆盖同编号结果。')
    lines.append(f'\n最终判定：**{status["decision"]}**。G2相对基座ACC/Hard/Prob/SEN/SPE为'+ '/'.join(f'{basecomp[k]:+.4f}' for k in KEYS[:5])+'pp。Prob-AUC改善本身不能证明完整成功；敏感度下降或类别间代价按表中真实数值报告。配对关闭影响小且训练对照没有一致收益时，不把整体变化强行归因于混淆配对。')
    q1=comp[R.GROUPS[0]+' minus E74.4-G2']
    q4=comp[g2+' minus E77.1-G2']
    conclusions=[
      ('双视图典型原型',q1,'基座'),('混淆配对训练增量',paircomp,'G1'),
      ('医学相对匿名语义',semcomp,'G3'),('相对既有持续困难记忆',q4,'E77.1-G2')]
    for name,value,reference in conclusions:
        finding='ACC和Hard-AUC点估计均较高；只支持本单seed固定协议下的差异' if value['acc']>1e-9 and value['hard_auc']>1e-9 else '未在ACC和Hard-AUC上同时优于对照，不能主张完整收益'
        lines.append(f'\n- {name} vs {reference}：{finding}；ΔSEN/ΔSPE为{value["sen"]:+.4f}/{value["spe"]:+.4f}pp。')
    lines.append('\n历史评估保留val=test选模、CV前RFE、full-graph transductive传播、原折划分与Fold-2两个重叠患者的原始处理。协议一致不等于无偏泛化；新增记忆隔离不能去除旧表征/图中的测试暴露。单seed固定协议的点估计不能证明跨seed稳定性或独立临床泛化，也不代表MICCAI2027发表标准。若本轮不支持，应停止自动追加同类训练和大规模搜索；下一步优先建立独立评估与预注册的临床条件增量检验，再决定是否值得继续端到端改进。')
    (HERE/'results/report.md').write_text('\n\n'.join(lines)+'\n')


def update_handoff(status,mech,inter):
    path=Path('/model/spromptgl/experiment/handoff.md')
    marker='<!-- EXPERIMENT77_3_DSCBPM_2026-11-10 -->'
    old=path.read_bytes()
    if marker.encode() in old:
        return
    summary=status['ten_fold'];comp=status['comparisons'];g2=R.GROUPS[1]
    ms=read(HERE/'results/mechanism_summary.json');x=ms['groups'][g2]
    migration=read(HERE/'results/migration_audit.json')['checks']
    text=f'''\n\n## Experiment 77.3 — DS-CBPM：疾病语义引导的混淆边界原型记忆\n\n- 状态：`{status['status']}`；研究判定：**{status['decision']}**；G1/G2/G3各10/10，共30/30正式端到端训练和独立最佳checkpoint审计完成。唯一有效实验目录：`{HERE}`，完整方法/逐折结果/机制/限制见`results/report.md`、`fold_metrics.csv`、`paired_differences.csv`、`mechanism.csv`、`interventions.csv`和`manifest.json`。\n- 动机：E77–E77.2的类别中心与FP/FN相似度缺少稳定CAM条件增量。新模块以异病混淆双方的差异取代单纯误诊中心相似度，学习额外可利用的有方向证据；不声称创造原始新信息。\n- 技术：每模态每类4个确定性CAM144原型，以同一成员对齐rawCORR256中心；CAM分4×36槽位，CORR严格对应基座CSV的64:320特征列。双视图同时leave-one-out，测试患者不参与构建，256维不解释为解剖ROI。\n- 配对：原分类头最近3次eval至少2次错误形成稳定FP/FN，新增15 epoch warmup后启用；每方向历史容量128、20 epoch过期，每困难患者最多2个可靠异病最近邻，混合CAM/CORR相似度匹配，对侧复用≤8。保存CAM上下文Key、双方rawCORR、ASD−TD差异Value、方向、相似度、错误频次/年龄/持续长度及可靠性。每查询排除含自身任一端点的配对。\n- 方向与融合：局部证据用tanh归一化的query-ASD与query-TD投影CORR余弦差，正负方向均保留；FP/FN来源不规定修正符号。患者条件门控综合典型/边界证据与基础不确定度，有界margin修正±0.8。没有大型Transformer或新主干。\n- 医学Prompt：沿用E77.1已审计8条医学文本和冻结CLIP RN50；候选特异语义影响原型/配对检索、差异维度及可靠性。G3使用8条预先固定alpha/beta匿名文本，同编码、维度、参数和完整训练流程。语义交换只能说明敏感性，不能证明医学知识有效。\n- 唯一基座仍是迁移后的`{BASE}`，E74.4-G2 Positive-DCR实际使用`PLNNoPEN`，本轮在DARG-CAM上附加模块，没有替换活动PEN。run.py SHA `{migration['run.py']['actual']}`；soft_positive_dcr.py SHA `{migration['soft_positive_dcr.py']['actual']}`；manifest SHA `{migration['manifest.json']['actual']}`。全部历史保护SHA见清单。\n- 路径适配：E51/E43旧链接已断裂，找到E72原始封存快照，10个锚点与源码逐个匹配原SHA。仅在新目录适配器内调整运行期路径全局量，没有改历史文件、创建链接或推测恢复。10/10历史最佳模型前向、预测/概率完全复现。\n- 完整继承：逐折E61 epoch10模型、所有原优化器状态、Python/NumPy/CPU/CUDA RNG、原4096训练患者对及顺序完全继承。新增参数使用独立CPU/CUDA fork初始化，三组同折相同非文本初值，公共随机流不受新模块影响。G1/G2共同15 epoch更新、类别记忆、优化器及预测10/10完全相同；公共RNG与可用E74.4逐epochdigest一致，三组共同轨迹一致。\n- 配置：seed0、原10折、最大总450 epoch、wait>200、原ACC严格最优选模、固定lr且无scheduler；原Adam lr0.0038/wd0.11，DARG及追加新组AdamW lr0.001/wd0.01。保留原全图传播与全部原损失；新增典型证据0.05、配对排序0.02，只有一套预先固定配置。G1为医学双视图典型；G2完整DS-CBPM；G3完整匿名。未重训历史基座或增加第四组。\n- 预检：模块关闭单步forward/backward/优化器更新与基座一致；CAM/CORR数据对齐、双视图成员、方向符号、合法训练成员、两侧自检索排除、非零分类/排序梯度、独立读取和严格恢复均PASS。独立checkpoint30/30标签、概率与全部指标精确复现，latest完整恢复状态也读回；模型/记忆/配对/文本/RNG/SHA均审计。保护{status['protected_files']}个历史/根源码/数据/产物SHA未变。\n- 数据边界：新增成员严格effective_train−explicit_test，Fold-2的一基行377/407从新记忆、配对和辅助监督排除；其原始基座训练/评估处理不改。协议保留val=test选模、CV前RFE、transductive全图、原折与重叠，不是无偏泛化。\n- 十折均值±population SD（%）：\n\n{table(summary,['E74.4-G2','E77.1-G2',*R.GROUPS])}\n\n- 平均配对差（pp，依次ACC/Hard-AUC/Prob-AUC/SEN/SPE/ASD-F1）：\n'''
    for name,v in comp.items():
        text+='\n  - '+name+'：`'+'/'.join(f'{v[k]:+.4f}' for k in KEYS)+'`。'
    text+='\n\n- 逐折ACC配对差（pp，完整六指标见paired_differences.csv）：\n\n| Fold | G1−Base | G2−Base | G3−Base | G2−G1 | G2−G3 | G2−E77.1-G2 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    ps=csv_rows(HERE/'results/paired_differences.csv')
    for f in range(1,11):
        text+='| '+str(f)+' | '+' | '.join(f"{next(float(v['delta_acc_pp']) for v in ps if v['comparison']==n and int(v['fold'])==f):+.4f}" for n in comp)+' |\n'
    text+=f'''\n- 双视图机制：G2每折ASD/TD有效中心均值{x['prototypes_asd']:.2f}/{x['prototypes_td']:.2f}，聚类覆盖{x['clustering_coverage']:.4f}，类内cos>0.98坍缩率{x['prototype_collapse_rate']:.6f}；同成员对齐100%。关闭CORR差异证据，测试mean|Δmargin|{x['raw_margin_effect_mean_abs_test']:.7f}。原始CORR仍作为检索上下文，故此为差异分数干预。\n- 混淆机制：G2最佳状态每折独立配对均值{x['pair_count']:.2f}，稳定FP/FN均值{x['stable_fp']:.2f}/{x['stable_fn']:.2f}；实际测试查询覆盖{x['boundary_query_coverage_test']:.4f}，边界证据mean绝对值{x['test_boundary_mean_abs']:.7f}、非零证据类别方向准确率{x['test_boundary_label_direction_accuracy']:.4f}。固定checkpoint配对关闭→完整：测试纠正{x['test_boundary_corrected']}、误翻转{x['test_boundary_harmed']}；训练纠正{x['train_boundary_corrected']}、误翻转{x['train_boundary_harmed']}。按折累计，跨折可能重复。\n'''
    for name,v in ms['g2_fixed_checkpoint_interventions'].items():
        text+=f"- G2固定干预{name}：mean|ΔP(ASD)|{v['probability_mean_abs_change']:.7f}，mean|Δmargin|{v['margin_mean_abs_change']:.7f}，硬预测变化{v['hard_changes']}，ΔACC/Hard/Prob/SEN/SPE为"+'/'.join(f'{v["delta_"+k+"_pp"]:+.4f}' for k in KEYS[:5])+"pp。\n"
    sem=comp[g2+' minus '+R.GROUPS[2]]
    text+=f'''- 语义训练对照G2−G3 ACC/Hard为{sem['acc']:+.4f}/{sem['hard_auc']:+.4f}pp。结合完整表判断医学语义；相当或不一致时不宣称独立医学收益。各槽位实际贡献和CORR主导程度见mechanism.csv，不把质量指标解释为ASD病理。\n- 九折描述性（排除Fold-2）ACC/Hard/Prob/SEN/SPE：'''
    for g in ['E74.4-G2',*R.GROUPS]:
        text+=' '+g+' `'+ '/'.join(f'{status["excluding_fold2"][g][k]["mean"]:.4f}' for k in KEYS[:5])+'%`；'
    text+='\n- 运行：首选/data新实验目录创建遭OSSFS EIO/errno5，训练前回退至唯一/model输出；基座保持/data搬迁后路径。worker恢复状态见execution.json。固定方法后未按测试结果改方法、Prompt、阈值或损失，无额外诊断训练/超参搜索。HP²L只取得摘要及出版商预览，完整正文403，未冒称全文精读。\n'
    text+=f'- 最终 **{status["decision"]}**：按基座硬分类性能、G2−G1配对增量、G2−G3医学语义对照和真实纠错共同判断；不能仅凭Prob-AUC或语义交换变化宣称成功。本轮限于历史单seed固定协议，未达到独立临床泛化或MICCAI2027发表验证标准。下一步优先独立评估与临床条件增量证据；若NO-GO，停止自动追加同类原型训练。\n\n{marker}\n'
    with path.open('a') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    after=path.read_bytes();assert after[:len(old)]==old and after[len(old):]==text.encode()
    R.save_json(HERE/'results/handoff_update.json',dict(status='APPENDED',path=str(path),
          before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),historical_prefix_unchanged=True))

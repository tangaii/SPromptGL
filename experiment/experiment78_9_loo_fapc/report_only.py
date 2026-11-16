#!/usr/bin/env python3
"""Independent result audit; never writes frozen training sources/checkpoints."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, time, types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
import run as R
import audit as A
from loo_fapc import unit

HERE=R.HERE; E8=R.E78_8
KEYS=("acc","hard_auc","prob_auc","sen","spe","asd_f1")

def arr(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def reference_module():
    m=types.ModuleType('e78_8_final_reference')
    p=E8/'fapc.py'; exec(compile(p.read_text(),str(p),'exec'),m.__dict__)
    return m

def require(condition,message):
    if not condition: raise RuntimeError(message)

def close(x,y,name,atol=2e-6,rtol=2e-5):
    require(torch.allclose(x,y,atol=atol,rtol=rtol),f'{name}: max error {float((x-y).abs().max().detach())}')
    return float((x-y).abs().max().detach())

def self_tests():
    torch.set_num_threads(1)
    old=reference_module(); cache=R.save_text_cache()
    tests={}; torch.manual_seed(78906)
    for mode in ('visual','text'):
        anchors=None if mode=='visual' else cache['medical_anchors']
        classes=None if mode=='visual' else cache['medical_classes']
        new=R.initialize(anchors,classes,mode,'medical' if mode=='text' else 'none',1,'cpu')
        ref=old.initialize(anchors,classes,mode,'medical' if mode=='text' else 'none',1,'cpu')
        require(all(torch.equal(p,dict(ref.named_parameters())[n]) for n,p in new.named_parameters()),'initial state mismatch')
        h=torch.randn(871,144,requires_grad=True); y=torch.tensor(([0,1]*435)+[0]); support=torch.arange(20)
        queries=torch.tensor([0,1,19,100]); base=F.log_softmax(torch.randn(871,2),-1); weights=torch.tensor([1.2,.9])
        test_query=torch.tensor([100,101,102])
        d=new(h,y,support,test_query,weights,base,training=True)
        e=ref(h,y,support,test_query,weights,base,training=True)
        tests[mode+'_full_support_logit_error']=close(d['semantic_logits'],e['semantic_logits'],'no-LOO/test forward')
        tests[mode+'_full_support_loss_error']=close(d['semantic_loss'],e['semantic_loss'],'no-LOO/test loss')
        g1=torch.autograd.grad(d['semantic_loss']+.01*d['keep_loss'],[h]+list(new.parameters()),retain_graph=True,allow_unused=True)
        g2=torch.autograd.grad(e['semantic_loss']+.01*e['keep_loss'],[h]+list(ref.parameters()),retain_graph=True,allow_unused=True)
        tests[mode+'_full_support_gradient_error']=max(close(a,b,'full support gradient',atol=2e-6,rtol=1e-4) for a,b in zip(g1,g2) if a is not None and b is not None)
        # A no-LOO diagnostic replacement must reproduce E78.8 on training queries.
        original=new._loo_directions
        def static_directions(a,labels,support,query):
            mu,dirs=ref._support_prototypes(a,labels,support)
            return mu[None].expand(len(query),-1,-1),dirs[None].expand(len(query),-1,-1),torch.ones_like(query,dtype=torch.bool),labels[query],torch.bincount(labels[support],minlength=2).to(a)
        new._loo_directions=static_directions
        d=new(h,y,support,support,weights,base,training=True)
        e=ref(h,y,support,support,weights,base,training=True)
        tests[mode+'_disabled_loo_logit_error']=close(d['semantic_logits'],e['semantic_logits'],'disabled LOO logits')
        tests[mode+'_disabled_loo_loss_error']=close(d['semantic_loss'],e['semantic_loss'],'disabled LOO loss')
        ga=torch.autograd.grad(d['semantic_loss']+.01*d['keep_loss'],[h]+list(new.parameters()),retain_graph=True,allow_unused=True)
        gb=torch.autograd.grad(e['semantic_loss']+.01*e['keep_loss'],[h]+list(ref.parameters()),retain_graph=True,allow_unused=True)
        tests[mode+'_disabled_loo_gradient_error']=max(close(a,b,'disabled LOO gradients',atol=2e-6,rtol=1e-4) for a,b in zip(ga,gb) if a is not None and b is not None)
        new._loo_directions=original
        # Separate query path z from prototype path a and check each autograd VJP.
        a=torch.randn(871,32,requires_grad=True); z=torch.randn(1,32,requires_grad=True)
        means,dirs,_,_,_=new._loo_directions(a,y,support,torch.tensor([0]))
        self_grad=torch.autograd.grad(means.square().sum(),a,retain_graph=True)[0]
        require(torch.equal(self_grad[0],torch.zeros_like(self_grad[0])),'own support feature leaked through LOO prototype')
        require(float(self_grad[1:20].norm())>0,'other support gradients lost')
        match=(unit(z)*(dirs[:,0]-dirs[:,1])).sum()
        gquery=torch.autograd.grad(match,z,retain_graph=True)[0]
        require(float(gquery.norm())>0,'query feature path incorrectly detached')
        tests[mode+'_self_gradient_zero_other_support_and_query_nonzero']=True
        # Explicit slow reference, support/query permutations, class swap, count edge.
        all_a=torch.randn(8,32); all_y=torch.tensor([0,0,1,1,1,1,0,1]); s=torch.arange(6); qs=torch.tensor([0,2,7])
        mu,v,*_=new._loo_directions(all_a,all_y,s,qs)
        manual=torch.stack([torch.stack([all_a[s[(s!=int(q)) & (all_y[s]==c)]].mean(0) for c in (0,1)]) for q in qs])
        close(mu,manual,'slow explicit means',atol=2e-7,rtol=1e-6)
        swapped_mu,swapped_v,*_=new._loo_directions(all_a,1-all_y,s,qs)
        close(swapped_mu,mu.flip(1),'class means swap'); close(swapped_v,v.flip(1),'signed class direction swap')
        perm, *_=new._loo_directions(all_a,all_y,s.flip(0),qs.flip(0)); close(perm.flip(0),mu,'query/support order')
        failed=False
        try:new._loo_directions(all_a,all_y,torch.tensor([0,2,3]),qs)
        except RuntimeError:failed=True
        require(failed,'class count<2 did not fail')
        tests[mode+'_manual_permutation_class_swap_and_count_boundary']=True
    # Test actual production Hook and its removal, including zero-main fallback.
    for name,gb,ga in [('same',[1.,0.],[.4,0.]),('opposed',[1.,0.],[-1.,.4]),('orthogonal',[1.,0.],[0.,1.]),('zero_aux',[1.,0.],[0.,0.]),('zero_base',[0.,0.],[1.,0.])]:
        x=torch.tensor([.4,.8],requires_grad=True); h=2*x; bg=torch.tensor(gb); ag=torch.tensor(ga)
        base=(h*bg).sum(); aux=(h*ag).sum(); safe=ag.clone(); dot=(bg*ag).sum(); nb=bg.norm()
        if dot<0 and nb>1e-12:safe-=dot/(nb.square()+1e-12)*bg
        if nb<=1e-12:safe.zero_()
        elif safe.norm()>.2*nb:safe*=.2*nb/(safe.norm()+1e-12)
        d={'h_original':h,'semantic_loss':aux/.1,'keep_loss':aux*0}
        stats=R._protected_backward({'total':base},d,base+aux)
        close(x.grad,2*(bg+safe),'production hook '+name,atol=1e-6,rtol=0)
        require(not h._backward_hooks,'hook remained registered after backward')
        require(np.isfinite(list(stats[k] for k in ('base_norm','aux_norm_ratio_safe','gradient_cosine'))).all(),'nonfinite protection')
        tests['production_hook_'+name]=True
    # Production training must reject nonfinite gradients before any optimizer step.
    toy=torch.nn.Linear(1,2,bias=False)
    h=toy(torch.ones(1,1)); base=(h*float('nan')).sum(); aux=h.sum()
    R._protected_backward({'total':base},{'h_original':h,'semantic_loss':aux/.1,'keep_loss':aux*0},base+aux)
    require(not R._finite_parameters(toy),'nonfinite original gradients escaped the training guard')
    require(not h._backward_hooks,'nonfinite-gradient hook was not removed')
    tests['production_nonfinite_training_guard_rejects_before_step']=True
    # Geometry and serialization-friendly report fields.
    raw=cache['embeddings_raw'].double(); anchor=cache['anchors'].double()
    tests['gram_max_abs_error_float32_readback']=float((raw@raw.T-anchor@anchor.T).abs().max())
    require(tests['gram_max_abs_error_float32_readback']<2e-6,'anchor geometry changed')
    json.dumps(tests,allow_nan=False)
    result={'status':'PASS','tests':tests,'report_only_source_sha256':R.sha_file(Path(__file__))}
    R.atomic_json(R.RESULTS/'auditor_self_tests.json',result)
    return result

def evaluate_task(group,fold,source_sha):
    taskfile,bestfile,latestfile,_=R.task_paths(group,fold)
    task=json.loads(taskfile.read_text()); cp=torch.load(bestfile,map_location='cpu',weights_only=False)
    latest=torch.load(latestfile,map_location='cpu',weights_only=False)
    require(task['status']=='COMPLETE','incomplete task')
    for obj in (task,cp,latest):
        require(obj['source_sha256']==source_sha and obj['group']==group and obj['fold']==fold,'task/checkpoint identity mismatch')
    require(R.sha_file(bestfile)==task['checkpoint_sha256'],'best SHA mismatch')
    state,_,ident=R.prepare(fold); support=R.support_ids(state); test=state['test_np']
    require(ident==task['identity']==cp['identity']==latest['identity'],'base/warmup identity mismatch')
    module=R._make_module(state,group); opt=R._new_optimizer(module)
    initial_sha=R.sha_object({n:p.detach().cpu() for n,p in module.named_parameters()})
    ref_group=R.CFG['group_specs'][group]['reference_group']; ref_task=json.loads((E8/'runtime/tasks'/f'{ref_group}_f{fold:02d}.json').read_text())
    require(initial_sha==task['initialization']['module_init_sha256']==ref_task['initialization']['module_init_sha256'],'E78.8 initial parameters changed')
    require(task['initialization']['support_ids']==support.tolist(),'support set changed')
    require(not np.intersect1d(support,test).size,'test support overlap')
    expected_anchor=module.anchors.clone(); expected_classes=module.prompt_classes.clone()
    R._restore(state,module,opt,cp['best_state']); module.eval()
    require(torch.equal(module.anchors,expected_anchor) and torch.equal(module.prompt_classes,expected_classes),'anchor drift/class change')
    require(not module.anchors.requires_grad and module.anchors.grad is None,'anchor gradient present')
    for key in ('opt_base','opt_cam','opt_added'):
        require(cp['best_state']['base_state'][key]['state'],'original optimizer missing state')
        require(latest['current']['base_state'][key]['state'],'latest original optimizer missing state')
    require(cp['best_state']['new_optimizer']['state'],'new optimizer missing state')
    require(latest['current']['new_optimizer']['state'],'latest new optimizer missing state')
    with torch.no_grad():
        out,d=R._forward(state,module,training=False,support=support,query=test)
        met=R._metrics(out['logp'],state)
        h=d['h_original']; basep=d['base_logp']; y=state['y']; weights=state['weights']
        train=module(h,y,support,support,weights,basep,training=False)
        train_darg=R._metrics(basep,state,support)
        yflip=y.clone(); yflip[torch.as_tensor(test,device=y.device)]=1-yflip[torch.as_tensor(test,device=y.device)]
        flipped=module(h,yflip,support,test,weights,basep,training=False)
        reverse=module(h,y,support[::-1].copy(),test[::-1].copy(),weights,basep,training=False)
    for key in KEYS:require(abs(float(met[key])-float(cp['metrics'][key]))<1e-12,'metric mismatch '+key)
    require(np.array_equal(arr(met['pred']),np.asarray(cp['test_prediction'])),'hard predictions differ')
    require(np.array_equal(arr(met['prob']),np.asarray(cp['test_probability'])),'probabilities differ')
    require(torch.equal(d['semantic_logits'],flipped['semantic_logits']),'test labels affect auxiliary output')
    require(torch.equal(d['final_logp'],flipped['final_logp']),'test labels affect DARG output')
    close(reverse['semantic_logits'].flip(0),d['semantic_logits'],'support/query permutation',atol=2e-6,rtol=2e-5)
    require(torch.equal(reverse['final_logp'].flip(0),d['final_logp']),'query order affects DARG')
    require(not bool(d['query_in_support'].any()),'test removed from support incorrectly')
    require(bool(train['query_in_support'].all()),'training query not treated as LOO')
    # Strict first-maximum ACC selection over every recorded epoch.
    history=cp['loss_history']; best_acc=-1.; best_epoch=None
    for row in history:
        if float(row['metrics']['acc'])>best_acc:best_acc=float(row['metrics']['acc']);best_epoch=int(row['epoch'])
    require(best_epoch==cp['best_epoch_1based']==task['best_epoch_1based'],'strict ACC first-best selection violated')
    require(int(latest['next_epoch'])==int(task['epochs_executed']),'latest final epoch mismatch')
    # Verify full final state can restore all tensors and optimizers, then restore best.
    saved_best=R._snapshot(state,module,opt)
    R._restore(state,module,opt,latest['current'])
    R._compare_tree(R._cpu_copy(R._snapshot(state,module,opt)),R._cpu_copy(latest['current']),'latest_readback')
    R._restore(state,module,opt,saved_best)
    e74_task=json.loads((R.BASE/'.runtime/tasks'/f'{R.BASE_GROUP}_f{fold:02d}.json').read_text())
    rng={}
    for name,reference in [('e74_4',e74_task),('e78_8',ref_task)]:
        byepoch={int(r['epoch']):r for r in reference['rng_trace']};common=0;mismatches=0
        for row in task['rng_trace']:
            if int(row['epoch']) in byepoch:
                common+=1; other=byepoch[int(row['epoch'])]
                mismatches+=sum(row[k]!=other[k] for k in ('rng_before','rng_after_train','rng_after_eval'))
        rng[name+'_common_epochs']=common; rng[name+'_mismatches']=mismatches
    require(rng['e74_4_mismatches']==0 and rng['e78_8_mismatches']==0,'public RNG mismatch')
    protections=[r['gradient_protection'] for r in history]
    first_grad=history[0]['gradient_audit_first_epoch']['classification']
    require(first_grad['darg_gfr']>0 and first_grad['cam']>0 and first_grad['module']>0,'first-step gradient disconnected')
    require(all(v>0 and np.isfinite(v) for v in first_grad['module_by_name'].values()),'new parameter gradient disconnected')
    train_y=arr(y[torch.as_tensor(support,device=y.device)]);test_y=arr(y[torch.as_tensor(test,device=y.device)])
    trainacc=float((arr(train['semantic_logp']).argmax(1)==train_y).mean());testacc=float((arr(d['semantic_logp']).argmax(1)==test_y).mean())
    old_ref=reference_module()
    source=R.CFG['group_specs'][group]['prompt_source']
    old_module=old_ref.initialize(None if module.mode=='visual' else module.anchors,
                                  None if module.mode=='visual' else module.prompt_classes,
                                  module.mode,source,fold,h.device)
    old_module.load_state_dict(module.state_dict(),strict=True)
    with torch.no_grad():
        self_inclusive=old_module(h,y,support,support,weights,basep,training=False)
    selfacc=float((arr(self_inclusive['semantic_logp']).argmax(1)==train_y).mean())
    # Ordinary full-support reference on the same live bank quantifies self participation.
    a=train['attention_features']; means=torch.stack([a[torch.as_tensor(support[train_y==c],device=a.device)].mean(0) for c in (0,1)])
    delta=means[0]-means[1]; full_dirs=unit(module.prototype_direction(unit(torch.stack([delta,-delta]))))
    loo_dirs=train['prototypes']
    # Renormalize in float64 before acos: at <0.1 degree, float32 norm
    # roundoff can be comparable to the angular change being measured.
    angle=torch.rad2deg(torch.acos((unit(loo_dirs.double())*unit(full_dirs.double())[None]).sum(-1).clamp(-1,1)))
    mean_cos=F.cosine_similarity(train['prototype_means'].double(),means[None].double(),dim=-1)
    counts=np.bincount(train_y,minlength=2)
    mech={'group':group,'fold':fold,'support_count':len(support),'support_asd':int(counts[0]),'support_td':int(counts[1]),
          'historical_train_test_overlap_1based':json.dumps((np.intersect1d(state['train_np'],test)+1).tolist()),
          'fold2_overlap_excluded':not bool(np.intersect1d(support,np.intersect1d(state['train_np'],test)).size),
          'own_support_mass_asd':1/float(counts[0]),'own_support_mass_td':1/float(counts[1]),
          'loo_mean_cosine_change_mean':float((1-mean_cos).mean().detach().cpu()),
          'loo_direction_angle_mean_deg':float(angle.mean().detach().cpu()),
          'loo_direction_angle_p50_deg':float(torch.quantile(angle.flatten(),.5).detach().cpu()),
          'loo_direction_angle_p95_deg':float(torch.quantile(angle.flatten(),.95).detach().cpu()),
          'loo_direction_asd_query_angle_deg':float(angle[torch.as_tensor(train_y==0,device=angle.device)].mean().detach().cpu()),
          'loo_direction_td_query_angle_deg':float(angle[torch.as_tensor(train_y==1,device=angle.device)].mean().detach().cpu()),
          'train_aux_accuracy':trainacc,'test_aux_accuracy':testacc,'train_test_aux_gap_pp':100*(trainacc-testacc),
          'same_checkpoint_self_inclusive_train_aux_accuracy':selfacc,
          'same_checkpoint_loo_vs_self_inclusive_aux_logit_mae':float((train['semantic_logits']-self_inclusive['semantic_logits']).abs().mean().cpu()),
          'same_checkpoint_loo_vs_self_inclusive_aux_hard_flips':int((arr(train['semantic_logp']).argmax(1)!=arr(self_inclusive['semantic_logp']).argmax(1)).sum()),
          'train_darg_accuracy':float(train_darg['acc']),'test_darg_accuracy':float(met['acc']),
          'protected_epochs':len(protections),'conflict_epochs':sum(p['conflict'] for p in protections),
          'projected_epochs':sum(p['projected'] for p in protections),'scaled_epochs':sum(p['scaled'] for p in protections),
          'mean_gradient_cosine':float(np.mean([p['gradient_cosine'] for p in protections])),
          'mean_aux_ratio_raw':float(np.mean([p['aux_norm_ratio_raw'] for p in protections])),
          'mean_aux_ratio_safe':float(np.mean([p['aux_norm_ratio_safe'] for p in protections])),
          'anchor_sha256':module.anchor_sha(),'anchor_frozen':True,'beta':json.dumps(arr(module.calibration_strength).tolist()),
          'semantic_loss_first':history[0]['losses']['semantic_loss'],'semantic_loss_last':history[-1]['losses']['semantic_loss'],
          'darg_cls_loss_first':history[0]['losses']['base_cls'],'darg_cls_loss_last':history[-1]['losses']['base_cls']}
    metricrow={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS',**{k:float(met[k]) for k in KEYS},
               'ba':float(met['hard_auc']),'tp':int(met['asd_tp']),'tn':int(met['td_tn']),'fp':int(met['td_fp']),'fn':int(met['asd_fn']),
               'best_epoch_1based':cp['best_epoch_1based'],'epochs_executed':task['epochs_executed'],
               'training_epochs':len(history),'wall_seconds':task['wall_seconds'],'peak_gpu_bytes':task['peak_gpu_bytes'],
               'checkpoint_sha256':task['checkpoint_sha256']}
    auditrow={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS',**rng,'latest_restored_exact':True,
              'initial_parameter_sha_matches_e78_8':True,'first_step_auxiliary_gradients_nonzero':True,
              'strict_acc_selection_verified':True,'anchors_unchanged':True,'support_and_test_label_isolation':True,
              'warmup_sha256':ident['warmup_sha256'],'checkpoint_sha256':task['checkpoint_sha256']}
    interventions=[]
    if group==R.GROUPS[1]:
        original=module.anchors.detach().clone(); base_sem=arr(d['semantic_logp'].exp()[:,0]); base_darg=arr(d['final_logp'].exp()[:,0])
        for name in ('no_calibration','anchor_only','anonymous_text','class_swap'):
            if name=='anonymous_text':module.anchors.copy_(R.save_text_cache()['anonymous_anchors'].to(module.anchors))
            try:
                with torch.no_grad():changed=module(h,y,support,test,weights,basep,training=False,intervention='full' if name=='anonymous_text' else name)
            finally:module.anchors.copy_(original)
            sem=arr(changed['semantic_logp'].exp()[:,0]); pred=arr(changed['semantic_logp']).argmax(1)
            interventions.append({'group':group,'fold':fold,'intervention':name,
                'mean_abs_semantic_probability_change':float(np.abs(sem-base_sem).mean()),
                'semantic_hard_flips':int((pred!=arr(d['semantic_logp']).argmax(1)).sum()),'aux_test_accuracy':float((pred==test_y).mean()),
                'mean_abs_darg_probability_change':float(np.abs(arr(changed['final_logp'].exp()[:,0])-base_darg).mean()),
                'darg_hard_flips':int((arr(changed['final_logp']).argmax(1)!=arr(d['final_logp']).argmax(1)).sum())})
    del state,module,opt,cp,latest,out,d,train,saved_best,old_module,self_inclusive
    torch.cuda.empty_cache()
    return metricrow,mech,auditrow,interventions

def comparison_rows(metrics):
    e8=A.read_csv(E8/'results/fold_metrics.csv');base=A.read_csv(R.BASE/'results/fold_metrics.csv')
    comparisons=[]
    for group in R.GROUPS:
        ref=R.CFG['group_specs'][group]['reference_group']
        comparisons.extend([(group,ref,metrics+e8),(group,R.BASE_GROUP,metrics+base)])
    comparisons.extend([(R.GROUPS[1],R.GROUPS[0],metrics),(R.GROUPS[1],R.GROUPS[2],metrics)])
    rows=[]
    for left,right,bank in comparisons:
        l={int(r['fold']):r for r in bank if r['group']==left}; rr={int(r['fold']):r for r in bank if r['group']==right}
        require(len(l)==len(rr)==10,'paired fold count mismatch')
        for scope,folds in [('all10',list(range(1,11))),('exclude_fold2',[f for f in range(1,11) if f!=2])]:
            for k in KEYS:
                d=np.asarray([100*(float(l[f][k])-float(rr[f][k])) for f in folds]);mean=float(d.mean());sd=float(d.std(ddof=1));half=float(t.ppf(.975,len(d)-1)*sd/np.sqrt(len(d)))
                rows.append({'comparison':left+' - '+right,'scope':scope,'metric':k,'n_folds':len(folds),
                  'mean_difference_pp':mean,'ci95_low_pp':mean-half,'ci95_high_pp':mean+half,
                  'wins':int((d>1e-12).sum()),'ties':int((np.abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),
                  'fold_differences_pp':json.dumps({str(f):float(v) for f,v in zip(folds,d)})})
    return rows

def disk_resume_probe():
    group=R.GROUPS[1];fold=1
    state,_,_,support,module,opt,_=R._initialized_task(fold,group,None)
    first=R._replay_step(state,module,opt,support)
    saved=R._snapshot(state,module,opt)
    probe=HERE/'runtime/serialized_resume_probe.pt'
    require(not probe.exists(),'resume probe path already occupied')
    try:
        R.atomic_torch(probe,saved)
        expected=R._replay_step(state,module,opt,support)
        with torch.no_grad():
            expected_out,expected_details=R._forward(state,module,training=False,support=support,query=state['test_np'])
            expected_outputs=R._cpu_copy({'darg_logp':expected_out['logp'],
                                         'semantic_logits':expected_details['semantic_logits'],
                                         'semantic_logp':expected_details['semantic_logp']})
        loaded=torch.load(probe,map_location='cpu',weights_only=False)
        R._restore(state,module,opt,loaded)
        actual=R._replay_step(state,module,opt,support)
        R._compare_tree(R._cpu_copy(expected),R._cpu_copy(actual),'serialized_two_step_replay')
        with torch.no_grad():
            actual_out,actual_details=R._forward(state,module,training=False,support=support,query=state['test_np'])
            actual_outputs=R._cpu_copy({'darg_logp':actual_out['logp'],
                                       'semantic_logits':actual_details['semantic_logits'],
                                       'semantic_logp':actual_details['semantic_logp']})
        R._compare_tree(expected_outputs,actual_outputs,'serialized_two_step_full_outputs')
        digest=R.sha_file(probe)
    finally:
        probe.unlink(missing_ok=True)
    del state,module,opt,saved,expected,actual,first
    torch.cuda.empty_cache()
    return {'status':'PASS','fold':fold,'group':group,'serialized_probe_sha256':digest,
            'models_four_optimizers_rng_losses_predictions_and_protection_exact':True,
            'full_darg_and_semantic_probability_outputs_exact':True,
            'original_best_latest_checkpoints_untouched':True}

def final_report(status,metrics,mechs,comps,intervs):
    def stats(rows,key):
        v=np.asarray([float(r[key]) for r in rows])*100
        return f'{v.mean():.4f} ± {v.std(ddof=0):.4f}'
    lines=['# Experiment 78.9 — LOO-FAPC final report','',f"STATUS: {status['status']}",'',
      f"单一正式目录：`{HERE}`。preferred OSSFS mkdir errno=5，回退/model。30/30正式训练，10 worker，墙钟{status['training_wall_seconds']/60:.2f}分钟；任务时间合计{status['training_task_seconds_sum']/3600:.3f}小时；单任务allocated峰值{max(r['peak_gpu_bytes'] for r in metrics)/2**30:.3f}GiB。",
      f"冻结训练源SHA `{status['source_sha256']}`；E78.8最终有效修正版 `corrected-neutral-anonymous-and-main-gradient-v2`。固定文本缓存SHA `{status['fixed_anchor_cache_sha256']}`。",
      '', '## 方法和来源','',
      '唯一基座E74.4-G2，全部任务独立从E61 epoch-10参数、三个优化器和全部RNG恢复。复用服务器真实soft_positive_dcr.py完整损失；原始weighted NLL、图损失与pair/adv/preserve权重保持。仅将训练支持查询的辅助原型改成向量化(sum-self)/(count-1)；ASD/TD两个竞争方向共用同一去除自身的支持集。测试查询始终使用完整合法训练支持。患者查询路径及其他支持患者原型路径保留梯度，自身支持路径梯度抵消。',
      '患者投影、疾病方向映射、beta、固定RN50锚点、辅助权重0.1、保持权重0.01、温度0.25、尺度10、20%梯度上限和私有初始化seed788000+fold均与E78.8相同。正式预测仍为DARG MLP，没有语义残差分类。',
      'SPromptGL提供GFR/CAM及八条Prompt依据；BrainPrompt提供两条ABIDE疾病级原文，TC映射TD。冻结参考借鉴KgCoOp，单侧CAM144保护借鉴PCGrad，但不是其多任务优化器复现。四篇论文和官方源码已读取；BrainPrompt文本文件SHA ce6a7bb9755c26cd9643b6867bac5d1496e93bf9a0d7d8d90944dd3b119c431c，与缓存原文核对。',
      '来源：[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[官方代码](https://github.com/wanxixi11/SPromptGL)；[BrainPrompt论文](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、[官方文本](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt)；[KgCoOp](https://arxiv.org/abs/2303.13283)、[代码](https://github.com/htyao89/KgCoOp)；[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)、[代码](https://github.com/tianheyu927/PCGrad)。',
      '', '## 六指标','', '均值 ± population SD，%；ASD=0，TD=1。Hard-AUC/BA和ASD概率ROC-AUC分别计算。',
      '', '|组别|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for g in R.GROUPS:lines.append('|'+g+'|'+'|'.join(stats([r for r in metrics if r['group']==g],k) for k in KEYS)+'|')
    historical=A.read_csv(R.BASE/'results/fold_metrics.csv')+A.read_csv(E8/'results/fold_metrics.csv')
    lines+=['','历史同协议参考（只读，不重训）：','','|组别|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for g in [R.BASE_GROUP]+[R.CFG['group_specs'][g]['reference_group'] for g in R.GROUPS]:lines.append('|'+g+'|'+'|'.join(stats([r for r in historical if r['group']==g],k) for k in KEYS)+'|')
    lines+=['','## 配对比较','','差值为百分点，95%为十折配对差的描述性t区间。全部六指标逐折差与W/T/L见paired_comparisons.csv。','','|比较|ACC差[95%区间]|W/T/L|Hard-AUC差|Prob-AUC差|去Fold2 ACC差|','|---|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(r['comparison'] for r in comps):
        def c(k,scope='all10'):return next(r for r in comps if r['comparison']==name and r['metric']==k and r['scope']==scope)
        a=c('acc');lines.append(f"|{name}|{a['mean_difference_pp']:+.4f} [{a['ci95_low_pp']:+.4f},{a['ci95_high_pp']:+.4f}]|{a['wins']}/{a['ties']}/{a['losses']}|{c('hard_auc')['mean_difference_pp']:+.4f}|{c('prob_auc')['mean_difference_pp']:+.4f}|{c('acc','exclude_fold2')['mean_difference_pp']:+.4f}|")
    lines+=['','去Fold 2九折结果：','','|组别|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for g in R.GROUPS:lines.append('|'+g+'|'+'|'.join(stats([r for r in metrics if r['group']==g and r['fold']!=2],k) for k in KEYS)+'|')
    lines+=['','## LOO、泛化差距与梯度机制','','|组别|LOO方向平均角度°|P95均值°|辅助训练ACC|辅助测试ACC|差距pp|冲突率|缩放率|安全梯度比|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in R.GROUPS:
        mm=[r for r in mechs if r['group']==g];n=sum(r['protected_epochs'] for r in mm)
        lines.append(f"|{g}|{np.mean([r['loo_direction_angle_mean_deg'] for r in mm]):.5f}|{np.mean([r['loo_direction_angle_p95_deg'] for r in mm]):.5f}|{np.mean([r['train_aux_accuracy'] for r in mm])*100:.4f}|{np.mean([r['test_aux_accuracy'] for r in mm])*100:.4f}|{np.mean([r['train_test_aux_gap_pp'] for r in mm]):.4f}|{sum(r['conflict_epochs'] for r in mm)/n*100:.2f}%|{sum(r['scaled_epochs'] for r in mm)/n*100:.2f}%|{np.mean([r['mean_aux_ratio_safe'] for r in mm]):.5f}|")
    old=A.read_csv(E8/'results/mechanism.csv')
    lines+=['','与E78.8辅助分类差距比较：','','|组别|E78.8训练ACC|E78.9训练ACC|E78.8测试ACC|E78.9测试ACC|','|---|---:|---:|---:|---:|']
    for g in R.GROUPS:
        ref=R.CFG['group_specs'][g]['reference_group'];mm=[r for r in mechs if r['group']==g];oo=[r for r in old if r['group']==ref]
        lines.append(f"|{g}|{np.mean([float(r['train_aux_accuracy']) for r in oo])*100:.4f}|{np.mean([r['train_aux_accuracy'] for r in mm])*100:.4f}|{np.mean([float(r['test_aux_accuracy']) for r in oo])*100:.4f}|{np.mean([r['test_aux_accuracy'] for r in mm])*100:.4f}|")
    lines+=['','在同一E78.9最佳Checkpoint上恢复包含自身的训练原型，仅用于量化LOO的即时作用：','','|组别|LOO训练辅助ACC|包含自身训练辅助ACC|辅助Logits MAE|辅助硬翻转总数|','|---|---:|---:|---:|---:|']
    for g in R.GROUPS:
        mm=[r for r in mechs if r['group']==g]
        lines.append(f"|{g}|{np.mean([r['train_aux_accuracy'] for r in mm])*100:.4f}|{np.mean([r['same_checkpoint_self_inclusive_train_aux_accuracy'] for r in mm])*100:.4f}|{np.mean([r['same_checkpoint_loo_vs_self_inclusive_aux_logit_mae'] for r in mm]):.7f}|{sum(r['same_checkpoint_loo_vs_self_inclusive_aux_hard_flips'] for r in mm)}|")
    masses=[r[k] for r in mechs for k in ('own_support_mass_asd','own_support_mass_td')]
    lines += ['',f"合法支持共{min(r['support_count'] for r in mechs)}–{max(r['support_count'] for r in mechs)}人，单个患者在对应类均值中的直接权重为{min(masses)*100:.4f}%–{max(masses)*100:.4f}%。LOO方向角度在重新归一化后使用float64计算；自身支持特征梯度为零的结论来自分离路径自动微分测试，不能外推为上游传导图中不存在患者间信息。",'',
              '|组别|梯度余弦折均值|原始辅助/主梯度范数比|安全范数比|冲突epoch|投影epoch|缩放epoch|','|---|---:|---:|---:|---:|---:|---:|']
    for g in R.GROUPS:
        mm=[r for r in mechs if r['group']==g]
        lines.append(f"|{g}|{np.mean([r['mean_gradient_cosine'] for r in mm]):.6f}|{np.mean([r['mean_aux_ratio_raw'] for r in mm]):.6f}|{np.mean([r['mean_aux_ratio_safe'] for r in mm]):.6f}|{sum(r['conflict_epochs'] for r in mm)}|{sum(r['projected_epochs'] for r in mm)}|{sum(r['scaled_epochs'] for r in mm)}|")
    lines+=['','固定G2 Checkpoint干预：','','|干预|辅助ASD概率平均绝对变化|辅助硬翻转|最终DARG硬翻转|','|---|---:|---:|---:|']
    for name in dict.fromkeys(r['intervention'] for r in intervs):
        rr=[r for r in intervs if r['intervention']==name]
        lines.append(f"|{name}|{np.mean([r['mean_abs_semantic_probability_change'] for r in rr]):.6f}|{sum(r['semantic_hard_flips'] for r in rr)}|{sum(r['darg_hard_flips'] for r in rr)}|")
    lines+=['','文本替换影响辅助分类，但最终DARG在冻结Checkpoint推理时不直接读取文本，因此零最终翻转是预期，不能据此否定训练期辅助梯度作用。医学内容价值以G2/G3独立训练为准。',
      '', '## 审计和科研判断','',f"30/30 best独立前向概率和六指标一致；30/30 latest全部模型/四优化器/RNG读回逐值一致。E74.4共有epoch {status['rng_common_epochs_vs_e74_4']}，不匹配 {status['rng_mismatches_vs_e74_4']}；E78.8共有epoch {status['rng_common_epochs_vs_e78_8']}，不匹配 {status['rng_mismatches_vs_e78_8']}。10/10 base-off parity，30/30私有初始化参数buffer匹配，LOO显式参考与自梯度抵消、完整Hook全参数合成、零辅助主梯度逐值恢复和两步恢复均通过。所有冻结输入按manifest复核。",
      f"判定：`{'`; `'.join(status['scientific_conclusions'])}`。",
      '判定区分：LOO实现生效不等于诊断增益；辅助训练ACC降低不等于辅助测试泛化改善；若仅相对E78.8医学组有局部改善而未超过匿名/Visual-only/基座，仍不能宣称完整方案GO。',
      '', '## 已知限制','', '单seed，保留val=test、测试ACC选Checkpoint、CV前预处理/RFE及全患者transductive图。Fold2历史重叠仅从新增支持/监督中排除，原历史路径未变。LOO排除的是投影后该患者自身的直接原型贡献；上游传导图仍可传播患者间信息，不能将此等同整体无泄漏。十折训练集重叠，描述性置信区间不是独立临床显著性结论。不自动开展E78.10。',
      '', '## 实现纠错记录','', '预检发现CUDA index_add_聚合导致约1e-6的不确定累加误差，正式训练前改为固定支持顺序的分类归约。该修复未改变LOO公式、参数、初始化、损失或超参数。所有正式任务均使用最终冻结SHA。report_only.py是审计/汇总补充，不修改冻结训练源码或Checkpoint。','']
    def delta(left,right,key='acc',scope='all10'):
        return next(c['mean_difference_pp'] for c in comps if c['comparison']==left+' - '+right and c['metric']==key and c['scope']==scope)
    med,vis,anon=R.GROUPS[1],R.GROUPS[0],R.GROUPS[2]
    lines+=['## 结果解释','',
        f"医学LOO组相对E78.8医学组的ACC变化{delta(med,R.CFG['group_specs'][med]['reference_group']):+.4f}pp；相对Visual-only {delta(med,vis):+.4f}pp、匿名文本 {delta(med,anon):+.4f}pp、DARG基座 {delta(med,R.BASE_GROUP):+.4f}pp。相对基座Hard-AUC变化{delta(med,R.BASE_GROUP,'hard_auc'):+.4f}pp，Prob-AUC变化{delta(med,R.BASE_GROUP,'prob_auc'):+.4f}pp；去Fold2后的ACC差{delta(med,R.BASE_GROUP,scope='exclude_fold2'):+.4f}pp。", 
        '医学内容的独立增量需同时优于匹配匿名对照和Visual-only；相对低于基座的旧医学方案改善，不能单独构成诊断收益证据。梯度保护的触发统计只说明算法工作，本实验没有独立改变保护规则，不能据此证明20%上限最优。',
        ('全部方向性GO条件满足，仍仅是历史协议下的探索性候选，需更严格的独立验证。' if 'GO_CANDIDATE' in status['scientific_conclusions'] else '完整LOO-FAPC不满足预定GO方向性条件，建议停止通过当前疾病原型与Prompt辅助监督改善DARG的路线；本次不自动开展E78.10。'),
        '', '## 复现入口','',
        '`python run.py --train --workers 10` 使用冻结训练实现并跳过已完成且SHA匹配的任务；独立审计及报告使用 `python report_only.py`。冻结audit.py保留原字节，最终审计/汇总入口明确为report_only.py。身份清单见manifest.json，逐折独立重载见results/checkpoint_audit.csv，完整预检见results/preflight.json。','']
    (R.RESULTS/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--available',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_tests(),indent=2));return
    frozen=R.verify_frozen_source();tests=self_tests();metrics=[];mechs=[];audits=[];intervs=[];errors=[]
    for g in R.GROUPS:
        for fold in range(1,11):
            taskpath,bestpath,_,_=R.task_paths(g,fold)
            if args.available and not taskpath.is_file():continue
            try:
                cache_path=HERE/'runtime/independent_audit'/f'{g}_f{fold:02d}.json'
                identity={'checkpoint_sha256':R.sha_file(bestpath),'source_sha256':frozen['source_sha256'],
                          'auditor_sha256':R.sha_file(Path(__file__))}
                cached=json.loads(cache_path.read_text()) if cache_path.exists() else None
                if cached and cached.get('identity')==identity:
                    m,me,a,iv=cached['metric'],cached['mechanism'],cached['audit'],cached['interventions']
                else:
                    m,me,a,iv=evaluate_task(g,fold,frozen['source_sha256'])
                    R.atomic_json(cache_path,{'identity':identity,'metric':m,'mechanism':me,'audit':a,'interventions':iv})
                metrics.append(m);mechs.append(me);audits.append(a);intervs.extend(iv)
                print(f'AUDIT PASS {g} fold={fold}',flush=True)
            except Exception as exc:
                import traceback;traceback.print_exc();errors.append({'group':g,'fold':fold,'error':repr(exc)})
    if args.available:
        print(json.dumps({'available_audited':len(audits),'errors':errors},indent=2));return
    R.atomic_json(R.RESULTS/'audit_status.json',{'status':'PARTIAL','independent_checkpoint_audits':len(audits),'audit_errors':errors})
    require(not errors and len(audits)==30,'independent audit did not pass 30/30')
    comps=comparison_rows(metrics)
    resume=disk_resume_probe()
    def diff(g,r,k='acc',scope='all10'):return next(c['mean_difference_pp'] for c in comps if c['comparison']==g+' - '+r and c['metric']==k and c['scope']==scope)
    medical=R.GROUPS[1];visual=R.GROUPS[0];anonymous=R.GROUPS[2];ref=R.CFG['group_specs'][medical]['reference_group']
    gain=diff(medical,R.BASE_GROUP)>0 and diff(medical,anonymous)>0 and diff(medical,visual)>0 and diff(medical,R.BASE_GROUP,'hard_auc')>=0 and diff(medical,R.BASE_GROUP,'prob_auc')>=0 and diff(medical,R.BASE_GROUP,scope='exclude_fold2')>0
    conclusions=['GO_CANDIDATE' if gain else 'NO_GO_LOO_FAPC']
    if diff(visual,R.CFG['group_specs'][visual]['reference_group'])>0 and diff(medical,ref)>0:conclusions.append('LOO_EFFECT_SUPPORTED')
    else:conclusions.append('LOO_EFFECT_NOT_SUPPORTED')
    if diff(medical,anonymous)<=0 or diff(medical,visual)<=0:conclusions.append('MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED')
    if not gain:conclusions.append('GRADIENT_PROTECTION_ACTIVE_BUT_NO_GAIN')
    controller=json.loads((R.LOGS/'controller.json').read_text())
    status={'status':'COMPLETE_AUDITED','valid_run_revision':R.CFG['run_revision'],'expected_tasks':30,'trained_tasks':30,'independent_checkpoint_audits':30,'latest_states_checked':30,
       'training_workers':len(controller['workers']),'training_wall_seconds':controller['finished']-controller['started'],
       'training_task_seconds_sum':sum(m['wall_seconds'] for m in metrics),'training_epochs':sum(m['training_epochs'] for m in metrics),
       'source_sha256':frozen['source_sha256'],'report_only_source_sha256':R.sha_file(Path(__file__)),
       'fixed_anchor_cache_sha256':R.sha_file(HERE/'runtime/text_embeddings.pt'),'anchors_unchanged':True,
       'rng_common_epochs_vs_e74_4':sum(a['e74_4_common_epochs'] for a in audits),'rng_mismatches_vs_e74_4':sum(a['e74_4_mismatches'] for a in audits),
       'rng_common_epochs_vs_e78_8':sum(a['e78_8_common_epochs'] for a in audits),'rng_mismatches_vs_e78_8':sum(a['e78_8_mismatches'] for a in audits),
       'audit_errors':[],'base_off_parity_passed':10,'scientific_conclusions':conclusions,'auditor_self_tests':tests['status'],
       'serialized_checkpoint_two_step_replay':resume,
       'root_data_and_historical_dependencies_unchanged':R.verify_frozen_source()['source_sha256']==frozen['source_sha256']}
    for name,rows in [('fold_metrics.csv',metrics),('mechanism.csv',mechs),('checkpoint_audit.csv',audits),('paired_comparisons.csv',comps),('interventions.csv',intervs)]:A.write_csv(R.RESULTS/name,rows)
    final_report(status,metrics,mechs,comps,intervs)
    hand=Path('/model/spromptgl/experiment/handoff.md');marker='## Experiment 78.9 LOO-FAPC final '+frozen['source_sha256'][:12]
    if marker not in hand.read_text():
        report=(R.RESULTS/'report.md').read_text()
        with hand.open('a',encoding='utf-8') as stream:stream.write('\n'+marker+'\n\n'+report+'\n')
    status['handoff_update']='APPENDED_OR_ALREADY_PRESENT';status['handoff_path']=str(hand)
    R.atomic_json(R.RESULTS/'audit_status.json',status)
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()

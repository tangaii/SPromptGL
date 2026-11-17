#!/usr/bin/env python3
"""Independent best-checkpoint replay, metric reconstruction and E82 synthesis."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import csv
import fcntl
import json
import math
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score
import run as r
from pssec import EPS, SPECS


SPECTRAL_AUDITS = {}


def independent_spectral_audit(cache,state):
    """Recompute statistics with NumPy, separately from construction code."""
    fold=state['fold']
    if fold in SPECTRAL_AUDITS:return SPECTRAL_AUDITS[fold]
    idx=np.asarray(cache['legal_train']);z=cache['warmup_cam'].numpy()[idx].astype(np.float64)
    y=np.asarray(state['labels_np'])[idx]
    assert r.sha(cache['warmup_cam'])==cache['feature_sha']
    assert r.sha(torch.as_tensor(y))==cache['legal_label_sha']
    capture=[];before=r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    handle=state['cam'].Outputlayer.register_forward_pre_hook(lambda _m,args:capture.append(args[0].detach().cpu().clone()))
    try:
        with torch.no_grad():initial=r.runtime().forward(state,r.BASE_GROUP,training=False)
    finally:handle.remove()
    assert len(capture)==1 and torch.equal(capture[0],cache['warmup_cam'])
    assert [r.sha(v) for v in initial['original_adjs']]==cache['warmup_graph_sha']
    assert [r.sha(v) for v in initial['unified_adjs']]==cache['warmup_unified_graph_sha']
    assert r.base_runtime().rng_sha(r.base_runtime().pack_rng())==before
    del initial,capture
    assert np.isin(idx,state['train_np']).all()
    assert not np.isin(idx,np.setdiff1d(state['test_np'],state['train_np'])).any()
    if fold==2:assert not np.isin([376,406],idx).any()
    def cov(a):
        b=a-a.mean(0);return b.T@b/(len(a)-1)
    s=cov(z);s0=cov(z[y==0]);s1=cov(z[y==1]);dm=z[y==0].mean(0)-z[y==1].mean(0)
    n0,n1=int((y==0).sum()),int((y==1).sum())
    eye=np.eye(144);s0s=.9*s0+.1*np.trace(s0)/144*eye;s1s=.9*s1+.1*np.trace(s1)/144*eye
    pool=(n0*s0+n1*s1)/(n0+n1);pools=(n0*s0s+n1*s1s)/(n0+n1)
    errors={}
    for variant in ('O','I'):
        b=cache['spectra'][variant];u=b['U'].numpy();ug=u[:,:8];ud=u[:,8:];p=eye-ug@ug.T
        assert b['n_train']==len(idx) and b['n_asd']==n0 and b['n_td']==n1
        for name,expected in [('covariance',s),('cov_asd',s0),('cov_td',s1),('mu',z.mean(0)),('delta_mean',dm),('pooled',pool),('pooled_shrunk',pools)]:
            assert np.allclose(b[name].numpy(),expected,atol=1e-10,rtol=1e-10),(fold,variant,name)
        assert np.allclose(u.T@u,np.eye(16),atol=1e-10,rtol=0)
        assert (u[np.argmax(abs(u),axis=0),np.arange(16)]>=0).all()
        gv=np.linalg.eigvalsh(s)[::-1][:8]
        assert np.allclose(np.diag(ug.T@s@ug),gv,atol=1e-10,rtol=1e-9)
        if variant=='O':
            mat=p@(s0-s1)@p
            nu=abs(np.diag(u.T@(s0-s1)@u))/(np.diag(u.T@pool@u)+EPS)
        else:
            d=p@dm;c=p@(s0s-s1s)@p;dd=np.outer(d,d);cc=c@c
            a=dd/(np.trace(dd)+EPS) if np.trace(dd)>EPS else np.zeros_like(dd)
            bb=cc/(np.trace(cc)+EPS) if np.trace(cc)>EPS else np.zeros_like(cc)
            mat=.5*a+.5*bb
            denom=np.diag(u.T@pools@u)+EPS
            nu=.5*(u.T@dm)**2/denom+.5*abs(np.diag(u.T@(s0s-s1s)@u))/denom
        if b['fallback']:mat=p@s@p
        rayleigh=np.diag(ud.T@mat@ud)
        residual=float(np.linalg.norm(mat@ud-ud*rayleigh))
        assert residual<1e-8*max(1.,np.linalg.norm(mat))
        spectrum=np.linalg.eigvalsh((mat+mat.T)/2)
        score=np.sort(abs(spectrum) if variant=='O' and not b['fallback'] else spectrum)[::-1][:8]
        actual=abs(rayleigh) if variant=='O' and not b['fallback'] else rayleigh
        assert np.allclose(actual,score,atol=1e-8,rtol=1e-7)
        lam=np.maximum(np.diag(u.T@s@u),EPS)
        assert np.allclose(b['lambda'].numpy(),lam,atol=1e-10,rtol=1e-10)
        assert np.allclose(b['nu'].numpy(),nu,atol=1e-8,rtol=1e-9)
        errors[variant]={'eigen_equation_residual_frobenius':residual,'orthogonality_max_abs':float(abs(u.T@u-eye[:16,:16]).max())}
    SPECTRAL_AUDITS[fold]=errors
    return errors


def verify_feature_formula(module,d):
    z=d['z'];u=module.U;h=(z-module.mu)@u
    assert torch.equal(h,d['h'])
    hh=h/torch.sqrt(module.lam+EPS)
    assert torch.equal(hh,d['hhat'])
    if 'a' in d:
        evidence=torch.stack([hh,hh.abs(),module.loglam.expand_as(h),module.nu.expand_as(h)],-1)
        tokens=module.norm(module.direction(u.T)[None]+module.evidence(evidence))
        assert torch.equal(tokens,d['tokens'])
        q=F.normalize(module.text_proj(module.text),dim=-1);k=module.key(d['tokens'])
        score=torch.einsum('kd,njd->nkj',q,k)/math.sqrt(32)
        null=(q@module.null/math.sqrt(32))[None,:,None].expand(len(z),-1,-1)
        attention=torch.softmax(torch.cat([score,null],-1)/.5,-1)
        assert torch.equal(attention,d['a'])
        coverage=1-(1-attention[:,:,:16]).prod(1)
        sem=coverage/(coverage.sum(-1,keepdim=True)+EPS)
        assert torch.equal(sem,d['w_sem'])
        entropy=-(sem*(sem+EPS).log()).sum(-1)/math.log(16)
        assert torch.equal(entropy,d['entropy'])
        assert torch.allclose(attention.sum(-1),torch.ones_like(attention[:,:,0]),atol=1e-6,rtol=0)
    if 'w_sp' in d:
        v=torch.stack([hh.abs(),module.loglam.expand_as(h),module.nu.expand_as(h)],-1)
        assert torch.equal(module.spectral(v).squeeze(-1).softmax(-1),d['w_sp'])
    if module.mode=='spectral':expected_w=d['w_sp']
    elif module.mode=='semantic':expected_w=d['w_sem']
    else:
        beta=((1-d['a'][:,:,16].mean(-1))*(1-d['entropy']).clamp(.2,.8)).detach()
        expected_w=beta[:,None]*d['w_sem']+(1-beta[:,None])*d['w_sp']
    assert torch.equal(expected_w,d['w'])
    raw=((expected_w-1/16)*h)@u.T
    c=.08*z.norm(dim=-1,keepdim=True).clamp_min(EPS);norm=raw.norm(dim=-1,keepdim=True)
    expected_delta=raw*(c*torch.tanh(norm/c)/norm.clamp_min(EPS))
    if module.fold==2:expected_delta[[376,406]]=0
    assert torch.allclose(expected_delta,d['delta'],atol=1e-7,rtol=1e-6)
    return True


def replay(state,module,with_grad=False):
    """Audit owns its hook and invokes locked DARG directly, not training forward."""
    calls={'output':0,'cam':0,'gcn':[0,0,0,0]};details={}
    module.eval()
    def pre(_m,args):
        calls['output']+=1
        assert len(args)==2 and args[0].shape==(871,144)
        z,d=module(args[0]);details.update(d)
        return (z,args[1])
    handles=[state['cam'].Outputlayer.register_forward_pre_hook(pre),
             state['cam'].register_forward_hook(lambda *_:calls.__setitem__('cam',calls['cam']+1))]
    for i,m in enumerate(state['model'].gcn):
        handles.append(m.register_forward_hook(lambda _m,_a,_o,i=i:calls['gcn'].__setitem__(i,calls['gcn'][i]+1)))
    try:
        with torch.set_grad_enabled(with_grad):out=r.runtime().forward(state,r.BASE_GROUP,training=False)
    finally:
        for h in handles:h.remove()
    assert calls=={'output':1,'cam':1,'gcn':[1,1,1,1]}
    return out,details,calls


def independently_recompute(logp,state):
    idx=np.asarray(state['test_np']);y=np.asarray(state['labels_np'])[idx]
    prob=logp.detach().exp().cpu().numpy()[idx];pred=prob.argmax(-1)
    values={'acc':accuracy_score(y,pred),'hard_auc':roc_auc_score(np.eye(2)[y],np.eye(2)[pred]),
        'prob_auc':roc_auc_score(y,prob[:,1]),'sen':recall_score(y,pred,pos_label=0),
        'spe':recall_score(y,pred,pos_label=1),'asd_f1':f1_score(y,pred,pos_label=0)}
    assert abs(values['hard_auc']-balanced_accuracy_score(y,pred))<1e-12
    historical=r.base_runtime().classification_metrics(logp,state['test'],state['y'])
    assert all(abs(values[k]-historical[k])<1e-12 for k in r.METRICS)
    return {k:float(v) for k,v in values.items()},prob,pred,y


def quantiles(x):
    x=x.detach().cpu().double().reshape(-1)
    q=torch.quantile(x,torch.tensor([0.,.05,.25,.5,.75,.95,1.],dtype=x.dtype))
    return {'mean':float(x.mean()),'sd':float(x.std(unbiased=False)),
            **{k:float(v) for k,v in zip(('min','p05','p25','p50','p75','p95','max'),q)}}


def add_distribution(row,prefix,x):
    row.update({prefix+'_'+k:v for k,v in quantiles(x).items()})


def mechanism(group,fold,state,module,cache,d):
    variant=SPECS[group][0];s=cache['spectra'][variant]
    legal=torch.as_tensor(cache['legal_train'],device=state['x'].device)
    test=torch.as_tensor(state['test_np'],device=state['x'].device)
    u=module.U;z=d['z'];delta=d['delta'];warm=cache['warmup_cam'].to(z.device)
    centered=z[legal]-module.mu
    warm_centered=warm[legal]-module.mu
    energy=centered@u;warm_energy=warm_centered@u
    row={'group':group,'fold':fold,'spectrum':variant,**module.counts(),
        'variance_coverage_warmup':s['variance_coverage'],
        'mean_discrimination_sum':float(s['mean_discrimination'].sum()),
        'covariance_discrimination_sum':float(s['covariance_discrimination'].sum()),
        'basis_orthogonality_max_abs':s['orthogonality_max_abs'],
        'disease_fallback':s['fallback'],'mean_term_zero':s['mean_term_zero'],'cov_term_zero':s['cov_term_zero'],
        'minimum_global_top8_boundary_gap':float(s['global_gaps'][7]),
        'disease_top8_boundary_gap':float(s['disease_selection_gaps'][7]),
        'oi_projector_frobenius':cache['spectra']['comparison']['projector_frobenius'],
        'oi_principal_angles_degrees':json.dumps(cache['spectra']['comparison']['principal_angles_degrees'].tolist()),
        'cam_drift_relative_l2_legal_train':float((z[legal]-warm[legal]).norm()/warm[legal].norm()),
        'cam_cosine_with_warmup_legal_train':float(F.cosine_similarity(z[legal],warm[legal],dim=-1).mean()),
        'projection_energy_ratio_best_to_warmup':float(energy.square().sum()/warm_energy.square().sum().clamp_min(EPS)),
        'projection_energy_best_by_component':json.dumps(energy.square().mean(0).cpu().tolist()),
        'projection_energy_warmup_by_component':json.dumps(warm_energy.square().mean(0).cpu().tolist()),
        'projection_energy_after_residual_ratio':float((((z[legal]+delta[legal]-module.mu)@u).square().sum())/energy.square().sum().clamp_min(EPS)),
        'variance_fraction_at_best':float(energy.square().sum()/centered.square().sum().clamp_min(EPS)),
        'legal_train_count':len(legal)}
    rel=delta.norm(dim=-1)/z.norm(dim=-1).clamp_min(EPS)
    assert torch.isfinite(rel).all() and float(rel.max())<=.080001
    add_distribution(row,'residual_relative_norm_test',rel[test])
    add_distribution(row,'residual_relative_norm_all',rel)
    row['near_limit_fraction_test']=float((rel[test]>=.0792).float().mean())
    weights=d['w'][test]
    ent=-(weights*(weights+EPS).log()).sum(-1)
    add_distribution(row,'weight_effective_components_test',ent.exp())
    row['component_mean_weights_test']=json.dumps(weights.mean(0).cpu().tolist())
    row['component_argmax_counts_test']=json.dumps(torch.bincount(weights.argmax(-1),minlength=16).cpu().tolist())
    row['component_nu']=json.dumps(s['nu'].tolist())
    row['component_total_variance']=json.dumps(s['lambda'].tolist())
    row['component_mean_discrimination']=json.dumps(s['mean_discrimination'].tolist())
    row['component_covariance_discrimination']=json.dumps(s['covariance_discrimination'].tolist())
    if 'a' in d:
        a=d['a'][test];per_prompt=a[:,:,:16].mean(0)
        similarity=F.normalize(per_prompt,dim=-1)@F.normalize(per_prompt,dim=-1).T
        row['prompt_pairwise_selection_cosine']=json.dumps(similarity.cpu().tolist())
        row['prompt_pairwise_selection_cosine_offdiag_mean']=float(similarity[~torch.eye(8,dtype=torch.bool,device=z.device)].mean())
        row['prompt_component_distributions']=json.dumps(per_prompt.cpu().tolist())
        row['prompt_null_means']=json.dumps(a[:,:,16].mean(0).cpu().tolist())
        add_distribution(row,'null_test',d['null'][test]);add_distribution(row,'semantic_entropy_test',d['entropy'][test])
        add_distribution(row,'semantic_coverage_effective_test',1/(d['w_sem'][test].square().sum(-1)+EPS))
        if 'w_sp' in d:
            row['semantic_spectral_cosine_test']=float(F.cosine_similarity(d['w_sem'][test],d['w_sp'][test],dim=-1).mean())
            row['semantic_spectral_l1_test']=float((d['w_sem'][test]-d['w_sp'][test]).abs().sum(-1).mean())
    if 'beta' in d:add_distribution(row,'beta_test',d['beta'][test])
    if fold==2:
        assert torch.count_nonzero(delta[[376,406]])==0
        assert not np.isin([376,406],cache['legal_train']).any()
        row['fold2_residual_rows_377_407_exact_zero']=True
    return row


def flips(reference,candidate,y):
    return {'corrected':int(((reference!=y)&(candidate==y)).sum()),
        'harmed':int(((reference==y)&(candidate!=y)).sum()),
        'net':int((candidate==y).sum()-(reference==y).sum()),'flips':int((reference!=candidate).sum())}


def audit_one(group,fold):
    assert r.task_complete(group,fold)
    task_path,best_path,latest_path,log_path=r.paths(group,fold)
    task=json.loads(task_path.read_text());best=r.torch_load(best_path);latest=r.torch_load(latest_path)
    state,warmup,identity=r.prepare(fold);cache=r.torch_load(r.RT/f'spectrum_f{fold:02d}.pt')
    module,opt,init=r.make_module(state,group,cache)
    independent_spectral_audit(cache,state)
    assert init==best['new_parameter_sha']==task['new_parameter_sha']
    for c in (best,latest):
        assert c['group']==group and c['fold']==fold and c['seed']==0
        assert c['identity']==identity and c['config']==r.CFG and c['spectra_sha']==cache['spectra_sha']
        assert c['source_sha']==r.implementation_sha()
    assert task['best_sha']==r.sha_file(best_path) and task['latest_sha']==r.sha_file(latest_path)
    assert r.sha(cache['spectra'])==cache['spectra_sha'] and r.sha(cache['legal_train'])==cache['legal_train_sha']
    assert np.array_equal(cache['legal_train'],r.legal_train(state))
    r.load_bundle(state,module,opt,best['current']);r.base_runtime().restore_rng(best['rng'])
    assert torch.equal(module.text.cpu(),r.torch_load(r.RT/'prompts.pt')[SPECS[group][2]])
    if fold==2:assert int(state['labels_np'][376])==0 and int(state['labels_np'][406])==1
    rng_before=r.base_runtime().rng_sha(r.base_runtime().pack_rng());out,d,calls=replay(state,module,with_grad=True)
    assert rng_before==best['rng_sha']==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    values,prob,pred,y=independently_recompute(out['logp'],state)
    verify_feature_formula(module,d)
    assert np.array_equal(pred,best['prediction']) and np.allclose(prob,best['probability'],atol=1e-8,rtol=0)
    assert all(abs(values[k]-best['metrics'][k])<1e-12 and abs(values[k]-task['metrics'][k])<1e-12 for k in r.METRICS)
    assert r.sha(module.U)==r.sha(cache['spectra'][SPECS[group][0]]['U'].float())
    assert r.sha(module.mu)==r.sha(cache['spectra'][SPECS[group][0]]['mu'].float())
    for buffer,key in [('lam','lambda'),('loglam','loglambda_normalized'),('nu','nu_normalized')]:
        assert torch.equal(getattr(module,buffer).cpu(),cache['spectra'][SPECS[group][0]][key].float())
    logs=[json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    # A resumed run can legitimately replay uncommitted epochs. Last record is
    # retained only after requiring all duplicates to be exactly reproducible.
    records={}
    for line in logs:
        ep=line['epoch']
        if ep in records:assert records[ep]==line
        records[ep]=line
    assert sorted(records)==list(range(11,task['epochs_executed']+1))
    chosen=max(records,key=lambda e:(records[e]['acc'],-e))
    assert chosen==task['best_epoch']==best['best_epoch']
    assert records[chosen]['acc']==best['metrics']['acc']
    assert len(task['rng_trace'])==task['epochs_executed']-10
    reference={x['epoch']:x for x in r.historical_task(fold)['rng_trace']}
    parity=0
    for t in task['rng_trace']:
        for k in ('rng_before','rng_after_train','rng_after_eval'):assert records[t['epoch']][k]==t[k]
        if t['epoch'] in reference:assert t==reference[t['epoch']];parity+=1
    row=mechanism(group,fold,state,module,cache,d)
    named=[(prefix+'.'+name,p) for prefix,m in [('darg',state['model']),('cam',state['cam']),('pssec',module)] for name,p in m.named_parameters() if p.requires_grad]
    gradients=torch.autograd.grad(out['cls'],[p for n,p in named]+[d['z'],out['fused']],allow_unused=True)
    squared={n:float(g.detach().double().square().sum().cpu()) if g is not None else 0. for (n,p),g in zip(named,gradients)}
    path_norms={prefix:math.sqrt(sum(v for n,v in squared.items() if n.startswith(prefix+'.') or n==prefix)) for prefix in
        ('darg','cam','cam.Outputlayer','pssec.direction','pssec.evidence','pssec.norm','pssec.text_proj','pssec.key','pssec.null','pssec.spectral')}
    path_norms['cam144']=float(gradients[-2].detach().norm().cpu());path_norms['upstream_darg320']=float(gradients[-1].detach().norm().cpu())
    assert all(math.isfinite(v) for v in path_norms.values()) and all(path_norms[k]>0 for k in ('darg','cam','cam.Outputlayer','cam144','upstream_darg320'))
    row['best_classification_gradient_norms']=json.dumps(path_norms)
    row.update(best_epoch=best['best_epoch'],wall_seconds=task['wall_seconds'],peak_gpu_bytes=task['peak_gpu_bytes'],
        best_total_gradient_norms=json.dumps(best['gradients']),first_classification_gradient_norms=json.dumps(task['first_classification_gradients']))
    baseline=r.torch_load(r.runtime().EXP/'.runtime/checkpoints'/f'{r.BASE_GROUP}_f{fold:02d}.best.pt')
    assert baseline['identity']==identity
    baseline_pred=np.asarray(baseline['test_prediction'])
    bp=np.asarray(baseline['test_probability'])
    assert np.array_equal(bp.argmax(-1),baseline_pred)
    bm={'acc':accuracy_score(y,baseline_pred),'hard_auc':balanced_accuracy_score(y,baseline_pred),
        'prob_auc':roc_auc_score(y,bp[:,1]),'sen':recall_score(y,baseline_pred,pos_label=0),
        'spe':recall_score(y,baseline_pred,pos_label=1),'asd_f1':f1_score(y,baseline_pred,pos_label=0)}
    assert all(abs(bm[k]-baseline['metrics'][k])<1e-12 for k in r.METRICS)
    row['vs_b0']=json.dumps(flips(baseline_pred,pred,y))
    interventions=[]
    if group=='G5':
        original_text=module.text.clone()
        prompts=r.torch_load(r.RT/'prompts.pt')
        with torch.no_grad():module.text.copy_(prompts['anonymous'].to(module.text.device))
        changed,_,_=replay(state,module);cm,cp,cr,_=independently_recompute(changed['logp'],state)
        interventions.append({'intervention':'same_geometry_anonymous','mean_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).mean()),
            'max_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).max()),**flips(pred,cr,y),'metrics':cm,
            'flipped_rows_1based':(state['test_np'][cr!=pred]+1).tolist()})
        with torch.no_grad():module.text.copy_(original_text)
        mode=module.mode;module.mode='spectral'
        try:changed,_,_=replay(state,module)
        finally:module.mode=mode
        cm,cp,cr,_=independently_recompute(changed['logp'],state)
        interventions.append({'intervention':'semantic_off','mean_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).mean()),
            'max_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).max()),**flips(pred,cr,y),'metrics':cm,
            'flipped_rows_1based':(state['test_np'][cr!=pred]+1).tolist()})
        row['prompt_readonly_interventions']=json.dumps(interventions)
    # Restore every Latest state and all four optimizers, and inspect its RNG
    # continuity against the last actually executed epoch.
    r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng'])
    assert r.sha(r.bundle(state,module,opt))==latest['bundle_sha']
    assert r.base_runtime().rng_sha(r.base_runtime().pack_rng())==latest['rng_sha']==task['rng_trace'][-1]['rng_after_eval']
    assert latest['next_epoch']==task['epochs_executed']
    assert latest['wait']>200 or latest['next_epoch']==450
    restore='bundle_and_rng_exact'
    if group=='G5' and fold==1:
        def next_step():
            r.zero(state,opt);o=r.forward(state,module,True);o['total'].backward();r.step(state,opt)
            with torch.no_grad():o=r.forward(state,module,False)
            return r.sha(r.bundle(state,module,opt)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),o['logp'].detach().cpu().clone()
        a=next_step();r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng']);b=next_step()
        assert a[:2]==b[:2] and torch.equal(a[2],b[2]);restore='next_train_eval_update_state_prob_rng_exact'
    result={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','metrics':values,'probability_max_abs_error':float(abs(prob-best['probability']).max()),
        'best_epoch':best['best_epoch'],'best_sha':r.sha_file(best_path),'latest_sha':r.sha_file(latest_path),
        'initial_identity':identity,'new_parameter_sha':init,'spectra_sha':cache['spectra_sha'],
        'historical_common_rng_epochs':parity,'calls':calls,'latest_restore':restore,'mechanism':row,
        'predictions':pred.tolist(),'labels':y.tolist(),'b0_metrics':{k:float(baseline['metrics'][k]) for k in r.METRICS},
        'b0_predictions':baseline_pred.tolist(),'rng_trace':task['rng_trace'],'interventions':interventions}
    del state,module,opt,out,d,best,latest,cache;torch.cuda.empty_cache()
    print(f'AUDIT PASS {group}/{fold}',flush=True)
    return result


def paired_rows(audits):
    by={g:{a['fold']:a['metrics'] for a in audits if a['group']==g} for g in r.GROUPS}
    by['B0']={a['fold']:a['b0_metrics'] for a in audits if a['group']=='G5'}
    comparisons=[(g,'B0') for g in r.GROUPS]+[('G2','G1'),('G5','G2'),('G3','G1'),('G5','G3'),('G5','G4'),('G5','G6')]
    rows=[]
    for left,right in comparisons:
        for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
            for metric in r.METRICS:
                diff=np.array([(by[left][f][metric]-by[right][f][metric])*100 for f in folds])
                half=(2.2621571628540993 if len(folds)==10 else 2.306004135204166)*diff.std(ddof=1)/math.sqrt(len(folds))
                rows.append({'comparison':f'{left}_vs_{right}','scope':scope,'metric':metric,'mean_difference_pp':float(diff.mean()),
                    'wins':int((diff>1e-12).sum()),'ties':int((abs(diff)<=1e-12).sum()),'losses':int((diff< -1e-12).sum()),
                    'descriptive_95ci_low_pp':float(diff.mean()-half),'descriptive_95ci_high_pp':float(diff.mean()+half),
                    'fold_differences_pp':json.dumps({f:float(v) for f,v in zip(folds,diff)})})
    return rows,by


def synthesize(audits):
    paired,by=paired_rows(audits);summaries=[];fold_rows=[]
    execution=json.loads((r.RT/'execution.json').read_text())
    for g in ('B0',)+r.GROUPS:
        for f,m in by[g].items():fold_rows.append({'group':g,'fold':f,**m})
        for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
            row={'group':g,'name':r.CFG['groups'].get(g,'E74.4-G2-Positive-DCR (historical; no retraining)'), 'scope':scope,'folds':len(folds)}
            for m in r.METRICS:
                x=np.array([by[g][f][m]*100 for f in folds]);row[m+'_mean_percent']=float(x.mean());row[m+'_population_sd_percent']=float(x.std(ddof=0))
            summaries.append(row)
    for a in audits:
        if a['group']=='G5':
            other=next(x for x in audits if x['group']=='G2' and x['fold']==a['fold'])
            a['mechanism']['vs_g2']=json.dumps(flips(np.array(other['predictions']),np.array(a['predictions']),np.array(a['labels'])))
    r.write_csv(r.RESULTS/'fold_metrics.csv',fold_rows)
    r.write_csv(r.RESULTS/'summary.csv',summaries)
    r.write_csv(r.RESULTS/'paired_comparisons.csv',paired)
    r.write_csv(r.RESULTS/'mechanism.csv',[a['mechanism'] for a in audits])
    def comparison(l,rr,metric='acc',scope='all10'):
        return next(x for x in paired if x['comparison']==f'{l}_vs_{rr}' and x['metric']==metric and x['scope']==scope)
    conclusion=scientific_conclusion(by,comparison,audits)
    lines=['# Experiment 82 — PSSEC 原版与改进版正式训练',
        '', '状态：COMPLETE_AUDITED；60/60 正式训练、60/60 Best 独立审计。B0 仅读取历史 Checkpoint，未重新训练。',
        '', '唯一正式目录：`'+str(r.HERE)+'`。seed=0，原固定10折，871名患者；ASD=0、TD=1。',
        '', f"训练调度耗时 {execution['wall_seconds']:.1f}s；独立worker峰值 {execution['max_workers_observed']}；正式失败/重试记录 {len(execution.get('failures',[]))}。精确PyTorch allocated峰值按任务记录，含驱动开销的全设备采样另见 runtime/device_peak.json（采样从正式执行中途开始）。",
        '', '## 训练和源码核对',
        '', '直接调用经锁定 E74.4-G2 prepare/forward；每任务独立继承 E61 epoch10 完整状态、三优化器和4096关系对。原 OutputLayer 每个训练/评估前向仅执行一次，PSSEC 在其 CAM144 输入上增强。O/I 谱基来自合法训练患者的 warmup CAM，冻结后共享；Fold 2 新增统计、正则排除377/407，残差始终为零。',
        '', '核对[原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) GFR/CAM/SPGS/RES 公式与模型图，区分论文公式与历史代码未进入 total 的 ALoss；继承的是 PLNNoPEN。核对[SSC正文](https://papers.miccai.org/miccai-2026/paper/4444_paper.pdf)及[官方源码](https://github.com/Hanyu1124/ssc)：SSC 使用文本谱空间及硬候选预算，E82 使用患者谱方向和连续覆盖。外部版本、URL、SHA256及全部历史依赖见 manifest；未复制 SSC 分类器。',
        '', '六组、16方向、32维Token、0.5温度、0.5/0.5疾病矩阵、8%限幅和0.001残差损失全程冻结。各组同折新增参数初始化完全相同（820000+fold），隔离 Python/NumPy/CPU/GPU RNG；无逐epoch设种，无新增随机前向。G2/G5仅谱基及统计不同，G5/G6仅固定文本输入不同。禁用分支不计算、不进入优化器。',
        '', '## 分类成绩', '', '均值 ± population SD，单位%。Hard-AUC 等于本二分类口径 BA；Probability-AUC 使用连续概率。']
    for scope in ('all10','without_fold2'):
        lines += ['', '### '+('全10折' if scope=='all10' else '排除Fold 2的9折'),'',
                  '|组别|ACC|Hard-AUC/BA|Probability-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
        for row in summaries:
            if row['scope']==scope:
                lines.append('|'+row['group']+' '+row['name']+'|'+'|'.join(f"{row[m+'_mean_percent']:.4f} ± {row[m+'_population_sd_percent']:.4f}" for m in r.METRICS)+'|')
    lines += ['', '## 配对比较', '', '单位百分点，逐折差后求均值。全部6项指标、每折增益、W/T/L和描述性配对t区间见 paired_comparisons.csv；这些共享、非独立测试折及测试选模结果不构成无偏显著性检验。', '',
        '|比较|范围|ACC Δ|Hard-AUC Δ|Prob-AUC Δ|SEN Δ|SPE Δ|F1 Δ|ACC W/T/L|','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for left,right in [('G2','B0'),('G5','B0'),('G2','G1'),('G5','G2'),('G3','G1'),('G5','G3'),('G5','G4'),('G5','G6')]:
        for scope in ('all10','without_fold2'):
            c=comparison(left,right,scope=scope)
            lines.append(f'|{left} vs {right}|{scope}|'+ '|'.join(f"{comparison(left,right,m,scope)['mean_difference_pp']:+.4f}" for m in r.METRICS)+f"|{c['wins']}/{c['ties']}/{c['losses']}|")
    lines += ['', '## 谱与选择机制','', '每个Best的16维投影能量、ν/方差、O/I主角度、边界特征值间隔、CAM漂移、有效方向数、残差分位数、Null、八Prompt分布和两两相似度、β分位数及分支梯度见 mechanism.csv。禁用路径记为空，不能解释为有效机制。']
    for group in r.GROUPS:
        rows=[a['mechanism'] for a in audits if a['group']==group]
        avg=lambda key:float(np.mean([x[key] for x in rows if key in x]))
        first=rows[0]
        sentence=f"{group}：有效/名义参数 {first['effective_parameters']}/{first['nominal_parameters']}；warmup总体方差覆盖 {avg('variance_coverage_warmup')*100:.2f}%；均值判别和 {avg('mean_discrimination_sum'):.5g}、协方差判别和 {avg('covariance_discrimination_sum'):.5g}；最佳CAM相对漂移 {avg('cam_drift_relative_l2_legal_train'):.4f}；有效方向 {avg('weight_effective_components_test_mean'):.3f}/16；残差范数 {avg('residual_relative_norm_test_mean')*100:.4f}%，接近限幅患者 {avg('near_limit_fraction_test')*100:.3f}%；Best epoch {np.mean([x['best_epoch'] for x in rows]):.1f}（{min(x['best_epoch'] for x in rows)}–{max(x['best_epoch'] for x in rows)}）；单任务GPU峰值上界 {max(x['peak_gpu_bytes'] for x in rows)/2**30:.2f} GiB。"
        if 'null_test_mean' in first:sentence+=f" Null {avg('null_test_mean'):.5f}，语义熵 {avg('semantic_entropy_test_mean'):.5f}，Prompt选择两两cos {avg('prompt_pairwise_selection_cosine_offdiag_mean'):.5f}。"
        if 'beta_test_mean' in first:sentence+=f" β均值 {avg('beta_test_mean'):.5f}，语义–谱cos {avg('semantic_spectral_cosine_test'):.5f}。"
        lines+=['',sentence]
    g5=[a for a in audits if a['group']=='G5']
    for name in ('same_geometry_anonymous','semantic_off'):
        rows=[next(i for i in a['interventions'] if i['intervention']==name) for a in g5]
        lines+=['',f"G5只读干预 {name}：ASD概率平均绝对变化 {np.mean([x['mean_abs_asd_probability_change'] for x in rows]):.8g}；硬翻转 {sum(x['flips'] for x in rows)}，原错→干预对 {sum(x['corrected'] for x in rows)}，原对→干预错 {sum(x['harmed'] for x in rows)}。"]
    for name in ('vs_b0','vs_g2'):
        fx=[json.loads(a['mechanism'][name]) for a in g5]
        lines+=['',f"G5 {name}：纠错 {sum(x['corrected'] for x in fx)}，误伤 {sum(x['harmed'] for x in fx)}，净纠错 {sum(x['net'] for x in fx)}（按折计数；Fold 2重叠已披露）。"]
    lines += ['', '## 独立审计和限制', '',
        '独立代码重建全部Best并重放原DARG→CAM→PSSEC→OutputLayer；硬预测完全相同，概率最大误差≤1e-8；六项指标同时与历史评价工具和独立sklearn计算核对。每份Latest的模型、全部优化器和RNG恢复哈希通过；另对G5/Fold1执行下一训练及评估更新的双重恢复重放。严格ACC首次最大值与完整epoch记录核对。所有共同历史epoch RNG三元组一致，并核对组间共有epoch；无人工逐epoch RNG回填。根目录、数据、warmup和历史Best SHA均未改变。',
        '', '历史协议偏倚完整保留：val=test选模、CV前标准化和RFE、transductive患者图以及Fold 2训练/测试重叠。因此结果不能作为无偏外部泛化性能，排除Fold 2也不能消除其余偏倚。',
        '', 'Null仅为可学习拒绝候选，未校准为医学不确定性。16谱分量是潜在方向，不是脑区或原4×4×9局部。匿名文本在原8向量子空间内固定正交旋转，Gram保持；由于Wt可训练，该变换可被重参数化吸收。G5/G6只说明同几何训练差异，Prompt能改变概率、带来独立收益、医学含义被验证是三个不同命题，不能互相替代。',
        '', '## 预定科研判断', '', *conclusion['answers'], '', '判定：**'+conclusion['decision']+'**。本实验不追加第七组、PSSEC-v2或参数搜索。']
    (r.RESULTS/'report.md').write_text('\n'.join(lines)+'\n')
    return conclusion,summaries,paired


def scientific_conclusion(by,compare,audits):
    delta=lambda l,rr,m='acc':compare(l,rr,m)['mean_difference_pp']
    metric_support=lambda l,rr:sum(delta(l,rr,m)>1e-9 for m in r.METRICS)
    answers=[]
    answers.append(f"A 原版：G2相对B0 ACC {delta('G2','B0'):+.4f} pp；相对G1 {delta('G2','G1'):+.4f} pp。实际方向见六项配对指标，不能仅凭概率响应认定语义增益。")
    improved=delta('G3','G1')>0 and delta('G5','G2')>0
    answers.append(f"B 改进谱构建：G3−G1 ACC {delta('G3','G1'):+.4f} pp，G5−G2 {delta('G5','G2'):+.4f} pp；{'两个对照方向一致' if improved else '未获得两个对照一致改善'}。构建时均值判别方向纳入由固定训练统计证实；它是否转化为分类收益需以这些对照共同判断。")
    complementary=delta('G5','G3')>0 and delta('G5','G4')>0
    answers.append(f"C 谱–语义互补：G5−G3 {delta('G5','G3'):+.4f} pp，G5−G4 {delta('G5','G4'):+.4f} pp；{'ACC超过两条单独路径，但仍属偏倚协议内的描述性支持' if complementary else '未同时超过两条路径，不支持两条路径均有独立贡献'}。")
    c=compare('G5','G6');flips_n=sum(i['flips'] for a in audits if a['group']=='G5' for i in a['interventions'] if i['intervention']=='same_geometry_anonymous')
    answers.append(f"D 医学Prompt：G5−G6 ACC {c['mean_difference_pp']:+.4f} pp，W/T/L {c['wins']}/{c['ties']}/{c['losses']}；匿名替换硬翻转{flips_n}次。Wt的可训练性造成几何等价和语义不可辨识，不能据此证明独立医学知识或医学含义得到验证。")
    all_fail=all(delta(g,'B0')<=0 for g in r.GROUPS)
    # Conservative evidence gate: competitive classification plus both factor
    # contrasts, both single-path controls, and stable prompt-control benefit.
    prompt_support=c['mean_difference_pp']>0 and c['wins']>c['losses'] and flips_n>0
    competitive=delta('G5','B0')>0 and metric_support('G5','B0')>=4
    decision='GO（仅支持继续研究，仍需无偏验证）' if competitive and improved and complementary and prompt_support else 'NO-GO（作为疾病Prompt第二创新模块证据不足）'
    if all_fail:decision='NO-GO（全部新增组ACC均未优于B0）'
    elif delta('G3','B0')>0 and not prompt_support:decision='NO-GO（疾病Prompt创新）；仅可记录谱表示增益'
    answers.append('E MICCAI 2027候选：'+decision+'；ACC单点越过历史基线不足以支持创新成立。')
    return {'decision':decision,'answers':answers,'all_new_acc_not_above_b0':all_fail}


def handoff(summary):
    path=Path('/model/spromptgl/experiment/handoff.md')
    body='\n\n---\n## Experiment 82 — PSSEC 正式训练与独立审计\n\n'+summary+'\n'
    with path.open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0,2);before=f.tell();f.write(body);f.flush();os.fsync(f.fileno());f.seek(before);readback=f.read();assert readback==body
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    return {'path':str(path),'status':'APPENDED_AND_READBACK_VERIFIED','appended_utf8_bytes':len(body.encode()),'append_sha256':r.sha(body)}


def main():
    started=time.time();audits=[]
    audit_source_at_start=r.sha_file(Path(__file__))
    r.runtime()
    from pssec import smooth_limit
    zeros=torch.zeros((2,144),dtype=torch.float64)
    raw=torch.stack([torch.zeros(144),torch.ones(144)]).double().requires_grad_()
    limited=smooth_limit(raw,zeros)
    assert torch.count_nonzero(limited[0])==0 and float(limited[1].norm())<=.08*EPS*(1+1e-12)
    limited.sum().backward();assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[0],torch.ones(144,dtype=torch.float64))
    manifest=json.loads((r.HERE/'manifest.json').read_text());protected=manifest['source_maps']['protected']
    assert all(r.sha_file(p)==s for p,s in protected.items()),'protected source identity changed'
    assert all(r.sha_file(p)==s for p,s in manifest['source_maps']['implementation'].items() if Path(p).name!='audit.py')
    manifest.setdefault('audit_implementation',{})['current_audit_sha']=r.sha_file(Path(__file__))
    for fold in range(1,11):
        for group in r.GROUPS:audits.append(audit_one(group,fold))
    cross=0
    for fold in range(1,11):
        records=[a for a in audits if a['fold']==fold]
        assert len({a['new_parameter_sha'] for a in records})==1
        epochs={}
        for a in records:
            for t in a['rng_trace']:
                if t['epoch'] in epochs:assert epochs[t['epoch']]==t;cross+=1
                else:epochs[t['epoch']]=t
    assert all(r.sha_file(p)==s for p,s in protected.items())
    assert r.sha_file(Path(__file__))==audit_source_at_start,'audit implementation changed during replay'
    conclusion,summaries,paired=synthesize(audits)
    prompts=r.torch_load(r.RT/'prompts.pt')
    # Persist compact independent audit evidence, avoiding patient tokens.
    for a in audits:del a['rng_trace']
    status={'status':'PASS','training_complete':60,'audited_checkpoints':60,'expected_checkpoints':60,
        'protected_source_sha_pass':True,'cross_group_common_epoch_triplets':cross,
        'historical_common_epoch_triplets':sum(a['historical_common_rng_epochs'] for a in audits),
        'probability_max_abs_error':max(a['probability_max_abs_error'] for a in audits),
        'checkpoint_recovery':'60 bundle/RNG exact; G5/F1 next train-eval update exact',
        'wall_seconds':time.time()-started,'zero_input_limit_and_zero_residual_derivative':'PASS',
        'warmup_cam_original_and_unified_graph_replay':'10/10 exact',
        'independent_numpy_spectral_checks':SPECTRAL_AUDITS,
        'anonymous_geometry':{k:v for k,v in prompts['metadata'].items() if not torch.is_tensor(v)},
        'anonymous_rotation_sha':prompts['rotation_sha'],
        'conclusion':conclusion,'tasks':audits}
    status['result_artifact_sha256']={f:r.sha_file(r.RESULTS/f) for f in ('fold_metrics.csv','summary.csv','paired_comparisons.csv','mechanism.csv','report.md')}
    r.atomic_json(r.RESULTS/'audit_status.json',status)
    manifest.update(status='COMPLETE_AUDITED',completed_tasks=60,independent_audits=60,
         scientific_conclusion=conclusion,audit_summary={k:v for k,v in status.items() if k!='tasks'})
    r.atomic_json(r.HERE/'manifest.json',manifest)
    report=(r.RESULTS/'report.md').read_text()
    events=manifest.get('engineering_events',[])
    error_text='\n执行错误/修复：首选OSSFS目录errno5后立即使用/model。\n'+ '\n'.join(json.dumps(e,ensure_ascii=False) for e in events)
    execution=json.loads((r.RT/'execution.json').read_text())
    error_text+='\n正式任务失败/重试记录：'+json.dumps(execution.get('failures',[]),ensure_ascii=False)+'；冻结方法未调参。'
    h=handoff(report+error_text)
    manifest['handoff_update']=h;r.atomic_json(r.HERE/'manifest.json',manifest)
    print(json.dumps({'status':'COMPLETE_AUDITED','handoff':h,'conclusion':conclusion},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

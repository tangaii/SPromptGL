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
from pcde_ps import EPS, GROUPS as GROUP_SPECS


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


def flips(reference,candidate,y):
    return {'corrected':int(((reference!=y)&(candidate==y)).sum()),
        'harmed':int(((reference==y)&(candidate!=y)).sum()),
        'net':int((candidate==y).sum()-(reference==y).sum()),'flips':int((reference!=candidate).sum())}


def replay(state,module,intervention='full',with_grad=False):
    module.eval();details={};calls={'output':0,'cam':0,'gcn':[0,0,0,0]}
    def pre(_m,args):
        calls['output']+=1
        assert len(args)==2 and args[0].shape==(871,144)
        changed,d=module(args[0],intervention);details.update(d)
        return changed,args[1]
    handles=[state['cam'].Outputlayer.register_forward_pre_hook(pre),state['cam'].register_forward_hook(lambda *_:calls.__setitem__('cam',calls['cam']+1))]
    for i,m in enumerate(state['model'].gcn):handles.append(m.register_forward_hook(lambda _m,_a,_o,i=i:calls['gcn'].__setitem__(i,calls['gcn'][i]+1)))
    before=r.sha(r.cpu_state(module))
    try:
        with torch.set_grad_enabled(with_grad):out=r.runtime().forward(state,r.BASE_GROUP,training=False)
    finally:
        for h in handles:h.remove()
    assert calls=={'output':1,'cam':1,'gcn':[1,1,1,1]} and before==r.sha(r.cpu_state(module))
    return out,details,calls


SPECTRAL_DONE = {}

def spectral_audit(c,state):
    fold=state['fold']
    if fold in SPECTRAL_DONE:return SPECTRAL_DONE[fold]
    captured=[]
    h=state['cam'].Outputlayer.register_forward_pre_hook(lambda _m,args:captured.append(args[0].detach().cpu().clone()))
    before=r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    try:
        with torch.no_grad():out=r.runtime().forward(state,r.BASE_GROUP,training=False)
    finally:h.remove()
    assert len(captured)==1 and torch.equal(captured[0],c['warmup_cam'])
    assert [r.sha(v) for v in out['original_adjs']]==c['warmup_graph_sha']
    assert [r.sha(v) for v in out['unified_adjs']]==c['warmup_unified_graph_sha']
    assert before==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    z=c['warmup_cam'].double().numpy()[c['legal_train']];labels=state['labels_np'][c['legal_train']]
    s=np.cov(z,rowvar=False,ddof=1);s0=np.cov(z[labels==0],rowvar=False,ddof=1);s1=np.cov(z[labels==1],rowvar=False,ddof=1)
    spec=c['spectrum'];u=spec['U'].numpy();ug=u[:,:8];ud=u[:,8:];p=np.eye(144)-ug@ug.T
    assert np.allclose(u.T@u,np.eye(16),atol=1e-10,rtol=0)
    assert np.allclose(spec['covariance'].numpy(),s,atol=1e-10,rtol=1e-10)
    assert np.allclose(spec['mu'].numpy(),z.mean(0),atol=1e-10,rtol=1e-10)
    assert np.allclose(np.diag(ug.T@s@ug),np.linalg.eigvalsh(s)[-8:][::-1],atol=1e-9,rtol=1e-9)
    mat=p@(s0-s1)@p
    if spec['fallback']:mat=p@s@p
    ray=np.diag(ud.T@mat@ud);err=np.linalg.norm(mat@ud-ud*ray)
    assert err<1e-8*max(1,np.linalg.norm(mat))
    assert np.allclose(np.sort(abs(ray)),np.sort(abs(np.linalg.eigvalsh(mat)))[-8:],atol=1e-8,rtol=1e-7)
    assert r.sha(c['spectrum'])==c['spectra_sha'] and r.sha(c['warmup_cam'])==c['feature_sha']
    assert r.sha(c['spectrum']['U'])==c['e82_original_U_sha']
    assert np.array_equal(c['legal_train'],r.legal_train(state))
    assert r.sha(torch.as_tensor(labels))==c['legal_label_sha']
    result={'orthogonality':float(abs(u.T@u-np.eye(16)).max()),'eigen_residual':float(err),'e82_original_U_exact':True,'warmup_feature_and_both_graph_paths_exact':True}
    SPECTRAL_DONE[fold]=result
    return result


def formulas(module,d):
    h=(d['z']-module.mu_warm)@module.U;assert torch.equal(h,d['h'])
    floor=torch.maximum(module.var_pool*1e-4,torch.full_like(module.var_pool,EPS))
    va=torch.maximum(.9*module.var_asd+.1*module.var_pool,floor);vt=torch.maximum(.9*module.var_td+.1*module.var_pool,floor)
    # Independent normal-density reconstruction in NumPy float64.
    hn=h.detach().cpu().double().numpy();ma=module.mean_asd.cpu().double().numpy();mt=module.mean_td.cpu().double().numpy()
    va=va.cpu().double().numpy();vt=vt.cpu().double().numpy()
    llr=np.log(np.sqrt(vt/va))-.5*(hn-ma)**2/va+.5*(hn-mt)**2/vt
    assert np.allclose(d['llr'].detach().cpu().numpy(),llr,atol=1e-4,rtol=2e-5)
    assert np.allclose(d['e'].cpu().numpy(),4*np.tanh(llr/4),atol=2e-5,rtol=2e-5)
    assert not d['e'].requires_grad
    assert float(d['gate_input'].min())>=-1e-5 and float(d['gate_input'].max())<=1.00001
    q=F.normalize(module.q0[None]+.25*d['meta'],dim=-1)
    assert torch.equal(q,d['q'])
    keys=module.key(d['tokens']);signed=torch.cat([-d['e'][:,None].expand(-1,4,-1),d['e'][:,None].expand(-1,4,-1)],1)
    scores=torch.einsum('nkd,njd->nkj',q,keys)/math.sqrt(32)+.2*signed
    null=torch.einsum('nkd,d->nk',q,module.null)/math.sqrt(32)
    a=torch.cat([scores,null[:,:,None]],-1).div(.5).softmax(-1);assert torch.equal(a,d['a'])
    assert torch.allclose(a.sum(-1),torch.ones_like(a[:,:,0]),atol=1e-6,rtol=0)
    for side,ix in [('asd',slice(4,8)),('td',slice(0,4))]:
        cover=1-(1-a[:,ix,:16]).prod(1);w=(cover+EPS)/(cover.sum(-1,keepdim=True)+16*EPS)
        assert torch.equal(w,d['w_'+side]) and torch.allclose(w.sum(-1),torch.ones_like(w[:,0]),atol=1e-6,rtol=0)
    raw=.5*(d['w_asd']-d['w_td'])*(h-module.mean_pool);raw=raw@module.U.T
    assert torch.equal(raw,d['delta_raw'])
    norm=raw.norm(dim=-1,keepdim=True);limit=.08*d['z'].norm(dim=-1,keepdim=True).clamp_min(EPS)
    ratio=norm/limit;scale=torch.where(ratio==0,torch.ones_like(ratio),torch.tanh(ratio)/ratio.clamp_min(1e-30))
    expected=raw*scale*d['gate'][:,None]
    if module.fold==2:expected[[376,406]]=0
    assert torch.allclose(expected,d['delta'],atol=1e-7,rtol=2e-6)
    assert float((d['delta'].norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)).max())<=.080001
    return True


def mechanism(state,module,c,d):
    idx=torch.as_tensor(state['test_np'],device=module.U.device);legal=module.legal
    row={'group':module.group,'fold':module.fold,**module.counts(),'spectrum_sha':c['spectra_sha'],
        'variance_coverage_warmup':c['spectrum']['variance_coverage'],
        'cam_drift_relative_train':float((d['z'][legal]-c['warmup_cam'].to(module.U.device)[legal]).norm()/c['warmup_cam'][c['legal_train']].norm().to(module.U.device)),
        'projection_energy_best':float(d['h'][legal].square().mean()),
        'projection_energy_best_by_direction':json.dumps(d['h'][legal].square().mean(0).detach().cpu().tolist()),
        'ema_updates':int(module.ema_updates),'ema_class_mean_asd':json.dumps(module.mean_asd.cpu().tolist()),
        'ema_class_mean_td':json.dumps(module.mean_td.cpu().tolist()),'ema_class_variance_asd':json.dumps(module.var_asd.cpu().tolist()),
        'ema_class_variance_td':json.dumps(module.var_td.cpu().tolist()),'ema_pool_mean':json.dumps(module.mean_pool.cpu().tolist()),
        'ema_pool_variance':json.dumps(module.var_pool.cpu().tolist())}
    with torch.no_grad():
        row['ema_mean_lag_relative']=float((d['h'][legal].mean(0)-module.mean_pool).norm()/module.var_pool.sum().sqrt().clamp_min(EPS))
        row['ema_variance_lag_relative']=float((d['h'][legal].var(0,unbiased=True)-module.var_pool).norm()/module.var_pool.norm().clamp_min(EPS))
    for prefix,x in [('llr',d['llr'][idx]),('signed_evidence',d['e'][idx]),('s_asd',d['s_asd'][idx]),('s_td',d['s_td'][idx]),
        ('support_difference',d['s_asd'][idx]-d['s_td'][idx]),('margin',d['margin'][idx]),('gate',d['gate'][idx]),
        ('null',d['a'][idx,:,16]),('query_conditioning_l2',.25*d['meta'][idx].norm(dim=-1)),
        ('residual_relative',d['delta'][idx].norm(dim=-1)/d['z'][idx].norm(dim=-1).clamp_min(EPS)),
        ('bounded_relative',d['delta_bounded'][idx].norm(dim=-1)/d['z'][idx].norm(dim=-1).clamp_min(EPS))]:add_distribution(row,prefix,x)
    row['llr_saturation_fraction']=float((d['e'][idx].abs()>=3.96).float().mean())
    row['gate_near_zero_fraction']=float((d['gate'][idx]<.05).float().mean());row['gate_near_one_fraction']=float((d['gate'][idx]>.95).float().mean())
    row['residual_near_8pct_fraction']=float((d['delta'][idx].norm(dim=-1)/d['z'][idx].norm(dim=-1).clamp_min(EPS)>=.0792).float().mean())
    row['side_cosine']=float(F.cosine_similarity(d['w_asd'][idx],d['w_td'][idx],dim=-1).mean())
    row['side_l1']=float((d['w_asd'][idx]-d['w_td'][idx]).abs().sum(-1).mean())
    for side in ('asd','td'):
        w=d['w_'+side][idx];row[side+'_effective_directions']=float((-(w*(w+EPS).log()).sum(-1)).exp().mean())
        row[side+'_direction_weights']=json.dumps(w.mean(0).detach().cpu().tolist())
    ap=d['a'][idx,:,:16];a=F.normalize(ap,dim=-1);cos=a@a.transpose(1,2)
    mask=~torch.eye(4,dtype=torch.bool,device=a.device)
    row['prompt_same_side_td_cosine']=float(cos[:,:4,:4][:,mask].mean());row['prompt_same_side_asd_cosine']=float(cos[:,4:,4:][:,mask].mean())
    row['prompt_cross_side_cosine']=float(cos[:,:4,4:].mean());row['prompt_selection_matrix']=json.dumps(cos.mean(0).detach().cpu().tolist())
    row['prompt_direction_distributions']=json.dumps(ap.mean(0).detach().cpu().tolist());row['prompt_null_means']=json.dumps(d['a'][idx,:,16].mean(0).detach().cpu().tolist())
    q=d['q'][idx];row['query_between_patient_rms']=float((q-q.mean(0)).square().mean().sqrt())
    row['query_asd_td_cosine']=float(F.cosine_similarity(q[:,4:].mean(1),q[:,:4].mean(1),dim=-1).mean())
    row['query_anchor_difference_l2']=float((q-module.q0[None]).norm(dim=-1).mean())
    # Mechanism diagnostic only: compare learned routing to scores containing just the preregistered LLR sign bias.
    e=d['e'][idx];signed=torch.cat([-e[:,None].expand(-1,4,-1),e[:,None].expand(-1,4,-1)],1)
    bias_only=torch.cat([.2*signed,torch.zeros_like(signed[:,:,:1])],-1).div(.5).softmax(-1)
    row['full_vs_llr_bias_only_attention_cosine']=float(F.cosine_similarity(d['a'][idx].flatten(1),bias_only.flatten(1),dim=-1).mean())
    for label,side in [(0,'asd'),(1,'td')]:
        hc=d['h'][legal][module.legal_labels==label]
        row['ema_'+side+'_mean_lag_relative']=float((hc.mean(0)-getattr(module,'mean_'+side)).norm()/getattr(module,'var_'+side).sum().sqrt().clamp_min(EPS))
        row['ema_'+side+'_variance_lag_relative']=float((hc.var(0,unbiased=True)-getattr(module,'var_'+side)).norm()/getattr(module,'var_'+side).norm().clamp_min(EPS))
    warm_h=(c['warmup_cam'].to(module.U.device)-module.mu_warm)@module.U
    row['projection_energy_best_to_warm_ratio']=float(d['h'][legal].square().sum()/warm_h[legal].square().sum().clamp_min(EPS))
    pred=d.get('prediction');y=torch.as_tensor(state['labels_np'][state['test_np']],device=idx.device)
    if pred is not None:
        epred=(d['margin'][idx]<0).long();row['evidence_classification_agreement']=float((epred==pred).float().mean())
        row['evidence_true_class_agreement']=float((epred==y).float().mean())
        wrong=pred!=y;row['wrong_patient_margin']=json.dumps(d['margin'][idx][wrong].detach().cpu().tolist())
        row['wrong_patient_gate']=json.dumps(d['gate'][idx][wrong].detach().cpu().tolist())
        row['gate_mean_correct']=float(d['gate'][idx][~wrong].mean())
        row['gate_mean_wrong']=float(d['gate'][idx][wrong].mean()) if wrong.any() else 0.
    if module.fold==2:assert torch.count_nonzero(d['delta'][[376,406]])==0
    return row


def reference_metrics(prob,y):
    pred=prob.argmax(-1)
    return {'acc':accuracy_score(y,pred),'hard_auc':balanced_accuracy_score(y,pred),'prob_auc':roc_auc_score(y,prob[:,1]),
        'sen':recall_score(y,pred,pos_label=0),'spe':recall_score(y,pred,pos_label=1),'asd_f1':f1_score(y,pred,pos_label=0)}


def audit_one(group,fold):
    assert r.task_complete(group,fold)
    tp,bp,lp,log=r.paths(group,fold);task=json.loads(tp.read_text());best=r.torch_load(bp);latest=r.torch_load(lp)
    state,warmup,identity=r.prepare(fold)
    if fold==2:assert int(state['labels_np'][376])==0 and int(state['labels_np'][406])==1
    c=r.torch_load(r.RT/f'spectrum_f{fold:02d}.pt');text=r.torch_load(r.RT/'prompts.pt')
    module,opt,init=r.make_module(state,group,c,text);spectral=spectral_audit(c,state)
    assert init==best['new_parameter_sha']==task['new_parameter_sha']
    for ck in (best,latest):
        assert ck['identity']==identity and ck['config']==r.CFG and ck['source_sha']==r.implementation_sha()
        assert ck['spectra_sha']==c['spectra_sha'] and ck['text_identity_sha']==text['identity_sha']
    assert task['best_sha']==r.sha_file(bp) and task['latest_sha']==r.sha_file(lp)
    r.load_bundle(state,module,opt,best['current']);r.base_runtime().restore_rng(best['rng'])
    assert r.sha(r.bundle(state,module,opt))==r.sha(best['current'])
    assert torch.equal(module.q0.cpu(),text['q_'+GROUP_SPECS[group][0]])
    assert torch.equal(module.U.cpu(),c['spectrum']['U'].float())
    assert int(module.ema_updates)==best['best_epoch']-10
    assert best['wait']==0 and best['next_epoch']==best['best_epoch'] and best['best_acc']==best['metrics']['acc']
    before=r.base_runtime().rng_sha(r.base_runtime().pack_rng());out,d,calls=replay(state,module,with_grad=True)
    values,prob,pred,y=independently_recompute(out['logp'],state);formulas(module,d)
    assert before==best['rng_sha']==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    assert np.array_equal(pred,best['prediction']) and np.allclose(prob,best['probability'],atol=1e-8,rtol=0)
    assert all(abs(values[k]-best['metrics'][k])<1e-12 and abs(values[k]-task['metrics'][k])<1e-12 for k in r.METRICS)
    logs=verified_logs(group,fold)
    assert sorted(logs)==list(range(11,task['epochs_executed']+1))
    chosen=max(logs,key=lambda e:(logs[e]['acc'],-e));assert chosen==best['best_epoch']==task['best_epoch']
    reference={v['epoch']:v for v in r.historical_task(fold)['rng_trace']};parity=0
    for v in task['rng_trace']:
        assert all(logs[v['epoch']][k]==v[k] for k in ('rng_before','rng_after_train','rng_after_eval'))
        if v['epoch'] in reference:assert v==reference[v['epoch']];parity+=1
    assert len(task['rng_trace'])==task['epochs_executed']-10
    assert all(logs[ep]['ema_updates']==ep-10 for ep in logs)
    assert logs[chosen]['ema_sha']==r.sha({n:getattr(module,n) for n in ('mean_asd','mean_td','var_asd','var_td','mean_pool','var_pool')})
    d['prediction']=torch.as_tensor(pred,device=module.U.device);row=mechanism(state,module,c,d)
    named=[(prefix+'.'+n,p) for prefix,m in [('darg',state['model']),('cam',state['cam']),('pcde',module)] for n,p in m.named_parameters() if p.requires_grad]
    grads=torch.autograd.grad(out['cls'],[p for n,p in named]+[d['z'],out['fused']],allow_unused=True,retain_graph=True)
    squared={n:float(g.detach().double().square().sum().cpu()) if g is not None else 0. for (n,p),g in zip(named,grads)}
    norms={pre:math.sqrt(sum(v for n,v in squared.items() if n.startswith(pre+'.') or n==pre)) for pre in ('darg','cam','cam.Outputlayer','pcde.direction','pcde.evidence','pcde.key','pcde.null','pcde.meta_asd','pcde.meta_td','pcde.gate.0','pcde.gate.2')}
    norms.update(cam144=float(grads[-2].norm().cpu()),darg320=float(grads[-1].norm().cpu()))
    assert all(math.isfinite(v) for v in norms.values()) and all(norms[n]>0 for n in ('darg','cam','cam.Outputlayer','cam144','darg320'))
    all_parameters={id(p):p for op in [state[k] for k in ('opt_base','opt_cam','opt_added')]+[opt] for group_params in op.param_groups for p in group_params['params'] if p.requires_grad}
    params=list(all_parameters.values());positions={id(p):i for i,p in enumerate(params)}
    residual=(d['delta'][module.legal].square().sum(-1)/(d['z'][module.legal].square().sum(-1)+EPS)).mean()
    total=out['total']+.01*d['evidence_loss']+.001*residual
    total_grads=torch.autograd.grad(total,params,allow_unused=True)
    optimizer_checks={}
    for name,optimizer in [(k,state[k]) for k in ('opt_base','opt_cam','opt_added')]+[('opt_pcde',opt)]:
        current=[p for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad]
        gradnorm=math.sqrt(sum(float(total_grads[positions[id(p)]].double().square().sum().cpu()) for p in current if total_grads[positions[id(p)]] is not None))
        assert math.isfinite(gradnorm) and gradnorm>0
        sd=optimizer.state_dict();moments=[v for st in sd['state'].values() for k,v in st.items() if torch.is_tensor(v)]
        assert all(torch.isfinite(v).all() for v in moments)
        optimizer_checks[name]={'total_gradient_norm_at_best_eval':gradnorm,'state_count':len(sd['state']),'state_sha':r.sha(sd),
            'nonzero_first_moment_tensors':sum(int(torch.count_nonzero(st['exp_avg'])>0) for st in sd['state'].values() if 'exp_avg' in st),
            'learning_rate':sd['param_groups'][0]['lr'],'weight_decay':sd['param_groups'][0]['weight_decay']}
    row['four_optimizer_audit']=json.dumps(optimizer_checks)
    row.update(best_classification_gradient_norms=json.dumps(norms),best_total_gradient_norms=json.dumps(best['gradients']),
        first_gradient_norms=json.dumps(task['first_classification_gradients']),best_epoch=best['best_epoch'],wall_seconds=task['wall_seconds'],peak_gpu_bytes=task['peak_gpu_bytes'])
    b0=r.torch_load(r.runtime().EXP/'.runtime/checkpoints'/f'{r.BASE_GROUP}_f{fold:02d}.best.pt');assert b0['identity']==identity
    b0prob=np.asarray(b0['test_probability']);bm=reference_metrics(b0prob,y);b0pred=b0prob.argmax(-1)
    assert all(abs(bm[k]-b0['metrics'][k])<1e-12 for k in r.METRICS)
    old=r.torch_load(r.E82/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt');assert old['identity']==identity
    oldprob=np.asarray(old['probability']);om=reference_metrics(oldprob,y)
    assert all(abs(om[k]-old['metrics'][k])<1e-12 for k in r.METRICS)
    row['vs_b0']=json.dumps(flips(b0pred,pred,y))
    idx=torch.as_tensor(state['test_np'],device=module.U.device);correct=torch.as_tensor(b0pred==y,device=module.U.device)
    row['residual_mean_b0_correct']=float((d['delta'][idx].norm(dim=-1)/d['z'][idx].norm(dim=-1).clamp_min(EPS))[correct].mean())
    interventions=[]
    if group=='G3':
        qsave=module.q0.clone()
        for name in ('anonymous','swap_sides','conditioning_off','gate_off'):
            with torch.no_grad():
                if name=='anonymous':module.q0.copy_(text['q_anonymous'].to(module.U.device))
                elif name=='swap_sides':module.q0.copy_(qsave[torch.tensor([4,5,6,7,0,1,2,3],device=module.U.device)])
                else:module.q0.copy_(qsave)
            changed,_,_=replay(state,module,name if name in ('conditioning_off','gate_off') else 'full')
            met,cp,cr,_=independently_recompute(changed['logp'],state)
            interventions.append({'intervention':name,'mean_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).mean()),
                'max_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).max()),**flips(pred,cr,y),'metrics':met,
                'flipped_rows_1based':(state['test_np'][cr!=pred]+1).tolist()})
        with torch.no_grad():module.q0.copy_(qsave)
        # Only diagnostic losses see labels; replacing held-out labels must leave full forward prediction equal.
        held=np.setdiff1d(state['test_np'],r.legal_train(state));saved_y=state['y'].clone();saved_np=state['labels_np'].copy()
        state['y'][held]=1-state['y'][held];state['labels_np'][held]=1-state['labels_np'][held]
        try:
            changed,_,_=replay(state,module)
            assert torch.equal(out['logp'],changed['logp'])
        finally:state['y'].copy_(saved_y);state['labels_np'][:]=saved_np
    row['readonly_interventions']=json.dumps(interventions)
    r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng'])
    assert r.sha(r.bundle(state,module,opt))==latest['bundle_sha']
    assert r.base_runtime().rng_sha(r.base_runtime().pack_rng())==latest['rng_sha']==task['rng_trace'][-1]['rng_after_eval']
    assert latest['next_epoch']==task['epochs_executed'] and int(module.ema_updates)==latest['next_epoch']-10
    assert latest['wait']>200 or latest['next_epoch']==450
    restore='all_parameters_ema_four_optimizers_rng_exact'
    if group=='G3' and fold==1:
        def next_update():
            r.zero(state,opt);o=r.forward(state,module,True);o['total'].backward();r.step(state,opt)
            with torch.no_grad():o=r.forward(state,module,False)
            return r.sha(r.bundle(state,module,opt)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),o['logp'].cpu().clone()
        a=next_update();r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng']);b=next_update()
        assert a[:2]==b[:2] and torch.equal(a[2],b[2]);restore='next_update_with_ema_state_probability_rng_exact'
    result={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','metrics':values,'prediction':pred.tolist(),'probability':prob.tolist(),'labels':y.tolist(),
        'b0_metrics':bm,'b0_prediction':b0pred.tolist(),'e82_metrics':om,'e82_prediction':oldprob.argmax(-1).tolist(),
        'best_epoch':best['best_epoch'],'best_sha':task['best_sha'],'latest_sha':task['latest_sha'],
        'new_parameter_sha':init,'identity':identity,'spectral_audit':spectral,'historical_common_rng_epochs':parity,
        'probability_max_abs_error':float(abs(prob-best['probability']).max()),'latest_restore':restore,'mechanism':row,'interventions':interventions,'rng_trace':task['rng_trace']}
    del out,d,state,module,opt,best,latest,b0,old;torch.cuda.empty_cache();print(f'AUDIT PASS {group}/{fold}',flush=True)
    return result
def aggregate(audits):
    from scipy.stats import t as student_t
    by={(a['group'],a['fold']):a for a in audits}
    foldrows=[];mechs=[]
    for f in range(1,11):
        for g in (*r.GROUPS,'B0','E82-G2'):
            a=by[(g,f)] if g in r.GROUPS else by[('G3',f)]
            m=a['metrics'] if g in r.GROUPS else a['b0_metrics'] if g=='B0' else a['e82_metrics']
            foldrows.append({'group':g,'fold':f,'best_epoch':a['best_epoch'] if g in r.GROUPS else '',**{k:100*m[k] for k in r.METRICS}})
        for g in r.GROUPS:
            row=by[(g,f)]['mechanism'].copy()
            for ref in ('G1','G2'):
                row['vs_'+ref.lower()]=json.dumps(flips(np.asarray(by[(ref,f)]['prediction']),np.asarray(by[(g,f)]['prediction']),np.asarray(by[(g,f)]['labels'])))
            mechs.append(row)
    summary=[];paired=[]
    groups=(*r.GROUPS,'B0','E82-G2')
    for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
        for g in groups:
            rows=[x for x in foldrows if x['group']==g and x['fold'] in folds]
            row={'group':g,'scope':scope,'n_folds':len(rows)}
            for k in r.METRICS:
                vals=np.asarray([x[k] for x in rows]);row[k+'_mean']=float(vals.mean());row[k+'_population_sd']=float(vals.std(ddof=0))
            summary.append(row)
        index={(x['group'],x['fold']):x for x in foldrows}
        for g,ref in [('G3','B0'),('G3','E82-G2'),('G3','G1'),('G3','G2'),('G2','E82-G2'),('G1','B0')]:
            for k in r.METRICS:
                delta=np.asarray([index[(g,f)][k]-index[(ref,f)][k] for f in folds]);mean=float(delta.mean())
                half=float(student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/math.sqrt(len(delta)))
                paired.append({'comparison':g+' - '+ref,'scope':scope,'metric':k,'mean_delta_pp':mean,
                    'population_sd_delta_pp':float(delta.std()),'descriptive_95_low':mean-half,'descriptive_95_high':mean+half,
                    'wins':int((delta>1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),'losses':int((delta< -1e-10).sum()),
                    **{'fold'+str(f)+'_delta_pp':float(v) for f,v in zip(folds,delta)}})
    r.write_csv(r.RESULTS/'fold_metrics.csv',foldrows);r.write_csv(r.RESULTS/'summary.csv',summary)
    r.write_csv(r.RESULTS/'paired_comparisons.csv',paired);r.write_csv(r.RESULTS/'mechanism.csv',mechs)
    def dm(g,ref,k):return next(x for x in paired if x['comparison']==g+' - '+ref and x['scope']=='all10' and x['metric']==k)
    def mean_mech(g,k):return float(np.mean([m[k] for m in mechs if m['group']==g]))
    comparisons=[]
    for g,ref in [('G3','B0'),('G3','E82-G2'),('G3','G1'),('G3','G2'),('G2','E82-G2'),('G1','B0')]:
        comparisons.append('|'+g+'−'+ref+'|'+'|'.join(f"{dm(g,ref,k)['mean_delta_pp']:+.4f}" for k in r.METRICS)+'|'+str(dm(g,ref,'acc')['wins'])+'/'+str(dm(g,ref,'acc')['ties'])+'/'+str(dm(g,ref,'acc')['losses'])+'|')
    interventions=[]
    for name in ('anonymous','swap_sides','conditioning_off','gate_off'):
        a=[i for x in audits if x['group']=='G3' for i in x['interventions'] if i['intervention']==name]
        interventions.append({'intervention':name,'mean_abs_probability_change':float(np.mean([x['mean_abs_asd_probability_change'] for x in a])),
            'max_abs_probability_change':float(max(x['max_abs_asd_probability_change'] for x in a)),
            **{k:sum(x[k] for x in a) for k in ('flips','corrected','harmed','net')}})
    texts=r.torch_load(r.RT/'prompts.pt');execution=json.loads((r.RT/'execution.json').read_text())
    report=['# Experiment 82.1 PCDE-PS','', '状态：COMPLETE_AUDITED；30/30正式训练，30/30 Best独立审计。',
        '',f'正式目录：`{r.HERE}`。仅seed=0、871人和原固定10折；ASD=0、TD=1。B0与E82均只读历史结果，所有新任务仅从对应E61 epoch10完整warmup初始化。',
        '', '## 冻结方法与源码核查','',
        '真实历史PLNNoPEN仅执行InputLayer→Encoder/FeedForward→原OutputLayer；原八条Prompt的SPGS/PEN路径已被历史方法移除。E82.1不重新引入原PEN。OutputLayer真实签名为forward(x,attn_embedding)，x=CAM144；新增模块通过try/finally hook增强其输入，每个训练/评估仅一次DARG/CAM/原分类头。',
        '', '核查[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[原仓库](https://github.com/wanxixi11/SPromptGL)、[项目仓库](https://github.com/tangaii/SPromptGL)、[CoCoOp论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)及[官方Meta-Net源码](https://github.com/KaiyangZhou/CoOp/blob/main/trainers/cocoop.py)、[SSC论文](https://papers.miccai.org/miccai-2026/paper/4444_paper.pdf)与[官方代码](https://github.com/Hanyu1124/ssc)。精确版本、SHA及实际依赖见manifest，原文与源码读取记录合并在runtime/source_review.json。CoCoOp条件化文本context来自CLIP图像特征；本实验不假定CAM与CLIP图像空间对齐。SSC的SKP来自文本谱空间，PCDE来自训练患者CAM谱空间；SSC硬Top-r、预算融合与本文连续双侧覆盖不同。',
        '',f"U逐折与E82原版O逐元素相同，CPU float64；类/总体方差均为n−1。EMA固定0.95旧+0.05当前，只用合法训练h.detach()，训练前向更新一次，评估只读。类方差收缩为0.9类+0.1全训练总体方差；逐分量下限max(1e−8,1e−4总体EMA方差)。e=4tanh(LLR/4)进入路由时detach；辅助BCE仍通过Token/Query/attention产生实际梯度。折2新增统计/损失排除377、407，残差严格0；历史训练重叠仍保留。",
        '', '0.25患者条件化、0.2有符号偏置、0.5温度、8%平滑限幅、0.01证据辅助损失、0.001残差正则和学习率全程冻结。G2门控同结构但不计算、不入优化器；G1/G3只有文本不同，初始参数与有效参数数一致。',
        '',f"文本秩{int(texts['rank'])}；匿名旋转Gram最大误差{float(texts['gram_error']):.3g}，运行float32误差{float(texts['gram_error_float32']):.3g}；文本相对变化{float(texts['text_relative_change']):.6f}。冻结共享投影P_text列正交误差{float(texts['projection_orthogonality']):.3g}，医学/匿名投影Gram误差{float(texts['projection_gram_medical']):.3g}/{float(texts['projection_gram_anonymous']):.3g}。后续编码器可部分吸收几何变换，不能单凭训练对照证明医学语义因果贡献。",'','## 六项指标','', '均值±population SD，单位%；Hard-AUC=本历史二分类硬预测BA，Prob-AUC用连续概率。']
    for scope in ('all10','without_fold2'):
        report+=['', '### '+scope,'','|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['scope']==scope:report.append('|'+row['group']+'|'+'|'.join(f"{row[k+'_mean']:.4f} ± {row[k+'_population_sd']:.4f}" for k in r.METRICS)+'|')
    report+=['','## 主要配对比较','', '单位百分点；全部逐折差、W/T/L及描述性95%配对t区间见paired_comparisons.csv，包含排除Fold2后的9折。共享测试折和test选模使其不构成无偏显著性检验。','', '|比较|ACC Δ|BA Δ|Prob-AUC Δ|SEN Δ|SPE Δ|F1 Δ|ACC W/T/L|','|---|---:|---:|---:|---:|---:|---:|---|',*comparisons,'','## 机制结果','']
    for g in r.GROUPS:
        selected=[m for m in mechs if m['group']==g];v=selected[0]
        report.append(f"{g}：有效/名义参数{v['effective_parameters']}/{v['nominal_parameters']}；warm谱方差覆盖{100*mean_mech(g,'variance_coverage_warmup'):.3f}%，CAM相对漂移{mean_mech(g,'cam_drift_relative_train'):.4f}，Best投影能量{mean_mech(g,'projection_energy_best'):.4f}，EMA均值/方差相对滞后{mean_mech(g,'ema_mean_lag_relative'):.4f}/{mean_mech(g,'ema_variance_lag_relative'):.4f}。")
        report.append(f"双侧cos={mean_mech(g,'side_cosine'):.6f}、L1={mean_mech(g,'side_l1'):.6f}；ASD/TD有效方向{mean_mech(g,'asd_effective_directions'):.3f}/{mean_mech(g,'td_effective_directions'):.3f}；同侧ASD/TD Prompt选择cos={mean_mech(g,'prompt_same_side_asd_cosine'):.6f}/{mean_mech(g,'prompt_same_side_td_cosine'):.6f}，跨侧={mean_mech(g,'prompt_cross_side_cosine'):.6f}。患者间Query RMS={mean_mech(g,'query_between_patient_rms'):.6f}，Query偏离冻结锚点L2={mean_mech(g,'query_anchor_difference_l2'):.6f}，LLR符号偏置单独attention与完整attention cos={mean_mech(g,'full_vs_llr_bias_only_attention_cosine'):.6f}。")
        report.append(f"Null均值{mean_mech(g,'null_mean'):.6f}，LLR饱和比例{100*mean_mech(g,'llr_saturation_fraction'):.3f}%；gate均值{mean_mech(g,'gate_mean'):.6f}，近0/近1比例{100*mean_mech(g,'gate_near_zero_fraction'):.3f}%/{100*mean_mech(g,'gate_near_one_fraction'):.3f}%；残差相对范数均值{100*mean_mech(g,'residual_relative_mean'):.4f}%，未乘gate的bounded均值{100*mean_mech(g,'bounded_relative_mean'):.4f}%，近8%比例{100*mean_mech(g,'residual_near_8pct_fraction'):.3f}%；Best epoch均值{mean_mech(g,'best_epoch'):.1f}。")
        report.append('')
    report+=['每折EMA六个状态、各分量能量、完整Prompt相似度矩阵、Query幅度、Margin/门控/残差分位数、错误患者Margin、原三个优化路径及新增路径梯度均见mechanism.csv。初始gate前层因最后层零权重而零梯度；第二次真实更新前已验证连通。','', '## G3 Best只读干预','', '|干预|ASD概率平均绝对变化|最大绝对变化|硬翻转|纠错|误伤|净纠错|','|---|---:|---:|---:|---:|---:|---:|']
    for x in interventions:report.append('|'+x['intervention']+'|'+f"{x['mean_abs_probability_change']:.9f}|{x['max_abs_probability_change']:.9f}|{x['flips']}|{x['corrected']}|{x['harmed']}|{x['net']}|")
    for ref in ('b0','g1','g2'):
        counts={k:sum(json.loads(m['vs_'+ref])[k] for m in mechs if m['group']=='G3') for k in ('corrected','harmed','net','flips')}
        report+=['',f"G3相对{ref.upper()}：纠错{counts['corrected']}、误伤{counts['harmed']}、净纠错{counts['net']}、硬翻转{counts['flips']}，按折患者计数。"]
    report+=['', '只读干预属于同一Best模型的即时响应，不等同于重新训练消融；类别交换仅交换锚点，仍保留按位置指定的LLR符号，不能由双侧差异本身推断文本医学含义。概率受到文本影响、文本带来独立收益、医学含义被验证是三个不同命题。','', '## 独立审计与工程记录','',
        f"30/30 Best由分离审计代码重放，概率误差≤1e−8、硬预测一致，六项指标与历史工具及sklearn相同。严格ACC首次最大值与每轮日志核对。共有历史epoch RNG三元组共{sum(a['historical_common_rng_epochs'] for a in audits)}条完全一致；三组同折初始参数/共有RNG一致。30份Latest均完整恢复全部参数、4优化器、EMA及RNG；G3/Fold1双重下一训练+评估更新数值完全一致。源模型/数据/warmup/历史Best保护SHA保持不变。",
        '',f"正式调度耗时{execution.get('wall_seconds',execution.get('elapsed_seconds',0)):.1f}s，最大worker {execution['max_workers_observed']}，失败事件{len(execution['failures'])}。峰值按任务PyTorch allocated记录；全部任务最高{max(m['peak_gpu_bytes'] for m in mechs)/2**30:.3f}GiB，不等于整设备驱动总占用。",
        '', '保留的协议偏倚：val=test选Best、CV前RFE/标准化、transductive患者图、Fold2两人训练测试重叠；九折结果也不能消除前三项偏倚。潜在16方向不是脑区；Gaussian支持是训练数据统计证据，不是病理事实；正常/异常分组不是已验证的ASD/TD知识。', '', '## 预注册科研判断','',
        f"A 疾病竞争证据：G2/G3相对E82-G2 ACC分别{dm('G2','E82-G2','acc')['mean_delta_pp']:+.4f}/{dm('G3','E82-G2','acc')['mean_delta_pp']:+.4f}pp；完整证据与条件化同时改变，不能归因于单一机制。",
        f"B 医学文本：G3−G1 ACC {dm('G3','G1','acc')['mean_delta_pp']:+.4f}pp，W/T/L {dm('G3','G1','acc')['wins']}/{dm('G3','G1','acc')['ties']}/{dm('G3','G1','acc')['losses']}。必须结合上述替换/交换干预及跨折方向判断，微小概率响应不等于独立医学收益。",
        f"C 可靠性门控：G3−G2 ACC {dm('G3','G2','acc')['mean_delta_pp']:+.4f}pp；门控即时干预和未乘/乘门控残差尺度共同解释其作用，不能把单纯幅值降低当成可靠性校准。",
        f"D 候选要求：G3−B0 ACC/BA/Prob-AUC分别{dm('G3','B0','acc')['mean_delta_pp']:+.4f}/{dm('G3','B0','hard_auc')['mean_delta_pp']:+.4f}/{dm('G3','B0','prob_auc')['mean_delta_pp']:+.4f}pp。最终GO/PARTIAL/NO-GO须同时依据分类与机制证据；本实验不追加组别或调参。"]
    conclusion=json.loads((r.HERE/'manifest.json').read_text()).get('scientific_conclusion')
    if conclusion:report+=['','## 最终科研结论','',conclusion['text']]
    events=json.loads((r.HERE/'manifest.json').read_text()).get('engineering_events',[])
    if events:report+=['','工程记录：'+'；'.join(events)]
    finalization=json.loads((r.HERE/'manifest.json').read_text()).get('checkpoint_metadata_finalization')
    if finalization:report+=['','Checkpoint工程事件：Best格式检查后补齐显式续训计数wait=0、next_epoch=best_epoch、best_acc和bundle_sha；逐份核对参数/优化器/EMA bundle及RNG哈希未改变。原文件SHA与补齐后SHA保留在manifest和任务记录，训练源码及方法配置不变。']
    peak=r.RT/'device_peak.json'
    if peak.exists():report+=['',f"正式训练整设备采样峰值{json.loads(peak.read_text())['sampled_peak_device_mib']}MiB，15秒采样可能遗漏瞬时峰值；存在另一会话并发E81.6负载，该数值不能单独归属于E82.1。"]
    (r.RESULTS/'report.md').write_text('\n'.join(report)+'\n')
    return {'summary':summary,'paired':paired,'interventions':interventions}


def raw_logs(group,fold):
    path=r.paths(group,fold)[3];valid={};invalid=[]
    for no,line in enumerate(path.read_text().splitlines(),1):
        try:
            v=json.loads(line);ep=v['epoch']
        except (json.JSONDecodeError,KeyError):invalid.append({'line':no,'prefix':line[:100]});continue
        if ep in valid:assert valid[ep]==v
        valid[ep]=v
    task=json.loads(r.paths(group,fold)[0].read_text())
    missing=sorted(set(range(11,task['epochs_executed']+1))-set(valid))
    return valid,invalid,missing


def verified_logs(group,fold):
    logs,invalid,missing=raw_logs(group,fold)
    if invalid or missing:
        path=r.RT/f'log_recovery_{group}_f{fold:02d}.json';proof=json.loads(path.read_text())
        assert proof['status']=='EXACT_REPLAY_PASS' and proof['original_log_sha']==r.sha_file(r.paths(group,fold)[3])
        assert proof['source_sha']==r.implementation_sha() and proof['missing_epochs']==missing
        assert proof['invalid_rows']==invalid and proof['terminal_bundle_exact'] and proof['terminal_rng_exact']
        for v in proof['recovered_rows']:
            assert v['epoch'] not in logs;logs[v['epoch']]=v
    return logs


def recover_log(group,fold):
    """Audit-only deterministic replay; never writes formal weights/tasks/raw logs."""
    valid,invalid,missing=raw_logs(group,fold);assert missing
    task_path,best_path,latest_path,log=r.paths(group,fold)
    task=json.loads(task_path.read_text());best=r.torch_load(best_path);latest=r.torch_load(latest_path)
    state,warmup,identity=r.prepare(fold);module,opt,_=r.make_module(state,group);base=r.base_runtime()
    best_epoch=best['best_epoch']
    if min(missing)>best_epoch:
        start=best_epoch;r.load_bundle(state,module,opt,best['current']);base.restore_rng(best['rng'])
        best_acc=best['metrics']['acc'];chosen=best_epoch;wait=0
    else:start=10;best_acc=-1.;chosen=-1;wait=0
    end=best_epoch if max(missing)<=best_epoch else task['epochs_executed']
    reference={x['epoch']:x for x in task['rng_trace']};recovered=[];matched=0;best_exact=False
    for epoch in range(start,end):
        before=base.rng_sha(base.pack_rng());r.zero(state,opt);out=r.forward(state,module,True)
        if epoch==10:r.classification_gradient_audit(out,state,module)
        if epoch==11 and module.gate_enabled:
            gp=torch.autograd.grad(out['cls'],module.gate[0].weight,retain_graph=True)[0];assert gp.norm()>0
        out['total'].backward();assert r.finite_grads(state,module);grad=r.gradients(state,module);r.step(state,opt)
        after=base.rng_sha(base.pack_rng())
        losses={k:float(out[k].detach().cpu()) for k in ('cls','graph','pair','adv','preserve','darg_total','res','evidence_loss','total')};del out
        with torch.no_grad():o=r.forward(state,module,False);metrics=base.classification_metrics(o['logp'],state['test'],state['y'])
        after_eval=base.rng_sha(base.pack_rng());executed=epoch+1
        item={'epoch':executed,'rng_before':before,'rng_after_train':after,'rng_after_eval':after_eval};assert item==reference[executed]
        improved=metrics['acc']>best_acc
        if improved:best_acc=float(metrics['acc']);chosen=executed;wait=0
        else:wait+=1
        row={'epoch':executed,'acc':float(metrics['acc']),'best_acc':best_acc,'best_epoch':chosen,'improved':improved,'wait':wait,
            'losses':losses,'gradients':grad,'ema_updates':int(module.ema_updates),
            'ema_sha':r.sha({n:getattr(module,n) for n in ('mean_asd','mean_td','var_asd','var_td','mean_pool','var_pool')}),**item}
        if executed in valid:
            assert row==valid[executed],f'Exact retained log row mismatch {group}/{fold}/{executed}';matched+=1
        if executed in missing:recovered.append(row)
        if executed==best_epoch:
            assert r.sha(r.bundle(state,module,opt))==r.sha(best['current']) and after_eval==best['rng_sha']
            assert np.array_equal(metrics['prob'],best['probability']);best_exact=True
        del o
    target=best if end==best_epoch else latest
    terminal_bundle=r.sha(r.bundle(state,module,opt));terminal_rng=base.rng_sha(base.pack_rng())
    assert terminal_bundle==r.sha(target['current']) and terminal_rng==target['rng_sha']
    assert len(recovered)==len(missing)
    proof={'status':'EXACT_REPLAY_PASS','group':group,'fold':fold,'source_sha':r.implementation_sha(),
        'original_log_sha':r.sha_file(log),'best_sha':r.sha_file(best_path),'latest_sha':r.sha_file(latest_path),
        'missing_epochs':missing,'invalid_rows':invalid,'start_epoch':start,'end_epoch':end,
        'replayed_epochs':end-start,'retained_rows_exactly_matched':matched,'best_bundle_verified':best_exact,
        'terminal_bundle_exact':True,'terminal_rng_exact':True,'terminal_bundle_sha':terminal_bundle,'terminal_rng_sha':terminal_rng,
        'recovered_rows':recovered,'formal_checkpoints_and_raw_logs_unchanged':True}
    r.atomic_json(r.RT/f'log_recovery_{group}_f{fold:02d}.json',proof)
    print(f'LOG RECOVERY EXACT {group}/{fold}: epochs {start+1}-{end}, missing {missing}, matched {matched}',flush=True)


def complete_best_metadata():
    """Serialization-only finalization; preserves every parameter/optimizer/EMA/RNG tensor."""
    changes=[]
    for fold in range(1,11):
        for group in r.GROUPS:
            assert r.task_complete(group,fold)
            tp,bp,lp,log=r.paths(group,fold);task=json.loads(tp.read_text());best=r.torch_load(bp)
            if all(k in best for k in ('wait','next_epoch','best_acc','bundle_sha')):continue
            original_sha=r.sha_file(bp);state_sha=r.sha(best['current']);rng_sha=r.sha(best['rng'])
            best.update(wait=0,next_epoch=best['best_epoch'],best_acc=best['metrics']['acc'],bundle_sha=state_sha)
            best['best_meta']={k:best[k] for k in ('best_epoch','metrics','probability','prediction','losses','gradients')}
            r.atomic_torch(bp,best);readback=r.torch_load(bp)
            assert r.sha(readback['current'])==state_sha and r.sha(readback['rng'])==rng_sha
            task.update(best_sha=r.sha_file(bp),original_best_sha_before_counter_metadata=original_sha)
            r.atomic_json(tp,task);changes.append({'group':group,'fold':fold,'original_sha':original_sha,'final_sha':task['best_sha'],'bundle_sha_unchanged':state_sha,'rng_sha_unchanged':rng_sha})
    manifest=json.loads((r.HERE/'manifest.json').read_text())
    manifest['checkpoint_metadata_finalization']={'reason':'Best needed explicit resume counters in addition to stored best_epoch and trace. Added wait=0, next_epoch=best_epoch, best_acc=ACC and bundle_sha. No tensor, probability, optimizer, EMA, RNG, selected epoch or training method changed.','changes':changes}
    manifest['preflight_audit_source_sha']=manifest['source_maps']['implementation'][str(Path(__file__))]
    manifest['source_maps']['implementation'][str(Path(__file__))]=r.sha_file(Path(__file__))
    r.atomic_json(r.HERE/'manifest.json',manifest)
    print('Best resume-counter metadata finalized; all state/RNG hashes unchanged.',flush=True)


def text_geometry_audit():
    t=r.torch_load(r.RT/'prompts.pt');p=t['P_text'].double().numpy();med=t['medical'].double().numpy();anon=t['anonymous'].double().numpy()
    errors={'projection_orthogonality':float(abs(p.T@p-np.eye(32)).max()),
        'anonymous_gram':float(abs(anon@anon.T-med@med.T).max()),
        'medical_retained_space':float(abs((med@p)@p.T-med).max()),
        'anonymous_retained_space':float(abs((anon@p)@p.T-anon).max())}
    assert errors['projection_orthogonality']<1e-6 and errors['anonymous_gram']<1e-3
    assert errors['medical_retained_space']<1e-5 and errors['anonymous_retained_space']<1e-5
    for kind,array in [('medical',med),('anonymous',anon)]:
        q=array@p;q=q/np.linalg.norm(q,axis=1,keepdims=True)
        errors[kind+'_q0']=float(abs(q-t['q_'+kind].double().numpy()).max())
        assert errors[kind+'_q0']<1e-6
        errors[kind+'_projection_gram']=float(abs((array@p)@(array@p).T-array@array.T).max())
        assert errors[kind+'_projection_gram']<1e-3
    assert np.linalg.norm(med-anon)/np.linalg.norm(med)>.1
    assert t['texts']==r.PROMPTS
    return errors


def main():
    started=time.time();manifest=json.loads((r.HERE/'manifest.json').read_text());audits=[]
    assert sum(r.task_complete(g,f) for g in r.GROUPS for f in range(1,11))==30
    text_check=text_geometry_audit()
    protected=manifest['source_maps']['protected']
    assert all(r.sha_file(p)==h for p,h in protected.items()),'historical protected source changed'
    for fold in range(1,11):
        for group in r.GROUPS:
            audits.append(audit_one(group,fold))
            r.atomic_json(r.RESULTS/'audit_status.json',{'status':'AUDITING','completed':len(audits),'expected':30,'audits':audits})
    for fold in range(1,11):
        selected=[a for a in audits if a['fold']==fold];assert len({a['new_parameter_sha'] for a in selected})==1
        traces=[{t['epoch']:t for t in a['rng_trace']} for a in selected]
        for ep in set.intersection(*(set(x) for x in traces)):assert traces[0][ep]==traces[1][ep]==traces[2][ep]
    assert all(r.sha_file(p)==h for p,h in protected.items())
    aggregates=aggregate(audits)
    status={'status':'COMPLETE_AUDITED','completed_tasks':30,'independent_audits':30,'audits':audits,
        'text_geometry_independent_audit':text_check,'protected_sources_unchanged':True,'same_fold_init_and_common_rng_pass':True,'wall_seconds':time.time()-started,
        'audit_source_sha':r.sha_file(Path(__file__)),'source_sha':r.implementation_sha(),'aggregates':aggregates}
    r.atomic_json(r.RESULTS/'audit_status.json',status)
    manifest.update(status='COMPLETE_AUDITED',completed_tasks=30,independent_audits=30,audit_wall_seconds=status['wall_seconds'],
        audited_implementation_sha=r.implementation_sha(),audit_source_sha=r.sha_file(Path(__file__)))
    r.atomic_json(r.HERE/'manifest.json',manifest);print('COMPLETE_AUDITED 30/30',flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='recover':recover_log(sys.argv[2],int(sys.argv[3]))
    else:main()

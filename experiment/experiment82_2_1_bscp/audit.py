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
if 'bscp_runner' in sys.modules:r=sys.modules['bscp_runner']
else:
    import run as r
from bscp import EPS, GROUPS as GROUP_SPECS, teacher_snapshot, teacher_logits, counterfactual, auxiliary

SPECTRAL_DONE={}

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

def reference_metrics(prob,y):
    pred=prob.argmax(-1)
    return {'acc':accuracy_score(y,pred),'hard_auc':balanced_accuracy_score(y,pred),'prob_auc':roc_auc_score(y,prob[:,1]),
        'sen':recall_score(y,pred,pos_label=0),'spe':recall_score(y,pred,pos_label=1),'asd_f1':f1_score(y,pred,pos_label=0)}

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


def independent_head(head,z):
    n,l=head.mlp_head[0],head.mlp_head[1]
    v=F.layer_norm(z,n.normalized_shape,n.weight.detach(),n.bias.detach(),n.eps)
    return F.linear(F.gelu(F.linear(v,l.weight.detach(),l.bias.detach())),head.classifier.weight.detach(),head.classifier.bias.detach())


def independent_limit(raw,z):
    cap=.08*z.norm(dim=-1,keepdim=True).clamp_min(EPS)
    x=raw.norm(dim=-1,keepdim=True)/cap
    return raw*torch.where(x==0,torch.ones_like(x),x.tanh()/x.clamp_min(1e-30))


@torch.no_grad()
def independent_cf(state,module,z,expected=None):
    legal=torch.as_tensor(r.legal_train(state),device=z.device);zz=z.detach()[legal];y=state['y'][legal]
    h=(zz-module.mu)@module.U
    raw=h[:,:,None]*module.U.T[None]
    d=independent_limit(raw,zz[:,None])
    inputs=torch.cat([zz[:,None],zz[:,None]+d,zz[:,None]-d],1)
    logits=independent_head(state['cam'].Outputlayer,inputs.flatten(0,1)).reshape(len(zz),33,2)
    my=logits.gather(2,y[:,None,None].expand(-1,33,1)).squeeze(2)
    mo=logits.gather(2,(1-y)[:,None,None].expand(-1,33,1)).squeeze(2)
    margins=my-mo
    gains=torch.stack([margins[:,1:17]-margins[:,:1],margins[:,17:]-margins[:,:1],torch.zeros_like(margins[:,1:17])],-1)
    target=F.softmax(gains/.1,dim=2)
    weights=torch.ones_like(margins[:,0]);weights[margins[:,0]<=0]=1.5
    rb=r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    if expected is None:expected=counterfactual(teacher_snapshot(state['cam'].Outputlayer),z,module.U,module.mu,legal,y)
    errors={k:float((v-expected[k]).abs().max()) for k,v in [('margins',margins),('gains',gains),('target',target),('weights',weights)]}
    assert errors['margins']<8e-6 and errors['gains']<8e-6 and errors['target']<2e-5 and errors['weights']==0,errors
    assert expected['target'].shape==(len(legal),16,3) and torch.equal(expected['correct'],logits[:,0].argmax(-1)==y)
    assert rb==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    assert not expected['target'].requires_grad
    assert not np.isin(legal.cpu(),state['test_np']).any(), 'new label supervision includes held-out patient'
    return expected,errors


def formulas(module,d):
    z=d['z'];h=(z-module.mu)@module.U;hh=h/torch.sqrt(module.lam+EPS)
    ll,nu=module.loglam.expand_as(h),module.nu.expand_as(h)
    assert torch.equal(h,d['h']) and torch.equal(hh,d['hhat'])
    sp=module.spectral
    wp=F.linear(F.gelu(F.linear(torch.stack([hh.abs(),ll,nu],-1),sp[0].weight,sp[0].bias)),sp[2].weight,sp[2].bias).squeeze(-1).softmax(-1)
    tokens=F.layer_norm(F.linear(module.U.T,module.direction.weight,module.direction.bias)[None]+F.linear(torch.stack([hh,hh.abs(),ll,nu],-1),module.evidence.weight,module.evidence.bias),(32,),module.norm.weight,module.norm.bias,module.norm.eps)
    keys=F.linear(tokens,module.key.weight)
    score=torch.einsum('kd,njd->nkj',module.q0,keys)/math.sqrt(32)
    null=(module.q0@module.null/math.sqrt(32))[None,:,None].expand(len(z),-1,-1)
    a=torch.cat([score,null],-1).div(.5).softmax(-1)
    c=1-(1-a[:,:,:16]).prod(1);ws=c/(c.sum(-1,keepdim=True)+EPS)
    H=-(ws*(ws+EPS).log()).sum(-1)/math.log(16)
    beta=((1-a[:,:,16].mean(-1))*(1-H).clamp(.2,.8)).detach()
    w=beta[:,None]*ws+(1-beta[:,None])*wp
    ctx=torch.einsum('nkj,kd->njd',a[:,:,:16],module.q0)/(a[:,:,:16].sum(1)[:,:,None]+EPS)
    inp=torch.cat([tokens,ctx,torch.stack([hh,hh.abs(),ws,wp,w,ll,nu],-1)],-1)
    pol=module.policy;logits=F.linear(F.gelu(F.linear(inp,pol[0].weight,pol[0].bias)),pol[2].weight,pol[2].bias)
    p=torch.cat([logits,torch.zeros_like(logits[:,:,:1])],-1).softmax(-1);s=p[:,:,0]-p[:,:,1]
    base=((w-1/16)*h)@module.U.T;corr=((w*s)*h)@module.U.T;total=base+.5*corr
    delta=independent_limit(total,z)
    if module.fold==2:delta=delta.clone();delta[[376,406]]=0
    for k,v in [('tokens',tokens),('a',a),('w_sp',wp),('w_sem',ws),('w',w),('context',ctx),('P',p),('raw_base',base),('raw_corr',corr),('raw_total',total),('delta',delta)]:
        assert torch.allclose(v,d[k],atol=2e-7,rtol=2e-6),k
    assert float((delta.norm(dim=1)/z.norm(dim=1).clamp_min(EPS)).max())<=.080001
    return True


def independent_aux(state,module,z):
    legal=torch.as_tensor(r.legal_train(state),device=z.device);y=state['y'][legal]
    head=teacher_snapshot(state['cam'].Outputlayer)
    rb=r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    out=auxiliary(module,head,z,legal,y)
    t,errors=independent_cf(state,module,z,out['cf'])
    with torch.no_grad():
        d=out['aux_details'];new=independent_head(state['cam'].Outputlayer,z.detach()[legal]+d['delta'][legal])
        old=independent_head(state['cam'].Outputlayer,z.detach()[legal]);diff=new-old
        gain=diff.gather(1,y[:,None]).squeeze(1)-diff.gather(1,(1-y)[:,None]).squeeze(1)
        action=-(t['target']*d['P'][legal].clamp_min(EPS).log()).sum(2).mean(1)
        harm=(-gain).relu();vals={}
        for name,v in [('action',action),('harm',harm)]:
            vals[name]=sum(float((v[y==c]*t['weights'][y==c]).sum()/t['weights'][y==c].sum()) for c in (0,1))*.5
            assert abs(vals[name]-float(out[name]))<2e-6,(name,vals[name],float(out[name]))
        assert torch.allclose(gain,out['gain'],atol=2e-6,rtol=2e-6)
    norm=r.auxiliary_gradient_audit(out,state,module)
    assert all(p.grad is None for p in head.parameters())
    assert rb==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    return out,dict(independent_cf_errors=errors,independent_losses=vals,isolated_gradient_norms=norm)


def action_alignment(p,t,mask=None):
    if mask is not None:p,t=p[mask],t[mask]
    if not len(p):return {'n':0,'cross_entropy':None,'top1':None,'mae':None}
    return {'n':len(p),'cross_entropy':float(-(t*p.clamp_min(EPS).log()).sum(-1).mean()),
        'top1':float((p.argmax(-1)==t.argmax(-1)).float().mean()),'mae':float((p-t).abs().mean())}


def mechanism(state,module,c,d,cf,gain,logs):
    legal=torch.as_tensor(r.legal_train(state),device=module.U.device);idx=torch.as_tensor(state['test_np'],device=module.U.device)
    y=state['y'][legal];t=cf['target'];p=d['P'][legal]
    row=dict(group=module.group,fold=module.fold,legal_count=len(legal),teacher_error_supervised_fraction=1.,
        teacher_correct_count=int(cf['correct'].sum()),teacher_error_count=int((~cf['correct']).sum()),
        target_entropy=float(-(t*t.clamp_min(EPS).log()).sum(-1).mean()),
        near_uniform_fraction=float(((t-1/3).abs().amax(-1)<=.02).float().mean()),
        target_mean=json.dumps(t.mean((0,1)).cpu().tolist()),policy_mean=json.dumps(p.mean((0,1)).cpu().tolist()),
        target_argmax=json.dumps(torch.bincount(t.argmax(-1).flatten(),minlength=3).div(t.shape[0]*16).cpu().tolist()),
        policy_argmax=json.dumps(torch.bincount(p.argmax(-1).flatten(),minlength=3).div(p.shape[0]*16).cpu().tolist()),
        gain_positive_fraction=float((gain>1e-6).float().mean()),gain_negative_fraction=float((gain< -1e-6).float().mean()),
        teacher_plus_gain_mean=float(cf['gains'][:,:,0].mean()),teacher_minus_gain_mean=float(cf['gains'][:,:,1].mean()),
        alignment=json.dumps(action_alignment(p,t)),error_alignment=json.dumps(action_alignment(p,t,~cf['correct'])))
    for name,x in [('beta',d['beta']),('raw_base',d['raw_base'].norm(dim=1)),('raw_corr',d['raw_corr'].norm(dim=1)),
        ('raw_total',d['raw_total'].norm(dim=1)),('delta',d['delta'].norm(dim=1)),
        ('delta_relative_test',d['delta'][idx].norm(dim=1)/d['z'][idx].norm(dim=1).clamp_min(EPS)),
        ('raw_cosine',F.cosine_similarity(d['raw_base'],d['raw_corr'],dim=1)),('gain',gain)]:add_distribution(row,name,x)
    ratio=d['delta'].norm(dim=1)/d['z'].norm(dim=1).clamp_min(EPS)
    row['raw_over_cap_fraction']=float((d['raw_total'].norm(dim=1)>=.08*d['z'].norm(dim=1)).float().mean())
    row['near_cap_fraction']=float((ratio>=.0792).float().mean())
    row['signed_positive_fraction']=float((d['s'][legal]>1e-6).float().mean());row['signed_negative_fraction']=float((d['s'][legal]< -1e-6).float().mean())
    row['semantic_spectral_cosine']=float(F.cosine_similarity(d['w_sem'][idx],d['w_sp'][idx],dim=-1).mean())
    row['semantic_spectral_l1']=float((d['w_sem'][idx]-d['w_sp'][idx]).abs().sum(-1).mean())
    pa=F.normalize(d['a'][idx,:,:16],dim=-1);cos=pa@pa.transpose(1,2);mask=~torch.eye(8,dtype=torch.bool,device=idx.device)
    row['prompt_pairwise_cosine']=float(cos[:,mask].mean());row['prompt_cosine_matrix']=json.dumps(cos.mean(0).cpu().tolist())
    row['prompt_selections']=json.dumps(d['a'][idx,:,:16].mean(0).cpu().tolist())
    row['prompt_null_mean']=json.dumps(d['a'][idx,:,16].mean(0).cpu().tolist())
    for name,mask in [('correct',cf['correct']),('error',~cf['correct']),('asd',y==0),('td',y==1),('hard_asd',(y==0)&(~cf['correct'])),('hard_td',(y==1)&(~cf['correct']))]:
        row[name+'_n']=int(mask.sum())
        row[name+'_gain_mean']=float(gain[mask].mean()) if mask.any() else None
        row[name+'_gain_positive_fraction']=float((gain[mask]>1e-6).float().mean()) if mask.any() else None
        row[name+'_gain_negative_fraction']=float((gain[mask]< -1e-6).float().mean()) if mask.any() else None
        row[name+'_alignment']=json.dumps(action_alignment(p,t,mask))
    for lab,name in [(0,'asd'),(1,'td')]:row[name+'_teacher_error_rate']=float((~cf['correct'][y==lab]).float().mean())
    for loss in ('action','harm'):
        row[loss+'_loss_first']=logs[11]['losses'][loss];row[loss+'_loss_last']=logs[max(logs)]['losses'][loss]
        row[loss+'_loss_best']=logs[max(e for e in logs if logs[e]['improved'])]['losses'][loss]
    row['train_error_supervised_fraction']=1.
    row['train_teacher_error_count_mean']=float(np.mean([x['teacher_error_count'] for x in logs.values()]))
    row['train_gain_positive_fraction_mean']=float(np.mean([x['gain_positive_fraction'] for x in logs.values()]))
    with torch.no_grad():
        initial,_,_=r.make_module(state,module.group,c)
        _,di=initial(c['warmup_cam'].to(module.U.device))
        row['initial_P_mean']=json.dumps(di['P'].mean((0,1)).cpu().tolist())
        row['initial_corr_norm_max']=float(di['raw_corr'].norm(dim=1).max())
        _,same=initial(d['z'])
        row['initial_policy_on_best_alignment']=json.dumps(action_alignment(same['P'][legal],t))
    return row


def verified_logs(group,fold):
    rows=r.read_logs(r.paths(group,fold)[3]);logs={v['epoch']:v for v in rows}
    assert len(rows)==len(logs)
    return logs


def audit_one(group,fold):
    assert r.task_complete(group,fold)
    tp,bp,lp,log=r.paths(group,fold);task=json.loads(tp.read_text());best=r.torch_load(bp);latest=r.torch_load(lp)
    state,warmup,identity=r.prepare(fold);c=r.torch_load(r.RT/f'spectrum_f{fold:02d}.pt');text=r.torch_load(r.RT/'prompts.pt')
    module,opt,init=r.make_module(state,group,c,text);spectral=spectral_audit(c,state)
    old=r.torch_load(r.E822/'runtime'/f'spectrum_f{fold:02d}.pt')
    assert torch.equal(c['spectrum']['U'],old['spectrum']['U']) and torch.equal(c['spectrum']['mu'],old['spectrum']['mu'])
    for ck in (best,latest):
        assert ck['identity']==identity and ck['config']==r.CFG and ck['source_sha']==r.implementation_sha()
        assert ck['new_parameter_sha']==init and ck['spectra_sha']==c['spectra_sha'] and ck['text_identity_sha']==text['identity_sha']
    assert task['best_sha']==r.sha_file(bp) and task['latest_sha']==r.sha_file(lp)
    r.load_bundle(state,module,opt,best['current']);r.base_runtime().restore_rng(best['rng'])
    assert r.sha(r.bundle(state,module,opt))==best['bundle_sha']==r.sha(best['current'])
    assert torch.equal(module.q0.cpu(),text['q_'+GROUP_SPECS[group]]) and torch.equal(module.U.cpu(),c['spectrum']['U'].float())
    assert best['wait']==0 and best['next_epoch']==best['best_epoch'] and best['best_acc']==best['metrics']['acc']
    before=r.base_runtime().rng_sha(r.base_runtime().pack_rng());out,d,calls=replay(state,module,with_grad=True)
    values,prob,pred,y=independently_recompute(out['logp'],state);formulas(module,d)
    assert before==best['rng_sha']==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    assert np.array_equal(pred,best['prediction']) and np.array_equal(prob,best['probability'])
    assert all(abs(values[k]-best['metrics'][k])<1e-12 and abs(values[k]-task['metrics'][k])<1e-12 for k in r.METRICS)
    logs=verified_logs(group,fold);assert sorted(logs)==list(range(11,task['epochs_executed']+1))
    chosen=max(logs,key=lambda e:(logs[e]['acc'],-e));assert chosen==best['best_epoch']==task['best_epoch']
    reference={v['epoch']:v for v in r.historical_task(fold)['rng_trace']};parity=0
    for v in task['rng_trace']:
        assert all(logs[v['epoch']][k]==v[k] for k in ('rng_before','rng_after_train','rng_after_eval'))
        if v['epoch'] in reference:assert v==reference[v['epoch']];parity+=1
    assert len(task['rng_trace'])==task['epochs_executed']-10
    aux,auxcheck=independent_aux(state,module,d['z']);cf=aux['cf']
    legal=torch.as_tensor(r.legal_train(state),device=module.U.device)
    with torch.no_grad():row=mechanism(state,module,c,d,cf,aux['gain'],logs)
    cg=r.classification_gradient_audit(out,state,module)
    optimizer_checks={}
    for name,optimizer in [(k,state[k]) for k in ('opt_base','opt_cam','opt_added')]+[('opt_bscp',opt)]:
        sd=optimizer.state_dict();assert all(torch.isfinite(v).all() for st in sd['state'].values() for v in st.values() if torch.is_tensor(v))
        assert len(sd['state'])>0
        optimizer_checks[name]=dict(state_sha=r.sha(sd),state_count=len(sd['state']),lr=sd['param_groups'][0]['lr'],weight_decay=sd['param_groups'][0]['weight_decay'])
    row.update(four_optimizer_audit=json.dumps(optimizer_checks),classification_gradient_norms=json.dumps(cg),best_epoch=best['best_epoch'],termination_epoch=task['epochs_executed'],**module.counts())
    refs={};refpred={};refprob={}
    for name,path,key in [('B0',r.runtime().EXP/'.runtime/checkpoints'/f'{r.BASE_GROUP}_f{fold:02d}.best.pt','test_probability'),('E82-G2',r.E82/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt','probability'),('E82.2-G2',r.E822/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt','probability')]:
        ck=r.torch_load(path);assert ck['identity']==identity
        pr=np.asarray(ck[key]);m=reference_metrics(pr,y)
        assert all(abs(m[k]-ck['metrics'][k])<1e-12 for k in r.METRICS)
        refs[name]=m;refpred[name]=pr.argmax(1).tolist();refprob[name]=pr.tolist()
    for name in refs:
        row['vs_'+name]=json.dumps(flips(np.array(refpred[name]),pred,y))
        for label,sub in [(0,'asd'),(1,'td')]:row['vs_'+name+'_'+sub]=json.dumps(flips(np.array(refpred[name])[y==label],pred[y==label],y[y==label]))
    interventions=[]
    if group=='G2':
        qsave=module.q0.clone();psha=r.sha(r.cpu_state(module));co_logp=None
        for name in ('anonymous','semantic_off','correction_off','residual_off','action_neutral'):
            with torch.no_grad():module.q0.copy_(text['q_anonymous'].to(module.U.device) if name=='anonymous' else qsave)
            changed,cd,_=replay(state,module,'full' if name=='anonymous' else name)
            met,cp,cr,_=independently_recompute(changed['logp'],state)
            response={'intervention':name,'mean_abs_probability_change':float(abs(cp[:,0]-prob[:,0]).mean()),'max_abs_probability_change':float(abs(cp[:,0]-prob[:,0]).max()),**flips(pred,cr,y),'metrics':met}
            for lab,sub in [(0,'asd'),(1,'td')]:response[sub]=flips(pred[y==lab],cr[y==lab],y[y==lab])
            response['policy_l1_change']=float((cd['P'][legal]-d['P'][legal]).abs().mean())
            response['action_alignment']=action_alignment(cd['P'][legal],cf['target'])
            if name=='correction_off':co_logp=changed['logp'].detach().clone()
            if name=='action_neutral':assert torch.equal(changed['logp'],co_logp)
            if name=='semantic_off':
                with torch.no_grad():
                    _,sd=module(d['z'],'semantic_off');module.q0.copy_(text['q_anonymous'].to(module.U.device));_,sa=module(d['z'],'semantic_off')
                    assert torch.equal(sd['delta'],sa['delta'])
            interventions.append(response);del changed,cd
        with torch.no_grad():module.q0.copy_(qsave)
        assert r.sha(r.cpu_state(module))==psha
        g1ck=r.torch_load(r.paths('G1',fold)[1]);g1,_,_=r.make_module(state,'G1',c,text);g1.load_state_dict(g1ck['current']['bscp']);g1.eval()
        with torch.no_grad():_,gd=g1(d['z'])
        row['trained_G1_same_G2_CAM_alignment']=json.dumps(action_alignment(gd['P'][legal],cf['target']))
        row['trained_G1_G2_same_CAM_policy_L1']=float((gd['P'][legal]-d['P'][legal]).abs().mean())
        del g1,g1ck,gd
    held=np.asarray(state['test_np']);saved_y=state['y'].clone();saved_np=state['labels_np'].copy()
    state['y'][held]=1-state['y'][held];state['labels_np'][held]=1-state['labels_np'][held]
    try:
        changed,_,_=replay(state,module);assert torch.equal(out['logp'],changed['logp'])
    finally:state['y'].copy_(saved_y);state['labels_np'][:]=saved_np
    if fold==2:assert torch.count_nonzero(d['delta'][[376,406]])==0
    del aux,out,d,changed
    r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng'])
    assert r.sha(r.bundle(state,module,opt))==latest['bundle_sha']
    assert r.base_runtime().rng_sha(r.base_runtime().pack_rng())==latest['rng_sha']==task['rng_trace'][-1]['rng_after_eval']
    assert latest['next_epoch']==task['epochs_executed'] and (latest['wait']>200 or latest['next_epoch']==450)
    def next_update():
        r.zero(state,opt);o=r.forward(state,module,True);o['total'].backward();r.step(state,opt);del o
        with torch.no_grad():ev=r.forward(state,module,False)
        return r.sha(r.bundle(state,module,opt)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),ev['logp'].cpu().clone()
    a=next_update()
    # Fresh model and optimizer objects, loading only disk Latest.
    state2,_,_=r.prepare(fold);module2,opt2,_=r.make_module(state2,group,c,text)
    disk=r.torch_load(lp);r.load_bundle(state2,module2,opt2,disk['current']);r.base_runtime().restore_rng(disk['rng'])
    r.zero(state2,opt2);o=r.forward(state2,module2,True);o['total'].backward();r.step(state2,opt2);del o
    with torch.no_grad():ev=r.forward(state2,module2,False)
    b=(r.sha(r.bundle(state2,module2,opt2)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),ev['logp'].cpu())
    assert a[:2]==b[:2] and torch.equal(a[2],b[2])
    result={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','metrics':values,'prediction':pred.tolist(),'probability':prob.tolist(),'labels':y.tolist(),
        'reference_metrics':refs,'reference_predictions':refpred,'best_epoch':best['best_epoch'],'termination_epoch':task['epochs_executed'],
        'best_sha':task['best_sha'],'latest_sha':task['latest_sha'],'new_parameter_sha':init,'identity':identity,'spectral_audit':spectral,
        'auxiliary_independent_audit':auxcheck,'historical_common_rng_epochs':parity,'latest_restore':'four_optimizers_fresh_object_next_train_eval_parameters_probability_rng_exact',
        'mechanism':row,'interventions':interventions,'rng_trace':task['rng_trace']}
    del state,state2,module,module2,opt,opt2,best,latest,ev;torch.cuda.empty_cache();print(f'AUDIT PASS {group}/{fold}',flush=True)
    return result


def aggregate(audits):
    by={(a['group'],a['fold']):a for a in audits};foldrows=[];mechs=[]
    allgroups=(*r.GROUPS,'B0','E82-G2','E82.2-G2')
    for f in range(1,11):
        for g in allgroups:
            a=by[(g,f)] if g in r.GROUPS else by[('G2',f)]
            m=a['metrics'] if g in r.GROUPS else a['reference_metrics'][g]
            foldrows.append({'group':g,'fold':f,'best_epoch':a['best_epoch'] if g in r.GROUPS else '',
                'termination_epoch':a['termination_epoch'] if g in r.GROUPS else '',**{k:100*m[k] for k in r.METRICS}})
        for g in r.GROUPS:
            row=by[(g,f)]['mechanism'].copy();other=by[('G1' if g=='G2' else 'G2',f)]
            row['vs_other_group']=json.dumps(flips(np.asarray(other['prediction']),np.asarray(by[(g,f)]['prediction']),np.asarray(by[(g,f)]['labels'])))
            row['readonly_interventions']=json.dumps(by[(g,f)]['interventions']);mechs.append(row)
    summary=[];paired=[];index={(x['group'],x['fold']):x for x in foldrows}
    comparisons=[('G2','B0'),('G2','E82-G2'),('G2','E82.2-G2'),('G2','G1'),('G1','B0')]
    def predictions(g,f):
        a=by[('G2',f)]
        return np.array(by[(g,f)]['prediction'] if g in r.GROUPS else a['reference_predictions'][g])
    for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
        for g in allgroups:
            rows=[index[(g,f)] for f in folds];row={'group':g,'scope':scope,'n_folds':len(rows)}
            for k in r.METRICS:
                v=np.array([x[k] for x in rows]);row[k+'_mean']=float(v.mean());row[k+'_population_sd']=float(v.std(ddof=0))
            summary.append(row)
        for g,ref in comparisons:
            fl=[flips(predictions(ref,f),predictions(g,f),np.array(by[(g,f)]['labels'])) for f in folds]
            for k in r.METRICS:
                delta=np.array([index[(g,f)][k]-index[(ref,f)][k] for f in folds])
                paired.append({'comparison':g+' - '+ref,'scope':scope,'metric':k,'mean_delta_pp':float(delta.mean()),
                    'population_sd_delta_pp':float(delta.std(ddof=0)),'wins':int((delta>1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),'losses':int((delta< -1e-10).sum()),
                    **{n:sum(v[n] for v in fl) for n in ('corrected','harmed','net','flips')},**{'fold'+str(f)+'_delta_pp':float(v) for f,v in zip(folds,delta)}})
    r.write_csv(r.RESULTS/'fold_metrics.csv',foldrows);r.write_csv(r.RESULTS/'summary.csv',summary)
    r.write_csv(r.RESULTS/'paired_comparisons.csv',paired);r.write_csv(r.RESULTS/'mechanism.csv',mechs)
    def dm(g,ref,k):return next(x for x in paired if x['comparison']==g+' - '+ref and x['scope']=='all10' and x['metric']==k)
    def mean(g,k):return float(np.mean([m[k] for m in mechs if m['group']==g and m[k] is not None]))
    interventions=[]
    for name in ('anonymous','semantic_off','correction_off','residual_off','action_neutral'):
        a=[i for x in audits if x['group']=='G2' for i in x['interventions'] if i['intervention']==name]
        interventions.append({'intervention':name,'mean_abs_probability_change':float(np.mean([x['mean_abs_probability_change'] for x in a])),
            'max_abs_probability_change':max(x['max_abs_probability_change'] for x in a),**{k:sum(x[k] for x in a) for k in ('flips','corrected','harmed','net')},
            'metrics_mean_percent':{k:100*float(np.mean([x['metrics'][k] for x in a])) for k in r.METRICS},
            'by_class':{sub:{k:sum(x[sub][k] for x in a) for k in ('corrected','harmed','net','flips')} for sub in ('asd','td')}})
    corrections={g:{scope:{k:sum(json.loads(m['vs_B0'+('' if scope=='all' else '_'+scope)])[k] for m in mechs if m['group']==g) for k in ('corrected','harmed','net','flips')} for scope in ('all','asd','td')} for g in r.GROUPS}
    from scipy.stats import spearmanr
    association={}
    for g in r.GROUPS:
        rows=[m for m in mechs if m['group']==g]
        xs=[-json.loads(m['alignment'])['cross_entropy'] for m in rows];ys=[by[(g,m['fold'])]['metrics']['acc']-by[(g,m['fold'])]['reference_metrics']['B0']['acc'] for m in rows]
        association[g]=float(spearmanr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else None
    dacc=dm('G2','B0','acc');dba=dm('G2','B0','hard_auc');dtext=dm('G2','G1','acc');anon=interventions[0];corr=interventions[2]
    classification_pass=dacc['mean_delta_pp']>0 and dba['mean_delta_pp']>=0 and dacc['wins']>dacc['losses']
    text_pass=dtext['mean_delta_pp']>0 and dtext['wins']>dtext['losses']
    impact_pass=anon['mean_abs_probability_change']>=.001 and anon['net']<0
    correction_pass=corr['net']<0
    sen_pass=dm('G2','E82.2-G2','sen')['mean_delta_pp']>0
    prob_pass=dm('G2','B0','prob_auc')['mean_delta_pp']>=-.1
    alignment_pass=association['G2'] is not None and association['G2']>0
    if all([classification_pass,text_pass,impact_pass,correction_pass,sen_pass,prob_pass,alignment_pass]):verdict='GO'
    elif (max(dm(g,'B0','acc')['mean_delta_pp'] for g in r.GROUPS)>0 and correction_pass) or (dba['mean_delta_pp']>0 and sen_pass and correction_pass):verdict='PARTIAL'
    else:verdict='NO-GO'
    con={'verdict':verdict,'classification_pass':classification_pass,'text_control_pass':text_pass,'text_impact_pass':impact_pass,
        'correction_pass':correction_pass,'ASD_sensitivity_pass':sen_pass,'probability_auc_pass':prob_pass,'alignment_association_pass':alignment_pass,
        'alignment_vs_test_delta_spearman':association,'corrections':corrections,'interventions':interventions,
        'g2_vs_references':{ref:{k:dm('G2',ref,k)['mean_delta_pp'] for k in r.METRICS} for ref in ('B0','E82-G2','E82.2-G2','G1')},
        'route_action':'STOP_NO_E82.2.2_NO_TUNING' if verdict=='NO-GO' else 'NO_ADDITIONAL_EXPERIMENTS_AUTHORIZED_IN_THIS_RUN'}
    report=['# Experiment 82.2.1 — BSCP','', '**Bidirectional Spectral Counterfactual Prompt Correction / 双向谱反事实Prompt纠错**','',
        f'状态：COMPLETE_AUDITED；20/20正式训练、20/20 Best、20/20 Latest独立恢复审计。实际目录：`{r.HERE}`。','',
        '## 源码与论文审查','',
        '[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)的GFR按模态构建患者多关系图，用GCN传播；CAM交互模态；SPGS按文本相似度加权局部信息，RES把局部表示聚合到全局。论文ABIDE为871人、10折，原四指标ACC/AUC/SEN/SPE，报告92.30/92.22/91.08/93.36%。这些论文指标不替代本次历史B0。','',
        '[原作者源码](https://github.com/wanxixi11/SPromptGL)完整PLN含PEN；本地历史真正加载vendor PLNNoPEN：InputLayer→Encoder/FeedForward→flatten144→Outputlayer。保留该实际路径。原分类头LayerNorm144→Linear144,36→GELU→Dropout→Linear36,2→log_softmax；训练/评估状态沿用历史，仅冻结教师显式跳过Dropout。','',
        '[SSC论文](https://papers.miccai.org/miccai-2026/paper/4444_paper.pdf)及[公开代码](https://github.com/Hanyu1124/ssc)用于多Prompt互补证据与冗余问题的动机参考。SSC的patch和文本谱不等于CAM潜在方向；未移植WSI网络。公开预算round实现与论文floor存在细节差异，对本实现无依赖。原件/远程commit/SHA均在manifest。','',
        '[E82.2指定提交](https://github.com/tangaii/SPromptGL/commit/9f104bda39cc9c4deaf90aa25cb09fd2c4a8d302)的psce/run/audit/report与本地只读来源逐文件核对；E82/E82.2仅工程与历史比较来源，不加载其训练后模型作初始化。','',
        '## 冻结设计与完整继承','',
        'G1匿名与G2医学只有固定q0不同；两组每折可训练参数完全同SHA，init_seed=822100+fold，text_seed=822199。8条原RN50文本保留Nromal拼写，不将前后4条映射为TD/ASD。Original O由合法训练warmup CAM144在CPU float64重建，8 PCA+8正交协方差差异方向，与E82.2每折U和mu逐元素一致；不拆成模态槽。','',
        '唯一基座为E74.4-G2 Positive-DCR，通过原path_adapter加载真实E61 epoch10模型、全部三优化器、4096原关系对和Python/NumPy/Torch CPU/CUDA RNG。DARG保留GraphLearn阈值支持、统一图归一化、四GCN与仅正DeltaA局部疾病残差。原损失直接使用历史out[total]，不重写。opt_base→opt_cam→opt_added→新增AdamW按原顺序更新；原分类损失和历史训练权重不改。','',
        '统计3-32-1分支、16 Token、8×17 Attention、Null、连续Coverage及detach beta均保留。71维Policy拼接Token32+按Prompt注意聚合Context32+7标量，71-32-2、固定不动logit0。末层零初始化令初始P各1/3、signed s=0；第二次更新审计所有有效层梯度。','',
        '每个训练Epoch复制当前原头为冻结eval教师，在一次历史CAM的合法训练行上向量化33输入（原始/16增强/16抑制）。真实类别Margin为logit_y−logit_1-y；tau=.10，target=softmax([gain_plus,gain_minus,0]/tau)。所有合法训练患者含误分类患者参与，Margin<=0权重1.5，按ASD/TD分别加权平均再各占一半。','',
        'raw_base=((w−1/16)h)Uᵀ；raw_corr=((w*s)h)Uᵀ；raw_total=raw_base+0.5raw_corr，最终统一smooth_limit至8%。L=L_DARG+.001L_res+.01L_action+.01L_harm，L_harm=类别均衡weighted ReLU(−teacher_gain)。辅助另用z.detach前向，仅新增参数接收其梯度，主分类仍贯穿DARG/CAM/BSCP。Fold2原训练成员不改，新增监督排除377/407且其delta归零。','',
        'seed0、原10折、epoch10开始、max450、wait>200，Best仅测试ACC严格提高且平局最早；Latest每10轮和终止保存四优化器/RNG/wait/Best/源码身份。无参数搜索、额外seed、新模态、历史模型重训。科研规则和降级规则见训练前冻结config.json。','',
        '## 六指标','', '10折或去Fold2的9折算术均值±population SD，单位%。Hard-AUC等于硬预测BA，Prob-AUC为连续TD概率ROC-AUC；ASD=0。折间训练患者重叠，折不是独立生物学重复，不报告无依据的显著性。']
    for scope in ('all10','without_fold2'):
        report+=['',f'### {scope}','','|组|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['scope']==scope:report.append('|'+row['group']+'|'+'|'.join(f"{row[k+'_mean']:.4f} ± {row[k+'_population_sd']:.4f}" for k in r.METRICS)+'|')
    report+=['','### 正式20折','','|组|折|Best|终止|ACC|BA|Prob-AUC|SEN|SPE|F1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in foldrows:
        if row['group'] in r.GROUPS:report.append(f"|{row['group']}|{row['fold']}|{row['best_epoch']}|{row['termination_epoch']}|"+'|'.join(f'{row[k]:.4f}' for k in r.METRICS)+'|')
    report+=['','## 配对差值','','单位pp；每个六指标的W/T/L、两种scope的逐折差及纠错/误伤均在paired_comparisons.csv。','','|比较|ACC|BA|Prob-AUC|SEN|SPE|F1|ACC W/T/L|','|---|---:|---:|---:|---:|---:|---:|---|']
    for g,ref in comparisons:report.append('|'+g+'−'+ref+'|'+'|'.join(f"{dm(g,ref,k)['mean_delta_pp']:+.4f}" for k in r.METRICS)+f"|{dm(g,ref,'acc')['wins']}/{dm(g,ref,'acc')['ties']}/{dm(g,ref,'acc')['losses']}|")
    report+=['','## 动作、困难训练患者与真实残差收益','']
    for g in r.GROUPS:
        rows=[m for m in mechs if m['group']==g]
        means=lambda k:np.mean([json.loads(m[k]) for m in rows],axis=0).tolist()
        report.append(f"{g}：Best eval教师正确/错误患者（折患者出现次数）{sum(m['teacher_correct_count'] for m in rows)}/{sum(m['teacher_error_count'] for m in rows)}；错误患者100%进入新增监督，训练全程同样100%。ASD/TD错误率均值{100*mean(g,'asd_teacher_error_rate'):.3f}%/{100*mean(g,'td_teacher_error_rate'):.3f}%。这些重复训练患者计数不是独立受试者n。")
        report.append(f"增强/抑制/不动soft target均值：{means('target_mean')}；教师argmax分布{means('target_argmax')}；Policy均值{means('policy_mean')}，argmax分布{means('policy_argmax')}。教师熵{mean(g,'target_entropy'):.6f}（最大ln3=1.098612），目标接近均匀比例{100*mean(g,'near_uniform_fraction'):.3f}%（各概率距1/3≤.02）。增强/抑制收益均值{mean(g,'teacher_plus_gain_mean'):.6f}/{mean(g,'teacher_minus_gain_mean'):.6f}。soft均值与argmax含义不同，不动argmax比例不能替代软目标质量。")
        report.append(f"Policy对教师top1一致率{100*np.mean([json.loads(m['alignment'])['top1'] for m in rows]):.3f}%；错误训练患者一致率{100*np.mean([json.loads(m['error_alignment'])['top1'] for m in rows]):.3f}%。L_action首/末{mean(g,'action_loss_first'):.6f}/{mean(g,'action_loss_last'):.6f}，L_harm首/末{mean(g,'harm_loss_first'):.6f}/{mean(g,'harm_loss_last'):.6f}。")
        report.append(f"raw_base/raw_corr/raw_total平均范数{mean(g,'raw_base_mean'):.6f}/{mean(g,'raw_corr_mean'):.6f}/{mean(g,'raw_total_mean'):.6f}；base/corr cosine={mean(g,'raw_cosine_mean'):.6f}；delta平均范数{mean(g,'delta_mean'):.6f}，测试相对范数{100*mean(g,'delta_relative_test_mean'):.4f}%，raw超过cap比例{100*mean(g,'raw_over_cap_fraction'):.3f}%，最终达到99% cap比例{100*mean(g,'near_cap_fraction'):.3f}%。有符号增强/抑制方向比例{100*mean(g,'signed_positive_fraction'):.3f}%/{100*mean(g,'signed_negative_fraction'):.3f}%。")
        report.append(f"真实整体残差教师Margin改善/损害比例{100*mean(g,'gain_positive_fraction'):.3f}%/{100*mean(g,'gain_negative_fraction'):.3f}%，平均gain={mean(g,'gain_mean'):.6f}；教师正确/错误gain {mean(g,'correct_gain_mean'):.6f}/{mean(g,'error_gain_mean'):.6f}；ASD/TD {mean(g,'asd_gain_mean'):.6f}/{mean(g,'td_gain_mean'):.6f}；困难ASD/TD {mean(g,'hard_asd_gain_mean'):.6f}/{mean(g,'hard_td_gain_mean'):.6f}。初始P=1/3且corr为零；Best策略、各亚群和训练日志见mechanism.csv。")
        report.append(f"Prompt两两选择cos={mean(g,'prompt_pairwise_cosine'):.8f}；w_sem/w_sp cos={mean(g,'semantic_spectral_cosine'):.6f}，L1={mean(g,'semantic_spectral_l1'):.6f}；beta={mean(g,'beta_mean'):.6f}。8×8矩阵、8 Prompt方向/Null分布、相同G2 CAM上的G1/G2策略差均已输出。")
        report.append('相对B0各类纠错/误伤/净纠错：'+json.dumps(corrections[g],ensure_ascii=False)+'。\n')
    report+=['动作对齐负交叉熵与测试ACC增量的折间Spearman：'+json.dumps(association)+'。这是10折描述性关联，不能建立训练机制导致泛化改善的因果链。','',
        '## G2 Best只读干预','','以下纠错/误伤方向是完整模型→干预模型；负净值表示移除该部分有害于当前模型。','','|干预|mean abs ΔP|最大ΔP|翻转|纠错|误伤|净纠错|','|---|---:|---:|---:|---:|---:|---:|']
    for x in interventions:report.append('|'+x['intervention']+f"|{x['mean_abs_probability_change']:.9f}|{x['max_abs_probability_change']:.9f}|{x['flips']}|{x['corrected']}|{x['harmed']}|{x['net']}|")
    report+=['','干预六指标均值、ASD/TD纠错和Policy目标对齐见audit_status及mechanism.csv。Medical→Anonymous仅换q0。Semantic-off预先固定w_sem=w_sp、w=w_sp、beta=0、context=0，其余统计Token和7标量不变；已检查再换Query不影响输出。它是定义明确的输入退化，仍可能产生训练外分布，不能视为重训消融。Correction-off仅令s=0；Action-neutral与之逐元素输出完全一致。Residual-off使用当前已共同适配的骨干，不等于历史B0重训。','',
        '## 与E82.2比较和研究判断','',
        'E82.2只用正确患者的削弱正贡献监督方向；BSCP同时学习方向与增强/抑制/不动，覆盖错误训练患者，增加整体残差L_harm。但G2−E82.2还包含初始化及固定文本几何随机流变化；只有两个正式组，不能把差异单独归因于困难患者或L_harm。Correction-off检验当前已训练网络的即时纠错作用，不替代单因素重训。','',
        f"**{verdict}**。冻结判定检查：分类稳定提高={classification_pass}，Prob-AUC可接受={prob_pass}，相对E82.2 ASD灵敏度提高={sen_pass}，双向纠错即时收益={correction_pass}，医学优于匿名正式对照={text_pass}，文本非忽略诊断影响={impact_pass}，训练对齐/测试增益解释一致={alignment_pass}。所有指标和纠错一起解释，不以一次ACC或辅助loss下降认定成功。",
        '', '若NO-GO，正式停止此路线，不追加E82.2.2、调参或新实验；若PARTIAL，只承认相应的反事实正则线索；即便GO也只表示进一步研究候选，不代表满足MICCAI 2027创新或可靠泛化。','',
        '## 工程审计与限制','',
        f"20/20 Best完整重载概率逐元素相同，六指标独立sklearn复算；最早最大ACC/早停核对。公共历史RNG三元组{sum(a['historical_common_rng_epochs'] for a in audits)}条全部一致，同折两组共有epoch RNG亦完全一致；每次新增前向/教师操作实际检查零共有RNG消耗，无逐epoch重设seed或RNG回写。20/20 Latest由全新模型对象和四优化器从磁盘恢复，下一训练+评估全部参数、概率和RNG精确一致，临时审计更新不写回正式结果。",
        '', '独立33输入functional分类头重算、Margin正负、soft target、1.5困难权重、类均衡动作及harm损失、辅助梯度隔离、主分类全链梯度、8%统一限幅、Fold2新增路径保护、交换全部测试标签不改推理输出均通过。根模型、数据和历史Checkpoint保护SHA前后相同。',
        '', '首选路径的最小写入触发OSSFS errno5，正式运行前切换/model并通过fsync/rename/读回；无中途IO迁移。保留原val=test选模、CV前标准化/RFE、transductive患者图和Fold2训练重叠，排除Fold2后的9折仍保留前三项限制。单seed、仅两组的设计不能隔离每一新增机制。训练Margin是当前分类器的局部特征干预，不是医学因果效应、脑区病理证据或独立泛化。CLIP与CAM缺乏经过验证的语义预训练对齐，可训练Token/Key仍可吸收Query几何。','']
    (r.RESULTS/'report.md').write_text('\n'.join(report)+'\n')
    return dict(summary=summary,paired=paired,conclusion=con,interventions=interventions)


def append_handoff(status):
    path=Path('/model/spromptgl/experiment/handoff.md');marker='## Experiment 82.2.1 — BSCP（COMPLETE_AUDITED）'
    # Include the entire bounded scientific report so no requested result is lost in handoff.
    report=(r.RESULTS/'report.md').read_text();addition='\n'+marker+'\n\n'+report+'\n'
    with path.open('r+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX);original=f.read()
        if marker in original:
            assert addition in original;return dict(path=str(path),append_verified=True,already_present=True,sha256=r.sha_file(path))
        f.seek(0,2);f.write(addition);f.flush();os.fsync(f.fileno());f.seek(0);readback=f.read()
        assert readback==original+addition
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    return dict(path=str(path),append_verified=True,prefix_unchanged=True,appended_characters=len(addition),sha256=r.sha_file(path))


def main():
    started=time.time();manifest=json.loads((r.HERE/'manifest.json').read_text());audits=[]
    assert sum(r.task_complete(g,f) for g in r.GROUPS for f in range(1,11))==20
    assert r.implementation_sha()==manifest['preflight']['source_sha']
    assert r.sha_file(Path(__file__))==manifest['source_maps']['implementation'][str(Path(__file__))]
    protected=manifest['source_maps']['protected'];assert all(r.sha_file(p)==h for p,h in protected.items()),'protected source mismatch'
    text_check=text_geometry_audit()
    for fold in range(1,11):
        for group in r.GROUPS:
            audits.append(audit_one(group,fold))
            r.atomic_json(r.RESULTS/'audit_status.json',{'status':'AUDITING','completed':len(audits),'expected':20,'audits':audits})
    for fold in range(1,11):
        a=[v for v in audits if v['fold']==fold];assert len({v['new_parameter_sha'] for v in a})==1
        t=[{v['epoch']:v for v in x['rng_trace']} for x in a]
        for ep in set(t[0])&set(t[1]):assert t[0][ep]==t[1][ep]
    assert all(r.sha_file(p)==h for p,h in protected.items())
    aggregates=aggregate(audits)
    protection_sha=r.sha(protected)
    status=dict(status='COMPLETE_AUDITED',completed_tasks=20,independent_audits=20,latest_restore_audits=20,audits=audits,
        text_geometry_independent_audit=text_check,protected_sources_unchanged=True,protected_source_map_sha=protection_sha,
        same_fold_init_and_common_rng_pass=True,wall_seconds=time.time()-started,audit_source_sha=r.sha_file(Path(__file__)),source_sha=r.implementation_sha(),aggregates=aggregates)
    status['handoff']=append_handoff(status)
    r.atomic_json(r.RESULTS/'audit_status.json',status)
    manifest.update(status='COMPLETE_AUDITED',completed_tasks=20,independent_audits=20,latest_restore_audits=20,
        audited_implementation_sha=r.implementation_sha(),audit_source_sha=r.sha_file(Path(__file__)),scientific_conclusion=aggregates['conclusion'],protected_source_map_sha=protection_sha,handoff=status['handoff'])
    r.atomic_json(r.HERE/'manifest.json',manifest)
    print('COMPLETE_AUDITED 20/20; handoff appended and readback verified',flush=True)

if __name__=='__main__':main()

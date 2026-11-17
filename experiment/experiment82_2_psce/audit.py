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
from psce import EPS, GROUPS as GROUP_SPECS, teacher_snapshot, counterfactual, cf_kl


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



@torch.no_grad()
def independent_cf(state,module,z):
    """Functional reconstruction from original head tensors, no teacher helper."""
    idx=torch.as_tensor(r.legal_train(state),device=z.device);zz=z.detach()[idx];y=state['y'][idx]
    u=module.U;h=(zz-module.mu)@u
    raw=h[:,:,None]*u.T[None];n=raw.norm(dim=-1,keepdim=True)
    cap=.08*zz.norm(dim=-1)[:,None,None].clamp_min(EPS)
    ratio=n/cap
    # Independent analytic expression, including exact zero perturbation.
    scale=torch.where(n==0,torch.ones_like(n),torch.tanh(ratio)/ratio.clamp_min(1e-30))
    perturb=raw*scale
    inputs=torch.cat([zz[:,None],zz[:,None]-perturb],1).reshape(-1,144)
    head=state['cam'].Outputlayer;norm=head.mlp_head[0];linear=head.mlp_head[1]
    hidden=F.layer_norm(inputs,norm.normalized_shape,norm.weight.detach(),norm.bias.detach(),norm.eps)
    hidden=F.gelu(F.linear(hidden,linear.weight.detach(),linear.bias.detach()))
    logits=F.linear(hidden,head.classifier.weight.detach(),head.classifier.bias.detach()).reshape(len(idx),17,2)
    margins=logits.gather(2,y[:,None,None].expand(-1,17,1)).squeeze(-1)-logits.gather(2,(1-y)[:,None,None].expand(-1,17,1)).squeeze(-1)
    D=margins[:,:1]-margins[:,1:];positive=D.clamp_min(0)
    target=positive/(positive.sum(-1,keepdim=True)+EPS)
    correct=logits[:,0].argmax(-1)==y
    valid=correct & (margins[:,0]>0) & (D.max(-1).values>1e-6)
    teacher=teacher_snapshot(head);before=r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    expected=counterfactual(teacher,z,u,module.mu,idx,y)
    assert before==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    errors={k:float((a-expected[k]).abs().max()) for k,a in [('D',D),('margins',margins),('target',target)]}
    assert errors['D']<8e-6 and errors['margins']<8e-6 and errors['target']<1e-4,errors
    assert torch.equal(valid,expected['valid']) and torch.equal(correct,expected['correct'])
    assert not expected['target'].requires_grad
    assert torch.allclose(target[valid].sum(-1),positive[valid].sum(-1)/(positive[valid].sum(-1)+EPS),atol=2e-6,rtol=0)
    assert float((perturb.norm(dim=-1)/zz.norm(dim=-1,keepdim=True).clamp_min(EPS)).max())<=.080001
    return expected,errors


def formulas(module,d):
    h=(d['z']-module.mu)@module.U;assert torch.equal(h,d['h'])
    hh=h/torch.sqrt(module.lam+EPS);assert torch.equal(hh,d['hhat'])
    loglam=module.loglam.expand_as(h);nu=module.nu.expand_as(h)
    wp=module.spectral(torch.stack([hh.abs(),loglam,nu],-1)).squeeze(-1).softmax(-1)
    assert torch.equal(wp,d['w_sp'])
    tokens=module.norm(module.direction(module.U.T)[None]+module.evidence(torch.stack([hh,hh.abs(),loglam,nu],-1)))
    scores=torch.einsum('kd,njd->nkj',module.q0,module.key(tokens))/math.sqrt(32)
    null=(module.q0@module.null/math.sqrt(32))[None,:,None].expand(len(h),-1,-1)
    a=torch.cat([scores,null],-1).div(.5).softmax(-1);assert torch.equal(a,d['a'])
    coverage=1-(1-a[:,:,:16]).prod(1);ws=coverage/(coverage.sum(-1,keepdim=True)+EPS)
    assert torch.equal(ws,d['w_sem']) and torch.allclose(ws.sum(-1),torch.ones_like(h[:,0]),atol=2e-6,rtol=0)
    H=-(ws*(ws+EPS).log()).sum(-1)/math.log(16);beta=((1-a[:,:,16].mean(-1))*(1-H).clamp(.2,.8)).detach()
    assert torch.equal(beta,d['beta']) and not d['beta'].requires_grad
    w=beta[:,None]*ws+(1-beta[:,None])*wp;assert torch.equal(w,d['w'])
    raw=((w-1/16)*h)@module.U.T;assert torch.equal(raw,d['delta_raw'])
    cap=.08*d['z'].norm(dim=-1,keepdim=True).clamp_min(EPS);norm=raw.norm(dim=-1,keepdim=True);x=norm/cap
    delta=raw*torch.where(norm==0,torch.ones_like(x),torch.tanh(x)/x.clamp_min(1e-30))
    if module.fold==2:delta[[376,406]]=0
    assert torch.allclose(delta,d['delta'],atol=1e-7,rtol=2e-6)
    assert float((delta.norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)).max())<=.080001
    return True


def alignment(weights,cf):
    from scipy.stats import rankdata
    valid=cf['valid'].cpu().numpy();w=weights.detach().cpu().double().numpy()[valid];t=cf['target'].cpu().double().numpy()[valid]
    if not len(w):return {'n':0,'kl':None,'spearman':None,'spearman_finite_n':0,'top1':None,'top3_overlap':None}
    kl=np.sum(np.where(t>0,t*np.log(np.maximum(t,1e-300)/np.maximum(w,EPS)),0),axis=-1)
    wr=rankdata(w,axis=1);tr=rankdata(t,axis=1);wr-=wr.mean(-1,keepdims=True);tr-=tr.mean(-1,keepdims=True)
    den=np.linalg.norm(wr,axis=-1)*np.linalg.norm(tr,axis=-1);good=den>0
    rho=np.sum(wr[good]*tr[good],-1)/den[good]
    topw=np.argsort(-w,axis=1,kind='stable')[:,:3];topt=np.argsort(-t,axis=1,kind='stable')[:,:3]
    overlap=np.asarray([len(set(a)&set(b))/3 for a,b in zip(topw,topt)])
    return {'n':int(len(w)),'kl':float(kl.mean()),'spearman':float(rho.mean()) if len(rho) else None,
        'spearman_finite_n':int(good.sum()),'top1':float((w.argmax(-1)==t.argmax(-1)).mean()),'top3_overlap':float(overlap.mean())}


def mechanism(state,module,c,d,cf,logs):
    idx=torch.as_tensor(state['test_np'],device=module.U.device);legal=torch.as_tensor(r.legal_train(state),device=module.U.device)
    row={'group':module.group,'fold':module.fold,**module.counts(),**r.teacher_summary(cf),'valid_count':int(cf['valid'].sum()),'legal_count':len(legal),
        'spectrum_sha':c['spectra_sha'],'variance_coverage_warmup':c['spectrum']['variance_coverage']}
    for prefix,x in [('beta',d['beta']),('beta_test',d['beta'][idx]),('null',d['null'][idx]),('entropy',d['entropy'][idx]),
        ('residual_relative',d['delta'][idx].norm(dim=-1)/d['z'][idx].norm(dim=-1).clamp_min(EPS)),
        ('mask_relative',cf['mask_relative']),('true_margin',cf['margins'][:,0]),('masked_margin',cf['margins'][:,1:]),('D',cf['D'])]:add_distribution(row,prefix,x)
    row['target_mass_deficit_max']=float((1-cf['target'][cf['valid']].sum(-1)).abs().max()) if cf['valid'].any() else None
    row['D_mean_by_direction']=json.dumps(cf['D'].mean(0).cpu().tolist());row['D_positive_fraction_by_direction']=json.dumps((cf['D']>1e-6).float().mean(0).cpu().tolist())
    row['D_negative_fraction_by_direction']=json.dumps((cf['D']< -1e-6).float().mean(0).cpu().tolist())
    row['target_mean_by_direction']=json.dumps(cf['target'][cf['valid']].mean(0).cpu().tolist()) if cf['valid'].any() else '[]'
    for kind,w in [('semantic',d['w_sem'][legal]),('spectral',d['w_sp'][legal])]:
        row.update({kind+'_'+k:v for k,v in alignment(w,cf).items()})
    pa=d['a'][idx,:,:16];pn=F.normalize(pa,dim=-1);cos=pn@pn.transpose(1,2);mask=~torch.eye(8,dtype=torch.bool,device=idx.device)
    row['prompt_pairwise_cosine']=float(cos[:,mask].mean());row['prompt_pairwise_matrix']=json.dumps(cos.mean(0).cpu().tolist())
    row['prompt_direction_distributions']=json.dumps(pa.mean(0).cpu().tolist());row['prompt_null_means']=json.dumps(d['a'][idx,:,16].mean(0).cpu().tolist())
    row['prompt_target_alignment']=json.dumps([alignment(d['a'][legal,k,:16]/d['a'][legal,k,:16].sum(-1,keepdim=True).clamp_min(EPS),cf) for k in range(8)])
    row['semantic_spectral_cosine']=float(F.cosine_similarity(d['w_sem'][idx],d['w_sp'][idx],dim=-1).mean())
    row['semantic_mean_weights']=json.dumps(d['w_sem'][idx].mean(0).cpu().tolist());row['spectral_mean_weights']=json.dumps(d['w_sp'][idx].mean(0).cpu().tolist())
    with torch.no_grad():
        warm=r.torch_load(r.RT/'prompts.pt');initial,_,_=r.make_module(state,module.group,c,warm);initial.eval()
        _,di=initial(c['warmup_cam'].to(module.U.device))
        # Same current CAM and teacher isolate routing change, as well as warm-CAM init.
        _,same=initial(d['z'])
    row['initial_selector_current_teacher_alignment']=json.dumps(alignment(same['w_sem'][legal],cf))
    row['initial_selector_current_cam_l1']=float((same['w_sem'][idx]-d['w_sem'][idx]).abs().sum(-1).mean())
    row['warm_to_best_semantic_l1']=float((di['w_sem'][idx]-d['w_sem'][idx]).abs().sum(-1).mean())
    row['cf_loss_first']=logs[min(logs)]['losses']['cf'];row['cf_loss_last']=logs[max(logs)]['losses']['cf']
    row['cf_loss_best']=logs[max(e for e in logs if logs[e]['improved'])]['losses']['cf']
    row['cf_gradient_norm_min']=min(x['cf_gradient_norm'] for x in logs.values());row['cf_gradient_norm_mean']=float(np.mean([x['cf_gradient_norm'] for x in logs.values()]))
    row['teacher_valid_fraction_training_mean']=float(np.mean([x['teacher_valid_fraction'] for x in logs.values()]));row['teacher_valid_fraction_training_min']=min(x['teacher_valid_fraction'] for x in logs.values())
    if module.fold==2:assert torch.count_nonzero(d['delta'][[376,406]])==0 and not np.isin([376,406],r.legal_train(state)).any()
    return row


def reference_metrics(prob,y):
    pred=prob.argmax(-1)
    return {'acc':accuracy_score(y,pred),'hard_auc':balanced_accuracy_score(y,pred),'prob_auc':roc_auc_score(y,prob[:,1]),
        'sen':recall_score(y,pred,pos_label=0),'spe':recall_score(y,pred,pos_label=1),'asd_f1':f1_score(y,pred,pos_label=0)}


def audit_one(group,fold):
    assert r.task_complete(group,fold)
    tp,bp,lp,log=r.paths(group,fold);task=json.loads(tp.read_text());best=r.torch_load(bp);latest=r.torch_load(lp)
    state,warmup,identity=r.prepare(fold);c=r.torch_load(r.RT/f'spectrum_f{fold:02d}.pt');text=r.torch_load(r.RT/'prompts.pt')
    module,opt,init=r.make_module(state,group,c,text);spectral=spectral_audit(c,state)
    assert init==best['new_parameter_sha']==task['new_parameter_sha']
    for ck in (best,latest):
        assert ck['identity']==identity and ck['config']==r.CFG and ck['source_sha']==r.implementation_sha()
        assert ck['spectra_sha']==c['spectra_sha'] and ck['text_identity_sha']==text['identity_sha'] and ck['text_kind']==GROUP_SPECS[group]
    assert task['best_sha']==r.sha_file(bp) and task['latest_sha']==r.sha_file(lp)
    r.load_bundle(state,module,opt,best['current']);r.base_runtime().restore_rng(best['rng'])
    assert r.sha(r.bundle(state,module,opt))==best['bundle_sha']==r.sha(best['current'])
    assert torch.equal(module.q0.cpu(),text['q_'+GROUP_SPECS[group]]) and torch.equal(module.U.cpu(),c['spectrum']['U'].float())
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
    cf,cferrors=independent_cf(state,module,d['z']);legal=torch.as_tensor(r.legal_train(state),device=module.U.device)
    with torch.no_grad():row=mechanism(state,module,c,d,cf,logs)
    cf_loss=cf_kl(d['w_sem'],legal,cf)
    if cf['valid'].any():assert abs(float(cf_loss)-row['semantic_kl'])<1e-5
    active=list(module.named_parameters());cg=torch.autograd.grad(cf_loss,[p for n,p in active],allow_unused=True,retain_graph=True)
    row['best_cf_gradient_norms']=json.dumps({n:float(g.norm()) if g is not None else 0 for (n,p),g in zip(active,cg)})
    named=[(prefix+'.'+n,p) for prefix,m in [('darg',state['model']),('cam',state['cam']),('psce',module)] for n,p in m.named_parameters() if p.requires_grad]
    grads=torch.autograd.grad(out['cls'],[p for n,p in named]+[d['z'],out['fused']],allow_unused=True,retain_graph=True)
    sq={n:float(g.detach().double().square().sum().cpu()) if g is not None else 0. for (n,p),g in zip(named,grads)}
    norms={pre:math.sqrt(sum(v for n,v in sq.items() if n.startswith(pre+'.') or n==pre)) for pre in ('darg','cam','cam.Outputlayer','psce.direction','psce.evidence','psce.key','psce.null','psce.spectral')}
    norms.update(cam144=float(grads[-2].norm()),darg320=float(grads[-1].norm()))
    assert all(math.isfinite(v) and v>0 for v in norms.values())
    all_parameters={id(p):p for op in [state[k] for k in ('opt_base','opt_cam','opt_added')]+[opt] for pg in op.param_groups for p in pg['params'] if p.requires_grad}
    params=list(all_parameters.values());positions={id(p):i for i,p in enumerate(params)}
    residual=(d['delta'][legal].square().sum(-1)/(d['z'][legal].square().sum(-1)+EPS)).mean()
    total=out['total']+.01*cf_loss+.001*residual;total_grads=torch.autograd.grad(total,params,allow_unused=True)
    optimizer_checks={}
    for name,optimizer in [(k,state[k]) for k in ('opt_base','opt_cam','opt_added')]+[('opt_psce',opt)]:
        current=[p for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad]
        gn=math.sqrt(sum(float(total_grads[positions[id(p)]].double().square().sum()) for p in current if total_grads[positions[id(p)]] is not None))
        assert math.isfinite(gn) and gn>0
        sd=optimizer.state_dict();assert all(torch.isfinite(v).all() for st in sd['state'].values() for v in st.values() if torch.is_tensor(v))
        optimizer_checks[name]={'total_gradient_norm_best':gn,'state_sha':r.sha(sd),'state_count':len(sd['state']),
            'nonzero_moments':sum(int(torch.count_nonzero(st['exp_avg'])>0) for st in sd['state'].values() if 'exp_avg' in st),
            'lr':sd['param_groups'][0]['lr'],'weight_decay':sd['param_groups'][0]['weight_decay']}
    row.update(four_optimizer_audit=json.dumps(optimizer_checks),best_classification_gradient_norms=json.dumps(norms),best_epoch=best['best_epoch'],
        wall_seconds=task['wall_seconds'],peak_gpu_bytes=task['peak_gpu_bytes'])
    b0=r.torch_load(r.runtime().EXP/'.runtime/checkpoints'/f'{r.BASE_GROUP}_f{fold:02d}.best.pt');assert b0['identity']==identity
    b0prob=np.asarray(b0['test_probability']);bm=reference_metrics(b0prob,y);b0pred=b0prob.argmax(-1)
    assert all(abs(bm[k]-b0['metrics'][k])<1e-12 for k in r.METRICS)
    old=r.torch_load(r.E82/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt');assert old['identity']==identity
    oldprob=np.asarray(old['probability']);om=reference_metrics(oldprob,y)
    assert all(abs(om[k]-old['metrics'][k])<1e-12 for k in r.METRICS)
    row['vs_b0']=json.dumps(flips(b0pred,pred,y));row['probability_change_b0_mean']=float(abs(prob[:,0]-b0prob[:,0]).mean())
    for label,name in [(0,'asd'),(1,'td')]:row['vs_b0_'+name]=json.dumps(flips(b0pred[y==label],pred[y==label],y[y==label]))
    interventions=[]
    if group=='G2':
        qsave=module.q0.clone();psha=r.sha(r.cpu_state(module))
        for name in ('anonymous','prompt_permutation','semantic_off','residual_off'):
            with torch.no_grad():
                if name=='anonymous':module.q0.copy_(text['q_anonymous'].to(module.U.device))
                elif name=='prompt_permutation':module.q0.copy_(qsave[torch.tensor([7,6,5,4,3,2,1,0],device=module.U.device)])
                else:module.q0.copy_(qsave)
            changed,cd,_=replay(state,module,'semantic_off' if name=='semantic_off' else 'noop' if name=='residual_off' else 'full')
            met,cp,cr,_=independently_recompute(changed['logp'],state)
            response={'intervention':name,'mean_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).mean()),
                'max_abs_asd_probability_change':float(abs(cp[:,0]-prob[:,0]).max()),**flips(pred,cr,y),'metrics':met,
                'flipped_rows_1based':(state['test_np'][cr!=pred]+1).tolist()}
            if name=='anonymous':
                row['fixed_best_anonymous_alignment']=json.dumps(alignment(cd['w_sem'][legal],cf))
                row['fixed_best_medical_anonymous_wsem_l1']=float((cd['w_sem'][legal]-d['w_sem'][legal]).abs().sum(-1).mean())
            if name=='prompt_permutation':assert np.array_equal(pred,cr) and abs(cp-prob).max()<2e-6
            interventions.append(response);del changed,cd
        with torch.no_grad():module.q0.copy_(qsave)
        assert r.sha(r.cpu_state(module))==psha
        held=np.setdiff1d(state['test_np'],r.legal_train(state));saved_y=state['y'].clone();saved_np=state['labels_np'].copy()
        state['y'][held]=1-state['y'][held];state['labels_np'][held]=1-state['labels_np'][held]
        try:
            changed,_,_=replay(state,module);assert torch.equal(out['logp'],changed['logp'])
        finally:state['y'].copy_(saved_y);state['labels_np'][:]=saved_np
        # Compare trained G1/G2 on the same current G2 CAM to avoid feature drift confounding.
        g1ck=r.torch_load(r.paths('G1',fold)[1]);g1,_,_=r.make_module(state,'G1',c,text);g1.load_state_dict(g1ck['current']['psce']);g1.eval()
        with torch.no_grad():_,gd=g1(d['z'])
        row['trained_g1_on_g2_cam_alignment']=json.dumps(alignment(gd['w_sem'][legal],cf))
        row['trained_g1_g2_same_cam_wsem_l1']=float((gd['w_sem'][legal]-d['w_sem'][legal]).abs().sum(-1).mean())
        del g1,g1ck,gd
    row['readonly_interventions']=json.dumps(interventions)
    r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng'])
    assert r.sha(r.bundle(state,module,opt))==latest['bundle_sha']
    assert r.base_runtime().rng_sha(r.base_runtime().pack_rng())==latest['rng_sha']==task['rng_trace'][-1]['rng_after_eval']
    assert latest['next_epoch']==task['epochs_executed'] and (latest['wait']>200 or latest['next_epoch']==450)
    # Two repeated next updates are audit-only; no checkpoint or training log is changed.
    def next_update():
        r.zero(state,opt);o=r.forward(state,module,True);o['total'].backward();r.step(state,opt)
        with torch.no_grad():ev=r.forward(state,module,False)
        return r.sha(r.bundle(state,module,opt)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),ev['logp'].cpu().clone()
    a=next_update();r.load_bundle(state,module,opt,latest['current']);r.base_runtime().restore_rng(latest['rng']);b=next_update()
    assert a[:2]==b[:2] and torch.equal(a[2],b[2])
    result={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','metrics':values,'prediction':pred.tolist(),'probability':prob.tolist(),'labels':y.tolist(),
        'b0_metrics':bm,'b0_prediction':b0pred.tolist(),'e82_metrics':om,'e82_prediction':oldprob.argmax(-1).tolist(),
        'best_epoch':best['best_epoch'],'best_sha':task['best_sha'],'latest_sha':task['latest_sha'],'new_parameter_sha':init,'identity':identity,
        'spectral_audit':spectral,'counterfactual_independent_max_errors':cferrors,'historical_common_rng_epochs':parity,
        'probability_max_abs_error':float(abs(prob-best['probability']).max()),'latest_restore':'four_optimizers_parameters_next_train_eval_probability_rng_exact',
        'mechanism':row,'interventions':interventions,'rng_trace':task['rng_trace']}
    del out,d,state,module,opt,best,latest,b0,old;torch.cuda.empty_cache();print(f'AUDIT PASS {group}/{fold}',flush=True)
    return result


def aggregate(audits):
    by={(a['group'],a['fold']):a for a in audits};foldrows=[];mechs=[]
    for f in range(1,11):
        for g in (*r.GROUPS,'B0','E82-G2'):
            a=by[(g,f)] if g in r.GROUPS else by[('G2',f)]
            m=a['metrics'] if g in r.GROUPS else a['b0_metrics'] if g=='B0' else a['e82_metrics']
            foldrows.append({'group':g,'fold':f,'best_epoch':a['best_epoch'] if g in r.GROUPS else '',**{k:100*m[k] for k in r.METRICS}})
        for g in r.GROUPS:
            row=by[(g,f)]['mechanism'].copy();other=by[('G1' if g=='G2' else 'G2',f)]
            row['vs_other_group']=json.dumps(flips(np.asarray(other['prediction']),np.asarray(by[(g,f)]['prediction']),np.asarray(by[(g,f)]['labels'])))
            mechs.append(row)
    summary=[];paired=[];index={(x['group'],x['fold']):x for x in foldrows}
    comparisons=[('G2','B0'),('G2','E82-G2'),('G2','G1'),('G1','B0')]
    for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
        for g in (*r.GROUPS,'B0','E82-G2'):
            rows=[index[(g,f)] for f in folds];row={'group':g,'scope':scope,'n_folds':len(rows)}
            for k in r.METRICS:
                v=np.array([x[k] for x in rows]);row[k+'_mean']=float(v.mean());row[k+'_population_sd']=float(v.std(ddof=0))
            summary.append(row)
        for g,ref in comparisons:
            for k in r.METRICS:
                delta=np.array([index[(g,f)][k]-index[(ref,f)][k] for f in folds])
                paired.append({'comparison':g+' - '+ref,'scope':scope,'metric':k,'mean_delta_pp':float(delta.mean()),
                    'population_sd_delta_pp':float(delta.std(ddof=0)),'wins':int((delta>1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),
                    'losses':int((delta< -1e-10).sum()),**{'fold'+str(f)+'_delta_pp':float(v) for f,v in zip(folds,delta)}})
    r.write_csv(r.RESULTS/'fold_metrics.csv',foldrows);r.write_csv(r.RESULTS/'summary.csv',summary)
    r.write_csv(r.RESULTS/'paired_comparisons.csv',paired);r.write_csv(r.RESULTS/'mechanism.csv',mechs)
    def dm(g,ref,k):return next(x for x in paired if x['comparison']==g+' - '+ref and x['scope']=='all10' and x['metric']==k)
    def mean(g,k):return float(np.mean([m[k] for m in mechs if m['group']==g]))
    interventions=[]
    for name in ('anonymous','prompt_permutation','semantic_off','residual_off'):
        a=[i for x in audits if x['group']=='G2' for i in x['interventions'] if i['intervention']==name]
        interventions.append({'intervention':name,'mean_abs_probability_change':float(np.mean([x['mean_abs_asd_probability_change'] for x in a])),
            'max_abs_probability_change':max(x['max_abs_asd_probability_change'] for x in a),**{k:sum(x[k] for x in a) for k in ('flips','corrected','harmed','net')}})
    corrections={g:{k:sum(json.loads(m['vs_b0'])[k] for m in mechs if m['group']==g) for k in ('corrected','harmed','net','flips')} for g in r.GROUPS}
    g2=[m for m in mechs if m['group']=='G2'];anon=[json.loads(m['fixed_best_anonymous_alignment']) for m in g2]
    target_anon_rho=float(np.mean([x['spearman'] for x in anon]));target_med_rho=mean('G2','semantic_spearman');target_sp_rho=mean('G2','spectral_spearman')
    med_anon_wins=sum(m['semantic_spearman']>a['spearman'] for m,a in zip(g2,anon))
    med_sp_wins=sum(m['semantic_spearman']>m['spectral_spearman'] for m in g2)
    dacc=dm('G2','B0','acc');dba=dm('G2','B0','hard_auc');dtext=dm('G2','G1','acc');anonint=interventions[0]
    classification_pass=dacc['mean_delta_pp']>0 and dba['mean_delta_pp']>0 and dacc['wins']>dacc['losses']
    text_pass=dtext['mean_delta_pp']>0 and dtext['wins']>dtext['losses']
    alignment_pass=target_med_rho>target_anon_rho and target_med_rho>target_sp_rho and med_anon_wins>=6 and med_sp_wins>=6
    impact_pass=anonint['flips']>0 and anonint['harmed']>=anonint['corrected']
    if classification_pass and text_pass and alignment_pass and impact_pass and dm('G2','B0','prob_auc')['mean_delta_pp']>=-.1 and corrections['G2']['net']>0:verdict='GO'
    elif dm('G1','B0','acc')['mean_delta_pp']>0 or (classification_pass and not text_pass):verdict='PARTIAL'
    else:verdict='NO-GO'
    from scipy.stats import spearmanr
    association={}
    for g in r.GROUPS:
        rows=[m for m in mechs if m['group']==g]
        for feature in ('semantic_spearman','semantic_kl','teacher_valid_fraction_training_mean'):
            xs=[m[feature] for m in rows];ys=[by[(g,m['fold'])]['metrics']['acc']-by[(g,m['fold'])]['b0_metrics']['acc'] for m in rows]
            association[g+'_'+feature+'_vs_delta_acc']={'spearman':float(spearmanr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else None,'n':10}
    conclusion={'verdict':verdict,'g2_vs_b0':{k:dm('G2','B0',k)['mean_delta_pp'] for k in r.METRICS},
        'medical_target_spearman':target_med_rho,'anonymous_same_best_target_spearman':target_anon_rho,'spectral_target_spearman':target_sp_rho,
        'medical_alignment_better_than_anonymous_folds':med_anon_wins,'medical_alignment_better_than_spectral_folds':med_sp_wins,
        'classification_pass':classification_pass,'text_control_pass':text_pass,'alignment_pass':alignment_pass,'nonnegligible_text_decision_impact_pass':impact_pass,
        'corrections':corrections,'training_alignment_test_delta_association':association,'text_interventions':interventions,'no_additional_training':'Stop this second-module route on NO-GO; no E82.2.1 tuning or new experiment.'}
    texts=r.torch_load(r.RT/'prompts.pt');execution=json.loads((r.RT/'execution.json').read_text())
    report=['# Experiment 82.2 — PSCE','', '**Prompt-guided Spectral Counterfactual Evidence Learning / 疾病Prompt引导的谱反事实证据学习**','',
        f'状态：COMPLETE_AUDITED；20/20正式训练，20/20 Best独立审计。实际目录：`{r.HERE}`。',
        '', '仅seed=0、871名患者、原固定10折。B0为E74.4-G2 DARG；所有新增组直接逐折恢复E61 epoch10模型、三个优化器、全部RNG和原4096训练关系对。历史B0、E82均只读，无训练后权重初始化。',
        '', '## 源码、论文与冻结方法','',
        '[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)和[原作者仓库](https://github.com/wanxixi11/SPromptGL)、[用户仓库](https://github.com/tangaii/SPromptGL)均已核查。原GFR按患者模态相似度建多关系图并作消息传播；CAM在患者内部作跨模态交互。论文SPGS/PEN局部语义路径在实际历史vendor PLNNoPEN中未执行。本次不恢复PEN。真实CAM144由InputLayer→Encoder/FeedForward后flatten产生，Hook仅改变原OutputLayer输入。原头执行LayerNorm(144)→Linear(144,36)→GELU→Dropout→Linear(36,2)→log_softmax。',
        '', '[SSC论文](https://papers.miccai.org/miccai-2026/paper/4444_paper.pdf)和[公开源码](https://github.com/Hanyu1124/ssc)提供多Prompt预算检索、文本谱补充、冗余处理及熵融合参考。公开代码的预算取整与论文floor细节不同；本实验只保留既定E82连续覆盖与融合，并未移植SSC网络。源码、论文原件SHA和远程commit见manifest/runtime/source_review.json。',
        '', 'G1=PSCE-Anonymous，G2=PSCE-Medical；两组唯一差异为冻结q0。Original O谱基由合法训练warmup CAM的8个PCA方向与8个正交类别协方差差异方向组成，CPU float64逐元素核对E82-G2，U冻结。统计分支仍为[|hhat|,norm_loglambda,norm_nu]→3-32-1→softmax。语义分支保留16 Token、8 Query、16+Null注意候选和连续覆盖，固定32维公共投影，无自由文本线性映射。',
        '', '每个训练Epoch从当前原OutputLayer复制冻结eval教师；仅合法训练原始CAM进入17组轻量批量分类。对每个方向的h_j U_j实施8%平滑受限遮蔽，绝非完整方向删除。D=真实类别原Margin−遮蔽后Margin，正贡献表示当前模型对正确类别的支持随削弱而下降。仅教师正确、Margin>0且max(D)>1e−6的合法训练患者产生detach的ReLU(D)/(sum+EPS)目标。目标不回传到教师或CAM；KL只监督可微w_sem。',
        '', 'L_total=L_cls+L_graph+0.05L_pair+0.01L_adv+0.01L_preserve+0.001L_res+0.01L_CF。历史优化器按opt_base→opt_cam→opt_added更新，再由AdamW(lr=0.0005,wd=0.01)更新PSCE。beta=((1−Null)*clamp(1−H,0.2,0.8)).detach；残差严格≤8%。epoch10继续至450或wait>200，仅测试ACC严格提高保存Best，平局保留最早。所有科研参数正式训练前冻结，无网格、额外种子或新增组。',
        '',f"文本秩{texts['rank']}；匿名Gram误差{float(texts['gram_error']):.3g}（float32 {float(texts['gram_error_float32']):.3g}）；公共投影正交误差{float(texts['projection_orthogonality']):.3g}；医学/匿名投影Gram误差{float(texts['projection_gram_medical']):.3g}/{float(texts['projection_gram_anonymous']):.3g}。精确保留原拼写Nromal brain volume。",'','## 六项指标','', '均值±population SD，单位%；Hard-AUC沿用历史硬预测BA定义，Prob-AUC为连续概率ROC-AUC。']
    for scope in ('all10','without_fold2'):
        report+=['', '### '+scope,'','|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['scope']==scope:report.append('|'+row['group']+'|'+'|'.join(f"{row[k+'_mean']:.4f} ± {row[k+'_population_sd']:.4f}" for k in r.METRICS)+'|')
    report+=['','### 全部逐折指标','','|组|折|Best epoch|ACC|BA|Prob-AUC|SEN|SPE|F1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in foldrows:
        if row['group'] in r.GROUPS:report.append(f"|{row['group']}|{row['fold']}|{row['best_epoch']}|"+'|'.join(f'{row[k]:.4f}' for k in r.METRICS)+'|')
    report+=['','## 配对比较','','单位百分点；两种scope全部六项指标逐折差与W/T/L见paired_comparisons.csv。历史test=val选模下仅为描述性结果，不构成无偏显著性证据。','','|比较|ACC Δ|BA Δ|Prob-AUC Δ|SEN Δ|SPE Δ|F1 Δ|ACC W/T/L|','|---|---:|---:|---:|---:|---:|---:|---|']
    for g,ref in comparisons:report.append('|'+g+'−'+ref+'|'+'|'.join(f"{dm(g,ref,k)['mean_delta_pp']:+.4f}" for k in r.METRICS)+f"|{dm(g,ref,'acc')['wins']}/{dm(g,ref,'acc')['ties']}/{dm(g,ref,'acc')['losses']}|")
    report+=['','## 反事实证据、Prompt和分类机制','']
    for g in r.GROUPS:
        report.append(f"{g}：训练全程教师有效比例均值{100*mean(g,'teacher_valid_fraction_training_mean'):.3f}%；Best独立eval教师正确/有效比例{100*mean(g,'teacher_correct_fraction'):.3f}%/{100*mean(g,'teacher_valid_fraction'):.3f}%；正/负方向数均值{mean(g,'positive_directions_mean'):.3f}/{mean(g,'negative_directions_mean'):.3f}。受限遮蔽相对范数均值{100*mean(g,'mask_relative_mean'):.4f}%，上限全折{100*max(m['mask_relative_max'] for m in mechs if m['group']==g):.4f}%。")
        report.append(f"w_sem对target：KL={mean(g,'semantic_kl'):.6f}、Spearman={mean(g,'semantic_spearman'):.6f}、Top1={100*mean(g,'semantic_top1'):.3f}%、Top3重合={100*mean(g,'semantic_top3_overlap'):.3f}%；w_sp对target：Spearman={mean(g,'spectral_spearman'):.6f}。训练CF loss首/末{mean(g,'cf_loss_first'):.6f}/{mean(g,'cf_loss_last'):.6f}，CF梯度范数均值{mean(g,'cf_gradient_norm_mean'):.6f}；KL下降与最终分类收益分别评价。")
        report.append(f"8 Prompt两两选择cos={mean(g,'prompt_pairwise_cosine'):.8f}；Null={mean(g,'null_mean'):.6f}、熵={mean(g,'entropy_mean'):.6f}；beta全患者均值={mean(g,'beta_mean'):.6f}；语义/统计cos={mean(g,'semantic_spectral_cosine'):.6f}；测试残差平均相对范数={100*mean(g,'residual_relative_mean'):.4f}%。")
        ct=corrections[g];report.append(f"相对B0纠错{ct['corrected']}、误伤{ct['harmed']}、净纠错{ct['net']}；硬翻转{ct['flips']}。ASD/TD分项、各方向贡献、Margin变化、Prompt平均分布、beta/Null/残差分位数、梯度与概率变化见mechanism.csv。")
        report.append('')
    report+=['逐折训练对齐/有效教师比例与测试ACC相对B0增量的描述性Spearman：'+json.dumps(association,ensure_ascii=False)+'。仅10折且历史test选模，不是泛化或因果证明。','']
    report+= [f"固定G2 Best及其训练CAM，医学/匿名/普通谱统计对同一教师target的Spearman分别{target_med_rho:.6f}/{target_anon_rho:.6f}/{target_sp_rho:.6f}。医学优于匿名{med_anon_wins}/10折，优于统计{med_sp_wins}/10折。两种文本的正式训练性能差由G2−G1评价，固定Best文本替换仅为即时响应。",'',
        '逐折机制含同一G2 CAM上的训练后G1/G2选择差、初始化选择器至Best变化、8条Prompt各自target相关和Top方向重合。高度同质Prompt不能作为8种医学机制证据。较低训练KL或较高训练相关不能直接证明测试泛化。','', '## G2 Best只读干预','',
        '|干预|平均绝对ASD概率变化|最大变化|硬翻转|纠错|误伤|净纠错|','|---|---:|---:|---:|---:|---:|---:|']
    for x in interventions:report.append('|'+x['intervention']+f"|{x['mean_abs_probability_change']:.9f}|{x['max_abs_probability_change']:.9f}|{x['flips']}|{x['corrected']}|{x['harmed']}|{x['net']}|")
    report+=['','Prompt顺序交换保持Query集合不变；对称覆盖理论上置换不变，已检查浮点误差与硬预测一致；它不是医学语义证据。residual_off比较当前已训练DARG/CAM骨干与PSCE决策，不是重新训练历史B0。','', '## 工程与独立审计','',
        f"20/20 Best完整重放，原始概率最大误差{max(a['probability_max_abs_error'] for a in audits):.3g}；六指标由sklearn独立复算。最早最大ACC与原始日志核对。共同历史Epoch三元组{sum(a['historical_common_rng_epochs'] for a in audits)}条严格一致；两组同折初始化及共同Epoch RNG严格一致。20/20 Latest全部参数和四个优化器/RNG恢复，并各自重复下一次训练+评估检验数值完全一致。审计临时下一步不写回正式checkpoint。",
        '', '独立公式审计直接从原头张量用functional LayerNorm/Linear/GELU/Linear复算17组Margin、贡献、筛选、target和遮蔽。教师零RNG、原头不受修改、测试标签交换预测不变、Fold2新增教师/损失排除及377/407残差为零均通过。原数据、根目录模型、历史warmup/Best及真实依赖SHA前后相同。',
        '',f"首选OSSFS目录创建出现errno 5，在正式训练前转入/model；本次运行最大worker={execution['max_workers_observed']}，CPU线程每进程1；正式耗时{execution.get('wall_seconds',0):.1f}s，失败事件{len(execution['failures'])}；整卡采样峰值{execution.get('sampled_peak_device_mib',0)}MiB，任务allocated峰值{max(m['peak_gpu_bytes'] for m in mechs)/2**30:.3f}GiB。采样可能漏掉瞬时峰值。",
        '', '保留历史val=test选Best、CV前标准化/RFE、transductive图和Fold2训练重叠；排除Fold2后的9折仍保留前三项偏倚。16维潜在方向不是医学脑区。教师来自当前模型，反映分类器依赖，可能是自蒸馏或正则化而非医学真值；CLIP文本与CAM没有经过验证的预训练对齐。固定文本投影后可训练Key/Token仍能吸收部分旋转；G2−E82-G2包含投影约束变化，不能完全归因于CF。',
        '', '## 科研结论','',f"**{verdict}**。分类稳定改善={classification_pass}，医学/匿名正式对照改善={text_pass}，医学target对齐稳定优于两种对照={alignment_pass}，文本具有非忽略硬决策影响={impact_pass}。"]
    if verdict=='NO-GO':report+=['','正式停止PSSEC/PCDE/PSCE第二创新模块路线；不追加E82.2.1调参，不自行启动新实验。']
    elif verdict=='PARTIAL':report+=['','只能认定数据驱动反事实证据存在进一步研究的线索，不能声称疾病Prompt医学语义已验证；本次不追加实验。']
    else:report+=['','GO仅表示值得在严格独立验证协议下继续研究，不表示已经满足MICCAI 2027创新与泛化要求；本次不追加实验。']
    (r.RESULTS/'report.md').write_text('\n'.join(report)+'\n')
    return {'summary':summary,'paired':paired,'interventions':interventions,'conclusion':conclusion}

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
        cfgrads,cfgnorm=r.aux_gradients(out,module);teacher_stats=r.teacher_summary(out['details']['cf'])
        out['total'].backward();assert r.finite_grads(state,module);grad=r.gradients(state,module);r.step(state,opt)
        after=base.rng_sha(base.pack_rng())
        losses={k:float(out[k].detach().cpu()) for k in ('cls','graph','pair','adv','preserve','darg_total','res','cf','total')};del out
        with torch.no_grad():o=r.forward(state,module,False);metrics=base.classification_metrics(o['logp'],state['test'],state['y'])
        after_eval=base.rng_sha(base.pack_rng());executed=epoch+1
        item={'epoch':executed,'rng_before':before,'rng_after_train':after,'rng_after_eval':after_eval};assert item==reference[executed]
        improved=metrics['acc']>best_acc
        if improved:best_acc=float(metrics['acc']);chosen=executed;wait=0
        else:wait+=1
        row={'epoch':executed,'acc':float(metrics['acc']),'best_acc':best_acc,'best_epoch':chosen,'improved':improved,'wait':wait,
            'losses':losses,'gradients':grad,'cf_gradient_norm':cfgnorm,**teacher_stats,**item}
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



def append_handoff(status):
    path=Path('/model/spromptgl/experiment/handoff.md');marker='## Experiment 82.2 — PSCE（COMPLETE_AUDITED）'
    ag=status['aggregates'];summary=[x for x in ag['summary'] if x['scope']=='all10'];con=ag['conclusion']
    lines=['',marker,'',f'执行目录：`{r.HERE}`；20/20正式训练、20/20 Best独立审计；唯一基座E74.4-G2 DARG，逐折E61 epoch10完整warmup，seed=0、871人、原固定10折。',
        '', '已核查SPromptGL原论文/原作者及用户仓库与SSC论文/公开代码，commit和SHA存manifest。实际PLNNoPEN未执行原论文PEN，本次不恢复。G1同几何匿名固定Query，G2原八医学Prompt固定Query；仅q0不同。Original O基精确相同、原谱统计/覆盖/融合和8%限幅保持；新增合法训练无Dropout教师的16方向受限反事实KL。',
        '', '|组|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:lines.append('|'+row['group']+'|'+'|'.join(f"{row[k+'_mean']:.4f} ± {row[k+'_population_sd']:.4f}" for k in r.METRICS)+'|')
    lines+=['','均值±population SD，单位%；完整10折、排除Fold2的9折、六指标配对逐折差及W/T/L见正式报告/CSV。']
    for name in ('G2 - B0','G2 - E82-G2','G2 - G1','G1 - B0'):
        vals=[next(x['mean_delta_pp'] for x in ag['paired'] if x['comparison']==name and x['scope']=='all10' and x['metric']==k) for k in r.METRICS[:3]]
        lines+=['',f'{name} ACC/BA/Prob-AUC差（pp）：'+ '/'.join(f'{v:+.4f}' for v in vals)+'。']
    g2=[a['mechanism'] for a in status['audits'] if a['group']=='G2']
    lines+=['',f"G2训练有效教师样本比例均值{100*np.mean([m['teacher_valid_fraction_training_mean'] for m in g2]):.3f}%；Best合法训练target Spearman：医学{con['medical_target_spearman']:.6f}、同Best匿名{con['anonymous_same_best_target_spearman']:.6f}、原统计{con['spectral_target_spearman']:.6f}。医学优于匿名/统计分别{con['medical_alignment_better_than_anonymous_folds']}/10、{con['medical_alignment_better_than_spectral_folds']}/10折。Prompt平均两两选择cos={np.mean([m['prompt_pairwise_cosine'] for m in g2]):.8f}。"]
    for x in ag['interventions']:lines+=['',f"只读{x['intervention']}：平均/最大绝对概率变化{x['mean_abs_probability_change']:.9f}/{x['max_abs_probability_change']:.9f}；硬翻转{x['flips']}、纠错{x['corrected']}、误伤{x['harmed']}。"]
    ct=con['corrections']['G2'];lines+=['',f"G2对B0纠错{ct['corrected']}、误伤{ct['harmed']}、净纠错{ct['net']}。共同历史RNG三元组{sum(a['historical_common_rng_epochs'] for a in status['audits'])}条严格一致；20/20 Latest参数/四优化器/RNG及重复下一训练+评估恢复精确一致，原模型/数据/历史SHA未变。",'',
        'OSSFS首选目录创建errno 5，正式训练前切换/model；运行异常详见runtime/execution.json。保留历史test=val选模、CV前标准化/RFE、transductive图及Fold2训练重叠；新增路径排除377/407。教师反映当前分类器依赖，不是医学真值或因果效应；可训练Key/Token仍可能吸收旋转。', '',f"科研判断：**{con['verdict']}**。G2−E82同时包含公共文本投影变化，不能全归因于CF。"]
    if con['verdict']=='NO-GO':lines+=['','正式停止PSSEC/PCDE/PSCE第二创新模块路线，不追加E82.2.1参数调优，不自行启动新实验。']
    addition='\n'.join(lines)+'\n'
    with path.open('r+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX);original=f.read()
        if marker in original:
            assert addition in original;return {'path':str(path),'append_verified':True,'already_present':True,'sha256':r.sha_file(path)}
        f.seek(0,2);f.write(addition);f.flush();os.fsync(f.fileno());f.seek(0);readback=f.read()
        assert readback==original+addition and marker in readback
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    return {'path':str(path),'append_verified':True,'prefix_unchanged':True,'appended_characters':len(addition),'sha256':r.sha_file(path)}


def main():
    started=time.time();manifest=json.loads((r.HERE/'manifest.json').read_text());audits=[]
    assert sum(r.task_complete(g,f) for g in r.GROUPS for f in range(1,11))==20
    assert r.implementation_sha()==manifest['preflight']['source_sha']
    assert r.sha_file(Path(__file__))==manifest['source_maps']['implementation'][str(Path(__file__))]
    text_check=text_geometry_audit();protected=manifest['source_maps']['protected']
    assert all(r.sha_file(p)==h for p,h in protected.items()),'historical protected source changed'
    for fold in range(1,11):
        for group in r.GROUPS:
            valid,invalid,missing=raw_logs(group,fold)
            if missing or invalid:recover_log(group,fold)
            audits.append(audit_one(group,fold))
            r.atomic_json(r.RESULTS/'audit_status.json',{'status':'AUDITING','completed':len(audits),'expected':20,'audits':audits})
    for fold in range(1,11):
        selected=[a for a in audits if a['fold']==fold];assert len({a['new_parameter_sha'] for a in selected})==1
        traces=[{t['epoch']:t for t in a['rng_trace']} for a in selected]
        for ep in set(traces[0])&set(traces[1]):assert traces[0][ep]==traces[1][ep]
    assert all(r.sha_file(p)==h for p,h in protected.items())
    aggregates=aggregate(audits)
    status={'status':'COMPLETE_AUDITED','completed_tasks':20,'independent_audits':20,'audits':audits,
        'text_geometry_independent_audit':text_check,'protected_sources_unchanged':True,'same_fold_init_and_common_rng_pass':True,
        'wall_seconds':time.time()-started,'audit_source_sha':r.sha_file(Path(__file__)),'source_sha':r.implementation_sha(),'aggregates':aggregates}
    status['handoff']=append_handoff(status)
    r.atomic_json(r.RESULTS/'audit_status.json',status)
    manifest.update(status='COMPLETE_AUDITED',completed_tasks=20,independent_audits=20,audit_wall_seconds=status['wall_seconds'],
        audited_implementation_sha=r.implementation_sha(),audit_source_sha=r.sha_file(Path(__file__)),scientific_conclusion=aggregates['conclusion'],handoff=status['handoff'])
    r.atomic_json(r.HERE/'manifest.json',manifest)
    with (r.RESULTS/'report.md').open('a') as f:f.write('\nHandoff已追加至`/model/spromptgl/experiment/handoff.md`并读回验证，原前缀保持不变。\n')
    print('COMPLETE_AUDITED 20/20; handoff appended and readback verified',flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='recover':recover_log(sys.argv[2],int(sys.argv[3]))
    else:main()

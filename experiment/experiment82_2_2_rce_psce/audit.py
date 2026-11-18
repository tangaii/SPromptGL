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

if 'rce_runner' in sys.modules:r=sys.modules['rce_runner']
else:
    import run as r

from rce_psce import EPS, GROUPS as GROUP_SPECS, teacher_snapshot, teacher_logits, counterfactual, auxiliary

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

def verified_logs(group,fold):
    rows=r.read_logs(r.paths(group,fold)[3]);logs={v['epoch']:v for v in rows}
    assert len(rows)==len(logs)
    return logs

def append_handoff(status):
    path=Path('/model/spromptgl/experiment/handoff.md');marker='## Experiment 82.2.2 — RCE-PSCE（COMPLETE_AUDITED）'
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


@torch.no_grad()
def independent_cf(state,module,z,expected=None):
    idx=torch.as_tensor(r.legal_train(state),device=z.device);zz=z.detach()[idx];y=state['y'][idx]
    components=((zz-module.mu)@module.U)[:,:,None]*module.U.T[None]
    d8=independent_limit(components,zz[:,None,:]);d4=d8*.5
    inputs=torch.stack([zz]+[zz-d4[:,j] for j in range(16)]+[zz-d8[:,j] for j in range(16)],dim=1)
    logits=independent_head(state['cam'].Outputlayer,inputs.reshape(-1,144)).reshape(-1,33,2)
    # Binary margins independently use the label sign rather than gather.
    margins=(logits[:,:,1]-logits[:,:,0])*(2*y[:,None]-1)
    D4=margins[:,0,None]-margins[:,1:17];D8=margins[:,0,None]-margins[:,17:]
    pos=(D8>0)&(D4>0);rho=torch.zeros_like(D8)
    rho[pos]=(2*torch.minimum(D8[pos],2*D4[pos]))/(D8[pos]+2*D4[pos]+1e-8)
    scores=torch.maximum(D8,torch.zeros_like(D8))*rho
    correct=logits[:,0].argmax(1)==y;valid=correct&(margins[:,0]>0)&(scores.max(1).values>1e-6)
    target=scores/(scores.sum(1,keepdim=True)+1e-8)
    cf=dict(D8=D8,D4=D4,rho=rho,target=target,score=scores,valid=valid,correct=correct,margin=margins[:,0],d8=d8,d4=d4,inputs=inputs)
    assert torch.equal(d4,.5*d8) and ((rho>=0)&(rho<=1)).all()
    assert float((d8.norm(dim=-1)/zz.norm(dim=-1)[:,None].clamp_min(EPS)).max())<=.080001
    assert float((d4.norm(dim=-1)/zz.norm(dim=-1)[:,None].clamp_min(EPS)).max())<=.040001
    assert not np.isin(r.legal_train(state),state['test_np']).any()
    if expected is not None:
        assert torch.equal(valid,expected['valid']) and torch.equal(correct,expected['correct'])
        errors={k:float((cf[k]-expected[k]).abs().max()) for k in ('D8','D4','rho','target','d8','d4','inputs')}
        assert errors['D8']<8e-6 and errors['D4']<8e-6 and errors['target']<2e-4 and errors['rho']<2e-3,errors
        cf['independent_errors']=errors
    return cf


def formulas(module,d):
    with torch.no_grad():
        pars=dict(module.named_parameters())
        def linear(v,n):return F.linear(v,pars[n+'.weight'],pars.get(n+'.bias'))
        h=(d['z']-module.mu)@module.U;hh=h/torch.sqrt(module.lam+EPS)
        ll=module.loglam.expand_as(h);nu=module.nu.expand_as(h)
        wp=linear(F.gelu(linear(torch.stack([hh.abs(),ll,nu],-1),'spectral.0')),'spectral.2').squeeze(-1).softmax(-1)
        token=linear(module.U.T,'direction')[None]+linear(torch.stack([hh,hh.abs(),ll,nu],-1),'evidence')
        token=F.layer_norm(token,(32,),pars['norm.weight'],pars['norm.bias'],module.norm.eps)
        scores=torch.einsum('kd,njd->nkj',module.q0,linear(token,'key'))/math.sqrt(32)
        null=(module.q0@pars['null']/math.sqrt(32))[None,:,None].expand(len(h),-1,-1)
        a=torch.cat((scores,null),-1).div(.5).softmax(-1)
        cov=1-(1-a[:,:,:16]).prod(1);ws=cov/(cov.sum(-1,keepdim=True)+EPS)
        entropy=-(ws*(ws+EPS).log()).sum(-1)/math.log(16)
        beta=((1-a[:,:,16].mean(1))*(1-entropy).clamp(.2,.8)).detach()
        w=beta[:,None]*ws+(1-beta[:,None])*wp
        raw=((w-1/16)*h)@module.U.T;delta=independent_limit(raw,d['z'])
        if module.fold==2:delta[[376,406]]=0
        for k,v in [('h',h),('hhat',hh),('tokens',token),('w_sp',wp),('a',a),('w_sem',ws),('beta',beta),('w',w),('delta_raw',raw),('delta',delta)]:
            assert torch.allclose(v,d[k],atol=1e-7,rtol=2e-6),k
        assert not d['beta'].requires_grad
        assert float((delta.norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)).max())<=.080001
    return True


@torch.no_grad()
def scale_audit(state,c):
    # Independent functional classifier and NumPy midpoint median, only warmup legal rows.
    text=r.torch_load(r.RT/'prompts.pt');module,_,_=r.make_module(state,'G2',c,text)
    cf=independent_cf(state,module,c['warmup_cam'].to(state['x'].device))
    vals=cf['D8'][cf['correct']].abs().cpu().double().numpy();vals=vals[vals>1e-6]
    median=float(np.median(vals)) if len(vals) else 0.;scale=max(median,.001)
    assert abs(scale-c['s_fold'])<1e-7,(scale,c['s_fold'])
    assert len(vals)==c['scale_details']['nonzero_count'] and cf['valid'].any()
    return dict(s_fold=scale,nonzero_count=len(vals),independent_numpy_median=median,legal_only=True)


def independent_aux(state,module,z):
    idx=torch.as_tensor(r.legal_train(state),device=z.device);y=state['y'][idx]
    head=state['cam'].Outputlayer;before=r.sha(r.cpu_state(head));rb=r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    teacher=teacher_snapshot(head);a=auxiliary(module,teacher,z,idx,y)
    cf=independent_cf(state,module,z,a['cf']);valid=cf['valid'];p=a['aux_details']['w_sem'][idx][valid];t=cf['target'][valid]
    kl=(torch.where(t>0,t*(t.clamp_min(1e-30).log()-p.clamp_min(EPS).log()),torch.zeros_like(t))).sum(-1).mean() if valid.any() else p.sum()*0
    lp=independent_head(head,z.detach()[idx]+a['aux_details']['delta'][idx])
    margin=(lp[:,1]-lp[:,0])*(2*y-1);gain=margin-cf['margin']
    maskmeans=[(-gain[valid&(y==lab)]).clamp_min(0).mean()/module.s_fold for lab in (0,1) if (valid&(y==lab)).any()]
    harm=torch.stack(maskmeans).mean() if maskmeans else gain.sum()*0
    assert torch.allclose(kl,a['cf_loss'],atol=2e-5,rtol=1e-5) and torch.allclose(harm,a['harm'],atol=2e-5,rtol=1e-5)
    assert torch.allclose(gain,a['gain'],atol=8e-6,rtol=1e-5)
    fake=dict(cf_loss=a['cf_loss'],harm=a['harm'])
    norms=r.auxiliary_gradient_audit(fake,state,module)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert before==r.sha(r.cpu_state(head)) and rb==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    assert not z.detach().requires_grad and all(p.grad is None for p in teacher.parameters())
    a['independent_cf']=cf
    return a,dict(cf_loss=float(kl.detach()),harm=float(harm.detach()),gradient_norms=norms,teacher_frozen=True,no_shared_rng=True,formula_errors=cf['independent_errors'])


def rank_similarity(x,y):
    from scipy.stats import rankdata
    a=rankdata(x.detach().cpu().double().numpy(),axis=-1);b=rankdata(y.detach().cpu().double().numpy(),axis=-1)
    a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True)
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1);valid=den>0
    vals=(a[valid]*b[valid]).sum(-1)/den[valid]
    return float(vals.mean()) if len(vals) else None


def reliability_stats(cf,y):
    v=cf['valid'];D8=cf['D8'];D4=cf['D4']
    row={'legal_count':len(y),'valid_count':int(v.sum()),'valid_fraction':float(v.float().mean()),
         'teacher_correct_count':int(cf['correct'].sum()),'teacher_error_count':int((~cf['correct']).sum()),
         'teacher_error_supervised_count':int((v&~cf['correct']).sum()),
         'sign_agreement':float((torch.sign(D8)==torch.sign(D4)).float().mean()),
         'scale_rank_spearman':rank_similarity(D8,D4),
         'stable_positive_fraction':float(((D8>0)&(D4>0)).float().mean())}
    add_distribution(row,'rho',cf['rho']);row['rho_mean_by_direction']=json.dumps(cf['rho'].mean(0).cpu().tolist())
    row['stable_positive_by_direction']=json.dumps(((D8>0)&(D4>0)).float().mean(0).cpu().tolist())
    for lab,n in [(0,'asd'),(1,'td')]:
        mask=y==lab;row[n+'_valid_fraction']=float(v[mask].float().mean());row[n+'_teacher_error_rate']=float((~cf['correct'][mask]).float().mean())
    return row


def gain_stats(gain,y,cf):
    g=gain.detach();row={'gain_mean':float(g.mean()),'gain_positive_fraction':float((g>1e-6).float().mean()),'gain_negative_fraction':float((g< -1e-6).float().mean())}
    for name,mask in [('correct',cf['correct']),('error',~cf['correct']),('valid',cf['valid']),('asd',y==0),('td',y==1),('error_asd',(~cf['correct'])&(y==0)),('error_td',(~cf['correct'])&(y==1))]:
        row[name+'_gain_n']=int(mask.sum());row[name+'_gain_mean']=float(g[mask].mean()) if mask.any() else None
        row[name+'_gain_positive_fraction']=float((g[mask]>1e-6).float().mean()) if mask.any() else None
        row[name+'_gain_negative_fraction']=float((g[mask]< -1e-6).float().mean()) if mask.any() else None
    return row


def historical_diagnostic(state,c,text):
    rng=r.base_runtime().pack_rng();saved=r.runtime().load_history().state_bundle(state)
    modes=[state[k].training for k in ('model','cam')]
    try:
        ck=r.torch_load(r.E822/'runtime/checkpoints'/f"G2_f{state['fold']:02d}.best.pt")
        r.runtime().load_history().load_bundle(state,ck['current']['base'])
        m,_,_=r.make_module(state,'G2',c,text)
        missing=m.load_state_dict(ck['current']['psce'],strict=False)
        assert missing.missing_keys==['s_fold'] and not missing.unexpected_keys
        out,d,_=replay(state,m)
        idx=torch.as_tensor(r.legal_train(state),device=m.U.device);y=state['y'][idx]
        cf=independent_cf(state,m,d['z'])
        lp=independent_head(state['cam'].Outputlayer,d['z'][idx]+d['delta'][idx]);gain=(lp[:,1]-lp[:,0])*(2*y-1)-cf['margin']
        row=dict(fold=state['fold'],source='read-only historical E82.2 G2 Best',**reliability_stats(cf,y),**gain_stats(gain,y,cf))
        row['residual_relative_mean']=float((d['delta'][idx].norm(dim=-1)/d['z'][idx].norm(dim=-1)).mean())
        row['residual_relative_test_mean']=float((d['delta'][state['test']].norm(dim=-1)/d['z'][state['test']].norm(dim=-1)).mean())
    finally:
        r.runtime().load_history().load_bundle(state,saved)
        for k,mode in zip(('model','cam'),modes):state[k].train(mode)
    assert r.base_runtime().rng_sha(r.base_runtime().pack_rng())==r.base_runtime().rng_sha(rng)
    return row


def mechanism(state,module,c,d,cf,gain,logs):
    idx=torch.as_tensor(r.legal_train(state),device=module.U.device);test=state['test'];y=state['y'][idx]
    row=dict(group=module.group,fold=module.fold,s_fold=float(module.s_fold),**module.counts(),**reliability_stats(cf,y),**gain_stats(gain,y,cf))
    for prefix,arr in [('residual_relative',d['delta'].norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)),('residual_relative_test',d['delta'][test].norm(dim=-1)/d['z'][test].norm(dim=-1).clamp_min(EPS)),('beta',d['beta'][test]),('null',d['null'][test]),('entropy',d['entropy'][test])]:add_distribution(row,prefix,arr)
    row['delta_norm_mean']=float(d['delta'].norm(dim=-1).mean());row['raw_norm_mean']=float(d['delta_raw'].norm(dim=-1).mean())
    row['residual_near_cap_fraction']=float((d['delta'].norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)>=.0792).float().mean())
    row['residual_near_zero_fraction']=float((d['delta'].norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)<1e-5).float().mean())
    original_score=cf['D8'].relu();oldvalid=cf['correct']&(cf['margin']>0)&(cf['D8'].max(-1).values>1e-6)
    oldcf=dict(target=original_score/(original_score.sum(-1,keepdim=True)+EPS),valid=oldvalid)
    row['original_target_l1_to_reliable']=float((oldcf['target'][cf['valid']]-cf['target'][cf['valid']]).abs().sum(-1).mean())
    for name,w in [('semantic',d['w_sem'][idx]),('spectral',d['w_sp'][idx])]:
        for target_name,target in [('reliable',cf),('single_scale',oldcf)]:row[name+'_'+target_name+'_alignment']=json.dumps(alignment(w,target))
    pa=d['a'][test,:,:16];pn=F.normalize(pa,dim=-1);cos=pn@pn.transpose(1,2);mask=~torch.eye(8,dtype=torch.bool,device=idx.device)
    row['prompt_pairwise_cosine']=float(cos[:,mask].mean());row['prompt_pairwise_matrix']=json.dumps(cos.mean(0).cpu().tolist())
    row['prompt_direction_distributions']=json.dumps(pa.mean(0).cpu().tolist());row['prompt_null_means']=json.dumps(d['a'][test,:,16].mean(0).cpu().tolist())
    row['prompt_reliable_alignments']=json.dumps([alignment(d['a'][idx,k,:16]/d['a'][idx,k,:16].sum(-1,keepdim=True).clamp_min(EPS),cf) for k in range(8)])
    row['semantic_spectral_cosine']=float(F.cosine_similarity(d['w_sem'][test],d['w_sp'][test],dim=-1).mean())
    row['semantic_spectral_l1']=float((d['w_sem'][test]-d['w_sp'][test]).abs().sum(-1).mean())
    row['semantic_mean_weights']=json.dumps(d['w_sem'][test].mean(0).cpu().tolist());row['spectral_mean_weights']=json.dumps(d['w_sp'][test].mean(0).cpu().tolist())
    with torch.no_grad():
        initial,_,_=r.make_module(state,module.group,c);_,di=initial(d['z'])
        row['initial_selector_current_teacher_alignment']=json.dumps(alignment(di['w_sem'][idx],cf))
        row['initial_selector_current_cam_l1']=float((di['w_sem'][test]-d['w_sem'][test]).abs().sum(-1).mean())
        oldck=r.torch_load(r.E822/'runtime/checkpoints'/f"{module.group}_f{module.fold:02d}.best.pt")
        original,_,_=r.make_module(state,module.group,c);missing=original.load_state_dict(oldck['current']['psce'],strict=False);assert missing.missing_keys==['s_fold']
        _,od=original(d['z'])
        row['original_PSCE_same_CAM_delta_l2']=float((od['delta'][test]-d['delta'][test]).norm(dim=-1).mean())
        row['original_PSCE_same_CAM_residual_relative']=float((od['delta'][test].norm(dim=-1)/d['z'][test].norm(dim=-1)).mean())
        row['original_PSCE_same_CAM_reliable_alignment']=json.dumps(alignment(od['w_sem'][idx],cf))
    for key in ('cf_loss','harm'):
        row[key+'_first']=logs[min(logs)]['losses'][key];row[key+'_last']=logs[max(logs)]['losses'][key]
        row[key+'_best_training']=logs[max(e for e in logs if logs[e]['improved'])]['losses'][key]
    for name in ('cls','cf_loss','harm'):
        key=name+'_psce_gradient_norm';vals=[v[key] for v in logs.values()]
        row[key+'_mean']=float(np.mean(vals));row[key+'_min']=float(min(vals));row[key+'_max']=float(max(vals));row[key+'_nonzero_fraction']=float(np.mean(np.asarray(vals)>0))
    row['valid_fraction_training_mean']=float(np.mean([v['teacher_valid_fraction'] for v in logs.values()]))
    hist=json.loads((r.RT/'historical_readonly_diagnostic.json').read_text())[module.fold-1]
    row['historical_E822_own_CAM_residual_relative_test_mean']=hist['residual_relative_test_mean']
    row['ratio_residual_to_E822_own_CAM']=row['residual_relative_test_mean']/hist['residual_relative_test_mean']
    row['historical_E822_own_CAM_gain_positive_fraction']=hist['gain_positive_fraction']
    if module.fold==2:assert torch.count_nonzero(d['delta'][[376,406]])==0
    return row


def check_training_losses(out,state):
    idx=torch.as_tensor(r.legal_train(state),device=state['x'].device);d=out['details']
    independent_res=(torch.linalg.vector_norm(d['delta'][idx],dim=1)**2/(torch.linalg.vector_norm(d['z'][idx],dim=1)**2+EPS)).mean()
    assert torch.allclose(independent_res,out['res'],atol=1e-9,rtol=1e-6)
    assert torch.equal(out['total'],out['darg_total']+.001*out['res']+.01*out['cf_loss']+.01*out['harm'])
    return float(independent_res.detach())


def audit_one(group,fold):
    assert r.task_complete(group,fold)
    tp,bp,lp,log=r.paths(group,fold);task=json.loads(tp.read_text());best=r.torch_load(bp);latest=r.torch_load(lp)
    state,warmup,identity=r.prepare(fold);c=r.torch_load(r.RT/f'spectrum_f{fold:02d}.pt');text=r.torch_load(r.RT/'prompts.pt')
    module,opt,init=r.make_module(state,group,c,text);spectral=spectral_audit(c,state);scale=scale_audit(state,c)
    initcheck=r.validate_initialization(state,group,module,c,text)
    old=r.torch_load(r.E822/'runtime'/f'spectrum_f{fold:02d}.pt')
    assert torch.equal(c['spectrum']['U'],old['spectrum']['U']) and torch.equal(c['spectrum']['mu'],old['spectrum']['mu'])
    for ck in (best,latest):
        assert ck['identity']==identity and ck['config']==r.CFG and ck['source_sha']==r.implementation_sha()
        assert ck['new_parameter_sha']==init and ck['spectra_sha']==c['spectra_sha'] and ck['text_identity_sha']==text['identity_sha']
    assert task['best_sha']==r.sha_file(bp) and task['latest_sha']==r.sha_file(lp)
    r.load_bundle(state,module,opt,best['current']);r.base_runtime().restore_rng(best['rng'])
    assert r.sha(r.bundle(state,module,opt))==best['bundle_sha']==r.sha(best['current'])
    assert torch.equal(module.q0.cpu(),text['q_'+GROUP_SPECS[group]]) and torch.equal(module.U.cpu(),c['spectrum']['U'].float())
    assert float(module.s_fold)==float(torch.tensor(c['s_fold']))
    assert best['wait']==0 and best['next_epoch']==best['best_epoch'] and best['best_acc']==best['metrics']['acc']
    before=r.base_runtime().rng_sha(r.base_runtime().pack_rng());out,d,calls=replay(state,module,with_grad=True)
    values,prob,pred,y=independently_recompute(out['logp'],state);formulas(module,d)
    assert before==best['rng_sha']==r.base_runtime().rng_sha(r.base_runtime().pack_rng())
    assert np.array_equal(pred,best['prediction']) and np.array_equal(prob,best['probability'])
    assert all(abs(values[k]-best['metrics'][k])<1e-12 and abs(values[k]-task['metrics'][k])<1e-12 for k in r.METRICS)
    logs=verified_logs(group,fold);assert sorted(logs)==list(range(11,task['epochs_executed']+1))
    chosen=max(logs,key=lambda e:(logs[e]['acc'],-e));assert chosen==best['best_epoch']==task['best_epoch']
    reference={v['epoch']:v for v in r.historical_task(fold)['rng_trace']};e822=r.e822_trace(group,fold);parity=0;parity822=0
    for v in task['rng_trace']:
        assert all(logs[v['epoch']][k]==v[k] for k in ('rng_before','rng_after_train','rng_after_eval'))
        if v['epoch'] in reference:assert v==reference[v['epoch']];parity+=1
        if v['epoch'] in e822:assert v==e822[v['epoch']];parity822+=1
    assert len(task['rng_trace'])==task['epochs_executed']-10
    aux,auxcheck=independent_aux(state,module,d['z']);cf=aux['cf']
    legal=torch.as_tensor(r.legal_train(state),device=module.U.device)
    with torch.no_grad():row=mechanism(state,module,c,d,cf,aux['gain'],logs)
    cg=r.classification_gradient_audit(out,state,module)
    row.update(best_eval_cf_loss=auxcheck['cf_loss'],best_eval_harm_loss=auxcheck['harm'])
    optimizer_checks={}
    for name,optimizer in [(k,state[k]) for k in ('opt_base','opt_cam','opt_added')]+[('opt_psce',opt)]:
        sd=optimizer.state_dict();assert all(torch.isfinite(v).all() for st in sd['state'].values() for v in st.values() if torch.is_tensor(v))
        assert len(sd['state'])>0
        optimizer_checks[name]=dict(state_sha=r.sha(sd),state_count=len(sd['state']),lr=sd['param_groups'][0]['lr'],weight_decay=sd['param_groups'][0]['weight_decay'])
    row.update(four_optimizer_audit=json.dumps(optimizer_checks),classification_gradient_norms=json.dumps(cg),best_epoch=best['best_epoch'],termination_epoch=task['epochs_executed'])
    refs={};refpred={}
    for name,path,key in [('B0',r.runtime().EXP/'.runtime/checkpoints'/f'{r.BASE_GROUP}_f{fold:02d}.best.pt','test_probability'),('E82-G2',r.E82/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt','probability'),('E82.2-G2',r.E822/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt','probability'),('E82.2.1-G2',r.E8221/'runtime/checkpoints'/f'G2_f{fold:02d}.best.pt','probability')]:
        ck=r.torch_load(path);assert ck['identity']==identity
        pr=np.asarray(ck[key]);m=reference_metrics(pr,y)
        assert all(abs(m[k]-ck['metrics'][k])<1e-12 for k in r.METRICS)
        refs[name]=m;refpred[name]=pr.argmax(1).tolist()
    for name in refs:
        row['vs_'+name]=json.dumps(flips(np.array(refpred[name]),pred,y))
        for label,sub in [(0,'asd'),(1,'td')]:row['vs_'+name+'_'+sub]=json.dumps(flips(np.array(refpred[name])[y==label],pred[y==label],y[y==label]))
    interventions=[]
    if group=='G2':
        qsave=module.q0.clone();psha=r.sha(r.cpu_state(module))
        for name in ('anonymous','semantic_off','residual_off'):
            with torch.no_grad():module.q0.copy_(text['q_anonymous'].to(module.U.device) if name=='anonymous' else qsave)
            changed,cd,_=replay(state,module,'full' if name=='anonymous' else 'noop' if name=='residual_off' else name)
            met,cp,cr,_=independently_recompute(changed['logp'],state)
            response={'intervention':name,'mean_abs_probability_change':float(abs(cp[:,0]-prob[:,0]).mean()),'max_abs_probability_change':float(abs(cp[:,0]-prob[:,0]).max()),**flips(pred,cr,y),'metrics':met}
            for lab,sub in [(0,'asd'),(1,'td')]:response[sub]=flips(pred[y==lab],cr[y==lab],y[y==lab])
            response['semantic_l1_change']=float((cd['w_sem'][legal]-d['w_sem'][legal]).abs().sum(-1).mean())
            if name=='semantic_off':
                with torch.no_grad():
                    _,sd=module(d['z'],'semantic_off');module.q0.copy_(text['q_anonymous'].to(module.U.device));_,sa=module(d['z'],'semantic_off')
                    assert torch.equal(sd['delta'],sa['delta'])
            interventions.append(response);del changed,cd
        with torch.no_grad():module.q0.copy_(qsave)
        assert r.sha(r.cpu_state(module))==psha
        g1ck=r.torch_load(r.paths('G1',fold)[1]);g1,_,_=r.make_module(state,'G1',c,text);g1.load_state_dict(g1ck['current']['psce']);g1.eval()
        with torch.no_grad():_,gd=g1(d['z'])
        row['trained_G1_same_G2_CAM_alignment']=json.dumps(alignment(gd['w_sem'][legal],cf))
        row['trained_G1_G2_same_CAM_semantic_L1']=float((gd['w_sem'][legal]-d['w_sem'][legal]).abs().sum(-1).mean())
        row['trained_G1_G2_same_CAM_delta_L2']=float((gd['delta'][legal]-d['delta'][legal]).norm(dim=-1).mean())
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
    assert latest['next_epoch']==task['epochs_executed'] and latest['wait']==latest['next_epoch']-latest['best_epoch']
    assert latest['wait']>200 or latest['next_epoch']==450
    r.zero(state,opt);o=r.forward(state,module,True);check_training_losses(o,state);o['total'].backward();r.step(state,opt);del o
    with torch.no_grad():ev=r.forward(state,module,False)
    a=(r.sha(r.bundle(state,module,opt)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),ev['logp'].cpu().clone())
    # Fresh model objects and four optimizer states loaded only from on-disk Latest.
    state2,_,_=r.prepare(fold);module2,opt2,_=r.make_module(state2,group,c,text)
    disk=r.torch_load(lp);r.load_bundle(state2,module2,opt2,disk['current']);r.base_runtime().restore_rng(disk['rng'])
    r.zero(state2,opt2);o=r.forward(state2,module2,True);check_training_losses(o,state2);o['total'].backward();r.step(state2,opt2);del o
    with torch.no_grad():ev=r.forward(state2,module2,False)
    b=(r.sha(r.bundle(state2,module2,opt2)),r.base_runtime().rng_sha(r.base_runtime().pack_rng()),ev['logp'].cpu())
    assert a[:2]==b[:2] and torch.equal(a[2],b[2])
    result={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','metrics':values,'prediction':pred.tolist(),'probability':prob.tolist(),'labels':y.tolist(),
        'reference_metrics':refs,'reference_predictions':refpred,'best_epoch':best['best_epoch'],'termination_epoch':task['epochs_executed'],
        'best_sha':task['best_sha'],'latest_sha':task['latest_sha'],'new_parameter_sha':init,'identity':identity,'spectral_audit':spectral,'scale_audit':scale,'initialization_audit':initcheck,
        'auxiliary_independent_audit':auxcheck,'historical_common_rng_epochs':parity,'e822_common_rng_epochs':parity822,'latest_restore':'four_optimizers_fresh_object_next_train_eval_parameters_probability_rng_exact',
        'mechanism':row,'interventions':interventions,'rng_trace':task['rng_trace']}
    del state,state2,module,module2,opt,opt2,best,latest,ev;torch.cuda.empty_cache();print(f'AUDIT PASS {group}/{fold}',flush=True)
    return result


def aggregate(audits):
    by={(a['group'],a['fold']):a for a in audits};foldrows=[];mechs=[]
    allgroups=(*r.GROUPS,'B0','E82-G2','E82.2-G2','E82.2.1-G2')
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
    comparisons=[('G2','B0'),('G2','E82-G2'),('G2','E82.2-G2'),('G2','E82.2.1-G2'),('G2','G1'),('G1','B0')]
    def predictions(g,f):return np.array(by[(g,f)]['prediction'] if g in r.GROUPS else by[('G2',f)]['reference_predictions'][g])
    for scope,folds in [('all10',list(range(1,11))),('without_fold2',[f for f in range(1,11) if f!=2])]:
        for g in allgroups:
            rows=[index[(g,f)] for f in folds];row={'group':g,'scope':scope,'n_folds':len(rows)}
            for k in r.METRICS:
                v=np.array([x[k] for x in rows]);row[k+'_mean']=float(v.mean());row[k+'_population_sd']=float(v.std(ddof=0))
            summary.append(row)
        for g,ref in comparisons:
            counts={}
            for sub in ('all','asd','td'):
                values=[]
                for f in folds:
                    y=np.array(by[(g,f)]['labels']);mask=np.ones(len(y),dtype=bool) if sub=='all' else y==(0 if sub=='asd' else 1)
                    values.append(flips(predictions(ref,f)[mask],predictions(g,f)[mask],y[mask]))
                counts.update({('' if sub=='all' else sub+'_')+n:sum(v[n] for v in values) for n in ('corrected','harmed','net','flips')})
            for k in r.METRICS:
                delta=np.array([index[(g,f)][k]-index[(ref,f)][k] for f in folds])
                paired.append({'comparison':g+' - '+ref,'scope':scope,'metric':k,'mean_delta_pp':float(delta.mean()),
                    'population_sd_delta_pp':float(delta.std(ddof=0)),'wins':int((delta>1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),'losses':int((delta< -1e-10).sum()),
                    **counts,**{'fold'+str(f)+'_delta_pp':float(v) for f,v in zip(folds,delta)}})
    r.write_csv(r.RESULTS/'fold_metrics.csv',foldrows);r.write_csv(r.RESULTS/'summary.csv',summary)
    r.write_csv(r.RESULTS/'paired_comparisons.csv',paired);r.write_csv(r.RESULTS/'mechanism.csv',mechs)
    def dm(g,ref,k,scope='all10'):return next(x for x in paired if x['comparison']==g+' - '+ref and x['scope']==scope and x['metric']==k)
    def mean(g,k):
        values=[m[k] for m in mechs if m['group']==g and m.get(k) is not None]
        return float(np.mean(values)) if values else None
    def alignmean(g,key,stat):
        vals=[json.loads(m[key])[stat] for m in mechs if m['group']==g]
        return float(np.mean([v for v in vals if v is not None]))
    interventions=[]
    for name in ('anonymous','semantic_off','residual_off'):
        a=[i for x in audits if x['group']=='G2' for i in x['interventions'] if i['intervention']==name]
        interventions.append({'intervention':name,'mean_abs_probability_change':float(np.mean([x['mean_abs_probability_change'] for x in a])),
            'max_abs_probability_change':max(x['max_abs_probability_change'] for x in a),**{k:sum(x[k] for x in a) for k in ('flips','corrected','harmed','net')},
            'metrics_mean_percent':{k:100*float(np.mean([x['metrics'][k] for x in a])) for k in r.METRICS},
            'by_class':{sub:{k:sum(x[sub][k] for x in a) for k in ('corrected','harmed','net','flips')} for sub in ('asd','td')}})
    corrections={g:{sub:{n:dm(g,'B0','acc')[('' if sub=='all' else sub+'_')+n] for n in ('corrected','harmed','net','flips')} for sub in ('all','asd','td')} for g in r.GROUPS}
    from scipy.stats import spearmanr
    association={}
    for g in r.GROUPS:
        rows=[m for m in mechs if m['group']==g]
        xs=[json.loads(m['semantic_reliable_alignment'])['spearman'] for m in rows]
        ys=[by[(g,m['fold'])]['metrics']['acc']-by[(g,m['fold'])]['reference_metrics']['B0']['acc'] for m in rows]
        association[g]=float(spearmanr(xs,ys).statistic) if np.std(xs)>0 and np.std(ys)>0 else None
    dacc=dm('G2','B0','acc');dtext=dm('G2','G1','acc');anon=interventions[0];res=interventions[2]
    checks={'ACC_stable':dacc['mean_delta_pp']>0 and dacc['wins']>dacc['losses'],
        'BA_not_lower':dm('G2','B0','hard_auc')['mean_delta_pp']>=0,
        'Prob_AUC_not_materially_lower':dm('G2','B0','prob_auc')['mean_delta_pp']>=-.1,
        'ASD_SEN_not_materially_lower':dm('G2','B0','sen')['mean_delta_pp']>=-.5,
        'beats_E822':dm('G2','E82.2-G2','acc')['mean_delta_pp']>0,
        'medical_beats_anonymous':dtext['mean_delta_pp']>0 and dtext['wins']>dtext['losses'],
        'meaningful_immediate_text_effect':anon['mean_abs_probability_change']>=.001 and anon['net']<0,
        'immediate_residual_diagnostic_benefit':res['net']<0,
        'target_alignment_improved':alignmean('G2','semantic_reliable_alignment','spearman')>alignmean('G2','initial_selector_current_teacher_alignment','spearman'),
        'no_zero_residual_collapse':mean('G2','ratio_residual_to_E822_own_CAM')>.1 and mean('G2','residual_relative_test_mean')>1e-5}
    verdict='GO' if all(checks.values()) else 'PARTIAL' if dacc['mean_delta_pp']>0 and checks['immediate_residual_diagnostic_benefit'] and checks['no_zero_residual_collapse'] else 'NO-GO'
    con=dict(verdict=verdict,frozen_checks=checks,corrections=corrections,interventions=interventions,
        alignment_vs_test_delta_spearman=association,
        g2_vs_references={ref:{k:dm('G2',ref,k)['mean_delta_pp'] for k in r.METRICS} for ref in ('B0','E82-G2','E82.2-G2','E82.2.1-G2','G1')},
        route_action='STOP_E82_SPECTRAL_SEMANTIC_NO_E82.2.3_NO_SEARCH' if verdict=='NO-GO' else 'NO_ADDITIONAL_EXPERIMENTS_AUTHORIZED',
        mean_mechanisms={g:{k:mean(g,k) for k in ('sign_agreement','scale_rank_spearman','stable_positive_fraction','valid_fraction','asd_valid_fraction','td_valid_fraction','rho_mean','gain_positive_fraction','gain_negative_fraction','gain_mean','asd_gain_mean','td_gain_mean','residual_relative_test_mean','ratio_residual_to_E822_own_CAM','prompt_pairwise_cosine','beta_mean','null_mean','entropy_mean','semantic_spectral_l1','cf_loss_psce_gradient_norm_mean','harm_psce_gradient_norm_mean','cls_psce_gradient_norm_mean')} for g in r.GROUPS})
    protected=json.loads((r.HERE/'manifest.json').read_text())['source_maps']['protected']
    code={p:h for p,h in protected.items() if str(r.REPO) in p and '/experiment/' not in p and p.endswith('.py')}
    data={p:h for p,h in protected.items() if '/data/ABIDE/' in p}
    checkpoints={p:h for p,h in protected.items() if p.endswith('.pt')}
    con['protection']={'protected_files':len(protected),'all_map_sha':r.sha(protected),'root_code_files':len(code),'root_code_map_sha':r.sha(code),'data_files':len(data),'data_map_sha':r.sha(data),'historical_pt_files':len(checkpoints),'historical_pt_map_sha':r.sha(checkpoints)}
    report=['# Experiment 82.2.2 — RCE-PSCE','', '**Reliability-aware Counterfactual Evidence for Prompt-guided Spectral Learning / 可靠性约束的Prompt谱反事实证据学习**','',
        f'状态：COMPLETE_AUDITED。实际目录：`{r.HERE}`。20/20正式训练、20/20 Best逐元素重放、20/20 Latest四优化器精确恢复均通过。','',
        '## 论文与源码核查','',
        '[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)的GFR构建多关系患者图，GCN传播患者信息，CAM交互模态，SPGS/PEN用疾病Prompt选择局部信息，RES融合局部和全局。论文ABIDE包含871患者，10折ACC/AUC/SEN/SPE为92.30/92.22/91.08/93.36%。这些数值不作为历史DARG结果。','',
        '[原作者代码](https://github.com/wanxixi11/SPromptGL)固定commit ba3b9446fa451fde6864d6555bdf324de0c4b64a的network.py、model.py、layers.py、main.py已核对。原论文完整PEN路径和历史真实vendor PLNNoPEN不同：本实验保留InputLayer→Encoder/FeedForward→flatten144→OutputLayer，不加回PEN。原分类头LayerNorm144→Linear144,36→GELU→Dropout→Linear36,2→log_softmax。教师复制当前头并显式跳过Dropout，不改变原头状态。','',
        '[SSC论文](https://papers.miccai.org/miccai-2026/paper/4444_paper.pdf)与[代码](https://github.com/Hanyu1124/ssc)固定commit f87559c4deb3b624395cb9e9b4f83b68e3fb4beb只用于多Prompt互补证据与冗余问题的动机；公开预算round与论文floor有细节差异，本实现不依赖它。WSI patch选择不等同于脑影像CAM144潜在谱方向，不能据此声称解剖学语义对齐。','',
        '[E82.2提交](https://github.com/tangaii/SPromptGL/commit/9f104bda39cc9c4deaf90aa25cb09fd2c4a8d302)和[E82.2.1提交](https://github.com/tangaii/SPromptGL/commit/47a15e30b06cc5bf48924a7b3f814c3c43341d93)的方法、训练、审计、配置、报告远程字节与本地SHA相同。两篇PDF及所查远程文件共23项重新下载，版本、路径和SHA在manifest及runtime/source_review.json。','',
        'DARG通过真实E74.4-G2运行适配器加载E61 epoch10，不手工复刻损失。GraphLearn保留正边支持和历史候选规则，统一邻接加I并归一化一次，四GCN各调用一次；Positive-DCR只传播adjacency−original的正增量，按历史模态坐标/门控回写。原total保留cls、graph、pair、adv、preserve原权重。','',
        '## 预注册方法与初始化','',
        '本次回到E82.2完整原PSCE，不含E82.2.1三动作Policy。G1/G2只有固定q0不同，各6081可训练参数。初始化822000+fold、匿名几何种子821991；原8条RN50文本含Nromal历史拼写，不将前后4条指定为TD/ASD。原PSCE最小实现保存在本目录，正常方法不import旧psce.py。独立验证只读原初始化代码，在隔离RNG重建，与当前逐参数torch.equal及历史保存的初始参数SHA一致；未声称存在历史独立初始张量文件。','',
        '每折仅合法训练warmup CAM用CPU float64重建Original O：8 PCA+8正交类别协方差差异方向。U、mu与E82.2逐元素一致。保留3-32-1统计分支、16谱Token、8×17 Attention、Null、连续Coverage、detach beta及raw=((w−1/16)*h)Uᵀ，合并残差8%限幅。不拆CAM144为模态槽。','',
        '每训练Epoch在合法训练行构造33个冻结教师输入：原始、16个z−0.5d8、16个z−d8，d8为原8% smooth_limit。D为原真实类别Margin减遮蔽Margin。正贡献两尺度一致才有rho=2min(D8,2D4)/(D8+2D4+EPS)，target归一化ReLU(D8)*rho。可靠患者同时满足教师正确、Margin>0及max(score)>1e−6。教师错误患者仍参与原分类训练，额外CF和harm不监督他们。','',
        's_fold在warmup教师正确的合法训练患者abs(D8)>1e−6值上，以float64中点中位数并下限1e−3构建，整个正式训练固定。L_harm在同一可靠有效患者上对ReLU(−gain)/s_fold按存在的ASD/TD类别等权均值。两项辅助损失均另跑PSCE(z.detach())，只更新PSCE；主分类损失仍贯穿DARG/CAM/PSCE/原头。L=L_DARG+.001L_res+.01L_CF+.01L_harm；新AdamW lr.0005 wd.01最后更新，原三优化器完整继承。','',
        'seed0、10折、epoch10继续、max450、wait>200，Best仅test ACC严格提高、平局保留最早。Latest每10Epoch与终止保存。Fold2原训练成员保持377/407重叠，新谱统计与辅助排除二人且delta=0。无新组、seed、搜索、模态或历史重训。所有阈值与判断已在正式训练前config.json锁定。','',
        '## 六项诊断指标','',
        '单位%，10折/去Fold2的9折均值±population SD。Hard-AUC是硬分类BA；Prob-AUC使用连续TD概率。折间训练患者重叠，折并非独立生物重复，不据此宣称统计显著。']
    for scope in ('all10','without_fold2'):
        report+=['',f'### {scope}','','|组|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['scope']==scope:report.append('|'+row['group']+'|'+'|'.join(f"{row[k+'_mean']:.4f} ± {row[k+'_population_sd']:.4f}" for k in r.METRICS)+'|')
    report+=['','### 正式20折','','|组|折|Best epoch|终止epoch|ACC|BA|Prob-AUC|SEN|SPE|F1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in foldrows:
        if row['group'] in r.GROUPS:report.append(f"|{row['group']}|{row['fold']}|{row['best_epoch']}|{row['termination_epoch']}|"+'|'.join(f'{row[k]:.4f}' for k in r.METRICS)+'|')
    report+=['','## 配对比较','','六指标差值、各指标W/T/L、逐折差值、ASD/TD纠错与误伤，以及去Fold2的全部对应比较见paired_comparisons.csv。纠错方向为历史对照→新模型。','','|比较|ACC|BA|Prob-AUC|SEN|SPE|F1|ACC W/T/L|纠错/误伤/净纠错|','|---|---:|---:|---:|---:|---:|---:|---|---|']
    for g,ref in comparisons:
        x=dm(g,ref,'acc');report.append('|'+g+'−'+ref+'|'+'|'.join(f"{dm(g,ref,k)['mean_delta_pp']:+.4f}" for k in r.METRICS)+f"|{x['wins']}/{x['ties']}/{x['losses']}|{x['corrected']}/{x['harmed']}/{x['net']}|")
    report+=['','## 双尺度可靠性、残差与文本机制','']
    for g in r.GROUPS:
        rows=[m for m in mechs if m['group']==g]
        report.append(f"{g}：4%/8%方向贡献符号一致率{100*mean(g,'sign_agreement'):.3f}%，排序Spearman={mean(g,'scale_rank_spearman'):.6f}，稳定正方向覆盖{100*mean(g,'stable_positive_fraction'):.3f}%；rho均值{mean(g,'rho_mean'):.6f}。可靠有效患者{sum(m['valid_count'] for m in rows)}/{sum(m['legal_count'] for m in rows)}，折均比例{100*mean(g,'valid_fraction'):.3f}%，ASD/TD分别{100*mean(g,'asd_valid_fraction'):.3f}%/{100*mean(g,'td_valid_fraction'):.3f}%。教师错误患者新增监督数始终0；他们仍接受主分类损失。训练出现次数不是独立受试者n。")
        for branch in ('semantic','spectral'):
            report.append(f"{g} {branch}对可靠target的Spearman/Top1/Top3分别{alignmean(g,branch+'_reliable_alignment','spearman'):.6f}/{alignmean(g,branch+'_reliable_alignment','top1'):.6f}/{alignmean(g,branch+'_reliable_alignment','top3_overlap'):.6f}；对原单尺度target的Spearman={alignmean(g,branch+'_single_scale_alignment','spearman'):.6f}。")
        report.append(f"初始选择器在相同Best CAM和教师上的Spearman={alignmean(g,'initial_selector_current_teacher_alignment','spearman'):.6f}；选择器L1变化{mean(g,'initial_selector_current_cam_l1'):.6f}。原单尺度与新target L1={mean(g,'original_target_l1_to_reliable'):.6f}。CF首/末{mean(g,'cf_loss_first'):.6f}/{mean(g,'cf_loss_last'):.6f}；harm首/末{mean(g,'harm_first'):.6f}/{mean(g,'harm_last'):.6f}。")
        report.append(f"主cls/CF/harm在PSCE上的平均梯度范数{mean(g,'cls_psce_gradient_norm_mean'):.6g}/{mean(g,'cf_loss_psce_gradient_norm_mean'):.6g}/{mean(g,'harm_psce_gradient_norm_mean'):.6g}（未乘各loss系数，CF/harm实际乘.01）。CF/harm非零梯度比例{100*mean(g,'cf_loss_psce_gradient_norm_nonzero_fraction'):.3f}%/{100*mean(g,'harm_psce_gradient_norm_nonzero_fraction'):.3f}%；辅助对CAM/DARG为0。")
        report.append(f"测试残差相对范数均值{100*mean(g,'residual_relative_test_mean'):.5f}%；相对历史E82.2-G2自身CAM的均值范数比{mean(g,'ratio_residual_to_E822_own_CAM'):.6f}；在相同当前CAM上与原PSCE残差L2差{mean(g,'original_PSCE_same_CAM_delta_l2'):.6f}。近零比例{100*mean(g,'residual_near_zero_fraction'):.3f}%，近8%上限比例{100*mean(g,'residual_near_cap_fraction'):.3f}%。范数变化不直接代表诊断收益。")
        report.append(f"全部合法训练患者最终残差Margin改善/损害比例{100*mean(g,'gain_positive_fraction'):.3f}%/{100*mean(g,'gain_negative_fraction'):.3f}%，平均gain={mean(g,'gain_mean'):.6f}；可靠患者gain={mean(g,'valid_gain_mean'):.6f}；ASD/TD分别{mean(g,'asd_gain_mean'):.6f}/{mean(g,'td_gain_mean'):.6f}；教师正确/错误患者{mean(g,'correct_gain_mean'):.6f}/{mean(g,'error_gain_mean'):.6f}。")
        report.append(f"8 Prompt选择两两cosine={mean(g,'prompt_pairwise_cosine'):.8f}；beta={mean(g,'beta_mean'):.6f}、Null={mean(g,'null_mean'):.6f}、语义熵={mean(g,'entropy_mean'):.6f}；w_sem与w_sp L1={mean(g,'semantic_spectral_l1'):.6f}。8×8矩阵、每方向rho分位数、Prompt分布和目标对齐均在mechanism.csv。")
        report.append('相对B0的ASD/TD及总体纠错/误伤：'+json.dumps(corrections[g],ensure_ascii=False)+'。')
    report+=['','s_fold各折固定值：'+json.dumps({f:r.torch_load(r.RT/f'spectrum_f{f:02d}.pt')['s_fold'] for f in range(1,11)}),'',
        'G1/G2在相同G2 CAM上语义权重L1差均值：'+str(mean('G2','trained_G1_G2_same_CAM_semantic_L1'))+'；匿名选择对相同医学教师的对齐已记录。可靠选择对齐与测试ACC相对B0增量的折间Spearman：'+json.dumps(association)+'。这是描述性关联，不能证明泛化因果。','',
        '训练前已用历史E82.2 G2 Best在合法训练患者进行只读双尺度诊断；逐折符号一致率、方向排序、可靠覆盖和ASD/TD残差Margin见runtime/historical_readonly_diagnostic.json。这些数据未用于改变固定系数。','',
        '## G2 Best只读干预','','纠错方向为完整当前模型→干预模型；net<0表示移除该部分使当前模型受损。','','|干预|mean abs ΔP|最大ΔP|翻转|纠错|误伤|净纠错|','|---|---:|---:|---:|---:|---:|---:|']
    for x in interventions:report.append('|'+x['intervention']+f"|{x['mean_abs_probability_change']:.9f}|{x['max_abs_probability_change']:.9f}|{x['flips']}|{x['corrected']}|{x['harmed']}|{x['net']}|")
    report+=['','Medical→Anonymous只换固定q0。Semantic-off严格使用E82.2原定义w=w_sp；换q0后该干预输出逐元素相同。Residual-off将整个残差归零，保留已共同适配的骨干和原头，不等于重训DARG。三项连续概率、硬预测、六指标及ASD/TD结果保存在audit_status.json。可靠target和单尺度target只在合法训练患者只读比较，不伪装为推理消融。','',
        '## 冻结科研判断','',f'**{verdict}**。逐项冻结条件：'+json.dumps(checks,ensure_ascii=False)+'。','',
        '文本改变训练轨迹、改变推理输出、提供独立医学知识是三种不同主张。只有同容量医学优于匿名、即时文本移除有可检测且有意义的诊断损害并且机制与测试收益一致，才构成进一步研究依据。仅KL降低或训练Margin提高不能支撑医学Prompt创新。','',
        '若NO-GO，正式停止E82系列谱–语义Prompt第二创新模块，不追加E82.2.3、不开展新超参数搜索、不自动启动其他实验。E82.2.1既有NO-GO保留，本次是用户另行授权的预注册验证；没有事后改写历史标准。GO也只代表值得进一步验证，不能认定MICCAI创新或外部泛化成立。','',
        '## 工程审计与研究限制','',
        f"20/20 Best完整载入并逐元素重放概率，独立sklearn六指标、最早最大ACC核对通过。DARG公共Epoch RNG三元组{sum(a['historical_common_rng_epochs'] for a in audits)}条，E82.2公共三元组{sum(a['e822_common_rng_epochs'] for a in audits)}条均严格一致；同Fold G1/G2公共Epoch也完全一致。新增教师及PSCE辅助前向逐次验证零共享RNG消耗，无逐Epoch seed重设或RNG回写。",
        '', '20/20 Latest分别在原对象与全新模型对象上从磁盘恢复全部四优化器，下一次训练+评估的所有参数、优化器、概率、RNG逐元素/哈希一致；wait、best_acc、next_epoch也核对。独立33输入、双尺度扰动、Margin、rho、有效筛选、s_fold、CF/harm/res、辅助梯度隔离、主分类全链梯度及Fold2新增路径保护均通过；交换测试标签不改变推理。',
        '', '保护SHA汇总：'+json.dumps(con['protection'],ensure_ascii=False)+'。manifest保存逐文件SHA；根模型代码、数据、历史源码和Checkpoint前后未变。',
        '', '首选OSSFS最小写入发生errno5；正式任务前切至/model并通过fsync/原子rename/读回。无训练中迁移，已完成折未重训。单设备98GB、每worker CPU线程1、最多10worker，资源与异常记录在runtime/execution.json。',
        '', '保留历史val=test选Best、CV前标准化与RFE、transductive图及Fold2训练测试重叠。去Fold2后仍有前述其他限制。历次E82研究反复基于相同测试折选择方向，存在额外的研究选择偏倚；本次预注册不能消除该历史偏倚。没有独立外部测试、嵌套CV或多seed。训练集局部模型反事实的一致性不是医学知识可靠性，潜在方向遮蔽不是生物学因果操作。','']
    r.atomic_json(r.RT/'scientific_conclusion.json',con)
    (r.RESULTS/'report.md').write_text('\n'.join(report)+'\n')
    return dict(summary=summary,paired=paired,conclusion=con,interventions=interventions)


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

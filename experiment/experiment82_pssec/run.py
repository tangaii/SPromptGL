#!/usr/bin/env python3
"""E82: locked E74.4-G2 inheritance, feature hook, resumable 60-task runner."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch
from pssec import PSSEC, SPECS, PROMPTS, EPS, build_spectra, anonymous_text, smooth_limit

HERE = Path(__file__).resolve().parent
REPO = Path('/data/adodas/SPromptGL-main')
CFG = json.loads((HERE/'config.json').read_text())
RT, RESULTS = HERE/'runtime', HERE/'results'
GROUPS = tuple(CFG['groups'])
BASE_GROUP = CFG['base_group']
METRICS = ('acc', 'hard_auc', 'prob_auc', 'sen', 'spe', 'asd_f1')
BASE = None
ADAPTER = Path('/model/spromptgl/experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py')
TEXT_SOURCE = REPO/'experiment/experiment60_dss_pen/.runtime/text_embeddings.pt'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def sha(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous()
            h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype,x.shape)).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):h.update(str(k).encode()+b'\0');visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
        else:h.update(json.dumps(x,sort_keys=True,default=str).encode()+b'\0')
    visit(value)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with tmp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def torch_load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    tmp=Path(path).with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def log_row(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()


def runtime():
    global BASE
    if BASE is None:
        spec=importlib.util.spec_from_file_location('e82_readonly_path_adapter',ADAPTER)
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
        BASE=adapter.load_base_runner()
        base_runtime().runtime()
        torch.set_num_threads(1)
    return BASE


def base_runtime():
    return runtime().load_history().load_e69().load_e61().load_base()


@contextmanager
def isolated_rng(seed=None):
    r=base_runtime();saved=r.pack_rng()
    try:
        if seed is not None:
            import random
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        yield
    finally:
        r.restore_rng(saved)
        assert r.rng_sha(r.pack_rng())==r.rng_sha(saved)


def paths(group,fold):
    key=f'{group}_f{fold:02d}'
    return RT/'tasks'/f'{key}.json',RT/'checkpoints'/f'{key}.best.pt',RT/'checkpoints'/f'{key}.latest.pt',RT/'logs'/f'{key}.jsonl'


def legal_train(state):
    idx=np.asarray(state['train_np'],dtype=np.int64)
    return idx[~np.isin(idx,[376,406])] if state['fold']==2 else idx


def historical_task(fold):
    return json.loads((runtime().EXP/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())


def prepare(fold):
    state,warmup,identity=runtime().prepare(fold)
    state['fold']=fold
    task=historical_task(fold)
    assert task['status']=='COMPLETE' and task['identity']==identity
    assert runtime().load_history().state_model_sha(state)==identity['model_cam_sha']
    assert runtime().sha_object({k:state[k].state_dict() for k in ('opt_base','opt_cam','opt_added')})==identity['optimizer_sha']
    assert base_runtime().rng_sha(base_runtime().pack_rng())==warmup['rng_sha']
    assert state['train_pairs_np'].shape==(4096,2) and np.isin(state['train_pairs_np'],state['train_np']).all()
    assert tuple(state['x'].shape)==(871,320)
    for k,(name,lr,wd) in CFG['historical_optimizers'].items():
        assert type(state[k]).__name__==name
        assert all(abs(g['lr']-lr)<1e-12 and abs(g['weight_decay']-wd)<1e-12 for g in state[k].param_groups)
    return state,warmup,identity


def source_maps():
    historical={str(Path(p).resolve()):sha_file(p) for p in runtime().source_manifest()}
    extra=[ADAPTER,TEXT_SOURCE]
    vendor=REPO/'experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment51_dugsa'
    extra += [vendor/'dugsa.py'] + list((vendor/'source').glob('*.py'))
    for f in ('run.py','soft_positive_dcr.py','config.json','manifest.json','audit.py','results/report.md','results/fold_metrics.csv','results/audit_status.json'):
        extra.append(runtime().EXP/f)
    extra += list((runtime().EXP/'.runtime/checkpoints').glob(f'{BASE_GROUP}_f*.best.pt'))
    extra += list((runtime().EXP/'.runtime/tasks').glob(f'{BASE_GROUP}_f*.json'))
    for p in extra:historical[str(p.resolve())]=sha_file(p)
    code={str(HERE/f):sha_file(HERE/f) for f in ('run.py','pssec.py','audit.py','config.json')}
    return {'protected':historical,'implementation':code}


def implementation_sha():
    return sha({f:sha_file(HERE/f) for f in ('run.py','pssec.py','config.json')})


def build_prompt_cache():
    path=RT/'prompts.pt'
    if path.exists():return torch_load(path)
    with isolated_rng():
        old=torch_load(TEXT_SOURCE)
        assert old['text_map']['original']==PROMPTS
        t=old['embeddings']['original'].float().contiguous()
        assert t.shape==(8,1024) and torch.isfinite(t).all()
        anon,meta=anonymous_text(t,CFG['anonymous_seed'])
        assert meta['gram_error_float64']<1e-9 and meta['gram_error_float32']<1e-3 and meta['relative_text_change']>.1
        payload={'medical':t,'anonymous':anon,'metadata':meta,'rotation_sha':sha(meta),
                 'source':str(TEXT_SOURCE),'source_sha256':sha_file(TEXT_SOURCE),'texts':PROMPTS,
                 'medical_sha':sha(t),'anonymous_sha':sha(anon),'encoder':'frozen CLIP RN50 raw embeddings; E60 original cache'}
        atomic_torch(path,payload)
    return payload


def build_fold_cache(state,identity):
    fold=state['fold'];path=RT/f'spectrum_f{fold:02d}.pt'
    if path.exists():
        saved=torch_load(path);assert saved['identity']==identity
        return saved
    with isolated_rng():
        modes=(state['model'].training,state['cam'].training)
        captured=[]
        def capture(_m,args):captured.append(args[0].detach().cpu().clone())
        hook=state['cam'].Outputlayer.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():out=runtime().forward(state,BASE_GROUP,training=False)
        finally:
            hook.remove();state['model'].train(modes[0]);state['cam'].train(modes[1])
        assert len(captured)==1 and captured[0].shape==(871,144)
        z=captured[0];idx=legal_train(state)
        labels=torch.as_tensor(state['labels_np'][idx])
        spectra=build_spectra(z[idx],labels)
        repeat=build_spectra(z[idx],labels)
        for s in ('O','I'):
            assert sha(spectra[s])==sha(repeat[s])
        saved={'fold':fold,'identity':identity,'legal_train':idx,'legal_train_sha':sha(idx),
               'legal_label_sha':sha(labels),'warmup_cam':z,'spectra':spectra,
               'spectra_sha':sha(spectra),'feature_sha':sha(z),
               'repeat_projector_max_abs':{s:float((spectra[s]['U']@spectra[s]['U'].T-repeat[s]['U']@repeat[s]['U'].T).abs().max()) for s in ('O','I')},
               'warmup_graph_sha':[sha(x) for x in out['original_adjs']],
               'warmup_unified_graph_sha':[sha(x) for x in out['unified_adjs']]}
        atomic_torch(path,saved)
        del out
    return saved


def make_module(state,group,cache=None,prompts=None):
    cache=torch_load(RT/f"spectrum_f{state['fold']:02d}.pt") if cache is None else cache
    prompts=torch_load(RT/'prompts.pt') if prompts is None else prompts
    variant,_,text=SPECS[group]
    with isolated_rng(CFG['initialization_seed_base']+state['fold']):
        module=PSSEC(cache['spectra'][variant],prompts[text],group,state['fold']).to(state['x'].device)
    param_sha=sha(dict(module.named_parameters()))
    optimizer=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
    return module,optimizer,param_sha


def forward(state,module,training=False,intervention='full'):
    module.train(training)
    counts={'output':0,'cam':0,'gcn':[0,0,0,0]};details={}
    def pre(_m,args):
        counts['output']+=1
        assert counts['output']==1 and len(args)==2 and args[0].shape==(871,144)
        changed,d=module(args[0],intervention=intervention)
        details.update(d)
        return (changed,args[1])
    handles=[state['cam'].Outputlayer.register_forward_pre_hook(pre),
        state['cam'].register_forward_hook(lambda *_:counts.__setitem__('cam',counts['cam']+1))]
    for i,gcn in enumerate(state['model'].gcn):
        handles.append(gcn.register_forward_hook(lambda _m,_a,_o,i=i:counts['gcn'].__setitem__(i,counts['gcn'][i]+1)))
    try:out=runtime().forward(state,BASE_GROUP,training=training)
    finally:
        for handle in handles:handle.remove()
    assert counts=={'output':1,'cam':1,'gcn':[1,1,1,1]},counts
    idx=torch.as_tensor(legal_train(state),device=state['x'].device)
    res=(details['delta'][idx].square().sum(-1)/(details['z'][idx].square().sum(-1)+EPS)).mean()
    out.update(darg_total=out['total'],res=res,details=details,calls=counts)
    out['total']=out['darg_total']+.001*res
    return out


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def bundle(state,module,opt):
    return {'base':runtime().load_history().state_bundle(state),'pssec':cpu_state(module),'opt_pssec':copy.deepcopy(opt.state_dict())}


def load_bundle(state,module,opt,saved):
    runtime().load_history().load_bundle(state,saved['base'])
    module.load_state_dict(saved['pssec'],strict=True);opt.load_state_dict(copy.deepcopy(saved['opt_pssec']))


def grad_norm(params):
    vals=[p.grad.detach().double().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt().cpu()) if vals else 0.


def gradients(state,module):
    out={k:grad_norm(state[k].parameters()) for k in ('model','cam')}
    out['outputlayer']=grad_norm(state['cam'].Outputlayer.parameters())
    for prefix in ('direction','evidence','norm','text_proj','key','null','spectral'):
        out[prefix]=grad_norm(p for n,p in module.named_parameters() if n==prefix or n.startswith(prefix+'.'))
    return out


def zero(state,opt):
    runtime().load_history().zero_optimizers(state);opt.zero_grad(set_to_none=True)


def step(state,opt):
    for k in ('opt_base','opt_cam','opt_added'):state[k].step()
    opt.step()


def finite_grads(state,module):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for m in (state['model'],state['cam'],module) for p in m.parameters())


def metric_copy(m):
    return {k:float(v) for k,v in m.items() if k not in ('pred','prob')}


def classification_gradient_audit(out,state,module):
    active=[(n,p) for n,p in module.named_parameters() if p.requires_grad]
    targets=[p for n,p in active]+[out['details']['z'],out['fused']]
    grads=torch.autograd.grad(out['cls'],targets,retain_graph=True,allow_unused=True)
    norms={n:float(g.detach().norm().cpu()) if g is not None else 0. for (n,p),g in zip(active,grads)}
    norms['cam144']=float(grads[-2].norm().cpu());norms['upstream_darg320']=float(grads[-1].norm().cpu())
    assert all(math.isfinite(v) for v in norms.values()),norms
    # The shared final scalar bias adds the same constant to all 16 logits;
    # softmax cancels it exactly. Require every enabled *path* to carry a
    # classification gradient, while recording this non-identifiable scalar.
    assert all(v>0 for n,v in norms.items() if n!='spectral.2.bias'),norms
    # Disabled parameters are absent from the actual forward graph and optimizer.
    assert all(p.grad is None for p in module.parameters() if not p.requires_grad)
    return norms


def preflight():
    for p in (RT/'tasks',RT/'checkpoints',RT/'logs',RESULTS):p.mkdir(parents=True,exist_ok=True)
    runtime();assert torch.cuda.is_available() and torch.cuda.device_count()==1
    protected=source_maps()
    manifest=json.loads((HERE/'manifest.json').read_text())
    manifest.update(status='PREFLIGHT_RUNNING',source_maps=protected,config=CFG,
        source_review={'spromptgl':'GFR Eq1-2, CAM Eq4, SPGS Eq5, RES Eq6 checked against actual source; historical total excludes unused ALoss; PLNNoPEN inherited',
          'ssc':'DBPC per-prompt top-r and consensus, SKP whitened text covariance/SVD, entropy selects branch budgets; E82 uses patient training CAM covariance and differentiable coverage; no SSC classifier copied',
          'e813':'read-only loader and try/finally Outputlayer hooks reviewed; E82 replaces input features before original classifier',
          'basis_readout':'16 latent directions are not anatomical regions or 4x4x9 patches'})
    atomic_json(HERE/'manifest.json',manifest)
    inheritance=[];group_checks=[];peak=0
    for fold in range(1,11):
        state,warmup,identity=prepare(fold)
        r=base_runtime();r0=r.rng_sha(r.pack_rng())
        cache=build_fold_cache(state,identity);prompts=build_prompt_cache()
        inits=[]
        for group in GROUPS:
            module,opt,init_sha=make_module(state,group,cache,prompts);inits.append(init_sha)
            if fold==1:
                out=forward(state,module,training=True)
                assert out['details']['w'].shape==(871,16)
                if group not in ('G1','G3'):assert out['details']['tokens'].shape==(871,16,32)
                assert all(torch.isfinite(v).all() for v in out['details'].values() if torch.is_tensor(v))
                g=classification_gradient_audit(out,state,module)
                zero(state,opt);out['total'].backward();assert finite_grads(state,module)
                group_checks.append({'group':group,**module.counts(),'init_sha':init_sha,'cls_gradient_norms':g,'calls':out['calls']})
                del out;r.restore_rng(warmup['rng'])
            del module,opt
        assert len(set(inits))==1 and r.rng_sha(r.pack_rng())==r0
        inheritance.append({'fold':fold,'identity':identity,'new_parameter_sha':inits[0],
             'spectra_sha':cache['spectra_sha'],'legal_train_sha':cache['legal_train_sha'],
             'legal_train_count':len(cache['legal_train']),'repeat_projector_max_abs':cache['repeat_projector_max_abs']})
        del state,cache;torch.cuda.empty_cache()
        print(f'preflight inheritance fold={fold} PASS',flush=True)
    # Exactly one historical no-operation forward/update parity test.
    state,warmup,identity=prepare(1);module,opt,_=make_module(state,'G5')
    r=base_runtime();initial=bundle(state,module,opt);rng=r.pack_rng()
    zero(state,opt);original=runtime().forward(state,BASE_GROUP,training=True)
    original['total'].backward()
    grad_sha=sha({k:{n:p.grad for n,p in state[k].named_parameters()} for k in ('model','cam')})
    for k in ('opt_base','opt_cam','opt_added'):state[k].step()
    expected=runtime().load_history().state_bundle(state);endrng=r.rng_sha(r.pack_rng())
    expected_logp=original['logp'].detach().clone();expected_losses={k:original[k].detach().clone() for k in ('cls','graph','pair','adv','preserve','total')}
    del original
    load_bundle(state,module,opt,initial);r.restore_rng(rng);zero(state,opt)
    out=forward(state,module,training=True,intervention='noop')
    assert torch.equal(out['logp'],expected_logp)
    assert all(torch.equal(out[k],v) for k,v in expected_losses.items())
    assert torch.count_nonzero(out['details']['delta'])==0
    out['total'].backward()
    assert grad_sha==sha({k:{n:p.grad for n,p in state[k].named_parameters()} for k in ('model','cam')})
    for k in ('opt_base','opt_cam','opt_added'):state[k].step()
    assert sha(expected)==sha(runtime().load_history().state_bundle(state))
    assert endrng==r.rng_sha(r.pack_rng())
    del out
    # Smooth limit edge cases and gradients, independent deterministic CPU data.
    z=torch.arange(144,dtype=torch.float64)[None].expand(5,-1)/144
    raw=torch.stack([torch.zeros(144),torch.ones(144)*1e-10,torch.ones(144)*.01,torch.ones(144),torch.ones(144)*1e3]).double().requires_grad_()
    limited=smooth_limit(raw,z)
    assert torch.count_nonzero(limited[0])==0
    assert torch.all(limited.norm(dim=-1)<=.08*z.norm(dim=-1)+1e-12)
    limited.sum().backward();assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[0],torch.ones(144,dtype=torch.float64))
    # Actual full checkpoint interruption/recovery replay: one next update.
    load_bundle(state,module,opt,initial);r.restore_rng(rng);zero(state,opt)
    o=forward(state,module,True);o['total'].backward();step(state,opt);del o
    saved={'current':bundle(state,module,opt),'rng':r.pack_rng()}
    replay_path=RT/'recovery_check.pt';atomic_torch(replay_path,saved)
    zero(state,opt);o=forward(state,module,True);o['total'].backward();step(state,opt);del o
    next_sha=sha(bundle(state,module,opt));next_rng=r.rng_sha(r.pack_rng())
    disk=torch_load(replay_path);load_bundle(state,module,opt,disk['current']);r.restore_rng(disk['rng'])
    zero(state,opt);o=forward(state,module,True);o['total'].backward();step(state,opt);del o
    assert sha(bundle(state,module,opt))==next_sha and r.rng_sha(r.pack_rng())==next_rng
    replay_path.unlink()
    peak=int(torch.cuda.max_memory_allocated())
    pre={'status':'PASS','source_sha':implementation_sha(),'inheritance':inheritance,
        'group_checks':group_checks,'noop_output_losses_gradients_three_optimizers_rng_exact':True,
        'checkpoint_next_update_state_and_rng_exact':True,'smooth_limit_and_uniform_exact':True,
        'peak_gpu_bytes':peak,'device':torch.cuda.get_device_name(0),'torch':torch.__version__}
    atomic_json(RT/'preflight.json',pre)
    manifest.update(status='PREFLIGHT_PASS',preflight=pre)
    atomic_json(HERE/'manifest.json',manifest)
    print('PREFLIGHT PASS',flush=True)


def task_complete(group,fold):
    task,best,latest,log=paths(group,fold)
    if not task.exists() or not best.exists() or not latest.exists():return False
    t=json.loads(task.read_text())
    return t.get('status')=='COMPLETE' and t.get('source_sha')==implementation_sha() and t.get('best_sha')==sha_file(best)


def train_worker(group,fold):
    start=time.time();state,warmup,identity=prepare(fold)
    task,best_path,latest_path,log=paths(group,fold)
    source=implementation_sha();pre=json.loads((RT/'preflight.json').read_text())
    assert pre['status']=='PASS' and pre['source_sha']==source
    if task_complete(group,fold):return
    cache=torch_load(RT/f'spectrum_f{fold:02d}.pt');assert cache['identity']==identity
    module,opt,init_sha=make_module(state,group,cache)
    assert init_sha==pre['inheritance'][fold-1]['new_parameter_sha']
    r=base_runtime();assert r.rng_sha(r.pack_rng())==warmup['rng_sha']
    meta={'group':group,'fold':fold,'seed':0,'identity':identity,'source_sha':source,
          'new_parameter_sha':init_sha,'spectra_sha':cache['spectra_sha'],'config':CFG,
          'spectrum':SPECS[group][0],'counts':module.counts()}
    epoch0,wait,best_acc,best_epoch=10,0,-1.,-1
    trace=[];best_meta=None;resumes=[]
    if latest_path.exists():
        latest=torch_load(latest_path)
        for k in meta:assert sha(meta[k])==sha(latest[k]),k
        load_bundle(state,module,opt,latest['current']);r.restore_rng(latest['rng'])
        assert sha(bundle(state,module,opt))==latest['bundle_sha'] and r.rng_sha(r.pack_rng())==latest['rng_sha']
        epoch0,wait,best_acc,best_epoch=latest['next_epoch'],latest['wait'],latest['best_acc'],latest['best_epoch']
        trace,best_meta=latest['trace'],latest['best_meta']
        resumes=latest.get('resumes',[])+[{'next_epoch':epoch0,'bundle_sha':latest['bundle_sha'],'rng_sha':latest['rng_sha']}]
    baseline_trace={t['epoch']:t for t in historical_task(fold)['rng_trace']}
    torch.cuda.reset_peak_memory_stats();executed=epoch0
    for epoch in range(epoch0,450):
        if wait>200:break
        before=r.rng_sha(r.pack_rng());zero(state,opt)
        out=forward(state,module,True)
        assert torch.isfinite(out['total'])
        if epoch==10:cls_grads=classification_gradient_audit(out,state,module)
        out['total'].backward();assert finite_grads(state,module)
        grad=gradients(state,module);step(state,opt)
        after=r.rng_sha(r.pack_rng())
        losses={k:float(out[k].detach().cpu()) for k in ('cls','graph','pair','adv','preserve','darg_total','res','total')}
        del out
        with torch.no_grad():
            evaluated=forward(state,module,False)
            metrics=r.classification_metrics(evaluated['logp'],state['test'],state['y'])
        after_eval=r.rng_sha(r.pack_rng())
        item={'epoch':epoch+1,'rng_before':before,'rng_after_train':after,'rng_after_eval':after_eval}
        if epoch+1 in baseline_trace:assert item==baseline_trace[epoch+1],f'common RNG mismatch {group}/{fold}/{epoch+1}'
        trace.append(item);executed=epoch+1
        improved=metrics['acc']>best_acc
        if improved:
            best_acc,best_epoch,wait=float(metrics['acc']),executed,0
            best_meta={'best_epoch':best_epoch,'metrics':metric_copy(metrics),'probability':metrics['prob'].copy(),
                       'prediction':metrics['pred'].copy(),'losses':losses,'gradients':grad}
            atomic_torch(best_path,{**meta,**best_meta,'current':bundle(state,module,opt),
                'rng':r.pack_rng(),'rng_sha':after_eval,'trace':trace.copy()})
        else:wait+=1
        log_row(log,{'epoch':executed,'acc':float(metrics['acc']),'best_acc':best_acc,'best_epoch':best_epoch,
             'improved':improved,'wait':wait,'losses':losses,'gradients':grad,**item})
        del evaluated
        if executed%10==0 or executed==450 or wait>200:
            current=bundle(state,module,opt);rng=r.pack_rng()
            atomic_torch(latest_path,{**meta,'current':current,'bundle_sha':sha(current),'rng':rng,'rng_sha':r.rng_sha(rng),
                'next_epoch':executed,'wait':wait,'best_acc':best_acc,'best_epoch':best_epoch,
                'best_meta':best_meta,'trace':trace,'resumes':resumes})
            print(f'{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch} wait={wait}',flush=True)
    assert best_meta is not None
    atomic_json(task,{**meta,'status':'COMPLETE','best_sha':sha_file(best_path),'latest_sha':sha_file(latest_path),
         'best_epoch':best_epoch,'epochs_executed':executed,'metrics':best_meta['metrics'],'rng_trace':trace,
         'wall_seconds':time.time()-start,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()),
         'resumes':resumes,'first_classification_gradients':cls_grads if epoch0==10 else 'audited in original first epoch'})
    print(f'COMPLETE {group} fold={fold}',flush=True)


def run_all(workers):
    assert 1<=workers<=10
    pre=json.loads((RT/'preflight.json').read_text());assert pre['status']=='PASS' and pre['source_sha']==implementation_sha()
    pending=[(g,f,0) for f in range(1,11) for g in GROUPS if not task_complete(g,f)]
    active={};failures=[];started=time.time();max_active=0;limit=workers
    while pending or active:
        while pending and len(active)<limit:
            g,f,attempt=pending.pop(0);file=RT/'logs'/f'{g}_f{f:02d}.stdout.log'
            stream=file.open('a')
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'run.py'),'worker','--group',g,'--fold',str(f)],stdout=stream,stderr=subprocess.STDOUT,cwd=HERE)
            active[p.pid]=(p,g,f,attempt,stream,file);max_active=max(max_active,len(active))
        for pid,(p,g,f,attempt,stream,file) in list(active.items()):
            code=p.poll()
            if code is None:continue
            stream.close();del active[pid]
            if code==0 and task_complete(g,f):print(f'FINISHED {g}/{f}; active={len(active)}',flush=True)
            else:
                tail=file.read_text()[-5000:];oom='out of memory' in tail.lower()
                failures.append({'group':g,'fold':f,'attempt':attempt+1,'returncode':code,'oom':oom,'tail':tail})
                if oom and attempt<2:
                    limit=max(1,limit-1);pending.insert(0,(g,f,attempt+1))
                else:print(f'FAILED {g}/{f}: {tail[-900:]}',flush=True)
        execution={'status':'RUNNING' if pending or active else 'TRAINED' if not failures else 'TRAINING_FINISHED_WITH_RETRIES_OR_FAILURES',
             'completed_tasks':sum(task_complete(g,f) for f in range(1,11) for g in GROUPS),
             'expected_tasks':60,'active_tasks':len(active),'pending_tasks':len(pending),'max_workers_observed':max_active,
             'worker_limit':limit,'elapsed_seconds':time.time()-started,'failures':failures}
        atomic_json(RT/'execution.json',execution)
        if active:time.sleep(2)
    assert all(task_complete(g,f) for f in range(1,11) for g in GROUPS),'some formal tasks failed; repair and resume Latest'
    execution.update(status='TRAINED',wall_seconds=time.time()-started);atomic_json(RT/'execution.json',execution)
    manifest=json.loads((HERE/'manifest.json').read_text());manifest.update(status='TRAINED_PENDING_INDEPENDENT_AUDIT',completed_tasks=60,execution=execution);atomic_json(HERE/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','all','worker']);p.add_argument('--workers',type=int,default=10);p.add_argument('--group',choices=GROUPS);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='all':run_all(a.workers)
    else:train_worker(a.group,a.fold)


if __name__=='__main__':main()

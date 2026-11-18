"""E82.2.2 RCE-PSCE: locked original DARG/PSCE initialization; 20 formal tasks."""

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

from rce_psce import RCEPSCE, PSCE, GROUPS as GROUP_SPECS, PROMPTS, EPS, build_spectra, text_geometry, smooth_limit, teacher_snapshot, teacher_logits, counterfactual, auxiliary, fixed_scale, reliability

HERE=Path(__file__).resolve().parent

REPO=Path('/data/adodas/SPromptGL-main')

CFG=json.loads((HERE/'config.json').read_text())

RT,RESULTS=HERE/'runtime',HERE/'results'

GROUPS=tuple(CFG['groups']);BASE_GROUP=CFG['base_group']

GROUPS=tuple(CFG['groups']);BASE_GROUP=CFG['base_group']

METRICS=('acc','hard_auc','prob_auc','sen','spe','asd_f1')

BASE=None

sys.modules['rce_runner']=sys.modules[__name__]

ADAPTER=REPO/'experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py'

E82=REPO/'experiment/experiment82_pssec'

E822=Path('/model/spromptgl/experiment/experiment82_2_psce')
E8221=Path('/model/spromptgl/experiment/experiment82_2_1_bscp')

TEXT_SOURCE=REPO/'experiment/experiment60_dss_pen/.runtime/text_embeddings.pt'

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
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

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

def implementation_sha():
    return sha({f:sha_file(HERE/f) for f in ('run.py','rce_psce.py','config.json')})

def build_prompt_cache():
    path=RT/'prompts.pt'
    if path.exists():return torch_load(path)
    with isolated_rng():
        old=torch_load(TEXT_SOURCE);assert old['text_map']['original']==PROMPTS
        t=old['embeddings']['original'].float().contiguous()
        payload=text_geometry(t,CFG['anonymous_seed'])
        assert payload['gram_error']<1e-9 and payload['gram_error_float32']<1e-3
        assert payload['projection_gram_medical']<1e-9 and payload['projection_gram_anonymous']<1e-9
        assert payload['projection_orthogonality']<1e-10 and payload['text_relative_change']>.1
        payload.update(source=str(TEXT_SOURCE),source_sha256=sha_file(TEXT_SOURCE),texts=PROMPTS)
        payload['identity_sha']=sha(payload)
        atomic_torch(path,payload)
    old_cache=torch_load(E822/'runtime/prompts.pt')
    assert payload['identity_sha']==old_cache['identity_sha']
    for key in ('q_medical','q_anonymous','P_text'):assert torch.equal(payload[key],old_cache[key]),key
    return payload

def build_fold_cache(state,identity):
    fold=state['fold'];path=RT/f'spectrum_f{fold:02d}.pt'
    if path.exists():
        c=torch_load(path);assert c['identity']==identity;return c
    with isolated_rng():
        entry_rng=base_runtime().rng_sha(base_runtime().pack_rng())
        captured=[]
        handle=state['cam'].Outputlayer.register_forward_pre_hook(lambda _m,args:captured.append(args[0].detach().cpu().clone()))
        modes=(state['model'].training,state['cam'].training)
        try:
            with torch.no_grad():out=runtime().forward(state,BASE_GROUP,training=False)
        finally:
            handle.remove();state['model'].train(modes[0]);state['cam'].train(modes[1])
        assert len(captured)==1 and captured[0].shape==(871,144)
        z=captured[0];idx=legal_train(state);labels=torch.as_tensor(state['labels_np'][idx])
        spectrum=build_spectra(z[idx],labels);repeat=build_spectra(z[idx],labels)
        assert sha(spectrum)==sha(repeat)
        old=torch_load(E82/'runtime'/f'spectrum_f{fold:02d}.pt')
        assert torch.equal(z,old['warmup_cam']) and torch.equal(spectrum['U'],old['spectra']['O']['U'])
        device=state['x'].device
        teacher=teacher_snapshot(state['cam'].Outputlayer)
        cf=counterfactual(teacher,z.to(device),spectrum['U'].float().to(device),spectrum['mu'].float().to(device),torch.as_tensor(idx,device=device),state['y'][torch.as_tensor(idx,device=device)])
        scale,scale_details=fixed_scale(cf)
        assert bool(cf['valid'].any()),f'design premise failed: fold {fold} has no reliable warmup targets'
        c={'s_fold':scale,'scale_details':scale_details,'initial_reliable_count':int(cf['valid'].sum()),'fold':fold,'identity':identity,'legal_train':idx,'legal_train_sha':sha(idx),'legal_label_sha':sha(labels),
            'warmup_cam':z,'spectrum':spectrum,'spectra_sha':sha(spectrum),'feature_sha':sha(z),
            'repeat_projector_max_abs':float((spectrum['U']@spectrum['U'].T-repeat['U']@repeat['U'].T).abs().max()),
            'e82_original_U_sha':sha(old['spectra']['O']['U']),
            'warmup_graph_sha':[sha(x) for x in out['original_adjs']],
            'warmup_unified_graph_sha':[sha(x) for x in out['unified_adjs']]}
        assert entry_rng==base_runtime().rng_sha(base_runtime().pack_rng()),'spectral caching consumed RNG'
        atomic_torch(path,c)
    return c

def make_module(state,group,cache=None,prompts=None):
    cache=torch_load(RT/f"spectrum_f{state['fold']:02d}.pt") if cache is None else cache
    prompts=torch_load(RT/'prompts.pt') if prompts is None else prompts
    with isolated_rng(CFG['initialization_seed_base']+state['fold']):
        module=RCEPSCE(cache['spectrum'],prompts,group,state['fold'],cache['s_fold']).to(state['x'].device)
    param_sha=sha(dict(module.named_parameters()))
    opt=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
    return module,opt,param_sha

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def bundle(state,module,opt):
    return {'base':runtime().load_history().state_bundle(state),'psce':cpu_state(module),'opt_psce':copy.deepcopy(opt.state_dict())}

def load_bundle(state,module,opt,saved):
    runtime().load_history().load_bundle(state,saved['base'])
    module.load_state_dict(saved['psce'],strict=True);opt.load_state_dict(copy.deepcopy(saved['opt_psce']))

def grad_norm(params):
    vals=[p.grad.detach().double().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt().cpu()) if vals else 0.

def gradients(state,module):
    out={k:grad_norm(state[k].parameters()) for k in ('model','cam')}
    out['outputlayer']=grad_norm(state['cam'].Outputlayer.parameters())
    for prefix in ('direction','evidence','norm','key','null','spectral'):
        out[prefix]=grad_norm(p for n,p in module.named_parameters() if n==prefix or n.startswith(prefix+'.'))
    return out

def zero(state,opt):runtime().load_history().zero_optimizers(state);opt.zero_grad(set_to_none=True)

def step(state,opt):
    for k in ('opt_base','opt_cam','opt_added'):state[k].step()
    opt.step()

def finite_grads(state,module):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for m in (state['model'],state['cam'],module) for p in m.parameters())

def metric_copy(m):return {k:float(v) for k,v in m.items() if k not in ('pred','prob')}

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
          'new_parameter_sha':init_sha,'spectra_sha':cache['spectra_sha'],'text_identity_sha':torch_load(RT/'prompts.pt')['identity_sha'],'config':CFG,
          'spectrum':'O','counts':module.counts(),'text_kind':GROUP_SPECS[group]}
    epoch0,wait,best_acc,best_epoch=10,0,-1.,-1
    cls_grads=None
    trace=[];best_meta=None;resumes=[]
    if latest_path.exists():
        latest=torch_load(latest_path)
        for k in meta:assert sha(meta[k])==sha(latest[k]),k
        load_bundle(state,module,opt,latest['current']);r.restore_rng(latest['rng'])
        assert sha(bundle(state,module,opt))==latest['bundle_sha'] and r.rng_sha(r.pack_rng())==latest['rng_sha']
        epoch0,wait,best_acc,best_epoch=latest['next_epoch'],latest['wait'],latest['best_acc'],latest['best_epoch']
        trace,best_meta=latest['trace'],latest['best_meta']
        retained=[v for v in read_logs(log) if v['epoch']<=epoch0]
        atomic_log(log,retained)
        resumes=latest.get('resumes',[])+[{'next_epoch':epoch0,'bundle_sha':latest['bundle_sha'],'rng_sha':latest['rng_sha']}]
    baseline_trace={t['epoch']:t for t in historical_task(fold)['rng_trace']}
    psce_trace=e822_trace(group,fold)
    if not latest_path.exists() and log.exists():atomic_log(log,[])
    torch.cuda.reset_peak_memory_stats();executed=epoch0
    for epoch in range(epoch0,450):
        if wait>200:break
        before=r.rng_sha(r.pack_rng());zero(state,opt)
        out=forward(state,module,True)
        assert torch.isfinite(out['total'])
        if epoch==10:cls_grads=classification_gradient_audit(out,state,module)
        loss_gradients=loss_gradient_norms(out,module)
        teacher_stats=teacher_summary(out['cf'],out['aux_details'],out['gain'],state['y'][torch.as_tensor(legal_train(state),device=state['x'].device)])
        out['total'].backward();assert finite_grads(state,module)
        grad=gradients(state,module);step(state,opt)
        after=r.rng_sha(r.pack_rng())
        losses={k:float(out[k].detach().cpu()) for k in ('cls','graph','pair','adv','preserve','darg_total','res','cf_loss','harm','total')}
        del out
        with torch.no_grad():
            evaluated=forward(state,module,False)
            metrics=r.classification_metrics(evaluated['logp'],state['test'],state['y'])
        after_eval=r.rng_sha(r.pack_rng())
        item={'epoch':epoch+1,'rng_before':before,'rng_after_train':after,'rng_after_eval':after_eval}
        if epoch+1 in baseline_trace:assert item==baseline_trace[epoch+1],f'common RNG mismatch {group}/{fold}/{epoch+1}'
        if epoch+1 in psce_trace:assert item==psce_trace[epoch+1],f'E82.2 RNG mismatch {group}/{fold}/{epoch+1}'
        trace.append(item);executed=epoch+1
        improved=metrics['acc']>best_acc
        if improved:
            best_acc,best_epoch,wait=float(metrics['acc']),executed,0
            best_meta={'best_epoch':best_epoch,'metrics':metric_copy(metrics),'probability':metrics['prob'].copy(),
                       'prediction':metrics['pred'].copy(),'losses':losses,'gradients':grad}
            atomic_torch(best_path,{**meta,**best_meta,'current':bundle(state,module,opt),
                'rng':r.pack_rng(),'rng_sha':after_eval,'trace':trace.copy(),
                'wait':0,'next_epoch':executed,'best_acc':best_acc,'best_meta':best_meta,'bundle_sha':sha(bundle(state,module,opt))})
        else:wait+=1
        log_row(log,{'epoch':executed,'acc':float(metrics['acc']),'best_acc':best_acc,'best_epoch':best_epoch,
             'improved':improved,'wait':wait,'losses':losses,'gradients':grad,**teacher_stats,**loss_gradients,**item})
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
    memory=torch.cuda.mem_get_info() if torch.cuda.is_initialized() else None
    info=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
    safe_workers=max(1,int((int(info)*2**20-4*2**30)/(pre['peak_gpu_bytes']*1.35+600*2**20)))
    workers=min(workers,safe_workers)
    pending=[(g,f,0) for f in range(1,11) for g in GROUPS if not task_complete(g,f)]
    active={};failures=[];started=time.time();max_active=0;limit=workers;peak=0
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
             'expected_tasks':20,'active_tasks':len(active),'pending_tasks':len(pending),'max_workers_observed':max_active,
             'worker_limit':limit,'elapsed_seconds':time.time()-started,'failures':failures}
        try:
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            peak=max(peak,int(sample[0]));execution['sampled_peak_device_mib']=peak
        except Exception:pass
        atomic_json(RT/'execution.json',execution)
        if active:time.sleep(3)
    assert all(task_complete(g,f) for f in range(1,11) for g in GROUPS),'some formal tasks failed; repair and resume Latest'
    execution.update(status='TRAINED',wall_seconds=time.time()-started);atomic_json(RT/'execution.json',execution)
    manifest=json.loads((HERE/'manifest.json').read_text());manifest.update(status='TRAINED_PENDING_INDEPENDENT_AUDIT',completed_tasks=20,execution=execution);atomic_json(HERE/'manifest.json',manifest)

def read_logs(path):
    if not Path(path).exists():return []
    return [json.loads(v) for v in Path(path).read_text().splitlines()]

def atomic_log(path,rows):
    tmp=Path(path).with_suffix('.tmp')
    with tmp.open('w') as f:
        for v in rows:f.write(json.dumps(v,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def source_maps():
    historical={str(Path(p).resolve()):sha_file(p) for p in runtime().source_manifest()}
    vendor=REPO/'experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment51_dugsa'
    extra=list(REPO.glob('*.py'))+list((REPO/'source').glob('*.py'))+[ADAPTER,TEXT_SOURCE,vendor/'dugsa.py']+list((vendor/'source').glob('*.py'))
    extra+=list((REPO/'data/ABIDE').rglob('*'))
    for root in (E82,E822,E8221,Path('/model/spromptgl/experiment/experiment82_1_pcde_ps')):
        extra+=list(root.glob('*.py'))+[root/f for f in ('config.json','manifest.json','results/report.md','results/audit_status.json','results/fold_metrics.csv') if (root/f).exists()]
        extra+=list((root/'runtime').glob('spectrum_f*.pt'))+list((root/'runtime').glob('prompts.pt'))
        extra+=list((root/'runtime/checkpoints').glob('G2_f*.best.pt'))
    extra+=list((runtime().EXP/'.runtime/checkpoints').glob(f'{BASE_GROUP}_f*.best.pt'))
    extra+=list((runtime().EXP/'.runtime/tasks').glob(f'{BASE_GROUP}_f*.json'))
    for p in extra:
        if p.is_file():historical[str(p.resolve())]=sha_file(p)
    return {'protected':historical,'implementation':{str(HERE/f):sha_file(HERE/f) for f in ('rce_psce.py','run.py','audit.py','config.json')}}

def classification_gradient_audit(out,state,module):
    named=[('darg.'+n,p) for n,p in state['model'].named_parameters() if p.requires_grad]+[('cam.'+n,p) for n,p in state['cam'].named_parameters() if p.requires_grad]+[('psce.'+n,p) for n,p in module.named_parameters()]
    gs=torch.autograd.grad(out['cls'],[p for _,p in named],retain_graph=True,allow_unused=True)
    norms={prefix:math.sqrt(sum(float(g.detach().double().square().sum()) for (n,_),g in zip(named,gs) if g is not None and n.startswith(prefix))) for prefix in ('darg.','cam.','psce.')}
    assert all(v>0 and math.isfinite(v) for v in norms.values()),norms
    return norms

def auxiliary_gradient_audit(out,state,module):
    named=[('darg.'+n,p) for n,p in state['model'].named_parameters() if p.requires_grad]+[('cam.'+n,p) for n,p in state['cam'].named_parameters() if p.requires_grad]+[('psce.'+n,p) for n,p in module.named_parameters()]
    gs=torch.autograd.grad(out['cf_loss']+out['harm'],[p for _,p in named],retain_graph=True,allow_unused=True)
    norms={prefix:math.sqrt(sum(float(g.detach().double().square().sum()) for (n,_),g in zip(named,gs) if g is not None and n.startswith(prefix))) for prefix in ('darg.','cam.','psce.')}
    assert norms['darg.']==norms['cam.']==0 and norms['psce.']>0,norms
    return norms



def e822_trace(group,fold):
    task=json.loads((E822/'runtime/tasks'/f'{group}_f{fold:02d}.json').read_text())
    return {v['epoch']:v for v in task['rng_trace']}


def load_reference_psce():
    # Read-only initialization reconstruction, never a runtime dependency of our method.
    import importlib.machinery
    name='pinned_original_e822_initialization_reference'
    if name not in sys.modules:
        loader=importlib.machinery.SourceFileLoader(name,str(E822/'psce.py'))
        spec=importlib.util.spec_from_loader(name,loader)
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;loader.exec_module(m)
    return sys.modules[name]


def validate_initialization(state,group,module,cache,prompts):
    original=load_reference_psce()
    with isolated_rng(CFG['initialization_seed_base']+state['fold']):
        reference=original.PSCE(cache['spectrum'],prompts,group,state['fold']).to(state['x'].device)
    actual=dict(module.named_parameters());expected=dict(reference.named_parameters())
    assert actual.keys()==expected.keys() and all(torch.equal(actual[k],expected[k]) for k in actual)
    historical=json.loads((E822/'runtime/preflight.json').read_text())['inheritance'][state['fold']-1]['new_parameter_sha']
    assert sha(actual)==sha(expected)==historical
    assert module.counts()['effective_parameters']==6081
    return dict(named_parameters_exact_to_reconstruction=True,sha=historical,
        evidence='original pinned initialization code + seed + saved historical initial SHA; no independent initial tensor file claimed')


def forward(state,module,training=False,intervention='full'):
    module.train(training)
    counts={'output':0,'cam':0,'gcn':[0,0,0,0]};details={};aux={}
    legal=torch.as_tensor(legal_train(state),device=state['x'].device)
    entry=base_runtime().rng_sha(base_runtime().pack_rng())
    teacher=teacher_snapshot(state['cam'].Outputlayer) if training and intervention!='noop' else None
    assert entry==base_runtime().rng_sha(base_runtime().pack_rng())
    def pre(_m,args):
        counts['output']+=1
        assert counts['output']==1 and len(args)==2
        rb=base_runtime().rng_sha(base_runtime().pack_rng())
        changed,d=module(args[0],intervention);details.update(d)
        if teacher is not None:aux.update(auxiliary(module,teacher,args[0],legal,state['y'][legal].detach()))
        assert rb==base_runtime().rng_sha(base_runtime().pack_rng()),'PSCE or teacher consumed shared RNG'
        return changed,args[1]
    handles=[state['cam'].Outputlayer.register_forward_pre_hook(pre),state['cam'].register_forward_hook(lambda *_:counts.__setitem__('cam',counts['cam']+1))]
    for i,gcn in enumerate(state['model'].gcn):
        handles.append(gcn.register_forward_hook(lambda _m,_a,_o,i=i:counts['gcn'].__setitem__(i,counts['gcn'][i]+1)))
    try:out=runtime().forward(state,BASE_GROUP,training=training)
    finally:
        for handle in handles:handle.remove()
    assert counts=={'output':1,'cam':1,'gcn':[1,1,1,1]},counts
    res=(details['delta'][legal].square().sum(-1)/(details['z'][legal].square().sum(-1)+EPS)).mean()
    if intervention=='noop':res=res*0
    out.update(darg_total=out['total'],res=res,cf_loss=aux.get('cf_loss',res*0),harm=aux.get('harm',res*0),details=details,calls=counts)
    out.update(aux)
    out['total']=out['darg_total']+.001*res+.01*out['cf_loss']+.01*out['harm']
    return out


def teacher_summary(cf,d,gain,y):
    valid=cf['valid'];correct=cf['correct'];g=gain.detach()
    out={'teacher_correct_count':int(correct.sum()),'teacher_error_count':int((~correct).sum()),
         'teacher_legal_count':len(y),'teacher_valid_count':int(valid.sum()),'teacher_valid_fraction':float(valid.float().mean()),
         'teacher_error_supervised_count':int((valid&~correct).sum()),'rho_mean':float(cf['rho'].mean()),
         'sign_agreement':float((torch.sign(cf['D8'])==torch.sign(cf['D4'])).float().mean()),
         'stable_positive_fraction':float(((cf['D8']>0)&(cf['D4']>0)).float().mean()),
         'gain_mean':float(g.mean()),'gain_positive_fraction':float((g>1e-6).float().mean()),
         'gain_negative_fraction':float((g< -1e-6).float().mean()),
         'residual_relative_mean':float((d['delta'].norm(dim=-1)/d['z'].norm(dim=-1).clamp_min(EPS)).mean().detach())}
    for lab,name in ((0,'asd'),(1,'td')):
        mask=y==lab
        out[name+'_teacher_error_rate']=float((~correct[mask]).float().mean())
        out[name+'_valid_fraction']=float(valid[mask].float().mean())
        out[name+'_gain_mean']=float(g[mask].mean())
    return out


def loss_gradient_norms(out,module):
    params=tuple(module.parameters());norms={}
    for key in ('cls','cf_loss','harm'):
        gs=torch.autograd.grad(out[key],params,retain_graph=True,allow_unused=True)
        norms[key+'_psce_gradient_norm']=math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
    return norms


def preflight():
    for p in (RT/'tasks',RT/'checkpoints',RT/'logs',RESULTS):p.mkdir(parents=True,exist_ok=True)
    runtime();assert torch.cuda.is_available() and torch.cuda.device_count()==1
    import importlib.machinery
    loader=importlib.machinery.SourceFileLoader('rce_independent_preflight',str(HERE/'audit.py'))
    spec=importlib.util.spec_from_loader(loader.name,loader)
    audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;loader.exec_module(audit)
    manifest=json.loads((HERE/'manifest.json').read_text())
    manifest.update(status='PREFLIGHT_RUNNING',config=CFG,source_maps=source_maps())
    atomic_json(HERE/'manifest.json',manifest)
    inheritance=[];checks=[];diagnostics=[];r=base_runtime()
    # Analytical reliability edge cases are separate from empirical targets.
    d8=torch.tensor([[2.,1.,-1.,0.,1e-3]]);d4=torch.tensor([[1.,-1.,1.,0.,5e-4]])
    rho=reliability(d8,d4)
    assert rho[0,0]>.999999 and rho[0,4]>.999 and torch.count_nonzero(rho[0,1:4])==0
    assert ((rho>=0)&(rho<=1)).all()
    for fold in range(1,11):
        state,warmup,identity=prepare(fold);rng=r.pack_rng();before=r.rng_sha(rng)
        cache=build_fold_cache(state,identity);old=torch_load(E822/'runtime'/f'spectrum_f{fold:02d}.pt')
        assert torch.equal(cache['spectrum']['U'],old['spectrum']['U']) and torch.equal(cache['spectrum']['mu'],old['spectrum']['mu'])
        texts=build_prompt_cache();inits=[];init_checks=[]
        for group in GROUPS:
            module,opt,init=make_module(state,group,cache,texts);inits.append(init)
            init_checks.append(validate_initialization(state,group,module,cache,texts))
            assert torch.equal(module.q0.cpu(),torch_load(E822/'runtime/prompts.pt')['q_'+GROUP_SPECS[group]])
            assert float(module.s_fold)==float(torch.tensor(cache['s_fold']))
            if fold in (1,2):
                initial=bundle(state,module,opt);zero(state,opt)
                head=state['cam'].Outputlayer;hs=sha(cpu_state(head));mode=head.training;rb=r.rng_sha(r.pack_rng())
                teacher=teacher_snapshot(head)
                assert rb==r.rng_sha(r.pack_rng()) and hs==sha(cpu_state(head)) and mode==head.training
                out=forward(state,module,True);d=out['details'];cf=out['cf']
                legal=torch.as_tensor(legal_train(state),device=module.U.device)
                assert d['tokens'].shape==(871,16,32) and d['a'].shape==(871,8,17)
                assert cf['inputs'].shape==(len(legal),33,144) and cf['target'].shape==(len(legal),16)
                assert torch.isfinite(cf['target']).all() and not cf['target'].requires_grad
                assert torch.allclose(cf['target'][cf['valid']].sum(-1),torch.ones_like(cf['margin'][cf['valid']]),atol=1e-5,rtol=0)
                assert (~cf['valid']|cf['correct']).all() and not np.isin(state['test_np'],legal.cpu()).any()
                audit.independent_cf(state,module,d['z'],cf);audit.formulas(module,d)
                ga=classification_gradient_audit(out,state,module);ag=auxiliary_gradient_audit(out,state,module)
                ng=loss_gradient_norms(out,module)
                out['total'].backward();assert finite_grads(state,module)
                grads={n:float(p.grad.norm()) if p.grad is not None else 0 for n,p in module.named_parameters()}
                assert all(v>0 for v in grads.values()),grads
                step(state,opt)
                with torch.no_grad():
                    zm=module.mu.expand(871,-1);zc=counterfactual(teacher,zm,module.U,module.mu,legal,state['y'][legal])
                    assert torch.count_nonzero(zc['D8'])==torch.count_nonzero(zc['D4'])==0
                    empty=auxiliary(module,teacher,zm,legal,state['y'][legal])
                    assert float(empty['cf_loss'])==float(empty['harm'])==0 and torch.isfinite(empty['cf_loss'])
                    if fold==2:assert torch.count_nonzero(d['delta'][[376,406]])==0 and not np.isin([376,406],legal.cpu()).any()
                checks.append({'group':group,'fold':fold,'classification':ga,'auxiliary':ag,'gradients':grads,'loss_gradient_norms':ng,'teacher_frozen':all(p.grad is None for p in teacher.parameters())})
                load_bundle(state,module,opt,initial);r.restore_rng(rng);del out,teacher
            del module,opt
        assert len(set(inits))==1 and r.rng_sha(r.pack_rng())==before
        spectral_check=audit.spectral_audit(cache,state);scale_check=audit.scale_audit(state,cache)
        diagnostics.append(audit.historical_diagnostic(state,cache,texts))
        assert r.rng_sha(r.pack_rng())==before
        inheritance.append({'fold':fold,'identity':identity,'warmup_rng_sha':before,'new_parameter_sha':inits[0],
            'spectra_sha':cache['spectra_sha'],'legal_train_sha':cache['legal_train_sha'],'legal_train_count':len(cache['legal_train']),
            'U_sha':sha(cache['spectrum']['U']),'e822_U_mu_exact':True,'initialization_checks':init_checks,
            's_fold':cache['s_fold'],'scale_audit':scale_check,'spectral_audit':spectral_check})
        del state,cache;torch.cuda.empty_cache();print(f'PREFLIGHT fold={fold} PASS',flush=True)
    atomic_json(RT/'historical_readonly_diagnostic.json',diagnostics)
    state,warmup,identity=prepare(1);module,opt,_=make_module(state,'G2');initial=bundle(state,module,opt);rng=r.pack_rng()
    zero(state,opt);o=runtime().forward(state,BASE_GROUP,training=True);o['total'].backward()
    expected_logp=o['logp'].detach().clone();losses={k:o[k].detach().clone() for k in ('cls','graph','pair','adv','preserve','total')}
    expected_grad=sha({k:{n:p.grad for n,p in state[k].named_parameters()} for k in ('model','cam')})
    for k in ('opt_base','opt_cam','opt_added'):state[k].step()
    expected=runtime().load_history().state_bundle(state);expected_rng=r.rng_sha(r.pack_rng());del o
    load_bundle(state,module,opt,initial);r.restore_rng(rng);zero(state,opt);o=forward(state,module,True,'noop')
    assert torch.equal(o['logp'],expected_logp) and all(torch.equal(o[k],v) for k,v in losses.items())
    o['total'].backward();assert expected_grad==sha({k:{n:p.grad for n,p in state[k].named_parameters()} for k in ('model','cam')})
    for k in ('opt_base','opt_cam','opt_added'):state[k].step()
    assert sha(expected)==sha(runtime().load_history().state_bundle(state)) and expected_rng==r.rng_sha(r.pack_rng());del o
    raw=torch.stack([torch.zeros(144),torch.ones(144)*1e-10,torch.ones(144)*.01,torch.ones(144)*1e3]).double().requires_grad_()
    zz=torch.arange(144,dtype=torch.float64)[None].expand(4,-1)/144;bounded=smooth_limit(raw,zz)
    assert torch.count_nonzero(bounded[0])==0 and torch.all(bounded.norm(dim=-1)<=.08*zz.norm(dim=-1)+1e-12)
    bounded.sum().backward();assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[0],torch.ones(144,dtype=torch.float64))
    load_bundle(state,module,opt,initial);r.restore_rng(rng)
    def next_update():
        zero(state,opt);o=forward(state,module,True);o['total'].backward();step(state,opt);del o
        with torch.no_grad():o=forward(state,module,False)
        return sha(bundle(state,module,opt)),r.rng_sha(r.pack_rng()),sha(o['logp'])
    next_update();file=RT/'recovery_check.pt';atomic_torch(file,{'current':bundle(state,module,opt),'rng':r.pack_rng()})
    a=next_update();disk=torch_load(file);load_bundle(state,module,opt,disk['current']);r.restore_rng(disk['rng']);b=next_update()
    assert a==b;file.unlink()
    load_bundle(state,module,opt,initial);r.restore_rng(rng);reference={t['epoch']:t for t in historical_task(1)['rng_trace']}
    trajectories=[]
    for epoch in (11,12):
        before=r.rng_sha(r.pack_rng());zero(state,opt);o=forward(state,module,True);o['total'].backward();step(state,opt);del o
        after=r.rng_sha(r.pack_rng())
        with torch.no_grad():o=forward(state,module,False)
        item=dict(epoch=epoch,rng_before=before,rng_after_train=after,rng_after_eval=r.rng_sha(r.pack_rng()))
        assert item==reference[epoch]==e822_trace('G2',1)[epoch],item
        trajectories.append(item);del o
    pre={'status':'PASS','source_sha':implementation_sha(),'inheritance':inheritance,'group_checks':checks,
        'noop_forward_losses_gradients_three_optimizers_rng_exact':True,'next_update_disk_restore_four_optimizers_rng_exact':True,
        'teacher_33_inputs_independent_reconstruction':True,'auxiliary_gradients_isolated':True,'two_actual_historical_rng_triplets':trajectories,
        'initialization_exact_original_reconstruction_and_historical_SHA':True,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()),
        'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name(0)}
    atomic_json(RT/'preflight.json',pre)
    assert all(sha_file(p)==h for p,h in manifest['source_maps']['protected'].items())
    manifest.update(status='PREFLIGHT_PASS',preflight=pre,source_maps=source_maps());atomic_json(HERE/'manifest.json',manifest)
    print('PREFLIGHT PASS',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','all','worker']);p.add_argument('--workers',type=int,default=10);p.add_argument('--group',choices=GROUPS);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='all':run_all(a.workers)
    else:train_worker(a.group,a.fold)

if __name__=='__main__':main()

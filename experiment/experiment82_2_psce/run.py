#!/usr/bin/env python3
"""E82.2 PSCE: locked DARG inheritance; 20 formally registered tasks."""
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
from psce import PSCE, GROUPS as GROUP_SPECS, PROMPTS, EPS, build_spectra, text_geometry, smooth_limit, teacher_snapshot, teacher_logits, counterfactual, cf_kl

HERE = Path(__file__).resolve().parent
REPO = Path('/data/adodas/SPromptGL-main')
CFG = json.loads((HERE/'config.json').read_text())
RT, RESULTS = HERE/'runtime', HERE/'results'
GROUPS = tuple(CFG['groups'])
BASE_GROUP = CFG['base_group']
METRICS = ('acc', 'hard_auc', 'prob_auc', 'sen', 'spe', 'asd_f1')
BASE = None
ADAPTER = REPO/'experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py'
E82 = REPO/'experiment/experiment82_pssec'
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
    for p in list(E82.glob('*.py'))+[E82/'config.json',E82/'manifest.json',E82/'results/report.md',E82/'results/fold_metrics.csv',E82/'runtime/prompts.pt']+list((E82/'runtime').glob('spectrum_f*.pt'))+list((E82/'runtime/checkpoints').glob('G2_f*.best.pt')):
        historical[str(p.resolve())]=sha_file(p)
    for p in list(Path('/model/spromptgl/experiment/experiment82_1_pcde_ps').glob('*.py'))+[Path('/model/spromptgl/experiment/experiment82_1_pcde_ps')/f for f in ('config.json','manifest.json','results/report.md','results/audit_status.json')]:historical[str(p.resolve())]=sha_file(p)
    code={str(HERE/f):sha_file(HERE/f) for f in ('run.py','psce.py','audit.py','config.json')}
    return {'protected':historical,'implementation':code}


def implementation_sha():
    return sha({f:sha_file(HERE/f) for f in ('run.py','psce.py','config.json')})


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
    return payload


def build_fold_cache(state,identity):
    fold=state['fold'];path=RT/f'spectrum_f{fold:02d}.pt'
    if path.exists():
        c=torch_load(path);assert c['identity']==identity;return c
    with isolated_rng():
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
        c={'fold':fold,'identity':identity,'legal_train':idx,'legal_train_sha':sha(idx),'legal_label_sha':sha(labels),
            'warmup_cam':z,'spectrum':spectrum,'spectra_sha':sha(spectrum),'feature_sha':sha(z),
            'repeat_projector_max_abs':float((spectrum['U']@spectrum['U'].T-repeat['U']@repeat['U'].T).abs().max()),
            'e82_original_U_sha':sha(old['spectra']['O']['U']),
            'warmup_graph_sha':[sha(x) for x in out['original_adjs']],
            'warmup_unified_graph_sha':[sha(x) for x in out['unified_adjs']]}
        atomic_torch(path,c)
    return c


def make_module(state,group,cache=None,prompts=None):
    cache=torch_load(RT/f"spectrum_f{state['fold']:02d}.pt") if cache is None else cache
    prompts=torch_load(RT/'prompts.pt') if prompts is None else prompts
    with isolated_rng(CFG['initialization_seed_base']+state['fold']):
        module=PSCE(cache['spectrum'],prompts,group,state['fold']).to(state['x'].device)
    param_sha=sha(dict(module.named_parameters()))
    opt=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
    return module,opt,param_sha



def forward(state,module,training=False,intervention='full'):
    module.train(training)
    counts={'output':0,'cam':0,'gcn':[0,0,0,0]};details={}
    legal=torch.as_tensor(legal_train(state),device=state['x'].device)
    labels=state['y'][legal].detach()
    teacher=teacher_snapshot(state['cam'].Outputlayer) if training and intervention!='noop' else None
    def pre(_m,args):
        counts['output']+=1
        assert counts['output']==1 and len(args)==2 and args[0].shape==(871,144)
        changed,d=module(args[0],intervention=intervention);details.update(d)
        if teacher is not None:
            rng=base_runtime().rng_sha(base_runtime().pack_rng())
            cf=counterfactual(teacher,args[0],module.U,module.mu,legal,labels)
            assert rng==base_runtime().rng_sha(base_runtime().pack_rng()),'counterfactual consumed RNG'
            details['cf']=cf;details['cf_loss']=cf_kl(d['w_sem'],legal,cf)
        return (changed,args[1])
    handles=[state['cam'].Outputlayer.register_forward_pre_hook(pre),
        state['cam'].register_forward_hook(lambda *_:counts.__setitem__('cam',counts['cam']+1))]
    for i,gcn in enumerate(state['model'].gcn):
        handles.append(gcn.register_forward_hook(lambda _m,_a,_o,i=i:counts['gcn'].__setitem__(i,counts['gcn'][i]+1)))
    try:out=runtime().forward(state,BASE_GROUP,training=training)
    finally:
        for handle in handles:handle.remove()
    del teacher
    assert counts=={'output':1,'cam':1,'gcn':[1,1,1,1]},counts
    res=(details['delta'][legal].square().sum(-1)/(details['z'][legal].square().sum(-1)+EPS)).mean()
    cf=details.get('cf_loss',details['w_sem'].sum()*0)
    out.update(darg_total=out['total'],res=res,cf=cf,details=details,calls=counts)
    out['total']=out['darg_total']+CFG['residual_loss_coefficient']*res+CFG['cf_loss_coefficient']*cf
    return out


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

def aux_gradients(out,module):
    active=[(n,p) for n,p in module.named_parameters() if not n.startswith('spectral.')]
    gs=torch.autograd.grad(out['cf'],[p for n,p in active],retain_graph=True,allow_unused=True)
    norms={n:float(g.detach().norm().cpu()) if g is not None else 0. for (n,p),g in zip(active,gs)}
    return norms,math.sqrt(sum(v*v for v in norms.values()))

def classification_gradient_audit(out,state,module):
    active=list(module.named_parameters())
    grads=torch.autograd.grad(out['cls'],[p for n,p in active]+[out['details']['z'],out['fused']],retain_graph=True,allow_unused=True)
    norms={n:float(g.detach().norm().cpu()) if g is not None else 0. for (n,p),g in zip(active,grads)}
    norms.update(cam144=float(grads[-2].norm().cpu()),upstream_darg320=float(grads[-1].norm().cpu()))
    assert all(math.isfinite(v) and v>0 for v in norms.values()),norms
    ev,total=aux_gradients(out,module)
    if out['details']['cf']['valid'].any():
        assert all(ev[n]>0 for n in ('key.weight','direction.weight','evidence.weight')),ev
    assert not out['details']['cf']['target'].requires_grad
    rg=torch.autograd.grad(out['res'],[module.key.weight,out['details']['z']],retain_graph=True)
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in rg)
    return {'classification':norms,'counterfactual':ev,'cf_norm':total,'residual_grad_norms':[float(g.norm()) for g in rg]}


def teacher_summary(cf):
    return {'teacher_correct_fraction':float(cf['correct'].float().mean()),'teacher_valid_fraction':float(cf['valid'].float().mean()),
        'positive_directions_mean':float((cf['D']>1e-6).float().sum(-1).mean()),
        'negative_directions_mean':float((cf['D']< -1e-6).float().sum(-1).mean()),
        'mask_relative_mean':float(cf['mask_relative'].mean()),'mask_relative_max':float(cf['mask_relative'].max())}


def preflight():
    for p in (RT/'tasks',RT/'checkpoints',RT/'logs',RESULTS):p.mkdir(parents=True,exist_ok=True)
    runtime();assert torch.cuda.is_available() and torch.cuda.device_count()==1
    manifest=json.loads((HERE/'manifest.json').read_text())
    sources=json.loads((RT/'source_review.json').read_text())
    external={k:{a:b for a,b in v.items() if a!='text'} for k,v in sources.items()}
    external['github']={k:{a:b for a,b in v.items() if a!='text'} for k,v in sources['github'].items()}
    manifest.update(status='PREFLIGHT_RUNNING',config=CFG,external_references=external,source_maps=source_maps(),
        decisions={'base':'Only E74.4 G2 path, initialized exclusively from E61 epoch10 complete state.',
        'historical_path':'Actual vendor PLNNoPEN InputLayer -> Encoder/FeedForward -> flatten CAM144 -> original OutputLayer; no SPGS/PEN reintroduction.',
        'output':'Original LayerNorm(144), Linear(144,36), GELU, Dropout, Linear(36,2), log_softmax. Eval teacher clone reuses mlp_head and classifier to obtain pre-softmax logits.',
        'spectral':'Exact E82 Original O algorithm in CPU float64, PCA8 plus orthogonal covariance-difference8; frozen.',
        'ssc':'Reference DBPC distributed prompt budgets, SKP text spectral energy, redundancy and entropy fusion; PSCE directions are patient CAM spectrum, not SSC text spectrum.',
        'text':'Exact original eight RN50 embeddings; reuse audited fixed common text geometry rule, no trainable text projection. Only q0 differs between groups.',
        'target':'ReLU(D)/(sum ReLU(D)+EPS), detached; valid-target mass audited against EPS regularization. Tolerance fixed 1e-6.',
        'teacher':'Deepcopy current head once before each training forward; eval, requires_grad False, no_grad. Hook-free, label access limited to legal training. No extra original head call.',
        'limitations':'Controlled bounded feature masking, not full removal or medical causal effects. Current-model teacher may induce self-distillation; trainable key/token may absorb query rotations.'})
    atomic_json(HERE/'manifest.json',manifest)
    inheritance=[];checks=[];r=base_runtime()
    for fold in range(1,11):
        state,warmup,identity=prepare(fold);rng=r.pack_rng();before=r.rng_sha(rng)
        cache=build_fold_cache(state,identity);texts=build_prompt_cache();inits=[]
        for group in GROUPS:
            module,opt,init=make_module(state,group,cache,texts);inits.append(init)
            if fold==1 or fold==2:
                initial=bundle(state,module,opt);zero(state,opt);head=state['cam'].Outputlayer
                headsha=sha(cpu_state(head));mode=head.training;rb=r.rng_sha(r.pack_rng());teacher=teacher_snapshot(head)
                assert sha(cpu_state(head))==headsha and mode==head.training and rb==r.rng_sha(r.pack_rng())
                assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
                out=forward(state,module,True);d=out['details'];cf=d['cf'];legal=torch.as_tensor(legal_train(state),device=module.U.device)
                assert d['tokens'].shape==(871,16,32) and d['a'].shape==(871,8,17) and cf['margins'].shape==(len(legal),17)
                assert all(torch.isfinite(x).all() for x in d.values() if torch.is_tensor(x))
                assert torch.allclose(cf['target'][cf['valid']].sum(-1),torch.ones_like(cf['margins'][cf['valid'],0]),atol=2e-6,rtol=0)
                oldz=d['z'].detach();br=r.rng_sha(r.pack_rng())
                with torch.no_grad():
                    z=oldz[legal];logits=teacher_logits(teacher,z);lp,_=teacher(z,None)
                    assert torch.allclose(logits[:,0]-logits[:,1],lp[:,0]-lp[:,1],atol=1e-6,rtol=1e-6)
                    cfzero=counterfactual(teacher,oldz,module.U,oldz.mean(0),legal,state['y'][legal])
                    for lab in (0,1):
                        mask=state['y'][legal]==lab
                        assert torch.allclose(cf['margins'][mask,0],logits[mask,lab]-logits[mask,1-lab],atol=1e-6,rtol=1e-6)
                    repeated=teacher_logits(teacher,torch.cat([z,z],0));assert torch.equal(repeated[:len(z)],repeated[len(z):])
                    assert torch.count_nonzero((repeated[:len(z)]-repeated[len(z):])[:,0])==0
                    zmean=module.mu.expand(871,-1);zerocf=counterfactual(teacher,zmean,module.U,module.mu,legal,state['y'][legal])
                    assert torch.count_nonzero(zerocf['D'])==0 and not zerocf['valid'].any()
                assert br==r.rng_sha(r.pack_rng()) and cf['mask_relative'].max()<=.080001
                assert torch.equal(cf['D']>0,(cf['margins'][:,:1]>cf['margins'][:,1:]))
                fake=dict(cf);fake['valid']=torch.zeros_like(cf['valid']);loss0=cf_kl(d['w_sem'],legal,fake)
                assert loss0.requires_grad and float(loss0.detach())==0
                ga=classification_gradient_audit(out,state,module);out['total'].backward();assert finite_grads(state,module);step(state,opt)
                if fold==2:
                    assert torch.count_nonzero(d['delta'][[376,406]])==0 and not np.isin([376,406],cache['legal_train']).any()
                checks.append({'fold':fold,'group':group,'init_sha':init,'calls':out['calls'],'gradients':ga,**teacher_summary(cf)})
                del out,d,cf
                with torch.no_grad():out=forward(state,module,False)
                held=np.setdiff1d(state['test_np'],legal_train(state));oldy=state['y'].clone();oldnp=state['labels_np'].copy()
                state['y'][held]=1-state['y'][held];state['labels_np'][held]=1-state['labels_np'][held]
                try:
                    with torch.no_grad():changed=forward(state,module,False)
                    assert torch.equal(out['logp'],changed['logp'])
                finally:state['y'].copy_(oldy);state['labels_np'][:]=oldnp
                # Original head and copied teacher RNG states are independent of labels.
                saved=sha(cpu_state(head));br=r.rng_sha(r.pack_rng());mode=head.training
                t=teacher_snapshot(head)
                with torch.no_grad():counterfactual(t,out['details']['z'],module.U,module.mu,legal,state['y'][legal])
                assert saved==sha(cpu_state(head)) and mode==head.training and br==r.rng_sha(r.pack_rng())
                load_bundle(state,module,opt,initial);r.restore_rng(rng);del out,changed,teacher,t
            del module,opt
        assert len(set(inits))==1 and r.rng_sha(r.pack_rng())==before
        inheritance.append({'fold':fold,'identity':identity,'new_parameter_sha':inits[0],'spectra_sha':cache['spectra_sha'],
            'legal_train_sha':cache['legal_train_sha'],'legal_train_count':len(cache['legal_train']),
            'U_sha':sha(cache['spectrum']['U']),'e82_U_exact':True,'orthogonality':cache['spectrum']['orthogonality_max_abs']})
        del state,cache;torch.cuda.empty_cache();print(f'PREFLIGHT fold={fold} PASS',flush=True)
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
    z=torch.arange(144,dtype=torch.float64)[None].expand(4,-1)/144;bounded=smooth_limit(raw,z)
    assert torch.count_nonzero(bounded[0])==0 and torch.all(bounded.norm(dim=-1)<=.08*z.norm(dim=-1)+1e-12)
    bounded.sum().backward();assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[0],torch.ones(144,dtype=torch.float64))
    load_bundle(state,module,opt,initial);r.restore_rng(rng);zero(state,opt);o=forward(state,module,True);o['total'].backward();step(state,opt);del o
    file=RT/'recovery_check.pt';atomic_torch(file,{'current':bundle(state,module,opt),'rng':r.pack_rng()})
    def next_update():
        zero(state,opt);o=forward(state,module,True);o['total'].backward();step(state,opt)
        with torch.no_grad():o=forward(state,module,False)
        return sha(bundle(state,module,opt)),r.rng_sha(r.pack_rng()),sha(o['logp'])
    a=next_update();disk=torch_load(file);load_bundle(state,module,opt,disk['current']);r.restore_rng(disk['rng']);b=next_update()
    assert a==b;file.unlink()
    pre={'status':'PASS','source_sha':implementation_sha(),'inheritance':inheritance,'group_checks':checks,
        'noop_forward_losses_gradients_three_optimizers_rng_exact':True,'next_update_disk_restore_four_optimizers_rng_exact':True,
        'teacher_snapshot_eval_frozen_no_rng_no_original_mutation':True,'teacher_17_batches_only_legal_rows':True,
        'zero_mask_margin_exact_zero':True,'ASD0_TD1_margin_sign_checked':True,'test_label_swap_prediction_exact':True,
        'smooth_limit_zero_identity_gradient_and_bound':True,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()),
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
    atomic_json(RT/'preflight.json',pre);manifest.update(status='PREFLIGHT_PASS',preflight=pre,source_maps=source_maps());atomic_json(HERE/'manifest.json',manifest)
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
          'new_parameter_sha':init_sha,'spectra_sha':cache['spectra_sha'],'text_identity_sha':torch_load(RT/'prompts.pt')['identity_sha'],'config':CFG,
          'spectrum':'O','counts':module.counts(),'text_kind':GROUP_SPECS[group]}
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
        cfgrads,cfgnorm=aux_gradients(out,module)
        teacher_stats=teacher_summary(out['details']['cf'])
        out['total'].backward();assert finite_grads(state,module)
        grad=gradients(state,module);step(state,opt)
        after=r.rng_sha(r.pack_rng())
        losses={k:float(out[k].detach().cpu()) for k in ('cls','graph','pair','adv','preserve','darg_total','res','cf','total')}
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
                'rng':r.pack_rng(),'rng_sha':after_eval,'trace':trace.copy(),
                'wait':0,'next_epoch':executed,'best_acc':best_acc,'best_meta':best_meta,'bundle_sha':sha(bundle(state,module,opt))})
        else:wait+=1
        log_row(log,{'epoch':executed,'acc':float(metrics['acc']),'best_acc':best_acc,'best_epoch':best_epoch,
             'improved':improved,'wait':wait,'losses':losses,'gradients':grad,'cf_gradient_norm':cfgnorm,**teacher_stats,**item})
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
        if active:time.sleep(2)
    assert all(task_complete(g,f) for f in range(1,11) for g in GROUPS),'some formal tasks failed; repair and resume Latest'
    execution.update(status='TRAINED',wall_seconds=time.time()-started);atomic_json(RT/'execution.json',execution)
    manifest=json.loads((HERE/'manifest.json').read_text());manifest.update(status='TRAINED_PENDING_INDEPENDENT_AUDIT',completed_tasks=20,execution=execution);atomic_json(HERE/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','all','worker']);p.add_argument('--workers',type=int,default=10);p.add_argument('--group',choices=GROUPS);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='all':run_all(a.workers)
    else:train_worker(a.group,a.fold)


if __name__=='__main__':main()

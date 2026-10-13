"""Independent, resumable four-config / one-fold AGESA pilot runner."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import difflib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from build_roi_semantics import (EXP, ROOT, EXTRA, atomic_json, canonical_sha,
                                sha256, validate_inputs, build, encode, embedding_diagnostics)

CONFIGS = {'F0': 'OFF', 'F1': 'REAL', 'F2': 'SHUFFLED', 'F3': 'ID_ONLY'}
GPU_ASSIGNMENT = {'F0': 0, 'F1': 1, 'F2': 0, 'F3': 1}
PROTOCOL = dict(datadir=str(ROOT/'data')+'/', datname='ABIDE', cpu=False, gpu='-1',
                nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
                n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode='weighted-cosine', MP_mode='GCN', MF_mode='concat',
                alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
                nclass=2, mode='simple-2', seed=0)
SPLIT = dict(outer_n_splits=10, outer_random_state=10, outer_shuffle=True,
             outer_fold_one_based=1, inner_validation_fraction=.15,
             inner_random_state=0, inner_stratified=True)
MANIFEST = EXP/'manifest.json'
_CTX = None


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def history_digest(path):
    h, count, size = hashlib.sha256(), 0, 0
    for p in sorted(Path(path).rglob('*')):
        if p.is_file():
            h.update((p.relative_to(path).as_posix()+'\0'+sha256(p)+'\n').encode())
            count += 1
            size += p.stat().st_size
    return dict(sha256=h.hexdigest(), files=count, bytes=size)


def integrity(full=False):
    m = read_json(MANIFEST)
    for rel, digest in m['protected_hashes'].items():
        if sha256(ROOT/rel) != digest:
            raise RuntimeError('protected input/source changed: '+rel)
    if full:
        for name, digest in m['protected_history'].items():
            if history_digest(ROOT/'experiment'/name) != digest:
                raise RuntimeError('historical experiment changed: '+name)
    if sha256(Path.home()/'.cache/clip/RN50.pt') != m['clip_weight_sha256']:
        raise RuntimeError('CLIP weights changed')
    return True


def create_patch():
    original = (ROOT/'model.py').read_text()
    patched = original.replace('from utils import *\n',
                               'from utils import *\nfrom roi_semantic_adapter import AGESA\n',1)
    anchor = '        self.ModalFusion.apply(my_weight_init)\n'
    assert patched.count(anchor) == 1
    patched = patched.replace(anchor, anchor + '''        self.roi_adapter = None
        if hyperpm.roi_condition != "OFF":
            # Register after all original initialization; preserve CPU/CUDA RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(120000 + hyperpm.seed)
                self.roi_adapter = AGESA(hyperpm.roi_embeddings, hyperpm.roi_indices).to(dev)
            assert sum(p.numel() for p in self.roi_adapter.parameters()) == 32865
            self.optimizer_MF.add_param_group({"params": self.roi_adapter.parameters(),
                                               "lr": hyperpm.lr, "weight_decay": 0.0})
''',1)
    anchor = '            prob,_,_ = self.ModalFusion(feat)\n'
    assert patched.count(anchor) == 1
    patched = patched.replace(anchor, '''            if self.roi_adapter is not None:
                feat, _ = self.roi_adapter(self.feat[:,64:320], feat)
''' + anchor,1)
    compile(patched,'isolated_model.py','exec')
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True),
                                       patched.splitlines(keepends=True),
                                       fromfile='a/model.py',tofile='b/model.py'))
    (EXP/'integration_patch.diff').write_text(patch)
    runtime = EXP/'_runtime'
    runtime.mkdir(exist_ok=True)
    (runtime/'model.py').write_text(patched)
    return sha256(EXP/'integration_patch.diff')


def splits(labels):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    outer = StratifiedKFold(n_splits=10,shuffle=True,random_state=10)
    outer_train,test = next(outer.split(np.zeros((len(labels),1)),labels))
    train,val = train_test_split(outer_train,test_size=.15,random_state=0,
                                 stratify=labels[outer_train])
    train, val = np.sort(train),np.sort(val)
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert len(train)+len(val)+len(test) == 871
    summary = {name:dict(n=len(indices),ASD=int((labels[indices]==0).sum()),
                         TD=int((labels[indices]==1).sum()),
                         row_index_sha256=canonical_sha(indices.tolist()))
               for name,indices in [('train',train),('validation',val),('test',test)]}
    return train,val,test,summary


def prepare():
    for sub in ['cache','results','checkpoints','logs']:
        (EXP/sub).mkdir(exist_ok=True)
    if not MANIFEST.exists():
        files=['network.py','model.py','layers.py','main.py','utils.py',
               'ABIDE-simple-2-concat-weighted-cosine.sh',
               'data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']
        files += ['experiment/experiment11_roi_mapping_recovery/results/'+p for p in
                  ['roi_111_labels.csv','roi_256_named_mapping.csv','roi_256_mapping.csv']]
        atomic_json(MANIFEST,dict(experiment='Experiment 12 AGESA Feasibility',
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    started_utc=utc(),run_status='AUDITED',tasks={},
                    protected_hashes={p:sha256(ROOT/p) for p in files},
                    protected_history={p.name:history_digest(p) for p in (ROOT/'experiment').glob('experiment*')
                                       if p.is_dir() and p!=EXP},
                    clip_weight_sha256=sha256(Path.home()/'.cache/clip/RN50.pt')))
    integrity(full=True)
    prompts = build()
    cache = encode(prompts)
    patch_hash = create_patch()
    _,_,labels = validate_inputs()
    _,_,_,counts = splits(labels)
    m = read_json(MANIFEST)
    m.update(protocol=PROTOCOL, split_protocol=SPLIT, split_counts=counts,
             configs=CONFIGS,gpu_assignment=GPU_ASSIGNMENT,
             prompts_sha256=sha256(EXP/'prompts.json'), patch_sha256=patch_hash,
             text_cache_sha256=sha256(EXP/'cache/roi_text_embeddings.pt'),
             adapter_sha256=sha256(EXP/'roi_semantic_adapter.py'),
             roi_mapping_verified=256,valid_semantic_edges=252,extra_semantic_edges=EXTRA,
             adapter_parameters=32865,architecture=dict(text_projector=[1024,32],
             value_projector=[3,32],bias=False,phi=['z','abs(z)','sign(z)'],
             alpha_shared_scalar=True,alpha_initial=0.,residual_scale=.1),
             initialization_policy='P0 RN50 cache loaded before common seed; new adapter fork_rng seed 120000',
             checkpoint_policy='strict inner-validation ACC improvement; first maximum retained; patience wait>200',
             probability_auc_positive_class='ASD=0; exp(logsoftmax)[:,0]',
             prepared_utc=utc())
    if m['run_status'] in ['AUDITED','PREPARED']:
        m['run_status']='PREPARED'
    atomic_json(MANIFEST,m)
    print('PREPARED: 256 verified indices; 252 anatomical texts; 4 masked edges; RN50 cache saved',flush=True)
    print('splits',counts,flush=True)
    return cache


def signature():
    m=read_json(MANIFEST)
    return canonical_sha({k:m[k] for k in ['protected_hashes','prompts_sha256','text_cache_sha256',
                                          'adapter_sha256','patch_sha256','protocol','split_protocol']})


def context():
    global _CTX
    if _CTX is not None:
        return _CTX
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(ROOT))
    import network
    import model as original_model
    cache=torch.load(EXP/'cache/roi_text_embeddings.pt',map_location='cpu',weights_only=False)
    assert network.attr_words == cache['texts']['P0']
    # Common cached P0 setup before seeding model initialization, never every epoch.
    network._prompt_text_features=cache['p0_features'].to('cuda')
    create_patch()
    spec=importlib.util.spec_from_file_location('agesa_isolated_model',EXP/'_runtime/model.py')
    patched=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    frame,modals,labels=validate_inputs()
    train,val,test,summary=splits(labels)
    _CTX=dict(torch=torch,network=network,original=original_model,patched=patched,cache=cache,
              features=frame.iloc[:,:-1].values,labels=labels,train=train,val=val,test=test,
              dims=[len(v) for v in modals.values()],split_counts=summary)
    return _CTX


def seed_rng(seed=0):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def make_agent(config, original=False):
    c=context()
    seed_rng()
    hp=SimpleNamespace(**PROTOCOL,roi_condition=CONFIGS[config],
                       roi_embeddings=c['cache']['banks'].get(CONFIGS[config]),
                       roi_indices=c['cache']['valid_feature_indices'])
    cls=(c['original'] if original else c['patched']).EvalHelper
    with contextlib.redirect_stdout(io.StringIO()):
        agent=cls(c['dims'],c['features'],c['labels'],hp,c['train'],c['val'])
    if original:
        agent.roi_adapter=None
    # True held-out labels remain solely in c['labels'], outside the training helper.
    agent.targ.fill_(-100)
    agent.targ[c['train']]=c['torch'].as_tensor(c['labels'][c['train']],device='cuda')
    agent.tst_idx=c['test']
    agent.MF_sav.close()
    agent.GCMP_sav.close()
    return agent


def model_modules(agent):
    modules=[agent.MessagePassing0,agent.ModalFusion]
    if agent.roi_adapter is not None: modules.append(agent.roi_adapter)
    return modules


def model_hash(agent, branch=False):
    h=hashlib.sha256()
    modules=[agent.roi_adapter] if branch else [agent.MessagePassing0,agent.ModalFusion]
    for mod in modules:
        if mod is None: continue
        for name,p in sorted(mod.named_parameters()):
            h.update(name.encode()+p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def forward(agent,condition=None):
    import torch
    for mod in model_modules(agent): mod.eval()
    with torch.no_grad():
        n,parts=agent.dataLayer(agent.feat)
        h,adj,aloss=agent.MessagePassing0(parts,n,agent.targ,agent.trn_idx)
        if agent.roi_adapter is None:
            enhanced,delta=h,torch.zeros_like(h[:,64:320])
        else:
            e=None if condition is None or condition=='OFF' else context()['cache']['banks'][condition].to('cuda')
            enhanced,delta=agent.roi_adapter(agent.feat[:,64:320],h,e,off=condition=='OFF')
        output,hidden,_=agent.ModalFusion(enhanced)
        assert torch.isfinite(output).all() and torch.isfinite(enhanced).all()
        return dict(output=output,probability=output.exp(),h=h,enhanced=enhanced,delta=delta,hidden=hidden,aloss=aloss)


def metrics(output,indices):
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix
    c=context()
    y=c['labels'][indices]
    p=output.detach().exp().cpu().numpy()[indices]
    pred=p.argmax(1)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return dict(acc=float((pred==y).mean()),ba=float(balanced_accuracy_score(y,pred)),
                probability_auc=float(roc_auc_score(y==0,p[:,0])),
                sensitivity_asd=float(cm[0,0]/cm[0].sum()),
                specificity_td=float(cm[1,1]/cm[1].sum()),
                asd_tp=int(cm[0,0]),asd_fn=int(cm[0,1]),td_fp=int(cm[1,0]),td_tn=int(cm[1,1]),
                n=len(indices))


def grad_norm(p):
    import torch
    if p.grad is None: return 0.
    if not torch.isfinite(p.grad).all(): raise RuntimeError('nonfinite gradient')
    return float(p.grad.norm().item())


def gradients(agent):
    d=dict(classifier=grad_norm(agent.ModalFusion.Outputlayer.classifier.weight))
    if agent.roi_adapter is not None:
        d.update(alpha=grad_norm(agent.roi_adapter.alpha),
                 project_text=grad_norm(agent.roi_adapter.project_text.weight),
                 project_value=grad_norm(agent.roi_adapter.project_value.weight))
    for module in model_modules(agent):
        for p in module.parameters():
            if p.grad is not None and not context()['torch'].isfinite(p.grad).all():
                raise RuntimeError('nonfinite model gradient')
    return d


def model_state(agent):
    return {name:{k:v.detach().cpu().clone() for k,v in mod.state_dict().items()}
            for name,mod in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion),
                             ('adapter',agent.roi_adapter)] if mod is not None}


def load_model(agent,state):
    agent.MessagePassing0.load_state_dict(state['gfr'])
    agent.ModalFusion.load_state_dict(state['pln'])
    if agent.roi_adapter is not None: agent.roi_adapter.load_state_dict(state['adapter'])


def atomic_torch(path,data):
    import torch
    tmp=path.with_suffix('.tmp')
    torch.save(data,tmp)
    tmp.replace(path)


def rng_state():
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([p.cpu() for p in state['cuda']])


def optimizer_audit(agent):
    groups=agent.optimizer_MP0.param_groups+agent.optimizer_MF.param_groups
    ids=[id(p) for g in groups for p in g['params']]
    actual=[id(p) for mod in model_modules(agent) for p in mod.parameters()]
    assert len(ids)==len(set(ids)) and set(ids)==set(actual)
    assert all(g['lr']==.0038 for g in groups)
    assert agent.optimizer_MP0.param_groups[0]['weight_decay']==.11
    assert agent.optimizer_MF.param_groups[0]['weight_decay']==.11
    if agent.roi_adapter is not None:
        assert agent.optimizer_MF.param_groups[1]['weight_decay']==0
    return dict(no_duplicate_parameters=True,no_missing_parameters=True,
                parameter_count=sum(p.numel() for mod in model_modules(agent) for p in mod.parameters()))


def preflight():
    import torch
    integrity(full=True)
    c=context()
    base=make_agent('F0',original=True)
    base_hash=model_hash(base)
    reference=forward(base)
    checks=dict(input_shape=list(base.feat.shape),raw_correlation_shape=list(base.feat[:,64:].shape),
                gfr_shape=list(reference['h'].shape),pln_output_shape=list(reference['output'].shape),
                hidden_shape=list(reference['hidden'].shape),backbone_parameters=optimizer_audit(base)['parameter_count'],
                mapping_256=True,valid_semantics_252=True,excluded_edges=EXTRA,
                heldout_labels_masked=bool((base.targ[np.concatenate([c['val'],c['test']])]==-100).all()),
                original_trunk_sha256=base_hash,configs={})
    assert checks['backbone_parameters']==1462957
    assert checks['gfr_shape']==[871,320] and checks['hidden_shape']==[871,36]
    shared_new_hash=None
    agents={}
    for config in CONFIGS:
        agent=make_agent(config)
        assert model_hash(agent)==base_hash
        f=forward(agent)
        error=float((f['output']-reference['output']).abs().max().item())
        assert error<=1e-6 and torch.equal(f['h'],reference['h'])
        assert torch.equal(f['enhanced'][:,:64],f['h'][:,:64])
        assert torch.count_nonzero(f['delta'])==0
        audit=optimizer_audit(agent)
        item=dict(initial_trunk_sha256=model_hash(agent),eval_parity_max_error=error,
                  zero_residual=True,unmodified_first_64=True,**audit)
        # Verify dropout parity with exactly restored RNG and the same train mode.
        state=rng_state()
        base.ModalFusion.train(); agent.ModalFusion.train()
        with torch.no_grad(): b=base.ModalFusion(reference['h'])[0]
        restore_rng(state)
        with torch.no_grad(): a=agent.ModalFusion(f['enhanced'])[0]
        item['train_mode_parity_max_error']=float((a-b).abs().max().item())
        assert item['train_mode_parity_max_error']<=1e-6
        if agent.roi_adapter is not None:
            ah=model_hash(agent,branch=True)
            if shared_new_hash is None: shared_new_hash=ah
            assert ah==shared_new_hash and float(agent.roi_adapter.alpha.detach())==0
            item['new_parameter_init_sha256']=ah
            item['additional_parameters']=audit['parameter_count']-1462957
            assert item['additional_parameters']==32865
            small,delta=agent.roi_adapter(agent.feat[:7,64:],f['h'][:7])
            assert list(small.shape)==[7,320] and list(delta.shape)==[7,256]
            assert torch.equal(small[:,:64],f['h'][:7,:64])
            item['variable_batch_adapter']=True
            agent.ModalFusion.eval()
            steps=[]
            for step in range(2):
                with contextlib.redirect_stdout(io.StringIO()): agent.run_epoch('simple-2')
                g=gradients(agent); steps.append(g)
                assert all(np.isfinite(list(g.values()))) and g['classifier']>0
            assert steps[0]['alpha']>0
            assert steps[0]['project_text']==steps[0]['project_value']==0
            assert steps[1]['project_text']>0 and steps[1]['project_value']>0
            item['two_step_gradients']=steps
            item['gradient_check']='PASS'
        checks['configs'][config]=item
        agents[config]=agent
    # Held-out label values have no influence on GFR, including its computed prior.
    n,parts=base.dataLayer(base.feat)
    altered=base.targ.clone(); hidden=np.concatenate([c['val'],c['test']]); altered[hidden]=1
    with torch.no_grad(): h2,_,loss2=base.MessagePassing0(parts,n,altered,c['train'])
    assert torch.equal(h2,reference['h']) and torch.equal(loss2,reference['aloss'])
    checks['heldout_label_invariance']=True
    agent=agents['F1']; adapter=agent.roi_adapter
    saved_alpha=adapter.alpha.detach().clone()
    with torch.no_grad(): adapter.alpha.fill_(.3)
    e=adapter.text_embeddings.detach().clone()
    z=agent.feat[:,64:]
    d=adapter.residual(z)
    e[0]=e[1]
    changed=adapter.residual(z,e)
    target=int(adapter.valid_indices[0]); other=[i for i in range(256) if i!=target]
    text_target=float((changed[:,target]-d[:,target]).abs().max().item())
    text_other=float((changed[:,other]-d[:,other]).abs().max().item())
    z2=z.clone(); z2[:,target]+=.05
    value=adapter.residual(z2)
    value_target=float((value[:,target]-d[:,target]).abs().max().item())
    value_other=float((value[:,other]-d[:,other]).abs().max().item())
    real=forward(agent,'REAL'); shuffled=forward(agent,'SHUFFLED'); off=forward(agent,'OFF')
    native=forward(agent)
    cf_error=float((native['output']-real['output']).abs().max().item())
    score_change=float((shuffled['delta']-real['delta']).abs().max().item())
    prediction_change=float((real['probability']-off['probability']).abs().max().item())
    assert text_target>0 and value_target>0 and text_other==value_other==0
    assert score_change>0 and prediction_change>0 and cf_error==0
    assert torch.count_nonzero(real['delta'][:,EXTRA])==0
    checks['edge_alignment']=dict(target_feature_index=target,diagnostic_alpha=.3,
                     text_target_delta_max_change=text_target,text_other_delta_max_change=text_other,
                     value_target_delta_max_change=value_target,value_other_delta_max_change=value_other,
                     real_shuffled_delta_max_change=score_change,
                     real_off_probability_max_change=prediction_change,
                     real_matches_normal_max_error=cf_error,extra_delta_max=0.)
    # CONSTANT is an offline input, never an extra training config.
    constant=forward(agent,'CONSTANT')
    checks['constant_offline_diagnostic']=dict(delta_mean_abs=float(constant['delta'].abs().mean()),
        real_probability_mean_change=float((constant['probability']-real['probability']).abs().mean()))
    with torch.no_grad(): adapter.alpha.copy_(saved_alpha)
    before=forward(agent)['output'].clone()
    temp=EXP/'checkpoints/preflight_restore.pt'
    atomic_torch(temp,dict(model=model_state(agent)))
    restored=make_agent('F1')
    load_model(restored,torch.load(temp,map_location='cpu',weights_only=False)['model'])
    restore_error=float((before-forward(restored)['output']).abs().max().item())
    assert restore_error<=1e-6
    temp.unlink()
    integrity(full=True)
    checks.update(checkpoint_restore_max_error=restore_error,protected_sources_unchanged=True,
                  history_experiments_unchanged=True,status='PASS',signature=signature(),completed_utc=utc())
    atomic_json(EXP/'results/preflight.json',checks)
    m=read_json(MANIFEST)
    m.update(preflight_status='PASS',preflight_sha256=sha256(EXP/'results/preflight.json'),
             backbone_parameters=1462957,initialization_parity=True)
    if m['run_status']!='COMPLETE': m['run_status']='PREFLIGHT_PASS'
    atomic_json(MANIFEST,m)
    print('PREFLIGHT PASS: exact P0 parity, same initialization, two-step gradients, edge locality, checkpoints',flush=True)


def require_preflight():
    p=read_json(EXP/'results/preflight.json')
    assert p['status']=='PASS' and p['signature']==signature()
    integrity()


def completed(config):
    p=EXP/'checkpoints'/f'{config}_result.json'
    if not p.exists(): return False
    result=read_json(p)
    best=EXP/'checkpoints'/f'{config}_best.pt'
    return result.get('status')=='COMPLETE' and result.get('signature')==signature() and best.exists() and sha256(best)==result['checkpoint_sha256']


def worker(config):
    import torch
    require_preflight()
    c=context(); agent=make_agent(config)
    optimizer_audit(agent)
    sig=signature(); started=time.monotonic()
    init_hash,branch_hash=model_hash(agent),model_hash(agent,branch=True)
    path=EXP/'checkpoints'/f'{config}_last.pt'
    tracker={k:0. for k in gradients(agent)}
    best_state=None; best_val=None; best_epoch=-1; best_acc=-1.; wait=0; start=0; best_grads={}
    previous_seconds=0.
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['signature']==sig and saved['config']==config
        load_model(agent,saved['model'])
        agent.optimizer_MP0.load_state_dict(saved['optimizer_mp'])
        agent.optimizer_MF.load_state_dict(saved['optimizer_mf'])
        best_state=saved['best_state']; best_val=saved['best_val']; best_acc=saved['best_acc']
        best_epoch=saved['best_epoch']; best_grads=saved['best_grads']; wait=saved['wait']
        tracker=saved['gradient_tracker']; start=saved['epoch']+1; previous_seconds=saved['seconds']
        restore_rng(saved['rng'])
        print(f'RESUME config={config} epoch={start} best_epoch={best_epoch}',flush=True)
    interrupted=[False]
    def stop(signum,frame): interrupted[0]=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    print(json.dumps(dict(config=config,condition=CONFIGS[config],pid=os.getpid(),
                         physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         logical_gpu=torch.cuda.current_device(),device=str(torch.cuda.get_device_properties(0)),
                         initial_trunk_sha256=init_hash,initial_adapter_sha256=branch_hash,
                         started_utc=utc()),ensure_ascii=False),flush=True)
    def save_last(epoch):
        atomic_torch(path,dict(config=config,signature=sig,epoch=epoch,model=model_state(agent),
             optimizer_mp=agent.optimizer_MP0.state_dict(),optimizer_mf=agent.optimizer_MF.state_dict(),
             rng=rng_state(),best_state=best_state,best_val=best_val,best_acc=best_acc,
             best_epoch=best_epoch,best_grads=best_grads,wait=wait,gradient_tracker=tracker,
             seconds=previous_seconds+time.monotonic()-started))
    last_epoch=start-1
    for epoch in range(start,PROTOCOL['nepoch']):
        if wait>PROTOCOL['early']: break
        print(f'epoch={epoch:03d}/{PROTOCOL["nepoch"]}',end=' ',flush=True)
        agent.run_epoch('simple-2')
        g=gradients(agent)
        for key,value in g.items(): tracker[key]=max(tracker[key],value)
        f=forward(agent)
        val=metrics(f['output'],c['val'])
        improved=val['acc']>best_acc
        if improved:
            best_acc=val['acc']; best_val=val; best_state=model_state(agent); best_epoch=epoch; wait=0; best_grads=g
        else: wait+=1
        last_epoch=epoch
        print(f' val_acc={val["acc"]:.6f} val_ba={val["ba"]:.6f} val_prob_auc={val["probability_auc"]:.6f} best={best_epoch} wait={wait}',flush=True)
        if improved or epoch%10==9 or wait>PROTOCOL['early'] or interrupted[0]: save_last(epoch)
        if interrupted[0]:
            print('INTERRUPTED: saved an entire completed optimization epoch',flush=True)
            return 130
    assert best_state is not None
    load_model(agent,best_state)
    f=forward(agent)
    test=metrics(f['output'],c['test'])
    best_path=EXP/'checkpoints'/f'{config}_best.pt'
    atomic_torch(best_path,dict(config=config,signature=sig,model=best_state,best_epoch=best_epoch,
                              validation=best_val,train_protocol=PROTOCOL,split_protocol=SPLIT))
    # Restore into a fresh object and verify the whole-population evaluation forward.
    fresh=make_agent(config)
    load_model(fresh,torch.load(best_path,map_location='cpu',weights_only=False)['model'])
    error=float((forward(fresh)['output']-f['output']).abs().max().item())
    assert error<=1e-6
    result=dict(status='COMPLETE',config=config,condition=CONFIGS[config],signature=sig,
                seed=0,fold=1,physical_gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),
                worker_pid=os.getpid(),initial_trunk_sha256=init_hash,initial_adapter_sha256=branch_hash,
                additional_parameters=0 if config=='F0' else 32865,
                best_epoch=best_epoch,epochs_executed=last_epoch+1,validation=best_val,test=test,
                gradient_max_norms=tracker,best_epoch_gradient_norms=best_grads,
                seconds=previous_seconds+time.monotonic()-started,
                checkpoint_restore_max_error=error,checkpoint_sha256=sha256(best_path),completed_utc=utc())
    assert config=='F0' or all(tracker[k]>0 for k in ['alpha','project_text','project_value'])
    atomic_json(EXP/'checkpoints'/f'{config}_result.json',result)
    path.unlink(missing_ok=True)
    print('COMPLETE '+json.dumps(result,ensure_ascii=False),flush=True)
    return 0


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def inference_diagnostics(results):
    import torch
    c=context(); test=c['test']; valid=c['cache']['valid_feature_indices'].tolist()
    base=make_agent('F0')
    load_model(base,torch.load(EXP/'checkpoints/F0_best.pt',map_location='cpu',weights_only=False)['model'])
    baseline=forward(base)
    diag,cf=[] ,[]
    for result in results:
        config=result['config']
        if config=='F0': continue
        agent=make_agent(config)
        with torch.no_grad():
            initial_projected=torch.nn.functional.normalize(agent.roi_adapter.project_text(agent.roi_adapter.text_embeddings),dim=-1)
        initial_geometry=embedding_diagnostics(initial_projected)
        load_model(agent,torch.load(EXP/'checkpoints'/f'{config}_best.pt',map_location='cpu',weights_only=False)['model'])
        own=forward(agent)
        initial_parameter_hash=model_hash(agent,branch=True)
        probabilities=own['probability'][test]
        row=dict(config=config,seed=0,fold=1,condition=CONFIGS[config],additional_parameters=32865,
                 alpha_initial=0.,alpha_trained=float(agent.roi_adapter.alpha.item()),
                 actual_gate_coefficient=float(.1*agent.roi_adapter.alpha.tanh().item()),
                 project_text_gradient_max=result['gradient_max_norms']['project_text'],
                 project_value_gradient_max=result['gradient_max_norms']['project_value'],
                 alpha_gradient_max=result['gradient_max_norms']['alpha'],
                 project_text_gradient_at_best=result['best_epoch_gradient_norms']['project_text'],
                 project_value_gradient_at_best=result['best_epoch_gradient_norms']['project_value'],
                 project_text_weight_norm=float(agent.roi_adapter.project_text.weight.detach().norm()),
                 project_value_weight_norm=float(agent.roi_adapter.project_value.weight.detach().norm()),
                 delta_valid_mean_abs=float(own['delta'][:,valid].abs().mean()),
                 delta_valid_max_abs=float(own['delta'][:,valid].abs().max()),
                 delta_extra_max_abs=float(own['delta'][:,EXTRA].abs().max()),
                 residual_relative_l2_mean=float((own['delta'].norm(dim=-1)/(own['h'][:,64:].norm(dim=-1)+1e-12)).mean()),
                 valid_response_nonzero_fraction=float((own['delta'][:,valid].abs()>1e-8).float().mean()),
                 original_first_64_max_change=float((own['enhanced'][:,:64]-own['h'][:,:64]).abs().max()),
                 f0_probability_mean_abs_change=float((probabilities-baseline['probability'][test]).abs().mean()),
                 f0_prediction_flip_fraction=float((probabilities.argmax(-1)!=baseline['probability'][test].argmax(-1)).float().mean()),
                 finite_prediction=True,checkpoint_restore_max_error=result['checkpoint_restore_max_error'])
        trained_stats=c['cache']['diagnostics']['conditions'][CONFIGS[config]]
        row.update(text_embedding_pair_cosine=trained_stats['pairwise_cosine_mean'],
                   text_embedding_effective_rank=trained_stats['effective_rank'],
                   text_embedding_centered_effective_rank=trained_stats['centered_effective_rank'])
        with torch.no_grad():
            projected=torch.nn.functional.normalize(agent.roi_adapter.project_text(agent.roi_adapter.text_embeddings),dim=-1)
            raw=agent.feat[:,64:].index_select(1,agent.roi_adapter.valid_indices)
            phi=torch.stack([raw,raw.abs(),raw.sign()],dim=-1)
            value=torch.nn.functional.normalize(agent.roi_adapter.project_value(phi),dim=-1)
            scores=(value*projected.unsqueeze(0)).sum(dim=-1)
        geometry=embedding_diagnostics(projected)
        row.update(projected_text_pair_cosine=geometry['pairwise_cosine_mean'],
                   initial_projected_text_pair_cosine=initial_geometry['pairwise_cosine_mean'],
                   initial_projected_text_effective_rank=initial_geometry['effective_rank'],
                   initial_projected_text_centered_effective_rank=initial_geometry['centered_effective_rank'],
                   projected_text_effective_rank=geometry['effective_rank'],
                   projected_text_centered_effective_rank=geometry['centered_effective_rank'],
                   projected_text_zero_vectors=int((projected.norm(dim=-1)==0).sum().item()),
                   matching_score_mean=float(scores.mean()),matching_score_std=float(scores.std(unbiased=False)),
                   matching_score_abs_mean=float(scores.abs().mean()),
                   matching_score_std_across_edges_mean=float(scores.std(dim=1,unbiased=False).mean()),
                   matching_score_std_across_subjects_mean=float(scores.std(dim=0,unbiased=False).mean()),
                   delta_mean_signed=float(own['delta'][:,valid].mean()))
        for condition in ['REAL','SHUFFLED','ID_ONLY','CONSTANT','OFF']:
            f=forward(agent,condition)
            prob=f['probability'][test]
            met=metrics(f['output'],test)
            change=float((prob-probabilities).abs().mean())
            flip=float((prob.argmax(-1)!=probabilities.argmax(-1)).float().mean())
            cf.append(dict(config=config,seed=0,fold=1,condition=condition,**met,
                        vs_trained_probability_mean_abs_change=change,
                        vs_trained_prediction_flip_fraction=flip,
                        vs_trained_delta_mean_abs_change=float((f['delta']-own['delta']).abs().mean()),
                        delta_valid_mean_abs=float(f['delta'][:,valid].abs().mean()),
                        delta_extra_max_abs=float(f['delta'][:,EXTRA].abs().max())))
            assert float(f['delta'][:,EXTRA].abs().max())==0
            if condition==CONFIGS[config]: assert change==0 and flip==0
            if condition=='OFF':
                row['off_probability_mean_abs_change']=change
                row['off_prediction_flip_fraction']=flip
        real=forward(agent,'REAL'); shuffled=forward(agent,'SHUFFLED'); ids=forward(agent,'ID_ONLY')
        row.update(real_shuffled_delta_mean_abs_change=float((real['delta']-shuffled['delta']).abs().mean()),
                   real_shuffled_probability_mean_abs_change=float((real['probability'][test]-shuffled['probability'][test]).abs().mean()),
                   real_shuffled_prediction_flip_fraction=float((real['probability'][test].argmax(-1)!=shuffled['probability'][test].argmax(-1)).float().mean()),
                   real_id_only_delta_mean_abs_change=float((real['delta']-ids['delta']).abs().mean()),
                   real_id_only_probability_mean_abs_change=float((real['probability'][test]-ids['probability'][test]).abs().mean()))
        assert model_hash(agent,branch=True)==initial_parameter_hash
        diag.append(row)
    write_csv(EXP/'results/alignment_diagnostics.csv',diag)
    write_csv(EXP/'results/counterfactual.csv',cf)
    return diag,cf


def refresh():
    integrity(full=True)
    results=[read_json(EXP/'checkpoints'/f'{config}_result.json') for config in CONFIGS if completed(config)]
    rows=[]
    for result in results:
        row={k:result[k] for k in ['config','condition','seed','fold','physical_gpu','additional_parameters',
                                 'best_epoch','epochs_executed','seconds','checkpoint_restore_max_error']}
        row.update({'val_'+k:v for k,v in result['validation'].items()})
        row.update(result['test'])
        rows.append(row)
    write_csv(EXP/'results/pilot_metrics.csv',rows)
    m=read_json(MANIFEST)
    for result in results:
        m['tasks'][result['config']]={k:result[k] for k in ['status','physical_gpu','worker_pid','best_epoch',
            'epochs_executed','checkpoint_restore_max_error','checkpoint_sha256','initial_trunk_sha256','initial_adapter_sha256','seconds']}
    if len(results)==4:
        assert len({r['initial_trunk_sha256'] for r in results})==1
        assert len({r['initial_adapter_sha256'] for r in results if r['config']!='F0'})==1
        diag,cf=inference_diagnostics(results)
        assert len(diag)==3 and len(cf)==15
        assert all(d['delta_extra_max_abs']==d['original_first_64_max_change']==0 for d in diag)
        integrity(full=True)
        by={r['config']:r for r in results}
        real=by['F1']['test']; control=[by[k]['test'] for k in ['F2','F3']]
        # Any support here is explicitly exploratory; one fold cannot establish medical efficacy.
        semantic='INCONCLUSIVE'
        if all(real['ba']>x['ba'] and real['probability_auc']>x['probability_auc'] for x in control) and diag[0]['real_shuffled_probability_mean_abs_change']>1e-6:
            semantic='SUPPORTED'
        elif all(real['ba']<x['ba'] and real['probability_auc']<x['probability_auc'] for x in control):
            semantic='NOT_SUPPORTED'
        m.update(run_status='COMPLETE',ENGINEERING_STATUS='PASS',SEMANTIC_STATUS=semantic,
                 PILOT_STATUS='COMPLETE',recommend_full_30_folds=semantic=='SUPPORTED',
                 completed_utc=m.get('completed_utc',utc()),protected_files_final_verified=True,history_final_verified=True,
                 completed_configs=4,completed_training_folds=4,alignment_diagnostic_rows=3,
                 counterfactual_rows=15,seed_fold_scope='seed 0 / outer fold 1 only')
        m['implementation_hashes']={name:sha256(EXP/name) for name in
                                    ['run.py','build_roi_semantics.py','roi_semantic_adapter.py','integration_patch.diff']}
        m['pilot_wall_seconds']=(dt.datetime.fromisoformat(m['completed_utc'])-
                                  dt.datetime.fromisoformat(m['pilot_started_utc'])).total_seconds()
        report(results,diag,cf,m)
        m['result_hashes']={name:sha256(EXP/'results'/name) for name in
                           ['code_audit.md','edge_semantic_manifest.csv','text_embedding_diagnostics.json',
                            'preflight.json','pilot_metrics.csv','alignment_diagnostics.csv','counterfactual.csv','report.md']}
        atomic_json(MANIFEST,m)
        shutil.rmtree(EXP/'_runtime',ignore_errors=True)
        for temp in (EXP/'checkpoints').glob('*.tmp'): temp.unlink()
        print_final(results,m)
    else:
        m['run_status']='PARTIAL'; m['PILOT_STATUS']='PARTIAL'
        atomic_json(MANIFEST,m)
        print(f'PARTIAL: {len(results)}/4 complete; no missing metrics are replaced by zero',flush=True)


def report(results,diag,cf,m):
    by={r['config']:r for r in results}
    text=read_json(EXP/'results/text_embedding_diagnostics.json')
    pre=read_json(EXP/'results/preflight.json')
    d={row['config']:row for row in diag}
    lines=['# 实验十二：AGESA 真实解剖连接语义对齐可行性', '',
           f'ENGINEERING_STATUS={m["ENGINEERING_STATUS"]}；SEMANTIC_STATUS={m["SEMANTIC_STATUS"]}；PILOT_STATUS=COMPLETE。', '',
           '## 数据流与接入结论', '',
           '完整代码级流程和保留的源码细节见 [code_audit.md](code_audit.md)。原始 GFR 为受试者图，输出320维；四模态按48/6/10/256分段，原PLN生成每模态36维并划分四个潜在9维token。token不是ROI。AGESA在隔离 `EvalHelper.run_epoch(simple-2)` 中 GFR 与 ModalFusion 之间接入，只增量修改 H[:,64:320]；逐连接配对始终使用原始CSV Z[:,e]，不把传播后的H坐标解释为连接强度。GFR、PLN、P0 Softmax-Max、分类头保持原代码，未修复 GFR 覆盖分支、未加入原 ALoss 或新损失。', '',
           '## ROI映射与静态文本', '',
           '256/256个特征与实验十一索引严格一致；原上三角索引唯一且端点位于0..110。名称由111标签表按端点索引关联，而非读取继承的UNKNOWN字段。252条英文模板为正确解剖连接身份，第一条为Left Thalamus与Left Superior Temporal Gyrus, anterior division；无ASD方向、因果或临床推断。ROI 82/atlas3455涉及特征195、225、232、236，保留原始数值且语义残差严格为0。SHUFFLED是固定seed1200的无固定点置换，文本全集和分布不变；ID_ONLY使用实际特征号；CONSTANT仅离线诊断。映射与文本来源哈希见edge_semantic_manifest.csv与prompts.json。', '',
           '## 冻结CLIP文本表征', '',
           f'RN50权重SHA256 `{m["clip_weight_sha256"]}`。冻结eval/no_grad一次编码真实252条、ID252条、常量1条和P0八句，共513项；SHUFFLED复用REAL向量。训练每epoch编码次数为0，最长文本含BOS/EOS为{text["token_length_max"]}/77 token。', '',
           '| 文本 | 平均两两cos | cos 5%/50%/95% | 完全重复数 | cos>=.999对数 | 有效秩 | 中心化有效秩 |',
           '|---|---:|---|---:|---:|---:|---:|']
    for k in ['REAL','SHUFFLED','ID_ONLY','CONSTANT']:
        x=text['conditions'][k]; q=x['pairwise_cosine_quantiles']
        lines.append(f'| {k} | {x["pairwise_cosine_mean"]:.6f} | {q["0.05"]:.5f}/{q["0.5"]:.5f}/{q["0.95"]:.5f} | {x["exact_duplicate_count"]} | {x["pairs_cosine_ge_0999"]} | {x["effective_rank"]:.2f} | {x["centered_effective_rank"]:.2f} |')
    lines += ['', '共享模板造成较高余弦相似度，需要结合中心化多样性和置换敏感性判断；可区分向量不自动意味着CLIP理解了可靠的医学功能。CONSTANT中心化秩为0、所有向量相同是预期对照。', '',
              '## AGESA、初始化与真实梯度', '',
              'ProjectText=Linear(1024,32,bias=False)，ProjectValue=Linear(3,32,bias=False)，phi(z)=[z,abs(z),sign(z)]。两路归一化后s为点积；delta=.1*tanh(alpha)*z*tanh(s)，共享alpha零初始化。F1/F2/F3各新增32,865参数；F0无新增。原模型1,462,957参数。旧参数Adam .0038/.11，新参数同lr、wd0。新模块在原my_weight_init之后注册，fork_rng隔离，不扰动原模型或dropout RNG。', '',
              f'四组已有网络初始化hash一致；三组投影初始化hash一致。初始eval/train-mode输出最大误差均不超过{max(max(x["eval_parity_max_error"],x["train_mode_parity_max_error"]) for x in pre["configs"].values()):.3e}，初始delta=0；所有优化器参数无重复或遗漏。首步alpha非零梯度、两投影梯度为0，第二步两投影均有非零有限梯度；没有为验证而改变训练初始门控。AGESA本身支持7人测试batch；PLN的871硬编码保留。', '',
              '| 配置 | alpha训练后 | 实际系数 | Text梯度最大范数 | Value梯度最大范数 | 有效delta绝对均值/最大值 | 相对L2残差 | 非零比例 |',
              '|---|---:|---:|---:|---:|---|---:|---:|']
    for row in diag:
        lines.append(f'| {row["config"]} | {row["alpha_trained"]:+.6f} | {row["actual_gate_coefficient"]:+.6f} | {row["project_text_gradient_max"]:.3e} | {row["project_value_gradient_max"]:.3e} | {row["delta_valid_mean_abs"]:.3e}/{row["delta_valid_max_abs"]:.3e} | {row["residual_relative_l2_mean"]:.6f} | {row["valid_response_nonzero_fraction"]:.6f} |')
    edge=pre['edge_alignment']
    lines += ['', '## 连接局部性与工程检验', '',
              f'诊断时临时alpha=.3，仅替换特征{edge["target_feature_index"]}的文本：目标残差最大变化{edge["text_target_delta_max_change"]:.3e}、非目标变化{edge["text_other_delta_max_change"]:.3e}；仅扰动该原始连接值+.05：目标变化{edge["value_target_delta_max_change"]:.3e}、非目标变化{edge["value_other_delta_max_change"]:.3e}。REAL/SHUFFLED残差最大差{edge["real_shuffled_delta_max_change"]:.3e}；真实分支与OFF分类概率最大差{edge["real_off_probability_max_change"]:.3e}。这些是机制响应检验，不是语义性能证明。训练alpha仍从0开始。', '',
              '4条异常delta在preflight、训练后及全部反事实条件均精确为0；前64维注入前后最大误差为0。验证/测试标签遮蔽和原GFR对held-out标签值不变性通过。初始和最终checkpoint均在新对象恢复后逐值核验，最大误差不超过1e-6，所有预测与梯度有限。', '',
              '## 严格单fold pilot协议', '',
              f'外层StratifiedKFold(10,shuffle=True,random_state=10)第1折；seed0；外层训练集合内用stratify/random_state0切15%验证集。train/validation/test={m["split_counts"]["train"]["n"]}/{m["split_counts"]["validation"]["n"]}/{m["split_counts"]["test"]["n"]}；类别数量见manifest。只用内部训练标签做NLL和GFR标签先验，内部val ACC严格增加选checkpoint（BA/AUC同步记录且不作为平局选择），外层test仅在最佳checkpoint固定后评估及进行无训练反事实。最大450epoch，wait>200早停；全871特征参与原transductive前向。', '',
              '沿用原全局预处理与RFE后的固定数据；上游选择信息共享仍是研究限制，不是完整独立泛化验证。新验证协议与历史实验不同，历史91.69/91.80%不用于本次直接比较。仅4组×seed0×fold1，没有30fold扩展、调参或修改文本。', '',
              '| 配置 | 文本 | GPU | 最佳epoch(0-based) | 已跑epoch | Val ACC | Test ACC | BA | Prob-AUC | SEN-ASD | SPE-TD |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        t=r['test']; lines.append(f'| {r["config"]} | {r["condition"]} | {r["physical_gpu"]} | {r["best_epoch"]} | {r["epochs_executed"]} | {100*r["validation"]["acc"]:.3f}% | {100*t["acc"]:.3f}% | {100*t["ba"]:.3f}% | {100*t["probability_auc"]:.3f}% | {100*t["sensitivity_asd"]:.3f}% | {100*t["specificity_td"]:.3f}% |')
    lines += ['', 'Probability-AUC以ASD(0)为阳性，使用exp(log-softmax)的第0列；BA是ASD sensitivity与TD specificity均值，不把hard-label AUC称为ROC-AUC。', '',
              '| 配对差值 | ACC pp | BA pp | Prob-AUC pp | SEN pp | SPE pp |',
              '|---|---:|---:|---:|---:|---:|']
    for control in ['F0','F2','F3']:
        a,b=by['F1']['test'],by[control]['test']
        lines.append('| F1−'+control+' | '+' | '.join(f'{100*(a[k]-b[k]):+.3f}' for k in ['acc','ba','probability_auc','sensitivity_asd','specificity_td'])+' |')
    lines += ['', '## 训练后文本干预与预测变化', '',
              '所有干预保持同一已选checkpoint、图、原始连接与P0路径；只覆盖AGESA文本或关闭残差，无重新初始化/优化/选择checkpoint。统计同一88人外层测试集，全部个体预测仅在内存中比较，不保存受试者ID或预测表。OFF与独立训练F0不是同一对照：OFF只关掉当前训练模型的分支。', '',
              '| 训练模型 | 推理文本 | Test ACC | BA | Prob-AUC | 相对训练文本概率变化 | 翻转率 | delta变化均值 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in cf:
        lines.append(f'| {r["config"]} | {r["condition"]} | {100*r["acc"]:.3f}% | {100*r["ba"]:.3f}% | {100*r["probability_auc"]:.3f}% | {r["vs_trained_probability_mean_abs_change"]:.3e} | {r["vs_trained_prediction_flip_fraction"]:.6f} | {r["vs_trained_delta_mean_abs_change"]:.3e} |')
    lines += ['', f'F1 REAL/SHUFFLED概率平均绝对差{d["F1"]["real_shuffled_probability_mean_abs_change"]:.3e}，REAL/ID_ONLY差{d["F1"]["real_id_only_probability_mean_abs_change"]:.3e}；F1关分支概率变化{d["F1"]["off_probability_mean_abs_change"]:.3e}。F1相对独立F0概率变化{d["F1"]["f0_probability_mean_abs_change"]:.3e}、翻转比例{d["F1"]["f0_prediction_flip_fraction"]:.6f}，该量同时包含优化轨迹改变，不能全部归因于文本。', '',
              '## 训练后文本投影是否保留差异', '',
              '| 配置 | 初始/训练后投影平均cos | 初始/训练后有效秩 | 初始/训练后中心化秩 | 匹配s均值/SD | 每人跨边s的SD均值 | delta有符号均值 |',
              '|---|---:|---:|---:|---|---:|---:|']
    for row in diag:
        lines.append(f'| {row["config"]} | {row["initial_projected_text_pair_cosine"]:.6f}/{row["projected_text_pair_cosine"]:.6f} | {row["initial_projected_text_effective_rank"]:.3f}/{row["projected_text_effective_rank"]:.3f} | {row["initial_projected_text_centered_effective_rank"]:.3f}/{row["projected_text_centered_effective_rank"]:.3f} | {row["matching_score_mean"]:+.5f}/{row["matching_score_std"]:.5f} | {row["matching_score_std_across_edges_mean"]:.5f} | {row["delta_mean_signed"]:+.3e} |')
    lines += ['', '这些统计使用已选checkpoint，不重新训练，也不按结果改变投影结构。投影后高余弦或低非中心化有效秩表示共有方向占主导；中心化秩能揭示残余差异，不能把非中心化低秩单独称作完全坍塌。负alpha只说明学得了反向残差系数，不表示文本语义医学方向正确或错误。', '',
              f'F1训练前后采用相同32维投影结构，其平均文本cos从{d["F1"]["initial_projected_text_pair_cosine"]:.6f}变为{d["F1"]["projected_text_pair_cosine"]:.6f}，有效秩从{d["F1"]["initial_projected_text_effective_rank"]:.3f}变为{d["F1"]["projected_text_effective_rank"]:.3f}。这是训练后共有方向占主导的直接几何证据，比仅比较原1024维CLIP与32维投影更能区分维度压缩与训练变化；仍保留非零中心化差异。跨边匹配分数的SD同时包含各边原始连接值不同，不能将它全部解释为ROI语义区分。', '',
              '## 工程与医学语义结论', '',
              '工程上完成了真实连接—文本的一一对应、局部残差敏感性、零门控parity、实际梯度与恢复。数据上252条文本的身份经过实验十一索引与标签双重核验，4条未知身份被掩蔽。该机制真正让正确位置的原始数值与该位置的文本共同控制特征残差，避免了潜在token冒充ROI。', '',
              f'当前语义判定为 **{m["SEMANTIC_STATUS"]}**。该判定描述本次单fold探索性趋势；即使REAL向量可区分且替换文本改变残差，也不证明医学知识有独立ASD价值。必须看REAL相对SHUFFLED和ID_ONLY的BA与连续AUC，而非只看F1−F0或梯度。没有统计显著性结论，也没有发现具体ASD脑区机制的证据。', '',
              f'实测F1相对F0/F2/F3的ACC差分别为{100*(by["F1"]["test"]["acc"]-by["F0"]["test"]["acc"]):+.3f}/{100*(by["F1"]["test"]["acc"]-by["F2"]["test"]["acc"]):+.3f}/{100*(by["F1"]["test"]["acc"]-by["F3"]["test"]["acc"]):+.3f}pp；BA差分别为'+ '/'.join(f'{100*(by["F1"]["test"]["ba"]-by[k]["test"]["ba"]):+.3f}' for k in ['F0','F2','F3'])+'pp；Probability-AUC差分别为'+ '/'.join(f'{100*(by["F1"]["test"]["probability_auc"]-by[k]["test"]["probability_auc"]):+.3f}' for k in ['F0','F2','F3'])+f'pp。F1−F0的ASD sensitivity差{100*(by["F1"]["test"]["sensitivity_asd"]-by["F0"]["test"]["sensitivity_asd"]):+.3f}pp、TD specificity差{100*(by["F1"]["test"]["specificity_td"]-by["F0"]["test"]["specificity_td"]):+.3f}pp。不能把AUC相近理解为概率或校准相同。四组验证checkpoint不同，单fold结果不能分离全部优化轨迹与抽样波动。', '',
              f'F1相对残差L2强度为{100*d["F1"]["residual_relative_l2_mean"]:.3f}%，有效delta绝对均值{d["F1"]["delta_valid_mean_abs"]:.3e}，并非近零分支。其正确/错配/编号/常量干预的类别结果均可由上表直接核对；真实配对相对于错误配对的概率变化远小于关闭全部残差。模型对连接文本有数值响应，但医学身份对应关系未体现分类特异性增益。关闭F1分支的表现需与完整F1及独立F0分别比较，不是重训后的基线。', '',
              '原始CLIP已区分252条向量，不能把负结果简单归因于原编码完全失效。结合F1投影的共有方向增强、真实/错配/常量替换影响小于OFF，当前更符合分支主要形成共有文本方向条件下的数值重加权、未有效利用正确解剖身份差异的解释；这里只是机制线索，不是因果定论。文本仅含连接身份而没有个体疾病知识，原主干已利用原连接值，单fold抽样与checkpoint变化也限制判断。本实验没有继续改文本、门控或隐藏维度追分。', '',
              f'是否建议完整10折×3seeds：**{"建议在预注册固定方案和独立验证下进一步核验该探索性趋势" if m["recommend_full_30_folds"] else "暂不建议直接扩大到30折性能搜索"}**。尚不建议立即和实验八至十的疾病分支组合；应先确认ROI身份比匿名身份/错误配对提供独立信息，再研究互补性。后续不得据本测试折结果调参。', '',
              '## 可复现性与运行状态', '',
              f'Git HEAD `{m["git_head"]}`；源码、CSV、modality字典、ROI映射及CLIP哈希见manifest。GPU0={"、".join(k for k,v in GPU_ASSIGNMENT.items() if v==0)}，GPU1={"、".join(k for k,v in GPU_ASSIGNMENT.items() if v==1)}，物理分配和进程PID记录在各任务日志/manifest与scheduler.log。四组主干与三组新参数初始化核验一致。', '',
              f'正式pilot与首次最终诊断从{m["pilot_started_utc"]}至{m["completed_utc"]}，壁钟约{m["pilot_wall_seconds"]:.1f}秒；各worker训练与最佳checkpoint恢复时间见pilot_metrics.csv。后续refresh只重建报告/已训练模型诊断，保留首次完成时间，不增加训练。', '',
              '运行器支持prepare/preflight/all/resume/refresh；每次恢复加载完整已完成epoch的模型、两个Adam、best-state及CPU/CUDA/NumPy/Python RNG。成功任务按signature与checkpoint哈希跳过，失败结果不会变成0分或进入汇总。最终保留4个最佳checkpoint、单份文本cache、必要代码/轻量表/日志，清理临时接入副本和last checkpoint。根源码、CSV和实验一至十一哈希最终核验不变；本次未commit或push。']
    (EXP/'results/report.md').write_text('\n'.join(lines)+'\n')


def print_final(results,m):
    print('Experiment 12 — AGESA Feasibility\n')
    print('CODE_AUDIT: COMPLETE\nROI_MAPPING_VERIFIED: 256/256\nVALID_SEMANTIC_EDGES: 252\nEXTRA_ROI_EDGES: 4')
    print('TEXT_EMBEDDING_STATUS: VALID (see cosine/rank diagnostics)\nINITIALIZATION_PARITY: PASS\nGRADIENT_CHECK: PASS\nEDGE_ALIGNMENT_CHECK: PASS')
    print('ENGINEERING_STATUS: '+m['ENGINEERING_STATUS']+'\n\nPILOT_STATUS: '+m['PILOT_STATUS'])
    for r in results:
        t=r['test']; print(f'{r["config"]} ACC / BA / Prob-AUC: {100*t["acc"]:.3f}% / {100*t["ba"]:.3f}% / {100*t["probability_auc"]:.3f}%')
    print('\nSEMANTIC_STATUS: '+m['SEMANTIC_STATUS'])
    print('RECOMMEND_FULL_30_FOLDS: '+str(m['recommend_full_30_folds']))
    print('REPORT_PATH: '+str(EXP/'results/report.md'),flush=True)


def schedule():
    import fcntl
    (EXP/'logs').mkdir(exist_ok=True)
    lock=(EXP/'logs/scheduler.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another Experiment 12 scheduler is running')
    def request_stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,request_stop)
    require_path=EXP/'results/preflight.json'
    if not require_path.exists():
        subprocess.run([sys.executable,str(EXP/'run.py'),'prepare'],check=True,env=worker_env(0))
        subprocess.run([sys.executable,str(EXP/'run.py'),'preflight'],check=True,env=worker_env(0))
    require_preflight()
    scheduler=(EXP/'logs/scheduler.log').open('a',buffering=1)
    def log(message):
        s=utc()+' '+message
        print(s,flush=True); scheduler.write(s+'\n')
    m=read_json(MANIFEST); m.update(run_status='RUNNING',pilot_started_utc=m.get('pilot_started_utc',utc()))
    atomic_json(MANIFEST,m)
    active={}; attempts={config:0 for config in CONFIGS}
    def launch(config):
        attempts[config]+=1
        stream=(EXP/'logs'/f'{config}_seed0_fold1.log').open('a',buffering=1)
        stream.write('\nATTEMPT '+str(attempts[config])+' '+utc()+'\n'); stream.flush()
        p=subprocess.Popen([sys.executable,str(EXP/'run.py'),'worker','--config',config],
                           cwd=ROOT,env=worker_env(GPU_ASSIGNMENT[config]),stdout=stream,stderr=subprocess.STDOUT)
        active[config]=(p,stream)
        m=read_json(MANIFEST);m['tasks'][config]=dict(status='RUNNING',physical_gpu=GPU_ASSIGNMENT[config],pid=p.pid,attempt=attempts[config]);atomic_json(MANIFEST,m)
        log(f'LAUNCH {config} physical_gpu={GPU_ASSIGNMENT[config]} pid={p.pid}')
    for config in CONFIGS:
        if completed(config): log('SKIP verified COMPLETE '+config)
        else: launch(config)
    try:
        while active:
            time.sleep(5)
            for config,(p,stream) in list(active.items()):
                if p.poll() is None: continue
                stream.close();del active[config]
                if p.returncode==0 and completed(config):
                    log('SUCCESS '+config)
                elif attempts[config]<2 and p.returncode!=130:
                    log(f'RETRY {config} exit={p.returncode}, resume last valid full epoch');launch(config)
                else:
                    log(f'FAILED_OR_INTERRUPTED {config} exit={p.returncode}')
                    m=read_json(MANIFEST);m['tasks'][config].update(status='FAILED' if p.returncode!=130 else 'INTERRUPTED',exit_code=p.returncode);atomic_json(MANIFEST,m)
    except KeyboardInterrupt:
        log('scheduler interrupted: workers save at completed epoch')
        for p,_ in active.values(): p.send_signal(signal.SIGTERM)
        for p,stream in active.values(): p.wait(timeout=60);stream.close()
        raise
    finally:
        scheduler.close();lock.close()
        (EXP/'logs/scheduler.lock').unlink(missing_ok=True)
    subprocess.run([sys.executable,str(EXP/'run.py'),'refresh'],check=True,env=worker_env(0))
    if read_json(MANIFEST)['run_status']!='COMPLETE': raise RuntimeError('pilot incomplete; inspect logs and resume')


def worker_env(gpu):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT)+os.pathsep+str(EXP),
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    return env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','all','resume','refresh','worker'])
    parser.add_argument('--config',choices=CONFIGS)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='preflight': preflight()
    elif args.action in ['all','resume']: schedule()
    elif args.action=='refresh': refresh()
    else:
        if not args.config: parser.error('worker requires --config')
        sys.exit(worker(args.config))


if __name__=='__main__':
    main()

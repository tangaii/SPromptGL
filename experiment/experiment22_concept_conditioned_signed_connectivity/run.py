#!/usr/bin/env python3
"""Frozen SABER22: independent signed FC mechanism experiment.
Infrastructure adapted once from the audited Exp21 textual template.
"""
import argparse, ast, copy, contextlib, csv, difflib, fcntl, hashlib, io, json, math, os, pickle
import random, re, shutil, subprocess, sys, tempfile, time, types, signal
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
sys.dont_write_bytecode=True
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if 'preflight' in sys.argv or 'prepare' in sys.argv:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
from contract22 import SPECS, TASKS, PAIRS, CF_GROUPS, CF_ACTIONS, HISTORICAL_REFERENCES
ROOT=Path(os.environ.get('SP22_ROOT','/data/adodas/SPromptGL-main')).resolve()
STATE=Path('/tmp')/('spromptgl22-state-'+hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
MANIFEST,RESULTS,LOGS=EXP/'manifest.json',EXP/'results',EXP/'logs'
HANDOFF=ROOT/'handoff.md'
P9=ROOT/'experiment/experiment9_prompt_disease_adversarial_interaction'
P19=ROOT/'experiment/experiment19_contrastive_disease_evidence_alignment'
P20=ROOT/'experiment/experiment20_grounded_concept_graph_alignment'
P21=ROOT/'experiment/experiment21_evidence_constrained_adaptive_alignment'
CACHE=EXP/'semantic_cache.npz'
CONFIGS=list(SPECS);CONDITIONS=list(CF_ACTIONS)
DESIGNS={c:dict(**asdict(s),prompt='P3' if s.base=='E9' else 'P0',disease=s.base=='E9',adv=False,use_graph=s.graph,family='SIGNED22') for c,s in SPECS.items()}
CORE=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']
AUDIT_COMMIT='c2f07d274a281861a79dd35b979449b6b25e992e'
EXPECTED_FOLDS=540;EXPECTED_TASKS=54;EXPECTED_DIAGNOSTICS=540;EXPECTED_CF_ROWS=2340;EXPECTED_CF_NA=270
METRICS=['acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td']
REPORT_METRICS=METRICS+['f1_asd']
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',nepoch=450,early=200,
 lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,
 GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,
 theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
assert CONFIGS==[f'M{i:02}' for i in range(18)] and len(TASKS)==54
_CTX=None

def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "wb") as handle:
        handle.write(data.encode() if isinstance(data, str) else data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buf.getvalue())


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save_checkpoint(path, obj):
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    atomic(path, b"SP22" + hashlib.sha256(data).hexdigest().encode() + data)


def checkpoint(path):
    import torch
    data = Path(path).read_bytes()
    assert data[:4] == b"SP22" and hashlib.sha256(data[68:]).hexdigest().encode() == data[4:68]
    return torch.load(io.BytesIO(data[68:]), map_location="cpu", weights_only=False)


def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda"]])


def rng_hash(saved):
    h = hashlib.sha256(pickle.dumps((saved["python"], saved["numpy"]), protocol=4))
    for v in [saved["cpu"]] + saved["cuda"]:
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    table = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    mapping = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
    assert table.columns[:-1].tolist() == [str(v) for fields in mapping.values() for v in fields]
    values = table.values
    assert values.shape == (871, 321) and np.isfinite(values).all()
    x, y = values[:, :-1], values[:, -1].astype(int) - 1
    assert int((y == 0).sum()) == 403 and int((y == 1).sum()) == 468
    pheno = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith(("AGE_AT_SCAN_", "SEX_"))]
    sites = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith("SITE_ID_")]
    assert len(pheno) == 28 and len(sites) == 20 and not set(pheno).intersection(sites)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    audit = dict(shape=list(x.shape), labels={"ASD_0": 403, "TD_1": 468},
                 modality_columns={k: list(map(str, v)) for k, v in mapping.items()},
                 pheno_age_sex_indices=pheno, site_indices=sites)
    return x, y, folds, audit


def assert_equal_tree(a, b):
    import numpy as np
    import torch
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for v, w in zip(a, b):
            assert_equal_tree(v, w)
    else:
        assert a == b


def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def correlation(x,y):
    import numpy as np
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    # A constant has no defined correlation; record 0 with a separate SD below.
    return float(np.corrcoef(x,y)[0,1]) if x.std()>1e-10 and y.std()>1e-10 else 0.


def read_json(path):
    # OSS/FUSE can briefly hide the target of an atomic rename.
    for attempt in range(8):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError,OSError):
            if attempt==7: raise
            time.sleep(.15*(attempt+1))


def historical_folders():
    return sorted(p for p in (ROOT/'experiment').iterdir() if p.is_dir() and
                  re.match(r'experiment\d+_',p.name) and 1<=int(p.name.split('_',1)[0][10:])<=21)


def build_baseline_sources():
    """Apply the exact accepted Experiment9 diff to disposable, hash-checked originals."""
    directory=STATE/'oracle'
    directory.mkdir(parents=True,exist_ok=True)
    for name in CORE: atomic(directory/name,(ROOT/name).read_bytes())
    result=subprocess.run(['patch','--batch','--fuzz=0','-p0','-i',str(P9/'alignment_patch.diff')],
                          cwd=directory,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    sources={name:(directory/name).read_text() for name in CORE}
    return sources


def agent_for(config,seed,fold,oracle=False):
    c=context();set_config(config);c.update(current_config=config,current_fold=fold)
    cls=c['model'].EvalHelper
    if oracle:
        directory=STATE/'oracle';directory.mkdir(parents=True,exist_ok=True)
        src=build_baseline_sources()
        # Use oracle layers with exactly the same accepted V capture.
        net=types.ModuleType('e22_oracle_network');sys.modules[net.__name__]=net
        exec(compile(src['network.py'],'oracle_network.py','exec'),net.__dict__)
        mod=types.ModuleType('e22_oracle_model');sys.modules[mod.__name__]=mod
        exec(compile(src['model.py'].replace('from network import *','from e22_oracle_network import *'),'oracle_model.py','exec'),mod.__dict__)
        cls=mod.EvalHelper
    tr,te=c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):
        a=cls([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    assert (a.val_idx==a.tst_idx).all()
    if oracle:a._e22_raw=a.dataLayer(a.feat)[1]
    return a


def branch(a):return getattr(a.ModalFusion,'evidence22',None)


def model_modules(a):
    modules=[('gfr',a.MessagePassing0),('pln',a.ModalFusion)]
    if a.ModalityDiscriminator is not None:modules.append(('discriminator',a.ModalityDiscriminator))
    return modules


def model_state(a):
    return {name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for name,module in model_modules(a)}


def load_state(a,saved):
    for name,module in model_modules(a):module.load_state_dict(saved[name])


def optim_state(a):
    return copy.deepcopy(dict(mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),
                discriminator=a.optimizer_D.state_dict() if a.optimizer_D else None))


def load_optim(a,saved):
    # Adam's CPU step tensors may otherwise alias the saved restore point.
    saved=copy.deepcopy(saved)
    a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf'])
    if a.optimizer_D:a.optimizer_D.load_state_dict(saved['discriminator'])


def weight_hash(a,auxiliary=False,legacy13=False):
    h=hashlib.sha256()
    for name,module in model_modules(a):
        for key,value in sorted(module.named_parameters()):
            if key.startswith('evidence22.')!=auxiliary:continue
            tag=key if legacy13 else name+'.'+key
            h.update(tag.encode()+value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(a,config):
    groups=a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups+(a.optimizer_D.param_groups if a.optimizer_D else [])
    parameters=[p for _,m in model_modules(a) for p in m.parameters()]
    supplied=[p for g in groups for p in g['params']]
    assert len(supplied)==len(set(map(id,supplied))) and set(map(id,supplied))==set(map(id,parameters))
    assert all(g['lr']==.0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]['weight_decay']==a.optimizer_MF.param_groups[0]['weight_decay']==.11
    assert all(g['weight_decay']==0. for g in a.optimizer_MF.param_groups[1:])
    total=sum(p.numel() for p in parameters)
    assert total==1462957+read_json(MANIFEST)['additional_parameters'][config],(config,total)
    return total


def forward(a,oracle=False):
    import torch
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat)
        hidden,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        if oracle:out,_,_=a.ModalFusion(hidden)
        else:out,_,_=a.ModalFusion(hidden,a._e22_raw)
        assert torch.isfinite(out).all()
    return out.clone()


def metrics(out,indices):
    import numpy as np
    from sklearn.metrics import confusion_matrix,roc_auc_score
    target=context()['y'][indices]
    probability=out.exp().detach().cpu().numpy()[indices];pred=probability.argmax(1)
    cm=confusion_matrix(target,pred,labels=[0,1])
    sen,spe=float(cm[0,0]/cm[0].sum()),float(cm[1,1]/cm[1].sum())
    return dict(acc=float((pred==target).mean()),ba=(sen+spe)/2,
                hard_auc=float(roc_auc_score(np.eye(2)[target],np.eye(2)[pred])),
                probability_auc=float(roc_auc_score(target==0,probability[:,0])),sensitivity_asd=sen,specificity_td=spe,f1_asd=float(2*cm[0,0]/max(1,2*cm[0,0]+cm[1,0]+cm[0,1])),
                brier_asd=float(((probability[:,0]-(target==0))**2).mean()),
                logloss=float(-np.log(probability[np.arange(len(target)),target].clip(1e-15,1)).mean()),
                ece_asd=ece(probability[:,0],target==0))


def step(a):
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')


def validate_cache(cache):
    import numpy as np
    assert cache['valid_idx'].shape==(252,) and sorted(set(range(256))-set(cache['valid_idx']))==[195,225,232,236]
    mask=cache['masks'];wrong=cache['masks_wrong'];assert mask.sum(1).tolist()==[72,56,45,32]
    assert np.array_equal(mask.astype(int)@mask.T.astype(int),wrong.astype(int)@wrong.T.astype(int))
    assert (cache['edge_permutation']!=np.arange(252)).all()
    assert np.array_equal(cache['line_graph'],cache['line_graph'].T) and np.diag(cache['line_graph']).all()
    for key in ('prior_real','prior_neutral','prior_random','prior_anonymous'):
        a=cache[key];assert a.shape==(4,252) and np.isfinite(a).all() and abs(a).max()<=1.
        assert all(a[k,mask[k]].std()>1e-7 for k in range(4)),key
    assert all(not np.array_equal(cache['prior_real'],cache[k]) for k in ('prior_neutral','prior_random','prior_anonymous'))


def signature():
    m=read_json(MANIFEST)
    return canonical({k:m[k] for k in ('source_data_sha256','code_sha256','frozen_file_sha256','protocol','configurations','split_sha256','clip_weight_sha256')})


def task_signature(config,seed):return canonical(dict(experiment_signature=signature(),config=config,seed=seed))


def worker(config,seed):
    import torch
    m=integrity();task=f'{config}_seed{seed}';gpu=int(os.environ['SP22_PHYSICAL_GPU'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)==str(m['tasks'][task]['physical_gpu'])
    lock=acquire_lock(STATE/(task+'.lock'));path=STATE/(task+'.json');cp=STATE/(task+'.pt');sig=task_signature(config,seed)
    if path.exists() and read_json(path)['status']=='COMPLETE':
        result=read_json(path);assert result['signature']==sig and len(result['metrics'])==10;print('SKIP_COMPLETE '+task,flush=True);return
    c=context();set_config(config)
    saved=checkpoint(cp) if cp.exists() else None
    if saved:
        assert saved['signature']==sig;result=saved['result']
        # Warm exactly the historical process caches, then restore its uninterrupted RNG.
        c['network'].get_prompt_text_features()
        if DESIGNS[config]['disease']:c['network'].get_pdai_disease_prototypes()
        restore_rng(saved['rng']);start_fold=saved['fold']+(saved['phase']=='BETWEEN_FOLDS')
        if saved['phase']=='BETWEEN_FOLDS':save_json(path,result)
        print(f'RESUME {task} fold={start_fold} phase={saved["phase"]}',flush=True)
    else:
        seed_rng(seed);start_fold=1
        result=dict(signature=sig,task=task,config=config,seed=seed,physical_gpu=gpu,status='RUNNING',metrics=[],diagnostics=[],counterfactual=[],started_utc=utc(),seconds=0.)
    print(f'WORKER {task} physical_gpu={gpu} logical=cuda:0 signature={sig}',flush=True)
    for fold in range(start_fold,11):
        entry=rng();a=agent_for(config,seed,fold);optimizer_audit(a,config)
        initial,legacy,aux=weight_hash(a),weight_hash(a,legacy13=True),weight_hash(a,True)
        tracker={k:0. for k in gradients(a)};best,best_epoch,best_model,wait,start,prior_seconds=0.,-1,None,0,0,0.;best_optim=None
        if saved and saved['phase']=='EPOCH' and saved['fold']==fold:
            load_state(a,saved['model']);load_optim(a,saved['optim']);restore_rng(saved['rng'])
            best,best_epoch,best_model,wait,start=[saved[k] for k in ('best_acc','best_epoch','best_model','wait','next_epoch')]
            best_optim=saved['best_optim']
            tracker,entry,initial,legacy,aux,prior_seconds=[saved[k] for k in ('tracker','entry_rng','initial','legacy','aux','fold_seconds')]
        saved=None;started=time.monotonic();last=start-1
        for epoch in (range(start,450) if wait<=200 else []):
            step(a)
            if epoch in (0,1) or (epoch+1)%25==0:
                for key,value in gradients(a).items():tracker[key]=max(tracker[key],value)
            out=forward(a);val=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            if val>best:
                best,best_epoch,best_model,wait=val,epoch,model_state(a),0
                best_optim=optim_state(a)
            else:wait+=1
            last=epoch;stop=wait>200 or epoch==449
            if (epoch+1)%25==0 or stop:
                save_checkpoint(cp,dict(signature=sig,phase='EPOCH',fold=fold,next_epoch=epoch+1,model=model_state(a),optim=optim_state(a),rng=rng(),
                    best_acc=best,best_epoch=best_epoch,best_model=best_model,best_optim=best_optim,wait=wait,tracker=tracker,entry_rng=entry,initial=initial,legacy=legacy,aux=aux,
                    fold_seconds=prior_seconds+time.monotonic()-started,result=result))
                print(f'EPOCH fold={fold} epoch={epoch} best_acc={best:.12f} wait={wait}',flush=True)
            if stop:break
        assert best_model is not None
        continuation=rng();load_state(a,best_model);load_optim(a,best_optim);out=forward(a);values=metrics(out,a.tst_idx)
        assert abs(values['acc']-best)<1e-12 and abs(values['ba']-values['hard_auc'])<1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=gpu,task_signature=sig,**values,best_epoch=best_epoch,epochs_executed=last+1,
                 selected_val_acc=best,train_n=len(a.trn_idx),test_n=len(a.tst_idx),val_equals_test=True,initial_trunk_sha256=initial,
                 initial_trunk_legacy13_sha256=legacy,initial_branch_sha256=aux if branch(a) else '',entry_rng_sha256=rng_hash(entry),additional_parameters=m['additional_parameters'][config])
        diagnostic=diagnostics(a,config,seed,fold,out,tracker);diagnostic['task_signature']=sig
        cf=counterfactual(a,config,seed,fold,out) if config in CF_GROUPS else []
        for r in cf:r['task_signature']=sig
        diagnostic['best_epoch']=best_epoch
        row['seconds']=prior_seconds+time.monotonic()-started
        result['metrics'].append(row);result['diagnostics'].append(diagnostic);result['counterfactual'].extend(cf)
        restore_rng(continuation)
        result.update(seconds=sum(r['seconds'] for r in result['metrics']),status='COMPLETE' if fold==10 else 'RUNNING',updated_utc=utc())
        save_checkpoint(cp,dict(signature=sig,phase='BETWEEN_FOLDS',fold=fold,rng=continuation,result=result));save_json(path,result)
        print('FOLD_COMPLETE '+json.dumps(row),flush=True)
        del a,best_model;torch.cuda.empty_cache()
    cp.unlink(missing_ok=True);shutil.rmtree(STATE/'tasks'/task,ignore_errors=True)
    print('TASK_COMPLETE '+task,flush=True);lock.close()


def live_workers():
    import psutil
    result={}
    for p in psutil.process_iter(['pid','cmdline','status']):
        try:
            args=p.info['cmdline'] or []
            if str(EXP/'run.py') in args and '--worker' in args and p.info['status']!='zombie':
                i=args.index('--worker');result[f'{args[i+1]}_seed{args[i+2]}']=p.pid
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def task_results():
    out=[]
    for c,s,g in TASKS:
        path=STATE/f'{c}_seed{s}.json'
        if not path.exists():continue
        t=read_json(path);assert t['signature']==task_signature(c,s) and t['physical_gpu']==g
        assert [r['fold'] for r in t['metrics']]==list(range(1,len(t['metrics'])+1))
        assert len(t['diagnostics'])==len(t['metrics'])
        out.append(t)
    if not out and read_json(MANIFEST)['completed_folds']==EXPECTED_FOLDS:
        # Finalization may be interrupted after deleting temporary checkpoints. Never retrain completed CV.
        metrics_=read_csv(RESULTS/'fold_metrics.csv');diag_=read_csv(RESULTS/'diagnostics.csv');cf_=read_csv(RESULTS/'counterfactual.csv')
        for c,s,g in TASKS:
            select=lambda rr:[r for r in rr if r['config']==c and int(r['seed'])==s]
            rr=select(metrics_);assert len(rr)==10
            for r in rr:
                for k in ('seed','fold','physical_gpu','best_epoch','epochs_executed'):r[k]=int(r[k])
                for k in REPORT_METRICS+['seconds','brier_asd','logloss','ece_asd']:r[k]=float(r[k])
            out.append(dict(task=f'{c}_seed{s}',config=c,seed=s,physical_gpu=g,signature=task_signature(c,s),status='COMPLETE',metrics=rr,diagnostics=select(diag_),counterfactual=select(cf_),seconds=sum(r['seconds'] for r in rr),updated_utc=utc()))
    return out


def refresh():
    lock=acquire_lock(EXP/'logs/.refresh.lock')
    try:
        m=read_json(MANIFEST)
        if m['run_status']=='COMPLETE' and not STATE.exists():return m
        completed=task_results();rows=[r for t in completed for r in t['metrics']]
        ds=[r for t in completed for r in t['diagnostics']];cf=[r for t in completed for r in t['counterfactual']]
        assert len(set((r['config'],r['seed'],r['fold']) for r in rows))==len(rows)
        for t in completed:m['tasks'][t['task']].update(status=t['status'],signature=t['signature'],folds=len(t['metrics']),seconds=t['seconds'],updated_utc=t['updated_utc'])
        m.update(completed_tasks=sum(t['status']=='COMPLETE' for t in completed),completed_folds=len(rows),diagnostic_rows=len(ds),counterfactual_rows=len(cf),updated_utc=utc())
        for name,v in [('fold_metrics',rows),('diagnostics',ds),('counterfactual',cf)]:save_csv(RESULTS/(name+'.csv'),v)
        save_json(MANIFEST,m);return m
    finally:lock.close()


def worker_env(gpu,task='preflight'):
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),SP22_PHYSICAL_GPU=str(gpu),SP22_TASK=task,
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')


def resources():
    import psutil
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    cards=[]
    for line in raw.strip().splitlines():
        index,name,total,used,utilization=[v.strip() for v in line.split(',')]
        cards.append(dict(index=int(index),name=name,total_mib=int(total.split()[0]),used_mib=int(used.split()[0]),utilization=int(utilization.split()[0])))
    rss=0
    for pid in live_workers().values():
        try:rss+=psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:pass
    cg=Path('/sys/fs/cgroup')
    if (cg/'memory.max').exists():
        limit=(cg/'memory.max').read_text().strip();used=int((cg/'memory.current').read_text())
        memstat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        reclaimable=int(memstat.get('inactive_file',0));cgroup_version=2
    else:
        cg=cg/'memory';limit=(cg/'memory.limit_in_bytes').read_text().strip();used=int((cg/'memory.usage_in_bytes').read_text())
        memstat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        reclaimable=int(memstat.get('total_inactive_file',memstat.get('inactive_file',0)));cgroup_version=1
    return dict(utc=utc(),gpu=cards,worker_rss_gib=rss/2**30,host_available_gib=psutil.virtual_memory().available/2**30,
                cgroup_available_gib=(int(limit)-used+reclaimable)/2**30 if limit!='max' else 999.,active_workers=len(live_workers()),
                cgroup_version=cgroup_version,cpu_load_1m=os.getloadavg()[0],cpu_affinity=len(os.sched_getaffinity(0)))


def stable_support(rows,left,right):
    return all(all(v>0 for v in paired(rows,left,right,k)['seeds']) for k in ('acc','ba','probability_auc'))


def array_sha(value):
    import numpy as np
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def ece(p,y):
    import numpy as np
    total=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        keep=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if keep.any():total+=float(keep.mean()*abs(p[keep].mean()-y[keep].mean()))
    return total


def summary(rows):
    import numpy as np
    result={}
    for config in dict.fromkeys(r['config'] for r in rows):
        result[config]={}
        for key in REPORT_METRICS+['brier_asd','logloss','ece_asd','best_epoch']:
            if not all(key in r and r[key]!='' for r in rows if r['config']==config):continue
            seeds=[float(np.mean([float(r[key]) for r in rows if r['config']==config and int(r['seed'])==s])) for s in range(3)]
            result[config][key]=dict(mean=float(np.mean(seeds)),sd=float(np.std(seeds)),seeds=seeds)
    return result


def paired(rows,left,right,key):
    import numpy as np
    refs=read_csv(RESULTS/'historical_references.csv') if right in HISTORICAL_REFERENCES else []
    rr=HISTORICAL_REFERENCES.get(right,right)
    lookup={(r['config'],int(r['seed']),int(r['fold'])):float(r[key]) for r in rows+refs}
    dif=np.array([[lookup[left,s,f]-lookup[rr,s,f] for f in range(1,11)] for s in range(3)])
    return dict(mean=float(dif.mean()),seeds=dif.mean(1).tolist(),wins=int((dif>1e-12).sum()),ties=int((abs(dif)<=1e-12).sum()),losses=int((dif< -1e-12).sum()))


def patched_sources():
    from integration_builder22 import patched_sources22
    sources=build_baseline_sources();sources['main.py']=(ROOT/'main.py').read_text()
    return patched_sources22(sources)


def make_sources(task='preflight'):
    directory=STATE/'tasks'/task/'sources';directory.mkdir(parents=True,exist_ok=True)
    for name in CORE:atomic(directory/name,(ROOT/name).read_bytes())
    r=subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(EXP/'integration_patch.diff')],cwd=directory,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    for name in CORE:assert sha(directory/name)==read_json(MANIFEST)['patched_source_sha256'][name],name
    link=directory/'CLIP'
    if not link.exists():link.symlink_to(ROOT/'CLIP',target_is_directory=True)
    work=directory.parent/'work';work.mkdir(exist_ok=True)
    for name in ('attn','graph'):(work/name).mkdir(exist_ok=True)
    os.chdir(work)
    return directory


def context():
    global _CTX
    if _CTX is None:
        import torch,numpy as np
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1,'GPU training only; exactly one visible GPU per worker'
        torch.use_deterministic_algorithms(True)
        directory=make_sources(os.environ.get('SP22_TASK','preflight'))
        sys.path.insert(0,str(directory));sys.path.insert(0,str(EXP))
        sys.modules['e22_runtime']=sys.modules[__name__]
        import network,model
        x,y,folds,audit=data()
        with np.load(CACHE,allow_pickle=False) as packed:cache={k:packed[k].copy() for k in packed.files}
        validate_cache(cache)
        _CTX=dict(torch=torch,network=network,model=model,x=x,y=y,folds=folds,audit=audit,cache=cache)
    return _CTX


def set_config(config):
    import torch
    torch.use_deterministic_algorithms(True)
    s=SPECS[config]
    os.environ.update(PDAI_PROMPT='P3' if s.base=='E9' else 'P0',PDAI_DISEASE='1' if s.base=='E9' else '0',PDAI_ADV='0',PDAI_PROMPTS_FILE=str(P9/'prompts.json'))


def install_evidence22(agent):
    from contract22 import install_evidence22 as attach
    c=context();agent._e22_raw=agent.dataLayer(agent.feat)[1]
    attach(agent,cache=c['cache'],train_idx=agent.trn_idx,raw_modalities=agent._e22_raw,device=agent.dev,config=c['current_config'],seed=int(agent.hyperpm.seed))


def gradients(a):
    out={}
    for name,p in branch(a).named_parameters():
        g=p.grad
        if g is not None:assert bool(g.isfinite().all()),('NONFINITE_GRADIENT',name)
        out[name]=float(g.detach().double().square().sum().sqrt()) if g is not None else 0.
    return out


def cache_audit(cache):
    import numpy as np
    validate_cache(cache)
    p11=ROOT/'experiment/experiment11_roi_mapping_recovery/results'
    atlas=read_csv(p11/'roi_111_labels.csv');edges=read_csv(p11/'roi_256_named_mapping.csv')
    assert len(atlas)==111 and len(edges)==256
    atlas={int(r['roi_index_zero_based']):r for r in atlas}
    assert sorted(atlas)==list(range(111))
    assert [int(r['correlation_feature_index']) for r in edges if r['usable_for_roi_semantic_prompt']=='True']==cache['valid_idx'].tolist()
    tri=np.triu_indices(111,1);texts=[]
    for e,r in enumerate(edges):
        assert int(r['correlation_feature_index'])==e and r['verification_status']=='VERIFIED_NUMERIC_UNIQUE'
        if e not in cache['valid_idx']:continue
        i,j=int(r['roi_i_zero_based']),int(r['roi_j_zero_based']);original=int(r['original_connection_index'])
        idx=int(np.where(cache['valid_idx']==e)[0][0])
        assert (i,j)==(int(tri[0][original]),int(tri[1][original]))==tuple(cache['roi_endpoints'][idx])
        assert atlas[i]['name_verified']==atlas[j]['name_verified']=='True'
        assert atlas[i]['column_order_verified']==atlas[j]['column_order_verified']=='True'
        assert int(r['roi_i_atlas_value'])==int(atlas[i]['atlas_integer_value']) and int(r['roi_j_atlas_value'])==int(atlas[j]['atlas_integer_value'])
        assert 'UNKNOWN' not in (atlas[i]['roi_name'],atlas[j]['roi_name'])
        texts.append(f"Functional connectivity between {atlas[i]['roi_name']} and {atlas[j]['roi_name']}.")
    ends=cache['roi_endpoints'];line=(ends[:,None,:,None]==ends[None,:,None,:]).any(-1).any(-1)
    assert np.array_equal(line,cache['line_graph'])
    m20=read_json(cache_origin().parent/'manifest.json')
    assert m20['semantic_audit']['gram_error']<1e-5
    return dict(verified_edges=252,excluded=[195,225,232,236],candidate_counts=cache['masks'].sum(1).tolist(),
        candidate_overlap=(cache['masks'].astype(int)@cache['masks'].T.astype(int)).tolist(),
        line_graph_symmetric=True,line_graph_self_loop=True,line_graph_directed_entries=int(line.sum()),
        gram_error=m20['semantic_audit']['gram_error'],edge_name_text_sha256=canonical(texts),status='NONCONSTANT_PRIOR',
        candidate_prior_std={k:[float(cache[k][d,cache['masks'][d]].std()) for d in range(4)] for k in ['prior_real','prior_neutral','prior_random']})


def cache_origin():
    candidates=[Path('/tmp/experiment20_grounded_concept_graph_alignment/semantic_cache.npz'),P20/'semantic_cache.npz']
    for p in candidates:
        if p.is_file():return p
    raise RuntimeError('BLOCKED: frozen Exp20 semantic cache unavailable; rebuild requires identical RN50 + provenance verification')


def verify_historical():
    m21=read_json(P21/'manifest.json')
    assert m21['run_status']=='COMPLETE' and m21['historical_parity_180']['status']=='PASS'
    source=P21/'results/fold_metrics.csv';rows=read_csv(source)
    assert len(rows)==600 and len({(r['config'],r['seed'],r['fold']) for r in rows})==600
    assert sha(ROOT/'data/ABIDE/processed_standard_data.csv')==m21['source_data_sha256'][str(ROOT/'data/ABIDE/processed_standard_data.csv')]
    _,_,folds,_=data();split=canonical([dict(train=a.tolist(),test=b.tolist()) for a,b in folds]);assert split==m21['split_sha256']
    subset=[r for r in rows if r['config'] in HISTORICAL_REFERENCES.values()]
    assert len(subset)==180 and all(math.isfinite(float(r[k])) for r in subset for k in METRICS)
    for r in subset:
        tr,te=folds[int(r['fold'])-1]
        assert int(r['train_n'])==len(tr) and int(r['test_n'])==len(te) and r['val_equals_test']=='True'
        assert abs(float(r['ba'])-float(r['hard_auc']))<1e-12
    if 'results_sha256' in m21:assert sha(source)==m21['results_sha256']['fold_metrics']
    return subset,dict(status='PASS',rows=180,source_sha256=sha(source),split_sha256=split,
        historical_parity_180=m21['historical_parity_180'],continuous_fold_rng_identical=False,
        caution='paired split only; later folds have method-specific early stopping and RNG divergence')


def prepare():
    import numpy as np
    if MANIFEST.exists():return integrity(full=True)
    assert HANDOFF.is_file(),'Root handoff missing; cannot recreate history'
    for p in (LOGS,RESULTS,STATE):p.mkdir(parents=True,exist_ok=True)
    m21=read_json(P21/'manifest.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    repo=Path('/tmp/spromptgl-publish21')
    if subprocess.run(['git','cat-file','-e',AUDIT_COMMIT],cwd=ROOT,capture_output=True).returncode==0:repo=ROOT
    comparisons={}
    for name in CORE:
        old=subprocess.check_output(['git','show',AUDIT_COMMIT+':'+name],cwd=repo)
        current=(ROOT/name).read_bytes();h=hashlib.sha256(current).hexdigest()
        assert old==current and h==m21['source_comparison'][name]['local'],('CORE_SHA_MISMATCH',name)
        comparisons[name]=dict(local=h,audit=hashlib.sha256(old).hexdigest(),equal=True)
    blocks=re.findall(r'```python\n(.*?)```',(ROOT/'prompt').read_text(),re.S)
    for filename,prefix in [('signed_evidence22.py','"""Experiment 22: signed'),('contract22.py','"""Frozen Exp22 registrations'),('integration_builder22.py','"""Experiment22 exact surgical')]:
        assert (EXP/filename).read_text()==next(v for v in blocks if v.startswith(prefix)),('AUTHORITATIVE_SOURCE_CHANGED',filename)
    origin=cache_origin();m20=read_json(origin.parent/'manifest.json')
    assert m20['run_status']=='COMPLETE' and sha(origin)==m20['frozen_file_sha256']['semantic_cache.npz']==m21['source_data_sha256'][str(origin)]
    atomic(CACHE,origin.read_bytes())
    with np.load(CACHE,allow_pickle=False) as packed:cache={k:packed[k].copy() for k in packed.files}
    semantic=cache_audit(cache);refs,hist_audit=verify_historical();save_csv(RESULTS/'historical_references.csv',refs)
    x,y,folds,da=data();sources=patched_sources()
    patch=''.join(''.join(difflib.unified_diff((ROOT/n).read_text().splitlines(True),sources[n].splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n in CORE)
    atomic(EXP/'integration_patch.diff',patch)
    assert sources['network.py'].index("x = self.evidence22(x, raw_modalities)")<sources['network.py'].index('text_features = self.prompt_text_features')
    assert sources['model.py'].count('self.ModalFusion(feat, self._e22_raw)')==3
    loss=sources['model.py'].index('loss = cls_loss + all_graph_loss')
    assert loss<sources['model.py'].index('loss = loss + branch.auxiliary_loss()',loss)<sources['model.py'].index('loss = loss + 0.1 * disease_loss',loss)
    params={c:(36865 if s.base=='E9' else 0)+(20525 if s.graph else 19437) for c,s in SPECS.items()}
    design=dict(configurations={c:asdict(s) for c,s in SPECS.items()},tasks=[dict(config=c,seed=s,physical_gpu=g) for c,s,g in TASKS],protocol=PROTOCOL,
        pairs=PAIRS,CF_GROUPS=CF_GROUPS,CF_ACTIONS=CF_ACTIONS,historical_references=HISTORICAL_REFERENCES,
        hyperparameters=dict(KL=.002,prior_tau=.8,contrast_tau=.4,eta_upper=.35,eta_initial=.1,graph_cos_coef=.4,graph_strength_coef=.45,residual_limit=.1),
        expected_counts=dict(configs=18,tasks=54,folds=540,diagnostics=540,cf=2340,cf_na=270,historical=180),
        clinical_alignment='CLINICAL_ALIGNMENT_NOT_ESTABLISHED; verified identities 0/871; training-supervised weak disease association',
        success_criterion='all 3 seed deltas ACC/BA/prob-AUC positive vs strong baseline AND numeric+neutral controls; anatomy control for medical claim; same-checkpoint sensitivity required, no significance claim',
        stop_criterion='integrity/parity/cache/determinism violation; no test-performance-driven editing or prescreen',
        input_prompt_paths=[str(P9/'prompts.json'),str(P19/'prompts.json')],graph_type='verified FC connection-as-node line graph',frozen_before_training=True)
    save_json(EXP/'design.json',design)
    history={str(p.resolve()):sha(p) for f in historical_folders() for p in f.rglob('*') if p.is_file()}
    dependencies=[ROOT/n for n in CORE]+[ROOT/'data/ABIDE/processed_standard_data.csv',ROOT/'data/ABIDE/modal_feat_dict.npy',ROOT/'prompt',
        P9/'alignment_patch.diff',P9/'prompts.json',P19/'prompts.json',P19/'run.py',P19/'integration_patch.diff',
        P21/'run.py',P21/'adaptive_evidence.py',P21/'manifest.json',P21/'results/report.md',P21/'results/fold_metrics.csv',
        ROOT/'experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv',ROOT/'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv']
    for p in (P19/'results').glob('*'): 
        if p.is_file():dependencies.append(p)
    dependencies+=list(ROOT.glob('*.pdf'))
    dependencies += [ROOT/'CLIP/clip'/n for n in ['clip.py','model.py','simple_tokenizer.py']]
    weights=Path.home()/'.cache/clip/RN50.pt';assert sha(weights)==m21['clip_weight_sha256']
    import torch,sklearn
    m=dict(experiment='SABER22 Concept Conditioned Signed Connectivity',run_status='PREPARED',prepared_utc=utc(),root=str(ROOT),experiment_directory=str(EXP),temporary_state=str(STATE),
        git_head_at_prepare=head,github_audit_commit=AUDIT_COMMIT,source_comparison=comparisons,protocol=PROTOCOL,configurations=DESIGNS,
        data_audit=da,semantic_audit=semantic,historical_reference_audit=hist_audit,additional_parameters=params,
        cache_provenance=dict(original_path=str(origin),original_sha256=sha(origin),exp20_manifest_sha256=sha(origin.parent/'manifest.json'),exp20_cache_generator_sha256=sha(origin.parent/'cache_reference.py'),exp20_semantic_audit=m20['semantic_audit'],runtime_uses_only_local_CACHE=True),
        source_data_sha256={str(p):sha(p) for p in dependencies},protected_history_sha256=history,
        code_sha256={n:sha(EXP/n) for n in ['run.py','signed_evidence22.py','contract22.py','integration_builder22.py']},
        frozen_file_sha256={n:sha(EXP/n) for n in ['design.json','integration_patch.diff','semantic_cache.npz','results/historical_references.csv']},
        patched_source_sha256={n:hashlib.sha256(sources[n].encode()).hexdigest() for n in CORE},cache_array_sha256={k:array_sha(v) for k,v in cache.items()},
        clip_weight_sha256=sha(weights),split_sha256=hist_audit['split_sha256'],handoff_prefix_bytes=HANDOFF.stat().st_size,handoff_prefix_sha256=sha(HANDOFF),
        versions=dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,scikit_learn=sklearn.__version__),
        pdai_environment=dict(PDAI_PROMPT='P0 or P3 exactly per spec',PDAI_DISEASE='0 or 1 exactly per spec',PDAI_ADV='0',PDAI_PROMPTS_FILE=str(P9/'prompts.json')),
        source_audit=dict(insertion='Encoder+FeedForward -> evidence22 -> original SPGS/RES',GFR_i1_overwrite_preserved=True,ALoss_not_backpropagated=True,test_equals_val=True,global_preCV_RFE_standardization=True,all871_transductive=True,
            original_paper_url='https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf',MMGPL_url='https://arxiv.org/html/2312.14574v2',
            source_c_adapter='single diff against root originals; Exp9 diff used solely to reconstruct authoritative baseline, not applied twice',
            clinical_alignment='CLINICAL_ALIGNMENT_NOT_ESTABLISHED',verified_clinical_patient_identities=0),
        preflight=None,completed_tasks=0,completed_folds=0,diagnostic_rows=0,counterfactual_rows=0,
        tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS})
    save_json(MANIFEST,m);write_readme(m);report()
    print('PREPARED '+json.dumps(dict(arms=18,tasks=54,new_folds=540,historical=180,cache_sha256=sha(CACHE))),flush=True)
    return m


def integrity(full=False):
    m=read_json(MANIFEST)
    for p,h in m['source_data_sha256'].items():assert sha(p)==h,('INTEGRITY source/data changed',p)
    for field in ('code_sha256','frozen_file_sha256'):
        for name,h in m[field].items():assert sha(EXP/name)==h,('INTEGRITY frozen file changed',name)
    if full:
        for p,h in m['protected_history_sha256'].items():assert sha(p)==h,('INTEGRITY history changed',p)
        assert sha(Path.home()/'.cache/clip/RN50.pt')==m['clip_weight_sha256']
        assert hashlib.sha256(HANDOFF.read_bytes()[:m['handoff_prefix_bytes']]).hexdigest()==m['handoff_prefix_sha256'],'INTEGRITY handoff prefix changed'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['git_head_at_prepare']
    return m
@contextlib.contextmanager
def intervention(a,action,fold):
    import numpy as np,torch
    b=branch(a);cache=context()['cache']
    buffers={k:v.detach().clone() for k,v in b.named_buffers()}
    original=(b.condition,b.variant,b.intervention,a._e22_raw,rng())
    try:
        if action=='NEUTRAL':b.prior.copy_(torch.as_tensor(cache['prior_neutral'],device=a.dev))
        elif action=='ISOMETRIC':b.prior.copy_(torch.as_tensor(cache['prior_random'],device=a.dev))
        elif action=='WRONG_ANATOMY':b.mask.copy_(torch.as_tensor(cache['masks_wrong'],device=a.dev))
        elif action=='NUMERIC_ONLY':b.prior.zero_()
        elif action=='ADAPT_OFF':b.intervention='ADAPT_OFF'
        elif action=='SIGN_OFF':b.variant='UNSIGNED'
        elif action=='SHARED_DOMAIN':b.variant='SHARED'
        elif action=='GRAPH_SEMANTIC_OFF':b.intervention='GRAPH_SEMANTIC_OFF'
        elif action=='GRAPH_STRENGTH_OFF':b.intervention='GRAPH_STRENGTH_OFF'
        elif action=='GRAPH_OFF':b.intervention='GRAPH_OFF'
        elif action=='BRANCH_OFF':b.intervention='BRANCH_OFF'
        elif action=='FC_BRANCH_SHUFFLED':
            rr=np.random.default_rng(2200000+fold);idx=np.asarray(a.tst_idx);perm=rr.permutation(idx)
            while np.any(perm==idx):perm=rr.permutation(idx)
            altered=a._e22_raw[3].clone();altered[idx]=altered[perm];a._e22_raw=[*a._e22_raw[:3],altered]
        elif action!='REAL':raise ValueError(action)
        yield
    finally:
        for k,v in b.named_buffers():v.copy_(buffers[k])
        b.condition,b.variant,b.intervention,a._e22_raw=original[:4]
        assert rng_hash(rng())==rng_hash(original[4]),'counterfactual consumed global RNG'
        restore_rng(original[4])


def counterfactual(a,config,seed,fold,real):
    import torch
    b=branch(a);te=a.tst_idx;rows=[];target=metrics(real,te)
    original=model_state(a);raw=[v.clone() for v in a._e22_raw];entry=rng()
    for action in CONDITIONS:
        row=dict(config=config,seed=seed,fold=fold,condition=action,applicable=True,retrained=False)
        if action in ('GRAPH_SEMANTIC_OFF','GRAPH_STRENGTH_OFF','GRAPH_OFF') and not b.use_graph:
            rows.append(dict(row,applicable=False,reason='NA: local arm, no graph'));continue
        with intervention(a,action,fold):
            out=forward(a);v=metrics(out,te)
            row.update(**v,prediction_flip=float((out[te].argmax(1)!=real[te].argmax(1)).double().mean()),
                probability_abs_change=float((out[te].exp()-real[te].exp()).abs().mean()),
                logprob_abs_change=float((out[te]-real[te]).abs().mean()),
                acc_change=v['acc']-target['acc'],ba_change=v['ba']-target['ba'],probability_auc_change=v['probability_auc']-target['probability_auc'],
                hard_auc_change=v['hard_auc']-target['hard_auc'])
        assert torch.equal(forward(a),real),('counterfactual REAL restoration',action)
        assert_equal_tree(original,model_state(a));assert all(torch.equal(v,w) for v,w in zip(raw,a._e22_raw))
        rows.append(row)
    assert rng_hash(entry)==rng_hash(rng()) and len(rows)==13
    return rows


def diagnostics(a,config,seed,fold,real,tracker):
    import numpy as np,torch
    from torch.nn import functional as F
    b=branch(a);spec=SPECS[config];te=a.tst_idx;tr=a.trn_idx;capture={}
    handle=b.register_forward_hook(lambda module,inp,out:capture.update(base=inp[0].detach().clone(),new=out.detach().clone()))
    try:assert torch.equal(forward(a),real)
    finally:handle.remove()
    base,new=capture['base'],capture['new'];ll=copy.deepcopy(b.last)
    att=ll['att'];delta=ll['delta'];rel=delta.norm(dim=-1)/(base[:,3].norm(dim=-1)+1e-8)
    assert rel.max()<=.100001 and torch.equal(new[:,:3],base[:,:3])
    norm=lambda v:float(v.detach().norm())
    counts=b.mask.sum(1);entropy=-(att*att.clamp_min(1e-30).log()).sum(-1)
    strength=att.sum(1);union=b.mask.any(0);eta_p=ll['eta_plus'];eta_n=ll['eta_minus']
    row=dict(config=config,seed=seed,fold=fold,base=spec.base,condition=spec.condition,variant=spec.variant,use_graph=spec.graph,complement=spec.complement,
        prior_sha256=array_sha(b.prior.cpu().numpy()),mask_sha256=array_sha(b.mask.cpu().numpy()),prediction_nonfinite=False,
        predicted_asd=int((real[te].argmax(1)==0).sum()),predicted_td=int((real[te].argmax(1)==1).sum()),
        train_fc_mean_min=float(b.center.min()),train_fc_mean_max=float(b.center.max()),train_fc_std_min=float(b.scale.min()),train_fc_std_max=float(b.scale.max()),
        train_contrast_mean=float(b.train_contrast.mean()),train_contrast_std=float(b.train_contrast.std(unbiased=False)),train_contrast_min=float(b.train_contrast.min()),train_contrast_max=float(b.train_contrast.max()),
        contrast_positive_fraction=float((b.train_contrast>0).float().mean()),train_reference_train_only=True,train_reference_uses_training_labels=True,
        alpha=float(b.alpha.detach()),eta_plus=json.dumps(eta_p.cpu().tolist()),eta_minus=json.dumps(eta_n.cpu().tolist()),
        eta_plus_delta=json.dumps((eta_p-.1).cpu().tolist()),eta_minus_delta=json.dumps((eta_n-.1).cpu().tolist()),
        positive_signed_exposure=json.dumps((att[te]*ll['signed'][te,None,:].clamp_min(0)).sum(-1).mean(0).cpu().tolist()),
        negative_signed_exposure=json.dumps((att[te]*(-ll['signed'][te,None,:]).clamp_min(0)).sum(-1).mean(0).cpu().tolist()),
        kl_train=float(ll['kl'][tr].mean()),kl_test=float(ll['kl'][te].mean()),train_kl_loss=float(b.auxiliary_loss()),
        normalized_attention_entropy=json.dumps((entropy[te]/counts.float().log()).mean(0).cpu().tolist()),
        attention_entropy=json.dumps(entropy[te].mean(0).cpu().tolist()),attention_static_abs_change=float((att[te]-ll['p0']).abs().mean()),
        attention_correction_abs_mean=float(ll['correction'][te][:,b.mask].abs().mean()),
        prior_candidate_std=json.dumps([float(b.prior[k,b.mask[k]].std(unbiased=False)) for k in range(4)]),
        top_feature_indices=json.dumps([[int(b.valid_idx[i]) for i in b.prior[k].masked_fill(~b.mask[k],-1e4).topk(5).indices.cpu().tolist()] for k in range(4)]),
        strength_cv=float((strength[te][:,union].std(-1,unbiased=False)/strength[te][:,union].mean(-1).clamp_min(1e-8)).mean()),
        strength_domain_spread=json.dumps([float(strength[te][:,b.mask[k]].std(-1,unbiased=False).mean()) for k in range(4)]),
        strength_histogram=json.dumps(torch.histc(strength[te].float().cpu(),bins=10,min=0,max=.2).cpu().tolist()),
        attention_histogram=json.dumps(torch.histc(att[te][:,b.mask].float().cpu(),bins=10,min=0,max=.2).cpu().tolist()),
        residual_relative_mean=float(rel[te].mean()),residual_relative_max=float(rel.max()),residual_norm_mean=float(delta[te].norm(dim=-1).mean()),
        residual_nonzero_fraction=float((rel[te]>1e-8).float().mean()),noncorr_delta_max=float((new[:,:3]-base[:,:3]).abs().max()),
        representation_cosine=float(F.cosine_similarity(base[te,3],new[te,3],dim=-1).mean()),
        residual_base_cosine_abs=float(F.cosine_similarity(base[te,3],delta[te],dim=-1).abs().mean()),
        gradient_norms=json.dumps(tracker,sort_keys=True),all_core_parameter_gradients_observed=all(v>0 and math.isfinite(v) for v in tracker.values()),
        gradient_check_scope='max norm across epochs 0,1 and every25; finite assertions at each sampling',prior_requires_grad=b.prior.requires_grad,
        disease_gamma=a.ModalFusion.last_disease_gamma,
        disease_probability_correlation=correlation(a.ModalFusion.last_disease_logits[te].softmax(-1)[:,0].detach().cpu().numpy(),a.ModalFusion.last_base_logprob[te].exp()[:,0].detach().cpu().numpy()) if spec.base=='E9' else '')
    with torch.no_grad():
        z,d,g=b._read(a._e22_raw[3]);x=torch.stack([torch.tanh(d/3),torch.tanh(d.abs()/3),z.tanh()],-1)
        encoded=b.encoder(x);h=b.edge_norm(encoded+b.edge_id.weight[None]);h0=b.edge_norm(encoded)
        row['edge_id_feature_change']=float((h[te]-h0[te]).norm(dim=-1).mean())
        row['branch_parameter_norm']=math.sqrt(sum(norm(p)**2 for p in b.parameters()))
        row.update(graph_cosine_factor_mean=0.,graph_strength_factor_mean=0.,graph_raw_weight_semantic_abs_change=0.,graph_raw_weight_strength_abs_change=0.,message_relative_change=0.,adjacency_patient_variance=0.,adjacency_outside_line_nonzero=0)
        if b.use_graph:
            updated,info=b._message(h,att,g);numeric=(g[:,b.src]*g[:,b.dst]).clamp_min(1e-8)
            cosfactor=torch.exp(.4*(info['cos']-.5)) if b.condition!='NUMERIC' else torch.ones_like(numeric)
            magfactor=torch.exp(.45*(info['logstrength'][:,b.src]+info['logstrength'][:,b.dst])*.5) if b.condition!='NUMERIC' else torch.ones_like(numeric)
            actual=numeric*cosfactor*magfactor;den=g.new_ones((len(g),252)).index_add(1,b.dst,actual)
            row.update(graph_cosine_factor_mean=float(cosfactor[te].mean()),graph_strength_factor_mean=float(magfactor[te].mean()),
                graph_raw_weight_semantic_abs_change=float((actual[te]-numeric[te]).abs().mean()),
                graph_raw_weight_strength_abs_change=float((actual[te]-numeric[te]*cosfactor[te]).abs().mean()),
                message_relative_change=float(((updated[te]-h[te]).norm(dim=-1)/h[te].norm(dim=-1).clamp_min(1e-8)).mean()),
                adjacency_patient_variance=float((actual[te]/den[te][:,b.dst]).var(0,unbiased=False).mean()))
    for action,prefix in [('BRANCH_OFF','branch_off'),('GRAPH_OFF','graph_off'),('GRAPH_STRENGTH_OFF','strength_off')]:
        if action!='BRANCH_OFF' and not b.use_graph:continue
        with intervention(a,action,fold):out=forward(a);v=metrics(out,te)
        row.update({prefix+'_'+k:v[k] for k in METRICS})
        row[prefix+'_probability_change']=float((out[te].exp()-real[te].exp()).abs().mean())
        assert torch.equal(forward(a),real)
    return row
def preflight():
    import numpy as np,torch
    from torch.nn import functional as F
    from signed_evidence22 import SignedConceptEvidence22
    m=integrity(full=True)
    if m.get('preflight') and m['preflight']['status']=='PASS':print('PREFLIGHT_ALREADY_PASS',flush=True);return m['preflight']
    c=context();seed_rng(22022);checks=dict(status='RUNNING',started_utc=utc(),checks={},baseline_parity={})
    torch.use_deterministic_algorithms(True)
    raw=torch.randn(15,256,device='cuda');base=torch.randn(15,4,36,device='cuda');tr=np.arange(10);labels=torch.arange(10,device='cuda')%2
    parts=[torch.zeros(15,n,device='cuda') for n in (48,6,10)]+[raw]
    for condition in ('REAL','NUMERIC','NEUTRAL','WRONG_ANATOMY','ISOMETRIC'):
        for graph in (False,True):
            for variant in ('SIGNED','UNSIGNED','SHARED'):
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(220000)
                    b=SignedConceptEvidence22(condition=condition,use_graph=graph,cache=c['cache'],raw_corr=raw,train_idx=tr,train_labels=labels,variant=variant).cuda()
                assert sum(p.numel() for p in b.parameters())==(20525 if graph else 19437)
                assert torch.equal(b(base,parts),base) and not b.prior.requires_grad and not b.train_contrast.requires_grad
                assert b.last['att'][:,~b.mask].count_nonzero()==0 and torch.allclose(b.last['att'].sum(-1),torch.ones(15,4,device='cuda'),atol=1e-6)
                assert float(b.auxiliary_loss())>=0
                assert not any(isinstance(layer,torch.nn.Linear) and layer.out_features==1024 for layer in b.modules())
                opt=torch.optim.Adam(b.parameters(),lr=.0038,weight_decay=0.)
                for it in range(2):
                    opt.zero_grad();out=b(base,parts);loss=(out*torch.arange(36,device='cuda')).mean()+b.auxiliary_loss();loss.backward()
                    assert b.alpha.grad is not None and torch.isfinite(b.alpha.grad) and float(b.alpha.grad.abs())>0
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (b.pos_logits,b.neg_logits))
                    assert b.pos_logits.grad.abs().sum()>0 and b.neg_logits.grad.abs().sum()>0
                    if it==1:
                        for name,p in b.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,(condition,graph,variant,name)
                    opt.step()
                with torch.no_grad():b.alpha.fill_(100);out=b(base,parts)
                assert torch.equal(out[:,:3],base[:,:3]) and ((out-base)[:,3].norm(dim=-1)/base[:,3].norm(dim=-1)).max()<=.100001
    for complement in (False,True):
        b=SignedConceptEvidence22(condition='REAL',use_graph=True,cache=c['cache'],raw_corr=raw,train_idx=tr,train_labels=labels,complement=complement).cuda()
        with torch.no_grad():b.alpha.fill_(.7);out=b(base,parts)
        assert ((out-base)[:,3].norm(dim=-1)/base[:,3].norm(dim=-1)).max()<=.100001
        if complement:assert F.cosine_similarity(base[:,3],b.last['delta']).abs().max()<1e-5
    checks['checks']['synthetic_30_cases_plus_complement']='PASS: parameters19437/20525, exact zero-alpha, candidate weights, train-KL, 8 coefficient gradients, 2-step all gradients,10% bound'
    d=torch.randn(15,252,device='cuda');d[1]=-d[0]
    with torch.no_grad():b.reliability.fill_(.7)
    b.variant='SIGNED';aa=b._attention(d)[0];assert (aa[0]-aa[1]).abs().max()>1e-6
    b.variant='UNSIGNED';aa=b._attention(d)[0];assert torch.equal(aa[0],aa[1])
    b2=SignedConceptEvidence22(condition='REAL',use_graph=False,cache=c['cache'],raw_corr=raw,train_idx=tr,train_labels=1-labels).cuda()
    assert torch.allclose(b2.train_contrast,-SignedConceptEvidence22(condition='REAL',use_graph=False,cache=c['cache'],raw_corr=raw,train_idx=tr,train_labels=labels).cuda().train_contrast,atol=1e-6)
    with torch.no_grad():b.pos_logits.copy_(torch.tensor([-2.,-1.,1.,2.],device='cuda'))
    b.variant='SIGNED';co=b._attention(d)[4];b.variant='SHARED';shared=b._attention(d)[4]
    # Divide the frozen content factor and common signed value to isolate coefficients.
    positive=(torch.tanh(d/2)*b.reliability).clamp_min(0)
    expect=.35*b.pos_logits.sigmoid().mean()*positive[:,None,:]-.35*b.neg_logits.sigmoid().mean()*(-torch.tanh(d/2)*b.reliability).clamp_min(0)[:,None,:]
    assert torch.allclose(shared,expect*(1+.5*b.prior)[None],atol=1e-7) and (co-shared).abs().max()>1e-5
    checks['checks']['signed_and_concept_specificity']='PASS: opposite d gives different SIGNED and identical UNSIGNED attention; swapped train labels reverses contrast; SHARED coefficient algebra exact'
    # Full verified252 support dense reference, values and gradients.
    n=3;h=torch.randn(n,252,32,device='cuda',requires_grad=True);w=torch.rand(n,len(b.src),device='cuda',requires_grad=True)
    den=h.new_ones(n,252).index_add(1,b.dst,w)
    sparse=(h+torch.zeros_like(h).index_add(1,b.dst,h[:,b.src]*w[...,None]))/den[...,None]
    dense=torch.diag_embed(torch.ones(n,252,device='cuda'));dense[:,b.dst,b.src]=w;dense=dense/dense.sum(-1,keepdim=True)
    expected=torch.bmm(dense,h);assert torch.allclose(sparse,expected,atol=1e-5,rtol=1e-5)
    gs=torch.autograd.grad(sparse.square().mean(),(h,w),retain_graph=True);gd=torch.autograd.grad(expected.square().mean(),(h,w))
    assert all(torch.allclose(x,z,atol=1e-5,rtol=1e-5) for x,z in zip(gs,gd))
    assert torch.allclose(dense.sum(-1),torch.ones(n,252,device='cuda'),atol=1e-6) and dense[:,~b.support].count_nonzero()==0
    checks['checks']['deterministic_sparse_dense']=dict(status='PASS',max_value_error=float((sparse-expected).abs().max()),max_gradient_error=max(float((x-y).abs().max()) for x,y in zip(gs,gd)),tolerance=1e-5,deterministic=torch.are_deterministic_algorithms_enabled())
    b.variant='SIGNED';b.intervention='REAL'
    att=b.mask.float()[None].repeat(n,1,1)/b.mask.sum(1)[None,:,None];att2=att.clone()
    exclusive=b.mask & (b.mask.sum(0)==1)[None]
    domain=next(k for k in range(4) if int(exclusive[k].sum())>=2)
    e,f=exclusive[domain].nonzero().flatten()[:2];amount=.4*att2[:,domain,f]
    att2[:,domain,e]+=amount;att2[:,domain,f]-=amount
    profile=lambda at:F.normalize(at.transpose(1,2),dim=-1,eps=1e-8)
    assert torch.equal(profile(att),profile(att2))
    gate=torch.full((n,252),.6,device='cuda')
    with torch.no_grad():
        actual,info=b._message(h.detach(),att,gate);changed,info2=b._message(h.detach(),att2,gate)
        assert (info['strength']-info2['strength']).abs().max()>1e-5 and (actual-changed).abs().max()>1e-6
        for intervention_ in ('GRAPH_STRENGTH_OFF','GRAPH_SEMANTIC_OFF'):
            b.intervention=intervention_;a1,_=b._message(h.detach(),att,gate);a2,_=b._message(h.detach(),att2,gate)
            assert torch.equal(a1,a2),intervention_
        b.intervention='REAL'
    checks['checks']['graph_absolute_magnitude']='PASS: equal normalized profiles but redistributed domain mass changes message; cosine-only/semantic-off remain identical; self-loop+nonuniform neighbours prevent row-normalization cancellation'
    del b,b2,h,w,dense,sparse,expected;torch.cuda.empty_cache()
    baselines={}
    def original_gradients(agent):
        return {name+'.'+key:p.grad.detach().cpu().clone() if p.grad is not None else None for name,mod in model_modules(agent) for key,p in mod.named_parameters() if not key.startswith('evidence22.')}
    def original_state(agent):
        state=model_state(agent);state['pln']={k:v for k,v in state['pln'].items() if not k.startswith('evidence22.')};return state
    def original_optim(agent):
        state=optim_state(agent)
        if branch(agent) is not None:
            extra=state['mf']['param_groups'].pop()['params'];state['mf']['state']={k:v for k,v in state['mf']['state'].items() if k not in extra}
        return state
    for cfg in ('M00','M10'):
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None
        seed_rng(0);entry=rng();native=agent_for(cfg,0,1,oracle=True)
        initial=model_state(native);post=rng();prediction=forward(native,oracle=True);opt0=optim_state(native)
        step(native);expected_grad=original_gradients(native);expected_state=model_state(native);expected_optim=optim_state(native);expected_rng=rng()
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(cfg,0,1);optimizer_audit(a,cfg)
        assert_equal_tree(initial,original_state(a));assert_equal_tree(opt0,original_optim(a))
        assert rng_hash(post)==rng_hash(rng()) and torch.equal(prediction,forward(a))
        step(a);assert_equal_tree(expected_grad,original_gradients(a));assert_equal_tree(expected_state,original_state(a));assert_equal_tree(expected_optim,original_optim(a));assert rng_hash(expected_rng)==rng_hash(rng())
        baselines[SPECS[cfg].base]=(initial,post,prediction,entry)
        checks['baseline_parity'][SPECS[cfg].base]='PASS: exact Exp9 initial original params/Adam/RNG/prediction + first original gradients/Adam update; later branch alpha intentionally diverges'
        del a,native;torch.cuda.empty_cache()
    branch_hashes={}
    for cfg in CONFIGS:
        initial,post,pred,entry=baselines[SPECS[cfg].base]
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(cfg,0,1);optimizer_audit(a,cfg)
        assert_equal_tree(initial,original_state(a));assert rng_hash(post)==rng_hash(rng()) and torch.equal(pred,forward(a))
        key=SPECS[cfg].graph;hsh=weight_hash(a,True);branch_hashes.setdefault(key,hsh);assert branch_hashes[key]==hsh
        del a;torch.cuda.empty_cache()
    checks['checks']['all18_zero_alpha']='PASS: 18 original-trunk initial parameters/RNG/prediction exact; graph/local same-family learned init hashes equal'
    c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;seed_rng(0)
    a=agent_for('M14',0,1);step(a);assert gradients(a)['alpha']>0
    start=dict(model=model_state(a),optim=optim_state(a),rng=rng());targets=[]
    for _ in range(3):
        step(a);g=gradients(a);assert all(v>0 and math.isfinite(v) for v in g.values()),g
        targets.append((model_state(a),optim_state(a),rng(),forward(a)))
    cp=STATE/'preflight_recovery.pt';save_checkpoint(cp,start);saved=checkpoint(cp)
    load_state(a,saved['model']);load_optim(a,saved['optim']);restore_rng(saved['rng'])
    for sm,so,sr,pr in targets:
        step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng());assert torch.equal(pr,forward(a))
    cp.unlink();checks['checks']['three_step_atomic_recovery']='PASS: full model/buffers/original2Adam+E9/branch groups + Python/NumPy/CPU/CUDA RNG exact'
    real=forward(a);cf=counterfactual(a,'M14',0,1,real);assert len(cf)==13 and all(r['applicable'] for r in cf)
    diagnostic=diagnostics(a,'M14',0,1,real,gradients(a))
    assert diagnostic['prediction_nonfinite']==False and diagnostic['residual_relative_max']<=.100001
    checks['checks']['full_fold_diagnostic_path']='PASS: complete graph diagnostic incl deterministic CPU histograms,branch/graph/strength interventions and exact REAL restoration'
    del a;torch.cuda.empty_cache()
    # Reconstruct once with shuffled heldout labels; compare construction AND first full update.
    c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;seed_rng(1);entry=rng()
    a=agent_for('M14',1,1);initial=model_state(a);after_init=rng();step(a);sm,so,sr=model_state(a),optim_state(a),rng()
    y=c['y'];altered=y.copy();te=c['folds'][0][1];altered[te]=altered[np.random.default_rng(22022).permutation(te)]
    c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry);c['y']=altered
    try:z=agent_for('M14',1,1)
    finally:c['y']=y
    assert_equal_tree(initial,model_state(z));assert rng_hash(after_init)==rng_hash(rng())
    step(z);assert_equal_tree(sm,model_state(z));assert_equal_tree(so,optim_state(z));assert rng_hash(sr)==rng_hash(rng())
    checks['checks']['heldout_label_invariance']='PASS: modified heldout labels passed into EvalHelper -> identical signed buffers,graph,full first train update/optimizer/RNG'
    checks['checks']['counterfactual_restoration']='PASS: all13 actions restore every registered buffer,weights,rawFC,RNG and bitwise REAL predictions; no cached adjacency exists in CodeA'
    checks.update(status='PASS',completed_utc=utc(),formal_fold_training=False,precision='float32; exact parity/recovery; dense tolerance1e-5; no pilot scores')
    m=integrity(full=True);m['preflight']=checks;save_json(MANIFEST,m)
    print('PREFLIGHT_PASS '+json.dumps(checks,ensure_ascii=False),flush=True)
    return checks
def validate_complete():
    m=integrity(full=True);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    expected={(c,s,f) for c in CONFIGS for s in range(3) for f in range(1,11)}
    keys=lambda rr:{(r['config'],int(r['seed']),int(r['fold'])) for r in rr}
    assert len(rows)==len(ds)==EXPECTED_FOLDS and keys(rows)==keys(ds)==expected
    assert all(math.isfinite(float(r[k])) for r in rows for k in REPORT_METRICS+['brier_asd','logloss','ece_asd'])
    assert all(sum(int(r['physical_gpu'])==g for r in rows)==270 for g in (0,1))
    assert len(cf)==EXPECTED_CF_ROWS and len({(r['config'],int(r['seed']),int(r['fold']),r['condition']) for r in cf})==EXPECTED_CF_ROWS
    assert sum(r['applicable']=='False' for r in cf)==EXPECTED_CF_NA
    assert all(math.isfinite(float(r[k])) for r in cf if r['applicable']=='True' for k in REPORT_METRICS+['prediction_flip','probability_abs_change','logprob_abs_change'])
    assert all(r['config'] in CF_GROUPS and r['condition'] in CONDITIONS for r in cf)
    for r in cf:
        expected_na=not SPECS[r['config']].graph and r['condition'] in ('GRAPH_OFF','GRAPH_STRENGTH_OFF','GRAPH_SEMANTIC_OFF')
        assert (r['applicable']=='False')==expected_na
    assert m['completed_tasks']==54 and m['historical_reference_audit']['status']=='PASS' and m['preflight']['status']=='PASS'
    refs,_=verify_historical();assert refs==read_csv(RESULTS/'historical_references.csv')
    assert all(float(r['residual_relative_max'])<=.100001 and float(r['noncorr_delta_max'])==0 and int(r['adjacency_outside_line_nonzero'])==0 for r in ds)
    assert all(r['all_core_parameter_gradients_observed']=='True' and r['prior_requires_grad']=='False' for r in ds)
    assert all(float(r['kl_train'])>=-1e-7 and float(r['kl_test'])>=-1e-7 for r in ds)
    assert all(abs(float(r['ba'])-float(r['hard_auc']))<1e-12 for r in rows)
    assert set(p.name for p in RESULTS.iterdir())<={'fold_metrics.csv','diagnostics.csv','counterfactual.csv','historical_references.csv','report.md'}
    return m


def write_readme(m):
    lines=['# SABER22：概念条件化有符号功能连接证据','',
        '固定18组、seed0/1/2各连续10折，共540新fold。额外导入实验21六组180历史fold，只用于paired-split比较。禁止自动commit/push。','',
        '三个指定Python模块逐字复制自根prompt；run.py由实验21审计模板一次复制并适配，不导入历史实验Python模块。训练只使用本目录semantic_cache.npz，不依赖临时实验20缓存。','',
        '数学：训练折全体FC估计center/scale；仅outer-train ASD/TD标签估计contrast，r=tanh(contrast/.4)，s=tanh(d/2)r。它是训练监督的弱疾病关联，不是临床症状或病理真值。',
        '四概念正负系数各4标量，.35sigmoid、初值.1；attention=masked softmax(P/.8+(1+.5P)(a+max(s,0)-a−max(−s,0)))；仅train的.002KL。UNSIGNED只去attention修正符号，数值encoder/readout仍带符号；SHARED保留8参数并取均值。',
        '图节点252条经HO端点核验的RFE保留FC连接；共享ROI决定固定line support，gate×概念profile余弦×绝对strength控制患者权重，单位self-loop+归一化聚合。不是新的物理脑FC拓扑。',
        '仅CORR潜在slot：[871,4,36] Encoder/FeedForward后、原SPGS前追加≤10%归一化残差；alpha=0，fork_rng 220000+seed。local/graph各19437/20525新增参数；E9类别头另36865且完全沿用实验9数学。',
        'E9互补组仅将delta方向正交于当前CORR潜在向量；不保证信息论互补。原四个9维局部token不是脑区。','',
        '|组|基座|prior|图|变体|正交|问题|','|---|---|---|---|---|---|---|']
    for c,s in SPECS.items():lines.append(f'|{c}|{s.base}|{s.condition}|{s.graph}|{s.variant}|{s.complement}|{s.explanation}|')
    lines+=['','原协议原样保留：871×321输入，特征48+6+10+256；ASD=0/403，TD=1/468；StratifiedKFold(10,shuffle=True,random_state=10)，450epoch/early200，lr.0038，原Adam WD.11，新分支WD0，dropout.35。',
        'val=test、test ACC严格提升选最佳checkpoint、全局CV前监督RFE与标准化、全871 transductive传播存在乐观偏差；仅作历史协议比较，不是独立无泄漏泛化。原GFR i==1覆盖和ALoss不反传均保留，未加入新诊断BCE或GRL。',
        '每卡固定27seed任务/270fold，起始上限4worker，每worker2CPU线程；根据cgroup/GPU显存下调。每25epoch和fold边界原子保存完整模型/Adam/RNG，可精确恢复。一个seed的10fold不重新播种；不同方法早停会分叉后续随机流。',
        '补丁重建：读取根CORE→在临时oracle fuzz0应用实验9patch→CodeC生成新源码→保存相对根CORE的唯一integration_patch.diff。worker只复制根CORE并fuzz0应用本补丁一次，SHA严格核对。原目录只读。',
        '六核心组同best checkpoint的13反事实=2340行，local三图操作270 NA；2070有效，无重训。图实时重算，恢复全部buffer/输入/权重/RNG及bitwise REAL预测。NUMERIC_ONLY保留REAL图架构，不代替独立数值组。','',
        f"准备根HEAD `{m['git_head_at_prepare']}`；审计GitHub `{AUDIT_COMMIT}`；缓存SHA `{sha(CACHE)}`。源代码、RN50、数据、Prompt、映射、cache原始数组和历史参考签名见manifest。",'',
        '临床身份链未建立（0/871可核验），未复制患者ID/原始临床记录。正确文本/数值/中性/随机/错解剖对照共同决定科学门槛，不由最高ACC单独决定。',
        '[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)提出GFR+PLN/SPGS/RES；[MMGPL](https://arxiv.org/html/2312.14574v2)已有疾病概念加权与语义Token图。概念条件化GNN/语言Prompt本身不能作为首次创新。','',
        '```bash',f'cd {ROOT}']
    for cmd in ('prepare','preflight','all','resume','refresh','report','validate'):lines.append(f'python experiment/{EXP.name}/run.py {cmd}')
    lines+=['```','','最终仅保留本README、四Python模块、run.py、静态cache、design、manifest、唯一integration_patch和五个结果文件/简短scheduler日志；临时模型/源码/worker日志仅在/tmp，完整验证后清理。']
    atomic(EXP/'README.md','\n'.join(lines)+'\n')


def science_gates(rows,cf):
    import numpy as np
    if len(rows)!=540:return {k:dict(status='INCONCLUSIVE',reason='正式540fold未完整') for k in ('medical_text','correct_anatomy','signed_fc','domain_specific','graph_magnitude','E9_complement','E9_synergy')}
    def supports(left,controls):return all(stable_support(rows,left,r) for r in controls)
    def sensitive(group,action):
        rr=[r for r in cf if r['config']==group and r['condition']==action and str(r['applicable']) in ('True','true')]
        return len(rr)==30 and all(np.mean([float(r['probability_abs_change']) for r in rr if int(r['seed'])==s])>1e-8 for s in range(3))
    def status(ok,pairs,extra=True):
        if ok and extra:return 'PASS'
        deltas=[paired(rows,l,r,k)['mean'] for l,r in pairs for k in ('acc','ba','probability_auc')]
        return 'FAIL' if any(x<=0 for x in deltas) or not extra else 'INCONCLUSIVE'
    definitions={
      'medical_text':('M00',['M01','M02','M04','P0'],sensitive('M00','NEUTRAL') and sensitive('M00','ISOMETRIC')),
      'correct_anatomy':('M00',['M03'],sensitive('M00','WRONG_ANATOMY')),
      'signed_fc':('M00',['M05'],sensitive('M00','SIGN_OFF')),
      'domain_specific':('M00',['M06'],sensitive('M00','SHARED_DOMAIN')),
      'E9_complement':('M13',['M10','E9'],True),
    }
    out={}
    for name,(left,controls,extra) in definitions.items():
        ps=[(left,r) for r in controls]
        out[name]=dict(status=status(supports(left,controls),ps,extra),pairs=ps,same_checkpoint_sensitivity=extra,criterion='每个对照ACC/BA/ProbAUC均须三个seed差值全正；仅内部描述标准')
    local=supports('M10',['M11','M12','E9']);graph=supports('M14',['M15','M16','E9'])
    out['E9_synergy']=dict(status='PASS' if local or graph else 'FAIL',local_all_controls=local,graph_all_controls=graph,criterion='plain正确语义组同时超过E9、数值及中性，三个seed三指标均正')
    mg=supports('M07',['M08','M09','M00'])
    magnitude=sensitive('M07','GRAPH_STRENGTH_OFF') and sensitive('M14','GRAPH_STRENGTH_OFF')
    positive_cf=True
    for cfg in ('M07','M14'):
        for key in ('acc_change','ba_change','probability_auc_change'):
            rr=[r for r in cf if r['config']==cfg and r['condition']=='GRAPH_STRENGTH_OFF']
            positive_cf &= all(np.mean([float(r[key]) for r in rr if int(r['seed'])==s])<0 for s in range(3))
    out['graph_magnitude']=dict(status='PASS' if mg and magnitude and positive_cf else ('FAIL' if not mg else 'INCONCLUSIVE'),trained_graph_controls=mg,same_checkpoint_magnitude_sensitive=magnitude,consistent_selected_checkpoint_benefit=bool(positive_cf),scope='固定checkpoint消融不能代替无magnitude重训；不是Shapley或外部医学证据')
    out['E9_complement']['graph_pairs']=[('M17',r) for r in ['M14','E9']]
    out['E9_complement']['graph_all_positive']=supports('M17',['M14','E9'])
    if out['E9_complement']['graph_all_positive']:out['E9_complement']['status']='PASS'
    return out


def report():
    import numpy as np
    m=read_json(MANIFEST);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv');refs=read_csv(RESULTS/'historical_references.csv')
    lines=['# 实验二十二：概念条件化有符号功能连接证据','',f"状态 **{m['run_status']}**；任务 {m['completed_tasks']}/54，新训练fold {len(rows)}/540，诊断 {len(ds)}/540，反事实 {len(cf)}/2340，历史参考 {len(refs)}/180。生成 {utc()}。",'',
        f"根HEAD `{m['git_head_at_prepare']}`；公开审计 `{AUDIT_COMMIT}`；六核心SHA相同。原输入SHA `{m['source_data_sha256'][str(ROOT/'data/ABIDE/processed_standard_data.csv')]}`；缓存SHA `{sha(CACHE)}`。全部代码、模型、文本、历史、数据签名见manifest。",'',
        '本轮严格遵守冻结18组，无改词、无超参搜索、无测试pilot。三个模型文件与用户prompt代码块逐字一致。原始两Adam、P0/P3/E9、GFR覆盖行为和ALoss不参与反传均保留。新分支插在Encoder+FeedForward后SPGS前，仅CORR潜在slot、残差≤10%。',
        '训练折标签决定的contrast是弱监督疾病关联，正负方向不是独立临床真值。身份链仍0/871可核验，不接入临床量表/缺失模式。48+6+10+256这四组输入不等于四种独立真实影像模态；四个9D latent token不对应ROI。',
        '历史协议val=test，使用测试ACC选最佳epoch，CV前全局监督RFE/标准化，871人transductive传播；结果有乐观偏差，只作历史协议内部比较，不是独立无泄漏泛化。三seed共享10fold且训练集高度重叠，不把30fold当独立样本、不报告伪显著性。后续需要独立站点与训练内选模。','',
        '## 原论文、实现和历史教训','',
        '[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)以受试者GFR和PLN语义门控/RES融合；源码中的四个latent局部Token不能按论文示意直接命名脑区。[MMGPL](https://arxiv.org/html/2312.14574v2)的概念加权和概念关系Token图是先行工作，本轮不主张概念条件化GNN或Prompt本身的新颖性。',
        'Exp19自由Key可吸收语义方向；Exp20固定先验预测依赖弱；Exp21修正同域共享且用abs(d)，单域profile归一化会消除幅度，E9仍强，临床监督链不可建。本轮只检验训练折有符号关联、域独立正负系数、图绝对strength和E9正交残差的增量。',
        'FC为LedoitWolf相关/Fisher-z后RFE选择的稀疏连接，不是直接全脑功能连接图。新图以252条核验FC连接为节点，共享HO端点为line边，改变患者消息权重而非真实物理脑拓扑。',
        '与Exp21旧adaptive相比同时改变signed机制、attention温度、KL与graph强度，因此新旧差只能描述组合，单机制归因依靠本轮M05/M06及固定checkpoint干预。','',
        '## 方法与冻结配置','',
        'center/std只由outer-train全体FC计算；contrast仅由outer-train ASD/TD计算。r=tanh(contrast/.4)，s=tanh(d/2)r；每域a+/a−=.35sigmoid初始.1；attention=softmax(P/.8+(1+.5P)(a+max(s,0)−a−max(−s,0)))；.002KL仅训练样本。',
        'UNSIGNED只去attention修正的符号，公共数值encoder与证据readout仍带符号，不能叫完全sign-blind。SHARED保留8参数后平均。graph权重=数值gate乘exp(.4(cos−.5))及exp(.45平均logstrength)，NUMERIC训练组无这两个图语义因子。',
        'P0/E9原主干从头训练各30fold，绝不载入历史best权重初始化。E9 36→1024、T=.2、class CE=.1、gamma=.2tanh(alpha)保留。新分支local19437/graph20525参数，WD0。正交残差只对CORR潜在方向操作，不能据此声称信息互补。','',
        '|组|基座|先验|图|变体|正交|新增参数（含E9头）|机制问题|','|---|---|---|---|---|---|---:|---|']
    for cfg,s in SPECS.items():lines.append(f'|{cfg}|{s.base}|{s.condition}|{s.graph}|{s.variant}|{s.complement}|{m["additional_parameters"][cfg]}|{s.explanation}|')
    lines+=['','## 历史六组（不可变180fold）','',f"实验21历史parity180={m['historical_reference_audit']['status']}，源SHA `{m['historical_reference_audit']['source_sha256']}`。数据/划分SHA与本轮相同；仅paired-split，不保证fold2以后随机流相同，旧checkpoint/患者OOF不可用，未伪造。",'',
        '|引用|旧组|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---|---:|---:|---:|---:|---:|---:|']
    if len(refs)==180:
        hs=summary(refs)
        for alias,old in HISTORICAL_REFERENCES.items():lines.append('|'+alias+'|'+old+'|'+'|'.join(f"{100*hs[old][k]['mean']:.5f}±{100*hs[old][k]['sd']:.5f}" for k in METRICS)+'|')
    lines+=['','## 工程审计与预检','',f"预检：{json.dumps(m.get('preflight'),ensure_ascii=False)}",'',
        f"资源：起始每卡最多4workers，各2CPU线程。固定GPU任务分配 {[(c,s,g) for c,s,g in TASKS]}；峰值 {m.get('resource_peaks',{})}；动态并发变化 {m.get('resource_throttling',[])}；训练墙钟 {m.get('training_wall_seconds','未完成')}秒。",'',
        '每seed连续10fold不重播种；worker任务锁/调度锁，模型、原Adam_MP/MF及E9/新分支group、所有buffer、Python/NumPy/CPU/CUDA RNG每25epoch及fold边界原子保存。失败最多两次原状态续跑；同(config,seed,fold)去重。每个最佳state另保存对应Adam；最终无600份权重。','']
    complete=len(rows)==540
    if complete:
        ss=summary(rows)
        lines+=['## 18组全部主结果','', '单位%，先seed内10fold均值，再3seed mean±population SD。Hard-AUC与BA数值相同、分别保留。ASD为正类，Prob-AUC使用exp(logprob)[:,0]；F1来自ASD TP/FP/FN且不用于checkpoint选择。','',
          '|组|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for cfg in CONFIGS:lines.append('|'+cfg+'|'+'|'.join(f"{100*ss[cfg][k]['mean']:.5f}±{100*ss[cfg][k]['sd']:.5f}" for k in REPORT_METRICS)+'|')
        for metric in ('acc','ba','probability_auc'):
            best=max(CONFIGS,key=lambda c:ss[c][metric]['mean']);lines.append(f"\n本轮最高{metric}：{best}，{100*ss[best][metric]['mean']:.5f}%；仅本轮18组最高，不自动代表超过E9。")
        lines+=['','## 三seed独立十折结果、epoch与校准','',
          '|组|seed|ACC|BA|Hard-AUC|Prob-AUC|SEN|SPE|F1|最佳epoch均值（0-based）|Brier-ASD|LogLoss|ECE-ASD(10bin)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cfg in CONFIGS:
            for seed in range(3):lines.append(f'|{cfg}|{seed}|'+'|'.join(f"{100*ss[cfg][k]['seeds'][seed]:.5f}" for k in REPORT_METRICS)+'|'+f"{ss[cfg]['best_epoch']['seeds'][seed]:.2f}|"+'|'.join(f"{ss[cfg][k]['seeds'][seed]:.7f}" for k in ['brier_asd','logloss','ece_asd'])+'|')
        lines+=['','历史组没有冻结患者OOF概率，不伪造旧组Brier/ECE或患者级配对。新组逐fold校准均值可比较，同样受test选模影响。','',
          '## 全部24预注册比较×六指标','', '单位pp；Δ顺序seed0/seed1/seed2。W/T/L只是同划分30fold描述，具有强依赖，不视为30独立样本。','',
          '|比较|指标|平均差pp|三个seed差pp|fold胜/平/负|','|---|---|---:|---|---|']
        for left,right in PAIRS:
            for key in METRICS:
                v=paired(rows,left,right,key);lines.append(f"|{left}−{right}|{key}|{100*v['mean']:+.5f}|"+'/'.join(f'{100*x:+.5f}' for x in v['seeds'])+f"|{v['wins']}/{v['ties']}/{v['losses']}|")
        lines+=['','## 新组之间epoch与概率校准差','', '|比较|Δ最佳epoch均值|ΔBrier|ΔLogLoss|ΔECE|','|---|---:|---:|---:|---:|']
        for left,right in PAIRS:
            if right in HISTORICAL_REFERENCES:continue
            lines.append(f'|{left}−{right}|'+ '|'.join(f"{paired(rows,left,right,k)['mean']:+.7f}" for k in ['best_epoch','brier_asd','logloss','ece_asd'])+'|')
    else:lines+=['## 正式结果尚未完整','',f'目前只有{len(rows)}fold，不输出不完整配置的最终排名或科学成败。']
    if ds:
        lines+=['','## 分支、signed关联、attention与图机制','',
            'gradient_norms为epoch0/1及每25epoch观测到的最大范数，不是best epoch的梯度。残差和反事实均来自ACC最佳state。alpha正负及contrast方向仅模型关联。','',
            '|组|相对残差均值/最大|alpha均值|训练/测试KL|域归一熵均值|正/负系数均值|strength CV|图message相对改变|branch-off Δp|graph-off Δp|strength-off Δp|','|---|---|---:|---|---|---|---:|---:|---:|---:|---:|']
        for cfg in CONFIGS:
            rr=[r for r in ds if r['config']==cfg]
            if not rr:continue
            mean=lambda k:float(np.mean([float(r[k]) for r in rr if r.get(k,'')!=''])) if any(r.get(k,'')!='' for r in rr) else 0.
            arr=lambda k:np.array([json.loads(r[k]) for r in rr]).mean(0).tolist()
            lines.append(f"|{cfg}|{mean('residual_relative_mean'):.7f}/{max(float(r['residual_relative_max']) for r in rr):.7f}|{mean('alpha'):+.6f}|{mean('kl_train'):.7g}/{mean('kl_test'):.7g}|"+','.join(f'{v:.5f}' for v in arr('normalized_attention_entropy'))+'|'+','.join(f'{v:.5f}' for v in arr('eta_plus'))+' / '+','.join(f'{v:.5f}' for v in arr('eta_minus'))+f"|{mean('strength_cv'):.6f}|{mean('message_relative_change'):.6f}|{mean('branch_off_probability_change'):.7g}|{mean('graph_off_probability_change'):.7g}|{mean('strength_off_probability_change'):.7g}|")
            lines.append(f"\n{cfg}：train contrast均值{mean('train_contrast_mean'):+.5f}、SD {mean('train_contrast_std'):.5f}，正方向比例{mean('contrast_positive_fraction'):.5f}；正暴露 {arr('positive_signed_exposure')}，负暴露 {arr('negative_signed_exposure')}；域strength spread {arr('strength_domain_spread')}；attention修正均值|C| {mean('attention_correction_abs_mean'):.6f}；ID embedding特征变化 {mean('edge_id_feature_change'):.6f}；原/新CORR cosine {mean('representation_cosine'):.7f}；delta对base绝对cos {mean('residual_base_cosine_abs'):.7g}；E9概率相关 {mean('disease_probability_correlation'):.6f}。")
            if SPECS[cfg].graph:lines.append(f"\n图{cfg}：cos因子{mean('graph_cosine_factor_mean'):.6f}、strength因子{mean('graph_strength_factor_mean'):.6f}；语义原始边权改变 {mean('graph_raw_weight_semantic_abs_change'):.7g}，幅度部分 {mean('graph_raw_weight_strength_abs_change'):.7g}；患者归一边权方差 {mean('adjacency_patient_variance'):.7g}。这些是机制量与非加性消融，不能当Shapley归因。")
        lines+=['','每fold完整正负系数、gradient norms、非零梯度标记、prior/mask hash、FC参考范围、contrast、attention/strength直方图及Top连接见唯一diagnostics.csv；不输出患者级表。']
    if cf:
        valid=[r for r in cf if r['applicable']=='True'];nas=[r for r in cf if r['applicable']=='False']
        lines+=['','## 同一最佳checkpoint反事实','',f'{len(cf)}总行={len(valid)}有效+{len(nas)} NA；完整应为2340=2070+270。NA只属于M00/M10/M13的三图干预。局部/图/原主干/原E9参数不更新；每种干预恢复buffer、rawFC、RNG和REAL预测bitwise一致。',
            'GRAPH_SEMANTIC_OFF去余弦+strength但保留数值gate及传播；GRAPH_STRENGTH_OFF仅去幅度；NUMERIC_ONLY只P归零，仍是REAL训练图，不是M08/M15数值模型。BRANCH_OFF可能分布外，不能当作重新训练P0/E9。返回log-probability的平均绝对变化列名logprob，不叫raw logits。','',
            '|组|干预|有效/NA|翻转比例|mean_abs_Δp|mean_abs_Δlogprob|ΔACC pp|ΔBA pp|ΔProbAUC pp|','|---|---|---|---:|---:|---:|---:|---:|---:|']
        for cfg in CF_GROUPS:
            for action in CONDITIONS:
                rr=[r for r in cf if r['config']==cfg and r['condition']==action];vv=[r for r in rr if r['applicable']=='True']
                if not vv:lines.append(f'|{cfg}|{action}|0/{len(rr)}|NA|NA|NA|NA|NA|NA|');continue
                mean=lambda k:np.mean([float(r[k]) for r in vv])
                lines.append(f"|{cfg}|{action}|{len(vv)}/{len(rr)-len(vv)}|{mean('prediction_flip'):.7f}|{mean('probability_abs_change'):.8g}|{mean('logprob_abs_change'):.8g}|{100*mean('acc_change'):+.5f}|{100*mean('ba_change'):+.5f}|{100*mean('probability_auc_change'):+.5f}|")
    gates=science_gates(rows,cf);m['scientific_conclusions']=gates;save_json(MANIFEST,m)
    lines+=['','## 冻结科学门槛（不是显著性检验）','']
    for key,v in gates.items():lines.append(f"- **{key}: {v['status']}** — {json.dumps(v,ensure_ascii=False)}")
    if complete:
        real_text=gates['medical_text']['status']=='PASS' and gates['correct_anatomy']['status']=='PASS';synergy=gates['E9_synergy']['status']=='PASS'
        lines+=['',f'正确医学文本与解剖必要性共同通过：{real_text}；E9协同通过：{synergy}。任一门槛失败不包装为医学知识创新。预测对文本敏感只说明依赖，不等于医学正确。',
            f"M14相对E9：ACC/BA/ProbAUC="+'/'.join(f"{100*paired(rows,'M14','E9',k)['mean']:+.5f}pp" for k in ['acc','ba','probability_auc'])+'；M17相对E9：'+'/'.join(f"{100*paired(rows,'M17','E9',k)['mean']:+.5f}pp" for k in ['acc','ba','probability_auc'])+'。',
            'MICCAI2027候选：本轮只提供内部机制证据。即便内部门槛全部通过，也需要外部临床/独立站点验证及文献新颖性审查；未通过时不具备疾病语义第一创新点的证据。',
            '后续唯一优先问题：在训练内选模和独立站点验证下，冻结真实语义先验能否稳定超过同容量数值与中性对照；不自动建立实验23，不改文本/超参。']
    if m.get('blocking_reason'):lines+=['',f"**阻塞：** {m['blocking_reason']}。保留PARTIAL/BLOCKED，未完成fold不造数。"]
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    return gates
def append_handoff():
    m=read_json(MANIFEST)
    if m['run_status']=='COMPLETE':validate_complete()
    assert HANDOFF.is_file(),'Root handoff missing; append forbidden'
    lock=acquire_lock(HANDOFF.with_name('handoff.md.lock'))
    try:
        original=HANDOFF.read_bytes();assert hashlib.sha256(original[:m['handoff_prefix_bytes']]).hexdigest()==m['handoff_prefix_sha256']
        stable=canonical(dict(experiment22=signature(),status=m['run_status'],folds=m['completed_folds']))
        marker=f'<!-- SABER22:{stable} -->'
        if marker.encode() in original:
            m.update(handoff_entry_signature=stable,handoff_sha256=sha(HANDOFF),handoff_update=dict(status='ROOT_APPENDED',path=str(HANDOFF),original_prefix_preserved=True));save_json(MANIFEST,m);return
        text='\n\n## 实验二十二：概念条件化有符号功能连接证据\n\n'+marker+'\n\n'
        text+=f"时间{utc()}，状态**{m['run_status']}**；目录 `{EXP}`。HEAD `{m['git_head_at_prepare']}`，公开审计 `{AUDIT_COMMIT}`；六核心SHA一致，全部源代码/文本/数据/先验/CLIP签名见manifest。\n\n"
        text+='历史动机：19自由Key重参数化、20固定先验依赖弱、21绝对值患者修正缺乏signed/domain差别且图profile丢幅度、临床身份0/871不可核验。22仅outer-train训练标签构建FC有符号弱疾病关联，冻结文本、同容量数值/中性/同Gram/错ROI、unsigned/shared、幅度图、E9正交残差。不是确证临床症状/病理。\n\n'
        text+=f"完成{m['completed_tasks']}/54任务、{m['completed_folds']}/540新fold、{m['diagnostic_rows']}/540诊断、{m['counterfactual_rows']}/2340反事实（完整时270 NA）；额外180历史fold没有重训。两GPU固定各27任务/270fold。耗时{m.get('training_wall_seconds','未完成')}秒；资源{m.get('resource_peaks',{})}。\n\n"
        for cfg,s in SPECS.items():text+=f'- {cfg}: {s.explanation}; prior={s.condition}, graph={s.graph}, variant={s.variant}, complement={s.complement}\n'
        rows=read_csv(RESULTS/'fold_metrics.csv')
        if len(rows)==540:
            ss=summary(rows);text+='\n全部组ACC/BA/HardAUC/ProbAUC/SEN-ASD/SPE-TD/F1，三seed mean±population SD，%：\n\n'
            for cfg in CONFIGS:text+='- '+cfg+': '+' / '.join(f"{100*ss[cfg][k]['mean']:.5f}±{100*ss[cfg][k]['sd']:.5f}" for k in REPORT_METRICS)+'\n'
            for left,right in PAIRS:text+='- '+left+'−'+right+': '+' / '.join(f"{k} {100*paired(rows,left,right,k)['mean']:+.5f}pp" for k in ['acc','ba','probability_auc'])+'\n'
        text+='\n工程预检：'+json.dumps(m.get('preflight'),ensure_ascii=False)+'\n\n'
        text+='科学门槛：'+json.dumps(m.get('scientific_conclusions'),ensure_ascii=False)+'\n\n'
        text+='五条主要认识：1. signed contrast来源仅训练诊断标签，不能提升为临床真值；2. 正确文本必须超过数值/中性/随机/错解剖，最高ACC不够；3. 图幅度需固定checkpoint消融与数值图共同审查，非Shapley；4. E9正交潜在方向不保证独立增量；5. val=test、全局RFE/标准化、transductive和共享fold限制泛化及显著性。详细实测量与反事实见report。\n'
        if m.get('blocking_reason'):text+='\n工程阻塞：'+m['blocking_reason']+'；没有完成的正式fold不作结论。\n'
        text+='\n后续唯一问题：训练内选模与独立站点下真实医学先验能否超过同容量数值/中性；不自动调词、调参或建立实验23。root CORE与实验1–21只读，根handoff完整原前缀逐字保留；未commit/push。\n'
        atomic(HANDOFF,original+text.encode())
        m=read_json(MANIFEST);m.update(handoff_entry_signature=stable,handoff_sha256=sha(HANDOFF),handoff_update=dict(status='ROOT_APPENDED',path=str(HANDOFF),original_prefix_preserved=True,original_prefix_sha256=m['handoff_prefix_sha256'],bytes_appended=len(text.encode())))
        save_json(MANIFEST,m)
    finally:lock.close()


def schedule(workers_per_gpu=4):
    assert 1<=workers_per_gpu<=4
    m=prepare()
    if m['run_status']=='COMPLETE':validate_complete();print('ALREADY_COMPLETE 54 tasks /540 folds',flush=True);return
    lock=acquire_lock(STATE/'scheduler.lock');(STATE/'worker_logs').mkdir(exist_ok=True)
    if not m.get('preflight') or m['preflight']['status']!='PASS':
        with open(STATE/'preflight.log','a') as out:subprocess.run([sys.executable,str(EXP/'run.py'),'preflight'],env=worker_env(0),stdout=out,stderr=subprocess.STDOUT,check=True)
    m=read_json(MANIFEST);m.update(run_status='RUNNING',workers_per_gpu=workers_per_gpu,formal_training_started=True)
    m.setdefault('started_unix',time.time());m.setdefault('started_utc',utc());m['launch_resources']=resources();save_json(MANIFEST,m)
    active={t:dict(pid=p,process=None,gpu=m['tasks'][t]['physical_gpu']) for t,p in live_workers().items()}
    failures=dict(m.get('failures_and_restarts',{}));limits={0:workers_per_gpu,1:workers_per_gpu};last_refresh=last_resources=0.
    print('SCHEDULER_START '+utc(),flush=True)
    try:
        while True:
            for task,info in list(active.items()):
                code=info['process'].poll() if info['process'] else (None if Path('/proc',str(info['pid'])).exists() else 0)
                if code is None:continue
                if 'log' in info:info['log'].close()
                del active[task]
                if code!=0:
                    failures[task]=failures.get(task,0)+1
                    logpath=STATE/'worker_logs'/(task+'.log');failure_text=logpath.read_text()[-12000:]
                    atomic(LOGS/('failed_'+task+'.log'),logpath.read_bytes())
                    if any(k in failure_text for k in ['INTEGRITY','PARITY','REAL restoration','deterministic']) or failures[task]>=3:raise RuntimeError('Task failed; formal run stopped: '+task+'; '+failure_text[-1600:])
                    print('EXACT_RESUME '+task+' from atomic checkpoint',flush=True)
            done={t['task'] for t in task_results() if t['status']=='COMPLETE'}
            now=time.monotonic()
            if now-last_resources>=60:
                m=read_json(MANIFEST);r=resources();peaks=m.setdefault('resource_peaks',dict(worker_rss_gib=0.,gpu0_mib=0,gpu1_mib=0))
                peaks['worker_rss_gib']=max(peaks['worker_rss_gib'],r['worker_rss_gib'])
                for card in r['gpu']:
                    g=card['index'];peaks[f'gpu{g}_mib']=max(peaks[f'gpu{g}_mib'],card['used_mib'])
                    newlimit=1 if card['used_mib']>.90*card['total_mib'] or r['cgroup_available_gib']<12 else (2 if r['cgroup_available_gib']<24 else workers_per_gpu)
                    if limits[g]!=newlimit:m.setdefault('resource_throttling',[]).append(dict(utc=utc(),gpu=g,new_limit=newlimit,resource=r));limits[g]=newlimit
                m['latest_resources']=r;save_json(MANIFEST,m);last_resources=now;print('RESOURCE '+json.dumps(r),flush=True)
            # Fixed27 tasks per GPU; prioritize costly graph tasks within each immutable assignment.
            pending=sorted(TASKS,key=lambda t:(not SPECS[t[0]].graph,t[0],t[1]))
            for gpu in (0,1):
                for config,seed,assigned in pending:
                    task=f'{config}_seed{seed}'
                    if assigned!=gpu or task in done or task in active:continue
                    if sum(v['gpu']==gpu for v in active.values())>=limits[gpu]:break
                    output=open(STATE/'worker_logs'/(task+'.log'),'a');output.write('\nATTEMPT_START '+utc()+'\n');output.flush()
                    proc=subprocess.Popen([sys.executable,str(EXP/'run.py'),'--worker',config,str(seed)],env=worker_env(gpu,task),cwd=EXP,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    active[task]=dict(pid=proc.pid,process=proc,gpu=gpu,log=output);print(f'LAUNCH {task} GPU{gpu} pid={proc.pid}',flush=True)
            if now-last_refresh>=20 or not active:
                m=refresh();m['failures_and_restarts']=failures
                for task in active:
                    if m['tasks'][task]['status']!='COMPLETE':m['tasks'][task]['status']='RUNNING'
                save_json(MANIFEST,m);last_refresh=now
                print(f'PROGRESS tasks={len(done)}/54 folds={m["completed_folds"]}/540 active={len(active)}',flush=True)
            if len(done)==54 and not active:break
            time.sleep(2)
        m=refresh();m.update(run_status='TRAINING_COMPLETE',training_completed_utc=utc(),training_wall_seconds=time.time()-m['started_unix']);save_json(MANIFEST,m)
        validate_complete();assert not live_workers();integrity(full=True)
        m=read_json(MANIFEST);m.update(run_status='COMPLETE',completed_utc=utc(),no_training_processes=True,
            results_sha256={n:sha(RESULTS/(n+'.csv')) for n in ['fold_metrics','diagnostics','counterfactual','historical_references']})
        save_json(MANIFEST,m);report();append_handoff();validate_complete()
        shutil.rmtree(STATE)
        m=read_json(MANIFEST);m.update(temporary_state_cleaned=True,report_sha256=sha(RESULTS/'report.md'));save_json(MANIFEST,m)
        for p in LOGS.glob('.*.lock'):p.unlink(missing_ok=True)
        print('COMPLETE 54/54 tasks;540/540 new folds;540 diagnostics;2340 counterfactual rows including270NA;GPU270/270',flush=True)
    except BaseException as exc:
        for info in active.values():
            if info['process'] and info['process'].poll() is None:os.killpg(info['pid'],signal.SIGTERM)
        for info in active.values():
            if info['process']:
                try:info['process'].wait(timeout=25)
                except subprocess.TimeoutExpired:os.killpg(info['pid'],signal.SIGKILL);info['process'].wait()
        m=refresh();m.update(run_status='BLOCKED',blocking_reason=repr(exc),failures_and_restarts=failures);save_json(MANIFEST,m)
        report();append_handoff();raise
    finally:lock.close()


def main():
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');worker(sys.argv[i+1],int(sys.argv[i+2]));return
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','all','resume','refresh','report','validate']);p.add_argument('--workers-per-gpu',type=int,default=4)
    args=p.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='preflight':preflight()
        elif args.command in ('all','resume'):schedule(args.workers_per_gpu)
        elif args.command=='refresh':refresh()
        elif args.command=='report':report()
        elif args.command=='validate':validate_complete();print('VALIDATE_COMPLETE_PASS',flush=True)
    except Exception as exc:
        if MANIFEST.exists() and args.command not in ('refresh','report','validate'):
            m=read_json(MANIFEST);m.update(run_status='BLOCKED',blocking_reason=repr(exc));save_json(MANIFEST,m);report();append_handoff()
        raise

if __name__=='__main__':main()

#!/usr/bin/env python3
"""SABER21: offline/clinical audit and frozen 20-arm adaptive evidence experiment.
Independent runtime; only immutable artifacts are read from earlier experiments.
"""
import argparse, ast, copy, contextlib, csv, difflib, fcntl, hashlib, io, json, math, os, pickle
import random, re, shutil, subprocess, sys, tempfile, time, types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
if 'preflight' in sys.argv or 'prepare' in sys.argv:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
EXP=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('SP21_ROOT','/data/adodas/SPromptGL-main')).resolve()
STATE=Path('/tmp')/('spromptgl21-state-'+hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
MANIFEST,RESULTS,LOGS=EXP/'manifest.json',EXP/'results',EXP/'logs'
HANDOFF=ROOT/'handoff.md'
CORE=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']
AUDIT_COMMIT='f11bf7966fcf5756c96f0b074e162827056d8f3d'
P9=ROOT/'experiment/experiment9_prompt_disease_adversarial_interaction'
P19=ROOT/'experiment/experiment19_contrastive_disease_evidence_alignment'
P20=Path(os.environ.get('SP21_EXP20',str(ROOT/'experiment/experiment20_grounded_concept_graph_alignment')))
if not P20.is_dir():P20=Path('/tmp/experiment20_grounded_concept_graph_alignment')
CACHE=P20/'semantic_cache.npz'
CONFIGS=[f'U{i:02}' for i in range(20)]
# The formal 20-row table is authoritative when prose shorthand U2--U5 differs.
_SPECS=[('P0',False,None,False,'NONE','T00'),('P3',True,None,False,'NONE','T02'),
 ('P0',False,'REAL',False,'STATIC','T05'),('P0',False,'REAL',True,'STATIC','T11'),
 ('P0',False,'NUMERIC',True,'STATIC','T10'),('P0',False,'NUMERIC',False,'STATIC','T04'),
 ('P0',False,'REAL',False,'ADAPTIVE',None),('P0',False,'NUMERIC',False,'ADAPTIVE',None),
 ('P0',False,'NEUTRAL',False,'ADAPTIVE',None),('P0',False,'WRONG_ANATOMY',False,'ADAPTIVE',None),
 ('P0',False,'ISOMETRIC',False,'ADAPTIVE',None),('P0',False,'REAL',True,'ADAPTIVE',None),
 ('P0',False,'NUMERIC',True,'ADAPTIVE',None),('P0',False,'NEUTRAL',True,'ADAPTIVE',None),
 ('P3',True,'REAL',False,'ADAPTIVE',None),('P3',True,'NUMERIC',False,'ADAPTIVE',None),
 ('P3',True,'REAL',True,'ADAPTIVE',None),('P3',True,'NUMERIC',True,'ADAPTIVE',None),
 ('P3',True,'NEUTRAL',False,'ADAPTIVE',None),('P3',True,'NEUTRAL',True,'ADAPTIVE',None)]
DESIGNS={c:dict(prompt=p,disease=d,adv=False,condition=k,use_graph=g,family=f,historical_config=h)
 for c,(p,d,k,g,f,h) in zip(CONFIGS,_SPECS)}
# Sparse CUDA index_add must replay exactly after a checkpoint.  Keep the six
# historical workers' original execution settings; only new adaptive workers
# and the mixed preflight need the deterministic cuBLAS workspace.
if 'preflight' in sys.argv or ('--worker' in sys.argv and
        DESIGNS[sys.argv[sys.argv.index('--worker')+1]]['family']=='ADAPTIVE'):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
TASKS=[(c,s,(i*3+s)%2) for i,c in enumerate(CONFIGS) for s in range(3)]
METRICS=['acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td']
CONDITIONS=['REAL','NEUTRAL','ISOMETRIC','ANATOMY_PERMUTED','NUMERIC_ONLY','ADAPT_OFF',
 'GRAPH_NUMERIC','GRAPH_OFF','FC_BRANCH_SHUFFLED','BRANCH_OFF']
CF_GROUPS=['U06','U11','U14','U16']
PAIRS=[('U06',r) for r in ['U02','U07','U08','U09','U10','U00']]+[('U11',r) for r in ['U03','U04','U12','U13','U00']]+[('U14',r) for r in ['U01','U15','U18']]+[('U16',r) for r in ['U01','U14','U17','U19']]
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',nepoch=450,early=200,
 lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,
 GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,
 theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
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
    atomic(path, b"SP21" + hashlib.sha256(data).hexdigest().encode() + data)


def checkpoint(path):
    import torch
    data = Path(path).read_bytes()
    assert data[:4] == b"SP21" and hashlib.sha256(data[68:]).hexdigest().encode() == data[4:68]
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


def summary(rows):
    import numpy as np
    result = {}
    configs = sorted(set(r["config"] for r in rows), key=lambda v: (v[0], int(v[1:])))
    for config in configs:
        result[config] = {}
        for key in METRICS:
            seeds = [float(np.mean([float(r[key]) for r in rows if r["config"] == config and int(r["seed"]) == seed])) for seed in range(3)]
            result[config][key] = dict(mean=float(np.mean(seeds)), sd=float(np.std(seeds)), seeds=seeds)
    return result


def paired(rows, left, right, key):
    import numpy as np
    lookup = {(r["config"], int(r["seed"]), int(r["fold"])):float(r[key]) for r in rows}
    differences = np.array([[lookup[left, seed, fold] - lookup[right, seed, fold] for fold in range(1, 11)] for seed in range(3)])
    return dict(mean=float(differences.mean()), seeds=differences.mean(1).tolist(),
                wins=int((differences > 1e-12).sum()), ties=int((np.abs(differences) <= 1e-12).sum()), losses=int((differences < -1e-12).sum()))


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
                  re.match(r'experiment\d+_',p.name) and 1<=int(p.name.split('_',1)[0][10:])<=20)


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






def context():
    global _CTX
    if _CTX is None:
        import torch,numpy as np
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        directory=make_sources(os.environ.get('SP21_TASK','preflight'))
        sys.path.insert(0,str(directory));sys.modules['e21_runtime']=sys.modules[__name__]
        import network,model
        x,y,folds,audit=data()
        with np.load(CACHE,allow_pickle=False) as packed:cache={k:packed[k].copy() for k in packed.files}
        _CTX=dict(torch=torch,network=network,model=model,x=x,y=y,folds=folds,audit=audit,cache=cache)
    return _CTX


def set_config(config):
    d=DESIGNS[config]
    import torch
    torch.use_deterministic_algorithms(d['family']=='ADAPTIVE')
    os.environ.update(PDAI_PROMPT=d['prompt'],PDAI_DISEASE=str(int(d['disease'])),PDAI_ADV=str(int(d['adv'])),
                      PDAI_PROMPTS_FILE=str(P9/'prompts.json'))


def agent_for(config,seed,fold,oracle=False):
    c=context();set_config(config);c.update(current_config=config,current_fold=fold)
    cls=c['model'].EvalHelper
    if oracle:
        directory=STATE/'oracle';directory.mkdir(parents=True,exist_ok=True)
        src=build_baseline_sources()
        # Use oracle layers with exactly the same accepted V capture.
        net=types.ModuleType('e21_oracle_network');sys.modules[net.__name__]=net
        exec(compile(src['network.py'],'oracle_network.py','exec'),net.__dict__)
        mod=types.ModuleType('e21_oracle_model');sys.modules[mod.__name__]=mod
        exec(compile(src['model.py'].replace('from network import *','from e21_oracle_network import *'),'oracle_model.py','exec'),mod.__dict__)
        cls=mod.EvalHelper
    tr,te=c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):
        a=cls([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    assert (a.val_idx==a.tst_idx).all()
    if oracle:a._e21_raw=a.dataLayer(a.feat)[1]
    return a


def branch(a):return getattr(a.ModalFusion,'evidence21',None)


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


def graph_state(a):return branch(a).graph_adj.detach().cpu().clone() if branch(a) and branch(a).use_graph and hasattr(branch(a),'graph_adj') else None


def load_graph(a,saved):
    if saved is not None and hasattr(branch(a),'graph_adj'):branch(a).graph_adj=saved.to(a.dev)


def weight_hash(a,auxiliary=False,legacy13=False):
    h=hashlib.sha256()
    for name,module in model_modules(a):
        for key,value in sorted(module.named_parameters()):
            if key.startswith('evidence21.')!=auxiliary:continue
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
        else:out,_,_=a.ModalFusion(hidden,a._e21_raw)
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
                probability_auc=float(roc_auc_score(target==0,probability[:,0])),sensitivity_asd=sen,specificity_td=spe)


def step(a):
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')


def gradients(a):
    result={}
    if branch(a):
        b=branch(a)
        for name,pp in [('alpha',[b.alpha]),('encoder',b.encoder.parameters()),('edge_id',b.edge_id.parameters()),
                        ('readout',b.readout.parameters()),('graph_proj',b.graph_proj.parameters() if b.use_graph else [])]:
            grads=[p.grad for p in pp if p.grad is not None]
            assert all(bool(g.isfinite().all()) for g in grads)
            result[name]=math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
    if branch(a) is not None and hasattr(branch(a),'eta_logit'):
        g=branch(a).eta_logit.grad;result['eta_logit']=float(g.abs()) if g is not None else 0.
    return result


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
    m=integrity();task=f'{config}_seed{seed}';gpu=int(os.environ['SP21_PHYSICAL_GPU'])
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
        tracker={k:0. for k in gradients(a)};best,best_epoch,best_model,wait,start,prior_seconds=0.,-1,None,0,0,0.
        if saved and saved['phase']=='EPOCH' and saved['fold']==fold:
            load_state(a,saved['model']);load_optim(a,saved['optim']);load_graph(a,saved['graph']);restore_rng(saved['rng'])
            best,best_epoch,best_model,wait,start=[saved[k] for k in ('best_acc','best_epoch','best_model','wait','next_epoch')]
            tracker,entry,initial,legacy,aux,prior_seconds=[saved[k] for k in ('tracker','entry_rng','initial','legacy','aux','fold_seconds')]
        saved=None;started=time.monotonic();last=start-1
        for epoch in (range(start,450) if wait<=200 else []):
            step(a)
            if epoch in (0,1) or (epoch+1)%25==0:
                for key,value in gradients(a).items():tracker[key]=max(tracker[key],value)
            out=forward(a);val=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            if val>best:best,best_epoch,best_model,wait=val,epoch,model_state(a),0
            else:wait+=1
            last=epoch;stop=wait>200 or epoch==449
            if (epoch+1)%25==0 or stop:
                save_checkpoint(cp,dict(signature=sig,phase='EPOCH',fold=fold,next_epoch=epoch+1,model=model_state(a),optim=optim_state(a),graph=graph_state(a),rng=rng(),
                    best_acc=best,best_epoch=best_epoch,best_model=best_model,wait=wait,tracker=tracker,entry_rng=entry,initial=initial,legacy=legacy,aux=aux,
                    fold_seconds=prior_seconds+time.monotonic()-started,result=result))
                print(f'EPOCH fold={fold} epoch={epoch} best_acc={best:.12f} wait={wait}',flush=True)
            if stop:break
        assert best_model is not None
        continuation=rng();load_state(a,best_model);out=forward(a);values=metrics(out,a.tst_idx)
        assert abs(values['acc']-best)<1e-12 and abs(values['ba']-values['hard_auc'])<1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=gpu,task_signature=sig,**values,best_epoch=best_epoch,epochs_executed=last+1,
                 selected_val_acc=best,train_n=len(a.trn_idx),test_n=len(a.tst_idx),val_equals_test=True,initial_trunk_sha256=initial,
                 initial_trunk_legacy13_sha256=legacy,initial_branch_sha256=aux if branch(a) else '',entry_rng_sha256=rng_hash(entry),additional_parameters=m['additional_parameters'][config])
        assert_historical_fold(row)
        diagnostic=diagnostics(a,config,seed,fold,out,tracker);diagnostic['task_signature']=sig
        cf=counterfactual(a,config,seed,fold,out) if config in CF_GROUPS else []
        for r in cf:r['task_signature']=sig
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
    if not out and read_json(MANIFEST)['completed_folds']==600:
        # Finalization may be interrupted after deleting temporary checkpoints. Never retrain completed CV.
        metrics_=read_csv(RESULTS/'fold_metrics.csv');diag_=read_csv(RESULTS/'diagnostics.csv');cf_=read_csv(RESULTS/'counterfactual.csv')
        for c,s,g in TASKS:
            select=lambda rr:[r for r in rr if r['config']==c and int(r['seed'])==s]
            rr=select(metrics_);assert len(rr)==10
            for r in rr:
                for k in ('seed','fold','physical_gpu','best_epoch','epochs_executed'):r[k]=int(r[k])
                for k in METRICS+['seconds']:r[k]=float(r[k])
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
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),SP21_PHYSICAL_GPU=str(gpu),SP21_TASK=task,
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


def schedule(workers_per_gpu=4):
    assert 1<=workers_per_gpu<=4
    audit();m=prepare()
    if m['run_status']=='COMPLETE':validate_complete();print('ALREADY_COMPLETE 60 tasks / 600 folds',flush=True);return
    lock=acquire_lock(STATE/'scheduler.lock')
    if not m.get('preflight') or m['preflight']['status']!='PASS':
        with open(LOGS/'preflight.log','a') as out:subprocess.run([sys.executable,str(EXP/'run.py'),'preflight'],env=worker_env(0),stdout=out,stderr=subprocess.STDOUT,check=True)
    m=read_json(MANIFEST);m.update(run_status='RUNNING',workers_per_gpu=workers_per_gpu,formal_training_started=True,staging_only=False)
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
                    text=(LOGS/(task+'.log')).read_text()[-10000:]
                    if 'HISTORICAL_PARITY_FAILED' in text or failures[task]>=3:raise RuntimeError('Task failed and formal run stopped: '+task+'; '+text[-1200:])
                    print('RECOVER '+task+' from atomic checkpoint',flush=True)
            done={t['task'] for t in task_results() if t['status']=='COMPLETE'}
            baseline_done=all(f'{c}_seed{s}' in done for c in CONFIGS[:6] for s in range(3))
            if baseline_done and not read_json(MANIFEST).get('historical_parity_180'):
                m=read_json(MANIFEST);m['historical_parity_180']=dict(status='PASS',folds=180,groups=CONFIGS[:6],metrics='All 6 metrics + best/executed epochs + initial trunk SHA + entry RNG SHA; 60 P0/E9 + 120 static folds',utc=utc());save_json(MANIFEST,m)
                print('HISTORICAL_PARITY_PASS 180/180 (120 static + 60 P0/E9); release U06-U19',flush=True)
            # Fixed GPU task assignment; long graph tasks first once historical paths are verified.
            pending=sorted(TASKS,key=lambda t:(not DESIGNS[t[0]]['use_graph'],t[0],t[1])) if baseline_done else TASKS
            for gpu in (0,1):
                for config,seed,assigned in pending:
                    task=f'{config}_seed{seed}'
                    if assigned!=gpu or task in done or task in active or (not baseline_done and config not in CONFIGS[:6]):continue
                    if sum(v['gpu']==gpu for v in active.values())>=limits[gpu]:break
                    output=open(LOGS/(task+'.log'),'a');output.write('\nATTEMPT_START '+utc()+'\n');output.flush()
                    proc=subprocess.Popen([sys.executable,str(EXP/'run.py'),'--worker',config,str(seed)],env=worker_env(gpu,task),cwd=EXP,stdout=output,stderr=subprocess.STDOUT)
                    active[task]=dict(pid=proc.pid,process=proc,gpu=gpu,log=output);print(f'LAUNCH {task} GPU{gpu} pid={proc.pid}',flush=True)
            now=time.monotonic()
            if now-last_refresh>=20 or not active:
                m=refresh();m['failures_and_restarts']=failures
                if now-last_resources>=60:
                    r=resources();peaks=m.setdefault('resource_peaks',dict(worker_rss_gib=0.,gpu0_mib=0,gpu1_mib=0))
                    peaks['worker_rss_gib']=max(peaks['worker_rss_gib'],r['worker_rss_gib'])
                    for card in r['gpu']:
                        g=card['index'];peaks[f'gpu{g}_mib']=max(peaks[f'gpu{g}_mib'],card['used_mib'])
                        newlimit=1 if card['used_mib']>85000 or r['cgroup_available_gib']<12 else (2 if r['cgroup_available_gib']<24 else workers_per_gpu)
                        if limits[g]!=newlimit:m.setdefault('resource_throttling',[]).append(dict(utc=utc(),gpu=g,new_limit=newlimit));limits[g]=newlimit
                    last_resources=now;print('RESOURCE '+json.dumps(r),flush=True)
                save_json(MANIFEST,m);last_refresh=now
                print(f'PROGRESS tasks={len(done)}/60 folds={m["completed_folds"]}/600 active={len(active)} baseline_gate={baseline_done}',flush=True)
            if len(done)==60 and not active:break
            time.sleep(2)
        m=refresh();m.update(run_status='TRAINING_COMPLETE',training_completed_utc=utc(),training_wall_seconds=time.time()-m['started_unix']);save_json(MANIFEST,m)
        validate_complete();report();assert not live_workers();integrity(full=True)
        shutil.rmtree(STATE)
        m=read_json(MANIFEST);m.update(run_status='COMPLETE',completed_utc=utc(),temporary_state_cleaned=True,no_training_processes=True,
            root_handoff_sha256=sha(ROOT/'handoff.md'),results_sha256={n:sha(RESULTS/(n+'.csv')) for n in ['fold_metrics','diagnostics','counterfactual']})
        save_json(MANIFEST,m);report();append_handoff()
        m=read_json(MANIFEST);m['report_sha256']=sha(RESULTS/'report.md');save_json(MANIFEST,m)
        for p in LOGS.glob('.*.lock'):p.unlink(missing_ok=True)
        print('COMPLETE 60/60 tasks; 600/600 folds; 600 diagnostics; 1200 counterfactual rows',flush=True)
    except Exception as exc:
        for info in active.values():
            if info['process'] and info['process'].poll() is None:info['process'].terminate()
        for info in active.values():
            if info['process']:
                try:info['process'].wait(timeout=25)
                except subprocess.TimeoutExpired:info['process'].kill();info['process'].wait()
        m=refresh();m.update(run_status='BLOCKED',blocking_reason=str(exc),failures_and_restarts=failures);save_json(MANIFEST,m)
        report();append_handoff();raise
    finally:lock.close()


def stable_support(rows,left,right):
    return all(all(v>0 for v in paired(rows,left,right,k)['seeds']) for k in ('acc','ba','probability_auc'))


def patched_sources():
    src=build_baseline_sources()
    src['main.py']=(ROOT/'main.py').read_text()
    model,network=src['model.py'],src['network.py']
    lines=model.splitlines(keepends=True)
    for node in reversed(ast.parse(model).body):
        if isinstance(node,ast.ClassDef) and node.name=='EvalHelper':
            for fn in reversed(node.body):
                if isinstance(fn,ast.FunctionDef) and fn.name=='write_pdai_outputs':
                    del lines[fn.lineno-1:fn.end_lineno]
    model=''.join(lines)
    anchor='    def forward(self, x):\n        bs = x.size(0)'
    assert network.count(anchor)==1
    network=network.replace(anchor,'    def forward(self, x, raw_modalities=None):\n        bs = x.size(0)')
    anchor='        text_features = self.prompt_text_features\n'
    assert network.count(anchor)==1
    network=network.replace(anchor,"        if hasattr(self, 'evidence21'):\n            x = self.evidence21(x, raw_modalities)\n"+anchor)
    anchor=next(line for line in model.splitlines() if line.startswith('    def run_epoch('))
    model=model.replace(anchor,"        from e21_runtime import install_evidence21\n        install_evidence21(self)\n\n"+anchor)
    # Historical simple-2 and _print_acc calls only. Other unused modes unchanged.
    tree=ast.parse(model);ll=model.splitlines(keepends=True)
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef) or cls.name!='EvalHelper':continue
        for fn in cls.body:
            if not isinstance(fn,ast.FunctionDef):continue
            if fn.name=='run_epoch':
                lo=next(i for i in range(fn.lineno-1,fn.end_lineno) if "elif mode == 'simple-2':" in ll[i])
                hi=next(i for i in range(lo+1,fn.end_lineno) if "elif mode == 'normal':" in ll[i])
                for i in range(lo,hi):
                    ll[i]=ll[i].replace('self.ModalFusion(feat)','self.ModalFusion(feat, self._e21_raw)')
                    if ll[i].strip()=='loss = cls_loss + all_graph_loss':
                        ll[i]+="            branch = getattr(self.ModalFusion, 'evidence21', None)\n            if branch is not None and hasattr(branch, 'auxiliary_loss'):\n                loss = loss + branch.auxiliary_loss()\n"
            elif fn.name=='_print_acc':
                for i in range(fn.lineno-1,fn.end_lineno):
                    ll[i]=ll[i].replace('self.ModalFusion(feat)','self.ModalFusion(feat, self._e21_raw)')
    model=''.join(ll)
    assert model.count('loss = loss + branch.auxiliary_loss()')==1
    assert model.count('self.ModalFusion(feat, self._e21_raw)')==3
    src['model.py'],src['network.py']=model,network
    return src


def make_sources(task='preflight'):
    directory=STATE/'tasks'/task/'sources';directory.mkdir(parents=True,exist_ok=True)
    for name in CORE:atomic(directory/name,(ROOT/name).read_bytes())
    for level,patch in [(0,P9/'alignment_patch.diff'),(1,EXP/'integration_patch.diff')]:
        r=subprocess.run(['patch','--batch','--fuzz=0',f'-p{level}','-i',str(patch)],cwd=directory,capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
    for name in CORE:assert sha(directory/name)==read_json(MANIFEST)['patched_source_sha256'][name]
    link=directory/'CLIP'
    if not link.exists():link.symlink_to(ROOT/'CLIP',target_is_directory=True)
    work=directory.parent/'work';work.mkdir(exist_ok=True)
    for name in ('attn','graph'):(work/name).mkdir(exist_ok=True)
    os.chdir(work)
    return directory


def install_evidence21(a):
    from adaptive_evidence import BoundedConceptEvidence21
    from static_evidence import ConceptEvidence20
    c=context();a._e21_raw=a.dataLayer(a.feat)[1];d=DESIGNS[c['current_config']]
    assert tuple(a._e21_raw[3].shape)==(871,256)
    if d['condition'] is None:return
    torch=c['torch'];static=d['family']=='STATIC'
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if a.dev.type=='cuda' else []):
        torch.manual_seed((200000 if static else 210000)+int(a.hyperpm.seed))
        cls=ConceptEvidence20 if static else BoundedConceptEvidence21
        kw={} if static else {'semantic_graph':d['condition']!='NUMERIC'}
        a.ModalFusion.evidence21=cls(condition=d['condition'],use_graph=d['use_graph'],cache=c['cache'],raw_corr=a._e21_raw[3],train_idx=a.trn_idx,**kw).to(a.dev)
    if static:a.ModalFusion.evidence21.prepare_graph(a._e21_raw[3])
    a.optimizer_MF.add_param_group(dict(params=list(a.ModalFusion.evidence21.parameters()),lr=a.hyperpm.lr,weight_decay=0.))


def historical_reference(config,seed,fold):
    hist=DESIGNS[config]['historical_config']
    if hist is None:return None
    ref=next(r for r in read_csv(P20/'results/fold_metrics.csv') if r['config']==hist and int(r['seed'])==seed and int(r['fold'])==fold)
    result={k:float(ref[k]) for k in METRICS}
    result.update({k:int(ref[k]) for k in ('best_epoch','epochs_executed')})
    result.update({k:ref[k] for k in ('initial_trunk_sha256','entry_rng_sha256')})
    return result


def assert_historical_fold(row):
    ref=historical_reference(row['config'],int(row['seed']),int(row['fold']))
    if ref is None:return
    for k,v in ref.items():
        if isinstance(v,str):assert row[k]==v,('HISTORICAL_PARITY_FAILED',row['config'],row['seed'],row['fold'],k,row[k],v)
        else:assert abs(float(row[k])-v)<1e-12,('HISTORICAL_PARITY_FAILED',row['config'],row['seed'],row['fold'],k,row[k],v)


def prepare():
    import numpy as np
    if MANIFEST.exists():return integrity(full=True)
    if not (RESULTS/'clinical_audit.json').exists() or not (RESULTS/'semantic_rank_audit.csv').exists():audit()
    LOGS.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True);STATE.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    audit_repo=Path('/tmp/spromptgl19-publish')
    if subprocess.run(['git','cat-file','-e',AUDIT_COMMIT],cwd=ROOT,capture_output=True).returncode==0:audit_repo=ROOT
    comparison={}
    for name in CORE:
        old=subprocess.check_output(['git','show',AUDIT_COMMIT+':'+name],cwd=audit_repo)
        new=(ROOT/name).read_bytes()
        comparison[name]=dict(local=hashlib.sha256(new).hexdigest(),audit=hashlib.sha256(old).hexdigest(),equal=new==old)
        assert old==new,('SOURCE_SHA_MISMATCH',name)
    m20=read_json(P20/'manifest.json');assert m20['run_status']=='COMPLETE'
    assert sha(CACHE)==m20['frozen_file_sha256']['semantic_cache.npz']
    assert sha(EXP/'static_evidence.py')==m20['code_sha256']['concept_evidence.py']
    with np.load(CACHE,allow_pickle=False) as a:cache={k:a[k].copy() for k in a.files}
    validate_cache(cache)
    x,y,folds,da=data()
    p11=ROOT/'experiment/experiment11_roi_mapping_recovery/results'
    roi=read_csv(p11/'roi_111_labels.csv');edges=read_csv(p11/'roi_256_named_mapping.csv')
    assert len(roi)==111 and len(edges)==256
    assert [int(r['correlation_feature_index']) for r in edges if r['usable_for_roi_semantic_prompt']=='True']==cache['valid_idx'].tolist()
    assert all(r['verification_status']=='VERIFIED_NUMERIC_UNIQUE' for r in edges)
    iu=np.triu_indices(111,1)
    for j,idx in enumerate(cache['valid_idx']):
        row=edges[idx];i,k=map(int,(row['roi_i_zero_based'],row['roi_j_zero_based']))
        original=int(row['original_connection_index'])
        assert (i,k)==(int(iu[0][original]),int(iu[1][original]))==tuple(cache['roi_endpoints'][j])
        assert roi[i]['name_verified']==roi[k]['name_verified']=='True'
        assert roi[i]['column_order_verified']==roi[k]['column_order_verified']=='True'
    hist={str(p.resolve()):sha(p) for f in historical_folders() for p in f.rglob('*') if p.is_file()}
    hist.update({str(p):sha(p) for p in P20.rglob('*') if p.is_file()})
    names=CORE+['data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy','data/ABIDE/phenotyic_processing.ipynb','data/ABIDE/data_processing.ipynb','CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py']
    sources={str(ROOT/n):sha(ROOT/n) for n in names}
    sources[str(ROOT/'prompt')]=sha(ROOT/'prompt')
    for p in ROOT.glob('*.pdf'):sources[str(p)]=sha(p)
    sources[str(CACHE)]=sha(CACHE)
    for dep in [P9/'prompts.json',P9/'alignment_patch.diff',P19/'prompts.json',P19/'semantic_cache.npz',P19/'cdea.py',P20/'concept_evidence.py',P20/'cache_reference.py',P20/'run.py',P20/'design.json',P20/'manifest.json',P20/'results/fold_metrics.csv',P20/'results/diagnostics.csv',P20/'results/counterfactual.csv']:
        sources[str(dep)]=sha(dep)
    phenotype=ROOT/'dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
    if phenotype.exists():sources[str(phenotype)]=sha(phenotype)
    base=build_baseline_sources();src=patched_sources()
    patch=''.join(''.join(difflib.unified_diff(base[n].splitlines(True),src[n].splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n in CORE)
    atomic(EXP/'integration_patch.diff',patch)
    design=dict(configurations=DESIGNS,tasks=[dict(config=c,seed=s,physical_gpu=g) for c,s,g in TASKS],protocol=PROTOCOL,
        pairs=PAIRS,counterfactual_conditions=CONDITIONS,counterfactual_groups=CF_GROUPS,cache_path=str(CACHE),
        prior_math='immutable Exp20: 4x252; real/neutral/same-Gram/wrong-anatomy',adaptive_math=dict(temperature=.5,eta_initial=.1,eta_max=.35,kl_weight=.02,graph_profile_scale=.7,residual_limit=.1),
        prompt_paths=[str(P9/'prompts.json'),str(P19/'prompts.json')],table_authoritative=True,
        historical_parity=dict(groups={c:DESIGNS[c]['historical_config'] for c in CONFIGS[:6]},formal_gate=180,static_gate_folds=120),
        new_weight_decay=0.,initialization_seed=dict(static='200000+seed',adaptive='210000+seed'),frozen_before_training=True)
    save_json(EXP/'design.json',design)
    params={c:(36865 if d['disease'] else 0)+((20517 if d['use_graph'] else 19429)+(d['family']=='ADAPTIVE') if d['condition'] else 0) for c,d in DESIGNS.items()}
    codes=['run.py','adaptive_evidence.py','static_evidence.py','offline_diagnostics.py','clinical_audit.py']
    m=dict(experiment='SABER21 Evidence Constrained Adaptive Alignment',run_status='PREPARED',prepared_utc=utc(),root=str(ROOT),experiment_directory=str(EXP),temporary_state=str(STATE),
        git_head_at_prepare=head,github_audit_commit=AUDIT_COMMIT,source_comparison=comparison,protocol=PROTOCOL,configurations=DESIGNS,
        data_audit=da,semantic_audit=m20['semantic_audit'],additional_parameters=params,source_data_sha256=sources,protected_history_sha256=hist,
        patched_source_sha256={n:hashlib.sha256(src[n].encode()).hexdigest() for n in CORE},
        frozen_file_sha256={n:sha(EXP/n) for n in ['design.json','integration_patch.diff']},code_sha256={n:sha(EXP/n) for n in codes},
        clip_weight_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),split_sha256=canonical([dict(train=tr.tolist(),test=te.tolist()) for tr,te in folds]),
        handoff_prefix_bytes=HANDOFF.stat().st_size,handoff_prefix_sha256=sha(HANDOFF),preflight=None,
        completed_tasks=0,completed_folds=0,diagnostic_rows=0,counterfactual_rows=0,
        tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS},
        audit_results=read_json(RESULTS/'clinical_audit.json'),A_checkpoints='CHECKPOINT_UNAVAILABLE_NO_EXACT_DECOMPOSITION',
        scientific_conclusions=dict(engineering='PENDING',medical_semantics='PENDING',adaptation='PENDING',graph_semantics='PENDING',e9_synergy='PENDING'))
    save_json(MANIFEST,m);write_readme(m);report()
    print('PREPARED',json.dumps(dict(configurations=20,tasks=60,folds=600,params=params,cache_sha256=sha(CACHE))),flush=True)
    return m


def integrity(full=False):
    m=read_json(MANIFEST)
    for path,h in m['source_data_sha256'].items():assert sha(path)==h,('source/data changed',path)
    for field in ('code_sha256','frozen_file_sha256'):
        for name,h in m[field].items():assert sha(EXP/name)==h,('frozen file changed',name)
    if full:
        for path,h in m['protected_history_sha256'].items():assert sha(path)==h,('history changed',path)
        assert sha(Path.home()/'.cache/clip/RN50.pt')==m['clip_weight_sha256']
        original=HANDOFF.read_bytes()[:m['handoff_prefix_bytes']]
        assert hashlib.sha256(original).hexdigest()==m['handoff_prefix_sha256'],'handoff original bytes changed'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['git_head_at_prepare']
    return m
def audit():
    """A+B only. This function never constructs a model or calls run_epoch."""
    import numpy as np,pandas as pd
    RESULTS.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    keys=[ROOT/'prompt',ROOT/'data/ABIDE/processed_standard_data.csv',CACHE,P20/'results/counterfactual.csv',P20/'results/diagnostics.csv',EXP/'offline_diagnostics.py',EXP/'clinical_audit.py']
    inputs={str(p):sha(p) for p in keys}
    out=RESULTS/'clinical_audit.json'
    if out.exists():
        old=read_json(out)
        if old.get('audit_input_sha256')==inputs and all((RESULTS/n).is_file() for n in ['semantic_rank_audit.csv','exp20_counterfactual_diagnostic.csv','clinical_coverage.csv']):return old
    from offline_diagnostics import summary_counterfactual,rank_audit
    from clinical_audit import full_audit
    cf=summary_counterfactual(P20);cf.to_csv(RESULTS/'exp20_counterfactual_diagnostic.csv',index=False)
    # These are descriptive ratios: not additive contributions, no patient vectors are available.
    print('A_COUNTERFACTUAL',cf.groupby('config')[['BRANCH_OFF','GRAPH_NUMERIC','NEGATIVE_TEXT','FC_BRANCH_SHUFFLED']].mean().to_json(),flush=True)
    ranks=rank_audit(ROOT/'data/ABIDE/processed_standard_data.csv',CACHE);ranks.to_csv(RESULTS/'semantic_rank_audit.csv',index=False)
    s=full_audit(ROOT,RESULTS)
    m20=read_json(P20/'manifest.json')
    old_state=Path(m20['temporary_state']);remaining=list(old_state.glob('*.pt')) if old_state.exists() else []
    s.update(audit_input_sha256=inputs,audit_completed_utc=utc(),training_executed_in_audit=False,
        experiment20_path=str(P20),checkpoint_status='CHECKPOINT_UNAVAILABLE_NO_EXACT_DECOMPOSITION' if not remaining else 'PARTIAL_CHECKPOINTS_NOT_FULL_SIGNED_30_FOLDS',
        checkpoint_files_found=len(remaining),rank_rows=len(ranks),counterfactual_proxy_rows=len(cf),
        numerical_only_exp20_intervention='prior=0 AND condition=NUMERIC; graph rebuilt numeric. Not orthogonal local-only intervention.',
        official_sources=[dict(url='https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf',role='ABIDE I field definitions, scores nonnegative, versions and reliability flags'),
            dict(url='https://arxiv.org/html/2312.14574v2',role='MMGPL Fig1, III-C Eq7-10 and III-D Eq11-14; concepts and semantic graphs already established'),
            dict(url='https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf',role='MGDR multigraph common/private SVD and contrastive representation'),
            dict(url='https://papers.miccai.org/miccai-2025/0861-Paper5129.html',role='SPromptGL local PDF + official publication'),
            dict(url='https://arxiv.org/abs/2504.16096',role='BrainPrompt ROI/subject/disease prompting; local PDF')])
    save_json(out,s)
    print('AUDIT_COMPLETE',json.dumps(dict(A=s['checkpoint_status'],B=s['status'],clinical=s['clinical_alignment'],rank_rows=len(ranks)),ensure_ascii=False),flush=True)
    return s


@contextlib.contextmanager
def intervention(a,condition,fold):
    import numpy as np,torch
    b=branch(a);assert b is not None
    assert hasattr(b,'intervention'),'Only adaptive core groups receive Exp21 counterfactuals'
    original=(b.prior.clone(),b.mask.clone(),b.intervention,a._e21_raw)
    try:
        cache=context()['cache'];b.intervention=condition
        if condition=='NEUTRAL':b.prior.copy_(torch.as_tensor(cache['prior_neutral'],device=a.dev))
        elif condition=='ISOMETRIC':b.prior.copy_(torch.as_tensor(cache['prior_random'],device=a.dev))
        elif condition=='ANATOMY_PERMUTED':b.mask.copy_(torch.as_tensor(cache['masks_wrong'],device=a.dev))
        elif condition=='NUMERIC_ONLY':b.prior.zero_() # Preserve semantic_graph: only the prior changes, as registered.
        elif condition=='FC_BRANCH_SHUFFLED':
            rr=np.random.default_rng(212000+fold);idx=a.tst_idx;perm=rr.permutation(idx)
            while np.any(idx==perm):perm=rr.permutation(idx)
            raw=a._e21_raw[3].clone();raw[idx]=raw[perm];a._e21_raw=[*a._e21_raw[:3],raw]
        elif condition in ('REAL','ADAPT_OFF','GRAPH_NUMERIC','GRAPH_OFF','BRANCH_OFF'):pass
        else:raise ValueError(condition)
        yield
    finally:
        b.prior.copy_(original[0]);b.mask.copy_(original[1]);b.intervention=original[2];a._e21_raw=original[3]


def counterfactual(a,config,seed,fold,real):
    import torch
    b=branch(a);te=a.tst_idx;rows=[];target=metrics(real,te)
    original=model_state(a);raw=[t.clone() for t in a._e21_raw];original_rng=rng()
    for condition in CONDITIONS:
        row=dict(config=config,seed=seed,fold=fold,condition=condition,applicable=True,retrained=False)
        if condition in ('GRAPH_NUMERIC','GRAPH_OFF') and not b.use_graph:
            rows.append(dict(row,applicable=False,reason='NA: no graph installed'));continue
        with intervention(a,condition,fold):
            out=forward(a);m=metrics(out,te)
            row.update(**m,prediction_flip=float((out[te].argmax(1)!=real[te].argmax(1)).float().mean()),
                probability_abs_change=float((out[te].exp()-real[te].exp()).abs().mean()),
                logit_abs_change=float((out[te]-real[te]).abs().mean()),
                acc_change=m['acc']-target['acc'],ba_change=m['ba']-target['ba'],probability_auc_change=m['probability_auc']-target['probability_auc'])
        assert torch.equal(forward(a),real),('REAL counterfactual restore',condition)
        assert_equal_tree(original,model_state(a));assert all(torch.equal(x,y) for x,y in zip(raw,a._e21_raw))
        rows.append(row)
    assert rng_hash(original_rng)==rng_hash(rng())
    assert len(rows)==10
    return rows


def diagnostics(a,config,seed,fold,real,tracker):
    import numpy as np,torch
    from torch.nn import functional as F
    b=branch(a);d=DESIGNS[config];te=a.tst_idx
    row=dict(config=config,seed=seed,fold=fold,condition=d['condition'] or 'NONE',family=d['family'],use_graph=d['use_graph'],prediction_nonfinite=False,
        gradient_norms=json.dumps(tracker,sort_keys=True),disease_gamma=a.ModalFusion.last_disease_gamma,
        disease_probability_correlation=correlation(a.ModalFusion.last_disease_logits[te].softmax(-1)[:,0].detach().cpu().numpy(),
            a.ModalFusion.last_base_logprob[te].exp()[:,0].detach().cpu().numpy()) if d['disease'] else '',train_reference_label_free=True)
    if b is None:return row
    capture={}
    handle=b.register_forward_hook(lambda module,inp,out:capture.update(base=inp[0].detach().clone(),new=out.detach().clone()))
    try:assert torch.equal(forward(a),real)
    finally:handle.remove()
    base,new=capture['base'],capture['new'];ll={k:v.clone() for k,v in b.last.items()}
    static=d['family']=='STATIC';mask=b.masks if static else b.mask
    att=ll['attention'] if static else ll['att'];delta=ll['delta'];norm=base[:,3].norm(dim=-1)
    rel=delta[:,3].norm(dim=-1)/(norm+1e-8);assert rel.max()<=.100001 and delta[:,:3].count_nonzero()==0
    mean,sd=(b.reference_mean,b.reference_std) if static else (b.ref_mean,b.ref_sd)
    entropy=-(att*att.clamp_min(1e-30).log()).sum(-1)
    z,dev,gate=b.measure(a._e21_raw[3]) if static else b._patient_data(a._e21_raw[3])
    with torch.no_grad():
        q=(dev/3).tanh();enc=b.encoder(torch.stack((q,q.abs(),z.tanh()),dim=-1))
        h=b.edge_norm(enc+b.edge_id.weight[None]);without_id=b.edge_norm(enc)
    row.update(alpha=float(b.alpha.detach()),eta=float(ll['eta']) if not static else '',kl_train=float(b.auxiliary_loss().detach()/.02) if not static else '',
        kl_test=float(ll['kl'][te].mean()) if not static else '',graph_effect=float(ll['graph_effect']) if not static else '',
        residual_relative_mean=float(rel[te].mean()),residual_relative_max=float(rel.max()),residual_nonzero_fraction=float((rel[te]>1e-8).float().mean()),
        noncorr_delta_max=float(delta[:,:3].abs().max()),original_base_norm_mean=float(norm[te].mean()),delta_norm_mean=float(delta[te,3].norm(dim=-1).mean()),
        representation_cosine=float(F.cosine_similarity(base[te,3],new[te,3],dim=-1).mean()),
        train_fc_mean_min=float(mean.min()),train_fc_mean_max=float(mean.max()),train_fc_std_min=float(sd.min()),train_fc_std_max=float(sd.max()),
        mask_counts=json.dumps(mask.sum(1).cpu().tolist()),prior_requires_grad=b.prior.requires_grad,
        prior_candidate_std=json.dumps([float(b.prior[k,mask[k]].std(unbiased=False)) for k in range(4)]),
        domain_attention_entropy=json.dumps(entropy[te].mean(0).cpu().tolist()),
        normalized_attention_entropy=json.dumps((entropy[te]/mask.sum(1).float().log()).mean(0).cpu().tolist()),
        top_feature_indices=json.dumps([[int(b.valid_idx[i]) for i in b.prior[k].masked_fill(~mask[k],-1e4).topk(5).indices.cpu().tolist()] for k in range(4)]),
        edge_id_feature_change=float((h[te]-without_id[te]).norm(dim=-1).mean()),
        actual_gradient_parameter_count=sum(int((p.grad!=0).sum()) for p in b.parameters() if p.grad is not None))
    if not static:
        row['patient_attention_abs_change']=float((att[te]-ll['p0'][None]).abs().mean())
        row['eta_bound_satisfied']=0<=float(ll['eta'])<=.35
    if b.use_graph:
        with torch.no_grad():
            if static:
                adj=b.graph_adj[te];row['adjacency_patient_variance']=float(adj.var(0,unbiased=False)[b.line_graph].mean())
                row['adjacency_outside_line_nonzero']=int(adj[:,~b.line_graph].count_nonzero())
                ha=b.graph_norm(h+F.gelu(b.graph_proj(torch.bmm(b.graph_adj,h))))
                row['message_relative_change']=float(((ha[te]-h[te]).norm(dim=-1)/(h[te].norm(dim=-1)+1e-8)).mean())
                row['graph_semantic_weight_abs_change']=''
            else:
                weights=(gate[:,b.src]*gate[:,b.dest]).clamp_min(1e-8)
                profile=F.normalize(att.transpose(1,2),dim=-1,eps=1e-8)
                factor=torch.exp(.7*((profile[:,b.src]*profile[:,b.dest]).sum(-1)-.5)) if b.semantic_graph else torch.ones_like(weights)
                actual=weights*factor
                den=torch.ones_like(gate).index_add(1,b.dest,actual)
                row['adjacency_patient_variance']=float((actual[te]/den[te][:,b.dest]).var(0,unbiased=False).mean())
                row['graph_semantic_weight_abs_change']=float((actual[te]-weights[te]).abs().mean())
                row['adjacency_outside_line_nonzero']=0
    else:row.update(adjacency_patient_variance=0.,adjacency_outside_line_nonzero=0,graph_semantic_weight_abs_change=0.)
    # Same-checkpoint removal diagnostics are also generated for static groups without text interventions.
    handle=b.register_forward_hook(lambda module,inputs,output:inputs[0])
    try:off=forward(a)
    finally:handle.remove()
    offm=metrics(off,te);row.update(branch_off_acc=offm['acc'],branch_off_ba=offm['ba'],branch_off_probability_auc=offm['probability_auc'],branch_off_probability_change=float((off[te].exp()-real[te].exp()).abs().mean()))
    if b.use_graph:
        b.use_graph=False
        try:go=forward(a);gm=metrics(go,te)
        finally:b.use_graph=True
        row.update(graph_off_acc=gm['acc'],graph_off_ba=gm['ba'],graph_off_probability_auc=gm['probability_auc'],graph_off_probability_change=float((go[te].exp()-real[te].exp()).abs().mean()))
    assert torch.equal(forward(a),real)
    return row
def preflight():
    import numpy as np,torch
    from torch.nn import functional as F
    from adaptive_evidence import BoundedConceptEvidence21
    from static_evidence import ConceptEvidence20
    m=integrity(full=True)
    if m.get('preflight') and m['preflight']['status']=='PASS':return m['preflight']
    c=context();checks=dict(status='RUNNING',started_utc=utc(),checks={},baseline_parity={})
    seed_rng(21021)
    raw=torch.randn(21,256,device='cuda');base=torch.randn(21,4,36,device='cuda');tr=np.arange(16)
    parts=[torch.zeros(21,n,device='cuda') for n in (48,6,10)]+[raw]
    for cond in ('REAL','NEUTRAL','NUMERIC','ISOMETRIC','WRONG_ANATOMY'):
        for graph in (False,True):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(210000);b=BoundedConceptEvidence21(condition=cond,use_graph=graph,semantic_graph=cond!='NUMERIC',cache=c['cache'],raw_corr=raw,train_idx=tr).cuda()
            assert sum(p.numel() for p in b.parameters())==(20518 if graph else 19430)
            assert torch.equal(b(base,parts),base)
            assert not b.prior.requires_grad and b.last['att'][:,~b.mask].count_nonzero()==0
            assert torch.allclose(b.last['att'].sum(-1),torch.ones(21,4,device='cuda'),atol=1e-6)
            assert float(b.auxiliary_loss())>=-1e-7 and 0<=float(b.last['eta'])<=.35
            assert not any(isinstance(layer,torch.nn.Linear) and layer.out_features==1024 for layer in b.modules())
            opt=torch.optim.Adam(b.parameters(),lr=.0038,weight_decay=0)
            for it in range(2):
                opt.zero_grad();out=b(base,parts);loss=(out*torch.arange(36,device='cuda')).mean()+b.auxiliary_loss();loss.backward()
                assert b.alpha.grad is not None and torch.isfinite(b.alpha.grad) and abs(float(b.alpha.grad))>0
                assert b.eta_logit.grad is not None and torch.isfinite(b.eta_logit.grad) and abs(float(b.eta_logit.grad))>0
                if it==1:
                    for name,layer in [('encoder',b.encoder),('edge_id',b.edge_id),('readout',b.readout)]+([('graph_proj',b.graph_proj)] if graph else []):
                        gg=[p.grad for p in layer.parameters()];assert all(g is not None and torch.isfinite(g).all() for g in gg) and sum(float(g.abs().sum()) for g in gg)>0,(name,cond)
                opt.step()
            with torch.no_grad():b.alpha.fill_(100);out=b(base,parts)
            delta=out-base;assert delta[:,:3].count_nonzero()==0 and float((delta[:,3].norm(dim=-1)/(base[:,3].norm(dim=-1)+1e-8)).max())<=.100001
    checks['checks']['ten_adaptive_cases']='PASS: shapes/parameters/alpha+eta gradients/two steps/KL/mask/zero-alpha/residual bound'
    # Explicit row-stochastic dense graph versus destination index_add, including gradients.
    n=7;h=torch.randn(n,252,32,device='cuda',requires_grad=True)
    w=torch.rand(n,len(b.src),device='cuda',requires_grad=True)
    den=h.new_ones(n,252).index_add(1,b.dest,w)
    sparse=(h+torch.zeros_like(h).index_add(1,b.dest,h[:,b.src]*w[...,None]))/den[...,None]
    dense=torch.diag_embed(torch.ones(n,252,device='cuda'))
    dense[:,b.dest,b.src]=w;dense=dense/dense.sum(-1,keepdim=True)
    expected=torch.bmm(dense,h);assert torch.allclose(sparse,expected,atol=1e-5,rtol=1e-5)
    gs=torch.autograd.grad(sparse.square().mean(),(h,w),retain_graph=True);gd=torch.autograd.grad(expected.square().mean(),(h,w))
    assert all(torch.allclose(a,z,atol=1e-5,rtol=1e-5) for a,z in zip(gs,gd))
    assert torch.allclose(dense.sum(-1),torch.ones(n,252,device='cuda'),atol=1e-6)
    assert dense[:,~b.support].count_nonzero()==0
    assert bool(c['cache']['line_graph'][b.dest.cpu(),b.src.cpu()].all())
    checks['checks']['dense_sparse_value_gradient_equivalence']=dict(status='PASS',max_output_error=float((sparse-expected).abs().max()),max_grad_error=max(float((x-y).abs().max()) for x,y in zip(gs,gd)),atol=1e-5)
    del b,dense,expected,sparse,h,w;torch.cuda.empty_cache()
    # Historical P0/E9 source parity and three Adam updates.
    baselines={}
    for cfg in ('U00','U01'):
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None
        seed_rng(0);entry=rng();native=agent_for(cfg,0,1,oracle=True)
        initial=model_state(native);post=rng();prediction=forward(native,oracle=True);targets=[]
        for _ in range(3):step(native);targets.append((model_state(native),optim_state(native),rng(),forward(native,oracle=True)))
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(cfg,0,1);optimizer_audit(a,cfg)
        assert_equal_tree(initial,model_state(a));assert rng_hash(post)==rng_hash(rng()) and torch.equal(prediction,forward(a))
        for sm,so,sr,pr in targets:
            step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng());assert torch.equal(pr,forward(a))
        baselines[cfg]=(initial,post,prediction,entry)
        checks['baseline_parity'][cfg]='PASS: original P9 source, init/RNG/predictions/three Adam steps exact'
        del a,native,targets;torch.cuda.empty_cache()
    # Independent Exp20 source reconstruction; never import its run.py or launch an old worker.
    oracle_dir=STATE/'oracle20';oracle_dir.mkdir(exist_ok=True)
    for name in CORE:atomic(oracle_dir/name,(ROOT/name).read_bytes())
    rr=subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(P20/'integration_patch.diff')],cwd=oracle_dir,capture_output=True,text=True)
    assert rr.returncode==0,rr.stdout+rr.stderr
    def oracle_install(a):
        a._e20_raw=a.dataLayer(a.feat)[1];a._e21_raw=a._e20_raw;d=DESIGNS[c['current_config']]
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(200000+int(a.hyperpm.seed))
            a.ModalFusion.concept_evidence20=ConceptEvidence20(condition=d['condition'],use_graph=d['use_graph'],cache=c['cache'],raw_corr=a._e20_raw[3],train_idx=a.trn_idx).to(a.dev)
        a.ModalFusion.concept_evidence20.prepare_graph(a._e20_raw[3])
        a.optimizer_MF.add_param_group(dict(params=list(a.ModalFusion.concept_evidence20.parameters()),lr=a.hyperpm.lr,weight_decay=0.))
    sys.modules['sp20_runtime']=SimpleNamespace(install_branch=oracle_install)
    def normalized_state(a):
        st=model_state(a);st['pln']={k.replace('concept_evidence20.','evidence21.'):v for k,v in st['pln'].items()};return st
    for cfg in CONFIGS[2:6]:
        net=types.ModuleType('e21_static_oracle_network');sys.modules[net.__name__]=net
        exec(compile((oracle_dir/'network.py').read_text(),'static_oracle_network.py','exec'),net.__dict__)
        mod=types.ModuleType('e21_static_oracle_model')
        exec(compile((oracle_dir/'model.py').read_text().replace('from network import *','from e21_static_oracle_network import *'),'static_oracle_model.py','exec'),mod.__dict__)
        set_config(cfg);c['current_config']=cfg;seed_rng(0);entry=rng();tr,te=c['folds'][0]
        with contextlib.redirect_stdout(io.StringIO()):native=mod.EvalHelper([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=0),tr,te)
        native.MF_sav.close();native.GCMP_sav.close()
        initial=normalized_state(native);post=rng();prediction=forward(native);targets=[]
        for _ in range(3):step(native);targets.append((normalized_state(native),optim_state(native),rng(),forward(native)))
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(cfg,0,1);optimizer_audit(a,cfg)
        assert_equal_tree(initial,model_state(a));assert rng_hash(post)==rng_hash(rng());assert torch.equal(prediction,forward(a))
        for sm,so,sr,pr in targets:
            step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng());assert torch.equal(pr,forward(a))
        checks['baseline_parity'][cfg]='PASS: Exp20 source, module/optimizer/RNG/three Adam steps exact; full 30-fold parity pending'
        del a,native,targets;torch.cuda.empty_cache()
    # Zero-alpha equivalence of all new arms; RNG isolated after original E9 install.
    new_hashes={}
    for cfg in CONFIGS[6:]:
        family='U01' if DESIGNS[cfg]['disease'] else 'U00';state,post,pred,entry=baselines[family]
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(cfg,0,1);optimizer_audit(a,cfg);ss=model_state(a);ss['pln']={k:v for k,v in ss['pln'].items() if not k.startswith('evidence21.')}
        assert_equal_tree(state,ss);assert rng_hash(post)==rng_hash(rng()) and torch.equal(pred,forward(a))
        key=DESIGNS[cfg]['use_graph'];val=weight_hash(a,True)
        if key not in new_hashes:new_hashes[key]=val
        assert new_hashes[key]==val
        del a;torch.cuda.empty_cache()
    checks['checks']['twenty_integrated_paths']='PASS: existing parameters/init/RNG/prediction exact; adaptive same-family parameter init identical'
    c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;seed_rng(0)
    a=agent_for('U16',0,1);step(a);assert gradients(a)['alpha']>0 and gradients(a)['eta_logit']>0
    start=dict(model=model_state(a),optim=optim_state(a),rng=rng(),graph=graph_state(a));targets=[]
    for _ in range(3):
        step(a);g=gradients(a);assert all(g[k]>0 for k in ('encoder','edge_id','readout','graph_proj','eta_logit'))
        targets.append((model_state(a),optim_state(a),rng(),forward(a)))
    cp=STATE/'preflight_recovery.pt';save_checkpoint(cp,start);saved=checkpoint(cp)
    load_state(a,saved['model']);load_optim(a,saved['optim']);restore_rng(saved['rng'])
    for sm,so,sr,pr in targets:
        step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng());assert torch.equal(pr,forward(a))
    cp.unlink();checks['checks']['three_step_atomic_recovery']='PASS: all buffers, train statistics, Adam groups, model and CPU/CUDA/Python/NumPy RNG exact'
    start=dict(model=model_state(a),optim=optim_state(a),rng=rng());labels=a.targ.clone()
    step(a);sm,so,sr=model_state(a),optim_state(a),rng()
    load_state(a,start['model']);load_optim(a,start['optim']);restore_rng(start['rng'])
    a.targ[a.tst_idx]=labels[np.random.default_rng(21021).permutation(a.tst_idx)]
    step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng())
    a.targ.copy_(labels);checks['checks']['heldout_labels']='PASS: full train update/attention/FC statistics invariant under test-label permutation'
    real=forward(a);cf=counterfactual(a,'U16',0,1,real);assert len(cf)==10 and all(r['applicable'] for r in cf)
    checks['checks']['counterfactual_restoration']='PASS: graph/local/eta/FC/OFF changes restore complete state + REAL predictions exactly'
    checks.update(status='PASS',completed_utc=utc(),formal_fold_training=False,full_fold_parity='Gate requires U00-U05 180/180 exact folds, including 120 static folds before adaptive groups launch')
    m=integrity(full=True);m['preflight']=checks;save_json(MANIFEST,m)
    print('PREFLIGHT_PASS '+json.dumps(checks,ensure_ascii=False),flush=True)
    return checks
def validate_complete():
    m=integrity(full=True);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    expected={(c,s,f) for c in CONFIGS for s in range(3) for f in range(1,11)}
    keys=lambda rr:{(r['config'],int(r['seed']),int(r['fold'])) for r in rr}
    assert len(rows)==len(ds)==600 and keys(rows)==keys(ds)==expected
    assert all(math.isfinite(float(r[k])) for r in rows for k in METRICS)
    assert all(sum(int(r['physical_gpu'])==g for r in rows)==300 for g in (0,1))
    assert len(cf)==1200 and len({(r['config'],int(r['seed']),int(r['fold']),r['condition']) for r in cf})==1200
    assert sum(r['applicable']=='False' for r in cf)==120
    assert m['completed_tasks']==60 and m['historical_parity_180']['status']=='PASS'
    for r in rows:assert_historical_fold(r)
    assert all(float(r.get('residual_relative_max') or 0)<=.100001 and int(r.get('adjacency_outside_line_nonzero') or 0)==0 for r in ds)
    assert all(0<=float(r['eta'])<=.35 for r in ds if r.get('eta','')!='')
    assert all(float(r['kl_train'])>=-1e-7 for r in ds if r.get('kl_train','')!='')
    assert all((RESULTS/n).exists() for n in ['exp20_counterfactual_diagnostic.csv','semantic_rank_audit.csv','clinical_coverage.csv','clinical_audit.json'])
    return m


def write_readme(m):
    lines=['# SABER Experiment 21：三线验证与有界自适应疾病证据对齐','',
        '一个实验包含A历史机制离线复核、B患者临床身份可用性审计、C固定20组正式训练。A/B失败不改变C的预注册组别。','',
        f'实验20只读输入位于 `{P20}`；静态语义缓存 `{CACHE}`，SHA256 `{sha(CACHE)}`。未复制数据/CLIP权重或旧runner，静态模块字节与Exp20原文件相同。','',
        '唯一新增自适应参数eta_logit：eta=0.35sigmoid(theta)，初始0.10；att=softmax(prior/0.5 + eta*tanh(abs(d)-1))，mask外0。d只用outer-train全体FC估计无标签均值/标准差。0.02KL仅训练索引。',
        '节点为252条双端HO核验FC连接；固定共享ROI支持集。图权重为患者数值gate乘exp(0.7*(概念attention profile余弦-0.5))，单位self-loop仅一次，按destination行归一化。图不改变物理脑FC拓扑。',
        'CORR残差=0.1tanh(alpha)||base_CORR||normalize(readout)，alpha零初始化。仅插入Encoder+FeedForward后、SPGS前；原P0/P3/E9损失、GFR、分类器、两个Adam保持。新参数WD=0，原WD=.11。',
        'Static local/graph新增19429/20517参数，adaptive各多1个；E9另36865。静态200000+seed，自适应210000+seed置于fork_rng。','',
        '|组|基座|模块|prior|图|历史对应|','|---|---|---|---|---|---|']
    for c,d in DESIGNS.items():lines.append(f"|{c}|{d['prompt']}"+('+D' if d['disease'] else '')+f"|{d['family']}|{d['condition']}|{d['use_graph']}|{d['historical_config']}|")
    lines+=['','按正式配置表处理U02=T05/U03=T11/U04=T10/U05=T04，表外文字的U2--U5 shorthand不用于重排。正式U00-U05共180fold（其中120静态fold）逐值验证历史六指标、epoch、主干初始化与入口RNG；全部通过后才解锁U06-U19。',
        '历史协议：871人，10fold固定random_state10，seed0/1/2连续十折不重播种，450epoch/early200、Adam .0038、dropout.35。val=test且test ACC选Checkpoint、CV前全队列RFE/标准化、全871人transductive传播，会产生乐观偏差，不称无泄漏独立泛化。',
        '每卡30任务300fold、最多4worker，每worker2CPU线程。动态资源压力仅降低并发。完整模型/两Adam/原E9/缓冲区/所有RNG每25epoch及fold边界原子恢复；同签名结果去重；完成后删除临时权重。','',
        '反事实四核心组×30fold×10条件=1200行，local图条件120行NA。固定best checkpoint，只改被指定分支因素，恢复原参数、prior、mask、FC输入及REAL预测逐值一致。所有响应为非加性效应，不能相加为贡献率。','',
        'B包只输出汇总。没有原subject_IDs和871x64逐行复现，不称患者身份匹配，不引入任何量表、缺失指示或临床损失。',
        'MMGPL已有概念加权与概念语义图；MGDR已有多图分解；BrainPrompt已有ROI/subject/disease多层提示。候选创新需正确医学语义的独立必要性与患者有界适应实证支持。','',
        f"根HEAD `{m['git_head_at_prepare']}`；GitHub审计 `{AUDIT_COMMIT}`；核心源码逐字节相同。代码/文本/数据/模型权重/先验hash见manifest。根核心与历史实验只读，根handoff仅追加。禁止自动commit/push。",'',
        '```bash',f'cd {ROOT}']
    for cmd in ('audit','prepare','preflight','all','resume','refresh','report'):lines.append(f'python {EXP}/run.py {cmd}')
    lines+=['```','','重建顺序：复制根核心→fuzz0应用实验9原alignment_patch.diff→fuzz0应用本实验integration_patch.diff。补丁相对实验9基线，严格保留原Prompt/疾病类别原型数学与加载RNG。']
    atomic(EXP/'README.md','\n'.join(lines)+'\n')


def scientific_status(rows,ds,cf):
    import numpy as np
    if len(rows)!=600:return dict(engineering='PENDING',medical_semantics='PENDING',adaptation='PENDING',graph_semantics='PENDING',e9_synergy='PENDING')
    good=lambda l,rs:all(stable_support(rows,l,r) for r in rs)
    valid=lambda g,k:[float(r['probability_abs_change']) for r in cf if r['config']==g and r['condition']==k and str(r['applicable'])=='True']
    semantic=good('U06',['U07','U08','U09','U10']) and good('U14',['U15','U18']) and all(np.mean(valid(c,'NEUTRAL'))>1e-5 for c in ('U06','U14'))
    adaptation=good('U06',['U02']) and good('U11',['U03']) and all(1e-4<float(r['eta'])<.3499 for r in ds if r['config'] in ('U06','U11'))
    graph=good('U11',['U12','U13']) and good('U16',['U17','U19']) and all(np.mean(valid(c,'GRAPH_NUMERIC'))>1e-5 for c in ('U11','U16'))
    synergy=good('U14',['U01']) or good('U16',['U01'])
    return dict(engineering='ENGINEERING_PASS',medical_semantics='MEDICAL_SEMANTICS_'+('PASS' if semantic else 'FAIL'),
        adaptation='ADAPTATION_'+('PASS' if adaptation else 'FAIL'),graph_semantics='GRAPH_SEMANTICS_'+('PASS' if graph else 'FAIL'),e9_synergy='E9_SYNERGY_'+('PASS' if synergy else 'FAIL'))


def report():
    import numpy as np,pandas as pd
    m=read_json(MANIFEST);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    clinical=read_json(RESULTS/'clinical_audit.json');old=pd.read_csv(RESULTS/'exp20_counterfactual_diagnostic.csv');ranks=pd.read_csv(RESULTS/'semantic_rank_audit.csv')
    lines=['# 实验二十一：三线验证与有界自适应疾病证据对齐','',f"状态 **{m['run_status']}**；{m['completed_tasks']}/60任务、{len(rows)}/600fold、{len(ds)}/600诊断、{len(cf)}/1200反事实。生成 {utc()}。",'',
        f"服务器HEAD `{m['git_head_at_prepare']}`；GitHub参照 `{AUDIT_COMMIT}`。六核心源码相同；只读保护{len(m['protected_history_sha256'])}历史文件（含/tmp实验20）。原数据SHA `{m['source_data_sha256'][str(ROOT/'data/ABIDE/processed_standard_data.csv')]}`；CLIP SHA `{m['clip_weight_sha256']}`；先验SHA `{sha(CACHE)}`。全签名见manifest。",'',
        '保留GFR i==1覆盖、ALoss不反传、原双Adam及全部PLN/SPGS/RES/P3/E9数学。新分支接于Encoder+FeedForward的[871,4,36]；潜在token不叫ROI；252条真实连接才是新图节点。E9 36→1024、T=.2、CE=.1、gamma=.2tanh(alpha)与历史相同。','',
        '固定20组不根据结果编辑Prompt、先验或超参。U02/U03/U04/U05按正式表分别复现T05/T11/T10/T04。新增adaptive仅一标量eta_logit，eta∈(0,.35)，初始化.1；局部温度.5，train-only KL权重.02，图profile系数.7，CORR残差≤10%。','',
        '历史协议存在val=test和测试折ACC选checkpoint、CV前全局监督RFE/标准化以及871人transductive传播，分数有乐观偏差。结果只能作历史内部机制比较，不是独立无泄漏泛化。三seed复用相同10fold，不能当30次独立研究。连续折早停会导致后续RNG分叉。','',
        '## A：实验20零重训机制复核','',f"状态：{clinical['checkpoint_status']}。原最终checkpoint已清理，不能恢复历史T11/T18患者级logit、残差或精确Shapley。未以新训练模型冒充旧状态。A/B审计未调用训练。",'',
        'GRAPH_NUMERIC只改图边语义权重、保持local REAL prior；NEGATIVE_TEXT同时改local先验和图先验；NUMERIC_ONLY把prior归0并设置NUMERIC重建图，因此不是正交的纯局部因素；BRANCH_OFF是全新分支移除；FC_BRANCH_SHUFFLED只置乱分支test FC。响应非加性，下面比值不能相加为100%。']
    rng_=np.random.default_rng(21021)
    for cfg,g in old.groupby('config'):
        den=g.BRANCH_OFF.to_numpy();small=int((den<1e-6).sum());mu=den.mean()
        lines+=['',f'**{cfg}**：30fold分母BRANCH_OFF均值={mu:.9g}，分母<1e-6为{small}/30。']
        for condition in ['GRAPH_NUMERIC','NEGATIVE_TEXT','NUMERIC_ONLY','FC_BRANCH_SHUFFLED']:
            v=g[condition].to_numpy();rat=v/np.maximum(den,1e-12)
            # Bootstrap resamples three seed clusters, retaining all ten correlated folds per draw.
            per=g.groupby('seed')[[condition,'BRANCH_OFF']].mean().to_numpy();draw=rng_.integers(0,3,size=(4000,3));mboot=per[draw].mean(1)
            ci=np.quantile(mboot[:,0]/np.maximum(mboot[:,1],1e-12),[.025,.975])
            lines.append(f'- {condition}: mean|Δp|={v.mean():.9g}；ratio-of-means={v.mean()/max(mu,1e-12):.6g}，seed-cluster描述性bootstrap区间[{ci[0]:.6g},{ci[1]:.6g}]；逐折ratio中位数/范围={np.median(rat):.6g}/[{rat.min():.6g},{rat.max():.6g}]。只有3 seed，不解读为独立样本显著性。')
    d20=pd.read_csv(P20/'results/diagnostics.csv')
    for cfg in ['T11','T18']:
        g=d20[d20.config==cfg]
        lines.append(f"- {cfg}保存的代理诊断：残差={g.residual_relative_mean.mean():.7g}；graph message relative change={g.message_relative_change.mean():.7g}；attention entropy={np.mean([json.loads(v) for v in g.domain_attention_entropy],axis=0).tolist()}。并非精确成分分解。")
    lines+=['','A2只用outer-train标签算252连接的abs(univariate AUC−.5)，每域固定500次域内prior排序置乱，160行记录。仍受到历史全局RFE选择偏差；10个训练集高度重叠，不作独立生物标志物验证。','',
        '|域|prior|十折Spearman均值/范围|经验p<.05折数|','|---|---|---|---:|']
    for (domain,bank),g in ranks.groupby(['domain','prior']):
        lines.append(f'|{domain}|{bank}|{g.spearman.mean():+.5f} [{g.spearman.min():+.5f},{g.spearman.max():+.5f}]|{int((g.perm_p<.05).sum())}/10|')
        tops=[set(json.loads(v)) for v in g.univariate_top5];jac=[len(tops[i]&tops[j])/len(tops[i]|tops[j]) for i in range(10) for j in range(i)]
        lines.append(f"域{domain} {bank} 固定prior Top5={g.prior_top5.iloc[0]}；train判别Top5十折Jaccard均值={np.mean(jac):.5f}。")
    lines+=['','## B：临床证据审计','',f"{clinical['status']} / {clinical['clinical_alignment']}。原CSV {clinical.get('source_rows')}行、{clinical.get('source_unique_sub_ids')}唯一ID；871行可验证对应{clinical['verified_processed_rows']}/871；871×64复核={clinical['prefix_comparison']}。原因：{clinical['reason']}。",'',
        'clinical_coverage.csv为1112人原始源表的ASD/TD×站点及年龄分组汇总，明确标记UNMATCHED，不能把它解释成871人的缺失率。无ID猜测，无临床字段/缺失指示输入主模型。',
        f"文件来源 `{clinical['phenotype_source']}` SHA `{clinical['phenotype_sha256']}`；限定搜索 {clinical['bounded_search']}；不遍历全盘。字典为ABIDE I官方字段说明；空白/NaN计missing，低于量表特定合法下限（包括-9999）单列invalid，不泛化所有负数或臆造-999缺失码。ADOS/ADI-R与SRS版本/研究可靠性差异使跨类可比性不能由覆盖率自动证明。",'',
        '|量表字段|源ASD有效N/总N/站点|源TD有效N/总N/站点|871队列可用性|','|---|---|---|---|']
    for field in clinical.get('coverage_summary',[]):
        counts=field['source_counts'];ss=['{valid}/{subjects}/{sites}'.format(**x) for x in counts]
        lines.append(f"|{field['field']}|{ss[0]}|{ss[1]}|{field['classification']}|")
    lines+=['',f"不存在的候选族：{clinical.get('absent_candidate_families')}。不输出SYMMETRIC_CLINICAL_SUPERVISION_AVAILABLE；无当前队列可验证真值，C包仅称解剖与文献先验+训练标签驱动的弱监督对齐。",'',
        '## C：正式20组','',
        '指标%，先各seed十折均值再三seed mean±population SD。ASD=0，SPE-TD；Hard-AUC是硬预测，Prob-AUC用exp(logprob)的ASD类0概率。','',
        '|组|机制|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---|---:|---:|---:|---:|---:|---:|']
    complete=[c for c in CONFIGS if sum(r['config']==c for r in rows)==30];stats=summary([r for r in rows if r['config'] in complete]) if complete else {}
    for cfg,d in DESIGNS.items():
        mechanism=d['prompt']+('+D' if d['disease'] else '')+' / '+d['family']+' / '+str(d['condition'])+(' graph' if d['use_graph'] else ' local')
        values=[f"{100*stats[cfg][k]['mean']:.5f}±{100*stats[cfg][k]['sd']:.5f}" for k in METRICS] if cfg in stats else ['未完成']*6
        lines.append('|'+cfg+'|'+mechanism+'|'+'|'.join(values)+'|')
    if m.get('blocking_reason'):lines+=['',f"工程阻塞：{m['blocking_reason']}"]
    lines+=['','每seed六指标均值（%）：']
    for cfg in complete:
        for s in range(3):lines.append(f"- {cfg} seed{s}: "+' / '.join(f"{100*stats[cfg][k]['seeds'][s]:.5f}" for k in METRICS))
    lines+=['','预注册全部配对：单位pp，[]为seed0/1/2均值差，()为30fold胜/平/负。相同索引配对，不误称全程同初始化。']
    for left,right in PAIRS:
        if left not in stats or right not in stats:continue
        values=[]
        for k in METRICS:
            p=paired(rows,left,right,k);values.append(f"{k} {100*p['mean']:+.5f} [{' / '.join(f'{100*v:+.5f}' for v in p['seeds'])}] ({p['wins']}/{p['ties']}/{p['losses']})")
        lines.append(f'- {left}−{right}: '+'; '.join(values))
    lines+=['','逐组机制（30个best checkpoints均值；范数和attention是模型关联，不是患者症状/脑病灶真值）：']
    for cfg in complete:
        g=[r for r in ds if r['config']==cfg]
        fields=['alpha','eta','kl_train','kl_test','patient_attention_abs_change','residual_relative_mean','residual_relative_max','residual_nonzero_fraction','graph_effect','graph_semantic_weight_abs_change','adjacency_patient_variance','representation_cosine','edge_id_feature_change','branch_off_probability_change','graph_off_probability_change','disease_probability_correlation']
        values=[]
        for k in fields:
            vv=[float(r[k]) for r in g if r.get(k,'')!='']
            if vv:values.append(f'{k}={np.mean(vv):.8g} [{min(vv):.5g},{max(vv):.5g}]')
        for k in ['domain_attention_entropy','normalized_attention_entropy','prior_candidate_std']:
            vv=[json.loads(r[k]) for r in g if r.get(k,'')!='']
            if vv:values.append(k+'='+json.dumps(np.mean(vv,axis=0).tolist()))
        grad=[json.loads(r['gradient_norms']) for r in g];grad_keys=sorted(set(k for row in grad for k in row))
        values.append('sampled_max_gradient_means='+json.dumps({k:float(np.mean([r.get(k,0) for r in grad])) for k in grad_keys}))
        lines.append('- '+cfg+': '+'; '.join(values))
    lines+=['','同checkpoint反事实：每次REAL恢复、所有参数/buffer及原FC输入均逐值核对；不重训、不重选epoch；BRANCH_OFF为分布变化。图/文本/eta/FC/传播效应非加性，不能相加为解释率。','',
        '|组|干预|有效/总行|翻转%|mean abs Δp|mean abs Δlogprob|ΔACC pp|ΔBA pp|ΔProbAUC pp|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cfg in CF_GROUPS:
        for cond in CONDITIONS:
            sub=[r for r in cf if r['config']==cfg and r['condition']==cond];valid=[r for r in sub if str(r['applicable'])=='True']
            if not sub:continue
            if not valid:lines.append(f'|{cfg}|{cond}|0/{len(sub)} NA|NA|NA|NA|NA|NA|NA|');continue
            vals=[np.mean([float(r[k]) for r in valid])*(100 if k in ('prediction_flip','acc_change','ba_change','probability_auc_change') else 1) for k in ['prediction_flip','probability_abs_change','logit_abs_change','acc_change','ba_change','probability_auc_change']]
            lines.append(f'|{cfg}|{cond}|{len(valid)}/{len(sub)}|'+'|'.join(f'{v:.8g}' for v in vals)+'|')
    sci=scientific_status(rows,ds,cf)
    if len(rows)==600:
        winners={k:max(stats,key=lambda c:stats[c][k]['mean']) for k in ('acc','ba','probability_auc')}
        lines+=['',f'最佳均值组（仅描述）：{winners}。20组均保留，不能凭最大值宣布成功。',f"科学门槛：{json.dumps(sci,ensure_ascii=False)}。按三seed ACC/BA/ProbAUC全部正向及正确阴性控制/反事实核对，不冒称统计显著性。",'',
            'U16相对成熟E9/U01：'+', '.join(f"{k} {100*paired(rows,'U16','U01',k)['mean']:+.5f}pp" for k in METRICS)+'。',
            '医学语义必要性若未通过，则即使模块非零响应、图传播明显或单组ACC上涨，也不能作为MICCAI2027第一创新点。图与无图非等容量，语义建图结论必须以同图容量REAL−NUMERIC/NEUTRAL和GRAPH_NUMERIC干预为依据。',
            '后续唯一科学问题：在训练折内选模和独立站点评估中，可验证患者临床证据是否能约束出超过纯数值FC的可迁移增量；先修复原始受试者身份链，再讨论临床监督。' if sci['medical_semantics'].endswith('FAIL') else '后续唯一科学问题：冻结当前方法后，在无测试参与的选模和跨站点评估中能否复现正确医学语义的独立增益。']
        m=read_json(MANIFEST);m['scientific_conclusions']=sci;save_json(MANIFEST,m)
    lines+=['','文献与创新边界：MMGPL已提出概念加权和基于概念profile的token语义图及GCN；本轮有界患者修正/固定ROI/阴性对照仍需联合实证，不能声称concept graph本身首次。[MMGPL作者全文](https://arxiv.org/html/2312.14574v2)。',
        '原GFR/SPGS来自[SPromptGL](https://papers.miccai.org/miccai-2025/0861-Paper5129.html)，多图共性/私有分解来自[MGDR](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)，多层ROI/受试者/疾病prompt已有[BrainPrompt](https://arxiv.org/abs/2504.16096)。MMGPL的1029人、MRI/FC、5fold×3seed、BiomedCLIP、72.39%不与当前协议横比。',
        'FC为LedoitWolf相关/Fisher-z经RFE保留的稀疏连接；当前列已全局标准化。固定解剖候选域未经患者症状真值验证。A2排序与attention都不能当病理或因果结论。',
        f"历史parity：{m.get('historical_parity_180','待180正式fold核验')}；GPU各30任务300fold，最多4worker×2CPU线程；墙钟 {m.get('training_wall_seconds','未完成')}秒；资源峰值 {m.get('resource_peaks',{})}。",f"根handoff追加状态：{m.get('handoff_update','待最终追加')}；临时权重清理={m.get('temporary_state_cleaned',False)}；没有执行commit/push。"]
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    return lines


def append_handoff():
    m=read_json(MANIFEST);lock=acquire_lock(LOGS/'.handoff.lock')
    try:
        original=HANDOFF.read_bytes();assert hashlib.sha256(original[:m['handoff_prefix_bytes']]).hexdigest()==m['handoff_prefix_sha256']
        signature_=canonical(dict(status=m['run_status'],folds=m['completed_folds'],results=sha(RESULTS/'fold_metrics.csv') if (RESULTS/'fold_metrics.csv').exists() else None,conclusions=m['scientific_conclusions']))
        if m.get('handoff_entry_signature')==signature_:return
        text='\n\n## 实验二十一：三线验证与有界自适应疾病证据对齐\n\n'
        text+=f"时间 {utc()}；状态 **{m['run_status']}**。目录 `{EXP}`。根HEAD `{m['git_head_at_prepare']}`，GitHub参照 `{AUDIT_COMMIT}`；代码/模型/数据/文本/先验SHA见manifest。\n\n"
        text+='A：原实验20 checkpoint已清理，仅复核聚合反事实非加性响应和outer-train先验排序；不能精确分解历史T11/T18向量。B：'+m['audit_results']['status']+'，验证对应'+str(m['audit_results']['verified_processed_rows'])+'/871，'+m['audit_results']['clinical_alignment']+'；未将临床字段或缺失模式用于训练。\n\n'
        text+='C：20固定组，U00 P0/U01 E9；U02–05严格复现T05/T11/T10/T04；U06–13检验有界患者FC修正、数值/中性/错ROI/同Gram及患者线图；U14–19比较E9局部/图的真实/数值/中性。eta=0.35sigmoid，初值.1；温度.5、train KL=.02、图profile .7、零alpha且CORR残差≤10%。无自由Key/BCE证据头，原主干与E9不变。\n\n'
        text+=f"工程 {m['completed_tasks']}/60任务、{m['completed_folds']}/600fold、{m['diagnostic_rows']}/600诊断、{m['counterfactual_rows']}/1200反事实（local图条件120 NA）；parity={m.get('historical_parity_180','待核验')}。\n\n"
        rows=read_csv(RESULTS/'fold_metrics.csv')
        if len(rows)==600:
            ss=summary(rows);text+='六指标ACC/BA/Hard-AUC/ProbAUC/SEN-ASD/SPE-TD，%，三seed mean±population SD：\n\n'
            for cfg in CONFIGS:text+='- '+cfg+': '+' / '.join(f"{100*ss[cfg][k]['mean']:.5f}±{100*ss[cfg][k]['sd']:.5f}" for k in METRICS)+'\n'
            for l,r in PAIRS:text+='- '+l+'−'+r+': '+' / '.join(f"{k} {100*paired(rows,l,r,k)['mean']:+.5f}pp" for k in ('acc','ba','probability_auc'))+'\n'
        text+='\n科学门槛：'+json.dumps(m['scientific_conclusions'],ensure_ascii=False)+'。不把分支敏感性或最高ACC等同正确医学语义。三seed共享fold，val=test/test ACC选模与CV前全局RFE/标准化存在乐观偏差；不是独立无泄漏泛化。\n'
        if m.get('blocking_reason'):text+='工程阻塞：'+m['blocking_reason']+'；未完成部分不做科学结论。\n'
        text+=f"运行墙钟 {m.get('training_wall_seconds','未完成')}秒，两卡各30任务300fold，资源 {m.get('resource_peaks',{})}。完整A/B/C、机制/反事实、三seed差值与W/T/L见 `{RESULTS/'report.md'}`。\n"
        text+='后续唯一问题：在可信患者身份链及训练内选模、独立站点评估中，医学证据能否提供超过纯数值FC的可迁移增量。原源码与实验1–20未改；本条保留此前handoff全部字节，未commit/push。\n'
        atomic(HANDOFF,original+text.encode())
        m=read_json(MANIFEST);m.update(handoff_entry_signature=signature_,handoff_sha256=sha(HANDOFF),root_handoff_sha256=sha(HANDOFF),handoff_update=dict(status='ROOT_APPENDED',path=str(HANDOFF),original_prefix_preserved=True,bytes_appended=len(text.encode())))
        save_json(MANIFEST,m)
    except OSError as exc:
        m=read_json(MANIFEST);m.update(run_status='BLOCKED',blocking_reason='Root handoff append failed: '+repr(exc),handoff_update=dict(status='FAILED',error=repr(exc)));save_json(MANIFEST,m)
        atomic(EXP/'handoff.md',HANDOFF.read_bytes()+text.encode());raise
    finally:lock.close()


def main():
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');worker(sys.argv[i+1],int(sys.argv[i+2]));return
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','prepare','preflight','all','resume','refresh','report']);p.add_argument('--workers-per-gpu',type=int,default=4)
    args=p.parse_args()
    try:
        if args.command=='audit':audit()
        elif args.command=='prepare':prepare()
        elif args.command=='preflight':preflight()
        elif args.command in ('all','resume'):schedule(args.workers_per_gpu)
        elif args.command=='refresh':refresh()
        elif args.command=='report':report();append_handoff();report()
    except Exception as exc:
        if MANIFEST.exists() and args.command not in ('refresh','report'):
            m=read_json(MANIFEST);m.update(run_status='BLOCKED',blocking_reason=repr(exc));save_json(MANIFEST,m);report();append_handoff()
        raise

if __name__=='__main__':main()

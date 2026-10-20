#!/usr/bin/env python3
"""Experiment29 standalone controller: prepare -> preflight -> all/resume -> report.

20 config tasks, seed0, continuous 10 folds each. Historical P0/E1 read only.
Historical root-source copies and all checkpoints live in this experiment only.
"""
from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
import argparse,ast,contextlib,csv,datetime as dt,fcntl,hashlib,io,json,pickle
import random,shutil,signal,subprocess,time,types,urllib.request
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import prompt_bank,integration
from method import CONFIGS,DESIGNS,SEEDS,METRICS,CONTRASTS,INTERVENTIONS,RULES,tasks,asdict
ROOT=Path('/data/adodas/SPromptGL-main')
EXP=Path(__file__).resolve().parent
RUNTIME=EXP/'.runtime'
SOURCE=RUNTIME/'sources'
RESULTS=EXP/'results'
MANIFEST=EXP/'manifest.json'
CACHE=EXP/'prompt_banks.npz'
HIST=ROOT/'experiment/experiment28_imaging_anchored_finegrained_semantic_alignment'
BASELINE=ROOT/'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv'
CORE=('main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh')
CODE=('method.py','prompt_bank.py','integration.py','run.py','integration.patch')
GPU={(c,s):g for c,s,g in tasks()}
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
    nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,
    n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',
    alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return prompt_bank.sha(p)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

@contextlib.contextmanager
def json_lock(exclusive):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'json.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def jread(path):
    for attempt in range(8):
        try:
            with json_lock(False):return json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt==7:raise
            time.sleep(.15)

def jwrite(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with json_lock(True):
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('w') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)

def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))

def cwrite(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def record_event(message):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'scheduler.log').open('a') as f:f.write(utc()+' '+str(message)+'\n');f.flush()
    print(message,flush=True)

def core_hashes():return {n:sha(ROOT/n) for n in CORE}
def code_hashes():return {n:sha(EXP/n) for n in CODE}
def build_sources():return integration.emit(ROOT,SOURCE,EXP/'integration.patch')
def modules(a):return (a.MessagePassing0,a.ModalFusion)

def data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    frame=pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv')
    mapping=np.load(ROOT/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert [len(v) for v in mapping.values()]==[48,6,10,256]
    assert list(frame.columns[:-1])==[str(c) for v in mapping.values() for c in v]
    assert list(frame.columns[64:320])==list(map(str,range(256)))
    x=frame.iloc[:,:-1].values;y=frame.iloc[:,-1].values.astype(int)-1
    assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
    names=list(map(str,mapping['PHENO']))
    ph=np.array([i for i,n in enumerate(names) if n.startswith(('AGE_AT_SCAN_','SEX_'))])
    si=np.array([i for i,n in enumerate(names) if n.startswith('SITE_ID_')])
    assert len(ph)==28 and len(si)==20 and not set(ph)&set(si)
    site=x[:,si].argmax(1)
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
    for tr,te in folds:assert not set(tr)&set(te) and set(tr)|set(te)==set(range(871))
    return dict(x=x,y=y,site=site,pheno_indices=ph,site_indices=si,folds=folds)

def seed_rng(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def rng():
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    import torch
    # A fresh resumed worker must also retain the original backend settings.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])

def rng_hash(s):
    h=hashlib.sha256(pickle.dumps((s['python'],s['numpy']),protocol=4))
    for v in [s['cpu']]+s['cuda']:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def model_state(a):return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in zip(('gfr','pln'),modules(a))}

def load_models(a,s):
    for k,m in zip(('gfr','pln'),modules(a)):m.load_state_dict(s[k])

def state_equal(a,b):
    import torch
    assert a.keys()==b.keys()
    for part in a:
        assert a[part].keys()==b[part].keys()
        for k in a[part]:assert torch.equal(a[part][k],b[part][k]),(part,k)

def checkpoint_write(p,obj):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=io.BytesIO();torch.save(obj,b);body=b.getvalue();tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(b'E29!'+hashlib.sha256(body).hexdigest().encode()+body);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def checkpoint_read(p):
    import torch
    b=Path(p).read_bytes();assert b[:4]==b'E29!' and hashlib.sha256(b[68:]).hexdigest().encode()==b[4:68]
    return torch.load(io.BytesIO(b[68:]),map_location='cpu',weights_only=False)

def recursive_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:recursive_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):recursive_equal(x,y)
    else:assert a==b,(a,b)

def corr(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.std()<1e-12 or b.std()<1e-12:return 0.
    return float(np.corrcoef(a,b)[0,1])

def eta_squared(values,groups):
    v=np.asarray(values);g=np.asarray(groups);den=float(((v-v.mean())**2).sum())
    return float(sum(len(v[g==s])*(v[g==s].mean()-v.mean())**2 for s in np.unique(g))/max(den,1e-20))

def signature():
    m=jread(MANIFEST)
    return canonical({k:m[k] for k in ('source_sha256','data_sha256','code_sha256','prompt_cache_sha256','design','rules','protocol')})

def verify_static():
    m=jread(MANIFEST)
    assert core_hashes()==m['source_sha256'],'Root core changed'
    assert code_hashes()==m['code_sha256'],'Frozen experiment code changed'
    assert sha(CACHE)==m['prompt_cache_sha256']
    for p,h in m['data_sha256'].items():assert sha(p)==h,p
    for p,h in m['reference_sha256'].items():assert sha(p)==h,p
    return m

def references():
    raw=readcsv(HIST/'results/fold_metrics.csv')
    hf=[r for r in readcsv(BASELINE) if r['config']=='F0' and int(r['seed'])==0]
    assert len(hf)==10
    result=[]
    for config,source in (('P0','E0_P0'),('E1','E1_NO_SPGS')):
        rows=[r for r in raw if r['config']==source and int(r['seed'])==0]
        assert len(rows)==10 and sorted(int(r['fold']) for r in rows)==list(range(1,11))
        for r in rows:
            f=int(r['fold']);counts={k:int(r[k]) for k in ('asd_tp','asd_fn','td_fp','td_tn')}
            tp,fn,fp,tn=(counts[k] for k in ('asd_tp','asd_fn','td_fp','td_tn'))
            if config=='P0':
                h=next(v for v in hf if int(v['fold'])==f)
                for key in ('acc','hard_auc','best_epoch','initial_trunk_sha256'):
                    if key=='hard_auc':assert abs(float(r[key])-float(h[key]))<2e-15,'Historical ROC/recall rounding '+str(f)
                    else:assert r[key]==h[key] or (key!='initial_trunk_sha256' and float(r[key])==float(h[key])),key
                assert all(int(h[k])==counts[k] for k in counts),'Historical confusion differs'
            result.append(dict(config=config,seed=0,fold=f,acc=float(r['acc']),auc=float(r['hard_auc']),
                sen_asd=tp/(tp+fn+1e-8),spe_td=tn/(tn+fp+1e-8),**counts,
                best_epoch=int(r['best_epoch']),initial_trunk_sha256=r['initial_trunk_sha256']))
    return result

def prepare():
    if MANIFEST.exists():verify_static();return
    started=utc();RESULTS.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    from concurrent.futures import ThreadPoolExecutor
    hashes=core_hashes()
    def remote_check(item):
        name,digest=item
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/tangaii/SPromptGL/{remote}/{name}',timeout=30) as f:body=f.read()
        assert hashlib.sha256(body).hexdigest()==digest,'Remote core mismatch '+name
        return name,digest
    with ThreadPoolExecutor(max_workers=6) as ex:remote_hashes=dict(ex.map(remote_check,hashes.items()))
    old=json.loads((HIST/'manifest.json').read_text());assert old['status']=='COMPLETE' and old['completed_folds']==300
    assert old['source_sha256']==hashes
    protected={Path(p) for p in old['protected_files_sha256'] if Path(p).is_file()}
    protected.update(p for folder in (ROOT/'experiment').glob('experiment*') if folder!=EXP for p in folder.rglob('*')
                     if p.is_file() and not any(part in ('.runtime','_state','__pycache__') for part in p.relative_to(folder).parts))
    protected.discard(ROOT/'handoff.md');protected.discard(ROOT/'prompt')
    protection={str(p):sha(p) for p in sorted(protected)}
    info=data();ref=references();assert len(ref)==20
    textinfo=prompt_bank.build(ROOT,EXP);build_sources()
    hand=(ROOT/'handoff.md').read_bytes()
    input_files=[ROOT/'data/ABIDE/processed_standard_data.csv',ROOT/'data/ABIDE/modal_feat_dict.npy']
    input_files+=list((ROOT/'experiment/experiment11_roi_mapping_recovery/results').glob('roi_*csv'))
    audited={}
    for folder in (ROOT/'experiment/experiment11_roi_mapping_recovery',ROOT/'experiment/experiment13_patient_specific_fc_prompt',HIST,
                   Path('/tmp/experiment28_supplement_e10_coarse_full')):
        if not folder.is_dir():continue
        for p in folder.glob('*.py'):
            content=p.read_text();ast.parse(content);audited[str(p)]={'sha256':sha(p),'lines':len(content.splitlines())}
        for p in (folder/'results').glob('*.md'):
            content=p.read_text();audited[str(p)]={'sha256':sha(p),'characters':len(content)}
    m=dict(experiment='29 CORR-dominant dual-level semantic alignment and selective auxiliary repair',status='PREPARED',
        started_utc=started,prepared_utc=utc(),source_local_head=local,source_remote_head=remote,
        source_sha256=hashes,source_remote_core_sha256=remote_hashes,source_remote_core_identical=True,
        data_sha256={str(p):sha(p) for p in input_files},
        reference_sha256={str(p):sha(p) for p in (BASELINE,HIST/'results/fold_metrics.csv',HIST/'manifest.json',HIST/'prompt_banks.npz',HIST/'prompt_banks.json')},
        protected_files_sha256=protection,original_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        original_handoff_size=len(hand),original_handoff_sha256=hashlib.sha256(hand).hexdigest(),
        prompt_source_sha256=sha(ROOT/'prompt'),prompt_cache_sha256=sha(CACHE),text_bank=textinfo,
        audit=audited,papers=dict(spromptgl='https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf',mmgpl='https://arxiv.org/html/2312.14574v2'),
        paper_local_sha256={p.name:sha(p) for p in ROOT.glob('*SPromptGL*.pdf')},
        design={c:asdict(DESIGNS[c]) for c in CONFIGS},rules=RULES,protocol=PROTOCOL,
        tasks=[dict(config=c,seed=s,gpu=g,status='PENDING',completed_folds=0) for c,s,g in tasks()],
        target_tasks=20,target_folds=200,completed_tasks=0,completed_folds=0,gpu_fold_targets={'0':100,'1':100},
        contrasts=CONTRASTS,historical_references=dict(P0='Exp13 F0 seed0 full10, Exp28 E0 exact parity',E1='Exp28 E1 seed0 full10'),
        expected_baseline_parameters=1462957,initialization='Historical Exp13/28 cold RN50 construction CPU RNG; appended branch fork_rng seed290000; no between-fold reseed',
        metrics_note='ONLY ACC/hard-label AUC/ASD SEN/TD SPE; mean/population SD over 10 folds of seed0. SEN/SPE original 1e-8 denominators.',
        duplicate_design_note='C3 and F3 are mathematically identical semantic-OFF controls; both are registered and trained independently as requested.',
        runner_provenance='Data/seed/RNG/state/equality/checkpoint helpers copied exactly from accepted Exp28 runner; checkpoint magic E29!',
        failures=[],engineering_fixes=[dict(stage='prepare',issue='Exp28 hard_auc is recall average; Exp13 uses ROC; folds5/6 differ by 1 ULP',
            resolution='Require exact confusion counts, ACC, best epoch and initialization hashes; hard-AUC tolerance2e-15 only. Formal metrics retain original ROC definition.')],handoff_appended=False)
    m['code_sha256']=code_hashes();jwrite(MANIFEST,m)
    record_event('PREPARED 20 tasks / seed0 / 200 folds. Frozen medical texts reused without encoding.')

def context():
    global _CTX
    if _CTX is not None:return _CTX
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Worker must see exactly one GPU'
    sys.path.insert(0,str(build_sources()))
    import model,network,layers,utils
    for mod in (model,network,layers,utils):assert Path(mod.__file__).resolve().parent==SOURCE.resolve()
    cache=prompt_bank.load(CACHE)
    assert tuple(network.attr_words)==prompt_bank.ORIGINAL
    def cached_p0():
        if network._prompt_text_features is None:
            encoder,_=network.clip.load('RN50',network.device);del encoder
            network._prompt_text_features=torch.from_numpy(cache['p0']).to('cuda')
        return network._prompt_text_features
    network.get_prompt_text_features=cached_p0
    from CLIP.clip.model import CLIP
    def forbidden(*args,**kwargs):raise RuntimeError('Text encoding forbidden in training worker')
    CLIP.encode_text=forbidden
    _CTX=dict(torch=torch,model=model,network=network,cache=cache,**data())
    return _CTX

def make_agent(config,seed,fold,labels=None,cls=None):
    c=context();tr,te=c['folds'][fold-1];y=c['y'] if labels is None else labels
    integration.configure(config=config,fold=fold,raw=c['x'],train=tr,test=te,labels_train=y[tr],
                          site=c['site'],pheno_indices=c['pheno_indices'],cache=c['cache'])
    with contextlib.redirect_stdout(io.StringIO()):
        a=(cls or c['model'].EvalHelper)([48,6,10,256],c['x'],y,SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close();assert np.array_equal(a.val_idx,a.tst_idx)
    return a

def parameter_hash(a,branch=False):
    h=hashlib.sha256()
    for m in modules(a):
        for n,p in sorted(m.named_parameters()):
            if ('semantic29.' in n)!=branch:continue
            h.update(n.encode()+p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def optimizer_audit(a):
    groups=a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups
    ids=[id(p) for g in groups for p in g['params']]
    params=[p for m in modules(a) for p in m.parameters()]
    assert len(ids)==len(set(ids)) and set(ids)==set(map(id,params))
    assert all(g['lr']==.0038 for g in groups) and groups[0]['weight_decay']==groups[1]['weight_decay']==.11
    b=getattr(a.ModalFusion,'semantic29',None)
    new=sum(p.numel() for p in b.parameters()) if b is not None else 0
    assert sum(p.numel() for p in params)==1462957+new
    assert len(groups)==(3 if b is not None else 2)
    if b is not None:assert groups[2]['weight_decay']==0
    return sum(p.numel() for p in params),new

def evaluate(a):
    torch=context()['torch']
    for m in modules(a):m.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat);z,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        out,_,_=a.ModalFusion(z,raw_input=a.feat)
    assert out.shape==(871,2) and torch.isfinite(out).all()
    return out

def metrics(out,test):
    from sklearn.metrics import roc_auc_score,confusion_matrix
    pred=out.detach().cpu().numpy()[test].argmax(1);y=context()['y'][test]
    tp,fn,fp,tn=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    auc=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred]))
    assert abs(auc-.5*(tp/(tp+fn)+tn/(tn+fp)))<1e-12
    return dict(acc=float((pred==y).mean()),auc=auc,sen_asd=tp/(tp+fn+1e-8),spe_td=tn/(tn+fp+1e-8),
                asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)

def gradient_norms(a):
    torch=context()['torch'];result={}
    for name,module in [('gfr',a.MessagePassing0),('pln_original',a.ModalFusion)]:
        params=[p for n,p in module.named_parameters() if name!='pln_original' or not n.startswith('semantic29.')]
        value=sum((p.grad.detach().square().sum() for p in params if p.grad is not None),a.feat.new_zeros(()))
        result[name]=float(value.sqrt())
    b=getattr(a.ModalFusion,'semantic29',None)
    if b:
        for name,module in list(b.named_children())+[(k,getattr(b,k)) for k in ('alpha_sem','alpha_aux')]:
            params=[module] if isinstance(module,torch.nn.Parameter) else list(module.parameters())
            result[name]=float(sum((p.grad.detach().square().sum() for p in params if p.grad is not None),a.feat.new_zeros(())).sqrt())
        for name in ('PHENO','ANAT','FUNC'):
            ps=list(b.selectors[name].parameters())+list(b.encoders[name].parameters())+list(b.routers[name].parameters())+list(b.values[name].parameters())
            result['aux_'+name]=float(sum((p.grad.detach().square().sum() for p in ps if p.grad is not None),a.feat.new_zeros(())).sqrt())
    assert all(np.isfinite(v) for v in result.values())
    return result

def raw_forward(a):
    n,parts=a.dataLayer(a.feat);z,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
    return a.ModalFusion(z,raw_input=a.feat)[0]

def smoke():
    c=context();torch=c['torch'];checks={}
    # Execute untouched root source in distinct namespaces, with historical cold
    # CLIP constructor RNG and frozen p0 embeddings, exactly as the accepted runner.
    original_network=types.ModuleType('_e29_unpatched_network');original_network.__file__=str(ROOT/'network.py')
    sys.modules[original_network.__name__]=original_network
    exec(compile((ROOT/'network.py').read_text(),str(ROOT/'network.py'),'exec'),original_network.__dict__)
    def cached_original():
        if original_network._prompt_text_features is None:
            model,_=original_network.clip.load('RN50',original_network.device);del model
            original_network._prompt_text_features=torch.from_numpy(c['cache']['p0']).to('cuda')
        return original_network._prompt_text_features
    original_network.get_prompt_text_features=cached_original
    original_model=types.ModuleType('_e29_unpatched_model')
    source=(ROOT/'model.py').read_text().replace('from network import *','from _e29_unpatched_network import *',1)
    exec(compile(source,str(ROOT/'model.py'),'exec'),original_model.__dict__)
    seed_rng(0);base=make_agent('P0',0,1,cls=original_model.EvalHelper)
    expected=model_state(base);after=rng();initial=parameter_hash(base)
    assert initial==next(r for r in references() if r['config']=='P0' and r['fold']==1)['initial_trunk_sha256']
    seed_rng(0);patched=make_agent('P0',0,1)
    state_equal(expected,model_state(patched));assert rng_hash(after)==rng_hash(rng())
    for m in modules(base):m.eval()
    with torch.no_grad():
        n,parts=base.dataLayer(base.feat);z,_,_=base.MessagePassing0(parts,n,base.targ,base.trn_idx)
        baseline_output=base.ModalFusion(z)[0]
    assert torch.equal(baseline_output,evaluate(patched))
    entry=rng()
    with contextlib.redirect_stdout(io.StringIO()):base.run_epoch('simple-2')
    expected=model_state(base);after=rng()
    restore_rng(entry)
    with contextlib.redirect_stdout(io.StringIO()):patched.run_epoch('simple-2')
    state_equal(expected,model_state(patched));recursive_equal(base.optimizer_MP0.state_dict(),patched.optimizer_MP0.state_dict())
    recursive_equal(base.optimizer_MF.state_dict(),patched.optimizer_MF.state_dict());assert rng_hash(after)==rng_hash(rng())
    checks['P0_untouched_initial_state_forward_two_adam_update_rng']='BITWISE_EXACT'
    checks['P0_seed0_fold1_initial_trunk_sha256']=initial
    del base,patched
    # Legacy Exp28 E1 network is authoritative for the SPGS-OFF/RES path.
    legacy_ns={'ast':ast}
    legacy_text=(HIST/'e28_integration.py').read_text();tree=ast.parse(legacy_text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='patch_network')
    exec(ast.get_source_segment(legacy_text,fn),legacy_ns)
    legacy=types.ModuleType('_e29_legacy_e1_network');sys.modules[legacy.__name__]=legacy
    exec(compile(legacy_ns['patch_network']((ROOT/'network.py').read_text()),str(HIST/'e28_integration.py'),'exec'),legacy.__dict__)
    legacy.get_prompt_text_features=lambda:c['network'].get_prompt_text_features()
    seed_rng(0);entry=rng();e1=make_agent('E1',0,1);oldhash=parameter_hash(e1);oldrng=rng_hash(rng());oldout=evaluate(e1)
    legacy_pln=legacy.PLN([48,6,10,256],SimpleNamespace(**PROTOCOL,seed=0)).to('cuda')
    legacy_pln.load_state_dict(e1.ModalFusion.state_dict());legacy_pln.e28_config='E1_NO_SPGS';legacy_pln.eval()
    with torch.no_grad():
        n,parts=e1.dataLayer(e1.feat);z,_,_=e1.MessagePassing0(parts,n,e1.targ,e1.trn_idx)
        assert torch.equal(legacy_pln(z)[0],oldout)
    checks['SPGS_OFF_RES_matches_historical_E28_E1']='BITWISE_EXACT'
    counts=[];newhash=None;selected=None
    for cfg in CONFIGS:
        restore_rng(entry);a=make_agent(cfg,0,1);b=a.ModalFusion.semantic29
        total,new=optimizer_audit(a);counts.append(new)
        assert parameter_hash(a)==oldhash and rng_hash(rng())==oldrng,cfg
        assert torch.equal(evaluate(a),oldout),cfg
        if newhash is None:newhash=parameter_hash(a,True)
        assert parameter_hash(a,True)==newhash,cfg
        assert b.alpha_sem.item()==b.alpha_aux.item()==0
        assert not any(v.requires_grad for n,v in b.named_buffers())
        assert b.original_text.shape==(8,64) and b.fine_text.shape==(252,2,64)
        if selected is None:selected=(b.selected_corr.clone(),b.selected_evidence.clone())
        assert torch.equal(selected[0],b.selected_corr) and torch.equal(selected[1],b.selected_evidence)
        assert not np.isin(b.selected_corr.cpu(),prompt_bank.EXCLUDED).any()
        assert b.selected_valid.sum(1).max()<=16
        perm=b.aux_permutation.cpu().numpy()
        for part in (a.trn_idx,a.tst_idx):assert set(perm[part])==set(part)
        assert np.array_equal(c['site'][perm],c['site'])
        del a,b
    assert len(set(counts))==1 and counts[0]==71277
    checks['all20_original_and_new_parameter_initialization_rng']='BITWISE_EXACT'
    checks['additional_parameters_all20']=71277
    checks['ROI252_unknown4_excluded_same_selection_all20']='PASS'
    # Actual train loss, model and both Adam updates unchanged by held-out labels.
    restore_rng(entry);a=make_agent('F2',0,1);start=rng()
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    expected=model_state(a);after=rng();first=gradient_norms(a)
    assert first['alpha_sem']>0 and first['alpha_aux']>0
    assert first['fc_encoder']>0 and first['query']>0
    labels=c['y'].copy();labels[a.tst_idx]=1-labels[a.tst_idx]
    restore_rng(entry);other=make_agent('F2',0,1,labels=labels)
    restore_rng(start)
    with contextlib.redirect_stdout(io.StringIO()):other.run_epoch('simple-2')
    state_equal(expected,model_state(other));recursive_equal(a.optimizer_MP0.state_dict(),other.optimizer_MP0.state_dict())
    recursive_equal(a.optimizer_MF.state_dict(),other.optimizer_MF.state_dict());assert rng_hash(after)==rng_hash(rng())
    checks['heldout_label_flip_statistics_forward_full_train_two_adams_rng']='BITWISE_EXACT'
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    second=gradient_norms(a)
    for name in ('fc_encoder','query','edge_id','semantic_decode','aux_decode','alpha_sem','alpha_aux','aux_PHENO','aux_ANAT','aux_FUNC'):
        assert second[name]>0,(name,second)
    checks['first_step_gradient_norms']=first;checks['second_step_gradient_norms']=second
    b=a.ModalFusion.semantic29;real=evaluate(a)
    assert b.last['delta'].abs().max()<=.10
    with torch.no_grad():
        cam=b.last['cam'].clone();savedmask=b.selected_valid.clone();b.selected_valid.zero_()
        b(cam,a.feat);terms=b.loss_terms(a.targ)
        assert terms['local_loss'].item()==0 and torch.count_nonzero(b.last['local_response'])==0
        b.selected_valid.copy_(savedmask)
        rawzero=a.feat.clone();rawzero[:,:64]=0;b(cam,rawzero)
        assert all(torch.count_nonzero(value)==0 for value in b.last['aux_values'].values())
    checks['no_valid_local_fallback_and_zero_aux_no_bias_bypass']='PASS'
    before_params=parameter_hash(a)+parameter_hash(a,True);before_rng=rng_hash(rng());real=evaluate(a)
    changed={}
    for condition in INTERVENTIONS:
        b.condition=condition;pred=evaluate(a);changed[condition]=float((pred.exp()-real.exp()).abs().mean())
        b.condition='REAL';assert torch.equal(real,evaluate(a)),condition
    assert before_params==parameter_hash(a)+parameter_hash(a,True) and before_rng==rng_hash(rng())
    assert changed['NUMERIC_TEXT']>0
    checks['same_checkpoint_text_intervention_REAL_restore_weights_rng']='BITWISE_EXACT'
    checks['two_step_intervention_probability_changes']=changed
    # Global/local losses must never supervise the auxiliary encoders directly.
    raw_forward(a);terms=b.loss_terms(a.targ)
    aux_params=[p for m in (b.selectors,b.encoders,b.routers,b.values,b.aux_decode) for p in m.parameters()]
    for name in ('global_loss','local_loss'):
        gs=torch.autograd.grad(terms[name],aux_params,allow_unused=True,retain_graph=True)
        assert all(v is None or torch.count_nonzero(v)==0 for v in gs),name
    checks['global_local_loss_do_not_supervise_auxiliary']='PASS'
    # Check the requested 25-epoch boundary and the next stochastic update.
    for epoch in range(2,25):
        with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    path=RUNTIME/'preflight_resume.pt'
    payload=dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),
                 epoch=24,best_model=model_state(a),wait=7,best=.5,best_epoch=10)
    checkpoint_write(path,payload)
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    expected=model_state(a);after=rng();prediction=evaluate(a)
    stored=checkpoint_read(path);restored=make_agent('F2',0,1)
    load_models(restored,stored['model']);restored.optimizer_MP0.load_state_dict(stored['mp']);restored.optimizer_MF.load_state_dict(stored['mf'])
    restore_rng(stored['rng'])
    with contextlib.redirect_stdout(io.StringIO()):restored.run_epoch('simple-2')
    state_equal(expected,model_state(restored));recursive_equal(a.optimizer_MP0.state_dict(),restored.optimizer_MP0.state_dict())
    recursive_equal(a.optimizer_MF.state_dict(),restored.optimizer_MF.state_dict())
    assert rng_hash(after)==rng_hash(rng()) and torch.equal(prediction,evaluate(restored));path.unlink()
    boundary=rng();next_agent=make_agent('F2',0,2);nextstate=model_state(next_agent);nextrng=rng()
    restore_rng(boundary);repeat=make_agent('F2',0,2)
    state_equal(nextstate,model_state(repeat));assert rng_hash(nextrng)==rng_hash(rng())
    checks['epoch25_disk_both_adams_rng_next_update_prediction']='BITWISE_EXACT'
    checks['fold_boundary_next_initialization_rng']='BITWISE_EXACT'
    checks['device']=str(torch.cuda.get_device_properties(0))
    jwrite(RUNTIME/'preflight_checks.json',checks)
    print('E29_GPU_PREFLIGHT_PASS',flush=True)

def worker_env(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),E29_PHYSICAL_GPU=str(gpu),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    return env

def preflight():
    prepare();m=verify_static()
    if m.get('preflight',{}).get('passed'):return
    from method import self_test
    self_test()
    path=RUNTIME/'preflight.log'
    with path.open('w') as f:
        p=subprocess.run([sys.executable,str(EXP/'run.py'),'_smoke'],cwd=RUNTIME,env=worker_env(0),stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('GPU preflight failed: '+path.read_text()[-6000:])
    checks=jread(RUNTIME/'preflight_checks.json')
    gpu1=subprocess.check_output([sys.executable,'-c',"import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0))"],env=worker_env(1),text=True).strip()
    checks['GPU1_visible_logical0']=gpu1
    m=jread(MANIFEST);m['preflight']=dict(passed=True,completed_utc=utc(),checks=checks);m['status']='PREFLIGHT_PASS'
    jwrite(MANIFEST,m);record_event('PREFLIGHT_PASS exact baseline/zero residual/label boundary/gradients/25epoch+fold resume. No formal folds trained.')

def rank_summary(x):
    x=np.asarray(x,np.float64);x=x-x.mean(0);s=np.linalg.svd(x,compute_uv=False)
    if float((s*s).sum())<1e-25:return dict(effective_rank=0.,variance=0.)
    p=s*s/float((s*s).sum());p=p[p>0]
    return dict(effective_rank=float(np.exp(-(p*np.log(p)).sum())),variance=float((x*x).sum()/len(x)))

def counterfactual(a,real):
    b=a.ModalFusion.semantic29;before=parameter_hash(a)+parameter_hash(a,True);before_rng=rng_hash(rng())
    rows=[];te=a.tst_idx;realprob=real.exp()[te,0];realpred=real[te].argmax(-1)
    for mode in INTERVENTIONS:
        b.condition=mode;out=evaluate(a);prob=out.exp()[te,0]
        rows.append(dict(condition=mode,probability_abs_change=float((prob-realprob).abs().mean()),
            prediction_flip_fraction=float((out[te].argmax(-1)!=realpred).float().mean()),**metrics(out,te)))
    b.condition='REAL';assert torch_equal(real,evaluate(a))
    assert before==parameter_hash(a)+parameter_hash(a,True) and before_rng==rng_hash(rng())
    return rows

def torch_equal(a,b):return context()['torch'].equal(a,b)

def diagnostics(a,config,fold,tracker):
    c=context();torch=c['torch'];b=a.ModalFusion.semantic29;evaluate(a);v=b.last
    def arr(t):return t.detach().cpu().numpy()
    tr=a.trn_idx;te=a.tst_idx;valid=arr(b.selected_valid);att=arr(v['attention'])
    h=arr(torch.nn.functional.normalize(v['h'],dim=-1));labels=c['y'][tr]
    margin=arr(v['global_logits'][:,0]-v['global_logits'][:,1])
    entropy=-(att*np.log(np.maximum(att,1e-12))).sum(1)
    counts=valid.sum(1);proto,_=b.text_bank()
    row=dict(config=config,seed=0,fold=fold,total_parameters=optimizer_audit(a)[0],additional_parameters=71277,
        semantic_enabled=b.design.semantic,soft_sparse=b.design.sparse,active_auxiliary=','.join(b.design.auxiliary),
        global_class_prototype_cosine=float((proto[0]*proto[1]).sum()),
        valid_connections_mean=float(counts.mean()),valid_connections_min=int(counts.min()),valid_connections_max=int(counts.max()),
        no_valid_fraction=float((counts==0).mean()),attention_entropy_mean=float(entropy.mean()),attention_effective_count=float(np.exp(entropy[counts>0]).mean()) if np.any(counts>0) else 0.,
        attention_effective_rank=rank_summary(att)['effective_rank'],anchor_effective_rank=rank_summary(h[tr])['effective_rank'],
        local_response_effective_rank=rank_summary(arr(v['local_response'])[tr])['effective_rank'],
        train_class_margin_asd=float(margin[tr][labels==0].mean()),train_class_margin_td=float(margin[tr][labels==1].mean()),
        train_class_margin_separation=float(margin[tr][labels==0].mean()-margin[tr][labels==1].mean()),
        alpha_sem=float(b.alpha_sem.detach()),alpha_aux=float(b.alpha_aux.detach()),
        residual_abs_max=float(v['delta'].abs().max()),residual_l2_mean=float(v['delta'].norm(dim=-1).mean()),
        residual_nonzero_fraction=float((v['delta'].norm(dim=-1)>1e-8).float().mean()),
        semantic_residual_l2_mean=float((.1*v['semantic_term'].tanh()).norm(dim=-1).mean()),
        auxiliary_residual_l2_mean=float((.1*v['auxiliary_term'].tanh()).norm(dim=-1).mean()),
        statistics_json=json.dumps(b.statistics['report'],sort_keys=True),
        sampled_gradient_norms_json=json.dumps(tracker,sort_keys=True),
        auxiliary_shuffled_fraction=float((arr(b.aux_permutation)!=np.arange(871)).mean()) if b.design.shuffled else 0.)
    assert row['residual_abs_max']<=.1+1e-7
    if b.design.semantic:
        positive=v['local_logits'].diagonal(dim1=1,dim2=2)*.2
        directions=b.direction[b.selected_ids]
        _,current_bank=b.text_bank()
        wrong=current_bank[b.roi_permutation][b.selected_ids,directions]
        wrong_cos=(v['query']*wrong).sum(-1)
        row['matched_local_cosine']=float(positive[b.selected_valid].mean()) if b.selected_valid.any() else 0.
        row['wrong_ROI_local_cosine']=float(wrong_cos[b.selected_valid].mean()) if b.selected_valid.any() else 0.
    else:row.update(matched_local_cosine=0.,wrong_ROI_local_cosine=0.)
    for name in ('PHENO','ANAT','FUNC'):
        gate=arr(v['feature_gates'][name]);router=arr(v['relevances'][name]).ravel();mean=gate.mean(1)
        row[name+'_gate_mean']=float(gate.mean());row[name+'_gate_std']=float(gate.std())
        row[name+'_gate_below_01_fraction']=float((gate<.1).mean())
        row[name+'_gate_below_001_fraction']=float((gate<.01).mean())
        row[name+'_effective_dimensions']=float((gate>.1).sum(1).mean())
        row[name+'_router_mean']=float(router.mean());row[name+'_router_std']=float(router.std())
        row[name+'_value_l2_mean']=float(v['aux_values'][name].norm(dim=-1).mean())
        row[name+'_gate_site_eta_squared']=eta_squared(mean,c['site'])
        row[name+'_gate_QC_max_abs_correlation']=max(abs(corr(mean,c['x'][:,j])) for j in range(48,64))
    # Best-checkpoint loss-specific gradients; no optimizer step or test labels.
    for m in modules(a):m.zero_grad(set_to_none=True)
    out=raw_forward(a);terms=b.loss_terms(a.targ);cls=c['model'].ClsLoss(out,a.targ,tr,a.weight)
    params=list(b.fc_encoder.parameters())+list(b.query.parameters())
    for name,loss in [('classification',cls)]+list(terms.items()):
        gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        row[name+'_anchor_query_gradient']=float(sum((g.detach().square().sum() for g in gradients if g is not None),out.new_zeros(())).sqrt())
        row[name+'_value']=float(loss.detach())
    b.condition='NUMERIC_TEXT';out=raw_forward(a);numeric_terms=b.loss_terms(a.targ)
    loss=.05*numeric_terms['global_loss']+.02*numeric_terms['local_loss']
    gradients=torch.autograd.grad(loss,params,allow_unused=True)
    row['numeric_text_semantic_gradient']=float(sum((g.detach().square().sum() for g in gradients if g is not None),out.new_zeros(())).sqrt())
    b.condition='REAL';real=evaluate(a);row['counterfactual_json']=json.dumps(counterfactual(a,real),sort_keys=True)
    return row

def task_name(config,seed=0):return config+'_seed'+str(seed)
def task_checkpoint(config,seed=0):return RUNTIME/'state'/(task_name(config,seed)+'.pt')
def task_json(config,seed=0):return RUNTIME/'tasks'/(task_name(config,seed)+'.json')

def task_result(config,seed=0):
    path=task_json(config,seed)
    if not path.exists():return None
    r=jread(path)
    assert r['config']==config and r['seed']==seed and r['signature']==signature()
    assert len(r['metrics'])==len(r['diagnostics'])
    assert [row['fold'] for row in r['metrics']]==list(range(1,len(r['metrics'])+1))
    return r

def worker(config,seed):
    assert config in CONFIGS and seed==0
    m=verify_static();assert m.get('preflight',{}).get('passed')
    assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['E29_PHYSICAL_GPU']==str(GPU[config,seed])
    c=context();torch=c['torch'];name=task_name(config,seed);sig=signature()
    (RUNTIME/'state').mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'state'/(name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=task_result(config,seed)
    if existing and existing['status']=='COMPLETE':print('SKIP_COMPLETE',name,flush=True);return
    path=task_checkpoint(config,seed);saved=checkpoint_read(path) if path.exists() else None
    if saved:
        assert saved['signature']==sig
        result=saved['result']
        assert not existing or len(result['metrics'])>=len(existing['metrics'])
        c['network']._prompt_text_features=torch.from_numpy(c['cache']['p0']).to('cuda')
        restore_rng(saved['rng']);jwrite(task_json(config,seed),result)
    else:
        assert existing is None or not existing['metrics'],'Cannot continue without boundary RNG'
        seed_rng(seed)
        result=dict(signature=sig,config=config,seed=seed,physical_gpu=GPU[config,seed],status='RUNNING',
                    metrics=[],diagnostics=[],seconds=0.,started_utc=utc(),resumes=0)
    if saved:result['resumes']+=1
    stop=[False]
    def signal_stop(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    begin=time.monotonic();oldseconds=result['seconds']
    print('TASK_START',name,'GPU',GPU[config,seed],'logical0','resumed',saved is not None,flush=True)
    for fold in range(len(result['metrics'])+1,11):
        if stop[0]:return
        resumed=saved is not None and saved['phase']=='EPOCH' and saved['fold']==fold
        entry=rng();a=make_agent(config,seed,fold);optimizer_audit(a)
        if resumed:
            load_models(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf'])
            restore_rng(saved['rng']);best_state=saved['best_model'];best=saved['best'];best_epoch=saved['best_epoch']
            wait=saved['wait'];start=saved['epoch']+1;initial=saved['initial'];initrng=saved['initial_rng'];tracker=saved['tracker']
        else:
            best_state=None;best=0.;best_epoch=-1;wait=0;start=0
            initial=parameter_hash(a);initrng=rng_hash(entry);tracker={}
        saved=None
        print('FOLD_START',name,fold,'init',initial,flush=True)
        def save_epoch(epoch):
            result['seconds']=oldseconds+time.monotonic()-begin
            checkpoint_write(path,dict(signature=sig,phase='EPOCH',fold=fold,epoch=epoch,model=model_state(a),
                 mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),
                 best_model=best_state,best=best,best_epoch=best_epoch,wait=wait,initial=initial,initial_rng=initrng,tracker=tracker,result=result))
        last=start-1
        for epoch in range(start,450):
            if wait>200:break
            with contextlib.redirect_stdout(io.StringIO()):
                a.run_epoch('simple-2')
                if epoch<2 or (epoch+1)%25==0:
                    for k,v in gradient_norms(a).items():tracker[k]=max(tracker.get(k,0),v)
                with torch.no_grad():_,acc=a.print_trn_acc('simple-2')
            if acc>best:best=float(acc);best_epoch=epoch;best_state=model_state(a);wait=0
            else:wait+=1
            last=epoch
            if (epoch+1)%25==0 or wait>200 or stop[0]:save_epoch(epoch)
            if stop[0]:print('INTERRUPTED_SAFE',name,fold,epoch,flush=True);return
        assert best_state is not None
        continuation=rng();load_models(a,best_state);out=evaluate(a);met=metrics(out,a.tst_idx)
        assert abs(met['acc']-best)<=1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=GPU[config,seed],**met,
             best_epoch=best_epoch,epochs_executed=last+1,selected_val_acc=best,train_n=len(a.trn_idx),test_n=len(a.tst_idx),
             initial_trunk_sha256=initial,initial_rng_sha256=initrng,additional_parameters=71277)
        diag=diagnostics(a,config,fold,tracker)
        result['metrics'].append(row);result['diagnostics'].append(diag)
        restore_rng(continuation);result['seconds']=oldseconds+time.monotonic()-begin
        result['status']='COMPLETE' if fold==10 else 'RUNNING'
        if fold==10:result['completed_utc']=utc()
        checkpoint_write(path,dict(signature=sig,phase='BOUNDARY',fold=fold,rng=continuation,result=result))
        jwrite(task_json(config,seed),result)
        print('FOLD_DONE',name,fold,'ACC',met['acc'],'AUC',met['auc'],'best',best_epoch,flush=True)
        del a,best_state,out
        torch.cuda.empty_cache()
    print('TASK_COMPLETE',name,result['seconds'],flush=True)

def collect():
    groups=[]
    for c,s,_ in tasks():
        r=task_result(c,s)
        if r:groups.append(r)
    rows=[v for r in groups for v in r['metrics']];diag=[v for r in groups for v in r['diagnostics']]
    for values in (rows,diag):assert len({(v['config'],v['seed'],v['fold']) for v in values})==len(values)
    return groups,rows,diag

def refresh():
    m=verify_static()
    # Completed task state is intentionally deleted. Never replace its final
    # manifest counts with an empty scan of the removed temporary task folder.
    if m['status']=='COMPLETE' and m.get('completed_folds')==200:
        rows,diag=load_results()
        assert len(rows)==len(diag)==200
        for name in ('fold_metrics.csv','diagnostics.csv'):
            assert sha(RESULTS/name)==m['result_sha256'][name]
        print('COMPLETE_REFRESH 200 immutable rows verified; no training states needed',flush=True)
        return [],rows,diag
    groups,rows,diag=collect()
    if rows:cwrite(RESULTS/'fold_metrics.csv',rows)
    if diag:cwrite(RESULTS/'diagnostics.csv',diag)
    index={r['config']:r for r in groups}
    for t in m['tasks']:
        r=index.get(t['config'])
        if r:t.update(status=r['status'],completed_folds=len(r['metrics']),seconds=r['seconds'],resumes=r['resumes'])
    m.update(completed_folds=len(rows),completed_tasks=sum(r['status']=='COMPLETE' for r in groups),
             gpu_completed_folds={str(g):sum(r['physical_gpu']==g for r in rows) for g in (0,1)},refreshed_utc=utc())
    jwrite(MANIFEST,m);return groups,rows,diag

def capacity():
    import psutil
    slots=4
    if psutil.virtual_memory().available<32*1024**3 or os.getloadavg()[0]>min(os.cpu_count(),32)*1.2:slots=2
    try:
        p=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
        if min(float(v.strip()) for v in p.splitlines())<10000:slots=1
    except (subprocess.SubprocessError,ValueError):pass
    return slots

def schedule():
    pending=[j for j in tasks() if (task_result(j[0],j[1]) or {}).get('status')!='COMPLETE']
    active=[];retries={};failure=None
    while pending or active:
        slots=capacity()
        for c,s,g in list(pending):
            if sum(v['gpu']==g for v in active)>=slots:continue
            log=RUNTIME/(task_name(c,s)+'.log');handle=log.open('a')
            p=subprocess.Popen([sys.executable,str(EXP/'run.py'),'_worker',c,str(s)],cwd=RUNTIME,
                env=worker_env(g),stdout=handle,stderr=subprocess.STDOUT)
            active.append(dict(process=p,gpu=g,config=c,seed=s,handle=handle,log=log))
            pending.remove((c,s,g));record_event(f'START {c} GPU{g} pid={p.pid} maxworkers={slots}')
        time.sleep(5)
        for job in list(active):
            code=job['process'].poll()
            if code is None:continue
            job['handle'].close();active.remove(job);c=job['config'];record_event(f'EXIT {c} code={code}')
            if code:
                tail=job['log'].read_text()[-10000:]
                # Retry only transient filesystem/process failures; numerical or
                # correctness errors stop all our workers and preserve exact state.
                transient=any(s in tail for s in ('Input/output error','FileNotFoundError','JSONDecodeError'))
                retries[c]=retries.get(c,0)+1
                event=dict(time=utc(),config=c,returncode=code,log_tail=tail,transient=transient)
                m=jread(MANIFEST);m['failures'].append(event);jwrite(MANIFEST,m)
                if transient and retries[c]<=2:pending.append((c,0,job['gpu']))
                else:failure=event;break
            else:
                r=task_result(c)
                if not r or r['status']!='COMPLETE':failure=dict(config=c,error='Worker exited without COMPLETE');break
        refresh()
        if failure:
            for job in active:job['process'].terminate()
            for job in active:
                try:job['process'].wait(timeout=55)
                except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait()
                job['handle'].close()
            m=jread(MANIFEST);m['status']='PARTIAL';jwrite(MANIFEST,m)
            raise RuntimeError('Formal task error; recoverable states retained: '+str(failure))

def load_results():
    if not (RESULTS/'fold_metrics.csv').is_file():return [],[]
    rows=readcsv(RESULTS/'fold_metrics.csv');diag=readcsv(RESULTS/'diagnostics.csv')
    for r in rows:
        for k in ('seed','fold','physical_gpu','asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed'):r[k]=int(r[k])
        for k in METRICS:r[k]=float(r[k])
    return rows,diag

def completed_artifact_audit():
    """Verify completed CSVs after cleanup, without reconstructing any training task."""
    m=verify_static();rows,diag=load_results()
    expected={(c,0,f) for c in CONFIGS for f in range(1,11)}
    assert len(rows)==len(diag)==200
    assert {(r['config'],r['seed'],r['fold']) for r in rows}==expected
    assert {(r['config'],int(r['seed']),int(r['fold'])) for r in diag}==expected
    assert len(m['tasks'])==20 and all(t['status']=='COMPLETE' and t['completed_folds']==10 for t in m['tasks'])
    for name in ('fold_metrics.csv','diagnostics.csv'):
        if name in m.get('result_sha256',{}):assert sha(RESULTS/name)==m['result_sha256'][name]
    for r in rows:
        assert r['physical_gpu']==GPU[r['config'],r['seed']]
        assert all(np.isfinite(r[k]) for k in METRICS)
        tp,fn,fp,tn=(r[k] for k in ('asd_tp','asd_fn','td_fp','td_tn'))
        assert abs(r['acc']-(tp+tn)/(tp+fn+fp+tn))<1e-14
        assert abs(r['auc']-.5*(tp/(tp+fn)+tn/(tn+fp)))<2e-15
        assert abs(r['sen_asd']-tp/(tp+fn+1e-8))<1e-14
        assert abs(r['spe_td']-tn/(tn+fp+1e-8))<1e-14
    for r in diag:
        assert float(r['residual_abs_max'])<=.1+1e-7
        cf=json.loads(r['counterfactual_json'])
        assert len(cf)==len(INTERVENTIONS) and {v['condition'] for v in cf}==set(INTERVENTIONS)
        real=next(v for v in cf if v['condition']=='REAL')
        assert real['probability_abs_change']==real['prediction_flip_fraction']==0
        for v in cf:assert all(np.isfinite(v[k]) for k in METRICS+('probability_abs_change','prediction_flip_fraction'))
    counts={str(g):sum(r['physical_gpu']==g for r in rows) for g in (0,1)}
    assert counts=={'0':100,'1':100}
    changed=[p for p,h in m['protected_files_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    assert not changed,'Protected history changed: '+str(changed)
    hand=(ROOT/'handoff.md').read_bytes()
    assert hashlib.sha256(hand[:m['original_handoff_size']]).hexdigest()==m['original_handoff_sha256']
    marker=b'<!-- EXPERIMENT29_CONNECTIVITY_SEMANTIC_SELECTIVE_AUXILIARY_FINAL -->'
    if m.get('handoff_appended'):assert hand.count(marker)==1
    return dict(verified_utc=utc(),unique_training_rows=200,unique_diagnostic_rows=200,
        counterfactual_conditions=200*len(INTERVENTIONS),gpu_folds=counts,
        protected_files_unchanged=len(m['protected_files_sha256']),handoff_prefix_unchanged=True)

def review_lines(rows,diag,table,pairs):
    """Post-training interpretation only; no changes to fitting or metric definitions."""
    lookup={(p['left'],p['right']):p['metrics'] for p in pairs}
    result=['','## 结果复核与机制判读','',
      '最高ACC/AUC均来自关闭语义的C3/F3（两者数学相同、独立执行结果逐值一致）。含真实语义组的最高ACC为C6和F6，仍低于P0；不根据这些结果重选主方法或追加实验。',
      'C2/F2相对P0及关闭语义对照均下降。C2不及数值语义码C8；F2对错误ROI/方向及数值码没有四指标一致优势，尤其ASD灵敏度不占优，因此医学文本必要性不成立。',
      '软稀疏相对Dense改善了ACC/AUC，但F2的SEN下降、漏诊增加；C2不优于同站点错配辅助，F2也在SEN上落后错配。不能把门控变小解释为已经识别医学互补信息。','',
      '### 所有新组相对P0/E1的四指标差值（pp）','',
      '|组|参照|ΔACC|ΔAUC|ΔSEN|ΔSPE|','|---|---|---:|---:|---:|---:|']
    for c in CONFIGS:
        if c not in table:continue
        for ref in ('P0','E1'):
            result.append('|'+c+'|'+ref+'|'+'|'.join(f'{100*(table[c][k]["mean"]-table[ref][k]["mean"]):+.5f}' for k in METRICS)+'|')
    result+=['','### 主方法机制数值','']
    for c in ('C2','F2'):
        rr=[r for r in diag if r['config']==c]
        if len(rr)!=10:continue
        avg=lambda k:float(np.mean([float(r[k]) for r in rr]))
        result.append(c+'：语义残差L2均值'+f'{avg("semantic_residual_l2_mean"):.6f}'+
            '，辅助残差L2均值'+f'{avg("auxiliary_residual_l2_mean"):.6f}'+
            '，非零响应比例'+f'{avg("residual_nonzero_fraction"):.6f}'+
            '；局部损失Query/Anchor梯度'+f'{avg("local_loss_anchor_query_gradient"):.7g}'+
            '。语义残差接近允许的L2上界0.6，提示有界输出的饱和风险；不是分支整体未接入。')
        result.append('PHENO/ANAT/FUNC的特征门均值分别为'+','.join(f'{avg(n+"_gate_mean"):.6f}' for n in ('PHENO','ANAT','FUNC'))+
            '，Router均值为'+','.join(f'{avg(n+"_router_mean"):.6f}' for n in ('PHENO','ANAT','FUNC'))+'；这些权重不能当作临床可靠度。')
        reals=[next(v for v in json.loads(r['counterfactual_json']) if v['condition']=='REAL') for r in rr]
        offs=[next(v for v in json.loads(r['counterfactual_json']) if v['condition']=='SEMANTIC_OFF') for r in rr]
        off_delta=100*np.mean([o['acc']-v['acc'] for o,v in zip(offs,reals)])
        retrained=lookup[c,c[0]+'3']['acc']['delta_pp']
        result.append('同Checkpoint关闭语义后ACC变化'+f'{off_delta:+.5f}pp'+
            '，但原模型相对从头训练的关闭语义组变化'+f'{retrained:+.5f}pp'+
            '。这清楚区分训练后依赖与独立性能增益，不能将前者包装成医学语义有效。')
    result+=['',
      '粗粒度局部Key逐值重复，所有患者都有16条有效边，因此其理想局部响应是同一个异常原型。浮点累加带来的极小差异会使按相对特征值计算的有效秩不为零；C系列该秩不能解释成新增语义维度，微小局部梯度也不是连接身份辨识证据。',
      'F2的GLOBAL_CLASS_SWAP仅交换全局原型顺序；softmax(h·Pᵀ)P对共同置换不变，故该反事实的零变化是算子的数学性质。全局CE的标签顺序仍在训练中起作用。C系列的局部重复原型同时取全局ASD原型，故同一干预还有局部Key改变，边界不同。',
      '固定Checkpoint文本替换/OFF只说明训练后计算依赖。semantic_decode还接收raw-CORR Anchor，因此关闭整个分支同时移除额外FC容量；不能把BRANCH依赖全归因文本内容。必须以完整训练的C8/F8/F9/F10判断必要性。',
      'C8/F10最终单位化前均值/协方差匹配，最终原型夹角与真实文本不同（真实全局cos约0.846、数值约0.216），因此不是完全相同几何的医学文案实验；本轮没有证明真实医学语义优越，也不声称已隔离纯语言含义。',
      '结论：NO-GO。保留原P0，不将C3/F3相对P0约0.115pp的单seed小幅领先包装为医学语义创新。后续优先真实互补测量、训练内选模或独立队列验证。']
    return result

def paired_results(rows):
    index={(r['config'],r['fold']):r for r in rows+references()};result=[]
    for a,b in CONTRASTS:
        if not all((c,f) in index for c in (a,b) for f in range(1,11)):continue
        effects={}
        for k in METRICS:
            delta=np.array([float(index[a,f][k])-float(index[b,f][k]) for f in range(1,11)])
            effects[k]=dict(delta_pp=float(delta.mean()*100),wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()),per_fold_delta_pp=(delta*100).tolist())
        result.append(dict(left=a,right=b,metrics=effects))
    return result

def report():
    m=verify_static();rows,diag=load_results();refs=references();pairs=paired_results(rows)
    table={}
    for c in ('P0','E1')+CONFIGS:
        rr=[r for r in rows+refs if r['config']==c]
        if len(rr)==10:table[c]={k:dict(mean=float(np.mean([float(r[k]) for r in rr])),sd_population=float(np.std([float(r[k]) for r in rr],ddof=0))) for k in METRICS}
    lookup={(p['left'],p['right']):p['metrics'] for p in pairs}
    def clear_win(a,b):
        return (a,b) in lookup and all(lookup[a,b][k]['delta_pp']>0 for k in ('acc','auc','sen_asd'))
    gates={c:dict(baseline=clear_win(c,'P0'),semantic_off=clear_win(c,c[0]+'3'),numeric=clear_win(c,'C8' if c=='C2' else 'F10'),
             correct_patient=clear_win(c,c[0]+'4'),dense=clear_win(c,c[0]+'1')) for c in ('C2','F2')}
    if 'F2' in gates:gates['F2'].update(roi=clear_win('F2','F8'),direction=clear_win('F2','F9'))
    # A strict qualitative screening rule declared before formal training; a
    # single-seed descriptive screen is never claimed a clinical significance test.
    scientific=('GO_CANDIDATE_SINGLE_SEED_ONLY' if any(all(v.values()) for v in gates.values()) else 'NO-GO') if len(rows)==200 else 'NOT_EVALUATED_INCOMPLETE'
    counts={c:dict(asd_fn=sum(int(r['asd_fn']) for r in rows+refs if r['config']==c),td_fp=sum(int(r['td_fp']) for r in rows+refs if r['config']==c)) for c in table}
    lines=['# 实验29：CORR主导的双层疾病语义对齐与选择性辅助修正','',
      f'工程状态：**{m["status"]}**；正式新增 **{len(rows)}/200折**，seed仅0；当前科研判定：**{scientific}**。','',
      '## 代码与论文审计','',
      '本地HEAD `'+m['source_local_head']+'`；远端main `'+m['source_remote_head']+'`。六核心源码逐字节相同，原始文件只读。',
      'GFR在四个受试者相似图上传播全部320维；i==1覆盖i==0的combined_feat，i==2/3继续累加。simple-2保留四图GraphConstructLoss平均和两Adam，ALoss仅计算/打印。',
      'CAM经过InputLayer、Encoder和FeedForward得到[871,4,36]，再分解为[871,4,4,9]潜在Token；这些位置不等于ROI。原SPGS将两类相似度Softmax后取max，失去类别方向；RES池化global_x并按softmax(part*global_x)逐维融合，输出分类器保持原样。',
      '输入PHENO48=年龄26/站点20/性别2，ANAT6/FUNC10是扫描质量指标，CORR256是RFE保留Fisher-z连接。新辅助只读取年龄/性别28和QC16，不移除原主干任何模态或站点。',
      '实验11通过数值重建恢复256连接身份，后续已核验HO名称/列序的252条用于文本；四条未知连接195/225/232/236只留原CORR。历史早期report里的UNKNOWN已由最终两张CSV验证列取代。',
      '实验28把高度同向文本先加权平均，再以SmoothL1监督FC+CAM anchor；两类教师中心cos约0.999988，教师共同方向和无配对CLIP域差导致类别区分弱。辅助是qa*qf*人口统计修正，关闭辅助的平均概率变化约0.00131。本轮改变为raw-CORR类别CE、显式连接Query–固定Key和推理语义响应，不将历史阴性解释为工程未接入。',
      'SPromptGL已有GFR/CAM/SPGS/RES，MMGPL已有概念相似度加权及概念响应Token图；本轮不宣称门控、概念加权、对比监督或语义图首创，也未新增患者图。参见[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)和[MMGPL §III-C/D](https://arxiv.org/html/2312.14574v2)。','',
      '## 冻结方法与公平控制','',
      '所有512个历史64维单位文本减同一公共均值后L2归一化。全局原型顺序[异常ASD,正常TD]；h_fc=GELU(LayerNorm(Linear(CORR256,64)))，训练全局CE温度0.2、系数0.05；不把CAM或辅助值作为语义监督Anchor。',
      'TD均值/std与ASD−TD方向复用实验28精确站点收缩公式；稳定性=站点方向一致率/(1+异质性)。异常证据ReLU(sign(d)*r−0.5)*clip(|d|,0,2)*稳定性；仅从252可信边稳定降序取最多16条，患者选择不读其标签。无候选时local response/loss为0，保留全局响应。',
      'Query输入h64、r/d/稳定性3维及连接ID16维，共83→64/LN/GELU/64；Key/Value固定且来自该真实连接的增强/减弱文本。C系列重复异常原型；F8只错配Key身份、F9只交换方向。局部候选内InfoNCE温度0.2/系数0.02；C重复原型的损失为log(有效候选数)，没有有效身份辨识梯度。',
      '逐连接匹配余弦/0.2加tanh(数值证据)为注意力logits；聚合后与h_fc、全局两类softmax加权文本响应拼接192→64/GELU/36。没有自由可学习文本Key投影。连接ID+Query本身能学习类别/身份代码，正确医学知识仍需负对照证明。',
      '每辅助模态独立Selector([h,x])→原输入维sigmoid、无偏置输入Linear→64/GELU、Router([h,u,h*u])→1sigmoid及无偏置Value64→64。三贡献拼接192→36无偏置。全零辅助产生严格零Value；FC只能条件化门控，不能通过Value bias单独产生辅助。',
      'Dense与Selective模块结构和初始参数完全相同；后者只有0.005×训练折活跃模态特征门/Router均值稀疏惩罚。为soft sparse gating，没有硬阈值，不声称精确特征删除；门控接近0比例另报。',
      'delta=0.10*tanh(tanh(alpha_sem)*semantic_candidate+tanh(alpha_aux)*aux_candidate)，两个alpha=0。仅CAM CORR位置追加，坐标≤0.10，L2≤0.6；这不是相对原特征10%限制。新参数71277均注册，关闭分支的有效参数/梯度另报。',
      'C8/F10用seed29008生成匹配512×64的冻结数值码；最终L2前均值/协方差严格相同，最终单位范数相同；最终协方差、逐对Gram及两类原型几何不完全匹配，不能把全部差异只归因医学含义。C3/F3语义关闭后数学相同，仍按预注册独立运行。','',
      '## 冻结20组','', '|组|粗/细|语义|辅助|软稀疏|患者错配|文本控制|','|---|---|---|---|---|---|---|']
    for c in CONFIGS:
        d=DESIGNS[c];lines.append(f'|{c}|'+('细' if d.fine else '粗')+f'|{d.semantic}|{",".join(d.auxiliary) or "无"}|{d.sparse}|{d.shuffled}|{d.text}|')
    lines+=['','## 正式四指标（seed0十折均值±population SD，%）','',
      'P0/E1为只读历史seed0完整十折；新200折不含历史参考。AUC按原代码硬标签one-hot计算，不是概率排序AUC。','',
      '|组|来源|ACC|AUC（硬标签）|SEN-ASD|SPE-TD|ASD FN|TD FP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for c,t in table.items():
        lines.append('|'+c+'|'+('HISTORICAL_READ_ONLY' if c in ('P0','E1') else 'NEW_SEED0_10_FOLDS')+'|'+'|'.join(f'{t[k]["mean"]*100:.5f}±{t[k]["sd_population"]*100:.5f}' for k in METRICS)+f'|{counts[c]["asd_fn"]}|{counts[c]["td_fp"]}|')
    if len(rows)==200:
        for prefix in ('C','F'):
            eligible=[c for c in CONFIGS if c.startswith(prefix)]
            lines.append(prefix+'系列各指标最高组：'+', '.join(k+'='+max(eligible,key=lambda c:table[c][k]['mean']) for k in METRICS)+'。排名不改变预设主方法C2/F2。')
    lines+=['','## 全部预设配对（pp）及逐折胜/平/负','', '|比较|指标|平均差pp|胜/平/负|fold1–10差pp|','|---|---|---:|---|---|']
    for pair in pairs:
        for k,item in pair['metrics'].items():
            lines.append(f'|{pair["left"]}−{pair["right"]}|{k}|{item["delta_pp"]:+.5f}|{item["wins"]}/{item["ties"]}/{item["losses"]}|'+','.join(f'{v:+.4f}' for v in item['per_fold_delta_pp'])+'|')
    lines+=['','## 机制诊断','',
      '文本几何：`'+json.dumps({k:v for k,v in m['text_bank'].items() if 'geometry' in k or 'cos' in k or 'error' in k or 'distance' in k},ensure_ascii=False)+'`。','',
      '|组|有效连接|注意力熵|局部响应秩|全局ASD−TD训练间隔|语义残差L2|辅助残差L2|总残差L2|非零比例|alpha语义/辅助|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def mean(c,key):
        rr=[r for r in diag if r['config']==c]
        return float(np.mean([float(r[key]) for r in rr])) if rr else 0.
    for c in CONFIGS:
        if c not in table:continue
        fields=('valid_connections_mean','attention_entropy_mean','local_response_effective_rank','train_class_margin_separation',
                'semantic_residual_l2_mean','auxiliary_residual_l2_mean','residual_l2_mean','residual_nonzero_fraction')
        lines.append('|'+c+'|'+'|'.join(f'{mean(c,k):.6f}' for k in fields)+f'|{mean(c,"alpha_sem"):.5f}/{mean(c,"alpha_aux"):.5f}|')
    lines+=['','### 门控、有效维数与梯度','', '|组|模态|特征门均值|<0.1比例|<0.01比例|有效维数(>0.1)|Router均值|站点η²|最大QC相关|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in ('C1','C2','C4','F1','F2','F4'):
        if c not in table:continue
        for name in ('PHENO','ANAT','FUNC'):
            keys=('gate_mean','gate_below_01_fraction','gate_below_001_fraction','effective_dimensions','router_mean','gate_site_eta_squared','gate_QC_max_abs_correlation')
            lines.append('|'+c+'|'+name+'|'+'|'.join(f'{mean(c,name+"_"+k):.6f}' for k in keys)+'|')
    lines+=['','|组|全局损失|局部损失|全局梯度|局部梯度|分类梯度|数值文本语义梯度|正确/错配ROI cosine|','|---|---:|---:|---:|---:|---:|---:|---|']
    for c in CONFIGS:
        if c not in table:continue
        keys=('global_loss_value','local_loss_value','global_loss_anchor_query_gradient','local_loss_anchor_query_gradient','classification_anchor_query_gradient','numeric_text_semantic_gradient')
        lines.append('|'+c+'|'+'|'.join(f'{mean(c,k):.7g}' for k in keys)+f'|{mean(c,"matched_local_cosine"):.6f}/{mean(c,"wrong_ROI_local_cosine"):.6f}|')
    lines+=['','### 同Checkpoint反事实（仅诊断，非重训证据）','', '|组|干预|ASD概率平均绝对变化|翻转率|ΔACC pp|ΔAUC pp|','|---|---|---:|---:|---:|---:|']
    for c in ('C2','F2'):
        rr=[json.loads(r['counterfactual_json']) for r in diag if r['config']==c]
        if not rr:continue
        for mode in INTERVENTIONS:
            cc=[next(t for t in fold if t['condition']==mode) for fold in rr]
            reals=[next(t for t in fold if t['condition']=='REAL') for fold in rr]
            lines.append('|'+c+'|'+mode+'|'+f'{np.mean([r["probability_abs_change"] for r in cc]):.8f}|{np.mean([r["prediction_flip_fraction"] for r in cc]):.8f}|'+
                         f'{100*np.mean([r["acc"]-real["acc"] for r,real in zip(cc,reals)]):+.5f}|{100*np.mean([r["auc"]-real["auc"] for r,real in zip(cc,reals)]):+.5f}|')
    lines+=['','## 科研判定与限制','',
       '冻结描述性筛选要求ACC/AUC/SEN均优于比较组。主组各项门槛：`'+json.dumps(gates,ensure_ascii=False)+'`；总体：**'+scientific+'**。此规则只用于保守判读，不用于追加组别或超参。',
       '医学文本必须优于数值码、错误ROI和方向；正确辅助必须优于同站点错配；选择性必须胜过Dense。仅最高ACC、有效梯度或同Checkpoint文本敏感不足以验证医学知识。若某维度有效而必要性控制失败，只保留对应工程或监督学习线索。',
       '仅seed0，不能说跨seed稳定。相同fold共享患者，十折不是十个独立研究；不同组早停引起后续折RNG分叉，同seed/fold不等于完全相同初始化。',
       '保留test=val选模、外层之前监督RFE/标准化、871人transductive和此前28轮探索，属于偏乐观内部历史可比指标，不称无泄漏泛化、医学因果协同或临床获益。方向和稳定性是训练标签衍生的弱监督数值证据，不是个体病灶真值。',
       'AUC是硬标签AUC，对二分类等于两类召回平均，不是独立概率排序证据。ANAT/FUNC不能当丰富独立影像流；辅助门控与站点/质量相关不是医学可靠度。',
       '后续建议：若必要性控制不能成立，保留P0强基线，停止在本队列静态文本上追分；优先真实互补测量与训练内选模/独立队列验证。若满足门槛，也仅为单seed候选，另行验证，不在本轮追加训练。','',
       '## 运行与完整性','',
       '预检：`'+json.dumps(m.get('preflight',{}),ensure_ascii=False)+'`。',
       '状态/任务：`'+json.dumps(m['tasks'],ensure_ascii=False)+'`；GPU折数：`'+json.dumps(m.get('gpu_completed_folds',{}))+'`；正式墙钟秒：`'+str(m.get('formal_seconds'))+'`。',
       '每25epoch、早停及fold边界保存原GFR/PLN/新增buffers、两个Adam、Python/NumPy/CPU/CUDA RNG、最佳模型/epoch/wait及已提交行；校验签名与载荷SHA后恢复，不混合失败attempt。',
       '源代码SHA：`'+json.dumps(m['source_sha256'])+'`；数据SHA：`'+json.dumps(m['data_sha256'])+'`；文本缓存SHA：`'+m['prompt_cache_sha256']+'`；实验代码SHA：`'+json.dumps(m['code_sha256'])+'`。',
       '历史保护：`'+json.dumps(m.get('protection',{}))+'`；handoff：`'+str(m.get('handoff_status','PENDING'))+'`。无自动commit/push。','',
       '## seed0完整逐折四指标','', '|组|fold|ACC|AUC|SEN|SPE|ASD FN|TD FP|best epoch(0-based)|实际epoch|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|'+r['config']+'|'+str(r['fold'])+'|'+'|'.join(f'{r[k]*100:.6f}' for k in METRICS)+f'|{r["asd_fn"]}|{r["td_fp"]}|{r["best_epoch"]}|{r["epochs_executed"]}|')
    if len(rows)==200:lines+=review_lines(rows,diag,table,pairs)
    tmp=RESULTS/'report.md.tmp'
    with tmp.open('w') as f:f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,RESULTS/'report.md')
    return dict(table=table,pairs=pairs,screening=gates,scientific=scientific,case_counts=counts)

def append_handoff(summary,status):
    m=jread(MANIFEST);target=ROOT/'handoff.md';marker='<!-- EXPERIMENT29_CONNECTIVITY_SEMANTIC_SELECTIVE_AUXILIARY_FINAL -->'
    text='\n\n'+marker+'\n## 实验二十九：CORR主导的双层疾病语义对齐与选择性辅助修正\n\n'
    text+='更新时间：'+utc()+'。STATUS='+status+'；'+summary['scientific']+'。\n\n'
    text+='本轮问题：真实疾病语义的类别对齐与真实FC连接身份匹配是否增加诊断信息；CORR条件化软稀疏辅助是否优于Dense及同站点错误患者配对。'+\
          '原SPromptGL已有GFR/CAM/SPGS/RES，MMGPL已有概念加权/Token图；实验28平均教师高度同向且文本不直接进入推理，本轮保留主干但以原始CORR64维Anchor、全局CE与局部InfoNCE及推理语义响应替代。\n\n'
    text+='精确方法：'+json.dumps(RULES,ensure_ascii=False)+'。\n\n20组冻结设计：'+json.dumps(m['design'],ensure_ascii=False)+'。\n\n'
    text+='正式完成'+str(m['completed_tasks'])+'/20任务、'+str(m['completed_folds'])+'/200fold；seed仅0，每任务连续十折；两GPU完成fold数'+json.dumps(m.get('gpu_completed_folds',{}))+'；正式墙钟秒'+str(m.get('formal_seconds'))+'。历史P0/E1只读，不重复训练。\n\n'
    text+='全部四指标（每组seed0十折均值±population SD，比例值）：'+json.dumps(summary['table'],ensure_ascii=False)+'。\n\n'
    text+='全部预设配对（pp及逐折胜/平/负/差）：'+json.dumps(summary['pairs'],ensure_ascii=False)+'。\n\n'
    text+='科研筛选：'+json.dumps(summary['screening'],ensure_ascii=False)+'。'+summary['scientific']+'。仅一个seed，不声称跨seed稳定；不能用最高ACC或有效梯度替代真实医学文本、正确患者配对及Sparse必要性。\n\n'
    text+='详细机制、逐折四指标和反事实见'+str(RESULTS/'report.md')+'；诊断200条目标，包含文本几何/Query匹配、损失梯度、门控及站点关系、各模态关闭及数值/错ROI/反向文本。'+\
          '全部20组新增注册参数71277；C3/F3关闭语义后数学相同仍独立运行。局部Token非ROI，ANAT/FUNC为QC；不把效应方向当病灶真值。\n\n'
    text+='代码/数据SHA：'+json.dumps(dict(core=m['source_sha256'],data=m['data_sha256'],code=m['code_sha256'],cache=m['prompt_cache_sha256']))+'。历史保护：'+json.dumps(m.get('protection',{}))+'。预检：'+json.dumps(m.get('preflight',{}),ensure_ascii=False)+'。\n\n'
    text+='历史协议test=val、外层前RFE/标准化、871全图transductive及跨轮探索保留；指标偏乐观，非独立无泄漏泛化。后续唯一建议为真实互补测量及训练内选模/独立队列验证；本轮不再追分或追加超参。无自动Git提交/推送。\n'
    pending=Path('/tmp/experiment29_handoff_pending.md')
    try:
        with (ROOT/'handoff.md.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for attempt in range(4):
                try:
                    with target.open('a+b',buffering=0) as f:
                        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);old=f.read()
                        assert hashlib.sha256(old[:m['original_handoff_size']]).hexdigest()==m['original_handoff_sha256']
                        if marker.encode() not in old:
                            assert f.write(text.encode())==len(text.encode());os.fsync(f.fileno())
                        new=target.read_bytes();assert new[:len(old)]==old and new.count(marker.encode())==1
                    m.update(handoff_appended=True,handoff_status='APPENDED_PREFIX_VERIFIED',handoff_updated_utc=utc(),handoff_sha256=sha(target));break
                except OSError:
                    if attempt==3:raise
                    time.sleep(.25*(attempt+1))
        if pending.exists():pending.unlink()
    except OSError as e:
        pending.write_text(text);m.update(handoff_appended=False,handoff_status='PENDING_ROOT_IO',handoff_error=str(e),handoff_pending=str(pending))
    jwrite(MANIFEST,m)

def finish():
    groups,rows,diag=refresh();assert len(groups)==20 and len(rows)==len(diag)==200
    assert all(g['status']=='COMPLETE' and len(g['metrics'])==10 for g in groups)
    assert [sum(r['physical_gpu']==g for r in rows) for g in (0,1)]==[100,100]
    assert all(all(np.isfinite(r[k]) for k in METRICS) for r in rows)
    m=verify_static()
    changed=[p for p,h in m['protected_files_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    assert not changed,'Historical/core files changed: '+str(changed)
    m.update(status='COMPLETE',completed_utc=utc(),protection=dict(status='UNCHANGED_SHA256',files=len(m['protected_files_sha256'])))
    m['formal_seconds']=(dt.datetime.fromisoformat(m['completed_utc'])-dt.datetime.fromisoformat(m['formal_started_utc'])).total_seconds()
    jwrite(MANIFEST,m);summary=report();append_handoff(summary,'COMPLETE');summary=report()
    m=jread(MANIFEST);m.update(scientific_conclusion=summary['scientific'],screening=summary['screening'],
        result_sha256={p.name:sha(p) for p in RESULTS.glob('*') if p.is_file()})
    jwrite(MANIFEST,m)
    print('STATUS=COMPLETE 20 tasks / 200 folds / GPU0=100 GPU1=100',flush=True)

def finalize_existing_results():
    audit=completed_artifact_audit();m=jread(MANIFEST)
    m.update(status='COMPLETE',post_training_verification=audit,finalized_utc=utc())
    for f in m['failures']:
        if f.get('error','').startswith('[Errno 39] Directory not empty:'):
            f.update(resolved_utc=utc(),resolution='Release controller lock before bounded cleanup; all 200 CSV rows retained without retraining.')
    jwrite(MANIFEST,m);summary=report()
    if not m.get('handoff_appended'):append_handoff(summary,'COMPLETE')
    m=jread(MANIFEST);m.update(scientific_conclusion=summary['scientific'],screening=summary['screening'],
        result_sha256={p.name:sha(p) for p in RESULTS.glob('*') if p.is_file()})
    jwrite(MANIFEST,m)
    print('STATUS=COMPLETE completed 200-fold artifacts verified; NO RETRAINING',flush=True)

def cleanup_runtime():
    """Call only AFTER releasing controller FD, including on NFS mounts."""
    assert RUNTIME==EXP/'.runtime'
    for attempt in range(5):
        try:
            if RUNTIME.exists():shutil.rmtree(RUNTIME)
            break
        except OSError:
            if attempt==4:raise
            time.sleep(.2*(attempt+1))
    assert not RUNTIME.exists()
    # jwrite creates a runtime lock; use a final atomic write without that lock.
    m=json.loads(MANIFEST.read_text());m['cleanup_status']='REMOVED';m['cleanup_verified_utc']=utc()
    tmp=MANIFEST.with_name(MANIFEST.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(m,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,MANIFEST)

def all_tasks():
    prepare();m=verify_static()
    if m.get('completed_folds')==200 and m.get('completed_tasks')==20:
        finalize_existing_results();return
    if m['status']=='COMPLETE':
        for name,h in m['result_sha256'].items():assert sha(RESULTS/name)==h
        print('ALREADY_COMPLETE verified; no retraining');return
    preflight();m=jread(MANIFEST);m['status']='RUNNING';m.setdefault('formal_started_utc',utc());jwrite(MANIFEST,m)
    schedule();finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','preflight','all','resume','refresh','report','finalize','_smoke','_worker'))
    p.add_argument('config',nargs='?');p.add_argument('seed',nargs='?',type=int);a=p.parse_args()
    if a.command=='_smoke':smoke();return
    if a.command=='_worker':worker(a.config,a.seed);return
    controller=None
    if a.command in ('all','resume','preflight','finalize'):
        RUNTIME.mkdir(parents=True,exist_ok=True);controller=(RUNTIME/'controller.lock').open('a')
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.command=='prepare':prepare()
        elif a.command=='preflight':preflight()
        elif a.command=='finalize':finalize_existing_results()
        elif a.command in ('all','resume'):all_tasks()
        elif a.command=='refresh':refresh()
        elif a.command=='report':
            report();m=jread(MANIFEST)
            if m['status']=='COMPLETE':m['result_sha256']={p.name:sha(p) for p in RESULTS.glob('*') if p.is_file()};jwrite(MANIFEST,m)
            print((RESULTS/'report.md').read_text())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if MANIFEST.exists():
            m=jread(MANIFEST);m['status']='PARTIAL' if m.get('formal_started_utc') else 'BLOCKED'
            m['failures'].append(dict(time=utc(),stage=a.command,error=str(exc)));jwrite(MANIFEST,m)
            # Keep the unique final handoff marker available while a concrete
            # engineering error is repaired. Append only when the run actually
            # ends (finish(), or an explicitly recorded terminal blockage).
            report()
        raise
    finally:
        if controller is not None:controller.close()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())['status']=='COMPLETE':cleanup_runtime()

if __name__=='__main__':main()


#!/usr/bin/env python3
"""Experiment28 controller. prepare -> preflight -> P0 parity30 -> other270.

All original files are read-only. Each config x seed is one isolated process
with one continuous ten-fold RNG stream. No test-score driven design changes.
"""
from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
import argparse,contextlib,csv,datetime as dt,fcntl,hashlib,io,json,pickle
from pathlib import Path
import random,shutil,signal,subprocess,time,types
from types import SimpleNamespace
import numpy as np
from e28_method import CONFIGS,SEEDS,CONTRASTS,INTERVENTIONS,RULES,tasks
import e28_prompts
import e28_integration

ROOT=Path('/data/adodas/SPromptGL-main')
EXP=Path(__file__).resolve().parent
RUNTIME=EXP/'.runtime'
SOURCE=RUNTIME/'sources'
RESULTS=EXP/'results'
MANIFEST=EXP/'manifest.json'
CACHE=EXP/'prompt_banks.npz'
CORE=('main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh')
CODE=('e28_method.py','e28_prompts.py','e28_integration.py','run.py','integration.patch')
BASELINE=ROOT/'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv'
E9REF=ROOT/'experiment/experiment9_prompt_disease_adversarial_interaction/results/fold_metrics.csv'
METRICS=('acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td','f1_asd')
GPU={(c,s):g for c,s,g in tasks()}
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
              nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,
              n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,
              GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,
              theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return e28_prompts.sha(p)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
@contextlib.contextmanager
def json_lock(exclusive):
    # Stable lock inode, independent of replaced JSON directory entries.
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'json.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def jread(p):
    # This mounted filesystem has exposed both temporarily missing and short
    # reads after rename. Bounded retries never accept persistently bad JSON.
    for attempt in range(8):
        try:
            with json_lock(False):return json.loads(Path(p).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt==7:raise
            time.sleep(.10)
def jwrite(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with json_lock(True):
        with tmp.open('w') as f:
            json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def cwrite(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def record_event(message):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'scheduler.log').open('a') as f:f.write(utc()+' '+str(message)+'\n');f.flush()
    print(message,flush=True)
def core_hashes():return {n:sha(ROOT/n) for n in CORE}
def code_hashes():return {n:sha(EXP/n) for n in CODE}
def signature():
    m=jread(MANIFEST)
    payload={k:m[k] for k in ('source_sha256','data_sha256','code_sha256','prompt_cache_sha256','design','rules','protocol')}
    # An audited I/O-only repair must not invalidate completed numerical states.
    # Current executable hashes are independently checked by verify_static().
    payload['code_sha256']=m.get('signature_code_sha256',m['code_sha256'])
    return canonical(payload)
def verify_static():
    m=jread(MANIFEST)
    assert core_hashes()==m['source_sha256'],'Root core modified'
    assert code_hashes()==m['code_sha256'],'Experimental code changed after freeze'
    assert sha(CACHE)==m['prompt_cache_sha256']
    assert sha(EXP/'prompt_banks.json')==m['prompt_json_sha256']
    for p,h in m['data_sha256'].items():assert sha(p)==h,p
    assert sha(BASELINE)==m['reference_p0_sha256']
    return m

def protected_snapshot():
    paths=set()
    previous=jread(ROOT/'experiment/experiment27_evidence_grounded_relational_prompt_cam/manifest.json')
    for p in previous['protected_files_sha256']:
        q=Path(p)
        if q.is_file() and q!=ROOT/'prompt' and q!=ROOT/'handoff.md':paths.add(q)
    for p in (ROOT/'experiment').glob('experiment*'):
        if p==EXP:continue
        paths.update(q for q in p.rglob('*') if q.is_file())
    for n in (20,23,26):
        for p in Path('/tmp').glob(f'experiment{n}_*'):
            if p.is_dir():paths.update(q for q in p.rglob('*') if q.is_file() and '.git' not in q.parts)
    paths.update(ROOT/n for n in CORE)
    paths.update(q for q in (ROOT/'CLIP').rglob('*') if q.is_file())
    paths.update(q for q in (ROOT/'data/ABIDE').glob('*') if q.is_file())
    return {str(p):sha(p) for p in sorted(paths)}

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

def historical_p0():
    rows=[r for r in readcsv(BASELINE) if r['config']=='F0']
    assert len(rows)==30 and len({(int(r['seed']),int(r['fold'])) for r in rows})==30
    result={}
    for r in rows:
        item={k:float(r[k]) for k in METRICS if k!='f1_asd'}
        tp,fn,fp,tn=(int(r[k]) for k in ('asd_tp','asd_fn','td_fp','td_tn'))
        item.update(f1_asd=2*tp/max(2*tp+fn+fp,1),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn,
                    best_epoch=int(r['best_epoch']),epochs_executed=int(r['epochs_executed']),
                    initial_trunk_sha256=r['initial_trunk_sha256'])
        result[int(r['seed']),int(r['fold'])]=item
    assert abs(np.mean([r['acc'] for r in result.values()])*100-91.69105)<.00001
    return result

def prepare():
    EXP.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    if MANIFEST.exists():
        verify_static();record_event('PREPARE_REUSE: frozen hashes verified');return
    started=utc();begin=time.monotonic()
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    import urllib.request
    hashes=core_hashes()
    for n,h in hashes.items():
        b=urllib.request.urlopen(f'https://raw.githubusercontent.com/tangaii/SPromptGL/{remote}/{n}',timeout=60).read()
        assert hashlib.sha256(b).hexdigest()==h,'GitHub/root core mismatch: '+n
    protection=protected_snapshot();info=data();historical_p0()
    hand=ROOT/'handoff.md';handbytes=hand.read_bytes() if hand.exists() else b''
    reports={}
    # Read complete history and relevant complete reports, not large epoch logs.
    _=handbytes.decode()
    for n in (3,8,9,10,11,13,14,15,16,17,18,19,20,22,23,24,25,26,27):
        candidates=list((ROOT/'experiment').glob(f'experiment{n}_*/results/report.md'))
        if not candidates:candidates=list(Path('/tmp').glob(f'experiment{n}_*/results/report.md'))
        assert candidates,'Missing historical report '+str(n)
        for p in candidates:reports[str(p)]=dict(sha256=sha(p),characters=len(p.read_text()))
    e28_integration.emit(ROOT,SOURCE,EXP/'integration.patch')
    if CACHE.exists() or (EXP/'prompt_banks.json').exists():
        assert CACHE.exists() and (EXP/'prompt_banks.json').exists()
        textinfo=jread(EXP/'prompt_banks.json')
    else:textinfo=e28_prompts.build(ROOT,EXP)
    design={c:dict(index=i,spgs=(i==0),new_module=(i>=2),auxiliary=(i>=4),
                   semantic_loss=(i>=2 and i!=8),purpose=(
                       'Original P0','No SPGS','Original coarse FC teacher','Fine FC teacher',
                       'Fine full conditional fusion','Frozen numeric codes','Wrong ROI identity',
                       'Wrong FC direction','No semantic training loss','Wrong within-site auxiliary patient')[i])
            for i,c in enumerate(CONFIGS)}
    m=dict(experiment='28 IA-FDSA',status='PREPARED' if textinfo['informative'] else 'STOPPED_TEXT_COLLAPSE',
           started_utc=started,prepared_utc=utc(),prepare_seconds=time.monotonic()-begin,
           source_local_head=local,source_remote_head=remote,source_remote_core_identical=True,
           source_sha256=hashes,data_sha256={str(ROOT/p):sha(ROOT/p) for p in
                ('data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy',
                 'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv',
                 'experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv')},
           protected_files_sha256=protection,protected_file_count=len(protection),
           original_handoff_size=len(handbytes),original_handoff_sha256=hashlib.sha256(handbytes).hexdigest(),
           original_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
           prompt_source_sha256=sha(ROOT/'prompt'),prompt_cache_sha256=sha(CACHE),
           prompt_json_sha256=sha(EXP/'prompt_banks.json'),clip_weights_sha256=textinfo['clip_weights_sha256'],
           reference_p0_sha256=sha(BASELINE),reference_e9_sha256=sha(E9REF),
           design=design,rules=RULES,protocol=PROTOCOL,contrasts=list(CONTRASTS),
           tasks=[dict(config=c,seed=s,gpu=g,folds=10,status='PENDING') for c,s,g in tasks()],
           target_folds=300,completed_folds=0,gpu_fold_targets={'0':150,'1':150},
           history_audit=reports,initialization=dict(baseline_parameters=1462957,new_parameters=28517,
               old_mlp_retained_registered_but_unused_in_E1_E9=True,
               lazy_clip_load='Match Exp13 first-agent RN50 construction RNG; cached encoding; no training text encoder calls'),
           paper_review=dict(spromptgl='https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf',
               mmgpl='https://arxiv.org/html/2312.14574v2',
               audit='Fig1, Section2 Eq1-7/Fig2/Fig3 SPromptGL; MMGPL III-C/III-D Eq7-14/Fig1/Fig2. Existing concept weighting and token graph methods are not claimed novel.',
               differences='Original latent tokens are not ROI; ANAT/FUNC are QA. Here supervised training-fold FC deviations weight frozen real edge text; no new semantic graph or query/key. Evaluation retains original optimistic protocol.'),
           engineering_fixes=[],failures=[],handoff_appended=False)
    m['code_sha256']=code_hashes();jwrite(MANIFEST,m)
    record_event('PREPARED frozen 10 configurations / 30 tasks / 300 folds; text informative='+str(textinfo['informative']))

def build_sources():return e28_integration.emit(ROOT,SOURCE,EXP/'integration.patch')

def context():
    global _CTX
    if _CTX is not None:return _CTX
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Worker must see one physical GPU'
    sys.path.insert(0,str(build_sources()))
    import network,model
    assert Path(network.__file__).resolve()==(SOURCE/'network.py').resolve()
    assert Path(model.__file__).resolve()==(SOURCE/'model.py').resolve()
    cache=e28_prompts.load(CACHE)
    assert tuple(network.attr_words)==e28_prompts.ORIGINAL
    # EXACT Exp13 historical cold initialization. Loading RN50 builds a model
    # and consumes CPU RNG; replacing encode_text alone preserves this trajectory.
    def cached_p0():
        if network._prompt_text_features is None:
            encoder,_=network.clip.load('RN50',network.device)
            del encoder
            network._prompt_text_features=torch.from_numpy(cache['p0']).to('cuda')
        return network._prompt_text_features
    network.get_prompt_text_features=cached_p0
    from CLIP.clip.model import CLIP
    def forbidden_encoding(*args,**kwargs):raise RuntimeError('Text encoding forbidden in training workers')
    CLIP.encode_text=forbidden_encoding
    _CTX=dict(torch=torch,network=network,model=model,cache=cache,**data())
    return _CTX

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
def make_agent(config,seed,fold,labels=None,cls=None):
    c=context();tr,te=c['folds'][fold-1];y=c['y'] if labels is None else labels
    e28_integration.configure(config=config,fold=fold,raw=c['x'],train=tr,test=te,
            labels_train=y[tr],site=c['site'],pheno_indices=c['pheno_indices'],cache=c['cache'])
    with contextlib.redirect_stdout(io.StringIO()):
        a=(cls or c['model'].EvalHelper)([48,6,10,256],c['x'],y,SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    assert np.array_equal(a.val_idx,a.tst_idx)
    return a
def modules(a):return (a.MessagePassing0,a.ModalFusion)
def model_state(a):return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in zip(('gfr','pln'),modules(a))}
def load_models(a,s):
    for k,m in zip(('gfr','pln'),modules(a)):m.load_state_dict(s[k])
def parameter_hash(a,branch=False):
    h=hashlib.sha256()
    for m in modules(a):
        for n,p in sorted(m.named_parameters()):
            if ('ia_fdsa.' in n)!=branch:continue
            h.update(n.encode()+p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def state_equal(a,b):
    import torch
    assert a.keys()==b.keys()
    for part in a:
        assert a[part].keys()==b[part].keys()
        for k in a[part]:assert torch.equal(a[part][k],b[part][k]),(part,k)
def optimizer_audit(a):
    groups=a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups
    ids=[id(p) for g in groups for p in g['params']]
    actual=[p for m in modules(a) for p in m.parameters()]
    assert len(ids)==len(set(ids)) and set(ids)==set(map(id,actual))
    assert all(g['lr']==.0038 for g in groups)
    assert groups[0]['weight_decay']==groups[1]['weight_decay']==.11
    count=sum(p.numel() for p in actual)
    branch=getattr(a.ModalFusion,'ia_fdsa',None)
    assert count==1462957+(28517 if branch is not None else 0),count
    if branch is not None:assert groups[2]['weight_decay']==0 and len(groups)==3
    else:assert len(groups)==2
    return count
def evaluate(a):
    import torch
    for m in modules(a):m.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat)
        z,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        out,_,_=a.ModalFusion(z,raw_input=a.feat)
    assert out.shape==(871,2) and torch.isfinite(out).all()
    return out
def metrics(out,test):
    from sklearn.metrics import roc_auc_score,confusion_matrix
    p=out.detach().exp().cpu().numpy()[test];y=context()['y'][test];pred=p.argmax(1)
    tp,fn,fp,tn=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    sen=tp/(tp+fn);spe=tn/(tn+fp);ba=(sen+spe)/2
    return dict(acc=float((y==pred).mean()),ba=ba,hard_auc=ba,
        probability_auc=float(roc_auc_score(y==0,p[:,0])),sensitivity_asd=sen,specificity_td=spe,
        f1_asd=2*tp/max(2*tp+fn+fp,1),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)

def checkpoint_write(p,obj):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=io.BytesIO();torch.save(obj,b);body=b.getvalue();tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(b'E28!'+hashlib.sha256(body).hexdigest().encode()+body);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def checkpoint_read(p):
    import torch
    b=Path(p).read_bytes();assert b[:4]==b'E28!' and hashlib.sha256(b[68:]).hexdigest().encode()==b[4:68]
    return torch.load(io.BytesIO(b[68:]),map_location='cpu',weights_only=False)

def gradient_norms(a):
    result={};active=0
    b=getattr(a.ModalFusion,'ia_fdsa',None)
    for prefix,m in [('gfr',a.MessagePassing0),('pln_original',a.ModalFusion)]:
        ps=[p for n,p in m.named_parameters() if prefix!='pln_original' or not n.startswith('ia_fdsa.')]
        result[prefix]=float(np.sqrt(sum(float(p.grad.detach().square().sum()) for p in ps if p.grad is not None)))
    if b is not None:
        for n in ('alpha','fc_encoder','context_encoder','demo_encoder','anat_gate','func_gate','decode'):
            obj=getattr(b,n);ps=[obj] if isinstance(obj,context()['torch'].nn.Parameter) else list(obj.parameters())
            result[n]=float(np.sqrt(sum(float(p.grad.detach().square().sum()) for p in ps if p.grad is not None)))
    for m in modules(a):
        active+=sum(p.numel() for p in m.parameters() if p.grad is not None and float(p.grad.detach().abs().max())>0)
    result['active_parameters']=active
    assert all(np.isfinite(v) for v in result.values())
    return result

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

def smoke():
    """Real CUDA integration tests only; no test-fold pilot training/selection."""
    c=context();torch=c['torch'];network=c['network'];checks={}
    # Independent, unpatched copies in memory use the same byte-identical root
    # layers/utils. They are never saved into or imported from root as new files.
    baseline_network=types.ModuleType('_e28_unpatched_network')
    baseline_network.__file__=str(ROOT/'network.py')
    exec(compile((ROOT/'network.py').read_text(),str(ROOT/'network.py'),'exec'),baseline_network.__dict__)
    sys.modules[baseline_network.__name__]=baseline_network
    def original_cached():
        if baseline_network._prompt_text_features is None:
            encoder,_=baseline_network.clip.load('RN50',baseline_network.device);del encoder
            baseline_network._prompt_text_features=torch.from_numpy(c['cache']['p0']).to('cuda')
        return baseline_network._prompt_text_features
    baseline_network.get_prompt_text_features=original_cached
    original_model=types.ModuleType('_e28_unpatched_model')
    original_model.__file__=str(ROOT/'model.py')
    text=(ROOT/'model.py').read_text().replace('from network import *','from _e28_unpatched_network import *',1)
    exec(compile(text,str(ROOT/'model.py'),'exec'),original_model.__dict__)
    seed_rng(0);o=make_agent('E0_P0',0,1,cls=original_model.EvalHelper)
    state0=model_state(o);rng0=rng();init0=parameter_hash(o)
    seed_rng(0);a=make_agent('E0_P0',0,1)
    state_equal(model_state(a),state0);assert rng_hash(rng())==rng_hash(rng0)
    assert init0==historical_p0()[0,1]['initial_trunk_sha256'],('historical first initialization mismatch',init0)
    optimizer_audit(a)
    for m in modules(o):m.eval()
    with torch.no_grad():
        n,parts=o.dataLayer(o.feat);z,_,_=o.MessagePassing0(parts,n,o.targ,o.trn_idx)
        expected=o.ModalFusion(z)[0]
    actual=evaluate(a);assert torch.equal(expected,actual)
    entry=rng()
    with contextlib.redirect_stdout(io.StringIO()):o.run_epoch('simple-2')
    expected_state=model_state(o);expected_rng=rng();g0=gradient_norms(o)
    restore_rng(entry)
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    state_equal(model_state(a),expected_state);recursive_equal(gradient_norms(a),g0)
    recursive_equal(a.optimizer_MP0.state_dict(),o.optimizer_MP0.state_dict())
    recursive_equal(a.optimizer_MF.state_dict(),o.optimizer_MF.state_dict())
    assert rng_hash(rng())==rng_hash(expected_rng)
    checks['P0_unpatched_init_forward_two_adam_first_update_rng']='EXACT_PASS'
    checks['P0_seed0_fold1_initial_hash']=init0
    del o,a
    # Warm CLIP buffer matches all later folds. Newly appended modules may not
    # advance this entry RNG or change any original model parameter.
    seed_rng(1);entry=rng();e1=make_agent('E1_NO_SPGS',1,1)
    e1state=parameter_hash(e1);e1rng=rng_hash(rng());e1out=evaluate(e1)
    counts=[]
    for cfg in CONFIGS[2:]:
        restore_rng(entry);b=make_agent(cfg,1,1)
        assert parameter_hash(b)==e1state and rng_hash(rng())==e1rng,cfg
        counts.append(optimizer_audit(b)-1462957)
        assert torch.equal(evaluate(b),e1out),cfg+' zero alpha != E1'
        branch=b.ModalFusion.ia_fdsa
        assert float(branch.alpha)==0 and not branch.teacher_target.requires_grad
        assert branch.teacher_target.shape==(871,64)
        assert torch.count_nonzero(branch.semantic_edge_weights[:,[195,225,232,236]])==0
        perm=branch.aux_permutation.cpu().numpy()
        for part in (b.trn_idx,b.tst_idx):assert set(perm[part])==set(part)
        assert np.array_equal(c['site'][perm],c['site'])
        if cfg=='E8_FINE_NO_ALIGN':
            before=evaluate(b)
            with torch.no_grad():branch.teacher_target.copy_(torch.randn_like(branch.teacher_target))
            assert torch.equal(before,evaluate(b)) and float(branch.semantic_loss())==0
        del b
    assert set(counts)=={28517}
    checks['new_groups_same_original_parameters_rng_zero_alpha_E1']='EXACT_PASS'
    checks['new_parameters_all_eight_groups']=28517
    checks['unknown_edges_zero_semantic_weight']='PASS'
    # Actual training gradient scope: changing every held-out label changes no
    # teacher, prediction, train gradient, optimizer update, or RNG.
    restore_rng(entry);b=make_agent('E4_FINE_FULL',1,1);teacher0=b.ModalFusion.ia_fdsa.teacher_target.clone()
    run_entry=rng()
    with contextlib.redirect_stdout(io.StringIO()):b.run_epoch('simple-2')
    expected=model_state(b);expected_rng=rng();grad=gradient_norms(b)
    y=c['y'].copy();y[b.tst_idx]=1-y[b.tst_idx]
    restore_rng(entry);altered=make_agent('E4_FINE_FULL',1,1,labels=y)
    assert torch.equal(teacher0,altered.ModalFusion.ia_fdsa.teacher_target)
    restore_rng(run_entry)
    with contextlib.redirect_stdout(io.StringIO()):altered.run_epoch('simple-2')
    state_equal(model_state(altered),expected);recursive_equal(grad,gradient_norms(altered))
    assert rng_hash(rng())==rng_hash(expected_rng)
    checks['heldout_label_flip_teacher_full_train_update']='EXACT_PASS'
    assert grad['alpha']>0 and grad['fc_encoder']>0 and grad['context_encoder']>0
    assert grad['decode']==0,'Zero alpha must block decoder classification gradient at first step'
    with contextlib.redirect_stdout(io.StringIO()):b.run_epoch('simple-2')
    g2=gradient_norms(b)
    for name in ('alpha','fc_encoder','context_encoder','demo_encoder','anat_gate','func_gate','decode'):
        assert g2[name]>0,(name,g2[name])
    evaluate(b);br=b.ModalFusion.ia_fdsa
    assert br.last['delta'].abs().max()<=.1+1e-7
    assert torch.isfinite(br.last['student']).all()
    checks['two_step_gradient_norms']=g2
    # End-to-end checkpoint through on-disk checksum, both original Adams and
    # CPU/CUDA/Python/NumPy RNG. Loading buffers happens before RNG restoration.
    path=RUNTIME/'preflight_resume.pt'
    state=dict(model=model_state(b),mp=b.optimizer_MP0.state_dict(),mf=b.optimizer_MF.state_dict(),rng=rng())
    checkpoint_write(path,state)
    with contextlib.redirect_stdout(io.StringIO()):b.run_epoch('simple-2')
    expected=model_state(b);expected_rng=rng();out=evaluate(b)
    saved=checkpoint_read(path);restored=make_agent('E4_FINE_FULL',1,1)
    load_models(restored,saved['model']);restored.optimizer_MP0.load_state_dict(saved['mp'])
    restored.optimizer_MF.load_state_dict(saved['mf']);restore_rng(saved['rng'])
    with contextlib.redirect_stdout(io.StringIO()):restored.run_epoch('simple-2')
    state_equal(model_state(restored),expected);assert rng_hash(rng())==rng_hash(expected_rng)
    assert torch.equal(out,evaluate(restored));path.unlink()
    checks['epoch_resume_model_adam_rng_prediction']='EXACT_PASS'
    boundary=rng();next_agent=make_agent('E4_FINE_FULL',1,2);next_state=model_state(next_agent);next_rng=rng()
    # Resume with a primed lazy text cache, THEN restore the exact boundary RNG.
    restore_rng(boundary);repeat_agent=make_agent('E4_FINE_FULL',1,2)
    state_equal(next_state,model_state(repeat_agent));assert rng_hash(next_rng)==rng_hash(rng())
    checks['fold_boundary_next_initialization_rng']='EXACT_PASS'
    evaluate(restored);param=parameter_hash(restored)+parameter_hash(restored,True);before_rng=rng_hash(rng());real=evaluate(restored)
    for condition in INTERVENTIONS:
        restored.ModalFusion.ia_fdsa.condition=condition;pred=evaluate(restored)
        assert torch.isfinite(pred).all()
        restored.ModalFusion.ia_fdsa.condition='REAL'
        assert torch.equal(real,evaluate(restored)),condition
    assert param==parameter_hash(restored)+parameter_hash(restored,True)
    assert before_rng==rng_hash(rng())
    checks['counterfactual_restore_weights_rng_REAL']='EXACT_PASS'
    checks['device']=str(torch.cuda.get_device_properties(0))
    jwrite(RUNTIME/'preflight_checks.json',checks)
    print('E28_GPU_PREFLIGHT_PASS',json.dumps(checks),flush=True)

def preflight():
    prepare();m=verify_static()
    if m['status']=='STOPPED_TEXT_COLLAPSE':finish('STOPPED_TEXT_COLLAPSE');return
    if m.get('preflight',{}).get('passed'):
        record_event('PREFLIGHT_REUSE: hashes verified');return
    from e28_method import self_test
    self_test()
    env=worker_env(0)
    log=RUNTIME/'preflight.log'
    with log.open('w') as f:
        run=subprocess.run([sys.executable,str(EXP/'run.py'),'_smoke'],cwd=RUNTIME,env=env,stdout=f,stderr=subprocess.STDOUT)
    if run.returncode:
        m=jread(MANIFEST);m['status']='BLOCKED_PREFLIGHT';m['failures'].append(dict(time=utc(),stage='preflight',log=log.read_text()[-16000:]))
        jwrite(MANIFEST,m);raise RuntimeError('GPU preflight failed: '+log.read_text()[-4000:])
    checks=jread(RUNTIME/'preflight_checks.json')
    m=jread(MANIFEST);m['preflight']=dict(passed=True,completed_utc=utc(),checks=checks)
    m['status']='PREFLIGHT_PASS';jwrite(MANIFEST,m)
    record_event('PREFLIGHT_PASS: exact baseline update/RNG, held-out-label exclusion, gradients, epoch/fold restore')

def representation(x,y):
    x=np.asarray(x,np.float64);y=np.asarray(y)
    center=x-x.mean(0);s=np.linalg.svd(center,compute_uv=False);p=s*s/max(float((s*s).sum()),1e-30)
    between=float(np.linalg.norm(x[y==0].mean(0)-x[y==1].mean(0)))
    within=float(np.mean([np.linalg.norm(row-x[y==label].mean(0)) for row,label in zip(x,y)]))
    return dict(effective_rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),
                across_patient_variance=float((center*center).sum()/len(x)),
                class_centroid_distance=between,class_separation_ratio=between/max(within,1e-12))
def corr(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.std()<1e-12 or b.std()<1e-12:return 0.
    return float(np.corrcoef(a,b)[0,1])
def eta_squared(values,groups):
    v=np.asarray(values);g=np.asarray(groups);den=float(((v-v.mean())**2).sum())
    return float(sum(len(v[g==s])*(v[g==s].mean()-v.mean())**2 for s in np.unique(g))/max(den,1e-20))

def diagnostics(a,config,seed,fold,tracker):
    import torch
    c=context();b=getattr(a.ModalFusion,'ia_fdsa',None)
    row=dict(config=config,seed=seed,fold=fold,total_parameters=optimizer_audit(a),
             additional_parameters=28517 if b is not None else 0,
             sampled_gradient_norms_json=json.dumps(tracker,sort_keys=True),
             active_parameters_sampled_max=int(tracker.get('active_parameters',0)),
             spgs_enabled=config=='E0_P0',semantic_loss_enabled=config in CONFIGS[2:] and config!='E8_FINE_NO_ALIGN')
    if b is None:return row
    evaluate(a)
    def arr(t):return t.detach().cpu().numpy()
    tr=a.trn_idx;te=a.tst_idx;y=c['y'][tr];last=b.last
    target=arr(b.teacher_target);active=arr(b.teacher_active);student=arr(torch.nn.functional.normalize(last['student'],dim=-1))
    row.update(teacher_valid_train=float(active[tr].mean()),teacher_valid_test=float(active[te].mean()),
        teacher_variance=float(target[tr].var(0).sum()),
        teacher_pair_distance_mean=float(np.linalg.norm(target[tr]-target[tr].mean(0),axis=1).mean()),
        teacher_class_cosine=float(e28_prompts.normalized(target[tr][y==0].mean(0))@e28_prompts.normalized(target[tr][y==1].mean(0))),
        teacher_student_cosine_train=float((student[tr][active[tr]]*target[tr][active[tr]]).sum(1).mean()),
        teacher_student_cosine_test=float((student[te][active[te]]*target[te][active[te]]).sum(1).mean()),
        semantic_loss=float(b.semantic_loss().detach()),alpha=float(b.alpha.detach()),
        residual_abs_max=float(last['delta'].abs().max()),
        residual_l2_mean=float(last['delta'].norm(dim=-1).mean()),
        residual_relative_mean=float((last['delta'].norm(dim=-1)/(last['cam'][:,3,:].norm(dim=-1)+1e-8)).mean()),
        residual_nonzero_fraction=float((last['delta'].norm(dim=-1)>1e-8).float().mean()),
        fc_norm_mean=float(last['fc'].norm(dim=-1).mean()),
        context_weighted_norm_mean=float((.25*last['context']).norm(dim=-1).mean()),
        auxiliary_norm_mean=float(last['conditional'].norm(dim=-1).mean()),
        auxiliary_anchor_ratio=float((last['conditional'].norm(dim=-1)/(last['student'].norm(dim=-1)+1e-8)).mean()),
        auxiliary_effective_shuffled_fraction=float((arr(b.aux_permutation)!=np.arange(871)).mean()),
        statistics_json=json.dumps(b.statistics['report'],sort_keys=True))
    assert row['residual_abs_max']<=.1+1e-7
    reps={}
    for name,value in [('raw_fc',a.feat[:,64:]),('fc',last['fc']),('context',last['context']),
                       ('anchor',last['student']),('joint',last['joint']),('teacher',b.teacher_target)]:
        reps[name]=representation(arr(value)[tr],y)
    row['representations_json']=json.dumps(reps,sort_keys=True)
    age=c['x'][:,:26].argmax(1)
    for name in ('qa','qf'):
        gate=arr(last[name]);mean=gate.mean(1)
        row[name+'_mean']=float(gate.mean());row[name+'_patient_std']=float(mean.std())
        row[name+'_feature_std_mean']=float(gate.std(0).mean())
        row[name+'_age_category_correlation']=corr(mean,age)
        row[name+'_site_eta_squared']=eta_squared(mean,c['site'])
        row[name+'_qc_abs_correlation_max']=max(abs(corr(mean,c['x'][:,j])) for j in range(48,64))
    # Loss-specific gradients on the SAME trained model, train indices only.
    # No optimizer step; the continuous next-fold RNG is restored by caller.
    for m in modules(a):m.eval();m.zero_grad(set_to_none=True)
    n,parts=a.dataLayer(a.feat);z,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
    out,_,_=a.ModalFusion(z,raw_input=a.feat)
    classification=c['model'].ClsLoss(out,a.targ,a.trn_idx,a.weight)
    semantic=b.semantic_loss()
    parameters=list(b.fc_encoder.parameters())+list(b.context_encoder.parameters())
    gc=torch.autograd.grad(classification,parameters,retain_graph=True,allow_unused=True)
    gs=torch.autograd.grad(semantic,parameters,allow_unused=True)
    def norm(gs):return float(torch.sqrt(sum((g.detach().square().sum() for g in gs if g is not None),out.new_zeros(()))))
    nc,ns=norm(gc),norm(gs)
    row.update(classification_gradient_anchor=nc,semantic_gradient_anchor=ns,
               semantic_classification_gradient_ratio=ns/max(nc,1e-12))
    del out,z,classification,semantic,gc,gs
    for m in modules(a):m.zero_grad(set_to_none=True)
    return row

def counterfactual(a,seed,fold,real):
    import torch
    b=a.ModalFusion.ia_fdsa;te=a.tst_idx
    parameter=parameter_hash(a)+parameter_hash(a,True);before=rng_hash(rng())
    reference=real.exp()[te,0].detach().clone();ref_pred=real[te].argmax(1)
    reference_delta=b.last['delta'].detach().clone();rows=[]
    for condition in INTERVENTIONS:
        b.condition=condition;out=evaluate(a);p=out.exp()[te,0]
        change=(p-reference).abs();last=b.last
        rows.append(dict(config='E4_FINE_FULL',seed=seed,fold=fold,condition=condition,
                         **metrics(out,te),probability_abs_change=float(change.mean()),
                         prediction_flip_fraction=float((out[te].argmax(1)!=ref_pred).float().mean()),
                         probability_change_p50=float(change.quantile(.5)),
                         probability_change_p90=float(change.quantile(.9)),
                         probability_change_p100=float(change.max()),
                         residual_l2_mean=float(last['delta'][te].norm(dim=-1).mean()),
                         residual_change_l2_mean=float((last['delta'][te]-reference_delta[te]).norm(dim=-1).mean()),
                         qa_mean=float(last['qa'][te].mean()),qf_mean=float(last['qf'][te].mean()),
                         auxiliary_norm_mean=float(last['conditional'][te].norm(dim=-1).mean())))
        b.condition='REAL';assert torch.equal(real,evaluate(a)),condition+' REAL restore failed'
    assert parameter==parameter_hash(a)+parameter_hash(a,True) and before==rng_hash(rng())
    assert rows[0]['probability_abs_change']==0 and rows[1]['probability_abs_change']==0
    return rows

def task_name(c,s):return c+'_seed'+str(s)
def task_json(c,s):return RESULTS/'tasks'/(task_name(c,s)+'.json')
def task_checkpoint(c,s):return RUNTIME/'state'/(task_name(c,s)+'.pt')
def task_result(c,s):
    p=task_json(c,s)
    if not p.exists():return None
    r=jread(p);assert r['signature']==signature() and r['config']==c and r['seed']==s
    assert len({r['fold'] for r in r['metrics']})==len(r['metrics'])
    if r['status']=='COMPLETE':assert len(r['metrics'])==len(r['diagnostics'])==10
    return r

def worker(config,seed):
    m=verify_static();assert m.get('preflight',{}).get('passed')
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(GPU[config,seed])
    c=context();torch=c['torch'];name=task_name(config,seed);sig=signature()
    (RUNTIME/'state').mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'state'/(name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=task_result(config,seed)
    if existing and existing['status']=='COMPLETE':print('SKIP_COMPLETE',name,flush=True);return
    p=task_checkpoint(config,seed);saved=checkpoint_read(p) if p.exists() else None
    if saved:
        assert saved['signature']==sig
        result=saved['result']
        assert not existing or len(result['metrics'])>=len(existing['metrics'])
        # Do not load RN50 again after restoring an interrupted stream.
        c['network']._prompt_text_features=torch.from_numpy(c['cache']['p0']).to('cuda')
        restore_rng(saved['rng']);jwrite(task_json(config,seed),result)
    else:
        assert existing is None or not existing['metrics'],'Completed fold without recoverable continuation RNG'
        seed_rng(seed)
        result=dict(signature=sig,config=config,seed=seed,physical_gpu=GPU[config,seed],
                    status='RUNNING',metrics=[],diagnostics=[],counterfactual=[],seconds=0.,started_utc=utc(),resumes=0)
    if saved:result['resumes']+=1
    stop=[False]
    def signal_stop(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    begin=time.monotonic();oldseconds=result['seconds'];history=historical_p0()
    print('TASK_START',name,'GPU',GPU[config,seed],'logical0','resumed',saved is not None,flush=True)
    for fold in range(len(result['metrics'])+1,11):
        if stop[0]:return
        resumed=saved is not None and saved['phase']=='EPOCH' and saved['fold']==fold
        entry_rng=rng();a=make_agent(config,seed,fold);optimizer_audit(a)
        if resumed:
            load_models(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf'])
            restore_rng(saved['rng']);best_state=saved['best_model'];best=saved['best'];best_epoch=saved['best_epoch']
            wait=saved['wait'];start=saved['epoch']+1;initial=saved['initial'];init_rng_hash=saved['init_rng_hash'];tracker=saved['tracker']
        else:
            best_state=None;best=0.;best_epoch=-1;wait=0;start=0
            initial=parameter_hash(a);init_rng_hash=rng_hash(entry_rng);tracker={}
        if config=='E0_P0':
            assert initial==history[seed,fold]['initial_trunk_sha256'],f'P0_INITIAL_PARITY_FAIL seed{seed} fold{fold}: {initial}'
        saved=None
        print('FOLD_START',name,fold,'init',initial,flush=True)
        def save_epoch(epoch):
            result['seconds']=oldseconds+time.monotonic()-begin
            checkpoint_write(p,dict(signature=sig,phase='EPOCH',fold=fold,epoch=epoch,
                 model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),
                 best_model=best_state,best=best,best_epoch=best_epoch,wait=wait,initial=initial,
                 init_rng_hash=init_rng_hash,tracker=tracker,result=result))
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
                 best_epoch=best_epoch,epochs_executed=last+1,selected_val_acc=best,
                 train_n=len(a.trn_idx),test_n=len(a.tst_idx),initial_trunk_sha256=initial,
                 initial_rng_sha256=init_rng_hash,additional_parameters=28517 if config in CONFIGS[2:] else 0)
        diag=diagnostics(a,config,seed,fold,tracker)
        cf=counterfactual(a,seed,fold,evaluate(a)) if config=='E4_FINE_FULL' else []
        result['metrics'].append(row);result['diagnostics'].append(diag);result['counterfactual'].extend(cf)
        restore_rng(continuation);result['seconds']=oldseconds+time.monotonic()-begin
        result['status']='COMPLETE' if fold==10 else 'RUNNING'
        if fold==10:result['completed_utc']=utc()
        # Commit continuation and all derived rows before declaring fold complete.
        checkpoint_write(p,dict(signature=sig,phase='BOUNDARY',fold=fold,rng=continuation,result=result))
        jwrite(task_json(config,seed),result)
        print('FOLD_DONE',name,fold,'ACC',met['acc'],'BA',met['ba'],'AUC',met['probability_auc'],'best',best_epoch,flush=True)
        if config=='E0_P0':
            expected=history[seed,fold]
            for key in ('acc','ba','probability_auc','best_epoch'):
                assert abs(row[key]-expected[key])<=1e-12,('P0_HISTORY_PARITY_FAIL',seed,fold,key,row[key],expected[key])
        del a,best_state,out
        torch.cuda.empty_cache()
    print('TASK_COMPLETE',name,result['seconds'],flush=True)

def collect():
    groups=[]
    for c,s,_ in tasks():
        r=task_result(c,s)
        if r:groups.append(r)
    metrics_rows=[row for r in groups for row in r['metrics']]
    diag=[row for r in groups for row in r['diagnostics']]
    cf=[row for r in groups for row in r['counterfactual']]
    for rows,keys in ((metrics_rows,('config','seed','fold')),(diag,('config','seed','fold')),
                      (cf,('config','seed','fold','condition'))):
        assert len({tuple(r[k] for k in keys) for r in rows})==len(rows),'Duplicate row ID'
    return groups,metrics_rows,diag,cf

def refresh():
    m=verify_static();groups,rows,diag,cf=collect()
    for name,content in [('fold_metrics.csv',rows),('diagnostics.csv',diag),('counterfactual.csv',cf)]:
        if content:cwrite(RESULTS/name,content)
    index={(r['config'],r['seed']):r for r in groups}
    for t in m['tasks']:
        r=index.get((t['config'],t['seed']))
        if r:t.update(status=r['status'],completed_folds=len(r['metrics']),seconds=r['seconds'],resumes=r['resumes'])
    m.update(completed_folds=len(rows),completed_tasks=sum(r['status']=='COMPLETE' for r in groups),
             gpu_completed_folds={str(g):sum(r['physical_gpu']==g for r in rows) for g in (0,1)},
             refreshed_utc=utc())
    jwrite(MANIFEST,m)
    return groups,rows,diag,cf

def parity():
    _,rows,_,_=collect();actual=[r for r in rows if r['config']=='E0_P0'];history=historical_p0()
    assert len(actual)==30,'P0 full30 not completed'
    differences=[]
    for r in actual:
        old=history[r['seed'],r['fold']]
        for k in ('acc','ba','probability_auc','best_epoch','initial_trunk_sha256'):
            if r[k]!=old[k]:differences.append(dict(seed=r['seed'],fold=r['fold'],field=k,current=r[k],historical=old[k]))
    assert not differences,'P0 exact historical parity failed: '+json.dumps(differences)
    m=jread(MANIFEST);m['p0_parity']=dict(passed=True,rows=30,fields=['acc','ba','probability_auc','best_epoch','initial_trunk_sha256'],
         checked_utc=utc(),maximum_acc_ba_auc_error=0.,initial_state_and_rng_preflight=True)
    jwrite(MANIFEST,m);record_event('P0_PARITY_PASS full30 metrics/best epoch/initial parameters; remaining270 allowed')

def worker_env(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),E28_PHYSICAL_GPU=str(gpu),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    return env
def capacity():
    import psutil
    slots=4
    if psutil.virtual_memory().available<96*1024**3 or os.getloadavg()[0]>os.cpu_count()*.9:slots=2
    try:
        p=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
        if min(float(v.strip()) for v in p.splitlines())<16000:slots=1
    except (subprocess.SubprocessError,ValueError):pass
    return slots
def schedule(jobs):
    pending=[j for j in jobs if not (task_result(j[0],j[1]) or {}).get('status')=='COMPLETE']
    active=[];failure=None
    while pending or active:
        slots=capacity()
        for job in list(pending):
            c,s,g=job
            if sum(x['gpu']==g for x in active)>=slots:continue
            log=RUNTIME/(task_name(c,s)+'.log');handle=log.open('a')
            process=subprocess.Popen([sys.executable,str(EXP/'run.py'),'_worker',c,str(s)],
                     cwd=RUNTIME,env=worker_env(g),stdout=handle,stderr=subprocess.STDOUT)
            active.append(dict(process=process,gpu=g,config=c,seed=s,handle=handle,log=log))
            pending.remove(job);record_event(f'START {task_name(c,s)} GPU{g} pid={process.pid} maxworkers={slots}')
        time.sleep(5)
        for item in list(active):
            code=item['process'].poll()
            if code is None:continue
            item['handle'].close();active.remove(item)
            record_event(f'EXIT {task_name(item["config"],item["seed"])} code={code}')
            if code:
                failure=dict(task=task_name(item['config'],item['seed']),returncode=code,
                             tail=item['log'].read_text()[-12000:],time=utc())
                break
        refresh()
        if failure:
            # Stop only our workers; each handler persists a complete epoch.
            for item in active:item['process'].terminate()
            for item in active:
                try:item['process'].wait(timeout=55)
                except subprocess.TimeoutExpired:item['process'].kill();item['process'].wait()
                item['handle'].close()
            m=jread(MANIFEST);m['failures'].append(failure);m['status']='PARTIAL';jwrite(MANIFEST,m)
            raise RuntimeError('Task failed; exact state retained: '+json.dumps(failure))

def summarize(rows,metric):
    seeds=[float(np.mean([float(r[metric]) for r in rows if int(r['seed'])==s])) for s in SEEDS]
    return dict(mean=float(np.mean(seeds)),sd_population=float(np.std(seeds,ddof=0)),per_seed=seeds)
def contrasts(rows):
    index={(r['config'],int(r['seed']),int(r['fold'])):r for r in rows};out=[]
    for left,right in CONTRASTS:
        if not all((c,s,f) in index for c in (left,right) for s in SEEDS for f in range(1,11)):continue
        result=dict(left=left,right=right,metrics={})
        for metric in METRICS:
            diffs=np.array([float(index[left,s,f][metric])-float(index[right,s,f][metric]) for s in SEEDS for f in range(1,11)])
            result['metrics'][metric]=dict(delta_pp=float(diffs.mean()*100),
                per_seed_delta_pp=(diffs.reshape(3,10).mean(1)*100).tolist(),
                wins=int((diffs>1e-12).sum()),ties=int((abs(diffs)<=1e-12).sum()),losses=int((diffs< -1e-12).sum()))
        out.append(result)
    return out

def detailed_results(m, table, paired, rows, diag, cf, scientific):
    """Report-only interpretation of all frozen contrasts; no model decisions."""
    if len(rows) != 300:
        return []
    pi = {(p['left'], p['right']): p['metrics'] for p in paired}
    def ds(c):
        return [r for r in diag if r['config'] == c]
    def mean(c, key):
        return float(np.mean([float(r[key]) for r in ds(c)]))
    def delta(a, b, k):
        return pi[a, b][k]['delta_pp']
    def seeds(a, b, k):
        return '/'.join(f'{v:+.4f}' for v in pi[a, b][k]['per_seed_delta_pp'])
    def supported(a, b):
        return all(pi[a, b][k]['delta_pp'] > 0 and
                   all(x > 0 for x in pi[a, b][k]['per_seed_delta_pp'])
                   for k in ('ba', 'probability_auc'))
    lines = ['', '## 针对本轮实测结果的逐项判读', '',
             '与MMGPL的可学习概念权重及Token语义图不同，本轮使用固定真实FC连接文本，并由训练折参考与患者FC数值确定教师，监督64维FC影像锚；没有新增自由概念Query或语义图。'
             '与原SPromptGL两主题压缩成Softmax-Max标量不同，E1–E9去掉该权重，E2–E9以有界CORR残差参与原RES；因此必须同时比较无SPGS的E1和完整P0，不能把旁路结构变化归为文本收益。']
    best = {k: max(table, key=lambda c: table[c][k]['mean']) for k in ('acc', 'ba', 'probability_auc')}
    lines.append('按三seed均值，ACC最高为' + best['acc'] + '，BA最高为' + best['ba'] +
                 '，连续Prob-AUC最高为' + best['probability_auc'] + '；这些排名不改变预设主方法E4及其判据。')
    for a,b,label in [
        ('E3_FINE_FC','E2_OLD_FC','细粒度相对原8句粗粒度教师'),
        ('E4_FINE_FULL','E3_FINE_FC','完整辅助条件相对CORR语义主锚'),
        ('E4_FINE_FULL','E5_NUMERIC_FULL','正确医学文本相对数值码'),
        ('E4_FINE_FULL','E6_WRONG_ROI','正确FC连接身份相对错配ROI'),
        ('E4_FINE_FULL','E7_WRONG_DIRECTION','正确增强/减弱方向相对反向绑定'),
        ('E4_FINE_FULL','E8_FINE_NO_ALIGN','语义监督相对关闭对齐损失'),
        ('E4_FINE_FULL','E9_WRONG_AUX_PAIR','正确患者辅助配对相对同站点错配'),
        ('E4_FINE_FULL','E1_NO_SPGS','新模块相对删除SPGS'),
        ('E4_FINE_FULL','E0_P0','新模块相对完整P0')]:
        conclusion = '满足BA/Prob-AUC均为正且三seed同向的内部候选门槛' if supported(a,b) else '未满足BA/Prob-AUC均为正且三seed同向的内部候选门槛'
        lines.append(f'{label}：ΔACC={delta(a,b,"acc"):+.5f}pp、ΔBA={delta(a,b,"ba"):+.5f}pp、'
                     f'ΔProb-AUC={delta(a,b,"probability_auc"):+.5f}pp；BA三seed={seeds(a,b,"ba")}pp，'
                     f'Prob-AUC三seed={seeds(a,b,"probability_auc")}pp。{conclusion}。')
    full = [r for r in rows if r['config']=='E4_FINE_FULL']
    base = [r for r in rows if r['config']=='E0_P0']
    fn = sum(r['asd_fn'] for r in full); fn0 = sum(r['asd_fn'] for r in base)
    fp = sum(r['td_fp'] for r in full); fp0 = sum(r['td_fp'] for r in base)
    lines.append(f'E4的ASD漏诊累计{fn}次，P0为{fn0}次；TD误报累计{fp}次，P0为{fp0}次。'
                 '这些为三个seed重复评估的病例事件数，并非三倍独立患者。')
    lines += ['', '### 表示是否坍塌及训练判别性', '',
              '|组|教师患者方差|教师ASD/TD中心cos|raw FC有效秩|anchor有效秩|teacher有效秩|anchor类分离比|joint类分离比|teacher类分离比|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    rep = {}
    for c in CONFIGS[2:]:
        rr = [json.loads(r['representations_json']) for r in ds(c)]
        def rm(name, key):return float(np.mean([r[name][key] for r in rr]))
        rep[c] = rr
        values = [mean(c,'teacher_variance'),mean(c,'teacher_class_cosine'),
                  rm('raw_fc','effective_rank'),rm('anchor','effective_rank'),rm('teacher','effective_rank'),
                  rm('anchor','class_separation_ratio'),rm('joint','class_separation_ratio'),rm('teacher','class_separation_ratio')]
        lines.append('|'+c+'|'+'|'.join(f'{v:.7g}' for v in values)+'|')
    lines.append('类分离比=两类训练中心距离/平均类内半径；此处全部以outer-train标签作事后诊断。'
                 'anchor与joint是同一已训练模型的两个位置，不能据此推断相对训练前的因果提升。'
                 'E2是同一粗粒度向量的加权平均，理论上没有患者语义差异；其极小非零方差及有效秩可由浮点归一化噪声产生，不能把噪声秩解读为丰富信息。'
                 '高student-teacher余弦而低教师类差异，可能只是拟合共同文本方向。')
    lines.append(f'E4教师患者方差={mean("E4_FINE_FULL","teacher_variance"):.8f}，'
                 f'两类中心余弦={mean("E4_FINE_FULL","teacher_class_cosine"):.8f}，'
                 f'训练student-teacher余弦={mean("E4_FINE_FULL","teacher_student_cosine_train"):.8f}。'
                 '教师并非逐患者完全相同，但两类中心高度同向；不能把非零患者方差或有效秩直接解释为强疾病语义区分性。')
    lines += ['', '### 梯度、有效训练参数与残差', '',
              '|组|已注册总参数|新增参数|采样有效参数范围|alpha梯度|FC梯度|context梯度|demo梯度|ANAT门梯度|FUNC门梯度|decode梯度|',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        xx=ds(c);gg=[json.loads(r['sampled_gradient_norms_json']) for r in xx]
        vals=[float(np.mean([g.get(k,0.) for g in gg])) for k in ('alpha','fc_encoder','context_encoder','demo_encoder','anat_gate','func_gate','decode')]
        active=[int(r['active_parameters_sampled_max']) for r in xx]
        lines.append(f'|{c}|{xx[0]["total_parameters"]}|{xx[0]["additional_parameters"]}|{min(active)}–{max(active)}|'+
                     '|'.join(f'{v:.5g}' for v in vals)+'|')
    lines.append('梯度统计为每折前两epoch及每25epoch采样的模块梯度范数最大值，再跨30折平均；'
                 '有效参数数目也仅为这些采样时刻非零梯度元素所属参数的最大计数，不等于全训练过程的精确活跃参数数目。'
                 'E2/E3辅助模块注册但不参与有效贡献；E1–E9原MLP保留注册但不参与前向，故不能把注册参数全部算作实际训练容量。')
    e4=ds('E4_FINE_FULL')
    residual=[float(r['residual_l2_mean']) for r in e4]
    nonzero=mean('E4_FINE_FULL','residual_nonzero_fraction')
    lines.append(f'E4残差平均L2={np.mean(residual):.7f}，逐fold均值范围{min(residual):.7f}–{max(residual):.7f}，'
                 f'非零响应比例={nonzero:.7f}，最大坐标绝对值={max(float(r["residual_abs_max"]) for r in e4):.7f}。'
                 f'平均相对CAM范数={mean("E4_FINE_FULL","residual_relative_mean"):.7f}；'
                 '该相对量只作诊断，数学约束仍是坐标≤0.10、L2≤0.6。'
                 + ('存在总体近零响应，需要结合alpha/梯度看待。' if np.mean(residual)<1e-5 else '未出现总体近零残差；分支被使用仍不等于正确医学文本有效。'))
    lines.append(f'E4语义/分类对anchor梯度范数比均值={mean("E4_FINE_FULL","semantic_classification_gradient_ratio"):.7g}；'
                 f'语义梯度均值={mean("E4_FINE_FULL","semantic_gradient_anchor"):.7g}，分类梯度均值={mean("E4_FINE_FULL","classification_gradient_anchor"):.7g}。'
                 '这是最佳Checkpoint的单次train-loss诊断，不是训练全过程累计贡献。')
    lines += ['', '### 辅助门控与站点关联', '',
              '|组|辅助/anchor|实际错配比例|QA/QF患者均值STD|QA/QF与年龄类别r|QA/QF站点η²|QA/QF最大QC绝对r|',
              '|---|---:|---:|---|---|---|---|']
    for c in CONFIGS[4:]:
        pair=lambda suffix:f'{mean(c,"qa_"+suffix):.6f}/{mean(c,"qf_"+suffix):.6f}'
        applied_shuffle=mean(c,'auxiliary_effective_shuffled_fraction') if c=='E9_WRONG_AUX_PAIR' else 0.
        lines.append(f'|{c}|{mean(c,"auxiliary_anchor_ratio"):.7f}|{applied_shuffle:.7f}|'+
                     '|'.join(pair(s) for s in ('patient_std','age_category_correlation','site_eta_squared','qc_abs_correlation_max'))+'|')
    lines.append('原始diagnostics.csv中的auxiliary_effective_shuffled_fraction记录各模型预生成置乱表的覆盖率，'
                 '该表仅在E9训练或AUX_PAIR_SHUFFLED干预时启用；上表按真实训练配置显示应用比例，其余组为0。原始诊断不作回写。')
    ss=[json.loads(r['statistics_json']) for r in e4 if int(r['seed'])==0]
    lines.append(f'十套fold训练统计中合格站点数范围={min(r["eligible_sites"] for r in ss)}–{max(r["eligible_sites"] for r in ss)}，'
                 f'平均站点方向一致率={np.mean([r["site_direction_agreement_mean"] for r in ss]):.7f}，'
                 f'平均站点效应异质性={np.mean([r["site_effect_heterogeneity_mean"] for r in ss]):.7f}。'
                 '同fold在三个seed共享统计，此处只计一次；QA与年龄/站点/QC关联不能解释为校准后的医学可靠度。')
    lines += ['', '### 反事实变化分布及实际依赖', '',
              '|干预|逐fold平均概率变化的p50均值|p90均值|逐fold最大变化的最大值|QA均值|QF均值|辅助范数|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for cond in INTERVENTIONS:
        rr=[r for r in cf if r['condition']==cond]
        vals=[np.mean([r[k] for r in rr]) for k in ('probability_change_p50','probability_change_p90')]
        vals += [max(r['probability_change_p100'] for r in rr)]
        vals += [np.mean([r[k] for r in rr]) for k in ('qa_mean','qf_mean','auxiliary_norm_mean')]
        lines.append('|'+cond+'|'+'|'.join(f'{v:.8f}' for v in vals)+'|')
    cfmean=lambda c,k:float(np.mean([r[k] for r in cf if r['condition']==c]))
    lines.append(f'关闭整条语义残差的ASD概率绝对变化={cfmean("SEMANTIC_RESIDUAL_OFF","probability_abs_change"):.7f}，'
                 f'关闭辅助修正={cfmean("AUX_OFF","probability_abs_change"):.7f}，'
                 f'关闭CAM上下文={cfmean("FC_ANCHOR_CONTEXT_OFF","probability_abs_change"):.7f}。'
                 '这些值说明已训练模型依赖哪些计算路径；必须与E4−E3/E8/E9完整重训结果共同解释。'
                 'SEMANTIC_TEACHER_OFF概率变化严格为0，是训练专用教师的预期性质；'
                 '正确/错误文本只通过训练损失作用，不能把同Checkpoint的teacher-off当作医学文本必要性检验。')
    lines += ['', '### 工程恢复与最终研究决策', '']
    start=dt.datetime.fromisoformat(m['started_utc']);end=dt.datetime.fromisoformat(m.get('finished_utc',utc()))
    lines.append(f'墙钟执行时间={(end-start).total_seconds():.2f}秒；开始{m["started_utc"]}，结束{m.get("finished_utc","运行中")}。'
                 '各GPU累计task运行秒包含同卡并行任务的重叠时间，不能与墙钟时间直接相加。')
    groups=[jread(task_json(c,s)) for c,s,g in tasks()]
    for gpu in (0,1):
        gg=[r for r in groups if int(r['physical_gpu'])==gpu]
        lines.append(f'GPU{gpu}：15任务/150折，累计task运行秒={sum(r["seconds"] for r in gg):.2f}；分配 '+
                     ', '.join(task_name(r['config'],r['seed']) for r in gg)+'。')
    for i,fix in enumerate(m.get('engineering_fixes',[]),1):
        concise={k:v for k,v in fix.items() if k!='diff'}
        lines.append(f'已审计工程修复{i}：'+json.dumps(concise,ensure_ascii=False,sort_keys=True)+'；逐行diff保留于manifest。')
    if scientific=='NO-GO':
        lines.append('最终科学决策：NO-GO。预设方法未通过全部正确文本、ROI身份、方向、语义监督和辅助患者配对的稳健对照；'
                     '当前证据不支持将IA-FDSA作为MICCAI 2027第一创新点。保留完整阴性结果，停止本数据与历史协议下静态FC文本追分；'
                     '优先增加真实互补影像/临床测量，并采用训练内选模、LOSO或独立队列检验。')
    else:
        lines.append('最终科学决策：内部候选。即使各预设对照均支持，也仅能进入采用训练内选模、LOSO或独立队列的外部验证；'
                     '当前历史协议不足以证明MICCAI 2027级别的临床泛化创新。')
    return lines


def report():
    m=verify_static();groups,rows,diag,cf=collect()
    table={c:{k:summarize([r for r in rows if r['config']==c],k) for k in METRICS}
           for c in CONFIGS if sum(r['config']==c for r in rows)==30}
    paired=contrasts(rows);history=historical_p0();reference={k:summarize([dict(seed=s,**v) for (s,f),v in history.items()],k) for k in METRICS}
    scientific='NOT_EVALUABLE'
    if len(rows)==300:
        pindex={(p['left'],p['right']):p for p in paired}
        required=[(a,b) for a,b in CONTRASTS]
        meets=True
        for a,b in required:
            for metric in ('ba','probability_auc'):
                item=pindex[a,b]['metrics'][metric]
                meets=meets and item['delta_pp']>0 and all(v>0 for v in item['per_seed_delta_pp'])
        scientific='CANDIDATE_REQUIRING_EXTERNAL_VALIDATION' if meets else 'NO-GO'
    summary=dict(status=m['status'],scientific_conclusion=scientific,completed_folds=len(rows),
        completed_tasks=sum(r['status']=='COMPLETE' for r in groups),metrics=table,contrasts=paired,
        historical_p0=reference,historical_exp9_e9=dict(acc=.9188262,ba=.9169151,probability_auc=.9382272),
        gpu_folds={str(g):sum(r['physical_gpu']==g for r in rows) for g in (0,1)},
        task_seconds={task_name(r['config'],r['seed']):r['seconds'] for r in groups})
    jwrite(RESULTS/'summary.json',summary)
    lines=['# 实验28：IA-FDSA影像锚定的细粒度疾病语义对齐','',
        f'状态：**{m["status"]}**；实际完成 **{len(rows)}/300折**；科学结论：**{scientific}**。','',
        '本轮验证细粒度FC文本、正确解剖身份/方向及正确辅助患者配对能否提供额外诊断信息。未按测试成绩修改Prompt、损失、网络或配置。','',
        '## 审计、来源与历史教训','',
        f'本地HEAD `{m["source_local_head"]}`；GitHub main `{m["source_remote_head"]}`。六核心源码逐字节一致；本轮仅在独立运行副本应用 integration.patch。',
        '原GFR四张受试者图、i==1覆盖、阈值0.9、四图正则平均、ALoss不反传、CAM特殊归一化及原RES/分类器均保留。CAM的4×36及4×4×9都是潜在Token，不能解释为具体ROI。ANAT6/FUNC10是扫描质量指标。',
        '原SPGS把两类相似度Softmax后取max，类别位置交换不改变权重。此前原型趋同、自由Key重参数化、中性/随机控制不劣、PHENO强制对齐、ROI旁路、QC校准及患者增益路由失败，是本轮设立严格负对照的动机。',
        'MMGPL已经提出概念权重和概念响应Token图（III-C/III-D，Eq.7–14）；SPromptGL提出GFR/CAM/SPGS/RES（Eq.1–7）。本轮不宣称上述机制首创，也不把文本可区分当作医学知识有效。',
        '文献：[MMGPL](https://arxiv.org/html/2312.14574v2)；[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)。MMGPL输入/样本数与协议不同，不能直接横比其ACC。','',
        '## 冻结文本及教师','',
        '保留原8句（含Nromal拼写）。依据实验11最终connection_name、atlas端点和验证列，252连接×增强/减弱=504句；195/225/232/236四条仍在原CORR256中，但语义权重严格为0。完整文本、索引、文本SHA和原始/低维向量分别在prompt_banks.json/npz。',
        'RN50冻结，P0八句独立编码并与实验12旧缓存逐值相同；512句共同单位归一化后做无中心SVD取64维，列符号固定，再单位归一化，不使用患者数据或标签。',
        'E2两种ASD关联方向均使用原异常四句平均向量；原正常四句参与共同文本基底。它是刻意粗粒度/近恒定教师，FC降低不能等同正常。',
        '数值对照E5：seed28005随机正交行坐标染色，最终归一化之前均值/协方差严格匹配文本，之后逐向量单位范数；个体配对Gram不相同，记录实际几何差异。E6 seed28006无固定点连接错配；E7仅交换方向。',
        '每fold TD均值/总体标准差仅来自train TD；d优先在同站点ASD/TD各至少2人时估计，先截断标准化差异至±2，按有效样本/(有效样本+20)收缩，再按调和有效样本权重汇总；无合格站点时才使用同规则全train估计。报告站点异质性，不能称病灶真值。',
        'r截断±3；a=ReLU(sign(d)r−0.5)×clip(|d|,0,2)。教师是252对应文本的加权单位向量，全部a为0时跳过。语义损失0.05×SmoothL1，仅train有效患者，不读取test标签。','',
        '## 模型、协议与预检','',
        'FC256→Linear64/LayerNorm/GELU；CAM CORR36→同类64维映射；anchor=FC+.25context。demo= tanh(Linear([年龄/性别28,anchor64]))；QA分别6/10→64 sigmoid；joint=anchor+.25qa×qf×demo。decode64→36；delta=.10tanh(alpha)tanh(decode)，alpha=0后安装。',
        '仅CORR CAM位置增加残差，再进入无SPGS权重的原RES。每坐标绝对残差≤0.10，36维L2≤0.6；本公式不是相对原特征10%的约束。alpha=0与E1逐值相等，不能误称与P0相等。',
        'E2–E9均注册28517个新参数，原1462957参数及原MLP注册顺序保留；E1–E9原MLP不参与前向。新分支在原my_weight_init之后fork_rng安装，新参数Adam lr=.0038/wd=0；原两Adam及原wd=.11保持。E2/E3的辅助模块注册但不参与有效贡献，实际梯度另报。',
        '原始协议：871人，403 ASD(0)/468 TD(1)，48/6/10/256列；10fold StratifiedKFold(shuffle=True,random_state=10)，seed0/1/2，每任务只播种一次连续10折；450epochs、wait>200、首次严格test/val ACC最高选模；dropout.35、18hidden、2heads、1layer，GC weighted-cosine/MP GCN/MF concat/simple-2。',
        'CLIP首次构建消耗的初始化RNG按实验13保留；之后使用固定缓存。新增模块不消耗原主干RNG；不同组早停后续折可能自然分叉，同seed/fold不代表完全同初始化。',
        '预检结果：`'+json.dumps(m.get('preflight',{}),ensure_ascii=False)+'`。',
        '完整P0历史parity：`'+json.dumps(m.get('p0_parity',{}),ensure_ascii=False)+'`。只有通过完整30折ACC/BA/ProbAUC/best epoch/初始化逐项核验后才允许其他270折。','',
        '## 十组正式设计','',
        '|组|方法|SPGS|语义监督|辅助修正|','|---|---|---|---|---|']
    for c,d in m['design'].items():lines.append(f'|{c}|{d["purpose"]}|{d["spgs"]}|{d["semantic_loss"]}|{d["auxiliary"]}|')
    lines+=['','## 实测主结果（先seed十折平均，再三seed均值±population SD，%）','',
        '|组|ACC|BA/Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|','|---|---:|---:|---:|---:|---:|---:|']
    display=('acc','ba','probability_auc','sensitivity_asd','specificity_td','f1_asd')
    def fmt(v):return f'{v["mean"]*100:.5f}±{v["sd_population"]*100:.5f}'
    for c in CONFIGS:
        if c in table:lines.append('|'+c+'|'+'|'.join(fmt(table[c][k]) for k in display)+'|')
        else:lines.append('|'+c+'|'+('未完成|'*6))
    lines+=['','历史实验13 P0只读参照：'+'；'.join(k+' '+fmt(reference[k]) for k in display)+'。',
        '历史实验9 E9：ACC91.88262%、BA91.69151%、Prob-AUC93.82272%，属于不同基座的历史描述性比较，绝不是本轮E9_WRONG_AUX_PAIR。','',
        '## 三个seed十折均值及病例计数','',
        '|组|seed|ACC|BA|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|ASD FN|TD FP|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in table:
        for s in SEEDS:
            rs=[r for r in rows if r['config']==c and r['seed']==s]
            lines.append('|'+c+'|'+str(s)+'|'+'|'.join(f'{np.mean([r[k] for r in rs])*100:.5f}' for k in display)+f'|{sum(r["asd_fn"] for r in rs)}|{sum(r["td_fp"] for r in rs)}|')
    lines+=['','## 全部预设配对（pp）','',
        '|比较|指标|均差pp|seed0/1/2差pp|逐折胜/平/负|','|---|---|---:|---|---|']
    for p in paired:
        for metric in METRICS:
            x=p['metrics'][metric]
            lines.append(f'|{p["left"]} − {p["right"]}|{metric}|{x["delta_pp"]:+.5f}|'+','.join(f'{v:+.5f}' for v in x['per_seed_delta_pp'])+f'|{x["wins"]}/{x["ties"]}/{x["losses"]}|')
    if 'E4_FINE_FULL' in table:
        lines+=['','E4相对历史实验9 E9（描述性）：'+', '.join(k+f' {(table["E4_FINE_FULL"][k]["mean"]-v)*100:+.5f}pp' for k,v in summary['historical_exp9_e9'].items())+'。']
    textinfo=jread(EXP/'prompt_banks.json')
    lines+=['','## 文本、教师和辅助机制诊断','',
        '文本几何：`'+json.dumps(textinfo['diagnostics'],ensure_ascii=False)+'`。',
        '统一短模板选择索引 '+str(textinfo['selected_template'])+'；最大token数 '+str(textinfo['token_count_max'])+'；truncate=False。',
        '|组|教师有效train比例|教师方差|student-teacher cos|alpha|残差L2|相对CAM残差|辅助/anchor|qa/qf患者STD|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in CONFIGS[2:]:
        ds=[r for r in diag if r['config']==c]
        if not ds:continue
        def avg(k):return np.mean([float(r[k]) for r in ds])
        keys=('teacher_valid_train','teacher_variance','teacher_student_cosine_train','alpha','residual_l2_mean','residual_relative_mean','auxiliary_anchor_ratio')
        lines.append('|'+c+'|'+'|'.join(f'{avg(k):.7f}' for k in keys)+f'|{avg("qa_patient_std"):.7f}/{avg("qf_patient_std"):.7f}|')
        lines.append('') if False else None
    lines+=['','每折diagnostics记录原始FC/FC编码/context/anchor/joint/teacher有效秩与类中心距离、teacher类余弦、语义/分类梯度比、各模块梯度、QA与年龄类别/站点/QC关联、站点差异稳定性及训练统计SHA。QA只是学习的条件系数，未经医学可靠度校准；高teacher类余弦或低方差意味着类别代码/近恒定目标，不能称个体医学知识成功。','',
        '## E4同Checkpoint反事实','',
        '|干预|ΔASD概率绝对均值|翻转率|ΔACC pp|ΔBA pp|ΔProbAUC pp|残差变化L2|','|---|---:|---:|---:|---:|---:|---:|']
    realcf={(r['seed'],r['fold']):r for r in cf if r['condition']=='REAL'}
    for condition in INTERVENTIONS:
        cs=[r for r in cf if r['condition']==condition]
        if not cs:continue
        values=[np.mean([r[k] for r in cs]) for k in ('probability_abs_change','prediction_flip_fraction')]
        values.extend(100*np.mean([r[k]-realcf[r['seed'],r['fold']][k] for r in cs]) for k in ('acc','ba','probability_auc'))
        values.append(np.mean([r['residual_change_l2_mean'] for r in cs]))
        lines.append('|'+condition+'|'+'|'.join(f'{v:.8f}' for v in values)+'|')
    lines+=['','SEMANTIC_TEACHER_OFF只关闭训练损失，不改变推理，预期概率变化严格0；其净作用应看E4−E8完整重训。PHENO_OFF将demo条件修正置0；ANAT/FUNC_QC_OFF把指定门控置1而移除该调制。AUX_PAIR_SHUFFLED只错配新增28+6+10，同站点且train/test内独立，不更改原主干或FC。单Checkpoint关闭分支不是重新训练消融，可能分布外。','',
        '## 资源、恢复、完整性','',
        f'完成任务 {summary["completed_tasks"]}/30，完成fold {len(rows)}/300，diagnostics {len(diag)}/300，counterfactual {len(cf)}/270；GPU0/1 fold数 {summary["gpu_folds"]}。每任务连续十折、最多每GPU4个worker/每worker2线程；资源不足只下调并发。',
        '25epoch及fold边界原子保存两个模型/两个Adam/所有RNG/最佳checkpoint/等待计数/签名及已提交行；fold状态先持久化再标记完成，不混合失败attempt。',
        '逐任务耗时秒：`'+json.dumps(summary['task_seconds'],sort_keys=True)+'`。',
        '异常/阻塞记录：`'+json.dumps(m['failures'],ensure_ascii=False)+'`。',
        '源码SHA：`'+json.dumps(m['source_sha256'],sort_keys=True)+'`。',
        '数据及ROI SHA：`'+json.dumps(m['data_sha256'],sort_keys=True)+'`。',
        f'RN50权重SHA `{m["clip_weights_sha256"]}`；Prompt缓存 `{m["prompt_cache_sha256"]}`；文本清单 `{m["prompt_json_sha256"]}`。',
        '实验源码SHA：`'+json.dumps(m['code_sha256'],sort_keys=True)+'`。','',
        '## 科学判断、评价偏差与后续','',
        f'本轮结论 **{scientific}**。预设主要配对须在BA和Prob-AUC同时改善且三个seed方向一致才列候选；单一最高ACC、有效梯度、低对齐损失、可解释连接词或单Checkpoint响应都不构成医学语义独立价值证据。',
        '细粒度Prompt是否优于粗粒度、文本是否优于数值码、正确ROI/方向是否优于错配、语义损失及正确辅助配对是否必要，分别以E3−E2、E4−E5/E6/E7/E8/E9及E4−E3/E1/E0判断；不省略阴性控制。',
        '本协议test=val且test ACC参与checkpoint选择，RFE在当前外层CV之前进行，QC有全队列预处理，全871人transductive传播，并已经历27轮探索，故结果是偏乐观内部证据。共享患者和fold的三个seed不是30个独立队列，不报告伪独立显著性、不称无泄漏泛化或临床因果获益。',
        'FC是RFE保留的Fisher-z稀疏连接；训练差异和候选文本不是已验证个体ASD病灶；QA不等于独立结构/功能成像流。',
        '若严格控制不成立，停止当前静态FC疾病文本路线并保留强基线，优先收集更互补的真实影像/临床模态并开展训练内选模的独立验证；若各项一致成立，也只能作为候选，仍需LOSO/独立队列验证。','']
    lines += detailed_results(m, table, paired, rows, diag, cf, scientific)
    tmp=RESULTS/'report.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(RESULTS/'report.md')
    return summary

def append_handoff(summary):
    m=jread(MANIFEST);target=ROOT/'handoff.md';marker='<!-- EXPERIMENT28_IA_FDSA_FINAL -->'
    stamp=utc()
    text='\n\n'+marker+'\n## 实验二十八：IA-FDSA影像锚定的细粒度疾病语义对齐\n\n'
    text+=f'更新时间：{stamp}。STATUS={m["status"]}，科学结论={summary["scientific_conclusion"]}。\n\n'
    text+='本轮保留原GFR/CAM/RES及历史协议，以原始FC及训练折TD参考/站点分层差异加权252真实连接的504条方向Prompt，在共享文本64维空间监督影像锚；年龄/性别与QA仅作受约束条件修正，替换SPGS。\n\n'
    text+='十组：'+', '.join(CONFIGS)+'。\n\n'
    text+=f'实际完成{summary["completed_folds"]}/300折、{summary["completed_tasks"]}/30任务，GPU分配/完成{summary["gpu_folds"]}。P0历史parity：{json.dumps(m.get("p0_parity",{}),ensure_ascii=False)}。\n\n'
    for c,stats in summary['metrics'].items():
        text+=c+'：'+', '.join(k+f'={stats[k]["mean"]*100:.5f}±{stats[k]["sd_population"]*100:.5f}%' for k in METRICS if k!='hard_auc')+'。\n'
    text+='\n预设配对：'+json.dumps(summary['contrasts'],ensure_ascii=False)+'。\n\n'
    text+='历史P0 ACC91.69105/BA91.53935/ProbAUC93.33520%；历史Exp9 E9 ACC91.88262/BA91.69151/ProbAUC93.82272%，不同于本轮E9。\n\n'
    text+='细粒度语义、正确ROI/方向、语义监督及辅助配对的必要性以完整重训负对照判断；teacher-off预期不改变推理，不能据此宣称语义无用；其他反事实和梯度/教师坍塌详见实验28/results/report.md。\n\n'
    text+='执行时间 '+m['started_utc']+' 至 '+stamp+'；任务耗时 '+json.dumps(summary['task_seconds'])+'。\n\n'
    text+='源码SHA '+json.dumps(m['source_sha256'])+'；数据SHA '+json.dumps(m['data_sha256'])+'；Prompt缓存SHA '+m['prompt_cache_sha256']+'；历史保护 '+json.dumps(m.get('protection',{}))+'。\n\n'
    text+='偏差与局限：test=val选模、外层前RFE/预处理、全871人transductive、多轮探索；不称独立无泄漏泛化、医学可靠度或个体病灶验证。若NO-GO，停止静态FC文本追分，优先真实互补模态和独立训练内选模验证。无自动Git操作。\n'
    with target.open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);old=f.read()
        prefix=old[:m['original_handoff_size']]
        assert hashlib.sha256(prefix).hexdigest()==m['original_handoff_sha256'],'Existing handoff prefix changed'
        if marker.encode() not in old:
            f.seek(0,2);f.write(text.encode());f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)
    m['handoff_appended']=True;m['handoff_updated_utc']=stamp;m['handoff_sha256']=sha(target);jwrite(MANIFEST,m)

def finish(status):
    groups,rows,diag,cf=refresh();m=jread(MANIFEST)
    if status=='COMPLETE':
        assert len(rows)==len(diag)==300 and len(cf)==270 and len(groups)==30
        assert all(r['status']=='COMPLETE' for r in groups)
        assert all(sum(r['physical_gpu']==g for r in rows)==150 for g in (0,1))
        assert m.get('p0_parity',{}).get('passed')
        for row in rows:
            assert all(np.isfinite(row[k]) for k in METRICS)
    differences=[p for p,h in m['protected_files_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    assert not differences,'Protected historical files changed: '+str(differences)
    m.update(status=status,finished_utc=utc(),protection=dict(status='UNCHANGED_SHA256',files=len(m['protected_files_sha256'])))
    jwrite(MANIFEST,m);summary=report()
    append_handoff(summary)
    m=jread(MANIFEST);m['artifacts_sha256']={str(p.relative_to(EXP)):sha(p) for p in RESULTS.glob('*') if p.is_file()}
    jwrite(MANIFEST,m)
    if status=='COMPLETE':
        # Only this experiment's temporary directory is eligible for cleanup.
        assert RUNTIME.parent==EXP and RUNTIME.name=='.runtime'
        shutil.rmtree(RUNTIME)
    print('STATUS='+status,'SCIENTIFIC='+summary['scientific_conclusion'],'folds',len(rows),flush=True)

def all_tasks():
    prepare();preflight();m=verify_static()
    if m['status']=='COMPLETE':
        assert len(collect()[1])==300;print('ALREADY_COMPLETE hashes and300 unique folds verified');return
    if m['status']=='STOPPED_TEXT_COLLAPSE':return
    m['status']='RUNNING_P0_PARITY' if not m.get('p0_parity',{}).get('passed') else 'RUNNING_FORMAL'
    jwrite(MANIFEST,m)
    if not m.get('p0_parity',{}).get('passed'):
        schedule([j for j in tasks() if j[0]=='E0_P0']);parity()
    m=jread(MANIFEST);m['status']='RUNNING_FORMAL';jwrite(MANIFEST,m)
    schedule([j for j in tasks() if j[0]!='E0_P0'])
    finish('COMPLETE')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','preflight','all','resume','report','refresh','_worker','_smoke'))
    p.add_argument('config',nargs='?');p.add_argument('seed',nargs='?',type=int);args=p.parse_args()
    if args.command=='_worker':worker(args.config,args.seed);return
    if args.command=='_smoke':smoke();return
    try:
        if args.command=='prepare':prepare()
        elif args.command=='preflight':preflight()
        elif args.command in ('all','resume'):all_tasks()
        elif args.command=='refresh':refresh()
        elif args.command=='report':
            report();print((RESULTS/'report.md').read_text())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if MANIFEST.exists():
            m=jread(MANIFEST)
            if not m['failures'] or str(exc) not in str(m['failures'][-1]):
                m['failures'].append(dict(time=utc(),stage=args.command,error=str(exc)))
            status='BLOCKED_P0_PARITY' if 'PARITY' in str(exc) else 'BLOCKED_ENGINEERING'
            m['status']=status;jwrite(MANIFEST,m)
            # Repairable interface errors retain state for an audited retry;
            # only a real historical parity stop is a final research stop.
            if status=='BLOCKED_P0_PARITY':
                try:finish(status)
                except Exception:traceback.print_exc()
        raise

if __name__=='__main__':main()




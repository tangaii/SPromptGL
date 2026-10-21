#!/usr/bin/env python3
"""Experiment30 standalone controller: prepare -> preflight -> all/resume -> report.

18 config tasks, seed0, continuous 10 folds each. Historical P0/E1 read only.
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
from method import CONFIGS,DESIGNS,SEEDS,METRICS,CONTRASTS,INTERVENTIONS,RULES,tasks,asdict_designs
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
        f.write(b'E30!'+hashlib.sha256(body).hexdigest().encode()+body);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def checkpoint_read(p):
    import torch
    b=Path(p).read_bytes();assert b[:4]==b'E30!' and hashlib.sha256(b[68:]).hexdigest().encode()==b[4:68]
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
    if den<=1.e-12*max(1.,float((v*v).sum())):return 0.
    value=sum(len(v[g==s])*(v[g==s].mean()-v.mean())**2 for s in np.unique(g))/den
    return float(np.clip(value,0.,1.))

def safe_corr(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.std()<1.e-12 or b.std()<1.e-12:return 0.
    return float(np.corrcoef(a,b)[0,1])

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
    """Read-only seed0 ten-fold P0 and E1 references."""
    raw=readcsv(HIST/'results/fold_metrics.csv')
    hf=[r for r in readcsv(BASELINE) if r['config']=='F0' and int(r['seed'])==0]
    assert len(hf)==10
    result=[]
    for config,source in (('P0','E0_P0'),('E1','E1_NO_SPGS')):
        rows=[r for r in raw if r['config']==source and int(r['seed'])==0]
        assert len(rows)==10
        for r in rows:
            f=int(r['fold']); counts={k:int(r[k]) for k in ('asd_tp','asd_fn','td_fp','td_tn')}
            tp,fn,fp,tn=(counts[k] for k in ('asd_tp','asd_fn','td_fp','td_tn'))
            if config=='P0':
                h=next(v for v in hf if int(v['fold'])==f)
                assert all(int(h[k])==counts[k] for k in counts)
                assert abs(float(r['acc'])-float(h['acc']))<1e-12
                assert abs(float(r['hard_auc'])-float(h['hard_auc']))<2e-15
                assert r['best_epoch']==h['best_epoch']
            result.append(dict(config=config,seed=0,fold=f,acc=float(r['acc']),auc=float(r['hard_auc']),
                sen_asd=tp/(tp+fn+1e-8),spe_td=tn/(tn+fp+1e-8),**counts,
                best_epoch=int(r['best_epoch']),initial_trunk_sha256=r.get('initial_trunk_sha256','')))
    return result

def prepare():
    if MANIFEST.exists():
        verify_static(); return
    started=utc(); RESULTS.mkdir(parents=True,exist_ok=True); RUNTIME.mkdir(exist_ok=True)
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    hashes=core_hashes()
    def remote_check(item):
        name,digest=item
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/tangaii/SPromptGL/{remote}/{name}',timeout=30) as f: body=f.read()
        assert hashlib.sha256(body).hexdigest()==digest, 'Remote core mismatch '+name
        return name,digest
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex: remote_hashes=dict(ex.map(remote_check,hashes.items()))
    old=json.loads((HIST/'manifest.json').read_text())
    assert old.get('status')=='COMPLETE' and old.get('completed_folds')==300
    assert old.get('source_sha256')==hashes
    info=data(); ref=references(); assert len(ref)==20
    textinfo=prompt_bank.build(ROOT,EXP); build_sources()
    hand=(ROOT/'handoff.md').read_bytes()
    input_files=[ROOT/'data/ABIDE/processed_standard_data.csv',ROOT/'data/ABIDE/modal_feat_dict.npy']
    input_files += list((ROOT/'experiment/experiment11_roi_mapping_recovery/results').glob('roi_*csv'))
    audited={}
    for folder in (ROOT/'experiment/experiment11_roi_mapping_recovery',ROOT/'experiment/experiment13_patient_specific_fc_prompt',HIST):
        if not folder.is_dir(): continue
        for q in folder.glob('*.py'):
            try: ast.parse(q.read_text())
            except SyntaxError: continue
            audited[str(q)]={'sha256':sha(q),'lines':len(q.read_text().splitlines())}
        for q in (folder/'results').glob('*.md'):
            audited[str(q)]={'sha256':sha(q),'characters':len(q.read_text())}
    protected={str(q):sha(q) for q in [ROOT/n for n in CORE] + [HIST/'manifest.json',HIST/'prompt_banks.npz',HIST/'prompt_banks.json',BASELINE]}
    m=dict(experiment='Experiment30 class-conditional semantic shared repair',status='PREPARED',
      started_utc=started,prepared_utc=utc(),source_local_head=local,source_remote_head=remote,
      source_sha256=hashes,source_remote_core_sha256=remote_hashes,source_remote_core_identical=True,
      data_sha256={str(q):sha(q) for q in input_files},reference_sha256={str(q):sha(q) for q in [BASELINE,HIST/'results/fold_metrics.csv',HIST/'manifest.json',HIST/'prompt_banks.npz',HIST/'prompt_banks.json']},
      protected_files_sha256=protected,original_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
      original_handoff_size=len(hand),original_handoff_sha256=hashlib.sha256(hand).hexdigest(),
      prompt_source_sha256=sha(ROOT/'prompt'),prompt_cache_sha256=sha(CACHE),text_bank=textinfo,
      audit=audited,papers={'spromptgl':'https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf','mmgpl':'https://arxiv.org/html/2312.14574v2'},
      paper_local_sha256={q.name:sha(q) for q in ROOT.glob('*SPromptGL*.pdf')},design={c:asdict_designs()[c] for c in CONFIGS},rules=RULES,protocol=PROTOCOL,
      tasks=[dict(config=c,seed=s,gpu=g,status='PENDING',completed_folds=0) for c,s,g in tasks()],target_tasks=18,target_folds=180,
      completed_tasks=0,completed_folds=0,gpu_fold_targets={'0':90,'1':90},contrasts=CONTRASTS,
      historical_references={'P0':'Exp13 F0 seed0 read-only 10 folds; Exp28 E0 parity','E1':'Exp28 E1 seed0 read-only 10 folds'},
      expected_baseline_parameters=1462957,expected_additional_parameters=43279,
      initialization='Original trunk RNG preserved; new branch fork_rng seed 300000; one seed0 stream continuous over ten folds',
      metrics_note='ACC, one-hot hard-label AUC (=BA), ASD sensitivity and TD specificity; mean/population SD over seed0 ten-fold metrics.',
      duplicate_design_note='Z0/Z1 and C3/F5 are distinct registered controls; configurations with equal flags are still independently trained.',
      runner_provenance='Experiment30 disposable runtime copies, exact root GFR/PLN/CAM/RES and two Adam optimizers; checkpoint magic E30!',failures=[],handoff_appended=False)
    m['code_sha256']=code_hashes(); jwrite(MANIFEST,m)
    record_event('PREPARED 18 tasks / seed0 / 180 folds; frozen 764-text bank and original P0 cache recorded.')


def context():
    global _CTX
    if _CTX is not None:return _CTX
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    sys.path.insert(0,str(build_sources()))
    for name in ('model','network','layers','utils'):
        sys.modules.pop(name,None)
    import model,network,layers,utils
    for mod in (model,network,layers,utils): assert Path(mod.__file__).resolve().parent==SOURCE.resolve()
    cache=prompt_bank.load(CACHE)
    assert tuple(network.attr_words)==prompt_bank.ORIGINAL
    def cached_p0():
        if network._prompt_text_features is None:
            network._prompt_text_features=torch.from_numpy(cache['p0']).to('cuda')
        return network._prompt_text_features
    network.get_prompt_text_features=cached_p0
    from CLIP.clip.model import CLIP
    CLIP.encode_text=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('CLIP text encoding is frozen and forbidden during training'))
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
            if ('semantic30.' in n)!=branch: continue
            h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def optimizer_audit(a):
    groups=a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups
    ids=[id(p) for g in groups for p in g['params']]
    params=[p for m in modules(a) for p in m.parameters()]
    assert len(ids)==len(set(ids)) and set(ids)==set(map(id,params))
    assert all(abs(g['lr']-.0038)<1e-12 for g in groups)
    assert groups[0]['weight_decay']==groups[1]['weight_decay']==.11
    b=getattr(a.ModalFusion,'semantic30',None); new=sum(p.numel() for p in b.parameters()) if b is not None else 0
    assert sum(p.numel() for p in params)==1462957+new
    assert len(groups)==(3 if b is not None else 2)
    if b is not None: assert groups[2]['weight_decay']==0 and new==43279
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
    pred=out.detach().cpu().numpy()[test].argmax(1); y=context()['y'][test]
    tp,fn,fp,tn=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    sen=tp/(tp+fn+1e-8); spe=tn/(tn+fp+1e-8)
    auc=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred]))
    return dict(acc=float((pred==y).mean()),auc=auc,sen_asd=sen,spe_td=spe,asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)

def gradient_norms(a):
    torch=context()['torch'];out={}
    for name,m in [('gfr',a.MessagePassing0),('pln_original',a.ModalFusion)]:
        ps=[p for n,p in m.named_parameters() if name!='pln_original' or not n.startswith('semantic30.')]
        out[name]=float(sum((p.grad.detach().square().sum() for p in ps if p.grad is not None),a.feat.new_zeros(())).sqrt())
    b=getattr(a.ModalFusion,'semantic30',None)
    if b is not None:
        groups={'fc_encoder':b.fc_encoder,'fc_query':b.fc_query,'scalar_query':b.scalar_query,'semantic_head':b.semantic_head,
                'fc_shared':b.fc_shared,'aux_encoder':b.aux_encoder,'aux_shared':b.aux_shared,'aux_private':b.aux_private,
                'aux_reconstruct':b.aux_reconstruct,'aux_router':b.aux_router,'aux_value':b.aux_value,'auxiliary_head':b.auxiliary_head}
        for name,m in groups.items():
            ps=[m] if isinstance(m,torch.nn.Parameter) else list(m.parameters())
            out[name]=float(sum((p.grad.detach().square().sum() for p in ps if p.grad is not None),a.feat.new_zeros(())).sqrt())
    assert all(np.isfinite(v) for v in out.values())
    return out

def raw_forward(a):
    n,parts=a.dataLayer(a.feat);z,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
    return a.ModalFusion(z,raw_input=a.feat)[0]


def smoke():
    """Real-GPU engineering checks; this never records a formal fold."""
    from method import training_statistics,select_edges
    import importlib.util
    c=context(); torch=c['torch']; checks={}
    cache=c['cache']
    assert cache['original'].shape==(8,64) and cache['fine_asd'].shape==(252,2,64)
    assert cache['fine_td'].shape==(252,64) and len(cache['valid_indices'])==252
    assert all(np.isfinite(cache[k]).all() for k in ('original','fine_asd','fine_td','p0'))
    checks['prompt_cache_shapes_finite']='PASS: 8 coarse, 252 connections, 764 frozen texts'

    # Compare the disposable P0 PLN against the unmodified network forward.
    seed_rng(0); p0=make_agent('P0',0,1); optimizer_audit(p0)
    assert not hasattr(p0.ModalFusion,'semantic30')
    saved_rng=rng()
    spec=importlib.util.spec_from_file_location('e30_original_network',ROOT/'network.py')
    original_network=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original_network;spec.loader.exec_module(original_network)
    original_network.get_prompt_text_features=lambda:torch.from_numpy(cache['p0']).to('cuda')
    original_pln=original_network.PLN([48,6,10,256],SimpleNamespace(**PROTOCOL)).to(p0.dev)
    original_pln.load_state_dict(p0.ModalFusion.state_dict(),strict=True);original_pln.eval();p0.ModalFusion.eval()
    with torch.no_grad():
        n,parts=p0.dataLayer(p0.feat);cam,_,_=p0.MessagePassing0(parts,n,p0.targ,p0.trn_idx)
        original_result=original_pln(cam);patched_result=p0.ModalFusion(cam,raw_input=p0.feat)
    recursive_equal(original_result,patched_result);restore_rng(saved_rng)
    checks['P0_original_vs_disposable_forward']='BITWISE_EXACT'
    checks['P0_parameter_count']=sum(p.numel() for m in modules(p0) for p in m.parameters())
    del original_pln,original_network,p0

    seed_rng(0); z0=make_agent('Z0',0,1); _,new=optimizer_audit(z0); assert new==43279
    seed_rng(0); c1=make_agent('C1',0,1); _,new2=optimizer_audit(c1); assert new2==43279
    assert parameter_hash(z0)==parameter_hash(c1)
    b=z0.ModalFusion.semantic30; zout=evaluate(z0); cout=evaluate(c1)
    assert torch.equal(zout,cout) and b.last['delta'].count_nonzero()==0
    seed_rng(0);p0_init=make_agent('P0',0,1)
    assert parameter_hash(z0)==parameter_hash(c1)==parameter_hash(p0_init)
    del p0_init
    checks['Z0_SPgs_off_initial_forward']='BITWISE_EXACT'
    checks['new_parameters']=new

    # Statistics and connection selection depend only on outer-train labels.
    tr,te=c['folds'][0]; flipped=c['y'].copy();flipped[te]=1-flipped[te]
    stats_a=training_statistics(c['x'][:,64:],tr,c['y'][tr],c['site'])
    stats_b=training_statistics(c['x'][:,64:],tr,flipped[tr],c['site'])
    for key in ('mean','std','direction','reliability','agreement','heterogeneity'):
        assert np.array_equal(stats_a[key],stats_b[key])
    select_a=select_edges(c['x'][:,64:],stats_a,cache['valid_indices'])
    select_b=select_edges(c['x'][:,64:],stats_b,cache['valid_indices'])
    for left,right in zip(select_a,select_b):assert np.array_equal(left,right)
    checks['heldout_label_flip_statistics_and_selection']='BITWISE_EXACT'
    assert not set(tr)&set(te) and len(set(tr)|set(te))==871

    # Zero heads, residual caps, non-CORR preservation, empty selection and
    # an all-zero auxiliary input are checked independently of training.
    b=c1.ModalFusion.semantic30
    assert b.semantic_head[-1].weight.count_nonzero()==0 and b.auxiliary_head[-1].weight.count_nonzero()==0
    probe_state={k:v.detach().clone() for k,v in b.state_dict().items()}
    cam=torch.randn((871,4,36),device='cuda');raw=torch.as_tensor(c['x'],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        b.semantic_head[-1].weight.normal_(0,.1);b.auxiliary_head[-1].weight.normal_(0,.1)
        out=b(cam,raw);den=cam[:,3,:].norm(dim=-1).clamp_min(1.e-8)
        assert torch.equal(out[:,:3],cam[:,:3])
        assert torch.all(b.last['sem_delta'].norm(dim=-1)<.05*den)
        assert torch.all(b.last['aux_delta'].norm(dim=-1)<.05*den)
        assert torch.all(b.last['delta'].norm(dim=-1)<.10*den)
        raw_zero=raw.clone();raw_zero[:,:64]=0
        b(cam,raw_zero);assert b.last['aux_delta'].count_nonzero()==0
    b.load_state_dict(probe_state)
    zero_stats=dict(mean=np.zeros(256),std=np.ones(256),direction=np.zeros(256),reliability=np.zeros(256))
    _,_,zero_weights,_,zero_active=select_edges(c['x'][:,64:],zero_stats,cache['valid_indices'])
    assert not zero_active.any() and not zero_weights.any() and np.isfinite(zero_weights).all()
    edge_weights=b.edge_weights.detach().clone()
    with torch.no_grad():
        b.edge_weights.zero_();b(cam,raw)
        assert b.last['semantic_value'].count_nonzero()==0 and torch.isfinite(b.last['class_logits']).all()
    b.edge_weights.copy_(edge_weights)
    checks['residual_bounds_cam_identity_zero_aux_and_empty_attention']='PASS'

    # Fine controls preserve the selected edge set and key dimensions while
    # changing the registered text assignment.
    seed_rng(0); fine=make_agent('F1',0,1); fb=fine.ModalFusion.semantic30
    edge_ids=fb.selected.detach().clone();fb.condition='REAL';_,real_keys=fb._text()
    text_changes={}
    for mode in ('WRONG_ROI','REVERSE_DIRECTION','SCRAMBLED_TEXT'):
        fb.condition=mode;_,keys=fb._text()
        assert keys.shape==(871,16,2,64) and not torch.equal(keys,real_keys)
        assert torch.equal(fb.selected,edge_ids);text_changes[mode]=True
    fb.condition='REAL';checks['fine_text_controls_keep_edges_and_change_keys']=text_changes

    seed_rng(0); shuffled=make_agent('C3',0,1);shuffled.MessagePassing0.eval();shuffled.ModalFusion.eval()
    feature_snapshot=shuffled.feat.detach().clone()
    with torch.no_grad():
        n,parts=shuffled.dataLayer(shuffled.feat);gfr_a,_,_=shuffled.MessagePassing0(parts,n,shuffled.targ,shuffled.trn_idx)
        shuffled.ModalFusion(gfr_a,raw_input=shuffled.feat)
        aux=shuffled.ModalFusion.semantic30
        expected={name:aux.aux_shared[name](aux.aux_encoder[name](shuffled.feat[aux.aux_perm][:,aux.pheno_ids] if name=='PHENO' else shuffled.feat[aux.aux_perm,48:54] if name=='ANAT' else shuffled.feat[aux.aux_perm,54:64])) for name in ('PHENO','ANAT','FUNC')}
        for name in expected:assert torch.equal(expected[name],aux.last['auxiliary_details'][name]['shared'])
        n,parts=shuffled.dataLayer(shuffled.feat);gfr_b,_,_=shuffled.MessagePassing0(parts,n,shuffled.targ,shuffled.trn_idx)
    assert torch.equal(feature_snapshot,shuffled.feat) and torch.equal(gfr_a,gfr_b)
    checks['shuffled_auxiliary_preserves_original_gfr_input']='BITWISE_EXACT'

    # Two optimizer updates must open gradients through zero-initialized heads.
    seed_rng(0); a=make_agent('C1',0,1)
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    first_grads=gradient_norms(a)
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    second_grads=gradient_norms(a)
    for key in ('fc_query','semantic_head','fc_shared','aux_router','aux_value'):
        assert second_grads[key]>0,(key,second_grads[key])
    checks['first_step_gradients']=first_grads;checks['second_step_gradients']=second_grads
    b=a.ModalFusion.semantic30
    out=raw_forward(a);terms=b.loss_terms(a.targ);cls=c['model'].ClsLoss(out,a.targ,a.trn_idx,a.weight)
    private=list(b.aux_private.parameters())
    class_grad=torch.autograd.grad(cls,private,retain_graph=True,allow_unused=True)
    rec_grad=torch.autograd.grad(terms['reconstruction_loss'],private,retain_graph=True,allow_unused=True)
    orth_grad=torch.autograd.grad(terms['orthogonality_loss'],private,retain_graph=True,allow_unused=True)
    norm=lambda gs:float(sum((g.detach().square().sum() for g in gs if g is not None),out.new_zeros(())).sqrt())
    assert norm(class_grad)==0 and norm(rec_grad)>0 and norm(orth_grad)>0
    checks['shared_private_gradient_boundary']=dict(classification_private=norm(class_grad),reconstruction_private=norm(rec_grad),orthogonality_private=norm(orth_grad))
    # Perturbing only held-out auxiliary values cannot change train-only losses.
    raw_changed=torch.as_tensor(c['x'].copy(),dtype=torch.float32,device='cuda');raw_changed[te,:64]=0
    zero_cam=torch.zeros((871,4,36),device='cuda')
    b(zero_cam,raw)
    terms_a=b.loss_terms(a.targ)
    b(zero_cam,raw_changed);terms_b=b.loss_terms(a.targ)
    for key in ('pair_loss','reconstruction_loss','orthogonality_loss'):
        assert torch.equal(terms_a[key],terms_b[key]),key
    checks['auxiliary_losses_training_indices_only']='BITWISE_EXACT'

    # The held-out label flip must leave one full two-Adam update unchanged.
    seed_rng(0); control=make_agent('C1',0,1)
    with contextlib.redirect_stdout(io.StringIO()):control.run_epoch('simple-2')
    labels=c['y'].copy();labels[te]=1-labels[te]
    seed_rng(0); changed=make_agent('C1',0,1,labels=labels)
    with contextlib.redirect_stdout(io.StringIO()):changed.run_epoch('simple-2')
    state_equal(model_state(control),model_state(changed))
    recursive_equal(control.optimizer_MP0.state_dict(),changed.optimizer_MP0.state_dict())
    recursive_equal(control.optimizer_MF.state_dict(),changed.optimizer_MF.state_dict())
    checks['heldout_label_flip_two_adam_update']='BITWISE_EXACT'

    # Simulate a 25-epoch interruption using a disk checkpoint, both Adam
    # states and all RNGs, then verify the next update and prediction exactly.
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(23):a.run_epoch('simple-2')
    payload=dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng())
    resume_path=RUNTIME/'preflight_resume.pt';checkpoint_write(resume_path,payload)
    expected_rng=payload['rng']
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    expected_model=model_state(a);expected_mp=a.optimizer_MP0.state_dict();expected_mf=a.optimizer_MF.state_dict();expected_pred=evaluate(a);expected_after=rng_hash(rng())
    seed_rng(0);resumed=make_agent('C1',0,1);stored=checkpoint_read(resume_path)
    load_models(resumed,stored['model']);resumed.optimizer_MP0.load_state_dict(stored['mp']);resumed.optimizer_MF.load_state_dict(stored['mf']);restore_rng(stored['rng'])
    with contextlib.redirect_stdout(io.StringIO()):resumed.run_epoch('simple-2')
    state_equal(expected_model,model_state(resumed));recursive_equal(expected_mp,resumed.optimizer_MP0.state_dict());recursive_equal(expected_mf,resumed.optimizer_MF.state_dict())
    assert torch.equal(expected_pred,evaluate(resumed)) and expected_after==rng_hash(rng())
    assert rng_hash(expected_rng)==rng_hash(payload['rng'])
    checks['epoch25_disk_resume_both_adams_rng_next_update_prediction']='BITWISE_EXACT'

    # Same-checkpoint interventions are diagnostics only and must restore the
    # original forward path without changing parameters or random streams.
    real=evaluate(a);before=parameter_hash(a)+parameter_hash(a,True);before_rng=rng_hash(rng());changes={}
    for mode in INTERVENTIONS:
        a.ModalFusion.semantic30.condition=mode;out=evaluate(a)
        changes[mode]=float((out.exp()-real.exp()).abs().mean().item())
    a.ModalFusion.semantic30.condition='REAL';assert torch.equal(real,evaluate(a))
    assert before==parameter_hash(a)+parameter_hash(a,True) and before_rng==rng_hash(rng())
    checks['intervention_changes']=changes
    checks['cam_non_corr_unchanged']='BITWISE_EXACT'
    jwrite(RUNTIME/'preflight_checks.json',checks)
    print('E30_GPU_PREFLIGHT_PASS',flush=True)


def worker_env(gpu):
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),E30_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    return env

def preflight():
    prepare(); m=verify_static()
    if m.get('preflight',{}).get('passed') and m.get('preflight',{}).get('version')==3:return
    from method import self_test
    self_test()
    for q in (EXP/'method.py',EXP/'prompt_bank.py',EXP/'integration.py',EXP/'run.py'):
        compile(q.read_text(),str(q),'exec')
    path=RUNTIME/'preflight.log'
    with path.open('w') as f:
        p=subprocess.run([sys.executable,str(EXP/'run.py'),'_smoke'],cwd=RUNTIME,env=worker_env(0),stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError('GPU preflight failed: '+path.read_text()[-10000:])
    checks=jread(RUNTIME/'preflight_checks.json')
    checks['python_syntax_compile']='PASS'
    gpu1=subprocess.check_output([sys.executable,'-c',"import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0))"],env=worker_env(1),text=True).strip()
    checks['GPU1_visible_logical0']=gpu1
    m=jread(MANIFEST); m['preflight']=dict(version=3,passed=True,completed_utc=utc(),checks=checks); m['status']='PREFLIGHT_PASS'; jwrite(MANIFEST,m)
    record_event('PREFLIGHT_V3_PASS: original P0 parity, label boundary, residual/gradient contracts and 25-epoch disk resume verified; no formal fold trained.')


def rank_summary(x):
    x=np.asarray(x,float); x=x-x.mean(0); s=np.linalg.svd(x,compute_uv=False)
    if float((s*s).sum())<1e-25:return dict(effective_rank=0.,variance=0.)
    p=s*s/float((s*s).sum()); p=p[p>0]
    return dict(effective_rank=float(np.exp(-(p*np.log(p)).sum())),variance=float((x*x).sum()/len(x)))

def counterfactual(a,real):
    b=a.ModalFusion.semantic30; before=parameter_hash(a)+parameter_hash(a,True); before_rng=rng_hash(rng()); te=a.tst_idx
    rp=real.exp()[te,0]; rpred=real[te].argmax(-1); rows=[]
    for mode in INTERVENTIONS:
        b.condition=mode; out=evaluate(a); p=out.exp()[te,0]; mm=metrics(out,te)
        rows.append(dict(condition=mode,probability_abs_change=float((p-rp).abs().mean()),prediction_flip_fraction=float((out[te].argmax(-1)!=rpred).float().mean()),**mm))
    b.condition='REAL'; assert torch_equal(real,evaluate(a)); assert before==parameter_hash(a)+parameter_hash(a,True); assert before_rng==rng_hash(rng())
    return rows

def torch_equal(a,b):return context()['torch'].equal(a,b)


def counterfactual(a,real):
    b=a.ModalFusion.semantic30;before=parameter_hash(a)+parameter_hash(a,True);before_rng=rng_hash(rng())
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
    c=context(); torch=c['torch']; b=a.ModalFusion.semantic30; evaluate(a); v=b.last; tr=a.trn_idx
    def npv(t):return t.detach().cpu().numpy()
    selected=npv(b.edge_active); counts=selected.sum(1); target=npv(v['target']); delta=npv(v['delta']); sem=npv(v['sem_delta']); aux=npv(v['aux_delta'])
    margin=npv(v['class_logits'][:,0]-v['class_logits'][:,1]); y=c['y'][tr]
    edge_weights=npv(b.edge_weights)
    edge_entropy=-(edge_weights*np.log(np.maximum(edge_weights,1.e-12))).sum(1)
    den=np.maximum(np.linalg.norm(target,axis=1),1.e-8)
    sem_relative=np.linalg.norm(sem,axis=1)/den
    aux_relative=np.linalg.norm(aux,axis=1)/den
    total_relative=np.linalg.norm(delta,axis=1)/den
    row=dict(config=config,seed=0,fold=fold,total_parameters=optimizer_audit(a)[0],additional_parameters=43279,
       semantic_enabled=b.design.semantic,auxiliary=b.design.auxiliary,text=b.design.text,pair_loss=b.design.pair_loss,
       selected_connections_mean=float(counts.mean()),selected_connections_min=int(counts.min()),selected_connections_max=int(counts.max()),selected_active_fraction=float(selected.mean()),
       edge_weight_entropy_mean=float(edge_entropy.mean()),edge_weight_effective_connections_mean=float(np.exp(edge_entropy).mean()),
       train_class_margin_asd=float(margin[tr][y==0].mean()),train_class_margin_td=float(margin[tr][y==1].mean()),train_class_margin_separation=float(margin[tr][y==0].mean()-margin[tr][y==1].mean()),
       alpha_sem=0.,alpha_aux=0.,residual_abs_max=float(np.abs(delta).max()),residual_l2_mean=float(np.linalg.norm(delta,axis=1).mean()),
       semantic_residual_l2_mean=float(np.linalg.norm(sem,axis=1).mean()),auxiliary_residual_l2_mean=float(np.linalg.norm(aux,axis=1).mean()),residual_nonzero_fraction=float((np.linalg.norm(delta,axis=1)>1e-8).mean()),
       relative_residual_mean=float(total_relative.mean()),semantic_relative_residual_mean=float(sem_relative.mean()),auxiliary_relative_residual_mean=float(aux_relative.mean()),
       semantic_relative_residual_max=float(sem_relative.max()),auxiliary_relative_residual_max=float(aux_relative.max()),total_relative_residual_max=float(total_relative.max()),
       statistics_json=json.dumps(b.statistics_report,sort_keys=True),
       sampled_gradient_norms_json=json.dumps(tracker,sort_keys=True),aux_shuffle_fraction=float((npv(b.aux_perm)!=np.arange(871)).mean()),
       pair_loss_value=float(v['pair_loss'].detach()),reconstruction_loss_value=float(v['reconstruction_loss'].detach()),orthogonality_loss_value=float(v['orthogonality_loss'].detach()))
    assert row['residual_abs_max']<=.10+1e-7
    assert row['semantic_relative_residual_max']<.05+1e-6
    assert row['auxiliary_relative_residual_max']<.05+1e-6
    assert row['total_relative_residual_max']<.10+1e-6
    if b.design.semantic:
        sim=npv(v['local_similarity']); probs=np.exp(sim-sim.max(-1,keepdims=True)); probs/=probs.sum(-1,keepdims=True); ent=-(probs*np.log(np.maximum(probs,1e-12))).sum(-1)
        coarse,keys=b._text()
        query=torch.nn.functional.normalize(b.fc_query(v['h']),dim=-1,eps=1.e-8)
        global_sim=(query@coarse.T)/.2
        local_diff=sim[:,:,0]-sim[:,:,1]
        global_diff=npv(global_sim[:,0]-global_sim[:,1])
        row.update(local_similarity_mean=float(sim[selected].mean()) if selected.any() else 0.,
            local_similarity_asd_mean=float(sim[:,:,0][selected].mean()) if selected.any() else 0.,
            local_similarity_td_mean=float(sim[:,:,1][selected].mean()) if selected.any() else 0.,
            local_similarity_diff_mean=float(local_diff[selected].mean()) if selected.any() else 0.,
            global_similarity_asd_mean=float(npv(global_sim[:,0]).mean()),global_similarity_td_mean=float(npv(global_sim[:,1]).mean()),
            global_similarity_diff_mean=float(global_diff.mean()),local_attention_entropy=float(ent[selected].mean()) if selected.any() else 0.,
            global_semantic_std=float(npv(v['class_logits']).std()))
    else:
        row.update(local_similarity_mean=0.,local_similarity_asd_mean=0.,local_similarity_td_mean=0.,local_similarity_diff_mean=0.,
            global_similarity_asd_mean=0.,global_similarity_td_mean=0.,global_similarity_diff_mean=0.,local_attention_entropy=0.,global_semantic_std=0.)
    if b.design.fine:
        query=torch.nn.functional.normalize(b.fc_query(v['h']),dim=-1,eps=1.e-8)
        local_query=torch.nn.functional.normalize(query[:,None,:]+b.scalar_query(b.edge_scalars),dim=-1,eps=1.e-8)
        correct_keys=torch.stack((b.fine_asd_text[b.selected, b.direction_index[b.selected]],b.fine_td_text[b.selected]),dim=2)
        wrong_keys=torch.stack((b.fine_asd_text[b.roi_permutation[b.selected], b.direction_index[b.selected]],b.fine_td_text[b.roi_permutation[b.selected]]),dim=2)
        correct_cos=(local_query[:,:,None,:]*correct_keys).sum(-1).detach().cpu().numpy()
        wrong_cos=(local_query[:,:,None,:]*wrong_keys).sum(-1).detach().cpu().numpy()
        row['roi_match_text_cos_mean']=float(correct_cos[selected].mean()) if selected.any() else 0.
        row['wrong_roi_match_text_cos_mean']=float(wrong_cos[selected].mean()) if selected.any() else 0.
        row['roi_match_selected_text_cos_mean']=row['roi_match_text_cos_mean']
        row['wrong_roi_selected_text_cos_mean']=row['wrong_roi_match_text_cos_mean']
    else:
        row.update(roi_match_text_cos_mean=0.,wrong_roi_match_text_cos_mean=0.,roi_match_selected_text_cos_mean=0.,wrong_roi_selected_text_cos_mean=0.)
    details=v['auxiliary_details']
    if details:
        routing=npv(details['routing'])
    else:
        # NONE configurations have no auxiliary tensors; represent their
        # explicit null route so the diagnostic schema stays rectangular.
        routing=np.zeros((len(target),4),dtype=np.float32)
        routing[:,-1]=1.
    for j,name in enumerate(('PHENO','ANAT','FUNC')):
        if details:
            sh=npv(details[name]['shared']); pr=npv(details[name]['private']); rt=npv(details[name]['router'])
            shared_l2=float(np.linalg.norm(sh,axis=1).mean())
            private_l2=float(np.linalg.norm(pr,axis=1).mean())
            shared_private_cos=float(np.mean(np.sum(sh*pr,axis=1)/(np.linalg.norm(sh,axis=1)*np.linalg.norm(pr,axis=1)+1e-8)))
            router_mean=float(rt.mean()); router_std=float(rt.std())
        else:
            shared_l2=private_l2=shared_private_cos=router_mean=router_std=0.
        row[name+'_routing_mean']=float(routing[:,j].mean()); row[name+'_routing_std']=float(routing[:,j].std()); row[name+'_router_mean']=router_mean; row[name+'_router_std']=router_std
        row[name+'_shared_l2']=shared_l2; row[name+'_private_l2']=private_l2; row[name+'_shared_private_cos']=shared_private_cos
    row['null_routing_mean']=float(routing[:,-1].mean()); row['null_routing_std']=float(routing[:,-1].std())
    train_site=c['site'][tr]
    for j,name in enumerate(('PHENO','ANAT','FUNC')):
        row[name+'_routing_site_eta2']=eta_squared(routing[tr,j],train_site)
    for name,start,end,j in (('ANAT',48,54,1),('FUNC',54,64,2)):
        qc=c['x'][tr,start:end];route=routing[tr,j]
        row[name+'_qc_route_mean_abs_corr']=float(np.mean([abs(safe_corr(qc[:,i],route)) for i in range(qc.shape[1])]))
        row[name+'_qc_site_mean_eta2']=float(np.mean([eta_squared(qc[:,i],train_site) for i in range(qc.shape[1])]))
    # Probe loss gradients without changing weights.
    for m in modules(a):m.zero_grad(set_to_none=True)
    out=raw_forward(a); terms=b.loss_terms(a.targ); cls=c['model'].ClsLoss(out,a.targ,tr,a.weight)
    watch=list(b.fc_encoder.parameters())+list(b.fc_query.parameters())+list(b.aux_encoder.parameters())
    for name,loss in [('classification',cls),('semantic_loss',terms['global_loss']),('pair_loss',terms['pair_loss']),('reconstruction_loss',terms['reconstruction_loss']),('orthogonality_loss',terms['orthogonality_loss']),('relative_aux_delta',terms['relative_aux_delta'])]:
        if loss.requires_grad:
            gs=torch.autograd.grad(loss,watch,retain_graph=True,allow_unused=True)
            grad_norm=float(sum((g.detach().square().sum() for g in gs if g is not None),out.new_zeros(())).sqrt())
        else:
            grad_norm=0.
        row[name+'_value']=float(loss.detach()); row[name+'_gradient_norm']=grad_norm
    b.condition='REAL'; real=evaluate(a); row['counterfactual_json']=json.dumps(counterfactual(a,real),sort_keys=True)
    return row


def task_name(config,seed=0):return config+'_seed'+str(seed)
def task_checkpoint(config,seed=0):return RUNTIME/'state'/(task_name(config,seed)+'.pt')
def task_json(config,seed=0):return RUNTIME/'tasks'/(task_name(config,seed)+'.json')

def task_result(config,seed=0):
    p=task_json(config,seed)
    if not p.exists():return None
    r=jread(p); assert r['signature']==signature() and r['config']==config and int(r['seed'])==seed
    assert len(r['metrics'])==len(r['diagnostics']) and [int(x['fold']) for x in r['metrics']]==list(range(1,len(r['metrics'])+1))
    return r

def worker(config,seed):
    assert config in CONFIGS and seed==0
    m=verify_static(); assert m.get('preflight',{}).get('passed')
    gpu=GPU[config,seed]; assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ.get('E30_PHYSICAL_GPU')==str(gpu)
    c=context(); torch=c['torch']; name=task_name(config,seed); sig=signature(); (RUNTIME/'state').mkdir(parents=True,exist_ok=True); (RUNTIME/'tasks').mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'state'/(name+'.lock')).open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=task_result(config,seed); ck=task_checkpoint(config,seed); saved=checkpoint_read(ck) if ck.exists() else None
    if existing and existing['status']=='COMPLETE': print('SKIP_COMPLETE',name,flush=True); return
    if saved:
        assert saved['signature']==sig; result=saved['result']; restore_rng(saved['rng'])
    else:
        seed_rng(seed); result=dict(signature=sig,config=config,seed=seed,physical_gpu=gpu,status='RUNNING',metrics=[],diagnostics=[],seconds=0.,started_utc=utc(),resumes=0)
    if saved:result['resumes']=int(result.get('resumes',0))+1
    begin=time.monotonic(); oldseconds=float(result.get('seconds',0.)); stop=[False]
    def stopit(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,stopit); signal.signal(signal.SIGINT,stopit)
    print('TASK_START',name,'GPU',gpu,'resumed',saved is not None,flush=True)
    for fold in range(len(result['metrics'])+1,11):
        if stop[0]:return
        resumed=bool(saved and saved.get('phase')=='EPOCH' and int(saved['fold'])==fold)
        entry=rng(); a=make_agent(config,seed,fold); optimizer_audit(a)
        if resumed:
            load_models(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf']);restore_rng(saved['rng'])
            best_state=saved['best_state'];best=float(saved['best']);best_epoch=int(saved['best_epoch']);wait=int(saved['wait']);start=int(saved['epoch'])+1;initial=saved['initial'];initial_rng=saved['initial_rng'];tracker=saved.get('tracker',{})
        else: best_state=None;best=-1.;best_epoch=-1;wait=0;start=0;initial=parameter_hash(a);initial_rng=rng_hash(entry);tracker={}
        saved=None; fold_start=time.monotonic(); last_epoch=start-1
        for epoch in range(start,450):
            if resumed and epoch==start and wait>200:
                break
            if stop[0]:return
            with contextlib.redirect_stdout(io.StringIO()):
                a.run_epoch('simple-2')
                if epoch<2 or (epoch+1)%25==0:
                    for k,v in gradient_norms(a).items():tracker[k]=max(float(tracker.get(k,0)),float(v))
                _,acc=a.print_trn_acc('simple-2')
            acc=float(acc)
            if acc>best:best=acc;best_epoch=epoch;best_state=model_state(a);wait=0
            else:wait+=1
            last_epoch=epoch
            if (epoch+1)%25==0 or wait>200 or stop[0]:
                result['seconds']=oldseconds+time.monotonic()-begin
                checkpoint_write(ck,dict(signature=sig,phase='EPOCH',fold=fold,epoch=epoch,model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),best_state=best_state,best=best,best_epoch=best_epoch,wait=wait,initial=initial,initial_rng=initial_rng,tracker=tracker,result=result))
            if wait>200:break
        assert best_state is not None
        continuation=rng();load_models(a,best_state); out=evaluate(a); met=metrics(out,a.tst_idx); assert abs(met['acc']-best)<=1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=gpu,**met,best_epoch=best_epoch,epochs_executed=last_epoch+1,selected_val_acc=best,train_n=len(a.trn_idx),test_n=len(a.tst_idx),initial_trunk_sha256=initial,initial_rng_sha256=initial_rng,additional_parameters=43279,wall_seconds=time.monotonic()-fold_start)
        diag=diagnostics(a,config,fold,tracker); result['metrics'].append(row);result['diagnostics'].append(diag);restore_rng(continuation);result['seconds']=oldseconds+time.monotonic()-begin;result['status']='COMPLETE' if fold==10 else 'RUNNING';result['last_fold_seconds']=time.monotonic()-fold_start
        checkpoint_write(ck,dict(signature=sig,phase='BOUNDARY',fold=fold+1,rng=continuation,result=result));jwrite(task_json(config,seed),result)
        print('FOLD_DONE',name,fold,'ACC',met['acc'],'AUC',met['auc'],flush=True);del a,best_state,out;torch.cuda.empty_cache()
    print('TASK_COMPLETE',name,result['seconds'],flush=True)

def collect():
    groups=[]
    for c,s,_ in tasks():
        r=task_result(c,s)
        if r:groups.append(r)
    rows=[v for r in groups for v in r['metrics']]; diag=[v for r in groups for v in r['diagnostics']]
    return groups,rows,diag

def refresh():
    m=verify_static(); groups,rows,diag=collect()
    if rows:cwrite(RESULTS/'fold_metrics.csv',rows)
    if diag:cwrite(RESULTS/'diagnostics.csv',diag)
    index={r['config']:r for r in groups}
    for t in m['tasks']:
        q=index.get(t['config'])
        if q:t.update(status=q['status'],completed_folds=len(q['metrics']),seconds=q.get('seconds',0.),resumes=q.get('resumes',0))
    m.update(completed_folds=len(rows),completed_tasks=sum(q['status']=='COMPLETE' for q in groups),gpu_completed_folds={str(g):sum(int(r['physical_gpu'])==g for r in rows) for g in (0,1)},refreshed_utc=utc()); jwrite(MANIFEST,m); return groups,rows,diag

def load_results():
    if not (RESULTS/'fold_metrics.csv').is_file():return [],[]
    rows=readcsv(RESULTS/'fold_metrics.csv'); diag=readcsv(RESULTS/'diagnostics.csv')
    for r in rows:
        for k in ('seed','fold','physical_gpu','asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed'):r[k]=int(r[k])
        for k in METRICS:r[k]=float(r[k])
    return rows,diag

def capacity():
    import psutil
    slots=4
    if psutil.virtual_memory().available<32*1024**3 or os.getloadavg()[0]>min(os.cpu_count() or 1,32)*1.2:slots=2
    return slots

def schedule():
    pending=[j for j in tasks() if (task_result(j[0],j[1]) or {}).get('status')!='COMPLETE']; active=[]; failure=None
    while pending or active:
        slots=capacity()
        for c,s,g in list(pending):
            if sum(q['gpu']==g for q in active)>=slots:continue
            log=RUNTIME/(task_name(c,s)+'.log'); h=log.open('a'); p=subprocess.Popen([sys.executable,str(EXP/'run.py'),'_worker',c,str(s)],cwd=RUNTIME,env=worker_env(g),stdout=h,stderr=subprocess.STDOUT)
            active.append(dict(process=p,gpu=g,config=c,seed=s,handle=h,log=log));pending.remove((c,s,g));record_event(f'START {c} GPU{g} pid={p.pid} slots={slots}')
        time.sleep(5)
        for q in list(active):
            code=q['process'].poll()
            if code is None:continue
            q['handle'].close();active.remove(q);record_event(f'EXIT {q["config"]} code={code}')
            if code:
                failure={'config':q['config'],'gpu':q['gpu'],'returncode':code,'tail':q['log'].read_text()[-12000:]};break
            if not task_result(q['config'],q['seed']) or task_result(q['config'],q['seed'])['status']!='COMPLETE':failure={'config':q['config'],'error':'worker exited incomplete'};break
        refresh()
        if failure:
            for q in active:q['process'].terminate()
            for q in active:
                try:q['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:q['process'].kill()
                q['handle'].close()
            m=jread(MANIFEST);m['status']='PARTIAL';m['failures'].append(dict(time=utc(),stage='schedule',error=str(failure)));jwrite(MANIFEST,m);raise RuntimeError('formal worker failed: '+str(failure))

def aggregate(rows):
    refs=references(); allrows=rows+[r for r in refs]
    table={}
    for c in ('P0','E1')+tuple(CONFIGS):
        rr=[r for r in allrows if r['config']==c];
        if not rr:continue
        table[c]={k:dict(mean=float(np.mean([float(r[k]) for r in rr])),sd_population=float(np.std([float(r[k]) for r in rr],ddof=0))) for k in METRICS}
        table[c]['asd_fn']=int(sum(int(r['asd_fn']) for r in rr));table[c]['td_fp']=int(sum(int(r['td_fp']) for r in rr))
    return table

def pair_stats(rows,left,right):
    a={(int(r['seed']),int(r['fold'])):r for r in rows if r['config']==left}; b=references() if right in ('P0','E1') else {(int(r['seed']),int(r['fold'])):r for r in rows if r['config']==right}
    if right in ('P0','E1'): b={(int(r['seed']),int(r['fold'])):r for r in b if r['config']==right}
    out={}
    for k in METRICS:
        d=np.array([float(a[key][k])-float(b[key][k]) for key in sorted(a)])
        out[k]=dict(delta_pp=float(100*d.mean()),wins=int((d>1e-12).sum()),ties=int((abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()),per_fold_delta_pp=[float(100*x) for x in d])
    return {'left':left,'right':right,'metrics':out}

def scientific_decision(table,pairs,complete):
    if not complete:return dict(status='INCOMPLETE',go_no_go='NO-GO',criteria=[])
    lookup={(p['left'],p['right']):p for p in pairs}
    criteria=[('F1','P0'),('F1','Z0'),('F1','F0'),('F1','F2'),('F1','F3'),('F1','F4'),('F1','F5'),('F1','F6'),('F1','F7'),('F1','F8'),
              ('C1','P0'),('C1','Z0'),('C1','C0'),('C1','C2'),('C1','C3'),('C1','C4'),('C1','C5'),('C1','C6')]
    missing=[key for key in criteria if key not in lookup]
    if missing:return dict(status='INCOMPLETE',go_no_go='NO-GO',criteria=criteria,missing=missing)
    required=[('F1','P0'),('F1','Z0'),('F1','F0'),('F1','F2'),('F1','F3'),('F1','F4'),('F1','F5'),('F1','F6'),('F1','F7'),('F1','F8'),
              ('C1','P0'),('C1','Z0'),('C1','C0'),('C1','C2'),('C1','C3'),('C1','C4'),('C1','C5'),('C1','C6')]
    failures=[]
    for left,right in required:
        p=lookup[left,right]
        for metric in ('acc','auc'):
            q=p['metrics'][metric]
            if q['delta_pp']<=.1 or q['wins']<7:
                failures.append(dict(comparison=f'{left}-{right}',metric=metric,delta_pp=q['delta_pp'],wins=q['wins']))
    return dict(status='GO' if not failures else 'NO-GO',go_no_go='GO' if not failures else 'NO-GO',
                criteria=required,minimum_mean_gain_pp=.1,minimum_fold_wins=7,failures=failures,
                rationale='GO requires both principal groups to exceed their frozen baseline, semantic and auxiliary controls by more than 0.1 pp in ACC and hard-AUC, with at least 7/10 paired-fold wins; otherwise NO-GO.')

def report():
    m=verify_static(); rows,diag=load_results(); table=aggregate(rows); pairs=[]
    refs=('P0','E1')
    for left,right in [('F1','P0'),('F1','Z0'),('F1','F0'),('F1','F2'),('F1','F3'),('F1','F4'),('F1','F5'),('F1','F6'),('F1','F7'),('F1','F8'),('C1','P0'),('C1','Z0'),('C1','C0'),('C1','C2'),('C1','C3'),('C1','C4'),('C1','C5'),('C1','C6')]:
        if left in table and right in table:pairs.append(pair_stats(rows,left,right))
    status='COMPLETE' if len(rows)==180 else m.get('status','PARTIAL')
    judgment=scientific_decision(table,pairs,len(rows)==180)
    lines=['# 实验30：类别条件语义共享修复（Class-Conditional Semantic Shared Repair）','',f'状态：**{status}**；正式新训练折 **{len(rows)}/180**；seed=0，每组连续10折。','',
      '本实验保留原始GFR、CAM、RES、分类器、四图损失、test=val选模、预外层RFE/标准化与871人transductive传播。新Z/C/F组关闭原SPGS，在同一PLN位置加入冻结CLIP类别/连接教师及共享辅助修复；P0/E1只读历史参照。','',
      '|组|来源|ACC|AUC/BA|SEN-ASD|SPE-TD|ASD FN|TD FP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for c in ('P0','E1')+tuple(CONFIGS):
        if c not in table:continue
        t=table[c]; lines.append('|'+c+'|'+('HISTORICAL_READ_ONLY' if c in ('P0','E1') else 'NEW_SEED0_10_FOLDS')+'|'+'|'.join(f'{100*t[k]["mean"]:.5f}±{100*t[k]["sd_population"]:.5f}' for k in METRICS)+f'|{t["asd_fn"]}|{t["td_fp"]}|')
    lines += ['','## 预设配对（百分点，逐fold；胜/平/负）','', '|比较|指标|均值差|胜/平/负|逐fold差|','|---|---|---:|---|---|']
    for q in pairs:
        for k,v in q['metrics'].items():lines.append(f'|{q["left"]}−{q["right"]}|{k}|{v["delta_pp"]:+.5f}|{v["wins"]}/{v["ties"]}/{v["losses"]}|'+','.join(f'{x:+.3f}' for x in v['per_fold_delta_pp'])+'|')
    lines += ['','## 配置与机制','', '|组|fine|semantic|auxiliary|text|pair loss|','|---|---:|---:|---|---|---:|']
    for c in CONFIGS:
        d=DESIGNS[c];lines.append(f'|{c}|{int(d.fine)}|{int(d.semantic)}|{d.auxiliary}|{d.text}|{int(d.pair_loss)}|')
    if diag:
        lines += ['','## 机制诊断（逐折诊断的组均值）','', '|组|有效连接|权重熵|局部相似差|全局相似差|同ROI文本余弦|错ROI文本余弦|语义残差比|辅助残差比|null路由|pair|recon|orth|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c in CONFIGS:
            rr=[q for q in diag if q['config']==c]
            if not rr:continue
            def av(k):return float(np.mean([float(q[k]) for q in rr]))
            lines.append('|'+c+'|'+'|'.join(f'{av(k):.6f}' for k in ('selected_connections_mean','edge_weight_entropy_mean','local_similarity_diff_mean','global_similarity_diff_mean','roi_match_text_cos_mean','wrong_roi_match_text_cos_mean','semantic_relative_residual_mean','auxiliary_relative_residual_mean','null_routing_mean','pair_loss_value','reconstruction_loss_value','orthogonality_loss_value'))+'|')
        lines += ['','### 路由、shared/private与QC/站点关联','', '|组|PHENO路由|ANAT路由|FUNC路由|null|PHENO站点η²|ANAT站点η²|FUNC站点η²|ANAT QC路由相关|FUNC QC路由相关|ANAT QC站点η²|FUNC QC站点η²|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c in CONFIGS:
            rr=[q for q in diag if q['config']==c]
            if not rr:continue
            def av(k):return float(np.mean([float(q[k]) for q in rr]))
            fields=('PHENO_routing_mean','ANAT_routing_mean','FUNC_routing_mean','null_routing_mean','PHENO_routing_site_eta2','ANAT_routing_site_eta2','FUNC_routing_site_eta2','ANAT_qc_route_mean_abs_corr','FUNC_qc_route_mean_abs_corr','ANAT_qc_site_mean_eta2','FUNC_qc_site_mean_eta2')
            lines.append('|'+c+'|'+'|'.join(f'{av(k):.6f}' for k in fields)+'|')
        lines += ['','### 梯度范数','', '|组|FC Query|语义解码器|CORR共享投影|辅助Router|辅助Value|私有投影|重构器|','|---|---:|---:|---:|---:|---:|---:|---:|']
        gradient_keys=('fc_query','semantic_head','fc_shared','aux_router','aux_value','aux_private','aux_reconstruct')
        for c in CONFIGS:
            rr=[q for q in diag if q['config']==c]
            if not rr:continue
            vals={k:[] for k in gradient_keys}
            for q in rr:
                g=json.loads(q['sampled_gradient_norms_json'])
                for k in gradient_keys:vals[k].append(float(g.get(k,0.)))
            lines.append('|'+c+'|'+'|'.join(f'{np.mean(vals[k]):.6g}' for k in gradient_keys)+'|')
        lines += ['','## 反事实（同checkpoint，仅F1/C1诊断）','', '|组|条件|ASD概率变化|预测翻转|ΔACC|ΔAUC|','|---|---|---:|---:|---:|---:|']
        for c in ('C1','F1'):
            rr=[q for q in diag if q['config']==c]
            for mode in INTERVENTIONS:
                vals=[]
                for q in rr:
                    z=json.loads(q['counterfactual_json']); vals.append(next(v for v in z if v['condition']==mode))
                if vals: lines.append('|'+c+'|'+mode+'|'+f'{np.mean([v["probability_abs_change"] for v in vals]):.8f}|{np.mean([v["prediction_flip_fraction"] for v in vals]):.8f}|{100*np.mean([v["acc"]-next(w for w in json.loads(q["counterfactual_json"]) if w["condition"]=="REAL")["acc"] for q,v in zip(rr,vals)]):+.5f}|{100*np.mean([v["auc"]-next(w for w in json.loads(q["counterfactual_json"]) if w["condition"]=="REAL")["auc"] for q,v in zip(rr,vals)]):+.5f}|')
    lines += ['','## 科研判定','',
      f'预设决策：**{judgment["go_no_go"]}**。规则已冻结为F1、C1分别相对P0/Z0、无辅助主干、文本/ROI/方向/错配/共享/私有/密集/关闭配对对照，在ACC与硬标签AUC上平均提升均须大于0.1个百分点且至少7/10个配对折获胜；任一条件不满足即NO-GO。该规则是保守工程判读，不把十个共享fold视为独立样本。','',
      '容量收益看F1−F0和C1−C0；文本/连接/方向必要性看F1−F2/F3/F4及C1−C2；正确患者辅助配对看F1−F5及C1−C3；共享机制看F1−F6/F7/F8和C1−C4/C5/C6。完整配对值、胜平负和折差见上表。若真实文本与错误文本接近、正确辅助与错配辅助接近、或shared不优于private/dense，则相应机制不成立；路由兼容度只称模型内兼容，不称临床标定扫描可靠度。','',
      '当前指标是内部历史可比结果：test=val参与选模、RFE和标准化在外层前完成、GFR对871人作transductive传播，不能声称无泄漏泛化、临床因果或病灶发现。4个9维PLN token不是ROI；ANAT/FUNC是辅助质量输入。','',
      '完整决策对象：`'+json.dumps(judgment,ensure_ascii=False,sort_keys=True)+'`','',
      '## 完整逐折指标','', '|组|fold|ACC|AUC/BA|SEN|SPE|ASD FN|TD FP|best epoch|epochs|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['config']+'|'+str(r['fold'])+'|'+'|'.join(f'{100*float(r[k]):.6f}' for k in METRICS)+f'|{r["asd_fn"]}|{r["td_fp"]}|{r["best_epoch"]}|{r["epochs_executed"]}|')
    lines += ['','## 完整性','',f'源SHA：`{json.dumps(m["source_sha256"],ensure_ascii=False)}`',f'代码SHA：`{json.dumps(m["code_sha256"],ensure_ascii=False)}`',f'Prompt缓存SHA：`{m["prompt_cache_sha256"]}`',f'GPU折数：`{json.dumps(m.get("gpu_completed_folds",{}))}`',f'状态：`{m.get("handoff_status","PENDING")}`','不自动Git commit/push。']
    tmp=RESULTS/'report.md.tmp';tmp.write_text('\n'.join(lines)+'\n');os.replace(tmp,RESULTS/'report.md')
    return dict(table=table,pairs=pairs,rows=len(rows),diagnostics=len(diag),scientific=judgment['go_no_go'],judgment=judgment)

def append_handoff(summary,status):
    m=jread(MANIFEST); marker='<!-- EXPERIMENT30_CLASS_CONDITIONAL_SEMANTIC_SHARED_REPAIR_FINAL -->'; target=ROOT/'handoff.md'
    rows,diag=load_results();table=summary['table'];pairs={(p['left'],p['right']):p for p in summary['pairs']}
    text='\n\n'+marker+'\n## 实验三十：类别条件语义共享修复\n\n更新时间：'+utc()+'；STATUS='+status+'；正式训练折 '+str(summary['rows'])+'/180；任务 '+str(sum(t['status']=='COMPLETE' for t in m['tasks']))+'/18。\n\n'
    text+='目的：以raw[:,64:320] CORR为诊断Anchor，使用ASD/TD类别竞争语义，并只把与CORR共享的PHENO年龄/性别28维、ANAT六项、FUNC十项辅助表示通过含null的Router注入PLN-CORR。与原SPGS相比，Z/C/F全部跳过原SPGS；P0只读审计原SPGS。与实验29相比，本轮删除edge_id/连接ID InfoNCE和每坐标固定0.1残差，改为训练折可靠度Top-16、同连接ASD/TD文本对、类别CE和相对CAM-CORR范数约束。\n\n'
    text+='预注册18组（seed0，每组原始连续10折）：Z0/Z1；C0–C6（粗粒度文本，依次含无辅助、shared、类别混排、站点内辅助错配、dense、private、无配对损失）；F0–F8（细粒度真实ROI文本，含shared主组、错ROI、反方向、文本混排、站点内辅助错配、dense、private、无配对损失）。没有依据早期成绩删组或调整参数。\n\n'
    text+='GPU分配：GPU0完成 '+str(m.get('gpu_completed_folds',{}).get('0',0))+'/90折（Z0,C0,C2,C4,C6,F0,F2,F4,F6）；GPU1完成 '+str(m.get('gpu_completed_folds',{}).get('1',0))+'/90折（Z1,C1,C3,C5,F1,F3,F5,F7,F8）；正式墙钟 '+f'{float(m.get("formal_seconds",0.)):.2f}'+'秒。\n\n'
    text+='正式四指标及错误数（均值±十折population SD；AUC是原始one-hot硬预测口径）：\n\n|组|ACC %|Hard-AUC %|SEN-ASD %|SPE-TD %|ASD漏诊|TD误报|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for config in ('P0','E1')+tuple(CONFIGS):
        if config not in table:continue
        q=table[config];source='历史只读' if config in ('P0','E1') else '本轮'
        text+='|'+config+' ('+source+')|'+'|'.join(f'{100*q[k]["mean"]:.5f}±{100*q[k]["sd_population"]:.5f}' for k in METRICS)+f'|{q["asd_fn"]}|{q["td_fp"]}|\n'
    text+='\n关键预设配对差，单位百分点（ACC / Hard-AUC / SEN-ASD / SPE-TD）：\n\n|比较|ACC|Hard-AUC|SEN-ASD|SPE-TD|\n|---|---:|---:|---:|---:|\n'
    contrasts=[('F1','P0'),('F1','Z0'),('F1','F0'),('F1','F2'),('F1','F3'),('F1','F4'),('F1','F5'),('F1','F6'),('F1','F7'),('F1','F8'),('C1','P0'),('C1','Z0'),('C1','C0'),('C1','C2'),('C1','C3'),('C1','C4'),('C1','C5'),('C1','C6')]
    for left,right in contrasts:
        p=pairs.get((left,right))
        if p:text+='|'+left+'−'+right+'|'+'|'.join(f'{p["metrics"][k]["delta_pp"]:+.4f}' for k in METRICS)+'|\n'
    text+='\n机制概况（十折诊断均值）：\n\n|组|有效连接数|总残差/原CORR|语义残差/原CORR|辅助残差/原CORR|null路由|pair|reconstruction|orthogonality|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for config in CONFIGS:
        rr=[r for r in diag if r['config']==config]
        if not rr:continue
        def dav(key):return float(np.mean([float(r[key]) for r in rr]))
        text+='|'+config+'|'+'|'.join(f'{dav(k):.6f}' for k in ('selected_connections_mean','relative_residual_mean','semantic_relative_residual_mean','auxiliary_relative_residual_mean','null_routing_mean','pair_loss_value','reconstruction_loss_value','orthogonality_loss_value'))+'|\n'
    text+='\n预检和完整性：764条RN50冻结文本、252条有效连接、P0 bitwise parity、Z0零残差、训练/测试标签边界、5%/10%范数上界、私有梯度边界及25 epoch双Adam/RNG磁盘恢复通过。源、数据、历史参考、缓存及代码SHA见 '+str(EXP/'manifest.json')+'；根核心源码与实验1–29的保护哈希未变。\n\n'
    text+='科研判定：**'+summary['scientific']+'**。'+summary['judgment'].get('rationale','训练未完整，暂不能判定。')+' 只有完整重训对照可支持机制归因；同checkpoint反事实仅用于描述依赖。兼容度不是临床标定扫描质量，训练折效应方向不是病灶真值。\n\n'
    text+='限制：val=test参与Checkpoint选择，RFE和标准化在外层划分前完成，GFR仍在871人上transductive传播；本结果是原协议内部可比指标，不代表无泄漏独立泛化、临床因果或病灶发现。\n\n'
    text+='完整报告：'+str(RESULTS/'report.md')+'。未执行Git commit/push。\n'
    try:
        with (ROOT/'handoff.md.lock').open('a+b') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            with target.open('a+b',buffering=0) as f:
                f.seek(0); old=f.read(); assert hashlib.sha256(old[:m['original_handoff_size']]).hexdigest()==m['original_handoff_sha256']
                if marker.encode() not in old:f.write(text.encode());f.flush();os.fsync(f.fileno())
            m.update(handoff_appended=True,handoff_status='APPENDED_PREFIX_VERIFIED',handoff_updated_utc=utc(),handoff_sha256=sha(target))
    except OSError as e:
        q=Path('/tmp/experiment30_handoff_pending.md');q.write_text(text);m.update(handoff_appended=False,handoff_status='PENDING_ROOT_IO',handoff_pending=str(q),handoff_error=str(e))
    jwrite(MANIFEST,m)

def finish():
    groups,rows,diag=refresh(); assert len(groups)==18 and len(rows)==len(diag)==180
    assert all(g['status']=='COMPLETE' and len(g['metrics'])==10 for g in groups)
    assert [sum(int(r['physical_gpu'])==g for r in rows) for g in (0,1)]==[90,90]
    m=verify_static(); changed=[q for q,h in m['protected_files_sha256'].items() if not Path(q).is_file() or sha(q)!=h]; assert not changed
    m.update(status='COMPLETE',completed_utc=utc(),formal_seconds=float(time.monotonic()-float(m.get('formal_started_monotonic',time.monotonic()))));jwrite(MANIFEST,m)
    summary=report();append_handoff(summary,'COMPLETE');m=jread(MANIFEST);m['result_sha256']={q.name:sha(q) for q in RESULTS.glob('*') if q.is_file()};m['scientific_conclusion']=summary['scientific'];jwrite(MANIFEST,m)
    print('STATUS=COMPLETE 18 tasks / 180 folds / GPU0=90 GPU1=90',flush=True)

def cleanup_runtime():
    if RUNTIME.exists():shutil.rmtree(RUNTIME)
    m=json.loads(MANIFEST.read_text());m['cleanup_status']='REMOVED';m['cleanup_verified_utc']=utc();tmp=MANIFEST.with_name(MANIFEST.name+'.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,MANIFEST)

def all_tasks():
    prepare();m=verify_static()
    if m.get('status')=='COMPLETE':
        for n,h in m.get('result_sha256',{}).items():assert sha(RESULTS/n)==h
        print('ALREADY_COMPLETE verified; no retraining');return
    preflight();m=jread(MANIFEST);m['status']='RUNNING'
    if not m.get('formal_started_utc'):
        m['formal_started_utc']=utc();m['formal_started_monotonic']=time.monotonic()
    else:m['last_resume_utc']=utc()
    jwrite(MANIFEST,m);schedule();finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','preflight','all','resume','refresh','report','_smoke','_worker'));p.add_argument('config',nargs='?');p.add_argument('seed',nargs='?',type=int);args=p.parse_args()
    if args.command=='_smoke':smoke();return
    if args.command=='_worker':worker(args.config,int(args.seed));return
    controller=None
    if args.command in ('all','resume','preflight'):
        RUNTIME.mkdir(parents=True,exist_ok=True);controller=(RUNTIME/'controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.command=='prepare':prepare()
        elif args.command=='preflight':preflight()
        elif args.command in ('all','resume'):all_tasks()
        elif args.command=='refresh':refresh()
        elif args.command=='report':
            report();print((RESULTS/'report.md').read_text())
    except Exception as exc:
        import traceback;traceback.print_exc()
        if MANIFEST.exists():
            m=jread(MANIFEST);m['status']='PARTIAL' if m.get('formal_started_utc') else 'BLOCKED';m.setdefault('failures',[]).append(dict(time=utc(),stage=args.command,error=str(exc)));jwrite(MANIFEST,m)
            try:report()
            except Exception:pass
        raise
    finally:
        if controller is not None:controller.close()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()).get('status')=='COMPLETE':cleanup_runtime()

if __name__=='__main__':main()

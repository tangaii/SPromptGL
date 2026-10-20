#!/usr/bin/env python3
"""Experiment28 supplement E10. Only three new continuous-ten-fold tasks.

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
from e28_method import CONFIGS,SEEDS,INTERVENTIONS,RULES,tasks
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
HIST=ROOT/'experiment/experiment28_imaging_anchored_finegrained_semantic_alignment'
E10='E10_COARSE_FULL'
CONTRASTS=((E10,'E2_OLD_FC'),('E4_FINE_FULL','E3_FINE_FC'),('E3_FINE_FC','E2_OLD_FC'),('E4_FINE_FULL',E10),(E10,'E3_FINE_FC'),(E10,'E4_FINE_FULL'))
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

def task_name(c,s):return c+'_seed'+str(s)

def task_json(c,s):return RUNTIME/'tasks'/(task_name(c,s)+'.json')

def task_checkpoint(c,s):return RUNTIME/'state'/(task_name(c,s)+'.pt')

def task_result(c,s):
    p=task_json(c,s)
    if not p.exists():return None
    r=jread(p);assert r['signature']==signature() and r['config']==c and r['seed']==s
    assert len({r['fold'] for r in r['metrics']})==len(r['metrics'])
    if r['status']=='COMPLETE':assert len(r['metrics'])==len(r['diagnostics'])==10
    return r

def worker(config,seed):
    assert config==E10 and seed in SEEDS
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
        if fold==1 and not resumed:
            reference=[r for r in readcsv(HIST/'results/fold_metrics.csv') if r['config']=='E4_FINE_FULL' and int(r['seed'])==seed and int(r['fold'])==1][0]
            assert initial==reference['initial_trunk_sha256'],'E4 first-fold trunk/RNG initialization differs'
            assert init_rng_hash==reference['initial_rng_sha256'],'E4 first-fold RNG entry differs'
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
        cf=[]
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

def worker_env(gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),E28_PHYSICAL_GPU=str(gpu),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    return env

def capacity():
    import psutil
    slots=2
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
    smoke_extra(c,torch,checks)

def verify_static():
    m=jread(MANIFEST)
    assert core_hashes()==m['source_sha256'],'Root source changed'
    assert code_hashes()==m['code_sha256'],'Supplement code changed after freeze'
    assert sha(CACHE)==m['prompt_cache_sha256']==sha(HIST/'prompt_banks.npz')
    for p,h in m['data_sha256'].items():assert sha(p)==h,p
    for n,h in m['historical_e28_sha256'].items():assert sha(HIST/n)==h,n
    assert sha(BASELINE)==m['reference_p0_sha256']
    return m

def prepare():
    if MANIFEST.exists():verify_static();return
    RESULTS.mkdir(exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    old=jread(HIST/'manifest.json');assert old['status']=='COMPLETE' and old['completed_folds']==300
    assert core_hashes()==old['source_sha256']
    assert sha(CACHE)==old['prompt_cache_sha256']
    for n in ('e28_prompts.py','e28_integration.py'):
        assert (EXP/n).read_bytes()==(HIST/n).read_bytes()
    rows=readcsv(HIST/'results/fold_metrics.csv')
    assert len(rows)==300 and len({(r['config'],r['seed'],r['fold']) for r in rows})==300
    info=data();historical_p0();e28_integration.emit(ROOT,SOURCE,EXP/'integration.patch')
    # Current hashes are frozen now, including every existing Exp28 artifact.
    protected={}
    paths={Path(p) for p in old['protected_files_sha256'] if Path(p).is_file()}
    paths.update(p for p in HIST.rglob('*') if p.is_file())
    for p in sorted(paths):protected[str(p)]=sha(p)
    hand=(ROOT/'handoff.md').read_bytes()
    remote=jread(EXP/'.audit_remote.json')
    fixes=(EXP/'.method_diff.txt').read_text()
    m=dict(experiment='28 supplement E10 coarse full',status='PREPARED',started_utc=utc(),
        source_local_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_remote_head=remote['remote'],remote_verified_files_sha256=remote['files'],source_sha256=core_hashes(),
        data_sha256=old['data_sha256'],reference_p0_sha256=sha(BASELINE),
        historical_e28_sha256={str(p.relative_to(HIST)):sha(p) for p in HIST.rglob('*') if p.is_file()},
        protected_files_sha256=protected,prompt_cache_sha256=sha(CACHE),
        prompt_json_sha256=sha(HIST/'prompt_banks.json'),clip_weights_sha256=old['clip_weights_sha256'],
        original_handoff_size=len(hand),original_handoff_sha256=hashlib.sha256(hand).hexdigest(),
        design={E10:dict(spgs=False,res=True,teacher='EXACT_E2_COARSE',full_aux=True,semantic_weight=.05,
             new_parameters=28517,student='anchor unchanged from E4',residual='coordinate absolute <=0.10, not relative 10%')},
        rules=RULES,protocol=PROTOCOL,code_sha256=code_hashes(),
        tasks=[dict(config=c,seed=s,gpu=g,folds=10,status='PENDING') for c,s,g in tasks()],
        target_folds=30,historical_folds_read_only=300,completed_folds=0,gpu_fold_targets={'0':20,'1':10},
        method_diff=fixes,runner_provenance=dict(path=str(HIST/'run.py'),sha256=sha(HIST/'run.py'),
            reused='data, cold RN50 initialization, make_agent, train/eval loop, metrics, two-Adam/RNG checkpoint, diagnostics, scheduler'),
        historical_scientific_conclusion='NO-GO unchanged',failures=[],handoff_appended=False,
        paper=dict(path=str(ROOT/'2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf'),
            sha256=sha(ROOT/'2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf'),
            review='Fig1/Section2.1-2.2/Eq1-7 and ablations read; actual GFR i=1 overwrite, ALoss exclusion, four graph mean, latent tokens, RES verified. Paper hyperparameters differ; historical repository protocol authoritative.'))
    jwrite(MANIFEST,m)
    for p in (EXP/'.audit_remote.json',EXP/'.method_diff.txt'):p.unlink()
    record_event('PREPARED only E10 seeds0/1/2; historical300 read-only; cache copied byte-for-byte')

def cpu_checks():
    import importlib.util,torch
    from e28_method import ImagingAnchoredAlignment,teacher,training_statistics
    spec=importlib.util.spec_from_file_location('_original_e28_method',HIST/'e28_method.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    from e28_method import self_test
    self_test();c=data();cache=e28_prompts.load(CACHE);checks={}
    for fold,(tr,te) in enumerate(c['folds'],1):
        args=dict(raw=c['x'],train=tr,test=te,y_train=c['y'][tr],site=c['site'],pheno_indices=c['pheno_indices'],cache=cache,fold=fold)
        torch.manual_seed(280001);a=ImagingAnchoredAlignment('E10_COARSE_FULL',**args)
        torch.manual_seed(280001);b=old.ImagingAnchoredAlignment('E2_OLD_FC',**args)
        torch.manual_seed(280001);e4=old.ImagingAnchoredAlignment('E4_FINE_FULL',**args)
        for n in ('teacher_target','teacher_active','semantic_edge_weights'):
            assert torch.equal(getattr(a,n),getattr(b,n)),(fold,n)
        assert a.full_aux and e4.full_aux and not b.full_aux
        assert list(a.named_parameters())[0][0]==list(e4.named_parameters())[0][0]
        assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in e4.parameters())==28517
        for (n,p),(nn,pp) in zip(a.named_parameters(),e4.named_parameters()):assert n==nn and torch.equal(p,pp)
        # All historical mathematical functions, including anchor loss, are unchanged.
        for n in ('training_statistics','teacher'):
            assert ast_source(EXP/'e28_method.py',n)==ast_source(HIST/'e28_method.py',n)
        for n in ('forward','semantic_loss'):
            assert ast_source(EXP/'e28_method.py',n,'ImagingAnchoredAlignment')==ast_source(HIST/'e28_method.py',n,'ImagingAnchoredAlignment')
        cam=torch.linspace(-1,1,871*4*36).reshape(871,4,36);raw=torch.tensor(c['x'],dtype=torch.float32)
        assert torch.equal(a(cam,raw),cam)
        b(cam,raw);assert torch.equal(a.last['student'],b.last['student']) and torch.equal(a.semantic_loss(),b.semantic_loss())
        with torch.no_grad():a.alpha.fill_(.5);e4.alpha.fill_(.5)
        assert torch.equal(a(cam,raw),e4(cam,raw)),'Only teacher may differ'
        # Existing E4 implementation is exactly reproduced; new E10 uses E2 targets.
        assert a.last['conditional'].norm()>0
        original_weights=a.semantic_edge_weights.clone()
        assert torch.count_nonzero(original_weights[:,[195,225,232,236]])==0
    checks['all10fold_E10_E2_teacher_active_weights_loss']='BITWISE_EXACT'
    checks['all10fold_E10_E4_parameters_aux_forward_nonzero_alpha']='BITWISE_EXACT'
    checks['teacher_forward_semantic_loss_source']='BYTE_IDENTICAL_FUNCTIONS'
    checks['frozen_coarse_cache']='REUSED_NO_CLIP_ENCODING'
    return checks

def ast_source(path,name,cls=None):
    import ast
    s=Path(path).read_text();tree=ast.parse(s)
    nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    n=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(s,n)

def smoke_extra(c,torch,checks):
    seed_rng(1);entry=rng();e1=make_agent('E1_NO_SPGS',1,1)
    expected=evaluate(e1);trunk=parameter_hash(e1);after=rng_hash(rng())
    agents={}
    for cfg in ('E2_OLD_FC','E4_FINE_FULL',E10):
        restore_rng(entry);a=make_agent(cfg,1,1);agents[cfg]=a
        assert parameter_hash(a)==trunk and rng_hash(rng())==after
        assert torch.equal(evaluate(a),expected) and optimizer_audit(a)==1491474
        assert float(a.ModalFusion.ia_fdsa.alpha)==0
    a=agents[E10];b=agents['E4_FINE_FULL'];e2=agents['E2_OLD_FC']
    assert parameter_hash(a,True)==parameter_hash(b,True)==parameter_hash(e2,True)
    recursive_equal(a.optimizer_MP0.state_dict(),b.optimizer_MP0.state_dict())
    recursive_equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict())
    assert torch.equal(a.ModalFusion.ia_fdsa.semantic_loss(),e2.ModalFusion.ia_fdsa.semantic_loss())
    checks['E10_E4_structure_original_new_parameters_optimizer_rng']='BITWISE_EXACT'
    checks['E10_E1_E4_zero_alpha_output']='BITWISE_EXACT'
    checks['E10_E2_same_student_semantic_loss']='BITWISE_EXACT'
    checks['total_parameters']=1491474;checks['additional_parameters']=28517
    run_entry=rng()
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    expected_state=model_state(a);expected_rng=rng();grad=gradient_norms(a)
    assert grad['alpha']>0 and grad['fc_encoder']>0 and grad['context_encoder']>0
    assert grad['decode']==grad['demo_encoder']==grad['anat_gate']==grad['func_gate']==0
    y=c['y'].copy();y[a.tst_idx]=1-y[a.tst_idx]
    restore_rng(entry);altered=make_agent(E10,1,1,labels=y)
    for n in ('teacher_target','teacher_active','semantic_edge_weights'):
        assert torch.equal(getattr(a.ModalFusion.ia_fdsa,n),getattr(altered.ModalFusion.ia_fdsa,n))
    restore_rng(run_entry)
    with contextlib.redirect_stdout(io.StringIO()):altered.run_epoch('simple-2')
    state_equal(model_state(altered),expected_state);recursive_equal(grad,gradient_norms(altered))
    recursive_equal(a.optimizer_MP0.state_dict(),altered.optimizer_MP0.state_dict())
    recursive_equal(a.optimizer_MF.state_dict(),altered.optimizer_MF.state_dict())
    assert rng_hash(rng())==rng_hash(expected_rng)
    checks['heldout_label_flip_full_train_update_models_both_adams_rng']='BITWISE_EXACT'
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    grad2=gradient_norms(a)
    for n in ('alpha','fc_encoder','context_encoder','demo_encoder','anat_gate','func_gate','decode'):assert grad2[n]>0,(n,grad2)
    checks['first_step_gradient_norms']=grad;checks['second_step_gradient_norms']=grad2
    real=evaluate(a);branch=a.ModalFusion.ia_fdsa
    assert branch.full_aux and branch.last['conditional'].norm()>0
    assert branch.last['delta'].abs().max()<=.1
    branch.condition='AUX_OFF';auxoff=evaluate(a);branch.condition='REAL'
    assert torch.equal(real,evaluate(a)) and (real-auxoff).abs().max()>0
    checks['auxiliary_conditional_reaches_classification']='PASS'
    path=RUNTIME/'preflight_resume.pt'
    state=dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng())
    checkpoint_write(path,state)
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
    expected_state=model_state(a);expected_rng=rng();out=evaluate(a)
    saved=checkpoint_read(path);restored=make_agent(E10,1,1)
    load_models(restored,saved['model']);restored.optimizer_MP0.load_state_dict(saved['mp']);restored.optimizer_MF.load_state_dict(saved['mf']);restore_rng(saved['rng'])
    with contextlib.redirect_stdout(io.StringIO()):restored.run_epoch('simple-2')
    state_equal(model_state(restored),expected_state)
    recursive_equal(a.optimizer_MP0.state_dict(),restored.optimizer_MP0.state_dict())
    recursive_equal(a.optimizer_MF.state_dict(),restored.optimizer_MF.state_dict())
    assert rng_hash(rng())==rng_hash(expected_rng) and torch.equal(evaluate(restored),out)
    path.unlink();checks['disk_checkpoint_model_two_adams_full_rng_next_update_prediction']='BITWISE_EXACT'
    boundary=rng();n=make_agent(E10,1,2);next_state=model_state(n);next_rng=rng()
    restore_rng(boundary);repeat=make_agent(E10,1,2)
    state_equal(model_state(repeat),next_state);assert rng_hash(next_rng)==rng_hash(rng())
    checks['next_fold_initialization_rng_resume']='BITWISE_EXACT'
    checks['device']=str(torch.cuda.get_device_properties(0))
    jwrite(RUNTIME/'preflight_checks.json',checks)
    print('E10_GPU_PREFLIGHT_PASS',json.dumps(checks),flush=True)

def preflight():
    prepare();m=verify_static()
    if m.get('preflight',{}).get('passed'):return
    cpu=cpu_checks()
    log=RUNTIME/'preflight.log'
    with log.open('w') as f:
        result=subprocess.run([sys.executable,str(EXP/'run.py'),'_smoke'],cwd=RUNTIME,env=worker_env(0),stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,'GPU preflight failed: '+log.read_text()[-6000:]
    checks=jread(RUNTIME/'preflight_checks.json');checks.update(cpu)
    result=subprocess.check_output([sys.executable,'-c',"import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))"],env=worker_env(1),text=True)
    checks['GPU1_visible_logical0']=result.strip()
    verify_static();m=jread(MANIFEST);m.update(status='PREFLIGHT_PASS',preflight=dict(passed=True,completed_utc=utc(),checks=checks));jwrite(MANIFEST,m)
    record_event('PREFLIGHT_PASS all10 teacher/structure; real GPU gradients, label boundary, two-Adam/RNG resume')

def load_results():
    if (RESULTS/'fold_metrics.csv').exists() and jread(MANIFEST)['status']=='COMPLETE':
        rows=readcsv(RESULTS/'fold_metrics.csv');diag=readcsv(RESULTS/'diagnostics.csv')
        for r in rows:
            for k in ('seed','fold','physical_gpu','asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed'):r[k]=int(r[k])
            for k in METRICS:r[k]=float(r[k])
        return rows,diag
    _,rows,diag,_=collect();return rows,diag

def report():
    m=verify_static();new,diag=load_results();historical=readcsv(HIST/'results/fold_metrics.csv')
    rows=historical+new;table={}
    for c in CONFIGS:
        rr=[r for r in rows if r['config']==c]
        if len(rr)==30:table[c]={k:summarize(rr,k) for k in METRICS}
    pairs=contrasts(rows);index={(r['config'],int(r['seed']),int(r['fold'])):r for r in rows}
    interaction={}
    if len(new)==30:
        for k in METRICS:
            diff=np.array([float(index['E4_FINE_FULL',s,f][k])-float(index['E3_FINE_FC',s,f][k])-float(index[E10,s,f][k])+float(index['E2_OLD_FC',s,f][k]) for s in SEEDS for f in range(1,11)])
            alternate=np.array([float(index['E4_FINE_FULL',s,f][k])-float(index[E10,s,f][k])-float(index['E3_FINE_FC',s,f][k])+float(index['E2_OLD_FC',s,f][k]) for s in SEEDS for f in range(1,11)])
            assert np.allclose(diff,alternate,atol=1e-14)
            interaction[k]=dict(delta_pp=float(diff.mean()*100),per_seed_delta_pp=(diff.reshape(3,10).mean(1)*100).tolist(),positive=int((diff>1e-12).sum()),zero=int((abs(diff)<=1e-12).sum()),negative=int((diff< -1e-12).sum()))
    counts={c:{str(s):dict(asd_fn=sum(int(r['asd_fn']) for r in rows if r['config']==c and int(r['seed'])==s),td_fp=sum(int(r['td_fp']) for r in rows if r['config']==c and int(r['seed'])==s)) for s in SEEDS} for c in table}
    hd=readcsv(HIST/'results/diagnostics.csv')
    combined_diag=hd+diag
    dk=('teacher_valid_train','teacher_valid_test','teacher_variance','teacher_class_cosine','teacher_student_cosine_train','teacher_student_cosine_test','alpha','residual_l2_mean','residual_relative_mean','auxiliary_norm_mean','auxiliary_anchor_ratio','qa_mean','qa_patient_std','qa_feature_std_mean','qf_mean','qf_patient_std','qf_feature_std_mean','semantic_gradient_anchor','classification_gradient_anchor')
    ds={c:{k:float(np.mean([float(r[k]) for r in combined_diag if r['config']==c])) for k in dk} for c in ('E2_OLD_FC','E3_FINE_FC','E4_FINE_FULL',E10) if any(r['config']==c for r in combined_diag)}
    for c,d in ds.items():
        rr=[r for r in combined_diag if r['config']==c]
        for g in ('qa','qf'):
            d[g+'_patient_variance_mean']=float(np.mean([float(r[g+'_patient_std'])**2 for r in rr]))
        grads=[json.loads(r['sampled_gradient_norms_json']) for r in rr]
        d['sampled_max_gradient_norms_mean']={k:float(np.mean([v.get(k,0.) for v in grads])) for k in ('alpha','fc_encoder','context_encoder','demo_encoder','anat_gate','func_gate','decode')}
    def consistent(a,b):
        pp=next(p['metrics'] for p in pairs if (p['left'],p['right'])==(a,b))
        return all(pp[k]['delta_pp']>0 and all(v>0 for v in pp[k]['per_seed_delta_pp']) for k in ('ba','probability_auc','sensitivity_asd'))
    decisions=dict(aux_coarse_supported=consistent(E10,'E2_OLD_FC') if len(new)==30 else False,
        aux_fine_supported=consistent('E4_FINE_FULL','E3_FINE_FC') if len(new)==30 else False,
        fine_with_full_aux_supported=consistent('E4_FINE_FULL',E10) if len(new)==30 else False,
        historical_E4_conclusion='NO-GO UNCHANGED',criterion='BA/probAUC/SEN each positive in mean and every seed; descriptive internal consistency only')
    summary=dict(status=m['status'],new_training_folds=len(new),historical_read_only_folds=300,
        metrics=table,provenance={c:'NEW_30_FOLDS' if c==E10 else 'HISTORICAL_READ_ONLY' for c in table},
        contrasts=pairs,interaction=interaction,case_counts=counts,diagnostics=ds,decisions=decisions,
        tasks=m['tasks'],runtime_seconds=m.get('wall_seconds'),preflight=m.get('preflight'),protection=m.get('protection'),
        handoff_status=m.get('handoff_status','PENDING'),delivery_status=m.get('delivery_status','TRAINING_COMPLETE'),
        scientific_conclusion='SUPPLEMENT_DESCRIPTIVE_ONLY; ORIGINAL_E4_NO_GO_UNCHANGED')
    jwrite(RESULTS/'summary.json',summary)
    lines=['# 实验28补充对照E10：粗粒度Prompt＋完整辅助条件修正','',
      f'状态 **{m["status"]}**。仅新增 **{len(new)}/30折、3个任务**；历史300折只读。执行目录：`{EXP}`。','',
      '本轮补全Prompt粒度×辅助条件修正2×2对照。E10与E4完整数学相同，只把fine教师换为原E2冻结coarse教师；不重编码CLIP、不改Prompt、student仍为anchor。',
      'E2的单一粗粒度向量来自原异常四句投影后均值归一化，并广播到252×2个位置；原正常四句只参与共享SVD基底。该教师理论上近常量，不能把其正则效果解释为独立疾病知识。','',
      '## 历史与新增全部11组','',
      '先分别计算每seed十折均值，再三seed均值±population SD，单位%。Hard-AUC=BA。','',
      '|组|来源|ACC|BA/Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|F1-ASD|','|---|---|---:|---:|---:|---:|---:|---:|']
    six=[k for k in METRICS if k!='hard_auc']
    for c,t in table.items():lines.append('|'+c+'|'+summary['provenance'][c]+'|'+'|'.join(f'{t[k]["mean"]*100:.5f}±{t[k]["sd_population"]*100:.5f}' for k in six)+'|')
    lines+=['','## E10三个seed及病例事件','', '|seed|ACC|BA|Prob-AUC|SEN|SPE|F1|ASD FN|TD FP|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    if E10 in table:
        for s in SEEDS:lines.append('|'+str(s)+'|'+'|'.join(f'{table[E10][k]["per_seed"][s]*100:.5f}' for k in six)+f'|{counts[E10][str(s)]["asd_fn"]}|{counts[E10][str(s)]["td_fp"]}|')
    lines+=['','## 全部指定配对与逐fold胜/平/负','', '|比较|指标|均差pp|seed0/1/2差pp|W/T/L|','|---|---|---:|---|---|']
    for p in pairs:
        for k in METRICS:
            q=p['metrics'][k];lines.append(f'|{p["left"]} − {p["right"]}|{k}|{q["delta_pp"]:+.5f}|'+','.join(f'{v:+.5f}' for v in q['per_seed_delta_pp'])+f'|{q["wins"]}/{q["ties"]}/{q["losses"]}|')
    lines+=['','## 2×2交互描述','',
        '交互=(E4−E3)−(E10−E2)=(E4−E10)−(E3−E2)。逐fold两公式数值一致；这是因子交互的内部描述，不是医学因果协同证明。','',
        '|指标|交互均值pp|seed0/1/2 pp|正/零/负fold|','|---|---:|---|---|']
    for k,q in interaction.items():lines.append(f'|{k}|{q["delta_pp"]:+.5f}|'+','.join(f'{v:+.5f}' for v in q['per_seed_delta_pp'])+f'|{q["positive"]}/{q["zero"]}/{q["negative"]}|')
    lines+=['','## 教师、辅助条件与有效梯度','', '|组|有效train/test比例|教师方差|ASD/TD中心cos|student-teacher train/test cos|辅助/anchor|alpha|残差L2|','|---|---|---:|---:|---|---:|---:|---:|']
    for c,d in ds.items():lines.append(f'|{c}|{d["teacher_valid_train"]:.6f}/{d["teacher_valid_test"]:.6f}|{d["teacher_variance"]:.9g}|{d["teacher_class_cosine"]:.9g}|{d["teacher_student_cosine_train"]:.6f}/{d["teacher_student_cosine_test"]:.6f}|{d["auxiliary_anchor_ratio"]:.7f}|{d["alpha"]:.7f}|{d["residual_l2_mean"]:.7f}|')
    if E10 in ds:
        d=ds[E10];lines+=['',f'E10 ANAT门均值{d["qa_mean"]:.7f}、患者均值SD的折平均{d["qa_patient_std"]:.7f}、患者均值方差的折平均{d["qa_patient_variance_mean"]:.9f}、逐维SD均值{d["qa_feature_std_mean"]:.7f}；FUNC门均值{d["qf_mean"]:.7f}、患者均值SD的折平均{d["qf_patient_std"]:.7f}、患者均值方差的折平均{d["qf_patient_variance_mean"]:.9f}、逐维SD均值{d["qf_feature_std_mean"]:.7f}。方差先逐fold计算SD平方，再跨fold平均，不能平方平均SD代替。',
        f'辅助修正L2均值{d["auxiliary_norm_mean"]:.7f}，相对anchor{d["auxiliary_anchor_ratio"]:.7f}。语义梯度到anchor为{d["semantic_gradient_anchor"]:.7g}，分类梯度{d["classification_gradient_anchor"]:.7g}。',
        '各模块训练采样最大梯度范数的折平均：`'+json.dumps(d['sampled_max_gradient_norms_mean'])+'`。',
        '逐fold详细各模块梯度、alpha、残差非零率、坐标上界、QA、训练统计SHA均保存在diagnostics.csv。alpha=0首步decoder/辅助梯度为零符合原E4规律；第二步梯度连通。']
    lines+=['','## 结论A–D','',
        f'A 辅助条件净价值：粗粒度E10−E2满足三seed BA/Prob-AUC/SEN一致正向：{decisions["aux_coarse_supported"]}；细粒度E4−E3：{decisions["aux_fine_supported"]}。各指标和seed差如上，不以单一ACC决定。',
        f'B 细粒度独立价值：相同完整辅助结构的E4−E10满足上述一致门槛：{decisions["fine_with_full_aux_supported"]}。若不满足，仅为阴性或指标权衡，不能声称细粒度全面有效。',
        'C 交互方向按上表逐指标和逐seed判读；三seed重复同一队列，不视为30个独立外部样本，不报告伪独立P值。',
        'D 原E4 ACC91.61442/BA91.42719/ProbAUC93.19882%，仍低于P0 91.69105/91.53935/93.33520%及历史Exp9 E9 91.88262/91.69151/93.82272%。原实验28 NO-GO判断保持。本轮历史E9_WRONG_AUX_PAIR不是历史Exp9 E9。',
        'E2/E10粗粒度教师各有效患者理论方向一致，其近零方差及中心cos≈1不提供患者类别区分。若E10增益出现，应优先解释为辅助融合、优化或常量语义正则线索，不证明医学知识必要性。','',
        '## 工程、协议与完整性','',
        '原GFR i==1覆盖、阈值.9、ALoss不加入simple-2损失、四图正则平均、原RES/分类器及两Adam保持。SPGS按E4关闭，原MLP继续注册但不参与前向。新分支28517参数、总参数1491474，新增wd=0、原wd=.11、lr=.0038。alpha=0与E1逐值相同。残差每坐标≤.10，不是相对CAM范数10%。',
        '871人403 ASD/468 TD，输入48/6/10/256。原10fold(seed划分10)、任务seed0/1/2，450epoch、wait>200、首次严格testACC最高Checkpoint。每任务只播种一次，连续10fold；早停后后续RNG自然分叉，不声称E4/E10各fold相同初始化。首fold入口RNG/主干hash与历史E4精确核对。',
        '保留test=val选模、外层CV前RFE/全体标准化、全871人transductive传播；多轮探索和测试参与选模使结果偏乐观，不能宣称独立无泄漏泛化。潜在4×36与4×4×9不是ROI；ANAT/FUNC为质量特征。',
        '预检：`'+json.dumps(m.get('preflight',{}),ensure_ascii=False)+'`。',
        '任务分配/耗时：`'+json.dumps(m['tasks'],ensure_ascii=False)+'`；墙钟秒：'+str(m.get('wall_seconds'))+'。',
        '历史/核心保护：`'+json.dumps(m.get('protection',{}),ensure_ascii=False)+'`；缓存SHA：`'+m['prompt_cache_sha256']+'`；源码SHA：`'+json.dumps(m['source_sha256'])+'`；实验源码SHA：`'+json.dumps(m['code_sha256'])+'`。',
        '不重训历史300fold；新fold_metrics/diagnostics仅E10各30条，历史表按原CSV读取。完成后清理本实验.runtime，不自动Git提交/推送。','',
        '下一步建议：维持原强基线及实验28止损判断；若局部交互值得关注，仅在训练内选模和独立队列验证，不据本轮测试成绩改Prompt或搜损失权重。','']
    if len(new)==30:
        lines+=['## 实测机制判读与交付状态','',
          'E10−E2的ACC三个seed均改善(+0.22989/+0.22989/+0.34483pp)，BA也三个seed改善；但Prob-AUC两个seed降低，均值−0.32976pp。因此辅助修正在粗粒度条件下有硬分类阈值指标收益，未产生一致排序收益。',
          '相同辅助结构的E4−E10：ACC两个seed降低、一个持平；BA两个seed降低；Prob-AUC两个seed上升、一个明显下降，均差−0.02361pp。ASD灵敏度均差+0.08130pp但方向不一致。没有支持细粒度教师全面优于粗粒度教师。',
          '交互在ACC、BA、F1和特异度三个seed均为负；概率AUC交互均值为正但seed1为负，灵敏度交互也不同向。没有稳定的正向因子交互。',
          'E10相对历史P0 ACC+0.11494pp、BA+0.06227pp，但Prob-AUC−0.11277pp、ASD灵敏度−0.73577pp；累计漏诊134次，P0为125次（重复三个seed的病例事件）。E10也未超过历史Exp9 E9的ACC/BA/Prob-AUC。单一最高ACC不能替代疾病语义必要性检验。',
          'E10教师与E2完全一致，类别中心cos≈1、患者方差≈2.05e−11，属近常量语义方向；其优化或辅助融合增益不等于独立医学知识。新分支残差L2≈0.324且辅助/anchor≈22.79%，不是近零响应失败。',
          'handoff状态：`'+m.get('handoff_status','PENDING')+'`；整体交付状态：`'+m.get('delivery_status','TRAINING_COMPLETE')+'`。',
          '若为PENDING_ROOT_IO，根目录追加/fsync返回Errno5；新增章节保存在`handoff_addendum.md`，可读旧历史前缀SHA仍一致。未确认根文件新章节持久化，不能宣称handoff已更新；I/O修复后运行`python run.py handoff`重试。',
          '正式训练完成后仅完善报告方差聚合、实测解释及handoff重试接口；训练冻结源码SHA和对应差异分别保存在manifest，训练/方法/协议/缓存/原始CSV均未改变。','']
    tmp=RESULTS/'report.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(RESULTS/'report.md')
    return summary

def append_handoff(summary):
    m=jread(MANIFEST);target=ROOT/'handoff.md';marker='<!-- EXPERIMENT28_SUPPLEMENT_E10_COARSE_FULL_FINAL -->'
    text='\n\n'+marker+'\n## 实验28补充对照E10：粗粒度Prompt＋完整辅助条件修正\n\n更新时间：'+utc()+'。STATUS='+m['status']+'。\n\n'
    text+=f'在{EXP}实际新增3任务/30折；历史E0–E9的300折只读。GPU0 seed0/2共20折，GPU1 seed1共10折。E10完整复用E4结构/损失/辅助机制，仅将fine教师换为原E2冻结coarse；28517新参数、原anchor语义监督保持。\n\n'
    text+='E10指标：'+json.dumps(summary['metrics'].get(E10,{}),ensure_ascii=False)+'。\n\n全部配对：'+json.dumps(summary['contrasts'],ensure_ascii=False)+'。\n\n'
    text+='2×2交互：'+json.dumps(summary['interaction'],ensure_ascii=False)+'。\n\n结论：'+json.dumps(summary['decisions'],ensure_ascii=False)+'。粗粒度教师近恒定，不能将辅助收益归因医学知识；原实验28 E4的NO-GO不变。\n\n'
    text+='GPU任务与耗时：'+json.dumps(m['tasks'],ensure_ascii=False)+'；总墙钟秒 '+str(m.get('wall_seconds'))+'。\n\n'
    text+='核心源码SHA '+json.dumps(m['source_sha256'])+'；缓存SHA '+m['prompt_cache_sha256']+'；历史保护 '+json.dumps(m.get('protection',{}))+'。预检 '+json.dumps(m.get('preflight',{}),ensure_ascii=False)+'。\n\n'
    text+='报告 '+str(RESULTS/'report.md')+'。保留test=val、前CV RFE及全图transductive的历史乐观偏差；后续仅建议独立队列/训练内选模验证，不追加追分参数。无自动Git提交或推送。\n'
    try:
        with target.open('a+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);old=f.read()
            assert hashlib.sha256(old[:m['original_handoff_size']]).hexdigest()==m['original_handoff_sha256'],'handoff prefix changed'
            if marker.encode() not in old:
                f.seek(0,2);written=f.write(text.encode());assert written==len(text.encode());f.flush();os.fsync(f.fileno())
            fcntl.flock(f,fcntl.LOCK_UN)
        updated=target.read_bytes();assert updated[:len(old)]==old
        assert updated.count(marker.encode())==1,'handoff append is not visible or duplicated'
        m.update(handoff_appended=True,handoff_updated_utc=utc(),handoff_sha256=sha(target),handoff_status='APPENDED_PREFIX_VERIFIED',delivery_status='COMPLETE')
    except OSError as exc:
        pending=EXP/'handoff_addendum.md';pending.write_text(text)
        m.update(handoff_appended=False,handoff_status='PENDING_ROOT_IO',handoff_error=str(exc),handoff_pending=str(pending),delivery_status='PARTIAL_HANDOFF_IO')
    jwrite(MANIFEST,m)

def finish():
    groups,rows,diag,cf=refresh();assert len(groups)==3 and len(rows)==len(diag)==30 and not cf
    assert all(g['status']=='COMPLETE' for g in groups)
    assert [sum(r['physical_gpu']==g for r in rows) for g in (0,1)]==[20,10]
    assert all(all(np.isfinite(r[k]) for k in METRICS) for r in rows)
    m=verify_static()
    changed=[p for p,h in m['protected_files_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    assert not changed,'Protected files changed '+str(changed)
    m.update(status='COMPLETE',finished_utc=utc(),protection=dict(status='UNCHANGED_SHA256',files=len(m['protected_files_sha256'])))
    m['wall_seconds']=(dt.datetime.fromisoformat(m['finished_utc'])-dt.datetime.fromisoformat(m['started_utc'])).total_seconds()
    jwrite(MANIFEST,m);summary=report();append_handoff(summary)
    m=jread(MANIFEST);m['artifacts_sha256']={str(p.relative_to(EXP)):sha(p) for p in RESULTS.glob('*') if p.is_file()};jwrite(MANIFEST,m)
    assert RUNTIME==EXP/'.runtime';shutil.rmtree(RUNTIME)
    print('STATUS=COMPLETE new30 historical300 readonly; handoff='+m['handoff_status'],flush=True)

def all_tasks():
    prepare();m=verify_static()
    if m['status']=='COMPLETE':
        for p,h in m['artifacts_sha256'].items():assert sha(EXP/p)==h
        print('COMPLETE VERIFIED; no retraining');return
    preflight();m=jread(MANIFEST);m['status']='RUNNING';m.setdefault('formal_started_utc',utc());jwrite(MANIFEST,m)
    schedule(tasks());finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','preflight','all','resume','refresh','report','handoff','_worker','_smoke'))
    p.add_argument('config',nargs='?');p.add_argument('seed',nargs='?',type=int);args=p.parse_args()
    if args.command=='_worker':worker(args.config,args.seed);return
    if args.command=='_smoke':smoke();return
    try:
        if args.command=='prepare':prepare()
        elif args.command=='preflight':preflight()
        elif args.command in ('all','resume'):all_tasks()
        elif args.command=='refresh':refresh()
        elif args.command=='report':
            report();m=jread(MANIFEST)
            m['artifacts_sha256']={str(p.relative_to(EXP)):sha(p) for p in RESULTS.glob('*') if p.is_file()};jwrite(MANIFEST,m)
            print((RESULTS/'report.md').read_text())
        elif args.command=='handoff':
            m=verify_static();assert m['status']=='COMPLETE'
            append_handoff(jread(RESULTS/'summary.json'));report();m=jread(MANIFEST)
            m['artifacts_sha256']={str(p.relative_to(EXP)):sha(p) for p in RESULTS.glob('*') if p.is_file()};jwrite(MANIFEST,m)
            print('HANDOFF_STATUS='+m['handoff_status'])
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if MANIFEST.exists():
            m=jread(MANIFEST);m['status']='BLOCKED_PREFLIGHT' if not m.get('preflight',{}).get('passed') else 'PARTIAL'
            m['failures'].append(dict(time=utc(),stage=args.command,error=str(exc)));jwrite(MANIFEST,m)
        raise

if __name__=='__main__':main()

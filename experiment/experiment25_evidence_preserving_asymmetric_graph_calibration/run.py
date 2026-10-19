"""Experiment25 EPAGC, run from SPromptGL-main/experiment/experiment25_.../.
Usage: python run.py preflight; python run.py full; python run.py report
Extra: python run.py worker B3 0 (only launched by scheduler).

Frozen 6 experimental arms x 3 seeds x 10 folds = 180 NEW training folds.
P0 uses 30 completed historical F0 seed-fold rows and is never re-trained.
"""
import os,sys,io,time,json,random,hashlib,argparse,contextlib,subprocess,concurrent.futures,shutil,fcntl,csv
from pathlib import Path
from types import SimpleNamespace
from e25_contract import DESIGNS,SEEDS,FOLDS,METRICS,tasks,validate
from e25_patch import emit,patch_network

ROOT=Path(os.environ.get('SPROMPTGL_ROOT','/data/adodas/SPromptGL-main')).resolve()
EXP=Path(__file__).resolve().parent
SOURCE=EXP/'_runtime_sources'
STATE=EXP/'_state'
RES=EXP/'results'
MANIFEST=EXP/'manifest.json'
BASELINE=ROOT/'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv'
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
 nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,
 n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',
 MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,
 theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None
CODE_FILES=('e25_gate.py','e25_contract.py','e25_patch.py','e25_runtime.py','run.py')


def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jwrite(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True,default=str))
    os.replace(tmp,path)
def jread(path):return json.loads(Path(path).read_text())
def cwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert rows
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def readcsv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def _cache_key():return {p:filehash(EXP/p) for p in CODE_FILES}
def _source():
    xs=('model.py','network.py','layers.py','utils.py',
        'data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy')
    return {name:filehash(ROOT/name) for name in xs}

def original_p0():
    assert BASELINE.is_file(),'Historical full 10-fold F0 CSV missing'
    raw=[r for r in readcsv(BASELINE) if r['config']=='F0']
    assert len(raw)==30
    assert {(int(r['seed']),int(r['fold'])) for r in raw}==set((s,f) for s in SEEDS for f in FOLDS)
    transformed=[]
    for row in raw:
        a={k:float(row[k]) for k in ('acc','ba','probability_auc')}
        a.update(seed=int(row['seed']),fold=int(row['fold']),
             sen_asd=float(row['sensitivity_asd']),spe_td=float(row['specificity_td']))
        # The historical F0 file doesn't contain F1; derive it from CONFUSION COUNTS.
        tp,fn,fp=(int(row[k]) for k in ('asd_tp','asd_fn','td_fp'))
        a['f1_asd']=2*tp/max(2*tp+fn+fp,1)
        transformed.append(a)
    import numpy as np
    mean_acc=100*np.mean([np.mean([r['acc'] for r in transformed if r['seed']==s]) for s in SEEDS])
    assert abs(mean_acc-91.691)<0.005,(mean_acc,'wrong historical baseline')
    return transformed


def get_data():
    import numpy as np,pandas as pd
    from sklearn.model_selection import StratifiedKFold
    df=pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv')
    mapping=np.load(ROOT/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert [len(v) for v in mapping.values()]==[48,6,10,256]
    assert df.shape==(871,321)
    x=df.iloc[:,:-1].to_numpy(dtype=np.float32)
    y=df.iloc[:,-1].to_numpy(dtype=np.int64)-1
    assert (y==0).sum()==403 and (y==1).sum()==468
    si=[i for i,v in enumerate(mapping['PHENO']) if str(v).startswith('SITE_ID_')]
    assert len(si)==20
    site=x[:,si].argmax(axis=1).astype(np.int64)
    assert len(set(site))==20
    folds=list(StratifiedKFold(n_splits=10,shuffle=True,random_state=10).split(x,y))
    return dict(x=x,y=y,site=site,folds=folds,site_columns=[str(mapping['PHENO'][i]) for i in si])


def build_sources():
    if not SOURCE.exists():
        emit(ROOT,SOURCE)
        (SOURCE/'CLIP').symlink_to(ROOT/'CLIP',target_is_directory=True)
    for n in ('model.py','network.py','layers.py','utils.py'):
        assert (SOURCE/n).exists()
    if not (SOURCE/'CLIP').exists():
        (SOURCE/'CLIP').symlink_to(ROOT/'CLIP',target_is_directory=True)
    assert (SOURCE/'network.py').read_text()==patch_network((ROOT/'network.py').read_text())
    for n in ('model.py','layers.py','utils.py'):
        assert filehash(SOURCE/n)==filehash(ROOT/n)
    return SOURCE


def context():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1,os.environ.get('CUDA_VISIBLE_DEVICES')
        info=get_data()
        sys.path.insert(0,str(build_sources()))
        import model,network,e25_runtime
        assert Path(network.__file__).resolve()==(SOURCE/'network.py').resolve()
        _CTX={**info,'torch':torch,'model':model,'network':network,'runtime':e25_runtime}
    return _CTX

def set_seed(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def rng():
    import torch,numpy as np
    return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(state):
    import torch,numpy as np
    random.setstate(state['py']);np.random.set_state(state['np'])
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda'])

def agent(config,seed,fold):
    c=context()
    tr,te=c['folds'][fold-1]
    c['runtime'].configure(config,fold,c['x'],tr,c['site'])
    with contextlib.redirect_stdout(io.StringIO()):
        a=c['model'].EvalHelper([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    import numpy as np
    assert np.array_equal(a.val_idx,a.tst_idx)
    return a

def model_state(a):
    return {name:{k:v.detach().cpu().clone() for k,v in mod.state_dict().items()}
      for name,mod in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}
def load_models(a,obj):
    a.MessagePassing0.load_state_dict(obj['gfr']);a.ModalFusion.load_state_dict(obj['pln'])


def evaluate(a):
    c=context();t=c['torch']
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with t.no_grad():
        m,x=a.dataLayer(a.feat)
        z,_,_=a.MessagePassing0(x,m,a.targ,a.trn_idx)
        out,_,_=a.ModalFusion(z)
    return out

def metrics(a):
    import numpy as np
    from sklearn.metrics import roc_auc_score,confusion_matrix
    pred=evaluate(a).exp().detach().cpu().numpy()[a.tst_idx]
    y=context()['y'][a.tst_idx]
    yp=pred.argmax(axis=1)
    tp,fn,fp,tn=map(int,confusion_matrix(y,yp,labels=[0,1]).ravel())
    sen=tp/max(tp+fn,1);spe=tn/max(tn+fp,1)
    return dict(acc=float((yp==y).mean()),ba=(sen+spe)/2,
        probability_auc=float(roc_auc_score(y==0,pred[:,0])),
        sen_asd=sen,spe_td=spe,f1_asd=2*tp/max(2*tp+fn+fp,1),
        asd_fn=fn,td_fp=fp,test_n=len(y))

def _taskpath(cfg,seed):return STATE/(f'{cfg}_s{seed}.pt')
def save_ckpt(path,obj):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    buf=io.BytesIO();torch.save(obj,buf)
    body=buf.getvalue();h=hashlib.sha256(body).hexdigest().encode()
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(b'E25!'+h+body);os.replace(tmp,path)
def load_ckpt(path):
    import torch
    b=Path(path).read_bytes();assert b[:4]==b'E25!'
    assert hashlib.sha256(b[68:]).hexdigest().encode()==b[4:68]
    return torch.load(io.BytesIO(b[68:]),map_location='cpu',weights_only=False)

def task_worker(cfg,seed):
    assert cfg in DESIGNS and seed in SEEDS
    assert os.environ.get('E25_PHYSICAL_GPU')==os.environ.get('CUDA_VISIBLE_DEVICES')
    c=context();ckpt=_taskpath(cfg,seed)
    manifest=jread(MANIFEST)
    assert _cache_key()==manifest['code_sha256']
    assert _source()==manifest['source_sha256']
    previous=load_ckpt(ckpt) if ckpt.exists() else None
    set_seed(seed)
    if previous:
        c['network'].get_prompt_text_features()  # avoid CLIP cache altering restored RNG
        restore_rng(previous['rng'])
    rows=list(previous['rows']) if previous else []
    next_fold=int(previous['fold']) if previous else 1
    if previous and previous['phase']=='BOUNDARY' and len(rows)==10:
        print('SKIP_COMPLETE',cfg,seed,flush=True);return
    assert len(rows)==next_fold-1
    init_hashes=[]
    for fold in range(next_fold,11):
        a=agent(cfg,seed,fold)
        # Initialization checksum only; no extra optimizer/forward step.
        m=model_state(a)
        digest=hashlib.sha256()
        for part in ('gfr','pln'):
            for name in sorted(m[part]):
                digest.update(name.encode());digest.update(m[part][name].contiguous().numpy().tobytes())
        initial=digest.hexdigest()
        if previous and previous['phase']=='EPOCH' and int(previous['fold'])==fold:
            load_models(a,previous['model'])
            a.optimizer_MP0.load_state_dict(previous['mp'])
            a.optimizer_MF.load_state_dict(previous['mf'])
            restore_rng(previous['rng'])
            best=float(previous['best']);best_epoch=int(previous['best_epoch'])
            best_state=previous['best_state'];wait=int(previous['wait'])
            start=int(previous['epoch'])
            initial=previous['initial']
        else:
            best=0.0;best_epoch=-1;best_state=None;wait=0;start=0
        previous=None
        start_time=time.monotonic();executed=start
        for epoch in range(start,450):
            with contextlib.redirect_stdout(io.StringIO()):
                a.run_epoch(mode='simple-2')
                _,val=a.print_trn_acc('simple-2')
            val=float(val)
            if val>best:
                best=val;best_epoch=epoch;best_state=model_state(a);wait=0
            else:wait+=1
            executed=epoch+1
            if executed%25==0 and wait<=200:
                save_ckpt(ckpt,dict(config=cfg,seed=seed,phase='EPOCH',fold=fold,
                    epoch=executed,best=best,best_epoch=best_epoch,best_state=best_state,
                    model=model_state(a),mp=a.optimizer_MP0.state_dict(),
                    mf=a.optimizer_MF.state_dict(),wait=wait,rng=rng(),rows=rows,initial=initial))
            if wait>200:break
        assert best_state is not None
        next_random=rng()
        load_models(a,best_state)
        c['runtime'].capture_stats(True)
        met=metrics(a)
        c['runtime'].capture_stats(False)
        assert abs(met['acc']-best)<1e-12,(cfg,seed,fold,met['acc'],best)
        stats=c['runtime'].stats()
        assert stats.get('edge_count',0)>0 and stats.get('mean_edge_multiplier',0)>=.8
        record=dict(config=cfg,seed=seed,fold=fold,physical_gpu=int(os.environ['E25_PHYSICAL_GPU']),
            **met,best_epoch=best_epoch,epochs_executed=executed,
            selected_val_acc=best,train_n=len(a.trn_idx),
            edge_count=stats['edge_count'],edge_multiplier=stats['mean_edge_multiplier'],
            qc_mean=stats['qc_mean'],initial_hash=initial,
            wall_seconds=time.monotonic()-start_time)
        rows.append(record)
        restore_rng(next_random)
        save_ckpt(ckpt,dict(config=cfg,seed=seed,phase='BOUNDARY',fold=fold+1,
            rng=next_random,rows=rows))
        print('FOLD',cfg,seed,fold,f"ACC={met['acc']:.5f}",flush=True)
        del a,best_state
    assert len(rows)==10
    print('DONE',cfg,seed,flush=True)

def gather():
    rows=[]
    for cfg in DESIGNS:
        for seed in SEEDS:
            path=_taskpath(cfg,seed)
            assert path.is_file(),(cfg,seed,'missing task')
            obj=load_ckpt(path)
            assert obj['phase']=='BOUNDARY' and obj['fold']==11 and len(obj['rows'])==10
            rows.extend(obj['rows'])
    assert len(rows)==180
    assert len({(r['config'],r['seed'],r['fold']) for r in rows})==180
    return sorted(rows,key=lambda r:(r['config'],r['seed'],r['fold']))

def per_seed(rows,metric):
    import numpy as np
    return [float(np.mean([r[metric] for r in rows if int(r['seed'])==s])) for s in SEEDS]

def statsfmt(rows,metric):
    import numpy as np
    a=np.array(per_seed(rows,metric))
    return f'{a.mean()*100:.3f}±{a.std(ddof=0)*100:.3f}'

def paired(raw,base):
    rindex={(int(r['seed']),int(r['fold'])):r for r in base}
    out=[]
    for cfg in DESIGNS:
        here=[r for r in raw if r['config']==cfg]
        for m in ('acc','ba','probability_auc'):
            diffs=[float(r[m])-float(rindex[(r['seed'],r['fold'])][m]) for r in here]
            import numpy as np
            a=np.array(diffs)
            out.append(dict(config=cfg,metric=m,delta_pp=float(100*a.mean()),
                wins=int((a>1e-10).sum()),ties=int((abs(a)<=1e-10).sum()),losses=int((a< -1e-10).sum()),
                per_seed_delta_pp=[float(100*np.mean([d for d,r in zip(diffs,here) if r['seed']==s])) for s in SEEDS]))
    return out

def write_report(raw):
    b=original_p0();both={'P0':[dict(config='P0',**r) for r in b]}
    both.update({k:[r for r in raw if r['config']==k] for k in DESIGNS})
    pairs=paired(raw,b)
    result=['# Experiment25 — EPAGC: 10-fold x 3-seed results','',
       '**历史P0是实验13 F0的已完成30折，不重新训练。**',
       '**所有新组是完整10折×3seed，每个seed先平均10折，再报告三seed均值±总体标准差。**',
       '原始val=test选模、前置RFE、transductive 871图均保留；不能称无泄漏泛化。','',
       '|组别|ACC %|BA / hard-AUC %|Prob-AUC %|SEN-ASD %|SPE-TD %|F1-ASD %|',
       '|---|---:|---:|---:|---:|---:|---:|']
    for cfg,rs in both.items():
        result.append('|'+cfg+'|'+ '|'.join(statsfmt(rs,k) for k in METRICS)+'|')
    result+=['','## 与历史P0的30折配对差（pp）','',
      '|组别|指标|Δ百分点|胜/平/负|逐seed差(pp)|','|---|---|---:|---|---|']
    for row in pairs:
        result.append(f"|{row['config']}|{row['metric']}|{row['delta_pp']:+.3f}|{row['wins']}/{row['ties']}/{row['losses']}|"+
         ', '.join(f'{v:+.3f}' for v in row['per_seed_delta_pp'])+'|')
    result+=['','## 机制审计','',
       '|组|原FC非对角边数均值|边乘数均值|QC相似度均值|平均最佳epoch|',
       '|---|---:|---:|---:|---:|']
    import numpy as np
    for cfg in DESIGNS:
        x=both[cfg]
        result.append('|'+cfg+'|'+ '|'.join(f"{np.mean([float(r[k]) for r in x]):.4f}" for k in
          ('edge_count','edge_multiplier','qc_mean','best_epoch'))+'|')
    result+=['','## 核心因果边界与风险','',
       'B3 vs B4 检查质量条件的患者对应；B3 vs B0 检查是否优于支持集平均强度常量收缩（不同模型实际门控预算未必严格相同）；',
       'B3 vs B5 检查站点校正的作用；B1/B2 检查单路质量条件。',
       '注意ANAT/FUNC是质量指标而不是独立的丰富影像表征，本研究只能声称质量条件的非对称关系校准。',
       'FC图保留原始>0.9边支持和自环，不改变原GFR其他三路融合逻辑。',
       '统计有测试集参与选模、预交叉验证RFE及多轮研究选择偏差，不代表独立临床性能。','']
    RES.mkdir(parents=True,exist_ok=True)
    tmp=RES/'report.tmp';tmp.write_text('\n'.join(result));os.replace(tmp,RES/'report.md')
    return pairs

def append_handoff(stage,body):
    text=f'\n\n## Experiment25 {stage}\n{body}\n'
    target=ROOT/'handoff.md'
    with target.open('ab',buffering=0) as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(text.encode());f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)

def preflight():
    validate()
    import e25_gate
    e25_gate.self_test()
    orig=original_p0();info=get_data()
    assert len(orig)==30
    assert not MANIFEST.exists(),'Preflight already ran; run full or report; do not overwrite state'
    assert ROOT==Path('/data/adodas/SPromptGL-main').resolve(),ROOT
    assert not (ROOT/'experiment'/'experiment24_conflict_aware_relational_alignment'/'.LOCK-EDIT').exists()
    build_sources()
    # Invariance of newly added gate to changes in held-out labels.
    from e25_gate import prepare
    tr,te=info['folds'][0]
    m=prepare('B3',1,info['x'],tr,info['site'])
    assert m['consensus'].shape==(871,871)
    payload=dict(status='PREFLIGHT_PASS',source_sha256=_source(),
        code_sha256=_cache_key(),reference_p0_file=str(BASELINE),
        reference_p0_sha256=filehash(BASELINE),history_rows=30,
        experiment_groups=list(DESIGNS),folds=10,seeds=list(SEEDS),
        new_fold_target=180,per_gpu_target=90,protocol=PROTOCOL,
        notes='Only original P0/F0 file read. Historical fixed protocol unchanged.')
    jwrite(MANIFEST,payload)
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']='0';env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    smoke=subprocess.run([sys.executable,str(EXP/'run.py'),'smoke'],
        env=env,cwd=EXP,capture_output=True,text=True)
    if smoke.returncode:
        raise RuntimeError('GPU smoke failed; NO formal training allowed: '+smoke.stderr[-5000:])
    manifest=jread(MANIFEST);manifest['gpu_smoke_pass']=True;jwrite(MANIFEST,manifest)
    append_handoff('PREFLIGHT_PASS','已审核原始源码、历史P0完整30折、site/训练折QC预处理；GPU forward、gradient、原模型关闭门控等价性通过。尚未训练。')
    print('PREFLIGHT PASS historical P0 30 folds; GPU smoke PASS')

def launch():
    assert MANIFEST.exists(),'preflight first'
    m=jread(MANIFEST)
    if m['status']=='COMPLETE':
        assert (RES/'fold_metrics.csv').exists() and (RES/'report.md').exists()
        assert filehash(RES/'fold_metrics.csv')==m['fold_metrics_sha256']
        assert filehash(RES/'report.md')==m['report_sha256']
        print('ALREADY_COMPLETE; 180-fold result hashes verified; no retraining');return
    assert m.get('gpu_smoke_pass') is True,'No real training before CUDA parity smoke'
    assert _cache_key()==m['code_sha256']
    assert _source()==m['source_sha256']
    jobs=tasks();print('TASKS',len(jobs),'GPU0=9 GPU1=9')
    import threading
    slots={0:threading.Semaphore(2),1:threading.Semaphore(2)}
    def start(job):
        cfg,seed,gpu=job
        with slots[gpu]:
            env=os.environ.copy()
            env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env['E25_PHYSICAL_GPU']=str(gpu)
            env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
            p=subprocess.run([sys.executable,str(EXP/'run.py'),'worker',cfg,str(seed)],
                 env=env,cwd=EXP,capture_output=True,text=True)
            if p.returncode:raise RuntimeError(f'GPU{gpu} {cfg} {seed}:\n{p.stderr[-6000:]}\n{p.stdout[-1000:]}')
            return p.stdout.strip()[-400:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures=[ex.submit(start,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
    raw=gather()
    assert sum(r['physical_gpu']==0 for r in raw)==sum(r['physical_gpu']==1 for r in raw)==90
    RES.mkdir(parents=True,exist_ok=True)
    cwrite(RES/'fold_metrics.csv',raw)
    differences=write_report(raw)
    manifest=jread(MANIFEST)
    manifest.update(status='COMPLETE',complete_rows=180,
        fold_metrics_sha256=filehash(RES/'fold_metrics.csv'),report_sha256=filehash(RES/'report.md'),
        paired_differences=differences)
    jwrite(MANIFEST,manifest)
    append_handoff('COMPLETE',
      '6组新增方法×3seeds×10folds=180折完成；历史P0为完整30折ACC 91.691%。'+
      ' 两卡各90折。逐折结果、负对照、研究结论见实验25/results/report.md。'+
      ' 若结果为阴性不可包装MICCAI2027创新点。')
    # Keep only necessary final files: 5 source files, manifest, 2 results.
    shutil.rmtree(STATE,ignore_errors=True)
    shutil.rmtree(SOURCE,ignore_errors=True)
    print('COMPLETE 180 folds + 30 archived P0 rows')

def smoke():
    import torch
    ctx=context();set_seed(0)
    a=agent('B3',0,1)
    a.MessagePassing0.eval();a.ModalFusion.eval()
    m,x=a.dataLayer(a.feat)
    ctx['runtime'].set_enabled(False)
    with torch.no_grad():
        base,_,_=a.MessagePassing0(x,m,a.targ,a.trn_idx)
        pred0,_,_=a.ModalFusion(base)
    ctx['runtime'].set_enabled(True)
    with torch.no_grad():
        changed,_,_=a.MessagePassing0(x,m,a.targ,a.trn_idx)
        pred1,_,_=a.ModalFusion(changed)
    assert not torch.equal(changed,base),'Gate did not affect real FC graph'
    assert not torch.equal(pred0,pred1),'Gate not on active prediction path'
    ctx['runtime'].set_enabled(False)
    with torch.no_grad():
        orig,_,_=a.MessagePassing0(x,m,a.targ,a.trn_idx)
    assert torch.equal(base,orig),'Disabled gate must be bitwise identity'
    ctx['runtime'].set_enabled(True)
    a.MessagePassing0.train();a.ModalFusion.train()
    a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad()
    gfr,_,_=a.MessagePassing0(x,m,a.targ,a.trn_idx)
    logp,_,_=a.ModalFusion(gfr)
    ce=ctx['model'].ClsLoss(logp,a.targ,a.trn_idx,a.weight)
    ce.backward()
    grad=a.MessagePassing0.GraphConstruct3.p.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.norm())>0
    # Compare uninterrupted vs resumed run_epoch (model, two Adam states, all RNG).
    set_seed(0)
    u=agent('B3',0,1)
    with contextlib.redirect_stdout(io.StringIO()):u.run_epoch('simple-2')
    epoch1=dict(model=model_state(u),mp=u.optimizer_MP0.state_dict(),
                mf=u.optimizer_MF.state_dict(),rng=rng())
    # Serialize exactly as controller checkpoints; optimizer tensors must be copied.
    buf=io.BytesIO();torch.save(epoch1,buf);buf.seek(0)
    epoch1=torch.load(buf,map_location='cpu',weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):u.run_epoch('simple-2')
    expected=model_state(u);rand_expected=rng()
    v=agent('B3',0,1)
    load_models(v,epoch1['model'])
    v.optimizer_MP0.load_state_dict(epoch1['mp'])
    v.optimizer_MF.load_state_dict(epoch1['mf'])
    restore_rng(epoch1['rng'])
    with contextlib.redirect_stdout(io.StringIO()):v.run_epoch('simple-2')
    got=model_state(v);rand_got=rng()
    for part in expected:
        for key in expected[part]:
            assert torch.equal(expected[part][key],got[part][key]),('resume failure',part,key)
    assert torch.equal(rand_expected['cpu'],rand_got['cpu'])
    for x,y in zip(rand_expected['cuda'],rand_got['cuda']):assert torch.equal(x,y)
    print('E25_CUDA_SMOKE_AND_RESUME_PASS')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('preflight','full','report','worker','smoke'))
    parser.add_argument('config',nargs='?');parser.add_argument('seed',nargs='?',type=int)
    args=parser.parse_args()
    if args.command=='preflight':preflight()
    elif args.command=='full':launch()
    elif args.command=='worker':task_worker(args.config,args.seed)
    elif args.command=='smoke':smoke()
    elif args.command=='report':
        m=jread(MANIFEST)
        assert m['status']=='COMPLETE','Use full to finish remaining tasks'
        assert filehash(RES/'fold_metrics.csv')==m['fold_metrics_sha256']
        assert filehash(RES/'report.md')==m['report_sha256']
        print((RES/'report.md').read_text())
if __name__=='__main__':main()


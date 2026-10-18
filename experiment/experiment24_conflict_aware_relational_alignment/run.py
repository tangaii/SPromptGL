"""Experiment24 autonomous runner. Usage: python run.py preflight|pilot|full|report|worker C S F.
Only experiment-local source copies are patched; root remains read-only except
append-only handoff. A seed's 10 folds share one uninterrupted RNG trajectory.
"""
import os,sys,io,math,time,json,copy,random,shutil,hashlib,pickle,argparse,contextlib,subprocess,concurrent.futures
from pathlib import Path
from types import SimpleNamespace

# Set ROOT explicitly only when repository is mounted at a different path.
ROOT=Path(os.environ.get('SPROMPTGL_ROOT','/data/adodas/SPromptGL-main')).resolve()
EXP=Path(__file__).resolve().parent
SOURCE=EXP/'_runtime_sources'
STATE=EXP/'_state'
RESULT=EXP/'results'
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
 nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,
 n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',
 MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,
 theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
from e24_contract import DESIGNS,SEEDS,FOLDS,PILOT_FOLDS,METRICS,tasks,assert_contract
from e24_patch import emit
_CTX=None


def sha(b):return hashlib.sha256(b).hexdigest()
def save_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,default=str));os.replace(tmp,p)
def read_json(p):return json.loads(Path(p).read_text())
def save_csv(p,rows):
    import csv
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,p)


def _ckptfile(config,seed):return STATE/f'{config}_s{seed}.pt'
def save_state(path,obj):
    import torch
    b=io.BytesIO();torch.save(obj,b);data=b.getvalue()
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(b'E24!'+sha(data).encode()+data);os.replace(tmp,path)
def load_state(path):
    import torch
    b=path.read_bytes();assert b[:4]==b'E24!' and sha(b[68:]).encode()==b[4:68]
    return torch.load(io.BytesIO(b[68:]),map_location='cpu',weights_only=False)

def seed_all(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def rng():
    import torch,numpy as np
    return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all())
def restore_rng(d):
    import torch,numpy as np
    random.setstate(d['py']);np.random.set_state(d['np']);torch.set_rng_state(d['cpu'])
    torch.cuda.set_rng_state_all(d['cuda'])

def source_audit():
    assert_contract()
    assert (ROOT/'network.py').exists() and (ROOT/'data/ABIDE/processed_standard_data.csv').exists()
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/n for n in ('model.py','network.py','layers.py','utils.py')] +
        [ROOT/'data/ABIDE/processed_standard_data.csv',ROOT/'data/ABIDE/modal_feat_dict.npy']}

def prepare():
    import numpy as np,pandas as pd
    from sklearn.model_selection import StratifiedKFold
    hashes=source_audit()
    df=pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv')
    mapping=np.load(ROOT/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert list(map(len,mapping.values()))==[48,6,10,256]
    assert df.shape==(871,321)
    x=df.iloc[:,:-1].to_numpy(dtype='float32');y=df.iloc[:,-1].to_numpy(dtype=int)-1
    assert (y==0).sum()==403 and (y==1).sum()==468
    siteidx=[i for i,v in enumerate(mapping['PHENO']) if str(v).startswith('SITE_ID_')]
    assert len(siteidx)==20
    rawsite=x[:,siteidx];site=rawsite.argmax(1)
    assert all((rawsite.max(1)>rawsite.min(1))) and len(set(site))>=15
    fold=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
    assert len(fold)==10
    return dict(hashes=hashes,x=x,y=y,site=site,folds=fold,
                siteidx=siteidx,site_distribution={str(k):int((site==k).sum()) for k in set(site)})

def _sources():
    if SOURCE.exists():
        for n in ('model.py','network.py','layers.py','utils.py'):
            assert (SOURCE/n).exists()
    else:emit(ROOT,SOURCE)
    lib=ROOT/'CLIP';assert lib.exists(),('missing CLIP dependency',lib)
    if not (SOURCE/'CLIP').exists():(SOURCE/'CLIP').symlink_to(lib,target_is_directory=True)
    return SOURCE

def context():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1,os.environ.get('CUDA_VISIBLE_DEVICES')
        info=prepare();sys.path.insert(0,str(_sources()))
        import network,model,e24_runtime
        assert Path(network.__file__).resolve()==(SOURCE/'network.py').resolve()
        _CTX={**info,'torch':torch,'network':network,'model':model,'runtime':e24_runtime}
    return _CTX

def agent(config,seed,fold):
    c=context();c['runtime'].configure(config,fold,c['site'])
    train,test=c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):
        a=c['model'].EvalHelper([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),train,test)
    a.MF_sav.close();a.GCMP_sav.close()
    assert np_equal(a.val_idx,a.tst_idx)
    return a

def np_equal(a,b):
    import numpy as np
    return bool(np.array_equal(a,b))

def module_state(a):
    return {n:{k:v.detach().cpu().clone() for k,v in obj.state_dict().items()} for n,obj in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}
def restore_models(a,d):
    a.MessagePassing0.load_state_dict(d['gfr']);a.ModalFusion.load_state_dict(d['pln'])

def evaluate(a):
    import torch
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat)
        features,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        pred,_,_=a.ModalFusion(features)
    return pred

def test_metrics(a):
    import numpy as np
    from sklearn.metrics import confusion_matrix,roc_auc_score
    o=evaluate(a).exp().detach().cpu().numpy()[a.tst_idx]
    yy=context()['y'][a.tst_idx]; pp=o.argmax(1)
    cm=confusion_matrix(yy,pp,labels=[0,1]);tp,fn,fp,tn=map(int,cm.ravel())
    sen=tp/max(tp+fn,1);spe=tn/max(tn+fp,1)
    return dict(acc=float((pp==yy).mean()),ba=(sen+spe)/2,
                probability_auc=float(roc_auc_score(yy==0,o[:,0])),
                sen_asd=sen,spe_td=spe,f1_asd=2*tp/max(2*tp+fp+fn,1),
                asd_fn=fn,td_fp=fp)

def _load_task(config,seed):
    path=_ckptfile(config,seed)
    if path.exists():
        s=load_state(path)
        assert s['config']==config and s['seed']==seed
        return s
    return None

def worker(config,seed,until):
    # RUN ONLY INSIDE GPU SUBPROCESS, not controller.
    c=context();torch=c['torch'];path=_ckptfile(config,seed)
    existing=read_json(EXP/'manifest.json')
    assert c['hashes']==existing['source_sha256'],'Original code/data changed since preflight'
    current_artifacts={p.name:sha(p.read_bytes()) for p in [EXP/n for n in
        ('e24_align.py','e24_contract.py','e24_patch.py','e24_runtime.py','run.py')]}
    assert current_artifacts==existing['artifact_sha256'], 'Experimental implementation changed after preflight'
    initial=_load_task(config,seed)
    seed_all(seed)
    if initial:
        # Baseline's cached RN50 must be ready BEFORE restoring saved RNG.
        c['network'].get_prompt_text_features()
        restore_rng(initial['rng'])
    rows=list(initial['rows']) if initial else []
    start_fold=int(initial['fold']) if initial else 1
    assert start_fold==len(rows)+1 or (initial and initial['phase']=='EPOCH' and start_fold==len(rows)+1)
    t_start=time.monotonic()
    for fold in range(start_fold,until+1):
        restored=initial if initial and initial['fold']==fold and initial['phase']=='EPOCH' else None
        a=agent(config,seed,fold)
        best=-1.;wait=0;best_epoch=-1;best_state=None;start_epoch=0
        if restored:
            restore_models(a,restored['model'])
            a.optimizer_MP0.load_state_dict(restored['mp'])
            a.optimizer_MF.load_state_dict(restored['mf'])
            restore_rng(restored['rng'])
            best=float(restored['best']);wait=int(restored['wait'])
            best_epoch=int(restored['best_epoch']);best_state=restored['best_state']
            start_epoch=int(restored['epoch'])
        for epoch in range(start_epoch,PROTOCOL['nepoch']):
            with contextlib.redirect_stdout(io.StringIO()):
                a.run_epoch(mode='simple-2')
                _,val=a.print_trn_acc('simple-2')
            if val>best:
                best=val;wait=0;best_epoch=epoch;best_state=module_state(a)
            else:wait+=1
            if (epoch+1)%25==0 and not(wait>PROTOCOL['early']):
                save_state(path,dict(config=config,seed=seed,phase='EPOCH',fold=fold,epoch=epoch+1,
                    best=best,best_epoch=best_epoch,wait=wait,model=module_state(a),
                    best_state=best_state,mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),
                    rng=rng(),rows=rows))
            if wait>PROTOCOL['early']:break
        executed=epoch+1
        assert best_state is not None
        restore_models(a,best_state)
        met=test_metrics(a)
        # Recompute diagnostics at the SELECTED checkpoint (not final epoch).
        # This is a read-only gradient query: NO backward, optimizer step or
        # test-label access in the added regularizer.
        if getattr(a,'_e24',None):
            with torch.enable_grad():
                a.MessagePassing0.eval();a.ModalFusion.eval()
                n,part=a.dataLayer(a.feat)
                features,_,_=a.MessagePassing0(part,n,a.targ,a.trn_idx)
                outputs,_,_=a.ModalFusion(features)
                ce=c['model'].ClsLoss(outputs,a.targ,a.trn_idx,a.weight)
                _=a._e24.loss(a.MessagePassing0,part,a.targ,a.trn_idx,ce)
            s=dict(a._e24.stats)
        else:s={}
        rows.append(dict(config=config,seed=seed,fold=fold,best_epoch=best_epoch,
            epochs_executed=executed,selected_val_acc=float(best),
            train_n=len(a.trn_idx),test_n=len(a.tst_idx),
            **met,align_calls=(a._e24.call_count if a._e24 else 0),
            support_mean=s.get('support_mean',0.),align_loss=s.get('alignment_loss_mean',0.),
            active_align_graphs=s.get('active',0),class_graph_loss=s.get('class_graph',0.),
            grad_cos_0=(s.get('grad_cos',[float('nan')]*3)+[float('nan')]*3)[0],
            grad_cos_1=(s.get('grad_cos',[float('nan')]*3)+[float('nan')]*3)[1],
            grad_cos_2=(s.get('grad_cos',[float('nan')]*3)+[float('nan')]*3)[2]))
        save_state(path,dict(config=config,seed=seed,phase='BOUNDARY',fold=fold+1,
            rng=rng(),rows=rows))
        del a
        initial=None
    print(json.dumps(dict(config=config,seed=seed,done=until,rows=len(rows),seconds=time.monotonic()-t_start)))


def all_rows(maxfold):
    rows=[]
    for cfg in DESIGNS:
        for seed in SEEDS:
            item=_load_task(cfg,seed)
            if item is not None:
                rows.extend(row for row in item['rows'] if row['fold']<=maxfold)
    return sorted(rows,key=lambda x:(x['config'],x['seed'],x['fold']))

def _historical_parity(rows):
    # Only A0 is required bit-for-bit to reproduce the historical baseline.
    import csv
    historical=[]
    for number,typ in [(13,'F0'),(17,'D0'),(18,'V0')]:
        d=list((ROOT/'experiment').glob(f'experiment{number}_*/results/fold_metrics.csv'))
        if d:
            with d[0].open() as f:rr=list(csv.DictReader(f))
            historical.append((number,typ,rr))
    assert historical,'No historical benchmark available for exact P0 parity'
    checked=0
    for row in rows:
        if row['config']!='A0':continue
        for num,typ,other in historical:
            ref=next(x for x in other if x['config']==typ and int(x['seed'])==row['seed'] and int(x['fold'])==row['fold'])
            for k in ('acc','ba','probability_auc'):
                assert abs(row[k]-float(ref[k]))<1e-12,(num,row,k,ref[k])
            assert row['best_epoch']==int(ref['best_epoch']),('epoch mismatch',row,ref['best_epoch'])
            checked+=1
    return checked

def pilot_decision(rows):
    import numpy as np
    means={}
    for cfg in DESIGNS:
        k=[r for r in rows if r['config']==cfg]
        assert len(k)==6,(cfg,len(k))
        means[cfg]={key:float(np.mean([row[key] for row in k])) for key in ('acc','ba','probability_auc')}
    def d(c,m):return 100*(means[c][m]-means['A0'][m])
    # Qualifying candidate must nearly retain P0 while beating pure
    # supervision and shuffled edge alignment on useful paired metrics.
    candidates=[]
    for c in ('A3','A4'):
        direct=means[c];p0=means['A0'];pure=means['A1'];wrong=means['A5']
        yes=(100*(direct['ba']-p0['ba'])>=-.30 and
             100*(direct['probability_auc']-p0['probability_auc'])>=-.30 and
             (100*(direct['ba']-pure['ba'])>=.15 or
              100*(direct['probability_auc']-pure['probability_auc'])>=.15) and
             direct['ba']>wrong['ba'] and direct['probability_auc']>wrong['probability_auc'])
        if yes:candidates.append(c)
    return dict(continue_full=bool(candidates),pilot_candidates=candidates,means=means,
        note='Fast futility screen only, not a statistical test; reuse test-set selection.' )

def make_report(rows,phase,decision=None):
    import numpy as np
    title=f'# Experiment24 CARA — {phase}\n\n'
    title+=('冻结阶段：先48折，有条件续跑至240折。'+'\n\n')
    title+='|组别|ACC|BA|Prob-AUC|SEN ASD|SPE TD|F1 ASD|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for cfg in DESIGNS:
        subset=[x for x in rows if x['config']==cfg]
        if not subset:continue
        def fmt(k):
            per=[np.mean([r[k] for r in subset if r['seed']==s]) for s in SEEDS]
            return f'{100*np.mean(per):.3f}±{100*np.std(per):.3f}'
        title+=f'|{cfg} {DESIGNS[cfg].name}|'+ '|'.join(fmt(k) for k in METRICS)+'|\n'
    title+='\n注：Hard-AUC在此二分类协议中等于BA；Prob-AUC使用类别0真实概率。\n'
    if decision:title+='\n## 预注册快速判断\n\n```json\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n```\n'
    title+='\n## 解释限制\n\n所有指标使用历史原始val=test选模及CV前RFE、871人transductive图；pilot用了同一批测试标签选最优checkpoint。三个seed并非独立队列。新增图正则只利用训练折标签；本报告不称独立泛化或生物标志物。\n'
    (RESULT/'report.md').write_text(title)


def handoff(stage,detail):
    import fcntl
    target=ROOT/'handoff.md'
    text='\n\n## Experiment24 '+stage+'\n'+str(detail)+'\n'
    try:
        with target.open('ab',buffering=0) as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.write(text.encode());f.flush();os.fsync(f.fileno())
            fcntl.flock(f,fcntl.LOCK_UN)
        return True
    except OSError as e:
        (EXP/'handoff_pending.txt').write_text(text+'ERROR='+repr(e))
        return False


def schedule(until):
    # Four processes, 2 physical GPUs; evenly split 12 tasks/card.
    assert until in (PILOT_FOLDS,10)
    jobs=[]
    for cfg,seed,gpu in tasks():
        obj=_load_task(cfg,seed)
        if obj and len(obj['rows'])>=until:continue
        jobs.append((cfg,seed,gpu))
    def launch(job):
        cfg,seed,gpu=job;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
        cmd=[sys.executable,str(EXP/'run.py'),'worker',cfg,str(seed),str(until)]
        p=subprocess.run(cmd,env=env,cwd=str(EXP),capture_output=True,text=True)
        if p.returncode!=0:raise RuntimeError(f'{job}\nSTDOUT={p.stdout[-1000:]}\nSTDERR={p.stderr[-5000:]}')
        return p.stdout.strip()
    # Fixed 2 slots/card. Avoid relying on GPU task order for RNG.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        sems={0:__import__('threading').Semaphore(2),1:__import__('threading').Semaphore(2)}
        def limit(job):
            with sems[job[2]]:return launch(job)
        futures=[pool.submit(limit,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
    result=all_rows(until)
    assert len(result)==len(DESIGNS)*len(SEEDS)*until
    parity=_historical_parity(result)
    save_csv(RESULT/'fold_metrics.csv',result)
    return result,parity


def preflight():
    import e24_align
    e24_align.self_test();source=source_audit()
    _sources()
    from e24_patch import patch
    assert sha(patch((ROOT/'model.py').read_text()).encode())==sha((SOURCE/'model.py').read_bytes())
    assert 'e24_runtime.install(self)' in (SOURCE/'model.py').read_text()
    print('Preflight: source and CPU contract passed; GPU parity is verified by A0 pilot')
    save_json(EXP/'manifest.json',dict(stage='PREFLIGHT',source_sha256=source,protocol=PROTOCOL,
       original_val_test_untouched=True,original_rfe_untouched=True,
       artifact_sha256={p.name:sha(p.read_bytes()) for p in [EXP/n for n in
         ('e24_align.py','e24_contract.py','e24_patch.py','e24_runtime.py','run.py')]},
       notes='Formal experiment begins with 48-fold pilot and A0 historical exact parity.'))
    # First official A0 fold doubles as a no-waste GPU regression oracle.
    # This 1 fold belongs to the planned 48-fold pilot; never train it twice.
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    proc=subprocess.run([sys.executable,str(EXP/'run.py'),'worker','A0','0','1'],
                        env=env,cwd=str(EXP),capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError('GPU baseline oracle failed: '+proc.stderr[-5000:])
    parity=_historical_parity(all_rows(1))
    assert parity>=1
    obj=read_json(EXP/'manifest.json')
    obj.update(stage='PREFLIGHT_GPU_PASS',gpu_baseline_parity_checks=parity)
    save_json(EXP/'manifest.json',obj)
    handoff('PREFLIGHT_GPU_PASS',f'CPU合同、GPU原P0首折实跑、历史逐值parity={parity}通过。')


def execute_pilot():
    assert (EXP/'manifest.json').exists(),'Run preflight first'
    rows,parity=schedule(PILOT_FOLDS)
    choice=pilot_decision(rows)
    save_json(RESULT/'decision.json',choice)
    save_json(EXP/'manifest.json',{**read_json(EXP/'manifest.json'),
        'stage':('PILOT_COMPLETE' if choice['continue_full'] else 'PILOT_STOP'),'pilot_folds':len(rows),'parity_checks':parity,'decision':choice})
    make_report(rows,'PILOT',choice)
    handoff('PILOT_COMPLETE',json.dumps(choice,ensure_ascii=False))
    print('PILOT',json.dumps(choice,ensure_ascii=False))


def execute_full():
    obj=read_json(RESULT/'decision.json')
    if not obj['continue_full']:
        print('STABLE_FUTILITY_STOP: fast gate false; do not launch extra 192 folds')
        return
    rows,parity=schedule(10)
    save_json(EXP/'manifest.json',{**read_json(EXP/'manifest.json'),
        'stage':'COMPLETE','formal_folds':len(rows),'parity_checks':parity})
    make_report(rows,'FULL',obj)
    handoff('COMPLETE','已完成240折；详见 results/report.md 和 results/fold_metrics.csv。')
    # State records are necessary for audit until full COMPLETE; only delete
    # after successful handoff and manifest+rows checks.
    print('FULL COMPLETE 240 folds; formal summary written')


def report():
    d=read_json(EXP/'manifest.json');stage=d['stage']
    rows=all_rows(10 if stage=='COMPLETE' else PILOT_FOLDS)
    decision=read_json(RESULT/'decision.json') if (RESULT/'decision.json').exists() else None
    make_report(rows,stage,decision)
    print('REPORT',stage,len(rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','pilot','full','report','worker']);
    parser.add_argument('config',nargs='?');parser.add_argument('seed',nargs='?');parser.add_argument('until',nargs='?');args=parser.parse_args()
    if args.command=='preflight':preflight()
    elif args.command=='pilot':execute_pilot()
    elif args.command=='full':execute_full()
    elif args.command=='report':report()
    else:worker(args.config,int(args.seed),int(args.until))


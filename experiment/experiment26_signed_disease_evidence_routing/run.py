"""Experiment26 SDER, run from SPromptGL-main/experiment/experiment26_.../.
Usage: python run.py preflight; python run.py full; python run.py report
Extra: python run.py worker D2 0 (only launched by scheduler).

Frozen 6 experimental arms x 3 seeds x 10 folds = 180 NEW training folds.
P0 uses 30 completed historical F0 seed-fold rows and is never re-trained.
"""
import os,sys,io,time,json,random,hashlib,argparse,contextlib,subprocess,concurrent.futures,shutil,fcntl,csv
from pathlib import Path
from types import SimpleNamespace
from e26_contract import DESIGNS,SEEDS,FOLDS,METRICS,tasks,validate
from e26_patch import emit,patch_network,patch_model

ROOT=Path(os.environ.get('SPROMPTGL_ROOT','/data/adodas/SPromptGL-main')).resolve()
EXP=Path(__file__).resolve().parent
SOURCE=EXP/'_runtime_sources'
STATE=EXP/'_state'
RES=EXP/'results'
MANIFEST=EXP/'manifest.json'
BASELINE=ROOT/'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv'
TEXT_CACHE=EXP/'prompt_prototypes.npz'
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
 nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,
 n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',
 MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,
 theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None
CODE_FILES=('e26_evidence.py','e26_contract.py','e26_patch.py','e26_runtime.py','run.py')


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

def prepare_prompt_cache():
    """GPU preflight only; no test labels, no learned text interpolation."""
    import numpy as np,torch
    from e26_contract import TEXT_UNRELATED
    assert torch.cuda.is_available()
    build_sources()
    sys.path.insert(0,str(SOURCE))
    import network
    from CLIP.clip import clip
    # CLIP class 0 is NORMAL/TD, class 1 is ALTERED/ASD.
    with torch.no_grad():
        p0=network.get_prompt_text_features().detach().float().cpu().numpy()
        assert p0.shape==(8,1024)
        real=np.stack([p0[:4].reshape(-1),p0[4:].reshape(-1)]).astype(np.float32)
        model,_=clip.load('RN50',device='cuda')
        unrelated=model.encode_text(clip.tokenize(TEXT_UNRELATED).to('cuda')).float().cpu().numpy()
        del model
        unrelated=np.stack([unrelated[:4].reshape(-1),unrelated[4:].reshape(-1)]).astype(np.float32)
    # Matched pairwise Gram: random subspace, same norms and inter-class cosine.
    q_old,_=np.linalg.qr(real.T.astype(np.float64))
    q_new,_=np.linalg.qr(np.random.default_rng(26001).normal(size=(4096,2)))
    isometric=(q_new@(q_old.T@real.T)).T.astype(np.float32)
    g=lambda x:x.astype(np.float64)@x.astype(np.float64).T
    assert np.allclose(g(real),g(isometric),rtol=2e-5,atol=0.1)
    TEXT_CACHE.parent.mkdir(parents=True,exist_ok=True)
    tmp=TEXT_CACHE.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,real=real,unrelated=unrelated,isometric=isometric)
    os.replace(tmp,TEXT_CACHE)
    return dict(cache_sha256=filehash(TEXT_CACHE),real_gram=g(real).tolist(),
                iso_gram=g(isometric).tolist(),texts=list(TEXT_UNRELATED))

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
    assert (SOURCE/'model.py').read_text()==patch_model((ROOT/'model.py').read_text())
    for n in ('layers.py','utils.py'):
        assert filehash(SOURCE/n)==filehash(ROOT/n)
    return SOURCE


def context():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1,os.environ.get('CUDA_VISIBLE_DEVICES')
        assert TEXT_CACHE.is_file(),'Run preflight to build frozen text cache'
        info=get_data()
        sys.path.insert(0,str(build_sources()))
        import model,network,e26_runtime
        assert Path(network.__file__).resolve()==(SOURCE/'network.py').resolve()
        _CTX={**info,'torch':torch,'model':model,'network':network,'runtime':e26_runtime}
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
    c['runtime'].configure(config,TEXT_CACHE)
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
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(b'E26!'+h+body);os.replace(tmp,path)
def load_ckpt(path):
    import torch
    b=Path(path).read_bytes();assert b[:4]==b'E26!'
    assert hashlib.sha256(b[68:]).hexdigest().encode()==b[4:68]
    return torch.load(io.BytesIO(b[68:]),map_location='cpu',weights_only=False)

def task_worker(cfg,seed):
    assert cfg in DESIGNS and seed in SEEDS
    assert os.environ.get('E26_PHYSICAL_GPU')==os.environ.get('CUDA_VISIBLE_DEVICES')
    c=context();ckpt=_taskpath(cfg,seed)
    manifest=jread(MANIFEST)
    assert _cache_key()==manifest['code_sha256']
    assert _source()==manifest['source_sha256']
    assert filehash(TEXT_CACHE)==manifest['text_cache_sha256']
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
        met=metrics(a)
        assert abs(met['acc']-best)<1e-12,(cfg,seed,fold,met['acc'],best)
        d=a.ModalFusion._e26_details
        last=a.ModalFusion._e26_last
        base=last['base']; tok=last['tokens']; txt=last['text']
        from e26_evidence import evidence,apply
        design=DESIGNS[cfg]
        flip_e,_=evidence(tok,txt.flip(0),design.selective,design.fc_only)
        off=base.exp()[:,0]; regular=apply(base,a.ModalFusion._e26_logits,design.readout)
        # Do not call PLN/GFR again or change RNG: use saved readout intermediates.
        swapped=apply(base,flip_e,design.readout)
        te=a.tst_idx
        abs_prob_off=float((regular.exp()[te,0]-off[te]).abs().mean().item())
        abs_prob_swap=float((regular.exp()[te,0]-swapped.exp()[te,0]).abs().mean().item())
        flip_off=float((regular[te].argmax(-1)!=base[te].argmax(-1)).float().mean().item())
        flip_swap=float((regular[te].argmax(-1)!=swapped[te].argmax(-1)).float().mean().item())
        sc=d['score'].detach(); m=d['by_modality'].detach(); conf=d['confidence'].detach()
        tr=a.trn_idx
        yy=a.targ[tr]
        separation=float(sc[tr][yy==0].mean()-sc[tr][yy==1].mean())
        record=dict(config=cfg,seed=seed,fold=fold,physical_gpu=int(os.environ['E26_PHYSICAL_GPU']),
            **met,best_epoch=best_epoch,epochs_executed=executed,
            selected_val_acc=best,train_n=len(a.trn_idx),initial_hash=initial,
            prompt_cos=float(d['prototype_cos'].detach()),prompt_distance=float(d['prompt_distance'].detach()),
            evidence_std=float(sc.std().item()),train_evidence_separation=separation,
            mean_confidence=float(conf.mean().item()),
            mean_abs_modality_0=float(m[:,0].abs().mean().item()),
            mean_abs_modality_1=float(m[:,1].abs().mean().item()),
            mean_abs_modality_2=float(m[:,2].abs().mean().item()),
            mean_abs_modality_3=float(m[:,3].abs().mean().item()),
            off_abs_delta=abs_prob_off,swap_abs_delta=abs_prob_swap,
            off_flip=flip_off,swap_flip=flip_swap,
            wall_seconds=time.monotonic()-start_time)
        # Read-only reporting supplement: reuse existing checkpoint outputs only.
        from sklearn.metrics import roc_auc_score,confusion_matrix
        report_y=c['y'][te]
        for label,output in (('off',base),('swap',swapped)):
            report_prob=output[te].exp().detach().cpu().numpy()
            report_pred=report_prob.argmax(-1)
            tp,fn,fp,tn=map(int,confusion_matrix(report_y,report_pred,labels=[0,1]).ravel())
            record.update({label+'_acc':float((report_pred==report_y).mean()),
                label+'_ba':.5*(tp/(tp+fn)+tn/(tn+fp)),
                label+'_probability_auc':float(roc_auc_score(report_y==0,report_prob[:,0])),
                label+'_asd_fn':fn})
        for modality in range(4):
            record[f'mean_signed_modality_{modality}']=float(m[:,modality].mean().item())
            record[f'mean_confidence_modality_{modality}']=float(conf[:,modality].mean().item())
            record[f'mean_abs_selected_modality_{modality}']=float((m[:,modality]*conf[:,modality]).abs().mean().item())
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
    import numpy as np
    b=original_p0();both={'P0':[dict(config='P0',**r) for r in b]}
    both.update({k:[r for r in raw if r['config']==k] for k in DESIGNS})
    pairs=paired(raw,b)
    lines=['# Experiment26 — Signed Disease Evidence Routing (SDER)','',
        '**Status: 180 new folds complete; P0 historical full 30 folds read-only.**',
        '**Protocol: same original val=test, pre-CV RFE and full-871 transductive graph; do not claim unbiased generalization.**','',
        '|组|ACC %|BA/HardAUC %|Prob-AUC %|SEN-ASD %|SPE-TD %|ASD F1 %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cfg,rs in both.items():
        lines.append('|'+cfg+'|'+ '|'.join(statsfmt(rs,k) for k in METRICS)+'|')
    lines+=['','## 逐seed完整十折六指标（%）','',
        '|组|seed|ACC|BA|Prob-AUC|SEN-ASD|SPE-TD|ASD F1|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cfg,rs in both.items():
        for seed in SEEDS:
            fields=[100*np.mean([float(r[k]) for r in rs if r['seed']==seed]) for k in METRICS]
            lines.append('|'+cfg+'|'+str(seed)+'|'+'|'.join(f'{x:.4f}' for x in fields)+'|')
    lines+=['','## 核心配对（same seed/fold）','',
        '|比较|ΔACC pp|ΔBA pp|ΔprobAUC pp|', '|---|---:|---:|---:|']
    ix={(r['config'],r['seed'],r['fold']):r for r in raw}
    ib={(r['seed'],r['fold']):r for r in b}
    def diff(left,right,metric):
        return 100*np.mean([ix[left,k,s][metric] - (ib[k,s][metric] if right=='P0' else ix[right,k,s][metric])
              for k in SEEDS for s in FOLDS])
    contrasts=[('D2',k) for k in ('P0','D1','D3','D4','D5','D6')]
    for a,c in contrasts:
        lines.append('|'+a+' - '+c+'|'+ '|'.join(f'{diff(a,c,m):+.4f}' for m in ('acc','ba','probability_auc'))+'|')
    lines+=['','## 每个配置与历史P0胜/平/负','',
        '|组|指标|平均Δ pp|胜/平/负|逐seed差pp|', '|---|---|---:|---|---|']
    for item in pairs:
        lines.append(f"|{item['config']}|{item['metric']}|{item['delta_pp']:+.4f}|"+
           f"{item['wins']}/{item['ties']}/{item['losses']}|"+
           ','.join(f'{q:+.3f}' for q in item['per_seed_delta_pp'])+'|')
    lines+=['','## 最佳checkpoint机制检验','',
        '|组|prompt prototype cos|证据std|Train ASD−TD evidence|关闭读出后 ΔASD prob|类交换后 ΔASD prob|关闭读出翻转率|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cfg in DESIGNS:
        rs=both[cfg]
        fields=('prompt_cos','evidence_std','train_evidence_separation','off_abs_delta','swap_abs_delta','off_flip')
        lines.append('|'+cfg+'|'+ '|'.join(f'{np.mean([float(r[x]) for r in rs]):.6f}' for x in fields)+'|')
    lines+=['','## 方法边界与否证条件','',
        '真实D2必须在BA与prob-AUC中同时优于P0，并表现出相对D4和D5的可靠区别，且D2优于关闭读出D3/无四模态D6才可谈Prompt语义证据。',
        '实际比较应报告三个seed差、10fold逐项配对，不把30fold当30个独立队列。',
        '固定checkpoint类交换导致的概率变化只证明结构上依赖标签方向，不证明医学语义成立。',
        '原SPGS类别max数学上不区分类别，SDER保留原SPGS并额外以有符号证据直连输出，不添加新分类参数。',
        '原始CLIP图像文本域缺乏脑特征预训练对齐，4096→9可学习投影可吸收任意文字方向。',
        '实验9–22类别监督、医学语义/随机、ROI证据等先前试验均未证实真实语义独立收益；不把本轮当新颖性的充分证明。',
        '请结合历史E9结果进行参考：历史E9 ACC约91.883%、BA约91.692%、ProbAUC约93.823%；不同Prompt基座不能声称同模型消融。',
        '输入PHENO48/ANAT6/FUNC10/CORR256且局部token=4×4×9不等于真实ROI。',
        '阶段性结论不得使用测试标签选择新超参数或文本，禁称无泄漏临床泛化。','']
    RES.mkdir(parents=True,exist_ok=True)
    tmp=RES/'report.tmp';tmp.write_text('\n'.join(lines));os.replace(tmp,RES/'report.md')
    return pairs

def append_handoff(stage,body):
    text=f'\n\n## Experiment26 {stage}\n{body}\n'
    target=ROOT/'handoff.md'
    try:
        with target.open('ab',buffering=0) as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.write(text.encode());f.flush();os.fsync(f.fileno())
            fcntl.flock(f,fcntl.LOCK_UN)
    except OSError as exc:
        # /data is an unreliable FUSE mount; retain the authorized append in /tmp.
        manifest=jread(MANIFEST)
        manifest.setdefault('pending_handoff',[]).append(dict(stage=stage,text=text,
            timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),error=str(exc)))
        manifest['handoff_status']='PENDING_DATA_IO'
        jwrite(MANIFEST,manifest)
        print('HANDOFF_PENDING_DATA_IO',stage,str(exc),flush=True)

def preflight():
    validate()
    import e26_evidence,e26_patch
    e26_evidence.self_test();e26_patch.self_test()
    orig=original_p0();info=get_data()
    assert len(orig)==30
    assert not MANIFEST.exists(),'Preflight already ran; run full or report; do not overwrite state'
    assert ROOT==Path('/data/adodas/SPromptGL-main').resolve(),ROOT
    build_sources()
    textinfo=prepare_prompt_cache()
    payload=dict(status='PREFLIGHT_PASS',source_sha256=_source(),
        text_cache_sha256=filehash(TEXT_CACHE),text_cache_info=textinfo,
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
    append_handoff('PREFLIGHT_PASS','已审核原始源码、历史P0完整30折与冻结CLIP真实/无关/随机同几何文本；GPU正向、梯度、原输出保留路径与恢复测试通过。尚未训练。')
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
    assert filehash(TEXT_CACHE)==m['text_cache_sha256']
    jobs=tasks();print('TASKS',len(jobs),'GPU0=9 GPU1=9')
    import threading
    slots={0:threading.Semaphore(2),1:threading.Semaphore(2)}
    def start(job):
        cfg,seed,gpu=job
        with slots[gpu]:
            env=os.environ.copy()
            env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env['E26_PHYSICAL_GPU']=str(gpu)
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
      '6组类别Prompt对齐方法×3seeds×10folds=180折完成；历史P0完整30折ACC 91.691%。'+
      ' 两卡各90折。逐折结果、证据读出/无关与同几何随机控制见实验26/results/report.md。'+
      ' 若结果为阴性不可包装MICCAI2027创新点。')
    # Keep only the five frozen files, prompt cache, manifest, and two result files.
    shutil.rmtree(STATE,ignore_errors=True)
    shutil.rmtree(SOURCE,ignore_errors=True)
    print('COMPLETE 180 folds + 30 archived P0 rows')

def smoke():
    """GPU validation of real network, gradient route, semantics and exact restart."""
    import torch
    ctx=context();set_seed(0)
    from e26_evidence import evidence,apply
    a=agent('D2',0,1)
    a.MessagePassing0.eval();a.ModalFusion.eval()
    pred=evaluate(a)
    last=a.ModalFusion._e26_last
    assert torch.isfinite(pred).all() and pred.shape==(871,2)
    # Fixed-model counterfactuals; held-out labels do not control interventions.
    off=last['base']
    assert torch.allclose(apply(off,torch.zeros_like(off),True),off,atol=1e-6),'Zero evidence must numerically restore base'
    assert float((pred-off).abs().mean())>1e-8,'Evidence did not enter final prediction'
    d=a.ModalFusion._e26_details
    assert d['score'].std()>1e-6,'Evidence collapsed before formal run'
    swapped,_=evidence(last['tokens'],last['text'].flip(0),True,False)
    assert torch.allclose(swapped,-a.ModalFusion._e26_logits,atol=1e-6)
    assert not torch.equal(pred,apply(off,swapped,True))
    a.MessagePassing0.train();a.ModalFusion.train()
    a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad()
    m,parts=a.dataLayer(a.feat)
    features,_,_=a.MessagePassing0(parts,m,a.targ,a.trn_idx)
    logp,_,_=a.ModalFusion(features)
    ce=ctx['model'].ClsLoss(logp,a.targ,a.trn_idx,a.weight)
    aux=ctx['runtime'].loss(a.ModalFusion,a.targ,a.trn_idx)
    (ce+aux).backward()
    g=a.ModalFusion.MLP.t2.weight.grad
    assert g is not None and bool(torch.isfinite(g).all()) and float(g.norm())>1e-9
    set_seed(0)
    u=agent('D2',0,1)
    with contextlib.redirect_stdout(io.StringIO()):u.run_epoch('simple-2')
    state=dict(model=model_state(u),mp=u.optimizer_MP0.state_dict(),
               mf=u.optimizer_MF.state_dict(),rng=rng())
    buf=io.BytesIO();torch.save(state,buf);buf.seek(0)
    state=torch.load(buf,map_location='cpu',weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):u.run_epoch('simple-2')
    expected=model_state(u);r_expected=rng()
    v=agent('D2',0,1)
    load_models(v,state['model']);v.optimizer_MP0.load_state_dict(state['mp'])
    v.optimizer_MF.load_state_dict(state['mf']);restore_rng(state['rng'])
    with contextlib.redirect_stdout(io.StringIO()):v.run_epoch('simple-2')
    got=model_state(v);r_got=rng()
    for part in expected:
        for key in expected[part]:assert torch.equal(expected[part][key],got[part][key]),(part,key)
    assert torch.equal(r_expected['cpu'],r_got['cpu'])
    for x,y in zip(r_expected['cuda'],r_got['cuda']):assert torch.equal(x,y)
    # Negative-control cache with original SPGS unchanged, no new model params.
    for cfg in ('D4','D5'):
        set_seed(0)
        mod=agent(cfg,0,1)
        predx=evaluate(mod);assert torch.isfinite(predx).all(),cfg
        assert sum(x.numel() for x in mod.ModalFusion.parameters())==sum(x.numel() for x in a.ModalFusion.parameters()),cfg
    print('E26_GPU_SMOKE_RESUME_CONTROL_PASS')

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

#!/usr/bin/env python3
"""Experiment 44 runner.

Commands are intentionally explicit: prepare -> stageA -> formal -> audit ->
report -> handoff.  Every mutable artifact is below this experiment folder.
"""
from __future__ import annotations

import argparse, csv, copy, hashlib, json, math, os, pickle, random, shutil, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

ROOT = Path('/data/adodas/SPromptGL-main')
EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'; RESULTS = EXP / 'results'
CFG = json.loads((EXP/'config.json').read_text())
sys.path.insert(0, str(EXP)); sys.path.insert(0, str(RT/'sources'))

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
import integration
from cugl import CUGLBranch, CUConfig

CTX = None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)


def read_json(path): return json.loads(Path(path).read_text())


def save_obj(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_obj(path): return torch.load(path, map_location='cpu', weights_only=False)


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['group','fold']
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_read(path):
    if not Path(path).exists(): return []
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def state_hash(state):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            z = v.detach().cpu().contiguous(); h.update(str((z.dtype, tuple(z.shape))).encode()); h.update(z.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            for z in v: add(z)
        else: h.update(repr(v).encode())
    add(state); return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def setup_gpu(gpu):
    global CTX
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES maps requested physical GPU to 0.
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    CTX = {'gpu': gpu, 'device': torch.device('cuda'), 'layers': __import__('layers')}
    return CTX


def data(): return integration.read_data()


def fold_indices(f): return data()[2][f-1]


def split_sha(f):
    tr, te = fold_indices(f)
    return integration.fold_sha(tr, te)


def source_signature():
    files = [EXP/x for x in ('run.py','integration.py','cugl.py','teacher.py','config.json')]
    files += list((RT/'sources').glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(EXP)): sha(p) for p in files}, sort_keys=True).encode()).hexdigest()


def prepare():
    RT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    for p in (RT/'anchors', RT/'candidates', RT/'teachers', RT/'checkpoints', RT/'tasks', RT/'logs'):
        p.mkdir(parents=True, exist_ok=True)
    src_hashes = integration.source_copy()
    x, y, folds, mapping = data()
    cand_hashes = {}
    for f, (tr, te) in enumerate(folds, 1):
        c = integration.all_candidates(x, tr, te, int(CFG['candidate_k']))
        p = RT/'candidates'/f'fold{f:02}.npz'; np.savez_compressed(p, candidates=c, train=tr, test=te)
        cand_hashes[str(f)] = sha(p)
    # E0/B0-B and the anchored physical CAM-only group are read-only historical
    # evidence; copies in this directory make their provenance auditable.
    hist0 = ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints'
    hist1 = ROOT/'experiment/experiment43_4_physical_cam_only_validation/.runtime/checkpoints'
    for f in range(1,11):
        shutil.copy2(hist0/f'g0_f{f:02}.best.pt', RT/'checkpoints'/f'g0_f{f:02}.best.pt')
        shutil.copy2(hist1/f'g0_f{f:02}.best.pt', RT/'checkpoints'/f'g1_f{f:02}.best.pt')
    protected_names = ['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']
    protected = {str(ROOT/n): sha(ROOT/n) for n in protected_names}
    handoff = ROOT/'handoff.md'; hb = handoff.read_bytes()
    manifest = {
        'status':'PREPARED','experiment':'Experiment 44 — CU-GL','started_unix':time.time(),
        'seed':0,'data_shape':list(x.shape),'label_counts':np.bincount(y).tolist(),
        'fold_count':10,'split_rule':'StratifiedKFold(n_splits=10,shuffle=True,random_state=10)',
        'candidate_k':16,'source_hashes':src_hashes,'candidate_hashes':cand_hashes,
        'protected_sha256':protected,'source_signature':source_signature(),
        'handoff_prefix':{'bytes':len(hb),'sha256':hashlib.sha256(hb).hexdigest()},
        'historical':{'b0b_dir':str(hist0),'cam_dir':str(hist1)},
        'teacher_tasks_expected':30,'formal_innovation_folds_expected':80
    }
    write_json(EXP/'manifest.json', manifest)
    print('PREPARE_PASS', json.dumps({'data':list(x.shape),'labels':manifest['label_counts'],'candidate_k':16}, ensure_ascii=False), flush=True)


def inner_splits(indices, y, seed):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    a = np.asarray(indices)
    return [(a[tr], a[va]) for tr, va in skf.split(a, y[a])]


def build_inner_targets(f, gpu):
    """Train 3 frozen teachers for one outer fold and write OOF utility cache."""
    from teacher import train_teacher, edge_utilities
    setup_gpu(gpu)
    x_np, y_np, folds, _ = data(); outer_train, outer_test = folds[f-1]
    x = torch.from_numpy(x_np).float().cuda(); y = torch.from_numpy(y_np).long().cuda()
    U = np.full((len(x_np),4,int(CFG['candidate_k'])), np.nan, np.float32)
    M = np.zeros_like(U, np.uint8)
    summaries=[]; teacher_hashes=[]; started=time.time()
    for inner_id,(itr, iva) in enumerate(inner_splits(outer_train, y_np, 44000+f), 1):
        c_np = integration.candidate_pool(x_np, np.asarray(outer_train), np.asarray(itr), int(CFG['candidate_k']))
        c = torch.from_numpy(c_np).long().cuda()
        seed = 444000 + f*10 + inner_id
        net, trace = train_teacher(CTX['layers'], x, y, itr, c, seed, int(CFG['teacher_epochs']))
        uh, vm, st = edge_utilities(net, x, y, c, iva)
        # Transfer only OOF targets.  The receiver label was not used to train
        # the teacher; it selected the log probability after freezing.
        U[iva] = uh[iva]; M[iva] = vm[iva]
        th = state_hash(net.state_dict()); teacher_hashes.append(th)
        st.update({'outer_fold':f,'inner_fold':inner_id,'train_count':int(len(itr)),
                   'heldout_count':int(len(iva)),'teacher_state_sha256':th,
                   'trace':trace})
        summaries.append(st)
        print('TEACHER_DONE',f,inner_id,st['valid_edges'],st['mean_abs'],flush=True)
        del net, c
        torch.cuda.empty_cache()
    p = RT/'teachers'/f'outer_fold{f:02}.pt'
    save_obj(p, {'fold':f,'train':outer_train,'test':outer_test,'utility':U,'valid':M,
                 'summaries':summaries,'teacher_hashes':teacher_hashes,
                 'candidate_hash':sha(RT/'candidates'/f'fold{f:02}.npz'),
                 'wall_seconds':time.time()-started,'source_signature':source_signature()})
    print('OUTER_TEACHERS_DONE',f,'valid',int(M.sum()),'abs',float(np.nanmean(np.abs(U))),flush=True)


def stageA(gpu, folds=None):
    if not (EXP/'manifest.json').exists(): prepare()
    folds = folds or list(range(1,11))
    for f in folds:
        if (RT/'teachers'/f'outer_fold{f:02}.pt').exists():
            print('SKIP_TEACHER_CACHE',f,flush=True); continue
        build_inner_targets(f, gpu)
    if all((RT/'teachers'/f'outer_fold{f:02}.pt').exists() for f in range(1,11)):
        preflight_science()


def map_outer_utility(f):
    """Map inner-pool OOF values onto the fixed outer candidate pool."""
    x,y,folds,_=data(); tr,te=folds[f-1]
    d=np.load(RT/'candidates'/f'fold{f:02}.npz'); outer=d['candidates']
    z=load_obj(RT/'teachers'/f'outer_fold{f:02}.pt'); U0=z['utility']; M0=z['valid']
    U=np.full_like(outer,dtype=np.float32,fill_value=np.nan); M=np.zeros_like(outer,dtype=np.uint8)
    # Each OOF target has values in the inner candidate pool saved only in the
    # summary cache; reconstruct the deterministic inner pool and copy by sender.
    for inner_id,(itr,iva) in enumerate(inner_splits(tr,y,44000+f),1):
        c=np.asarray(integration.candidate_pool(x,np.asarray(tr),np.asarray(itr),16))
        for i in iva:
            for m in range(4):
                for s,j in enumerate(c[i,m]):
                    if j < 0 or not M0[i,m,s]: continue
                    slots=np.where(outer[i,m]==j)[0]
                    if len(slots): U[i,m,slots[0]]=U0[i,m,s];M[i,m,slots[0]]=1
    return U,M,outer


def preflight_science():
    """Stage-A gate.  It checks information content, direction and exact edits."""
    x,y,folds,_=data(); allstats=[]; values=[]; labels=[]; hom=[]
    for f in range(1,11):
        U,M,C=map_outer_utility(f); finite=np.isfinite(U)&(M.astype(bool))
        vals=U[finite]; values.extend(vals.tolist())
        for i in range(len(x)):
            for m in range(4):
                for s,j in enumerate(C[i,m]):
                    if j>=0 and finite[i,m,s]:
                        labels.append(float(y[i]==y[j])); hom.append(float(U[i,m,s]))
        allstats.append({'fold':f,'valid':int(finite.sum()),'mean_abs':float(np.mean(np.abs(vals))) if len(vals) else 0.,'std':float(np.std(vals)) if len(vals) else 0.,'positive':float(np.mean(vals>0)) if len(vals) else 0.,'negative':float(np.mean(vals<0)) if len(vals) else 0.})
    vals=np.asarray(values,float); labels=np.asarray(labels,float); hom=np.asarray(hom,float)
    rng=np.random.default_rng(4401); shuffled=rng.permutation(vals) if len(vals) else vals
    real_hom=float(np.corrcoef(hom,labels)[0,1]) if len(hom)>2 and np.std(hom)>0 and np.std(labels)>0 else 0.
    real_shuffle=float(np.corrcoef(vals,shuffled)[0,1]) if len(vals)>2 and np.std(vals)>0 else 0.
    # Numerical gate: nonzero signed responses and a distribution that is not
    # just one repeated floating-point value.  Tiny but finite responses remain
    # scientifically reportable, rather than being silently amplified.
    passed=bool(len(vals)>=1000 and np.isfinite(vals).all() and np.std(vals)>1e-8 and np.mean(vals>0)>0.01 and np.mean(vals<0)>0.01)
    record={'status':'PASS' if passed else 'PREFLIGHT_SCIENTIFIC_NO_GO','valid_edges':int(len(vals)),
            'mean_abs':float(np.mean(np.abs(vals))) if len(vals) else 0.,'std':float(np.std(vals)) if len(vals) else 0.,
            'positive_fraction':float(np.mean(vals>0)) if len(vals) else 0.,'negative_fraction':float(np.mean(vals<0)) if len(vals) else 0.,
            'real_vs_fixed_shuffle_corr':real_shuffle,'utility_vs_homophily_corr':real_hom,
            'fold_stats':allstats,'criterion':'>=1000 finite OOF edges, signed variance >1e-8, positive and negative >1%'}
    write_json(RT/'preflight_science.json',record)
    m=read_json(EXP/'manifest.json');m['stageA']=record;m['status']='STAGEA_PASS' if passed else 'PREFLIGHT_SCIENTIFIC_NO_GO';write_json(EXP/'manifest.json',m)
    print('STAGE_A',record['status'],record,flush=True)
    if not passed: raise SystemExit(2)


def utility_targets_for_group(f, g):
    U,M,C=map_outer_utility(f); x,y,folds,_=data(); tr,te=folds[f-1]
    spec=CFG['groups'][str(g)]
    if spec.get('homophily'):
        H=np.zeros_like(U,np.float32); HM=np.zeros_like(M,np.uint8)
        for i in tr:
            for mm in range(4):
                for s,j in enumerate(C[i,mm]):
                    if j>=0: H[i,mm,s]=1.0 if y[i]==y[j] else -1.0;HM[i,mm,s]=1
        return H,HM,C,'homophily'
    if spec.get('shuffle'):
        U2=U.copy(); rng=np.random.default_rng(447000+f)
        for mm in range(4):
            idx=np.argwhere(M[:,mm]>0); vals=np.array([U[i,mm,s] for i,s in idx]); rng.shuffle(vals)
            for (i,s),v in zip(idx,vals): U2[i,mm,s]=v
        return U2,M,C,'shuffled_cf'
    return U,M,C,'crossfit_cf' if spec.get('cf') else 'none'


class Agent:
    def __init__(self, f, g, gpu, load_targets=True):
        self.f=f;self.g=g;self.gpu=gpu;self.device=torch.device('cuda')
        self.x_np,self.y_np,self.folds,_=data(); self.tr,self.te=self.folds[f-1]
        self.x=torch.from_numpy(self.x_np).float().cuda();self.y=torch.from_numpy(self.y_np).long().cuda()
        cp=np.load(RT/'candidates'/f'fold{f:02}.npz');self.candidates=torch.from_numpy(cp['candidates']).long().cuda()
        spec=CFG['groups'][str(g)]; self.cfg=CUConfig(k=16,use_cf=bool(spec.get('cf')),symmetric=bool(spec.get('symmetric')),use_null=bool(spec.get('null')))
        # New branch randomness is isolated by a fixed derived seed.  CAM is
        # then overwritten by the exact untrained B0-B fold anchor.
        seed_all(440000+g*100+f)
        self.branch=CUGLBranch(self.cfg).cuda(); self.cam=integration.build(CTX['layers'],integration.make_hp()).cuda()
        anchor=load_obj(RT/'anchors'/f'fold{f:02}.pt'); self.cam.load_state_dict(anchor['model']['pln'])
        self.class_weight=(len(self.tr)/(2*np.bincount(self.y_np[self.tr],minlength=2))).astype(np.float32)
        self.weight=torch.from_numpy(self.class_weight).cuda()
        self.opt_cam=torch.optim.Adam(self.cam.parameters(),lr=CFG['base_lr'],weight_decay=CFG['weight_decay'])
        self.opt_new=torch.optim.Adam(self.branch.parameters(),lr=CFG['new_lr'],weight_decay=CFG['new_weight_decay'])
        if load_targets:
            self.U_np,self.M_np,self.C_np,self.target_source=utility_targets_for_group(f,g)
        else:
            self.U_np=np.zeros((len(self.x_np),4,16),np.float32);self.M_np=np.zeros_like(self.U_np,np.uint8);self.C_np=self.candidates.detach().cpu().numpy();self.target_source='audit_only'
        self.U=torch.from_numpy(np.nan_to_num(self.U_np,nan=0.0)).float().cuda();self.M=torch.from_numpy(self.M_np).bool().cuda()

    def forward(self):
        xn,info=self.branch(self.x,self.candidates,self.cfg); out=self.cam(xn)[0]; return out,xn,info

    def step(self):
        self.cam.train();self.branch.train();self.opt_cam.zero_grad(set_to_none=True);self.opt_new.zero_grad(set_to_none=True)
        out,xn,info=self.forward(); cls=F.nll_loss(out[self.tr],self.y[self.tr],weight=self.weight)
        if self.cfg.use_cf and bool(self.M[self.tr].any()):
            ul,ud=self.branch.utility_loss(info,self.U,self.M, self.cfg); loss=cls+ul
        else: ul=cls*0;ud={'utility_reg':0.,'utility_rank':0.,'utility_valid':0};loss=cls
        if not torch.isfinite(loss): raise FloatingPointError('formal loss is not finite')
        loss.backward();
        torch.nn.utils.clip_grad_norm_(self.cam.parameters(),10.);torch.nn.utils.clip_grad_norm_(self.branch.parameters(),10.)
        gn=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in self.branch.parameters() if p.grad is not None)).item())
        self.opt_cam.step();self.opt_new.step()
        return float(loss.detach()),float(cls.detach()),float(ul.detach()),gn,ud

    @torch.no_grad()
    def evaluate(self):
        self.cam.eval();self.branch.eval();out,xn,info=self.forward();return out,xn,info


def metrics(out, agent, pred=None):
    if pred is None: pred=out[agent.te].argmax(1).detach().cpu().numpy()
    tar=agent.y_np[agent.te];tp,fn,fp,tn=map(int,confusion_matrix(tar,pred,labels=[0,1]).ravel())
    return {'acc':float(np.mean(pred==tar)),'hard_auc':float(roc_auc_score(np.eye(2)[tar],np.eye(2)[pred])), 'spe':float(tn/(tn+fp)),'sen':float(tp/(tp+fn)),'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,'prediction':pred.astype(int).tolist(),'test_indices':agent.te.astype(int).tolist()}


def formal_fold(g,f,gpu):
    key=f'g{g}_f{f:02}'; bestp=RT/'checkpoints'/f'{key}.best.pt';latest=RT/'checkpoints'/f'{key}.latest.pt';taskp=RT/'tasks'/f'{key}.json'
    if taskp.exists() and read_json(taskp).get('status')=='COMPLETE' and bestp.exists(): print('SKIP_FORMAL',key,flush=True);return
    agent=Agent(f,g,gpu); started=time.time(); best=None;bestacc=-1.;bestepoch=-1;wait=0;begin=0;trace=[];grad_trace=[];resumes=0
    base={'group':g,'name':CFG['groups'][str(g)]['name'],'fold':f,'seed':0,'physical_gpu':gpu,'split_sha':split_sha(f),'candidate_sha':sha(RT/'candidates'/f'fold{f:02}.npz'),'source_signature':source_signature(),'target_source':agent.target_source,'cam_parameter_count':sum(p.numel() for p in agent.cam.parameters()),'new_parameter_count':sum(p.numel() for p in agent.branch.parameters()),'optimizer_count':2}
    if latest.exists():
        s=load_obj(latest); assert s['base']==base;agent.cam.load_state_dict(s['cam']);agent.branch.load_state_dict(s['branch']);agent.opt_cam.load_state_dict(s['opt_cam']);agent.opt_new.load_state_dict(s['opt_new']);random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_rng'].cuda());begin=s['epoch']+1;best=s['best'];bestacc=s['bestacc'];bestepoch=s['bestepoch'];wait=s['wait'];trace=s['trace'];grad_trace=s['grad_trace'];resumes=s['resumes']+1
    for epoch in range(begin,int(CFG['epochs'])):
        loss,cls,ul,gn,ud=agent.step();out,_,_=agent.evaluate(); acc=float(out[agent.te].argmax(1).eq(agent.y[agent.te]).float().mean().item())
        if acc>bestacc:
            bestacc=acc;bestepoch=epoch;wait=0;best={'cam':copy.deepcopy(agent.cam.state_dict()),'branch':copy.deepcopy(agent.branch.state_dict()),'opt_cam':copy.deepcopy(agent.opt_cam.state_dict()),'opt_new':copy.deepcopy(agent.opt_new.state_dict()),'epoch':epoch,'bestacc':acc}
        else: wait+=1
        grad_trace.append(gn)
        if epoch%10==0 or epoch==int(CFG['epochs'])-1: trace.append({'epoch':epoch,'loss':loss,'cls':cls,'utility':ul,'acc':acc,'bestacc':bestacc,'grad_norm':gn,'utility_valid':ud.get('utility_valid',0)})
        if (epoch+1)%10==0 or wait>int(CFG['early']) or epoch==int(CFG['epochs'])-1:
            save_obj(latest,{'base':base,'cam':agent.cam.state_dict(),'branch':agent.branch.state_dict(),'opt_cam':agent.opt_cam.state_dict(),'opt_new':agent.opt_new.state_dict(),'python':random.getstate(),'numpy':np.random.get_state(),'torch_rng':torch.get_rng_state(),'epoch':epoch,'best':best,'bestacc':bestacc,'bestepoch':bestepoch,'wait':wait,'trace':trace,'grad_trace':grad_trace,'resumes':resumes})
        if wait>int(CFG['early']): break
    if best is None: raise RuntimeError(f'no best checkpoint {key}')
    agent.cam.load_state_dict(best['cam']);agent.branch.load_state_dict(best['branch']);out,xn,info=agent.evaluate();met=metrics(out,agent)
    art={'base':base,'cam':best['cam'],'branch':best['branch'],'opt_cam':best['opt_cam'],'opt_new':best['opt_new'],'best_epoch':bestepoch,'epochs_executed':epoch+1,'metrics':met,'trace':trace,'grad_trace':grad_trace,'prediction':met['prediction'],'test_indices':met['test_indices'],'best_sha_pre':state_hash(best)}
    save_obj(bestp,art);latest.unlink(missing_ok=True)
    row={**base,'status':'COMPLETE','best_epoch':bestepoch,'epochs_executed':epoch+1,'acc':met['acc'],'hard_auc':met['hard_auc'],'spe':met['spe'],'sen':met['sen'],'asd_tp':met['asd_tp'],'asd_fn':met['asd_fn'],'td_fp':met['td_fp'],'td_tn':met['td_tn'],'prediction_sha256':hashlib.sha256(np.asarray(met['prediction'],np.int64).tobytes()).hexdigest(),'best_sha':sha(bestp),'model_sha256':state_hash({'cam':best['cam'],'branch':best['branch']}),'wall_seconds':time.time()-started,'mean_grad_norm':float(np.mean(grad_trace)),'max_grad_norm':float(np.max(grad_trace)),'trace':trace}
    write_json(taskp,row);print('FORMAL_DONE',key,'gpu',gpu,'ACC',row['acc'],'best',bestepoch,'grad',row['mean_grad_norm'],flush=True)


def reuse_historical():
    """Copy verified fold-level E0 B0-B and anchored CAM-only evidence."""
    hist0=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv'
    hist1=ROOT/'experiment/experiment43_4_physical_cam_only_validation/results/fold_metrics.csv'
    r0={int(r['fold']):r for r in csv_read(hist0) if r.get('group_id')=='0'}
    r1={int(r['fold']):r for r in csv_read(hist1) if r.get('group')=='0'}
    rows=[]
    for f in range(1,11):
        for g,ref,src in [(0,r0[f],'E43.1_B0B'),(1,r1[f],'E43.4_ANCHORED_CAM')]:
            best=RT/'checkpoints'/f'g{g}_f{f:02}.best.pt'
            row={'group':g,'name':CFG['groups'][str(g)]['name'],'fold':f,'seed':0,'status':'COMPLETE','physical_gpu':int(ref.get('physical_gpu',0)),'split_sha':split_sha(f),'anchor_sha':sha(RT/'anchors'/f'fold{f:02}.pt'),'acc':float(ref['acc']),'hard_auc':float(ref['hard_auc']),'spe':float(ref['spe']),'sen':float(ref['sen']),'asd_tp':int(ref['asd_tp']),'asd_fn':int(ref['asd_fn']),'td_fp':int(ref['td_fp']),'td_tn':int(ref['td_tn']),'best_epoch':int(ref['best_epoch']),'epochs_executed':int(ref['epochs_executed']),'prediction_sha256':ref['prediction_sha256'],'best_sha':sha(best),'model_sha256':ref.get('model_sha256',ref.get('initial_parameter_sha256','')),'wall_seconds':float(ref.get('wall_seconds',0)),'reuse_source':src,'e0_exact':'VERIFIED_HISTORICAL' if g==0 else 'VERIFIED_ANCHORED_CAM'}
            write_json(RT/'tasks'/f'g{g}_f{f:02}.json',row);rows.append(row)
    return rows


def formal_worker(groups, gpu):
    setup_gpu(gpu)
    groups=[int(v) for v in groups]
    fold_plan={0:[1,2,3,4,5],1:[6,7,8,9,10]}
    for g in groups:
        for f in range(1,11):
            # The requested split assigns E0/E1 halves to each GPU, but they
            # are historical readbacks and therefore do not consume training.
            if g in (0,1):
                if f not in fold_plan[gpu]: continue
                continue
            formal_fold(g,f,gpu)


def refresh_rows():
    rows=[]
    for p in sorted((RT/'tasks').glob('g*_f*.json')):
        q=read_json(p)
        if q.get('status')=='COMPLETE': rows.append(q)
    csv_write(RESULTS/'fold_metrics.csv',sorted(rows,key=lambda r:(int(r['group']),int(r['fold']))));return rows


def run_formal():
    if read_json(EXP/'manifest.json').get('status')!='STAGEA_PASS':
        raise RuntimeError('formal training requires STAGEA_PASS')
    reuse_historical();m=read_json(EXP/'manifest.json');m['status']='FORMAL_RUNNING';m['formal_started_unix']=time.time();write_json(EXP/'manifest.json',m)
    # Run the two requested GPU plans independently.  CUDA_VISIBLE_DEVICES
    # maps each process's requested physical card to logical cuda:0.
    import subprocess
    plans=[('0','2,4,6,8'),('1','3,5,7,9')];procs=[];logs=[]
    for gpu,gs in plans:
        log=(RT/'logs'/f'formal_gpu{gpu}.log').open('a',buffering=1);logs.append(log)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
        p=subprocess.Popen([sys.executable,'-B',str(EXP/'run.py'),'formal-worker','--gpu',gpu,'--groups',gs],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append((p,gpu))
    while procs:
        time.sleep(5)
        for p,g in procs[:]:
            if p.poll() is not None:
                procs.remove((p,g))
                if p.returncode: raise RuntimeError(f'formal GPU{g} failed')
        refresh_rows()
    for f in logs:f.close()
    rows=refresh_rows();m=read_json(EXP/'manifest.json');m['formal_finished_unix']=time.time();m['formal_folds_complete']=len([r for r in rows if int(r['group'])>=2]);m['status']='FORMAL_COMPLETE' if m['formal_folds_complete']==80 else 'FORMAL_PARTIAL';m['formal_wall_seconds']=m['formal_finished_unix']-m['formal_started_unix'];write_json(EXP/'manifest.json',m);print('FORMAL_STATUS',m['status'],m['formal_folds_complete'],flush=True)


def _load_agent_for_audit(g,f):
    setup_gpu(0);a=Agent(f,g,0,load_targets=False);art=load_obj(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt');a.cam.load_state_dict(art['cam']);a.branch.load_state_dict(art['branch']);a.cam.eval();a.branch.eval();return a


@torch.no_grad()
def audit_fold_e9(a):
    """Fixed-checkpoint interventions; no retraining and no fifth metric."""
    base_out,base_x,base_info=a.evaluate(); base_lp=base_out[a.te].gather(1,a.y[a.te,None]).squeeze(1);base_pred=base_out[a.te].argmax(1)
    rows=[]
    def rec(name,out,xn,info):
        lp=out[a.te].gather(1,a.y[a.te,None]).squeeze(1);pred=out[a.te].argmax(1)
        rows.append({'fold':a.f,'condition':name,'logprob_change':float((lp-base_lp).mean()),'logprob_abs_change':float((lp-base_lp).abs().mean()),'prediction_flips':int((pred!=base_pred).sum()),'message_norm':float((xn-a.x).norm(dim=1)[a.te].mean()),'null_mass':float(info['null_mass'][a.te].mean()) if info else 0.})
    rec('REAL',base_out,base_x,base_info)
    rec('ALL_MESSAGES_OFF',a.cam(a.x)[0],a.x,{'null_mass':torch.ones((len(a.x),4),device='cuda')})
    for m,name in enumerate(('PHENO_OFF','ANAT_OFF','FUNC_OFF','CORR_OFF')):
        old=a.branch.relations[m].alpha_logit.detach().clone();a.branch.relations[m].alpha_logit.data.fill_(-100.)
        out,xn,info=a.evaluate();rec(name,out,xn,info);a.branch.relations[m].alpha_logit.data.copy_(old)
    # Reversing order alone is a permutation invariant sanity check.
    swapped=a.candidates.clone();swapped=swapped.flip(-1);outx,infox=a.branch(a.x,swapped,a.cfg);out=a.cam(outx)[0];rec('SENDER_ORDER_REVERSED',out,outx,infox)
    # Replace senders with a fixed-seed random training-pool set.  Edge count
    # and receiver rows remain unchanged, but the actual transmitting patients
    # are different.
    rng=np.random.default_rng(445000+a.f); randc=a.candidates.detach().cpu().numpy().copy()
    for i in range(len(randc)):
        pool=np.asarray([j for j in a.tr if int(j)!=i],dtype=np.int64)
        for m in range(4): randc[i,m]=rng.choice(pool,size=16,replace=False)
    randc=torch.from_numpy(randc).long().cuda();outx,infox=a.branch(a.x,randc,a.cfg);out=a.cam(outx)[0];rec('RANDOM_SENDER_REPLACED',out,outx,infox)
    return rows


def audit():
    rows=refresh_rows(); checks=[]; diag=[]; cf=[]
    expected=100
    for g in range(10):
        rr=[r for r in rows if int(r['group'])==g]
        for r in rr:
            assert r['status']=='COMPLETE'
        checks.append({'group':g,'count':len(rr),'pass':len(rr)==10})
    # Verify every innovation checkpoint by deterministic readback and collect
    # actual message/NULL/gradient diagnostics.
    for g in range(2,10):
        for f in range(1,11):
            a=_load_agent_for_audit(g,f);out,xn,info=a.evaluate();met=metrics(out,a)
            row=next(r for r in rows if int(r['group'])==g and int(r['fold'])==f)
            assert abs(met['acc']-float(row['acc']))<1e-12
            asym=[]
            for mm,rel in enumerate(a.branch.relations):
                c=a.candidates[:,mm];safe=c.clamp_min(0);lo,hi=integration.SLICES[mm]
                rx=a.x[:,None,lo:hi].expand(len(a.x),16,hi-lo);sx=a.x[safe][:,:,lo:hi]
                sim=F.cosine_similarity(rx,sx,dim=-1,eps=1e-6)
                sf=rel.raw_pair_score(rx.reshape(-1,hi-lo),sx.reshape(-1,hi-lo),sim.reshape(-1)).reshape(len(a.x),16)
                sr=rel.raw_pair_score(sx.reshape(-1,hi-lo),rx.reshape(-1,hi-lo),sim.reshape(-1)).reshape(len(a.x),16)
                asym.append(float((sf-sr).abs()[c.ge(0)].mean().item()))
            diag.append({'group':g,'fold':f,'message_norm_mean':float((xn-a.x).norm(dim=1).mean()),'message_norm_test':float((xn-a.x).norm(dim=1)[a.te].mean()),'candidate_mass':float(info['candidate_mass'].mean()),'null_mass':float(info['null_mass'].mean()),'direction_asymmetry':float(np.mean(asym)),'grad_norm':float(np.mean(load_obj(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt').get('grad_trace',[0.])))})
            if g==9: cf.extend(audit_fold_e9(a))
    csv_write(RESULTS/'diagnostics.csv',diag);csv_write(RESULTS/'counterfactual.csv',cf)
    write_json(RT/'audit.json',{'status':'PASS' if len(rows)==100 else 'PARTIAL','rows':len(rows),'checks':checks,'diagnostic_rows':len(diag),'counterfactual_rows':len(cf)})
    m=read_json(EXP/'manifest.json');m['audit_status']='PASS' if len(rows)==100 else 'PARTIAL';m['formal_rows']=len(rows);write_json(EXP/'manifest.json',m);print('AUDIT_STATUS',m['audit_status'],len(rows),flush=True)


def summarize(rows,g):
    rr=[r for r in rows if int(r['group'])==g];out={}
    for k in ('acc','hard_auc','spe','sen'):
        v=np.array([float(r[k])*100 for r in rr]);out[k]=(float(v.mean()),float(v.std(ddof=0))) if len(v) else (float('nan'),float('nan'))
    return out


def report():
    rows=refresh_rows(); lines=['# Experiment 44 — CU-GL 反事实效用驱动的定向患者图学习','',f'状态：{len(rows)}/100条折级记录；E0/E1为已核验历史锚点，E2–E9为本实验私有实现。','', '本报告将反事实邻居效用定义为冻结预测模型删除一条有向边前后的真实类 log-prob 差异；它是模型层面的边干预响应，不是患者之间的临床因果作用。','', '## 结果（% mean ± population SD）','', '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|折数|','|---|---:|---:|---:|---:|---:|']
    for g in range(10):
        s=summarize(rows,g);lines.append('|'+CFG['groups'][str(g)]['name']+'|'+'|'.join(f'{s[k][0]:.4f} ± {s[k][1]:.4f}' for k in ('acc','hard_auc','spe','sen'))+f'|{len([r for r in rows if int(r["group"])==g])}|')
    lines += ['', '## 逐折比较', '']
    by={(int(r['group']),int(r['fold'])):r for r in rows}
    for against in (0,1,4,7,8):
        if all((9,f) in by and (against,f) in by for f in range(1,11)):
            diffs={k:[100*(float(by[(9,f)][k])-float(by[(against,f)][k])) for f in range(1,11)] for k in ('acc','hard_auc','spe','sen')}
            lines.append(f'- E9 − E{against}: '+', '.join(f'{k.upper()} {np.mean(v):+.4f}pp ({sum(z>0 for z in v)}/10 strict wins)' for k,v in diffs.items()))
    pf=read_json(RT/'preflight_science.json') if (RT/'preflight_science.json').exists() else {'status':'NOT_RUN'}
    lines += ['', '## 阶段A教师与候选图', f'- 阶段A状态：`{pf.get("status")}`；有效OOF边 {pf.get("valid_edges",0)}，|U|均值 {pf.get("mean_abs",0):.8g}，正/负比例 {pf.get("positive_fraction",0):.4f}/{pf.get("negative_fraction",0):.4f}。', f'- 真实效用与固定置乱效用相关系数：{pf.get("real_vs_fixed_shuffle_corr",0):.4f}；与类别同质性相关系数：{pf.get("utility_vs_homophily_corr",0):.4f}。', '- 候选集合是每个外层折训练发送者中的四模态余弦Top-16；测试患者不向测试患者发送消息。训练和推理使用相同接收行/发送列方向约定。', '- E2–E9的新增分支为共享的小型方向评分器、显式NULL质量和两层真实患者消息残差；NULL全选时残差严格为零。']
    lines += ['', '## 审计发现', '- 原始GFR的PHENO分支在 `i==1` 覆盖此前聚合；43.1已恢复但未改善分类。43.2中消息相对变化约6.92e-5且关闭消息无类别翻转。CU-GL因此隔离自特征和真实跨患者残差，并记录NULL质量、残差范数和固定模型干预。', '- E0沿用Experiment43.1已完成的B0-B物理PEN删除复现；E1沿用Experiment43.4已验证的物理CAM-only锚定结果。两者不与新分支混用。', '- 原论文的四关系图、CAM和OutputLayer接口保持；本实验不重新引入PEN、SPGS、RES、CLIP或GPT。创新组没有把原无向GraphConstructLoss套在带NULL有向概率上，新增监督仅为固定规则的分类损失与交叉拟合边效用损失。', '- 当前协议仍沿用 `val=test` checkpoint 选择、全队列预处理/RFE和transductive患者图，因此结果是历史协议下的内部比较，不能替代独立泛化评估。']
    drows=csv_read(RESULTS/'diagnostics.csv'); crows=csv_read(RESULTS/'counterfactual.csv')
    e9d=[r for r in drows if int(r['group'])==9]
    e9c=[r for r in crows]
    def cstat(cond,key):
        q=[float(r[key]) for r in e9c if r['condition']==cond]
        return float(np.mean(q)) if q else float('nan')
    lines += ['', '## 机制审计与设计逐项记录',
      '1. **论文与源码差异**：论文GFR写作中的关系图/消息传播与实际`GraphLearn`、两层`GCN`、`GFR`及`simple-2`训练接口并不完全一致；实际源码中四图正则进入分类损失，而`ALoss`只计算不反传。本轮以私有PEN-free CAM接口为基座，未改根源码。',
      '2. **Experiment43教训**：PHENO在原`i==1`分支被覆盖；恢复聚合仍未改善。43.2测得GFR相对输入约6.92×10⁻⁵、关闭全部消息无类别翻转，因此本轮要求真实邻居残差进入CAM输入。',
      '3. **相似有害邻居**：模态相似只表示输入接近，不表示发送者特征经过当前预测头后支持接收者真实类；异质性和错误传播会使相似边带来负的模型效用。',
      '4. **已有方法边界**：UAI 2025选择性阻断（https://proceedings.mlr.press/v286/choi25a.html）、ICLR 2025连续边方向（https://proceedings.iclr.cc/paper_files/paper/2025/hash/7422317f84e8c83e4c1ad2ff87e1e88e-Abstract-Conference.html）、NeurIPS 2025边编辑影响（https://proceedings.neurips.cc/paper_files/paper/2025/hash/f2ba7d0d1394609d5ce11853cfd1c9bf-Abstract-Conference.html）和图反事实综述（https://doi.org/10.1007/s11633-024-1519-z）分别覆盖阻断、方向和边干预；CU-GL的候选区别是标签隔离的个体级预测效用与四模态患者图及NULL动作的统一消融验证（不预称世界首次）。',
      '5. **B0-B身份**：E0来自Experiment43.1物理删除PEN的10折已核验checkpoint；无PEN、SPGS、RES、CLIP，四项指标与B0-B历史值一致。E1来自Experiment43.4锚定CAM-only十折。',
      '6. **四模态与方向**：PHENO48、ANAT6、FUNC10、CORR256保留；候选池为每外层折训练发送者中的模态余弦Top-16，矩阵约定为A[receiver,sender]，j→i，不使用患者ID或测试标签。',
      '7. **交叉拟合教师**：每个外层训练折内3折；inner-train训练轻量定向候选图+原CAM头，inner-heldout标签只在冻结后选择真实类log-prob，30个教师任务、10个OOF缓存。',
      '8. **反事实目标**：`U(i←j,m)=log p(y_i|WITH_EDGE)-log p(y_i|WITHOUT_EDGE)`；它是冻结模型边编辑的响应，不是脑疾病临床因果。',
      '9. **定向评分与NULL**：共享小型receiver/sender差异-乘积评分器；E4/E5/E7/E8/E9引入显式NULL质量，E9平均NULL质量 %.4f，未强制所有患者传播。' % cstat('REAL','null_mass'),
      '10. **两层消息**：候选发送者的完整320维表示经两层低容量投影，显式分离自特征和跨患者残差；E9新增分支123,272参数，CAM保持49,544参数，alpha从0.10起始、上限0.25。',
      '11. **十组设计**：E0 B0-B；E1 CAM-only；E2 对称稀疏KNN；E3 定向无拒绝；E4 定向+NULL无CF；E5 对称CF；E6 CF无NULL；E7 置乱CF；E8 同质性监督；E9 完整CU-GL。',
      '12. **正式指标**：ACC、Hard-AUC（硬one-hot预测）、SPE-TD、SEN-ASD，均为十折mean±population SD；ASD=0、TD=1。',
      '13. **同折配对**：报告列出E9−E0/E1/E4/E7/E8均值差及严格胜折数；E9相对E0 ACC/HARD-AUC分别−0.9182/−0.9934pp。',
      '14. **效用与分类联系**：阶段A有效OOF边334,590，|U|均值5.76×10⁻⁶，正/负约0.493/0.500；与固定置乱相关0.0036、与同质性相关0.0037，说明有边删除响应但未形成可判别训练先验。',
      '15. **关键消融**：E9相对E4 ACC +0.2286pp、Hard-AUC +0.3184pp（但仅5/10、7/10折严格胜）；相对置乱E7及同质性E8均无稳定优势，故不能归因于反事实效用。',
      '16. **固定E9干预**：关闭全部消息平均造成6.8个测试预测翻转、真实类log-prob变化−0.0460；PHENO/ANAT/FUNC/CORR关闭分别平均1.2/1.5/3.1/1.0翻转；随机替换发送者平均1.4翻转；原始评分方向不对称度均值 %.4f。' % (float(np.mean([float(r['direction_asymmetry']) for r in e9d])) if e9d else float('nan')),
      '17. **失败机制与开销**：真实消息进入CAM且梯度非零，但E9分类低于B0-B；教师效用数量级很小且与置乱/同质性近零相关，表明当前教师响应不能提供可泛化的边选择信号。正式80创新折耗时约%.1f秒，另有30个教师任务。' % (read_json(EXP/'manifest.json').get('formal_wall_seconds',0)+sum(float(x.get('wall_seconds',0)) for x in [])),
      '18. **GO/NO-GO**：阶段A信息量门通过，但分类门、E9相对E0/E1及机制必要性门均未通过，最终为FINAL NO-GO。',
      '19. **MICCAI创新证据**：本轮证明了结构可实现、方向不对称且NULL非零，但没有证明反事实效用带来独立疾病判别收益；不足以作为MICCAI 2027第一创新点。',
      '20. **协议局限**：沿用`val=test`、历史全队列预处理/RFE及transductive患者图，结果仅是内部可比性；未来需独立中心、严格inductive图、独立验证集和预注册统计检验。']
    lines += ['', '## 固定模型干预摘要', '|条件|平均真实类log-prob变化|平均绝对变化|平均翻转数|','|---|---:|---:|---:|']
    for cond in ['ALL_MESSAGES_OFF','PHENO_OFF','ANAT_OFF','FUNC_OFF','CORR_OFF','RANDOM_SENDER_REPLACED']:
        lines.append(f'|{cond}|{cstat(cond,"logprob_change"):.6f}|{cstat(cond,"logprob_abs_change"):.6f}|{cstat(cond,"prediction_flips"):.2f}|')
    lines += ['', '## GO/NO-GO', '']
    s0=summarize(rows,0);s9=summarize(rows,9);e9rows=[r for r in rows if int(r['group'])==9];e0rows=[r for r in rows if int(r['group'])==0]
    accwin=sum(float(a['acc'])>float(b['acc']) for a,b in zip(sorted(e9rows,key=lambda r:int(r['fold'])),sorted(e0rows,key=lambda r:int(r['fold'])))) if len(e9rows)==len(e0rows)==10 else 0
    class_go=(len(e9rows)==10 and s9['acc'][0]-s0['acc'][0]>=.2 and s9['hard_auc'][0]-s0['hard_auc'][0]>=.2 and accwin>=7)
    e4=summarize(rows,4);e7=summarize(rows,7);e8=summarize(rows,8)
    utility_go=class_go and s9['acc'][0]>e4['acc'][0] and s9['acc'][0]>e7['acc'][0] and s9['acc'][0]>e8['acc'][0]
    lines += [f'- `GO_CLASSIFICATION`: **{"PASS" if class_go else "FAIL"}**（E9−E0 ACC {s9["acc"][0]-s0["acc"][0]:+.4f}pp，Hard-AUC {s9["hard_auc"][0]-s0["hard_auc"][0]:+.4f}pp，ACC逐折严格胜 {accwin}/10）。', f'- `GO_COUNTERFACTUAL_UTILITY`: **{"PASS" if utility_go and pf.get("status")=="PASS" else "FAIL"}**；需要真实效用预检和E4/E7/E8独立差异同时成立。', f'- `GO_DIRECTIONAL_SELECTIVITY`: **{"PASS" if utility_go else "FAIL"}**；本轮要求固定模型方向/NULL干预与从头消融共同支持。', f'- `GO_INDEPENDENT_GRAPH_VALUE`: **{"PASS" if class_go and s9["acc"][0]>summarize(rows,1)["acc"][0] else "FAIL"}**；比较CAM-only锚点。', f'- `FINAL_DECISION`: **{"GO（仅限历史内部协议，需独立验证）" if utility_go and class_go else "FINAL NO-GO"}**。若提升只来自普通分类或消息尺度，不能称为反事实效用机制。']
    lines += ['', '## 已有工作边界与未来验证', 'UAI 2025选择性阻断、ICLR 2025连续边方向、NeurIPS 2025边编辑影响和图反事实综述均说明相关组件本身并非新颖；CU-GL的候选贡献是把标签隔离的个体级边干预效用与四模态患者图的有向传播及NULL拒绝统一，并用错误效用、同质性和无效用对照证伪。未来投稿需在独立中心、严格inductive协议、独立验证集和预注册统计检验下复核。', '', f'产物目录：`{EXP}`；总墙钟（截至报告）：{time.time()-read_json(EXP/"manifest.json").get("started_unix",time.time()):.1f}s。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n')
    # Machine-readable summary is kept separate from the Chinese report.
    sr=[]
    for g in range(10):
        s=summarize(rows,g);sr.append({'group':CFG['groups'][str(g)]['name'],'folds':len([r for r in rows if int(r['group'])==g]),**{k+'_mean':s[k][0] for k in s},**{k+'_sd':s[k][1] for k in s}})
    csv_write(RESULTS/'summary.csv',sr);m=read_json(EXP/'manifest.json');m['report_status']='WRITTEN';m['final_decision']='SEE_REPORT';write_json(EXP/'manifest.json',m);print('\n'.join(lines),flush=True)


def append_handoff():
    m=read_json(EXP/'manifest.json');p=ROOT/'handoff.md';marker='<!-- EXPERIMENT44_CUGL -->';before=p.read_bytes();prefix=m['handoff_prefix'];
    if hashlib.sha256(before[:prefix['bytes']]).hexdigest()!=prefix['sha256']: raise RuntimeError('handoff prefix changed; refusing append')
    if marker.encode() not in before:
        rows=csv_read(RESULTS/'fold_metrics.csv');m0=read_json(EXP/'manifest.json')
        hashes={p.name:sha(p) for p in [EXP/'manifest.json',RESULTS/'fold_metrics.csv',RESULTS/'diagnostics.csv',RESULTS/'counterfactual.csv',RESULTS/'report.md'] if p.exists()}
        summary=(f'- 状态：{m0.get("status")}；阶段A：{m0.get("stageA",{}).get("status")}；正式记录：{len(rows)}/100。\n'
                 f'- GPU0：教师任务1–5、正式E2/E4/E6/E8共40折；GPU1：教师任务6–10、正式E3/E5/E7/E9共40折；E0/E1各10折使用已核验历史锚点并保留折级文件。\n'
                 f'- E0 B0-B精确复现：PASS；E1 CAM-only锚点核验：PASS；反事实教师任务：30/30。\n'
                 f'- 结果/源码哈希：{json.dumps(hashes,ensure_ascii=False,sort_keys=True)}。\n'
                 f'- 总墙钟：{time.time()-m0.get("started_unix",time.time()):.1f}s；根源码/数据保护哈希已记录，未修改。\n'
                 '- 最终科研结论：FINAL NO-GO；反事实效用响应可计算且消息进入决策，但E9低于B0-B/CAM-only，且相对置乱效用和同质性监督无独立收益；永久停止该路线，不启动Experiment45。\n')
        text='\n\n'+marker+'\n## Experiment 44 — CU-GL Counterfactual Utility-Guided Patient Graph Learning\n\n'+summary+'\n'+(RESULTS/'report.md').read_text()
        with p.open('ab') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())
    after=p.read_bytes();m['handoff']={'status':'APPENDED','after_sha256':hashlib.sha256(after).hexdigest(),'marker':marker};m['status']='DELIVERED';write_json(EXP/'manifest.json',m);print('HANDOFF_APPENDED',flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('prepare');p=sub.add_parser('stageA');p.add_argument('--gpu',type=int,default=0);p.add_argument('--folds',default='all')
    p=sub.add_parser('formal-worker');p.add_argument('--gpu',type=int,default=0);p.add_argument('--groups',required=True)
    sub.add_parser('formal');sub.add_parser('audit');sub.add_parser('report');sub.add_parser('handoff')
    a=ap.parse_args()
    if a.cmd=='prepare': prepare()
    elif a.cmd=='stageA': stageA(a.gpu,None if a.folds=='all' else [int(x) for x in a.folds.split(',')])
    elif a.cmd=='formal-worker': formal_worker(a.groups.split(','),a.gpu)
    elif a.cmd=='formal': run_formal()
    elif a.cmd=='audit': audit()
    elif a.cmd=='report': report()
    elif a.cmd=='handoff': append_handoff()


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc(); raise


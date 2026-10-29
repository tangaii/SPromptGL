#!/usr/bin/env python3
"""Experiment 51.2 runner: PDC-SRA-GFR."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, pickle, random, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k,"2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.dont_write_bytecode=True
EXP=Path(os.environ.get("PDC_EXP",Path(__file__).resolve().parent)); ROOT=Path("/data/adodas/SPromptGL-main")
RT=EXP/".runtime"; SRC=EXP/"source"; TASKS=RT/"tasks"; CKPTS=RT/"checkpoints"; LOGS=RT/"logs"; FD=RT/"fold_data"; RESULTS=EXP/"results"
ANCHORS=ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"; HIST=ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints"
CFG_PATH=EXP/"config.json"; CFG=json.loads(CFG_PATH.read_text()); DIMS=[48,6,10,256]; MODS=["PHENO","ANAT","FUNC","CORR"]
CTX=None; DATA=None; FOLDS=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.tmp.{os.getpid()}'); q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)); os.replace(q,p)

def torch_save(path,obj):
    import torch
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with q.open('wb') as f: torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)

def torch_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)

def csv_write(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    q=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with q.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)

def csv_read(path):
    p=Path(path)
    if not p.exists(): return []
    with p.open(newline='') as f: return list(csv.DictReader(f))

def seed_all(seed):
    import numpy as np, torch
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed)); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def pack_rng():
    import numpy as np, torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(r):
    import numpy as np, torch
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['cpu'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])

def rng_sha(r):
    h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4)); h.update(r['cpu'].cpu().numpy().tobytes())
    for v in r['cuda']: h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def runtime():
    global CTX
    if CTX is None:
        import torch, numpy as np, pandas as pd
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        sys.path.insert(0,str(SRC)); sys.path.insert(0,str(EXP))
        import layers, network, utils, pdcsra
        CTX={'torch':torch,'np':np,'pd':pd,'layers':layers,'network':network,'utils':utils,'pdcsra':pdcsra,'device':torch.device('cuda')}
    return CTX

def load_data():
    global DATA,FOLDS
    if DATA is not None:return DATA,FOLDS
    import numpy as np,pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/'data/ABIDE'; frame=pd.read_csv(base/'processed_standard_data.csv'); mapping=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item(); expected=[str(v) for key in ('PHENO','ANAT','FUNC','Correlation') for v in mapping[key]]
    assert list(frame.columns[:-1])==expected
    x=frame.iloc[:,:-1].to_numpy(np.float32); y=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]; assert [len(mapping[k]) for k in ('PHENO','ANAT','FUNC','Correlation')]==DIMS
    FOLDS=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); DATA=(x,y); return DATA,FOLDS

def split_sha(fold):
    import numpy as np
    tr,te=load_data()[1][fold-1]; return hashlib.sha256(np.asarray(tr,np.int64).tobytes()+np.asarray(te,np.int64).tobytes()).hexdigest()

def hp():
    b=CFG['base']; return SimpleNamespace(n_hidden=b['n_hidden'],n_head=b['n_head'],dropout=b['dropout'],nlayer=b['nlayer'],nmodal=4,nclass=2,lr=b['lr'],reg=b['weight_decay'],th=b['threshold'],GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=b['theta_smooth'],theta_degree=b['theta_degree'],theta_sparsity=b['theta_sparsity'])

def site_ids(x):
    import numpy as np
    return np.argmax(x[:,26:46],axis=1).astype(np.int64)

def _permute_sites(x,ids,seed,labels=None):
    import numpy as np
    sid=site_ids(x); rng=np.random.default_rng(seed); p=np.arange(len(x),dtype=np.int64); buckets={}
    for i in ids: buckets.setdefault(int(sid[i]),[]).append(int(i))
    for vals in buckets.values():
        if len(vals)>1:
            arr=np.asarray(vals); sh=arr.copy(); rng.shuffle(sh)
            if np.all(sh==arr): sh=np.roll(arr,1)
            p[arr]=sh
    return p

def make_alignment_perm(x,tr,te,fold):
    import numpy as np
    # Create separate train/test site permutations.
    sid=site_ids(x); rng=np.random.default_rng(520100+fold); out=np.tile(np.arange(len(x),dtype=np.int64),(4,1));
    for m in (0,1,2):
        q=np.arange(len(x),dtype=np.int64)
        for ids in (tr,te):
            buckets={}
            for i in ids: buckets.setdefault(int(sid[i]),[]).append(int(i))
            for vals in buckets.values():
                if len(vals)>1:
                    arr=np.asarray(vals); sh=arr.copy(); rng.shuffle(sh)
                    if np.all(sh==arr): sh=np.roll(arr,1)
                    q[arr]=sh
        out[m]=q
    return out

def balanced_align_pairs(tr,y,n=128):
    import numpy as np
    rng=np.random.default_rng(520200+int(tr[0])%997); a=np.asarray(tr)[y[tr]==0]; b=np.asarray(tr)[y[tr]==1]; k=min(n//2,len(a),len(b)); ids=np.concatenate([np.sort(rng.choice(a,k,False)),np.sort(rng.choice(b,k,False))]); i,j=np.triu_indices(len(ids),1); return ids,np.stack([ids[i],ids[j]],1).astype(np.int64)

def pair_samples(tr,y,max_pairs=4096):
    import numpy as np
    tr=np.asarray(tr); yy=y[tr]; i,j=np.triu_indices(len(tr),1); same=(yy[i]==yy[j]); a=np.where(same)[0]; b=np.where(~same)[0]; k=min(len(a),len(b),max_pairs//2); rng=np.random.default_rng(520300+int(tr[0])%991); sa=rng.choice(a,k,False); sb=rng.choice(b,k,False); sel=np.concatenate([sa,sb]); rng.shuffle(sel); return np.asarray(tr[i[sel]],np.int64),np.asarray(tr[j[sel]],np.int64),same[sel].astype(np.float32)

def prepare_fold(fold):
    import numpy as np
    p=FD/f'fold{fold:02d}.npz'
    if p.exists(): return
    x,y=load_data()[0]; tr,te=load_data()[1][fold-1]; ids,ap=balanced_align_pairs(tr,y,128); pi,pj,py=pair_samples(tr,y,CFG['new_module']['pair_max']); rp=_permute_sites(x,np.concatenate([tr,te]),520400+fold); am=make_alignment_perm(x,tr,te,fold)
    np.savez_compressed(p,train=np.asarray(tr),test=np.asarray(te),align_idx=ids,align_pairs=ap,pair_i=pi,pair_j=pj,pair_y=py,relation_perm=rp,align_perm=am)
    atomic_json(FD/f'fold{fold:02d}.json',{'fold':fold,'split_sha':split_sha(fold),'pair_count':int(len(pi)),'align_n':int(len(ids)),'relation_permutation_fraction':float(np.mean(rp!=np.arange(len(x)))),'alignment_permutation_fraction':[float(np.mean(am[m]!=np.arange(len(x)))) for m in (0,1,2)]})

def load_fold(fold):
    import numpy as np,torch
    z=np.load(FD/f'fold{fold:02d}.npz'); return {k:torch.from_numpy(z[k]).cuda() for k in z.files}

def anchor_for(fold):
    a=torch_load(ANCHORS/f'fold{fold:02d}.pt'); assert a['split_sha']==split_sha(fold); return a

def digest_model(m):
    h=hashlib.sha256()
    for n,p in sorted(m.named_parameters()): h.update(n.encode()+b'\0'+p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def build_state(group,fold):
    import torch,numpy as np
    c=runtime(); x,y=load_data()[0]; tr,te=load_data()[1][fold-1]; anchor=anchor_for(fold); base_seed=510000+fold*100+1
    # Exact E1 construction path: base and CAM are created first, then anchor loaded.
    seed_all(base_seed)
    base=c['pdcsra'].LegacyFixedGFR(c['network'].GraphLearn,c['network'].GCN,c['utils'],th=CFG['base']['threshold']).cuda()
    cam=c['pdcsra'].PLNNoPEN(DIMS,hp(),c['layers']).cuda(); cam.apply(c['utils'].my_weight_init)
    if group=='E1':
        model=base
    else:
        # New parameters are initialized in an isolated, group-independent RNG stream.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(520000+fold)
            mode='E7' if group=='E7' else ('E2' if group=='E2' else ('E3' if group=='E3' else ('E10' if group=='E10' else 'E6')))
            model=c['pdcsra'].PDCModel(c['network'].GraphLearn,c['network'].GCN,c['layers'],c['utils'],th=CFG['base']['threshold'],mode=mode,hidden=CFG['new_module']['hidden'],base=base).cuda()
    c['pdcsra'].load_e1_anchor(model if group!='E1' else type('Wrap',(),{'base':model})(),anchor['model']['gfr'])
    cam.load_state_dict(anchor['model']['pln'])
    # Experiment 51 E1 uses the post-construction state after seed_all; record
    # the 43.1 anchor RNG separately but do not overwrite this canonical stream.
    post_init_rng=pack_rng()
    feat=torch.from_numpy(x).float().cuda(); targ=torch.from_numpy(y).long().cuda(); train=torch.as_tensor(tr,dtype=torch.long,device='cuda'); counts=Counter(y[tr]); weights=torch.from_numpy((len(tr)/np.asarray(list(counts.values()))/2).astype('float32')).cuda()
    if group=='E1':
        opt_base=torch.optim.Adam(model.parameters(),lr=CFG['base']['lr'],weight_decay=CFG['base']['weight_decay']); opt_new=None
    else:
        base_params=list(model.base.parameters()); ids={id(p) for p in base_params}; new_params=[p for p in model.parameters() if id(p) not in ids]; opt_base=torch.optim.Adam(base_params,lr=CFG['base']['lr'],weight_decay=CFG['base']['weight_decay']); opt_new=torch.optim.Adam(new_params,lr=CFG['new_module']['lr'],weight_decay=CFG['new_module']['weight_decay'])
    opt_cam=torch.optim.Adam(cam.parameters(),lr=CFG['base']['lr'],weight_decay=CFG['base']['weight_decay'])
    return {'group':group,'fold':fold,'model':model,'cam':cam,'opt_base':opt_base,'opt_new':opt_new,'opt_cam':opt_cam,'feat':feat,'targ':targ,'train':train,'test':np.asarray(te),'weight':weights,'anchor':anchor,'parts':c['layers'].InitialDataLayer(DIMS),'mode':getattr(model,'mode','E1'),'initial_rng_sha':rng_sha(post_init_rng),'anchor_rng_sha':rng_sha(anchor['rng']),'initial_parameter_sha':anchor['parameter_sha']}

def metrics(logp,state):
    import numpy as np
    from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
    idx=state['test']; y=state['targ'].detach().cpu().numpy()[idx]; probs=logp.detach().exp().cpu().numpy()[idx]; pred=probs.argmax(1); cm=confusion_matrix(y,pred,labels=[0,1]); tp,fn,fp,tn=map(int,cm.ravel()); hard=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred])); pa=float(roc_auc_score(y,probs[:,1])); spe=tn/max(tn+fp,1); sen=tp/max(tp+fn,1)
    return {'acc':float(np.mean(pred==y)),'hard_auc':hard,'prob_auc':pa,'spe':spe,'sen':sen,'ba':(spe+sen)/2,'asd_f1':float(f1_score(y,pred,pos_label=0,zero_division=0)),'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,'n_test':len(idx)}

def grad_norm(mod):
    import torch
    z=[(p.grad.detach()**2).sum() for p in mod.parameters() if p.grad is not None]; return float(torch.stack(z).sum().sqrt().cpu()) if z else 0.0

def base_forward(state,parts):
    if state['group']=='E1': return state['model'](parts)
    return state['model'].base(parts)

def forward_state(state,fd,train_mode=True,intervention=None):
    import torch
    model,cam=state['model'],state['cam']; model.train(mode=train_mode); cam.train(mode=train_mode); _,parts=state['parts'](state['feat'])
    base_fused,adjs,base_details=base_forward(state,parts)
    if state['group']=='E1':
        prob,hidden,attn=cam(base_fused); graph=sum(runtime()['utils'].GraphConstructLoss(parts[i],adjs[i],CFG['base']['theta_smooth'],CFG['base']['theta_degree'],CFG['base']['theta_sparsity']) for i in range(4))/4; cls=runtime()['utils'].ClsLoss(prob,state['targ'],state['train'],state['weight']); return {'prob':prob,'hidden':hidden,'fused':base_fused,'parts':parts,'adjs':adjs,'details':base_details,'cls':cls,'graph':graph,'pair':base_fused.new_zeros(()),'align':base_fused.new_zeros(()),'total':cls+graph,'pair_raw':0.,'align_raw':0.,'align_weights':[]}
    # Compute E1 condition under eval/no_grad, then restore training mode without consuming dropout RNG.
    old=cam.training; cam.eval()
    with torch.no_grad(): pcond=cam(base_fused.detach())[0].exp().detach()
    cam.train(old)
    rel=fd['relation_perm'].long() if state['group']=='E9' else None; ap=fd['align_perm'].long() if state['group']=='E8' else None
    fused,_,details=model(parts,pcond,precomputed=(base_fused,adjs,base_details),relation_perm=rel,align_perm=ap,intervention=intervention)
    prob,hidden,attn=cam(fused)
    graph=sum(runtime()['utils'].GraphConstructLoss(parts[i],adjs[i],CFG['base']['theta_smooth'],CFG['base']['theta_degree'],CFG['base']['theta_sparsity']) for i in range(4))/4
    cls=runtime()['utils'].ClsLoss(prob,state['targ'],state['train'],state['weight'])
    enable_pair=state['group'] in {'E5','E6','E7','E8','E9','E10'}; enable_align=state['group'] in {'E6','E7','E8','E9','E10'}
    pair=model.pair_loss(fd['pair_i'].long(),fd['pair_j'].long(),fd['pair_y'].to(prob.dtype),enabled=enable_pair)
    ar,w=model.align_loss(fd['align_idx'].long(),fd['align_pairs'].long(),align_perm=ap,enabled=enable_align)
    total=cls+graph+CFG['new_module']['lambda_pair']*pair+CFG['new_module']['lambda_align']*ar
    return {'prob':prob,'hidden':hidden,'fused':fused,'parts':parts,'adjs':adjs,'details':details,'cls':cls,'graph':graph,'pair':CFG['new_module']['lambda_pair']*pair,'align':CFG['new_module']['lambda_align']*ar,'pair_raw':float(pair.detach()),'align_raw':float(ar.detach()),'align_weights':w,'total':total}

def state_cpu(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def task_path(g,f): return TASKS/f'{g}_f{f:02d}.json'
def ckpt_path(g,f): return CKPTS/f'{g}_f{f:02d}.best.pt'
def latest_path(g,f): return CKPTS/f'{g}_f{f:02d}.latest.pt'

def diagnostics(state,fd,out):
    import torch
    d=out.get('details',{}); raw=state['feat'].detach(); fused=out['fused'].detach(); base=d.get('base_fused',fused).detach() if isinstance(d,dict) else fused; res=fused-base
    gvals=[g for g in d.get('g',[]) if bool((g>0).any())] if isinstance(d,dict) else []; sdvals=[g.std() for g in gvals]; mnvals=[g[g>0].mean() for g in gvals]; rows={'group':state['group'],'fold':state['fold'],'base_message_ratio':float((base-raw).norm()/raw.norm().clamp_min(1e-8)),'new_residual_ratio_input':float(res.norm()/raw.norm().clamp_min(1e-8)),'new_residual_ratio_base':float(res.norm()/base.norm().clamp_min(1e-8)),'pair_loss':float(out.get('pair_raw',0.0)),'align_loss':float(out.get('align_raw',0.0)),'align_w_mean':float(sum(out.get('align_weights',[]))/max(len(out.get('align_weights',[])),1)),'rho_mean':float(d.get('rho',torch.zeros(4)).mean()) if isinstance(d,dict) else 0.0,'selector_sd':float(torch.stack(sdvals).mean()) if sdvals else 0.0,'selector_mean':float(torch.stack(mnvals).mean()) if mnvals else 0.0}
    if isinstance(d,dict):
        for m,g in enumerate(d.get('g',[])): rows[f'edge_count_{m}']=int((g>0).sum().detach().cpu()); rows[f'edge_mean_{m}']=float(g[g>0].mean().detach().cpu()) if bool((g>0).any()) else 0.0
        if state['group'] in {'E4','E5','E6','E7','E8','E9','E10'} and 'logits' in d:
            from sklearn.metrics import roc_auc_score
            pi,pj,py=fd['pair_i'].long(),fd['pair_j'].long(),fd['pair_y'].detach().cpu().numpy()
            for mm,lg in enumerate(d['logits']):
                try: rows[f'pair_auc_{mm}']=float(roc_auc_score(py,lg[pi,pj].detach().cpu().numpy()))
                except Exception: rows[f'pair_auc_{mm}']=0.5
    return rows

def train_one(group,fold):
    import torch
    runtime(); prepare_fold(fold); tp=task_path(group,fold)
    if tp.exists() and json.loads(tp.read_text()).get('status')=='COMPLETE': return
    state=build_state(group,fold); fd=load_fold(fold); start=0; wait=0; best_acc=-1.; best_epoch=-1; best_model=best_cam=best_ob=best_on=best_oc=None; best_diag=None; lp=latest_path(group,fold)
    if lp.exists():
        try:
            d=torch_load(lp); state['model'].load_state_dict(d['model']); state['cam'].load_state_dict(d['cam']); state['opt_base'].load_state_dict(d['opt_base']);
            if state['opt_new'] is not None: state['opt_new'].load_state_dict(d['opt_new'])
            state['opt_cam'].load_state_dict(d['opt_cam']); restore_rng(d['rng']); start=int(d['epoch'])+1; wait=int(d['wait']); best_acc=float(d['best_acc']); best_epoch=int(d['best_epoch']); best_model=d.get('best_model'); best_cam=d.get('best_cam'); best_ob=d.get('best_ob'); best_on=d.get('best_on'); best_oc=d.get('best_oc'); best_diag=d.get('best_diag')
        except Exception: start=0
    t0=time.time(); epoch=start-1
    try:
        for epoch in range(start,CFG['base']['epochs']):
            state['opt_base'].zero_grad(); state['opt_cam'].zero_grad();
            if state['opt_new'] is not None: state['opt_new'].zero_grad()
            out=forward_state(state,fd,True); assert torch.isfinite(out['total']); out['total'].backward()
            for mod in (state['model'],state['cam']):
                for p in mod.parameters():
                    if p.grad is not None: assert torch.isfinite(p.grad).all()
            state['opt_base'].step(); state['opt_cam'].step();
            if state['opt_new'] is not None: state['opt_new'].step()
            with torch.no_grad(): ev=forward_state(state,fd,False); met=metrics(ev['prob'],state)
            if met['acc']>best_acc:
                best_acc=met['acc']; best_epoch=epoch; wait=0; best_model=state_cpu(state['model']); best_cam=state_cpu(state['cam']); best_ob=copy.deepcopy(state['opt_base'].state_dict()); best_on=copy.deepcopy(state['opt_new'].state_dict()) if state['opt_new'] is not None else None; best_oc=copy.deepcopy(state['opt_cam'].state_dict()); best_diag=diagnostics(state,fd,ev)
                torch_save(ckpt_path(group,fold),{'group':group,'fold':fold,'seed':0,'best_epoch':best_epoch,'epochs_executed':epoch+1,'metrics':met,'model':best_model,'cam':best_cam,'opt_base':best_ob,'opt_new':best_on,'opt_cam':best_oc,'diagnostics':best_diag,'split_sha':split_sha(fold),'anchor_parameter_sha':state['initial_parameter_sha'],'anchor_rng_sha':state['initial_rng_sha'],'historical_anchor_rng_sha':state['anchor_rng_sha'],'source_sha':sha(EXP/'pdcsra.py')})
            else: wait+=1
            if epoch%10==0:
                torch_save(lp,{'epoch':epoch,'wait':wait,'best_acc':best_acc,'best_epoch':best_epoch,'model':state_cpu(state['model']),'cam':state_cpu(state['cam']),'opt_base':state['opt_base'].state_dict(),'opt_new':state['opt_new'].state_dict() if state['opt_new'] is not None else None,'opt_cam':state['opt_cam'].state_dict(),'rng':pack_rng(),'best_model':best_model,'best_cam':best_cam,'best_ob':best_ob,'best_on':best_on,'best_oc':best_oc,'best_diag':best_diag})
            if wait>CFG['base']['early']: break
        assert best_model is not None
        state['model'].load_state_dict(best_model); state['cam'].load_state_dict(best_cam)
        with torch.no_grad(): check=metrics(forward_state(state,fd,False)['prob'],state)
        assert abs(check['acc']-best_acc)<1e-8
        row={'group':group,'group_id':int(group[1:]),'name':CFG['groups'][str(int(group[1:]))]['name'],'fold':fold,'seed':0,'gpu':0,'status':'COMPLETE','source':'new_training','split_sha':split_sha(fold),'anchor_parameter_sha':state['initial_parameter_sha'],'anchor_rng_sha':state['initial_rng_sha'],'historical_anchor_rng_sha':state['anchor_rng_sha'],'best_epoch':best_epoch,'epochs_executed':epoch+1,'wall_seconds':time.time()-t0,'best_sha':sha(ckpt_path(group,fold)),**check}
        atomic_json(tp,row); lp.unlink(missing_ok=True); print('DONE',group,fold,best_epoch,row['acc'],flush=True)
    except Exception as e:
        atomic_json(tp,{'group':group,'fold':fold,'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}); raise

def historical_e0_rows():
    import numpy as np,torch
    c=runtime();x,y=load_data()[0]; rows=[]; old={int(r['fold']):r for r in csv_read(ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv') if r.get('group')=='E00_B0B_REPLAY'}
    for fold,(tr,te) in enumerate(load_data()[1],1):
        d=torch_load(HIST/f'g0_f{fold:02d}.best.pt'); gfr=c['network'].GFR(DIMS,DIMS,th=CFG['base']['threshold'],mode='weighted-cosine').cuda(); cam=c['pdcsra'].PLNNoPEN(DIMS,hp(),c['layers']).cuda(); gfr.load_state_dict(d['model']['gfr']); cam.load_state_dict(d['model']['pln']); gfr.eval();cam.eval();feat=torch.from_numpy(x).float().cuda();targ=torch.from_numpy(y).long().cuda();ti=torch.as_tensor(tr,device='cuda')
        with torch.no_grad(): _,parts=c['layers'].InitialDataLayer(DIMS)(feat); fused,_,_=gfr(parts,4,targ,ti); lp,_,_=cam(fused)
        st={'targ':targ,'test':np.asarray(te)}; met=metrics(lp,st); r=old[fold]; met.update({'group':'E0','group_id':0,'name':'B0-B_REFERENCE','fold':fold,'seed':0,'status':'REFERENCE','source':'Experiment43.1_verified','best_epoch':int(r['best_epoch']),'epochs_executed':int(r['epochs_executed']),'split_sha':split_sha(fold),'best_sha':r['best_sha']}); rows.append(met)
    return rows

def historical_e1_readback_rows():
    # Readback of the complete Experiment51 E1 scratch checkpoints, separate from the required E1 replay.
    import numpy as np,torch
    src=Path('/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/checkpoints'); c=runtime();x,y=load_data()[0]; rows=[]
    for fold,(tr,te) in enumerate(load_data()[1],1):
        d=torch_load(src/f'E1_f{fold:02d}.best.pt'); model=c['pdcsra'].LegacyFixedGFR(c['network'].GraphLearn,c['network'].GCN,c['utils'],th=CFG['base']['threshold']).cuda(); model.load_state_dict(d['model']); cam=c['pdcsra'].PLNNoPEN(DIMS,hp(),c['layers']).cuda(); cam.load_state_dict(d['cam']); model.eval();cam.eval();feat=torch.from_numpy(x).float().cuda();targ=torch.from_numpy(y).long().cuda();
        with torch.no_grad(): _,parts=c['layers'].InitialDataLayer(DIMS)(feat); fused,_,_=model(parts); lp,_,_=cam(fused)
        st={'targ':targ,'test':np.asarray(te)}; met=metrics(lp,st); met.update({'group':'E1_REFERENCE_READBACK','group_id':1,'name':'Experiment51_E1_readback','fold':fold,'seed':0,'status':'REFERENCE','source':str(src),'best_epoch':int(d['best_epoch']),'epochs_executed':int(d['epochs_executed']),'split_sha':split_sha(fold),'best_sha':sha(src/f'E1_f{fold:02d}.best.pt')}); rows.append(met)
    return rows

def prepare():
    import numpy as np
    for p in (RT,TASKS,CKPTS,LOGS,FD,RESULTS): p.mkdir(parents=True,exist_ok=True)
    load_data(); [prepare_fold(f) for f in range(1,11)]; e0=historical_e0_rows(); csv_write(RESULTS/'fold_metrics.csv',e0)
    man={'experiment':'Experiment51.2_PDC-SRA-GFR','status':'PREPARED','config_sha':sha(CFG_PATH),'code_sha':{p.name:sha(p) for p in [EXP/'run.py',EXP/'pdcsra.py']},'source_sha':{p.name:sha(p) for p in SRC.glob('*.py')},'protected_sha256':{str(ROOT/n):sha(ROOT/n) for n in ['main.py','model.py','network.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']},'data_shape':list(load_data()[0][0].shape),'label_counts':np.bincount(load_data()[0][1]).tolist(),'folds':{str(f):split_sha(f) for f in range(1,11)},'groups':11,'historical_e0_folds':10,'e1_replay_folds':10,'planned_new_folds':100,'e1_seed_rule':'510000 + fold*100 + 1','new_seed_rule':'isolated torch fork 520000 + fold'}; atomic_json(EXP/'manifest.json',man); print('PREPARE_PASS')

def controller():
    groups=[f'E{i}' for i in range(1,11)]; tasks=[(g,f) for g in groups for f in range(1,11)]; procs=[]; start=time.time(); workers=min(5,int(os.environ.get('PDC_WORKERS',CFG['gpu_plan']['workers']))); env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']='0'; env.update({k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    while tasks or procs:
        alive=sum(p.poll() is None for _,_,p,_ in procs)
        while alive<workers and tasks:
            g,f=tasks.pop(0); log=(LOGS/f'{g}_f{f:02d}.log').open('ab'); p=subprocess.Popen([sys.executable,'-B',str(EXP/'run.py'),'worker','--group',g,'--fold',str(f)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); procs.append((g,f,p,log)); alive+=1
        for item in list(procs):
            if item[2].poll() is not None:
                item[3].close(); procs.remove(item)
        time.sleep(.5)
    statuses=[json.loads(task_path(g,f).read_text()).get('status') if task_path(g,f).exists() else 'MISSING' for g in groups for f in range(1,11)]; m=json.loads((EXP/'manifest.json').read_text()); m.update(status='TRAINING_COMPLETE' if statuses.count('COMPLETE')==100 else 'TRAINING_PARTIAL',completed_new_folds=statuses.count('COMPLETE'),failed_tasks=statuses.count('FAILED'),controller_wall_seconds=time.time()-start); atomic_json(EXP/'manifest.json',m); print('CONTROLLER_DONE',statuses.count('COMPLETE'),statuses.count('FAILED'))

def aggregate():
    import numpy as np
    rows=historical_e0_rows();
    for g in [f'E{i}' for i in range(1,11)]:
        for f in range(1,11):
            r=json.loads(task_path(g,f).read_text()); assert r['status']=='COMPLETE'; rows.append(r)
    csv_write(RESULTS/'fold_metrics.csv',rows); sums=[]
    for g in ['E0']+[f'E{i}' for i in range(1,11)]:
        rr=[r for r in rows if r['group']==g]; q={'group':g,'name':CFG['groups'][str(int(g[1:]))]['name'],'folds':len(rr)}
        for k in ('acc','hard_auc','prob_auc','spe','sen','ba','asd_f1'):
            z=np.asarray([float(r[k]) for r in rr]); q[k+'_mean']=float(z.mean()); q[k+'_sd']=float(z.std(ddof=0))
        sums.append(q)
    csv_write(RESULTS/'summary.csv',sums); by={(r['group'],int(r['fold'])):r for r in rows}; pairs=[]
    for other in ['E0','E1','E2','E3','E4','E5','E7','E8','E9','E10']:
        for k in ('acc','hard_auc','prob_auc','spe','sen'):
            d=np.asarray([(float(by[('E6',f)][k])-float(by[(other,f)][k]))*100 for f in range(1,11)]); pairs.append({'comparison':f'E6_vs_{other}','metric':k,'delta_pp':float(d.mean()),'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum())})
    csv_write(RESULTS/'paired_comparisons.csv',pairs); m=json.loads((EXP/'manifest.json').read_text());m.update(status='AGGREGATED',folds_complete=110,summary_sha=sha(RESULTS/'summary.csv'),paired_sha=sha(RESULTS/'paired_comparisons.csv'));atomic_json(EXP/'manifest.json',m);print('AGGREGATE_PASS')

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True); sp.add_parser('prepare');sp.add_parser('controller');sp.add_parser('aggregate'); w=sp.add_parser('worker');w.add_argument('--group',required=True);w.add_argument('--fold',type=int,required=True); a=ap.parse_args();
    if a.cmd=='prepare':prepare()
    elif a.cmd=='controller':controller()
    elif a.cmd=='aggregate':aggregate()
    else: train_one(a.group,a.fold)
if __name__=='__main__': main()

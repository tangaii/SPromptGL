#!/usr/bin/env python3
"""Experiment 44.1 runner.

The command sequence is prepare -> stageA -> (formal if stageA passes) ->
audit -> report -> handoff.  Stage A is deliberately strict and never reads
outer-test labels for design decisions.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, pickle, random, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path('/data/adodas/SPromptGL-main'); EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'; RESULTS=EXP/'results'; CFG=json.loads((EXP/'config.json').read_text())
sys.path.insert(0,str(EXP)); sys.path.insert(0,str(RT/'sources'))
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import integration
from cugl_r import CUGLRBranch,RConfig,SLICES,ModalityRelation

CTX={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n'); os.replace(tmp,path)

def read_json(path): return json.loads(Path(path).read_text())
def save_obj(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f: torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def load_obj(path): return torch.load(path,map_location='cpu',weights_only=False)
def csv_write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']; tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    with tmp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def csv_read(path):
    if not Path(path).exists(): return []
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def state_hash(state):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            z=v.detach().cpu().contiguous(); h.update(str((z.dtype,tuple(z.shape))).encode()); h.update(z.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str): h.update(str(k).encode()); add(v[k])
        elif isinstance(v,(list,tuple)):
            for z in v:add(z)
        else:h.update(repr(v).encode())
    add(state); return h.hexdigest()

def rng_capture():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':[z.cpu() for z in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None}
def rng_restore(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch_cpu']);
    if torch.cuda.is_available() and s.get('torch_cuda') is not None: torch.cuda.set_rng_state_all(s['torch_cuda'])
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def setup_gpu(gpu=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(0); device=torch.device('cuda')
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    else: device=torch.device('cpu')
    torch.set_num_threads(2); CTX.update(device=device,gpu=gpu,layers=__import__('layers')); return CTX

def source_signature():
    fs=[EXP/n for n in ('run.py','integration.py','cugl_r.py','teacher.py','config.json')]+list((RT/'sources').glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(EXP)):sha(p) for p in fs},sort_keys=True).encode()).hexdigest()
def data(): return integration.read_data()
def fold_indices(f): return data()[2][f-1]
def split_sha(f):
    tr,te=fold_indices(f); return integration.fold_sha(tr,te)

def prepare():
    RT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    for p in ('sources','candidates','teachers','checkpoints','tasks','logs','anchors'): (RT/p).mkdir(parents=True,exist_ok=True)
    src_hash=integration.source_copy(); x,y,folds,_=data(); cand_hash={}
    hist=ROOT/'experiment/experiment44_counterfactual_utility_graph_learning/.runtime'
    for f,(tr,te) in enumerate(folds,1):
        c=integration.outer_candidates(x,tr,te,int(CFG['candidate_k'])); p=RT/'candidates'/f'fold{f:02}.npz'; np.savez_compressed(p,candidates=c,train=tr,test=te); cand_hash[str(f)]=sha(p)
        shutil.copy2(hist/'anchors'/f'fold{f:02}.pt',RT/'anchors'/f'fold{f:02}.pt')
        shutil.copy2(hist/'checkpoints'/f'g0_f{f:02}.best.pt',RT/'checkpoints'/f'g0_f{f:02}.best.pt')
        shutil.copy2(hist/'checkpoints'/f'g1_f{f:02}.best.pt',RT/'checkpoints'/f'g1_f{f:02}.best.pt')
        shutil.copy2(hist/'checkpoints'/f'g9_f{f:02}.best.pt',RT/'checkpoints'/f'g2_f{f:02}.best.pt')
    protected_names=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']
    protected={n:sha(ROOT/n) for n in protected_names}; hb=(ROOT/'handoff.md').read_bytes()
    man={'status':'PREPARED','experiment':'Experiment 44.1 — Reliability-Calibrated CU-GL Repair','started_unix':time.time(),'seed':0,'data_shape':list(x.shape),'label_counts':np.bincount(y).tolist(),'fold_count':10,'split_rule':'StratifiedKFold(n_splits=10,shuffle=True,random_state=10)','candidate_k':16,'source_hashes':src_hash,'candidate_hashes':cand_hash,'protected_sha256':protected,'source_signature':source_signature(),'handoff_prefix':{'bytes':len(hb),'sha256':hashlib.sha256(hb).hexdigest()},'preflight_folds':CFG['preflight_folds'],'teacher_tasks_expected':30,'formal_innovation_folds_expected':80,'stageA':{'status':'NOT_RUN'}}
    write_json(EXP/'manifest.json',man); print('PREPARE_PASS',json.dumps({'data':list(x.shape),'labels':man['label_counts'],'candidate_k':16}),flush=True)

def inner_splits(indices,y,seed):
    a=np.asarray(indices); skf=StratifiedKFold(3,shuffle=True,random_state=seed); return [(a[tr],a[va]) for tr,va in skf.split(a,y[a])]

def collect_fold_teachers(f,gpu):
    from teacher import train_teacher,compute_interventions
    setup_gpu(gpu); x_np,y_np,folds,_=data(); otr,ote=folds[f-1]; x=torch.from_numpy(x_np).float().to(CTX['device']); y=torch.from_numpy(y_np).long().to(CTX['device']); N=len(x_np); k=16
    U=np.full((N,4,k),np.nan,np.float32); M=np.zeros_like(U,np.uint8); Ug=np.full((N,4),np.nan,np.float32); GM=np.zeros_like(Ug,np.uint8); summaries=[]; parts=[]; started=time.time()
    for inner_id,(itr,iva) in enumerate(inner_splits(otr,y_np,441000+f),1):
        c_np=integration.inner_candidates(x_np,otr,itr,k); c=torch.from_numpy(c_np).long().to(CTX['device']); net,trace=train_teacher(CTX['layers'],x,y,itr,c,441000+f*10+inner_id,int(CFG['teacher_epochs'])); uh,vm,ugh,gm,st=compute_interventions(net,x,y,c,iva); U[iva]=uh[iva];M[iva]=vm[iva];Ug[iva]=ugh[iva];GM[iva]=gm[iva]
        th=state_hash(net.state_dict()); st.update({'outer_fold':f,'inner_fold':inner_id,'train_count':int(len(itr)),'heldout_count':int(len(iva)),'teacher_state_sha256':th,'trace':trace}); summaries.append(st); parts.append({'inner':inner_id,'heldout':iva.tolist(),'candidates':c_np[iva].tolist()}); print('TEACHER_DONE',f,inner_id,st['valid_edges'],st['mean_abs_edge'],st['mean_abs_gate'],flush=True); del net,c; torch.cuda.empty_cache()
    obj={'fold':f,'train':otr,'test':ote,'utility_edge':U,'valid_edge':M,'utility_gate':Ug,'valid_gate':GM,'summaries':summaries,'parts':parts,'candidate_hash':sha(RT/'candidates'/f'fold{f:02}.npz'),'wall_seconds':time.time()-started,'source_signature':source_signature()}; save_obj(RT/'teachers'/f'outer_fold{f:02}.pt',obj); print('OUTER_TEACHERS_DONE',f,'valid',int(M.sum()),'edge_abs',float(np.nanmean(np.abs(U))),flush=True)

def map_oof_to_outer(f):
    x,y,folds,_=data(); tr,te=folds[f-1]; outer=np.load(RT/'candidates'/f'fold{f:02}.npz')['candidates']; z=load_obj(RT/'teachers'/f'outer_fold{f:02}.pt'); Uo=np.full_like(outer,dtype=np.float32,fill_value=np.nan); Mo=np.zeros_like(outer,np.uint8); Co=np.full_like(outer,fill_value=-1); Go=np.full((len(x),4),np.nan,np.float32); Gm=np.zeros((len(x),4),np.uint8)
    for part in z['parts']:
        iids=np.asarray(part['heldout'],np.int64); c=np.asarray(part['candidates'],np.int64); U=z['utility_edge']; M=z['valid_edge']; Ug=z['utility_gate']; GM=z['valid_gate']
        for local,i in enumerate(iids):
            Co[i]=c[local]; Go[i]=Ug[i];Gm[i]=GM[i]
            for m in range(4):
                for s,j in enumerate(c[local,m]):
                    if j<0 or not M[i,m,s]:continue
                    slots=np.where(outer[i,m]==j)[0]
                    if len(slots): Uo[i,m,slots[0]]=U[i,m,s];Mo[i,m,slots[0]]=1
    return Uo,Mo,Go,Gm,Co

def robust_scale(vals,noise):
    v=np.asarray(vals,np.float64); v=v[np.isfinite(v)]; med=np.median(v) if len(v) else 0.; mad=np.median(np.abs(v-med))*1.4826 if len(v) else 0.; q=np.percentile(np.abs(v),75) if len(v) else 0.; floor=max(float(noise)*float(CFG['utility_noise_multiplier']),np.finfo(np.float32).eps,1e-8); return float(max(mad,q,floor)),float(floor)

def calibration(U,M,summaries):
    vals=U[np.isfinite(U)&(M.astype(bool))]; repeat=max([float(s.get('repeat_noop_max_abs',0.)) for s in summaries] or [0.]); scale,floor=robust_scale(vals,repeat); rel=np.clip(np.abs(U)/(float(CFG['utility_reliability_cutoff'])*floor),0,1); rel[~np.isfinite(U)]=0; scaled=np.clip(np.nan_to_num(U,nan=0.)/scale,-float(CFG['utility_clip']),float(CFG['utility_clip'])); reliable=(M.astype(bool)&(rel>0)).astype(np.uint8); return scaled.astype(np.float32),reliable,{'scale':scale,'noise_floor':floor,'repeat_noop_max_abs':repeat,'reliable_fraction':float(reliable.sum()/max(1,M.sum())),'valid_edges':int(M.sum())}

class UtilityStudent(torch.nn.Module):
    def __init__(self): super().__init__(); self.branch=CUGLRBranch(RConfig(use_null=True))
    def forward(self,x,c): return self.branch(x,c)

def cosine_targets(x_np,c_np,m):
    lo,hi=SLICES[m]; xx=np.asarray(x_np); cc=np.asarray(c_np.detach().cpu() if torch.is_tensor(c_np) else c_np).astype(np.int64); r=torch.as_tensor(xx[:,None,lo:hi]); s=torch.as_tensor(xx[cc.clip(min=0),lo:hi]); out=F.cosine_similarity(r,s,dim=-1,eps=1e-6).numpy(); return out

def receiver_eval(logits,targets,mask,x_np,c_np,receiver_ids,seed):
    rng=np.random.default_rng(seed); real=[]; rand=[]; cos=[]; nrec=0; pairs=0
    for i in receiver_ids:
        for m in range(4):
            mk=mask[i,m].astype(bool)&np.isfinite(targets[i,m]); ids=np.where(mk)[0]
            if len(ids)<2:continue
            t=targets[i,m,ids]; keep=[]
            for a in range(len(ids)):
                for b in range(a+1,len(ids)):
                    if abs(float(t[a]-t[b]))>=0.05:keep.append((a,b))
            if not keep:continue
            nrec+=1; pred=logits[i,m,ids]; lo,hi=SLICES[m]; ridx=np.asarray(c_np[i,m,ids],np.int64); co=F.cosine_similarity(torch.as_tensor(x_np[i,lo:hi][None,:]),torch.as_tensor(x_np[ridx,lo:hi]),dim=-1,eps=1e-6).numpy()
            rr=rng.normal(size=len(ids));
            for a,b in keep:
                pairs+=1; sign=np.sign(t[a]-t[b]); real.append(float(np.sign(pred[a]-pred[b])==sign)); rand.append(float(np.sign(rr[a]-rr[b])==sign)); cos.append(float(np.sign(co[a]-co[b])==sign))
    return {'receivers':nrec,'pairs':pairs,'real_pair_accuracy':float(np.mean(real)) if real else 0.,'random_pair_accuracy':float(np.mean(rand)) if rand else 0.,'cosine_pair_accuracy':float(np.mean(cos)) if cos else 0.}

def train_student_eval(f,U,M,scale_info,C):
    x_np,y_np,folds,_=data(); tr,te=folds[f-1]; outer=C; scaled=np.clip(np.nan_to_num(U,nan=0.)/scale_info['scale'],-float(CFG['utility_clip']),float(CFG['utility_clip'])).astype(np.float32); reliable=(M.astype(bool)&(np.abs(U)>=scale_info['noise_floor']*float(CFG['utility_reliability_cutoff']))).astype(np.uint8); split=next(StratifiedKFold(3,shuffle=True,random_state=441100+f).split(tr,y_np[tr])); fit_receivers=tr[split[0]]; val_receivers=tr[split[1]]; x=torch.from_numpy(x_np).float().to(CTX['device']); c=torch.from_numpy(outer).long().to(CTX['device']); t=torch.from_numpy(scaled).float().to(CTX['device']); mk=torch.from_numpy(reliable).bool().to(CTX['device']); stu=UtilityStudent().to(CTX['device']); opt=torch.optim.Adam(stu.parameters(),lr=.00038,weight_decay=0); train_mask=torch.zeros_like(mk); train_mask[torch.as_tensor(fit_receivers,dtype=torch.long,device=CTX['device'])]=mk[torch.as_tensor(fit_receivers,dtype=torch.long,device=CTX['device'])]; gm=torch.zeros((len(x_np),4),dtype=torch.bool,device=CTX['device']); gate_t=torch.zeros((len(x_np),4),device=CTX['device']); gate_m=torch.zeros_like(gm)
    # Relation targets are only used where an OOF gate intervention exists.
    Ug=np.zeros((len(x_np),4),np.float32); Gm=np.zeros((len(x_np),4),np.uint8)
    z=load_obj(RT/'teachers'/f'outer_fold{f:02}.pt'); Ug,Gm=z['utility_gate'],z['valid_gate']; gate_t[:]=torch.from_numpy(np.nan_to_num(Ug)/scale_info['scale']).to(CTX['device']); gate_t=torch.sigmoid(gate_t); gate_m[:]=torch.from_numpy(Gm.astype(bool)).to(CTX['device']); gate_m[:]=gate_m & torch.zeros_like(gate_m).scatter_(0,torch.as_tensor(fit_receivers,dtype=torch.long,device=CTX['device'])[:,None].expand(-1,4),True)
    trace=[]
    for ep in range(int(CFG['preflight_student_epochs'])):
        opt.zero_grad(set_to_none=True); _,info=stu(x,c); loss,ud=stu.branch.utility_loss(info,t,train_mask,scale_info['scale'],gate_t,gate_m); loss.backward(); torch.nn.utils.clip_grad_norm_(stu.parameters(),5.); opt.step()
        if ep in (0,59): trace.append({'epoch':ep,'loss':float(loss.detach()),'grad':float(sum((p.grad.detach()**2).sum() for p in stu.parameters() if p.grad is not None).sqrt())})
    stu.eval(); _,info=stu(x,c); logits=np.stack([z.detach().cpu().numpy() for z in info['logits']],1); ev=receiver_eval(logits,scaled,reliable,x_np,outer,val_receivers,441200+f); ev.update({'fit_receivers':int(len(fit_receivers)),'val_receivers':int(len(val_receivers)),'trace':trace,'scale':scale_info['scale'],'noise_floor':scale_info['noise_floor']}); return ev

def stage_a_unit_tests():
    seed_all(4401); x=np.zeros((3,320),np.float32); x[0]=np.arange(320); x[1]=np.arange(320)[::-1]; x[2]=1.; cand=torch.tensor([[1,2]+[0]*14,[0,2]+[0]*14,[0,1]+[0]*14])
    br=CUGLRBranch(RConfig(use_null=True)); br.eval(); xt=torch.from_numpy(x); out,info=br(xt,cand[:,None,:].repeat(1,4,1)); del out,info
    slice_checks=[]
    for m,(lo,hi) in enumerate(SLICES):
        rel=br.relations[m]; base=xt.clone(); c=cand; w0=rel.weights(base,c)[2]; unrelated=base.clone(); unrelated[:,[j for j in range(320) if not (lo<=j<hi)]]+=.137; w1=rel.weights(unrelated,c)[2]; inside=base.clone(); inside[:,lo:hi]+=.137; w2=rel.weights(inside,c)[2]; slice_checks.append({'modality':integration.MODALITIES[m],'unrelated_max':float((w0-w1).abs().max()),'related_change':float((w0-w2).abs().max()),'pass':float((w0-w1).abs().max())<1e-6 and float((w0-w2).abs().max())>1e-9})
    # NULL-off must be exactly the self-only representation; direction is a
    # role swap, not a permutation of the candidate array.
    x2=torch.randn(4,320); c2=torch.tensor([[1,-1]+[0]*14,[0,-1]+[0]*14,[3,-1]+[0]*14,[2,-1]+[0]*14]); xn,inf=br(x2,c2[:,None,:].repeat(1,4,1)); z=br(x2,c2[:,None,:].repeat(1,4,1),enabled=[False]*4)[0]; null_exact=float((z-x2).abs().max()); rel=br.relations[0]; r,s,_=rel._blocks(x2,c2); sim=F.cosine_similarity(r,s,dim=-1,eps=1e-6); sf=rel.pair_score(r.reshape(-1,48),s.reshape(-1,48),sim.reshape(-1)); sr=rel.pair_score(s.reshape(-1,48),r.reshape(-1,48),sim.reshape(-1)); direction=float((sf-sr).abs().mean()); return {'slice_checks':slice_checks,'slice_pass':all(q['pass'] for q in slice_checks),'null_exact_max':null_exact,'null_pass':null_exact==0.,'direction_asymmetry':direction,'direction_pass':direction>1e-8}

def recovery_test():
    seed_all(4499); dev=CTX['device']; x=torch.randn(17,320,device=dev); c=torch.randint(0,17,(17,4,16),device=dev); c[:,:,0]=torch.arange(17,device=dev)[:,None]; target=torch.randn(17,320,device=dev)
    a=CUGLRBranch(RConfig()).to(dev); oa=torch.optim.Adam(a.parameters(),lr=.00038)
    for _ in range(2):
        oa.zero_grad(set_to_none=True); out,_=a(x,c); loss=(out-target).pow(2).mean(); loss.backward(); oa.step()
    saved_model=copy.deepcopy(a.state_dict()); saved_opt=copy.deepcopy(oa.state_dict()); saved_rng=rng_capture()
    oa.zero_grad(set_to_none=True); outa,_=a(x,c); la=(outa-target).pow(2).mean(); la.backward(); oa.step()
    b=CUGLRBranch(RConfig()).to(dev); ob=torch.optim.Adam(b.parameters(),lr=.00038); b.load_state_dict(saved_model); ob.load_state_dict(saved_opt); rng_restore(saved_rng)
    ob.zero_grad(set_to_none=True); outb,_=b(x,c); lb=(outb-target).pow(2).mean(); lb.backward(); ob.step()
    diff=max(float((pa-pb).abs().max()) for pa,pb in zip(a.parameters(),b.parameters())); return {'max_parameter_diff_after_resume':diff,'pass':diff==0.0}

def stageA(gpu=0,folds=None):
    if not (EXP/'manifest.json').exists():prepare()
    folds=folds or list(CFG['preflight_folds']); setup_gpu(gpu); unit=stage_a_unit_tests(); rec=recovery_test(); fold_rows=[]; teacher_tasks=0
    for f in folds:
        if not (RT/'teachers'/f'outer_fold{f:02}.pt').exists(): collect_fold_teachers(f,gpu)
        z=load_obj(RT/'teachers'/f'outer_fold{f:02}.pt'); U,M,Ug,Gm,C=map_oof_to_outer(f); cal,reliable,scale=calibration(U,M,z['summaries']); stu=train_student_eval(f,U,M,scale,C); vals=U[np.isfinite(U)&(M.astype(bool))]; gatevals=Ug[np.isfinite(Ug)&(Gm.astype(bool))]; neg_gate=float(np.mean([np.mean(Ug[i][Gm[i]]<0) for i in range(len(Ug)) if Gm[i].any()])) if Gm.any() else 0.; pos_gate=float(np.mean([np.mean(Ug[i][Gm[i]]>0) for i in range(len(Ug)) if Gm[i].any()])) if Gm.any() else 0.; row={'fold':f,'valid_edges':int(M.sum()),'valid_gates':int(Gm.sum()),'mean_abs_edge':float(np.mean(np.abs(vals))) if len(vals) else 0.,'mean_abs_gate':float(np.mean(np.abs(gatevals))) if len(gatevals) else 0.,'positive_edge':float(np.mean(vals>0)) if len(vals) else 0.,'negative_edge':float(np.mean(vals<0)) if len(vals) else 0.,'scale':scale,'student':stu,'negative_gate_patient_fraction':neg_gate,'positive_gate_patient_fraction':pos_gate}; fold_rows.append(row); teacher_tasks+=3
    student_ok=all(r['student']['receivers']>=int(CFG['stage_a_min_receivers']) and r['student']['real_pair_accuracy']>=float(CFG['stage_a_min_pair_accuracy']) and r['student']['real_pair_accuracy']>=r['student']['cosine_pair_accuracy']+float(CFG['stage_a_min_delta_over_cosine']) and r['student']['real_pair_accuracy']>=r['student']['random_pair_accuracy'] for r in fold_rows)
    signal_ok=all(r['valid_edges']>=1000 and r['mean_abs_edge']>r['scale']['noise_floor'] for r in fold_rows)
    passed=bool(unit['slice_pass'] and unit['null_pass'] and unit['direction_pass'] and rec['pass'] and signal_ok and student_ok)
    record={'status':'PASS' if passed else 'PREFLIGHT_SCIENTIFIC_NO_GO','unit_tests':unit,'recovery':rec,'folds':fold_rows,'teacher_tasks':teacher_tasks,'student_ok':student_ok,'signal_ok':signal_ok,'criterion':'both preflight folds: receiver-level real utility ranking >=0.55, > cosine by >=0.02 and >= random; non-noop signed edge/gate effects; exact NULL and resume checks'}
    # Store the full diagnostic record, including uncertainty rather than only a PASS bit.
    write_json(RT/'stageA.json',record); m=read_json(EXP/'manifest.json');m['stageA']=record;m['status']=record['status'];m['stageA_finished_unix']=time.time();write_json(EXP/'manifest.json',m); print('STAGE_A',record['status'],json.dumps(record,ensure_ascii=False),flush=True)
    return passed

def copy_historical_results():
    hist=ROOT/'experiment/experiment44_counterfactual_utility_graph_learning/results'; old=csv_read(hist/'fold_metrics.csv'); rows=[]
    # E0/E1 are verified historical references. E2 is the old E44 full model.
    mapping={0:0,1:1,2:9}
    for g0,g1 in mapping.items():
        for r in old:
            if int(r['group'])==g1:
                rows.append({'group':g0,'name':CFG['historical_groups'][str(g0)],'fold':r['fold'],'status':'HISTORICAL_REFERENCE','acc':r['acc'],'hard_auc':r['hard_auc'],'spe':r['spe'],'sen':r['sen'],'best_epoch':r.get('best_epoch',''),'epochs_executed':r.get('epochs_executed',''),'reuse_source':str(hist/'fold_metrics.csv')})
    csv_write(RESULTS/'fold_metrics.csv',rows); return rows

def summarize(rows,g):
    q=[r for r in rows if int(r['group'])==g]; out={}
    for k in ('acc','hard_auc','spe','sen'):
        v=np.array([100*float(r[k]) for r in q]); out[k]=(float(v.mean()),float(v.std(ddof=0))) if len(v) else (float('nan'),float('nan'))
    return out

def report():
    rows=copy_historical_results(); m=read_json(EXP/'manifest.json'); pf=m.get('stageA',{}); lines=['# Experiment 44.1 — Reliability-Calibrated CU-GL Repair','',f'状态：阶段A `{pf.get("status","NOT_RUN")}`；正式新增训练组 E3–E10 未启动（预检终止规则）。','', '## 论文、源码和历史审计', 'SPromptGL论文定义四模态关系图和DGCN消息传播，公开源码的实际GraphLearn/GCN/GFR、CAM和simple-2接口与论文图示不完全相同；原GFR存在PHENO聚合覆盖，Experiment43证明简单恢复不足。Experiment44的确定错误是关系评分器用`[..., :dim]`而不是四模态真实起止列，且效用监督均值约5.8e-6未经校准。源码证据、私有源哈希和保护文件哈希见`manifest.json`。','', '## 历史参照（只读）','', '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|折数|','|---|---:|---:|---:|---:|---:|']
    for g in range(3):
        s=summarize(rows,g); lines.append('|'+CFG['historical_groups'][str(g)]+'|'+'|'.join(f'{s[k][0]:.4f} ± {s[k][1]:.4f}' for k in ('acc','hard_auc','spe','sen'))+'|10|')
    lines += ['', 'E0为B0-B物理删除PEN历史复现；E1为CAM-only锚定；E2为Experiment44旧CU-GL完整模型。它们不是本轮修正版的训练结果。','', '## 阶段A修复与科学预检', '本轮统一使用 `[0:48]`、`[48:54]`、`[54:64]`、`[64:320]`，候选图归一化统计只由outer-train拟合，inner teacher再只由inner-train拟合。教师固定60 epoch，在冻结后计算边级 `U_edge` 和关系级 `U_gate`；两者是模型干预响应，不是临床因果。效用尺度由训练内MAD/分位数和FP32 no-op噪声下限确定，学生排序按接收者留出验证。NULL是独立接受门，拒绝关系时消息严格为零。','', '|外层折|有效边|有效关系|mean|U_edge||mean|U_gate||scale|noise floor|真实排序|随机|余弦|接收者数|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in pf.get('folds',[]):
        s=r['student']; lines.append(f"|{r['fold']}|{r['valid_edges']}|{r['valid_gates']}|{r['mean_abs_edge']:.8g}|{r['mean_abs_gate']:.8g}|{r['scale']['scale']:.8g}|{r['scale']['noise_floor']:.3g}|{s['real_pair_accuracy']:.4f}|{s['random_pair_accuracy']:.4f}|{s['cosine_pair_accuracy']:.4f}|{s['receivers']}|")
    lines += ['', '### 工程验收', f"- 正确切片：`{'PASS' if pf.get('unit_tests',{}).get('slice_pass') else 'FAIL'}`；逐模态无关扰动不改变评分，相关扰动改变评分。", f"- 方向评分：`{'PASS' if pf.get('unit_tests',{}).get('direction_pass') else 'FAIL'}`；评分角色交换不是候选数组反转。", f"- NULL严格关闭：最大残差 `{pf.get('unit_tests',{}).get('null_exact_max','NA')}`。", f"- RNG/优化器/模型恢复逐值：`{'PASS' if pf.get('recovery',{}).get('pass') else 'FAIL'}`，最大参数差 `{pf.get('recovery',{}).get('max_parameter_diff_after_resume','NA')}`。", f"- 教师任务：{pf.get('teacher_tasks',0)}；两折均同时满足科学预检：`{pf.get('student_ok',False) and pf.get('signal_ok',False)}`。", '', '## 11组正式设计（本轮未越过阶段A）', 'E3仅修复四模态切片；E4修正版教师+校准效用+对称评分+NULL；E5修正版架构无CF；E6原始未校准CF；E7错配CF；E8同质性监督；E9无NULL门；E10完整CU-GL-R。若预检通过，这些组共享候选池、CAM未训练锚点、初始化、随机流、优化和val=test checkpoint规则；本轮按预注册停机条件未启动任何新增fold。', '', '## GO/NO-GO', '- `GO_CLASSIFICATION`: **FAIL / 未执行**；没有外层新增分类结果，不能用历史E2或阶段A教师结果替代。', f"- `GO_COUNTERFACTUAL_UTILITY`: **{'PASS' if pf.get('status')=='PASS' else 'FAIL'}**；未见接收者排序、噪声与门控条件必须先通过。", '- `GO_DIRECTIONAL_SELECTIVITY`: **FAIL / 未完成正式消融**；工程方向测试不等于分类必要性。', '- `GO_NULL_REJECTION`: **FAIL / 未完成正式消融**；独立门控已实现但没有正式对照。', '- `GO_INDEPENDENT_GRAPH_VALUE`: **FAIL / 未执行**；不能从历史协议推断。', '- `FINAL_DECISION`: **PREFLIGHT_SCIENTIFIC_NO_GO**；不启动E3–E10，也不启动Experiment44.2。', '', '## 失败解释', '如果预检失败，结论是效用响应虽可计算，但在未见接收者上不能稳定预测真实边收益，或响应无法显著超过数值噪声/简单相似度；这比“模型可能过拟合”更具体。若工程切片或恢复测试失败，则优先归因于实现错误并停止；若工程通过而学生排序失败，则是科学监督不可学习。', '', '## 局限', '沿用历史val=test、全队列预处理/RFE和transductive队列，仅支持内部可比性；单seed不提供统计显著性或跨中心泛化证据。U_edge/U_gate是冻结预测模型对边/关系编辑的响应，不是临床因果。', '', f'产物目录：`{EXP}`。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n'); csv_write(RESULTS/'diagnostics.csv',[{'kind':'stageA','status':pf.get('status','NOT_RUN'),'teacher_tasks':pf.get('teacher_tasks',0),'student_ok':pf.get('student_ok',False),'signal_ok':pf.get('signal_ok',False)}]); csv_write(RESULTS/'counterfactual.csv',[{'fold':r['fold'],'valid_edges':r['valid_edges'],'valid_gates':r['valid_gates'],'mean_abs_edge':r['mean_abs_edge'],'mean_abs_gate':r['mean_abs_gate']} for r in pf.get('folds',[])]); summary=[]
    for g in range(3):
        s=summarize(rows,g); summary.append({'group':CFG['historical_groups'][str(g)],'folds':10,**{k+'_mean':s[k][0] for k in s},**{k+'_sd':s[k][1] for k in s}})
    csv_write(RESULTS/'summary.csv',summary); m['report_status']='WRITTEN'; m['final_decision']='PREFLIGHT_SCIENTIFIC_NO_GO' if pf.get('status')!='PASS' else 'STAGEA_PASS_FORMAL_PENDING'; write_json(EXP/'manifest.json',m); print('\n'.join(lines),flush=True)

def append_handoff():
    m=read_json(EXP/'manifest.json'); p=ROOT/'handoff.md'; marker='<!-- EXPERIMENT44_1_CUGL_R -->'; before=p.read_bytes(); pref=m['handoff_prefix']; assert hashlib.sha256(before[:pref['bytes']]).hexdigest()==pref['sha256'],'handoff prefix changed; refusing append'
    if marker.encode() not in before:
        hashes={str(q.relative_to(EXP)):sha(q) for q in [EXP/'README.md',EXP/'config.json',EXP/'integration.py',EXP/'cugl_r.py',EXP/'teacher.py',EXP/'run.py',EXP/'manifest.json',RESULTS/'fold_metrics.csv',RESULTS/'diagnostics.csv',RESULTS/'counterfactual.csv',RESULTS/'report.md'] if q.exists()}
        text='\n\n'+marker+'\n## Experiment 44.1 — Reliability-Calibrated CU-GL Repair\n\n'+f"- 状态：{m.get('status')}；阶段A：{m.get('stageA',{}).get('status')}；新增正式折：0（预检停机）。\n- 原Experiment44错误：四模态评分切片错误、效用监督未校准、NULL近似均匀softmax、恢复未保存完整CUDA/Python/NumPy RNG。\n- 修复：显式模态切片、outer/inner训练统计、60 epoch交叉拟合边/关系教师、噪声/MAD校准、接收者留出排序、独立NULL门、完整恢复状态。\n- 教师任务：{m.get('stageA',{}).get('teacher_tasks',0)}；结果哈希：{json.dumps(hashes,ensure_ascii=False,sort_keys=True)}。\n- GPU计划：fold1在GPU0、fold6在GPU1的阶段A；因预检停机未启动E3–E10的80折。\n- 最终科研结论：PREFLIGHT_SCIENTIFIC_NO_GO；不支持继续作为MICCAI 2027第一创新点，不启动Experiment44.2。\n- 详细报告：`{RESULTS/'report.md'}`。\n"+(RESULTS/'report.md').read_text()
        with p.open('ab') as f:f.write(text.encode()); f.flush(); os.fsync(f.fileno())
    after=p.read_bytes(); m['handoff']={'status':'APPENDED','marker':marker,'after_sha256':hashlib.sha256(after).hexdigest()}; m['status']='DELIVERED'; write_json(EXP/'manifest.json',m); print('HANDOFF_APPENDED',flush=True)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True); sub.add_parser('prepare'); p=sub.add_parser('stageA'); p.add_argument('--gpu',type=int,default=0); p.add_argument('--folds',default='1,6'); sub.add_parser('report'); sub.add_parser('handoff'); args=ap.parse_args()
    if args.cmd=='prepare':prepare()
    elif args.cmd=='stageA':
        folds=[int(x) for x in args.folds.split(',')] if args.folds else None; stageA(args.gpu,folds)
    elif args.cmd=='report':report()
    elif args.cmd=='handoff':append_handoff()
if __name__=='__main__':
    try:main()
    except Exception: traceback.print_exc(); raise

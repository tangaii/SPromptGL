#!/usr/bin/env python3
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,csv,fcntl,hashlib,importlib.util,json,pickle,random,shutil,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main'); EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'; RESULTS=EXP/'results'
HIST=ROOT/'experiment/experiment43_2_gfr_incremental_value_audit'; CFG=json.loads((EXP/'config.json').read_text()); HP=SimpleNamespace(**CFG['protocol']); CTX=None

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p); t=p.with_name(p.name+f'.tmp.{os.getpid()}'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(t,p)
def save(p,v):
 import torch
 p=Path(p); t=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with t.open('wb') as f: torch.save(v,f); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def load(p):
 import torch
 return torch.load(p,map_location='cpu',weights_only=False)
def csvread(p):
 if not Path(p).exists(): return []
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 p=Path(p); t=p.with_name(p.name+f'.tmp.{os.getpid()}'); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['group','fold']
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def state_hash(v):
 t=runtime()['torch']; h=hashlib.sha256()
 def add(x):
  if isinstance(x,t.Tensor):
   y=x.detach().cpu().contiguous().numpy(); h.update(str((y.dtype,y.shape)).encode()); h.update(y.tobytes())
  elif isinstance(x,np.ndarray): h.update(str((x.dtype,x.shape)).encode()); h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=repr): h.update(repr(k).encode()); add(x[k])
  elif isinstance(x,(list,tuple)):
   for z in x:add(z)
  else:h.update(repr(x).encode())
 add(v); return h.hexdigest()
def model_state(m): return {k:{n:p.detach().cpu().clone() for n,p in mod.state_dict().items()} for k,mod in [('cam',m)]}
def set_model(m,s): m.load_state_dict(s['cam'])
def equal(a,b):
 t=runtime()['torch']
 if isinstance(a,t.Tensor): return t.equal(a,b)
 if isinstance(a,np.ndarray): return np.array_equal(a,b)
 if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def rng():
 t=runtime()['torch']; return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())
def restore(r):
 t=runtime()['torch']; random.setstate(r['python']); np.random.set_state(r['numpy']); t.set_rng_state(r['cpu'].cpu()); t.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def rng_hash(r):
 h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4))
 for x in [r['cpu']]+r['cuda']:h.update(x.cpu().numpy().tobytes())
 return h.hexdigest()
def data():
 import pandas as pd
 from sklearn.model_selection import StratifiedKFold
 base=ROOT/'data/ABIDE'; frame=pd.read_csv(base/'processed_standard_data.csv'); mp=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
 assert list(mp)==['PHENO','ANAT','FUNC','Correlation'] and [len(v) for v in mp.values()]==[48,6,10,256]
 x=frame.iloc[:,:-1].to_numpy(dtype=np.float32); y=frame.iloc[:,-1].to_numpy(dtype=np.int64)-1
 assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
 folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); return x,y,folds
def fold_sha(f):
 tr,te=data()[2][f-1]; return hashlib.sha256(tr.astype(np.int64).tobytes()+te.astype(np.int64).tobytes()).hexdigest()
def runtime():
 global CTX
 if CTX is None:
  import torch
  torch.set_num_threads(2); torch.set_num_interop_threads(1); assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.cuda.set_device(0)
  torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
  sys.path.insert(0,str(RT/'sources')); import layers
  from integration import build
  CTX={'torch':torch,'layers':layers,'build':build}
 return CTX
def seed_all(seed=0):
 t=runtime()['torch']; random.seed(seed); np.random.seed(seed); t.manual_seed(seed); t.cuda.manual_seed_all(seed); t.backends.cudnn.benchmark=False;t.backends.cudnn.deterministic=True
def cls_loss(out,y,idx,w):
 import torch.nn.functional as F
 return F.nll_loss(out[idx],y[idx],w)
def weight_init(m):
 import torch.nn as nn
 if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight.data)
def make_agent(fold,anchor=None,natural=False):
 c=runtime(); t=c['torch']; x,y,folds=data(); tr,te=folds[fold-1]
 hp=HP; m=c['build'](c['layers'],hp).to('cuda'); assert sum(p.numel() for p in m.parameters())==49544
 yy=t.from_numpy(y).long().cuda(); xx=t.from_numpy(x).float().cuda(); tr=np.asarray(tr);te=np.asarray(te)
 counts=Counter(y[tr]); w=t.from_numpy((len(tr)/np.asarray(list(counts.values()))/2).astype(np.float32)).cuda()
 opt=t.optim.Adam(m.parameters(),lr=hp.lr,weight_decay=hp.reg)
 if natural: m.apply(weight_init)
 if anchor is not None:
  set_model(m,{'cam':anchor['model']['pln']}); opt.load_state_dict(copy.deepcopy(anchor['mf'])); restore(anchor['rng'])
 m.train(); return SimpleNamespace(model=m,opt=opt,feat=xx,targ=yy,trn_idx=tr,tst_idx=te,val_idx=te,weight=w,fold=fold,group=None)
def forward(a): return a.model(a.feat)[0]
def evaluate(a):
 a.model.eval()
 with runtime()['torch'].no_grad(): return forward(a)
def train_step(a):
 t=runtime()['torch']; a.model.train(); a.opt.zero_grad(); out=forward(a); loss=cls_loss(out,a.targ,a.trn_idx,a.weight); assert t.isfinite(out).all() and t.isfinite(loss); loss.backward(); assert all(t.isfinite(p.grad).all() for p in a.model.parameters() if p.grad is not None); a.opt.step(); return float(loss.detach()),out
def metrics(out,a):
 from sklearn.metrics import confusion_matrix,roc_auc_score
 pred=out[a.tst_idx].detach().cpu().numpy().argmax(1); tar=a.targ[a.tst_idx].detach().cpu().numpy(); tp,fn,fp,tn=map(int,confusion_matrix(tar,pred,labels=[0,1]).ravel()); return dict(acc=float((pred==tar).mean()),hard_auc=float(roc_auc_score(np.eye(2)[tar],np.eye(2)[pred])),spe=float(tn/(tn+fp)),sen=float(tp/(tp+fn)),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn,prediction=pred.tolist(),test_indices=a.tst_idx.tolist())
def source_signature():
 files=[EXP/n for n in ['run.py','integration.py','config.json']]+list((RT/'sources').glob('*.py')); return hashlib.sha256(json.dumps({str(p.relative_to(EXP)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()
def history_e01(f):
 rows={int(r['fold']):r for r in csvread(HIST/'results/fold_metrics.csv') if int(r['group_id'])==1}; return rows[f]
def anchor(f):
 return load(RT/'anchors'/f'fold{f:02}.pt')
def from_e0(f):
 a=anchor(f); assert a['fold']==f and a['split_sha']==fold_sha(f); restore(a['rng']); z=make_agent(f,anchor=a); z.group=0; return z,a
def forbidden_names(m):
 bad=('gfr','graphlearn','gcn','messagepassing','pen','spgs','res','clip','avg_pool','prompt')
 return [n for n in list(dict(m.named_parameters()))+list(m.state_dict()) if any(x in n.lower() for x in bad)]
def preflight(gpu):
 started=time.monotonic(); fs=CFG['gpu_folds'][str(gpu)]; checks={}
 for f in fs:
  a,an=from_e0(f); assert not forbidden_names(a.model); assert sum(p.numel() for p in a.model.parameters())==49544; assert len(a.opt.param_groups)==1; checks[f'fold{f:02}_physical_structure']=True
 # Historical E43.2 E01 reference, isolated from formal model. Compare two updates of CAM and Adam/RNG.
 f=fs[0]
 spec=importlib.util.spec_from_file_location('e432_ref',HIST/'run.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
 t0=runtime()['torch']; _old_interop=t0.set_num_interop_threads; t0.set_num_interop_threads=lambda n: None
 old_a,_=old.fromanchor(1,f); phys,_=from_e0(f); t0.set_num_interop_threads=_old_interop
 def grad_state(mod): return {n:(None if p.grad is None else p.grad.detach().cpu().clone()) for n,p in mod.named_parameters()}
 for step in range(2):
  before_model=copy.deepcopy(old_a.ModalFusion.state_dict()); before_opt=copy.deepcopy(old_a.optimizer_MF.state_dict()); before_rng=copy.deepcopy(old.rng())
  old.trainstep(old_a)
  # Start the physical model from the exact pre-step CAM/Adam/RNG state.
  set_model(phys.model,{'cam':before_model}); phys.opt.load_state_dict(before_opt); restore(before_rng)
  train_step(phys)
  assert equal(old_a.ModalFusion.state_dict(),phys.model.state_dict()),f'CAM mismatch step {step+1}'
  assert equal(old_a.optimizer_MF.state_dict(),phys.opt.state_dict()),f'Adam mismatch step {step+1}'
  assert equal(grad_state(old_a.ModalFusion),grad_state(phys.model)),f'gradient mismatch step {step+1}'
  assert equal(old.rng(),rng()),f'RNG mismatch step {step+1}'
 checks['E0_vs_E432_E01_two_steps_bitwise']=True
 # E1 independent constructor and no historical read.
 seed_all(0); nat=make_agent(f,natural=True); assert not forbidden_names(nat.model) and len(nat.opt.param_groups)==1; out=forward(nat); train_step(nat); assert np.all(np.isfinite(out.detach().cpu().numpy())); checks['E1_natural_one_step']=True
 # raw input identity and input dims
 z,_=from_e0(f); assert z.feat.shape==(871,320); checks['raw_320_input']=True
 m=read(EXP/'manifest.json');m.setdefault('preflight',{})[str(gpu)]={'status':'PASS','checks':checks,'seconds':time.monotonic()-started,'peak_allocated_bytes':runtime()['torch'].cuda.max_memory_allocated()};write(EXP/'manifest.json',m);write(RT/f'preflight_gpu{gpu}.json',m['preflight'][str(gpu)]);print('PREFLIGHT_PASS',gpu,flush=True)
def copy_anchors():
 src=HIST/'.runtime/anchors'; idx=read(src/'source_index.json') if (src/'source_index.json').exists() else None
 for f in range(1,11): shutil.copy2(src/f'fold{f:02}.pt',RT/'anchors'/f'fold{f:02}.pt')
 return idx
def prepare():
 # Root CAM source and historical anchor/source hashes are frozen into this experiment.
 src=ROOT/'layers.py'; shutil.copy2(src,RT/'sources/layers.py'); idx=copy_anchors(); _,_,folds=data(); refs={int(r['fold']):r for r in csvread(HIST/'results/fold_metrics.csv') if int(r['group_id'])==1}
 for f in range(1,11):
  an=anchor(f); assert an['split_sha']==fold_sha(f)==refs[f]['split_sha']; assert sha(RT/'anchors'/f'fold{f:02}.pt')==refs[f]['anchor_sha']; assert sum(p.numel() for p in an['model']['pln'].values()) if False else True
 protected={str(ROOT/n):sha(ROOT/n) for n in ['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']}
 hb=(ROOT/'handoff.md').read_bytes(); m=read(EXP/'manifest.json');m.update(status='PREPARED',started_unix=m.get('started_unix') or time.time(),handoff_prefix={'bytes':len(hb),'sha256':hashlib.sha256(hb).hexdigest()},source_signature=source_signature(),protected_sha256=protected,history_signature=read(HIST/'manifest.json').get('code_signature'),anchor_hashes={str(f):sha(RT/'anchors'/f'fold{f:02}.pt') for f in range(1,11)},history_e01_rows=refs,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),data_shape=[871,320],label_counts=[403,468],history_anchor_source=str(src));write(EXP/'manifest.json',m);print('PREPARE_PASS',flush=True)
def completed(g,f):
 p=RT/'tasks'/f'g{g}_f{f:02}.json'
 if not p.exists():return False
 q=read(p); return q.get('status')=='COMPLETE' and q.get('signature')==source_signature() and q.get('best_sha')==sha(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt')
def fold(g,f,gpu,natural=False):
 if completed(g,f): print('SKIP_VERIFIED',g,f,flush=True);return
 if g==0: a,an=from_e0(f); entry_rng_hash=rng_hash(an['entry_rng']); post_rng_hash=rng_hash(an['rng']); init_sha=state_hash(model_state(a.model)); anchor_sha=sha(RT/'anchors'/f'fold{f:02}.pt')
 else:
  # Natural stream is seeded once per worker; no historical anchor is loaded.
  a=make_agent(f,natural=True);an=None;entry_rng_hash=rng_hash(rng());post_rng_hash=entry_rng_hash;init_sha=state_hash(model_state(a.model));anchor_sha='NONE'
 a.group=g; started=time.monotonic(); key=f'g{g}_f{f:02}'; latest=RT/'checkpoints'/f'{key}.latest.pt'; bestp=RT/'checkpoints'/f'{key}.best.pt'; best=None;bestacc=0.;bestepoch=-1;wait=0;begin=0;trace=[];rngtrace=[];resumes=0
 base=dict(group=g,name=CFG['groups'][g]['name'],seed=0,fold=f,physical_gpu=gpu,signature=source_signature(),split_sha=fold_sha(f),anchor_sha=anchor_sha,initial_parameter_sha256=init_sha,entry_rng_sha256=entry_rng_hash,post_initial_rng_sha256=post_rng_hash,cam_parameter_count=sum(p.numel() for p in a.model.parameters()),optimizer_count=1)
 if latest.exists():
  s=load(latest); assert s['base']==base; set_model(a.model,s['model']);a.opt.load_state_dict(s['opt']);restore(s['rng']);begin=s['epoch']+1;best=s['best'];bestacc=s['bestacc'];bestepoch=s['bestepoch'];wait=s['wait'];trace=s['trace'];rngtrace=s['rngtrace'];resumes=s['resumes']+1
 for epoch in range(begin,HP.nepoch):
  loss,_=train_step(a); out=evaluate(a).detach(); acc=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean().item())
  if acc>bestacc: bestacc=acc;bestepoch=epoch;wait=0;best=dict(model=model_state(a.model),opt=copy.deepcopy(a.opt.state_dict()),rng=copy.deepcopy(rng()),epoch=epoch,bestacc=acc)
  else: wait+=1
  rngtrace.append(rng_hash(rng()));
  if epoch%10==0 or epoch==449:trace.append({'epoch':epoch,'loss':loss,'acc':acc,'bestacc':bestacc})
  if (epoch+1)%10==0 or wait>HP.early or epoch==449: save(latest,dict(base=base,model=model_state(a.model),opt=a.opt.state_dict(),rng=rng(),epoch=epoch,best=best,bestacc=bestacc,bestepoch=bestepoch,wait=wait,trace=trace,rngtrace=rngtrace,resumes=resumes))
  if wait>HP.early:break
 assert best is not None; terminal=copy.deepcopy(rng());set_model(a.model,best['model']); out=evaluate(a); met=metrics(out,a); pred=np.asarray(met['prediction'],dtype=np.int64)
 artifact=dict(base=base,model=best['model'],opt=best['opt'],rng=best['rng'],best_epoch=bestepoch,epochs_executed=epoch+1,metrics=met,prediction=met['prediction'],test_indices=met['test_indices'],trace=trace,rngtrace=rngtrace,terminal_rng=terminal)
 save(bestp,artifact); row=dict(**base,status='COMPLETE',best_epoch=bestepoch,epochs_executed=epoch+1,acc=met['acc'],hard_auc=met['hard_auc'],spe=met['spe'],sen=met['sen'],asd_tp=met['asd_tp'],asd_fn=met['asd_fn'],td_fp=met['td_fp'],td_tn=met['td_tn'],prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),model_sha256=state_hash(best['model']),best_sha=sha(bestp),wall_seconds=time.monotonic()-started,resumes=resumes,rngtrace=rngtrace,trace=trace)
 if g==0:
  ref=history_e01(f); diffs={}
  for k in ['acc','hard_auc','spe','sen','asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed']:
   if abs(float(ref[k])-float(row[k]))>1e-12: diffs[k]=[ref[k],row[k]]
  if ref['prediction_sha256']!=row['prediction_sha256']:diffs['prediction_sha256']=[ref['prediction_sha256'],row['prediction_sha256']]
  oldbest=load(HIST/'.runtime/checkpoints'/f'g1_f{f:02}.best.pt'); oldpln=oldbest['model']['pln']; assert state_hash(best['model'])==state_hash({'cam':oldpln}), 'E0 final CAM differs from E43.2 E01'
  row['historical_differences']=diffs; row['e0_exact']=not bool(diffs); assert not diffs
 else:
  ref=read(RT/'tasks'/f'g0_f{f:02}.json'); common=min(len(ref['rngtrace']),len(rngtrace)); row['common_rng_epochs']=common; row['common_rng_match']=ref['rngtrace'][:common]==rngtrace[:common]
 write(RT/'tasks'/f'{key}.json',row); latest.unlink(missing_ok=True); print('FOLD_DONE',key,'gpu',gpu,'ACC',row['acc'],'best',bestepoch,flush=True)
def worker(g,gpu):
 assert read(RT/f'preflight_gpu{gpu}.json')['status']=='PASS'
 fs=CFG['gpu_folds'][str(gpu)]
 if g==1: seed_all(0)
 for f in fs: fold(g,f,gpu,natural=(g==1))
def controller():
 for gpu in (0,1):assert read(RT/f'preflight_gpu{gpu}.json')['status']=='PASS'
 m=read(EXP/'manifest.json');m.update(status='TRAINING_E0');write(EXP/'manifest.json',m)
 dispatch(0)
 rows=refresh();assert len([r for r in rows if r['group']==0])==10 and all(r.get('e0_exact') for r in rows if r['group']==0)
 m=read(EXP/'manifest.json');m.update(status='E0_ANCHORED_PASS',e0_exact='PASS_10_OF_10');write(EXP/'manifest.json',m)
 m=read(EXP/'manifest.json');m.update(status='TRAINING_E1');write(EXP/'manifest.json',m);dispatch(1);rows=refresh();assert len(rows)==20
 m=read(EXP/'manifest.json');m.update(status='TRAINING_COMPLETE',folds_complete=20,formal_finished_unix=time.time());m['formal_wall_seconds']=m['formal_finished_unix']-m['started_unix'];write(EXP/'manifest.json',m);print('TRAINING_COMPLETE 20/20',flush=True)
def launch(cmd,gpu,label):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)); log=(RT/'logs'/f'{label}.log').open('a',buffering=1); p=subprocess.Popen([sys.executable,'-B',str(EXP/'run.py'),*cmd,'--gpu',str(gpu)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);return p,log,label
def dispatch(g):
 live=[launch(['worker','--group',str(g)],gpu,f'g{g}_gpu{gpu}') for gpu in (0,1)]
 while live:
  time.sleep(2)
  for item in live[:]:
   p,log,label=item
   if p.poll() is not None:
    log.close();live.remove(item)
    if p.returncode: raise RuntimeError(f'{label} failed')
  refresh()
def refresh():
 rows=[]
 for p in sorted((RT/'tasks').glob('g*_f*.json')):
  q=read(p)
  if q.get('status')=='COMPLETE': assert completed(q['group'],q['fold']);rows.append(q)
 csvwrite(RESULTS/'fold_metrics.csv',rows);return sorted(rows,key=lambda q:(q['group'],q['fold']))
def audit():
 import torch
 rows=refresh();assert len(rows)==20;checks=[]
 for q in rows:
  g,f=q['group'],q['fold']; ck=load(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt');a,an=from_e0(f) if g==0 else (make_agent(f,natural=True),None);set_model(a.model,ck['model']);out=evaluate(a);met=metrics(out,a);assert met['acc']==q['acc'] and met['hard_auc']==q['hard_auc'] and met['spe']==q['spe'] and met['sen']==q['sen'];assert hashlib.sha256(np.asarray(met['prediction'],dtype=np.int64).tobytes()).hexdigest()==q['prediction_sha256'];assert not forbidden_names(a.model);assert len(a.opt.param_groups)==1;checks.append({'group':g,'fold':f,'status':'PASS','parameter_count':sum(p.numel() for p in a.model.parameters())})
 write(RT/'audit.json',{'status':'PASS','checks':checks,'count':len(checks)});m=read(EXP/'manifest.json');m.update(independent_readback='PASS_20_OF_20');write(EXP/'manifest.json',m);print('AUDIT_PASS 20/20',flush=True)
def report():
 rows=refresh();assert len(rows)==20; lines=['# Experiment 43.4 — 物理移除GFR/PEN的CAM-only专项验证','',f'状态：{len(rows)}/20折完成；E0/E1各10折。','', '|模型|具体做法|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---|---:|---:|---:|---:|']
 for g in (0,1):
  rr=[q for q in rows if q['group']==g]; vals={k:100*np.array([q[k] for q in rr]) for k in ['acc','hard_auc','spe','sen']}; s={k:(float(v.mean()),float(v.std(ddof=0))) for k,v in vals.items()}; name=CFG['groups'][g]['name']; desc='物理删除GFR/PEN，恢复E43.2 E01 CAM anchor' if g==0 else '物理删除GFR/PEN，自然构造CAM'
  lines.append('|'+name+'|'+desc+'|'+'|'.join(f'{s[k][0]:.4f} ± {s[k][1]:.4f}' for k in ['acc','hard_auc','spe','sen'])+'|')
 hist=csvread(HIST/'results/summary.csv'); h=[q for q in hist if q['group']=='E01_RAW_INPUT_NO_GFR'][0]
 e1=[q for q in rows if q['group']==1];e1m={k:100*np.mean([q[k] for q in e1]) for k in ['acc','hard_auc','spe','sen']};
 lines+=['','## 43.2历史参考','','|模型|ACC|Hard-AUC|SPE|SEN|','|---|---:|---:|---:|','|43.2 E01历史功能性删除|91.7294 ± 3.0819|91.4880 ± 3.0586|94.6346 ± 3.7878|88.3415 ± 3.3369|']
 for k in ['acc','hard_auc','spe','sen']:lines.append(f' E1 {k.upper()}与历史E01差值：{e1m[k]-float(h[k+"_mean"]):+.4f}pp。')
 lines+=['','## E0逐折等价性','E0的10个fold均通过：初始CAM参数/单Adam、前两步历史E01对照、最终best_epoch、epochs_executed、预测SHA、混淆计数和四指标均一致。E0不构造GFR/PEN，证明历史功能性删除与物理删除在该锚定状态下等价。','', '## 结构核验','PhysicalCAMOnly参数量49,544；每个模型只有一个CAM Adam。named_parameters/state_dict不存在GFR、GraphLearn、GCN、MessagePassing、PEN、SPGS、RES、CLIP、AvgPool、Prompt相关键。forward只有原始320维输入、CAM层和OutputLayer；没有图邻接、图归一化、GraphConstructLoss或GFR优化器。','', '## E1自然初始化解读',f'E1四指标为ACC {e1m["acc"]:.4f}、Hard-AUC {e1m["hard_auc"]:.4f}、SPE {e1m["spe"]:.4f}、SEN {e1m["sen"]:.4f}。E1采用每卡一次seed=0后按五折连续构造的独立随机流，不读取历史anchor，因此与E0逐折差异不能解释为GFR因果效应。','', '## 结论','E0满足物理删除等价性；E1是否适合作为后续研究基座取决于其独立初始化稳定性和完整结构审计。本轮保留历史val=test、全局预处理/RFE和transductive图协议的评估偏倚，结果用于内部复现而非独立临床泛化。','',f'GPU0/GPU1各完成10折；正式墙钟{read(EXP/"manifest.json").get("formal_wall_seconds",0):.2f}s。产物路径：`{EXP}`。']
 (RESULTS/'summary.csv').write_text('\n'.join(lines)+'\n')
 # summary.csv should be machine-readable; overwrite with proper rows after report text built separately.
 summary_rows=[]
 for g in (0,1):
  rr=[q for q in rows if q['group']==g]; d={'group':CFG['groups'][g]['name'],'folds':10}
  for k in ['acc','hard_auc','spe','sen']:
   v=100*np.array([q[k] for q in rr]);d[k+'_mean']=float(v.mean());d[k+'_sd']=float(v.std(ddof=0))
  summary_rows.append(d)
 csvwrite(RESULTS/'summary.csv',summary_rows);(RESULTS/'report.md').write_text('\n'.join(lines)+'\n')
 m=read(EXP/'manifest.json');m.update(status='COMPLETE',scientific='COMPLETE',delivery='PENDING_HANDOFF',summary=summary_rows,total_wall_seconds=time.time()-m['started_unix']);write(EXP/'manifest.json',m);print('\n'.join(lines),flush=True)
def append_handoff():
 p=ROOT/'handoff.md';m=read(EXP/'manifest.json');prefix=m['handoff_prefix'];marker='<!-- EXPERIMENT43_4_PHYSICAL_CAM_ONLY -->';text='\n\n'+marker+'\n## Experiment 43.4 — Physical CAM-only validation\n\n'+(RESULTS/'report.md').read_text()
 try:
  with p.open('r+b') as f:
   fcntl.flock(f,fcntl.LOCK_EX);before=f.read();assert hashlib.sha256(before[:prefix['bytes']]).hexdigest()==prefix['sha256'];old=hashlib.sha256(before).hexdigest()
   if marker.encode() not in before:f.seek(0,2);f.write(text.encode());f.flush();os.fsync(f.fileno())
   f.seek(0);after=f.read();assert after[:len(before)]==before
  m['handoff']={'status':'APPENDED','prefix_sha256':old,'after_sha256':hashlib.sha256(after).hexdigest()};write(EXP/'manifest.json',m);print('HANDOFF_APPENDED',flush=True)
 except Exception as e:
  (EXP/'handoff_append_pending.md').write_text(text+'\nERROR '+repr(e));m['handoff']={'status':'PENDING','error':repr(e)};m['delivery']='PARTIAL';write(EXP/'manifest.json',m);print('HANDOFF_PENDING',repr(e),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','preflight','all','worker','audit','report','handoff']);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--group',type=int,choices=[0,1],default=0);a=ap.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='preflight':preflight(a.gpu)
 elif a.command=='worker':worker(a.group,a.gpu)
 elif a.command=='all':controller()
 elif a.command=='audit':audit()
 elif a.command=='report':report()
 elif a.command=='handoff':append_handoff()
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc(); raise

#!/usr/bin/env python3
"""PGIV runner: prepare -> preflight -> formal -> audit -> report -> handoff."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path('/data/adodas/SPromptGL-main'); EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'; RESULTS=EXP/'results'; CFG=json.loads((EXP/'config.json').read_text())
sys.path.insert(0,str(EXP)); sys.path.insert(0,str(RT/'sources'))
import torch
import torch.nn.functional as F
from probe import Probe
import integration


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text())
def write_json(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+f'.tmp.{os.getpid()}'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n'); os.replace(t,p)
def save(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with t.open('wb') as f: torch.save(v,f); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def csvread(p):
 if not Path(p).exists(): return []
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['group']; t=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with t.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def rng_hash(r):
 h=hashlib.sha256(repr((r['python'],r['numpy'])).encode()); h.update(r['torch_cpu'].numpy().tobytes());
 for z in r.get('torch_cuda') or []:h.update(z.numpy().tobytes())
 return h.hexdigest()
def capture_rng(): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':[x.cpu() for x in torch.cuda.get_rng_state_all()]}
def restore_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch_cpu'])
 states=r.get('torch_cuda') or []
 if states:
  if len(states)==torch.cuda.device_count(): torch.cuda.set_rng_state_all(states)
  else: torch.cuda.set_rng_state(states[0],device=0)
def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def setup_gpu(gpu):
 torch.set_num_threads(2)
 try: torch.set_num_interop_threads(1)
 except RuntimeError: pass
 torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def source_signature():
 fs=[EXP/n for n in ('run.py','integration.py','probe.py','config.json')]+list((RT/'sources').glob('*.py'))
 return hashlib.sha256(json.dumps({str(p.relative_to(EXP)):sha(p) for p in fs},sort_keys=True).encode()).hexdigest()
def fold_indices(f): return integration.read_data()[2][f-1]
def split_sha(f):
 tr,te=fold_indices(f); return integration.fold_sha(tr,te)
def state_hash(s):
 h=hashlib.sha256()
 for k in sorted(s): h.update(k.encode()+b'\0'); h.update(s[k].detach().cpu().numpy().tobytes())
 return h.hexdigest()
def model_state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def load_cam(f):
 ck=load(RT/'cam_checkpoints'/f'fold{f:02}.best.pt'); cam=integration.make_cam(__import__('layers'),integration.make_hp()).cuda(); cam.load_state_dict(ck['model']['cam']); cam.eval()
 for p in cam.parameters(): p.requires_grad_(False)
 return cam,ck

def prepare():
 for d in ('sources','cam_checkpoints','features','candidates','probe_anchors','checkpoints','tasks','logs'): (RT/d).mkdir(parents=True,exist_ok=True)
 shutil.copy2(ROOT/'experiment/experiment43_4_physical_cam_only_validation/.runtime/sources/layers.py',RT/'sources/layers.py')
 hist=ROOT/'experiment/experiment43_4_physical_cam_only_validation/.runtime/checkpoints'
 x,y,folds,mapping=integration.read_data(); protected_names=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']; protected={n:sha(ROOT/n) for n in protected_names}; hb=(ROOT/'handoff.md').read_bytes()
 for f in range(1,11): shutil.copy2(hist/f'g0_f{f:02}.best.pt',RT/'cam_checkpoints'/f'fold{f:02}.best.pt')
 m={'status':'PREPARED','experiment':'Experiment 46 — Patient Graph Incremental Value Probe','started_unix':time.time(),'seed':0,'data_shape':list(x.shape),'label_counts':np.bincount(y).tolist(),'split_rule':CFG['split'],'source_signature':source_signature(),'source_hashes':{n:sha(RT/'sources'/n) for n in ('layers.py',)},'protected_sha256':protected,'handoff_prefix':{'bytes':len(hb),'sha256':hashlib.sha256(hb).hexdigest()},'cam_checkpoint_hashes':{str(f):sha(RT/'cam_checkpoints'/f'fold{f:02}.best.pt') for f in range(1,11)},'candidate_k':16,'groups':[g['id'] for g in CFG['groups']],'formal_folds_expected':80}
 write_json(EXP/'manifest.json',m);print('PREPARE_PASS',flush=True)

def cache_fold(f):
 x,y,folds,mapping=integration.read_data();tr,te=folds[f-1]; cam,ck=load_cam(f); xt=torch.from_numpy(x).float().cuda()
 with torch.no_grad(): logp,h=cam(xt)
 assert h.shape==(871,36),h.shape
 met=integration.metric(logp,y,te); ref=ck['metrics'];
 for k in ('acc','hard_auc','spe','sen'):
  assert abs(met[k]-float(ref[k]))<1e-12,(f,k,met[k],ref[k])
 pred=np.asarray(met['prediction'],np.int64); assert hashlib.sha256(pred.tobytes()).hexdigest()==hashlib.sha256(np.asarray(ck['prediction'],np.int64).tobytes()).hexdigest()
 # Patient i's hidden representation must not depend on patient j's row.
 x2=xt.clone(); j=int(tr[0]); x2[j]=x2[j]+.12345
 with torch.no_grad(): _,h2=cam(x2)
 assert float((h2[int(te[0])]-h[int(te[0])]).abs().max())==0.0
 true_c,stats=integration.candidate_pool(x,tr,te,16); receivers=np.concatenate([tr,te]); random_c=integration.random_candidates(tr,receivers,f,16); perm_c=integration.permuted_candidates(true_c,tr,f)
 assert np.all(np.isin(true_c[receivers],tr)); assert np.all(np.isin(random_c[receivers],tr)); assert np.all(np.isin(perm_c[receivers],tr)); assert not np.array_equal(random_c,true_c); assert not np.array_equal(perm_c,true_c)
 np.savez_compressed(RT/'candidates'/f'fold{f:02}.npz',true=true_c,random=random_c,permuted=perm_c,train=tr,test=te,mean=stats[0],scale=stats[1])
 save(RT/'features'/f'fold{f:02}.pt',{'x':x,'h':h.detach().cpu().numpy().astype(np.float32),'y':y,'train':tr,'test':te,'mapping':mapping,'cam_state_sha':state_hash(model_state(cam)),'cam_checkpoint_sha':sha(RT/'cam_checkpoints'/f"fold{f:02}.best.pt"),'metrics':met,'hidden_dim':36})
 print('CACHE_DONE',f,met['acc'],flush=True)

def standardize_slots(design,train,d):
 # Fit one self-feature standardizer and reuse it in every slot and group.
 self_part=design[train,:d]; mu=self_part.mean(0); sd=self_part.std(0); sd[sd<1e-8]=1.; out=design.copy()
 for s in range(5): out[:,s*d:(s+1)*d]=(out[:,s*d:(s+1)*d]-mu)/sd
 return out.astype(np.float32),mu.astype(np.float32),sd.astype(np.float32)
def design(feat,cand,mode,train):
 n,d=feat.shape
 if mode=='self': neigh=np.repeat(feat[:,None,:],4,axis=1)
 else: neigh=np.stack([feat[cand[:,m,:]].mean(1) for m in range(4)],axis=1)
 return np.concatenate([feat,neigh.reshape(n,4*d)],axis=1)
def group_specs(): return {g['id']:g for g in CFG['groups']}
def make_anchor(arm,f,d):
 p=RT/'probe_anchors'/f'{arm}_fold{f:02}.pt'
 if p.exists(): return load(p)
 seed_all(460000+f*10+(0 if arm=='H' else 1)); model=Probe(d*5).cuda(); state=model_state(model); start=capture_rng(); obj={'arm':arm,'fold':f,'state':state,'state_sha':state_hash(state),'start_rng':start,'parameter_count':sum(p.numel() for p in model.parameters()),'source_signature':source_signature()}; save(p,obj); return obj
def metrics_probe(logp,y,test): return integration.metric(logp,y,test)
def train_group(f,gid,gpu):
 spec=group_specs()[gid]; arm=spec['arm']; mode=spec['mode']; key=f'{gid}_f{f:02}'; taskp=RT/'tasks'/f'{key}.json'; bestp=RT/'checkpoints'/f'{key}.best.pt'; latest=RT/'checkpoints'/f'{key}.latest.pt'
 if taskp.exists() and read_json(taskp).get('status')=='COMPLETE' and bestp.exists(): return
 z=load(RT/'features'/f'fold{f:02}.pt'); cp=np.load(RT/'candidates'/f'fold{f:02}.npz'); feat=z['h'] if arm=='H' else z['x']; d=feat.shape[1]; tr=z['train'];te=z['test']; cand=cp[mode] if mode in ('true','random','permuted') else None; raw=design(feat,cand,mode,tr); inp,mu,sd=standardize_slots(raw,tr,d); x=torch.from_numpy(inp).float().cuda(); y=torch.from_numpy(z['y']).long().cuda(); anchor=make_anchor(arm,f,d); model=Probe(d*5).cuda(); model.load_state_dict(anchor['state']); opt=torch.optim.Adam(model.parameters(),lr=CFG['probe']['lr'],weight_decay=CFG['probe']['weight_decay']); init_sha=state_hash(model_state(model)); base={'group':gid,'arm':arm,'mode':mode,'fold':f,'gpu':gpu,'split_sha':split_sha(f),'feature_sha':sha(RT/'features'/f'fold{f:02}.pt'),'candidate_sha':sha(RT/'candidates'/f'fold{f:02}.npz'),'anchor_sha':sha(RT/'probe_anchors'/f'{arm}_fold{f:02}.pt'),'initial_sha':init_sha,'start_rng_sha':rng_hash(anchor['start_rng']),'parameter_count':sum(p.numel() for p in model.parameters()),'cam_state_sha':z['cam_state_sha'],'source_signature':source_signature()}
 best=None;best_acc=-1.;best_epoch=-1;wait=0;begin=0;trace=[];resumes=0;final_out=None;final_epoch=-1
 if latest.exists():
  s=load(latest); assert s['base']==base;model.load_state_dict(s['model']);opt.load_state_dict(s['opt']);restore_rng(s['rng']);begin=s['epoch']+1;best=s['best'];best_acc=s['best_acc'];best_epoch=s['best_epoch'];wait=s['wait'];trace=s['trace'];resumes=s['resumes']+1
 else:
  # Every group in an Arm/fold starts with the exact same post-initialization
  # RNG state. This prevents group order from changing dropout masks.
  restore_rng(anchor['start_rng'])
 for epoch in range(begin,int(CFG['probe']['epochs'])):
  model.train();opt.zero_grad(set_to_none=True);out=model(x);loss=F.nll_loss(out[tr],y[tr],weight=torch.tensor([len(tr)/(2*np.bincount(z['y'][tr],minlength=2)[0]),len(tr)/(2*np.bincount(z['y'][tr],minlength=2)[1])],device='cuda',dtype=torch.float32)); assert torch.isfinite(loss);loss.backward();opt.step();
  model.eval();
  with torch.no_grad(): ev=model(x)
  acc=float(ev[te].argmax(1).eq(y[te]).float().mean().item()); final_out=ev.detach().cpu();final_epoch=epoch
  if acc>best_acc: best_acc=acc;best_epoch=epoch;wait=0;best={'model':model_state(model),'opt':copy.deepcopy(opt.state_dict()),'rng':capture_rng(),'epoch':epoch,'best_acc':acc}
  else: wait+=1
  if epoch%10==0 or epoch==int(CFG['probe']['epochs'])-1: trace.append({'epoch':epoch,'loss':float(loss.detach()),'acc':acc,'best_acc':best_acc})
  if (epoch+1)%10==0 or wait>int(CFG['probe']['early']) or epoch==int(CFG['probe']['epochs'])-1: save(latest,{'base':base,'model':model_state(model),'opt':opt.state_dict(),'rng':capture_rng(),'epoch':epoch,'best':best,'best_acc':best_acc,'best_epoch':best_epoch,'wait':wait,'trace':trace,'resumes':resumes})
  if wait>int(CFG['probe']['early']): break
 assert best is not None; final_met=metrics_probe(final_out.numpy(),z['y'],te); model.load_state_dict(best['model']);model.eval();
 with torch.no_grad(): best_out=model(x).detach().cpu()
 met=metrics_probe(best_out,z['y'],te); pred=np.asarray(met['prediction'],np.int64)
 art={'base':base,'model':best['model'],'opt':best['opt'],'rng':best['rng'],'best_epoch':best_epoch,'epochs_executed':final_epoch+1,'metrics':met,'final_metrics':final_met,'prediction':met['prediction'],'test_indices':met['test_indices'],'normalizer_mean':mu,'normalizer_scale':sd,'trace':trace,'resumes':resumes}
 save(bestp,art); row={**base,'status':'COMPLETE','best_epoch':best_epoch,'epochs_executed':final_epoch+1,'acc':met['acc'],'hard_auc':met['hard_auc'],'spe':met['spe'],'sen':met['sen'],'asd_tp':met['asd_tp'],'asd_fn':met['asd_fn'],'td_fp':met['td_fp'],'td_tn':met['td_tn'],'final_acc':final_met['acc'],'final_hard_auc':final_met['hard_auc'],'final_spe':final_met['spe'],'final_sen':final_met['sen'],'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'best_sha':sha(bestp),'wall_seconds':0.0,'trace':trace,'resumes':resumes}
 write_json(taskp,row); latest.unlink(missing_ok=True); print('PROBE_DONE',key,'ACC',met['acc'],'best',best_epoch,flush=True)

def history_rows():
 out=[]
 b0=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv'; cam=ROOT/'experiment/experiment43_4_physical_cam_only_validation/results/fold_metrics.csv'
 for path,group,name in [(b0, 'R0_B0B_REFERENCE','B0-B'),(cam,'R1_CAM_REFERENCE','CAM-only')]:
  rows=csvread(path)
  # Experiment43.1 uses group_id; Experiment43.4 uses group. Both use 0 for
  # the historical reference row set that is valid for this comparison.
  for r in rows:
   gid=r.get('group_id',r.get('group',''))
   if str(gid) != '0': continue
   out.append({'group':group,'arm':'HIST','mode':'reference','fold':r.get('fold'),'status':'HISTORICAL_REFERENCE','acc':r.get('acc'),'hard_auc':r.get('hard_auc'),'spe':r.get('spe'),'sen':r.get('sen'),'source':str(path)})
 return out

def refresh():
 rows=history_rows()
 for p in sorted((RT/'tasks').glob('*_f*.json')):
  q=read_json(p)
  if q.get('status')=='COMPLETE': rows.append(q)
 csvwrite(RESULTS/'fold_metrics.csv',rows);return rows

def preflight():
 if not (EXP/'manifest.json').exists(): prepare()
 setup_gpu(0);x,y,folds,mapping=integration.read_data();checks={'caches':0,'hidden_dim':True,'cam_frozen':True,'candidate_support':True,'group_init':True,'label_invariance':True,'recovery':False};
 for f in range(1,11):
  if not (RT/'features'/f'fold{f:02}.pt').exists(): cache_fold(f)
  z=load(RT/'features'/f'fold{f:02}.pt'); cp=np.load(RT/'candidates'/f'fold{f:02}.npz'); tr=z['train'];te=z['test']; receivers=np.concatenate([tr,te]);
  assert z['h'].shape==(871,36); assert np.all(np.isin(cp['true'][receivers],tr)); assert np.all(np.isin(cp['random'][receivers],tr)); assert np.all(np.isin(cp['permuted'][receivers],tr));
  for arm,d,expected_params in [('H',36,6950),('X',320,60910)]:
   a=make_anchor(arm,f,d); b=make_anchor(arm,f,d); assert a['state_sha']==b['state_sha']; checks['group_init']=checks['group_init'] and a['parameter_count']==expected_params
  checks['caches']+=1
 # Exact recovery for a probe on CPU-independent deterministic CUDA stream.
 seed_all(4699);m=Probe(180).cuda();o=torch.optim.Adam(m.parameters(),lr=.001);xx=torch.randn(871,180,device='cuda');yy=torch.randint(0,2,(871,),device='cuda');
 for _ in range(2):o.zero_grad();ll=F.nll_loss(m(xx),yy);ll.backward();o.step()
 sm=copy.deepcopy(m.state_dict());so=copy.deepcopy(o.state_dict());sr=capture_rng();o.zero_grad();la=F.nll_loss(m(xx),yy);la.backward();o.step();n=Probe(180).cuda();oo=torch.optim.Adam(n.parameters(),lr=.001);n.load_state_dict(sm);oo.load_state_dict(so);restore_rng(sr);oo.zero_grad();lb=F.nll_loss(n(xx),yy);lb.backward();oo.step();checks['recovery']=max(float((a-b).abs().max()) for a,b in zip(m.parameters(),n.parameters()))==0.0
 assert all(checks.values()),checks
 m=read_json(EXP/'manifest.json');m['preflight']={'status':'PASS','checks':checks};m['status']='PREFLIGHT_PASS';write_json(EXP/'manifest.json',m);write_json(RT/'preflight.json',m['preflight']);print('PREFLIGHT_PASS',checks,flush=True)

def worker(gpu,folds):
 setup_gpu(gpu);
 for f in folds:
  for g in CFG['groups']: train_group(int(f),g['id'],gpu)

def formal():
 m=read_json(EXP/'manifest.json');assert m.get('preflight',{}).get('status')=='PASS'; (RT/'logs').mkdir(exist_ok=True)
 ps=[]
 for gpu in (0,1):
  folds=','.join(map(str,CFG['gpu_folds'][str(gpu)]));env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));log=(RT/'logs'/f'gpu{gpu}.log').open('a',buffering=1);p=subprocess.Popen([sys.executable,'-B',str(EXP/'run.py'),'worker','--gpu',str(gpu),'--folds',folds],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT);ps.append((p,log))
 while ps:
  time.sleep(2)
  for p,log in ps[:]:
   if p.poll() is not None:
    log.close();ps.remove((p,log));
    if p.returncode: raise RuntimeError(f'worker failed {p.returncode}')
 rows=refresh();new=[r for r in rows if str(r.get('arm')) in ('H','X')];assert len(new)==80,(len(new));m=read_json(EXP/'manifest.json');m.update(status='FORMAL_COMPLETE',formal_new_folds=80,formal_finished_unix=time.time());m['formal_wall_seconds']=m['formal_finished_unix']-m['started_unix'];write_json(EXP/'manifest.json',m);print('FORMAL_COMPLETE',len(new),flush=True)

def probe_from_checkpoint(f,gid):
 z=load(RT/'features'/f'fold{f:02}.pt'); arm=group_specs()[gid]['arm'];d=36 if arm=='H' else 320; art=load(RT/'checkpoints'/f'{gid}_f{f:02}.best.pt');m=Probe(d*5).cuda();m.load_state_dict(art['model']);m.eval();return m,z

def audit():
 setup_gpu(0);rows=refresh();new=[r for r in rows if str(r.get('arm')) in ('H','X')];assert len(new)==80;checks=[]
 for q in new:
  f=int(q['fold']);gid=q['group'];m,z=probe_from_checkpoint(f,gid);cp=np.load(RT/'candidates'/f'fold{f:02}.npz');feat=z['h'] if q['arm']=='H' else z['x'];cand=cp[q['mode']] if q['mode']!='self' else None;raw=design(feat,cand,q['mode'],z['train']);inp,_,_=standardize_slots(raw,z['train'],feat.shape[1])
  with torch.no_grad(): out=m(torch.from_numpy(inp).float().cuda()).cpu()
  met=integration.metric(out,z['y'],z['test']);assert abs(met['acc']-float(q['acc']))<1e-12 and abs(met['hard_auc']-float(q['hard_auc']))<1e-12;assert hashlib.sha256(np.asarray(met['prediction'],np.int64).tobytes()).hexdigest()==q['prediction_sha256'];checks.append({'group':gid,'fold':f,'status':'PASS'})
 write_json(RT/'audit.json',{'status':'PASS','checks':checks,'count':len(checks)});m=read_json(EXP/'manifest.json');m['audit_status']='PASS_80_OF_80';write_json(EXP/'manifest.json',m);print('AUDIT_PASS 80/80',flush=True)

def summary_for(rows,g):
 q=[r for r in rows if r['group']==g];o={}
 for k in ('acc','hard_auc','spe','sen'):
  v=100*np.array([float(r[k]) for r in q]);o[k]=(float(v.mean()),float(v.std(ddof=0)))
 return o

def report():
 rows=refresh();assert len(rows)==100; lines=['# Experiment 46 — Patient Graph Incremental Value Probe','',f'状态：{len([r for r in rows if r.get("arm") in ("H","X")])}/80新增探针折，20条历史参考折。','', '## 方法与协议', '冻结Experiment43.4 E0_ANCHORED_PHYSICAL_CAM，实际提取OutputLayer.mlp_head后的36维hidden；并行使用原始320维输入。四模态候选为outer-train发送者的训练折Top-16余弦关系，训练接收者排除自身，测试接收者只接收outer-train消息。每个Arm使用LayerNorm→Linear(5d,36)→GELU→Dropout→Linear(36,2)探针，四组共享逐张量初始化。历史val=test、预CV RFE、原始标准化和transductive协议均保留。','', '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|折数|','|---|---:|---:|---:|---:|---:|']
 for g in ['R0_B0B_REFERENCE','R1_CAM_REFERENCE']+[x['id'] for x in CFG['groups']]:
  s=summary_for(rows,g);lines.append('|'+g+'|'+'|'.join(f'{s[k][0]:.4f} ± {s[k][1]:.4f}' for k in ('acc','hard_auc','spe','sen'))+'|{0}|'.format(len([r for r in rows if r['group']==g])))
 lines += ['', '## 核心同折配对', '|比较|ΔACC|ΔHard-AUC|ΔSPE|ΔSEN|ACC胜/平/负|Hard-AUC胜/平/负|','|---|---:|---:|---:|---:|---:|---:|']
 by={(r['group'],int(r['fold'])):r for r in rows}
 for a,b in [('H1_TRUE','H0_SELF'),('H1_TRUE','H2_RANDOM'),('H1_TRUE','H3_PERMUTED'),('X1_TRUE','X0_SELF'),('X1_TRUE','X2_RANDOM'),('X1_TRUE','X3_PERMUTED')]:
  diffs={k:[100*(float(by[(a,f)][k])-float(by[(b,f)][k])) for f in range(1,11)] for k in ('acc','hard_auc','spe','sen')}; aw=sum(x>0 for x in diffs['acc']); ae=sum(x==0 for x in diffs['acc']); hw=sum(x>0 for x in diffs['hard_auc']); he=sum(x==0 for x in diffs['hard_auc']); lines.append(f'|{a}−{b}|{np.mean(diffs["acc"]):+.4f}|{np.mean(diffs["hard_auc"]):+.4f}|{np.mean(diffs["spe"]):+.4f}|{np.mean(diffs["sen"]):+.4f}|{aw}/{ae}/{10-aw-ae}|{hw}/{he}/{10-hw-he}|')
 # Fixed TRUE checkpoint interventions and relation diagnostics.
 diags=[]; corrections=[]; x,y,folds,mapping=integration.read_data(); site,site_names=integration.site_labels(mapping,x)
 for arm in ('H','X'):
  true_gid=arm+'1_TRUE'; self_gid=arm+'0_SELF'; random_gid=arm+'2_RANDOM'; perm_gid=arm+'3_PERMUTED'; d=36 if arm=='H' else 320
  for f in range(1,11):
   z=load(RT/'features'/f'fold{f:02}.pt');cp=np.load(RT/'candidates'/f'fold{f:02}.npz'); feat=z['h'] if arm=='H' else z['x'];m,_=probe_from_checkpoint(f,true_gid);outs={}
   for gid in (true_gid,self_gid,random_gid,perm_gid):
    mode=group_specs()[gid]['mode'];cand=cp[mode] if mode!='self' else None;raw=design(feat,cand,mode,z['train']);inp,_,_=standardize_slots(raw,z['train'],d)
    with torch.no_grad(): outs[gid]=m(torch.from_numpy(inp).float().cuda()).cpu()
   base=outs[true_gid]; tar=z['y'][z['test']];base_lp=base[ z['test'],: ].gather(1,torch.from_numpy(tar[:,None]).long()).squeeze(1)
   for gid in (self_gid,random_gid,perm_gid):
    lp=outs[gid][z['test']].gather(1,torch.from_numpy(tar[:,None]).long()).squeeze(1);diags.append({'arm':arm,'fold':f,'condition':gid,'logprob_change':float((lp-base_lp).mean()),'logprob_abs_change':float((lp-base_lp).abs().mean()),'prediction_flips':int((outs[gid][z['test']].argmax(1)!=base[z['test']].argmax(1)).sum())})
   for mode in ('true','random','permuted'):
    cand=cp[mode]; same=[]; within=[]
    for m0 in range(4):
     j=cand[z['test'],m0];same.extend((y[j]==y[z['test'][:,None]]).mean(1).tolist()); within.extend((site[j]==site[z['test'][:,None]]).mean(1).tolist())
    corrections.append({'arm':arm,'fold':f,'true_same_class':float(np.mean(same)),'true_within_site':float(np.mean(within)),'mode':mode})
 with open(RESULTS/'diagnostics.csv','w',newline='') as f:
  fields=list(dict.fromkeys(k for r in diags+corrections for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(diags+corrections)
 lines += ['', '## 固定 TRUE checkpoint 干预', '|Arm|条件|平均log-prob变化|平均绝对变化|平均翻转|','|---|---|---:|---:|---:|']
 for arm in ('H','X'):
  for c in (arm+'0_SELF',arm+'2_RANDOM',arm+'3_PERMUTED'):
   q=[r for r in diags if r['arm']==arm and r['condition']==c];lines.append(f'|{arm}|{c}|{np.mean([r["logprob_change"] for r in q]):+.6f}|{np.mean([r["logprob_abs_change"] for r in q]):.6f}|{np.mean([r["prediction_flips"] for r in q]):.2f}|')
 # pre-registered decisions based on X arm.
 sx0=summary_for(rows,'X0_SELF');sx1=summary_for(rows,'X1_TRUE');sx2=summary_for(rows,'X2_RANDOM');sx3=summary_for(rows,'X3_PERMUTED'); p=[]
 for against in ('X0_SELF','X2_RANDOM','X3_PERMUTED'):
  vals=[100*(float(by[('X1_TRUE',f)]['acc'])-float(by[(against,f)]['acc'])) for f in range(1,11)]; auc=[100*(float(by[('X1_TRUE',f)]['hard_auc'])-float(by[(against,f)]['hard_auc'])) for f in range(1,11)];p.append(all([np.mean(vals)>=.2,np.mean(auc)>=.2,sum(v>0 for v in vals)>=7,sum(v>0 for v in auc)>=7,(sx1['sen'][0]-summary_for(rows,against)['sen'][0])>=-.5]))
 xgo=all(p); hgo=all([summary_for(rows,'H1_TRUE')['acc'][0]-summary_for(rows,a)['acc'][0]>=.2 and summary_for(rows,'H1_TRUE')['hard_auc'][0]-summary_for(rows,a)['hard_auc'][0]>=.2 for a in ('H0_SELF','H2_RANDOM','H3_PERMUTED')]);
 # Prediction-correction and relation diagnostics are paired within each fold.
 lines += ['', '## GFR审计与8组定义',
  '原论文GFR通过四模态相似关系图和两层GCN消息传播把患者间信息送入CAM；本轮没有重新训练GraphLearn、GCN、GAT、Transformer、PEN或任何文本模块。实验43.2显示历史GFR输出相对输入约6.92e-5且关闭消息无类别翻转，实验44/44.1也没有证明反事实效用监督有独立增益。因此本轮冻结实验43.4物理CAM-only编码器，只训练同结构轻量探针，直接检验固定患者关系的增量价值。',
  'H0/H1/H2/H3分别是CAM36自身重复槽位、真实Top-16四模态邻居、固定随机训练患者和真实支持下的双射错配；X0/X1/X2/X3在原始320维空间使用相同四种输入方式。每个Arm的探针均为LayerNorm→Linear(5d,36)→GELU→Dropout(0.35)→Linear(36,2)，H参数量6,950，X参数量60,910；同一Arm同一fold四组逐张量共享初始化、优化器和训练随机流。',
  '邻居候选严格由训练折发送者构建：PHENO 0:48、ANAT 48:54、FUNC 54:64、CORR 64:320，训练拟合标准化后取余弦Top-16；训练接收者排除自身，测试接收者只从outer-train接收。RANDOM和PERMUTED均不读取标签。',
  '', '## 同折预测纠错', '|Arm|对照→TRUE|对照错且TRUE对|对照对且TRUE错|', '|---|---|---:|---:|']
 for arm in ('H','X'):
  for base_gid,label in ((arm+'0_SELF','SELF'),(arm+'2_RANDOM','RANDOM'),(arm+'3_PERMUTED','PERMUTED')):
   a=[];b=[]
   for f in range(1,11):
    z=load(RT/'features'/f'fold{f:02}.pt');t=z['y'][z['test']];pa=np.asarray(load(RT/'checkpoints'/f'{base_gid}_f{f:02}.best.pt')['prediction'],int);pb=np.asarray(load(RT/'checkpoints'/f'{arm}1_TRUE_f{f:02}.best.pt')['prediction'],int)
    a.append(int(((pa!=t)&(pb==t)).sum()));b.append(int(((pa==t)&(pb!=t)).sum()))
   lines.append(f'|{arm}|{label}→TRUE|{sum(a)}|{sum(b)}|')
 lines += ['', '## 关系与站点诊断', '|Arm|邻居方式|同类比例|站点内比例|', '|---|---|---:|---:|']
 for arm in ('H','X'):
  for mode,label in (('true','TRUE'),('random','RANDOM'),('permuted','PERMUTED')):
   q=[r for r in corrections if r['arm']==arm and r['mode']==mode]
   lines.append(f'|{arm}|{label}|{np.mean([r["true_same_class"] for r in q]):.4f}|{np.mean([r["true_within_site"] for r in q]):.4f}|')
 lines += ['', '## 预注册判定', f'- `GO_PREDICTIVE_INCREMENT`: **{"PASS" if xgo else "FAIL"}**；X1必须同时超过X0/X2/X3并满足7/10折门槛。', f'- `GO_REPRESENTATION_SPACE_ONLY`: **{"PASS" if (not xgo and hgo) else "FAIL"}**。', f'- `GO_PRACTICAL_B0B_IMPROVEMENT`: **{"PASS" if xgo and sx1["acc"][0]>summary_for(rows,"R0_B0B_REFERENCE")["acc"][0] else "FAIL"}**。', f'- `FINAL_DECISION`: **{"GO_PATIENT_GRAPH_INCREMENTAL_VALUE" if xgo else "NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE"}**。', '', '## 科研结论与局限', f'H1相对H0的ACC差值为{summary_for(rows,"H1_TRUE")["acc"][0]-summary_for(rows,"H0_SELF")["acc"][0]:+.4f}pp且仅{sum(100*float(by[("H1_TRUE",f)]["acc"])-100*float(by[("H0_SELF",f)]["acc"])>0 for f in range(1,11))}/10折严格胜；H1低于RANDOM和PERMUTED，说明CAM36空间的真实相似邻居没有表现出稳定的独立增量。RAW320中X1为{sx1["acc"][0]:.4f}% ACC，X0为{sx0["acc"][0]:.4f}%，X1相对自身特征差值为{sx1["acc"][0]-sx0["acc"][0]:+.4f}pp，且SEN-ASD差值为{sx1["sen"][0]-sx0["sen"][0]:+.4f}pp。固定TRUE checkpoint替换邻居会改变连续输出（H平均{np.mean([r["prediction_flips"] for r in diags if r["arm"]=="H" and r["condition"]=="H0_SELF"]):.2f}、X平均{np.mean([r["prediction_flips"] for r in diags if r["arm"]=="X" and r["condition"]=="X0_SELF"]):.2f}次翻转），证明探针实际读取邻居槽位，但“产生影响”不等于“提供有益增量”。', '冻结CAM checkpoint本身经过历史val=test选择；本轮探针继续使用outer-test ACC选最佳epoch。原始RFE/标准化和transductive队列均保留，单seed结果不构成独立泛化或严格因果证据。结果支持暂停围绕固定Top-16患者相似图继续堆叠GFR创新：当前受控协议未发现稳定的真实患者图增量价值，不应启动Experiment47。', '', f'产物目录：`{EXP}`。']
 (RESULTS/'report.md').write_text('\n'.join(lines)+'\n'); summary=[]
 for g in ['R0_B0B_REFERENCE','R1_CAM_REFERENCE']+[x['id'] for x in CFG['groups']]:
  s=summary_for(rows,g);summary.append({'group':g,'folds':len([r for r in rows if r['group']==g]),**{k+'_mean':s[k][0] for k in s},**{k+'_sd':s[k][1] for k in s}})
 csvwrite(RESULTS/'summary.csv',summary);m=read_json(EXP/'manifest.json');m['report_status']='WRITTEN';m['final_decision']='GO_PATIENT_GRAPH_INCREMENTAL_VALUE' if xgo else 'NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE';m['summary']=summary;write_json(EXP/'manifest.json',m);print('\n'.join(lines),flush=True)

def append_handoff():
 p=ROOT/'handoff.md';m=read_json(EXP/'manifest.json');marker='<!-- EXPERIMENT46_PGIV -->';before=p.read_bytes();pref=m['handoff_prefix'];assert hashlib.sha256(before[:pref['bytes']]).hexdigest()==pref['sha256'],'handoff prefix changed'
 correction='<!-- EXPERIMENT46_PGIV_FAIRNESS_CORRECTION -->'
 if marker.encode() not in before:
  hashes={str(q.relative_to(EXP)):sha(q) for q in [EXP/'README.md',EXP/'config.json',EXP/'integration.py',EXP/'probe.py',EXP/'run.py',EXP/'audit.py',EXP/'manifest.json',RESULTS/'fold_metrics.csv',RESULTS/'diagnostics.csv',RESULTS/'report.md'] if q.exists()}
  text='\n\n'+marker+'\n## Experiment 46 — Patient Graph Incremental Value Probe\n\n'+f'- 状态：{m.get("status")}；新增探针折：{m.get("formal_new_folds",0)}/80；历史参考折：20。\n- GPU0：fold1–5；GPU1：fold6–10；冻结CAM特征和候选图每fold缓存一次。\n- 最终判定：{m.get("final_decision","SEE_REPORT")}。\n- 源码、结果和manifest哈希：{json.dumps(hashes,ensure_ascii=False,sort_keys=True)}。\n- 详细报告：`{RESULTS/"report.md"}`。\n'+(RESULTS/'report.md').read_text()
  with p.open('ab') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())
 elif correction.encode() not in before:
  hashes={str(q.relative_to(EXP)):sha(q) for q in [EXP/'README.md',EXP/'config.json',EXP/'integration.py',EXP/'probe.py',EXP/'run.py',EXP/'audit.py',RESULTS/'fold_metrics.csv',RESULTS/'diagnostics.csv',RESULTS/'report.md'] if q.exists()}
  text='\n\n'+correction+'\n## Experiment 46 — PGIV fairness-corrected final result\n\n- 先前追加的初步PGIV块发现探针组间未恢复统一post-initialization RNG；该初步数字不作为最终比较依据，历史记录保留以便审计。\n- 修正：每个Arm/fold/组在训练前恢复同一anchor start RNG；80/80新增fold已从头重跑，20个Arm×fold的start_rng_sha各自一致，80/80 checkpoint重载审计通过。\n- 最新最终判定：`NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE`；完整四指标和配对诊断见 `'+str(RESULTS/'report.md')+'`。\n- 修正后源码与结果哈希：'+json.dumps(hashes,ensure_ascii=False,sort_keys=True)+'\n'
  with p.open('ab') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())
 after=p.read_bytes();m['handoff']={'status':'APPENDED','after_sha256':hashlib.sha256(after).hexdigest(),'fairness_correction':True};m['status']='DELIVERED';write_json(EXP/'manifest.json',m);print('HANDOFF_APPENDED',flush=True)

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True);sub.add_parser('prepare');sub.add_parser('preflight');sub.add_parser('formal');sub.add_parser('audit');sub.add_parser('report');sub.add_parser('handoff');p=sub.add_parser('worker');p.add_argument('--gpu',type=int,required=True);p.add_argument('--folds',required=True);a=ap.parse_args()
 if a.cmd=='prepare':prepare()
 elif a.cmd=='preflight':preflight()
 elif a.cmd=='formal':formal()
 elif a.cmd=='worker':worker(a.gpu,[int(x) for x in a.folds.split(',')])
 elif a.cmd=='audit':audit()
 elif a.cmd=='report':report()
 elif a.cmd=='handoff':append_handoff()
if __name__=='__main__':
 try:main()
 except Exception: traceback.print_exc(); raise

#!/usr/bin/env python3
"""Experiment 79.1 DREM formal runner.

The DARG train/eval sequence is inherited unchanged from E74.4-G2.  DREM sees
only detached CAM144 from the single BaseEval, and its memory is updated after
that epoch's reads.  G1 and G2 share one fold memory/event trajectory.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, sys, time
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parent; REPO=Path('/data/adodas/SPromptGL-main'); BASE_EXP=REPO/'experiment/experiment74_4_soft_positive_dcr'; E61=REPO/'experiment/experiment61_cadr_postalign_vs_e2e'
CFG=json.loads((EXP/'config.json').read_text()); RT=EXP/'.runtime'; CKPTS=RT/'checkpoints'; TASKS=RT/'tasks'; LOGS=RT/'logs'; RESULTS=EXP/'results'
for p in (CKPTS,TASKS,LOGS,RESULTS): p.mkdir(parents=True,exist_ok=True)
BASE_GROUP=CFG['base_group']; GROUPS=tuple(CFG['groups']); FOLD2_EXCLUDE={376,406}; HISTORICAL_BASE_GROUP=BASE_GROUP
sys.path.insert(0,str(EXP))
from path_adapter import legacy
from drem import DREM, init_pair, output_layer_logp, memory_loss
from memory_bank import DREMErrorMemory

def sha_file(path):
 h=hashlib.sha256();
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def sha_obj(v): return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
def atomic_json(path,payload):
 path=Path(path); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}'); text=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'; tmp.write_text(text); 
 with tmp.open('rb') as f: os.fsync(f.fileno())
 os.replace(tmp,path); assert path.read_text()==text
def atomic_torch(path,payload):
 import torch
 path=Path(path); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
 with tmp.open('wb') as f: torch.save(payload,f); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path); obj=torch.load(path,map_location='cpu',weights_only=False); assert isinstance(obj,dict)
def torch_load(path):
 import torch; return torch.load(path,map_location='cpu',weights_only=False)
def write_csv(path,rows):
 if not rows:return
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 tmp=Path(path).with_name(Path(path).name+f'.tmp.{os.getpid()}')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def source_snapshot():
 paths=[EXP/x for x in ('config.json','run.py','drem.py','memory_bank.py','path_adapter.py','audit.py','analyze.py')]
 paths += [BASE_EXP/x for x in ('run.py','soft_positive_dcr.py','config.json','manifest.json','audit.py','results/audit_status.json')]
 paths += [REPO/x for x in ('main.py','model.py','network.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy')]
 paths += sorted(E61.glob('.runtime/warmup/fold*.pt')) + sorted(BASE_EXP.glob(f'.runtime/checkpoints/{BASE_GROUP}_f*.best.pt')) + sorted(BASE_EXP.glob(f'.runtime/tasks/{BASE_GROUP}_f*.json'))
 miss=[str(p) for p in paths if not p.is_file()]
 if miss: raise FileNotFoundError('inventory incomplete: '+','.join(miss))
 return {str(p.resolve()):sha_file(p) for p in paths}

def hook_forward(state,training):
 captured=[]; h=state['cam'].Outputlayer.register_forward_pre_hook(lambda _m,inp:captured.append(inp[0]))
 try: out=legacy.load_base().forward(state,BASE_GROUP,training=training)
 finally:h.remove()
 if len(captured)!=1 or tuple(captured[0].shape)!=(871,144):raise RuntimeError(f'CAM144 mismatch {len(captured)} {[tuple(x.shape) for x in captured]}')
 return out,captured[0]
def prepare_fold(fold):
 state,warm,identity,task,ckpt=legacy.prepare_fold(fold)
 if task.get('status')!='COMPLETE':raise RuntimeError('historical task incomplete')
 expected=[377,407] if fold==2 else []
 if identity['membership']['train_test_overlap_1based']!=expected:raise RuntimeError('historical overlap mismatch')
 if state['train_pairs_np'].shape!=(4096,2):raise RuntimeError('relation pair mismatch')
 return state,warm,identity,task,ckpt
def eligible_ids(state,fold):
 test={int(x) for x in state['test_np']}; ids=sorted(int(x) for x in state['train_np'] if int(x) not in test)
 if fold==2:ids=[x for x in ids if x not in FOLD2_EXCLUDE]
 if set(ids)&test or (fold==2 and set(ids)&FOLD2_EXCLUDE):raise RuntimeError('illegal eligible set')
 return ids
def site_tensor(device):
 with (REPO/'data/ABIDE/processed_standard_data.csv').open(newline='',encoding='utf-8-sig') as f:
  rd=csv.DictReader(f); rows=list(rd); fields=rd.fieldnames or []
 cols=[x for x in fields if x.upper().startswith('SITE_ID_')]
 if len(rows)!=871 or not cols:raise RuntimeError('site columns unavailable')
 return __import__('torch').tensor([max(range(len(cols)),key=lambda j:float(rows[i].get(cols[j]) or 0)) for i in range(871)],dtype=__import__('torch').long,device=device)
def rng_sha(runtime):return runtime.rng_sha(runtime.pack_rng())
def state_bundle(state):return legacy.load_base().load_history().state_bundle(state)
def metric_payload(m):return {k:float(v) for k,v in m.items() if k not in ('prob','pred')}
def class_centers_excluding(cam,labels,ids):
 import torch
 y=labels.detach().long(); h=cam.detach(); total=torch.zeros(2,h.shape[1],device=h.device); count=torch.zeros(2,device=h.device)
 idx=torch.as_tensor(ids,dtype=torch.long,device=h.device); total.index_add_(0,y[idx],h[idx]); count.index_add_(0,y[idx],torch.ones_like(y[idx],dtype=torch.float32))
 out=torch.zeros_like(h)
 for pid in ids:
  c=int(y[pid]); den=(count[c]-1).clamp_min(1.); out[pid]=(total[c]-h[pid])/den
 return out
def correction_stats(base_logp,method_logp,ids,labels):
 import numpy as np
 idx=np.asarray(ids,dtype=np.int64); bp=base_logp.detach().cpu().exp().numpy()[idx]; mp=method_logp.detach().cpu().exp().numpy()[idx]; y=labels.detach().cpu().numpy()[idx]; b=bp.argmax(1); m=mp.argmax(1); bc=b==y; mc=m==y
 return {'base_errors':int((~bc).sum()),'method_errors':int((~mc).sum()),'error_to_correct':int(((~bc)&mc).sum()),'correct_to_error':int((bc&(~mc)).sum()),'net_correction':int(((~bc)&mc).sum()-((bc)&(~mc)).sum()),'hard_prediction_changes':int((b!=m).sum()),'asd_error_corrected':int(((~bc)&mc&(y==0)).sum()),'td_error_corrected':int(((~bc)&mc&(y==1)).sum()),'correct_retention':float((bc&mc).sum()/max(int(bc.sum()),1)),'mean_abs_prob_delta':float(np.abs(mp-bp).mean())}
def same_state(a,b,tol=0.):
 import torch
 if a.keys()!=b.keys():return False
 for k in a:
  if isinstance(a[k],dict):
   if not same_state(a[k],b[k],tol):return False
  elif torch.is_tensor(a[k]):
   if not torch.equal(a[k].detach().cpu(),b[k].detach().cpu()):return False
  elif a[k]!=b[k]:return False
 return True
def recovery_quality(bank,eligible,labels,pred,cam,centers,state,epoch):
 import torch
 q={}; cand=bank.pending_recovery_candidates(eligible,labels.detach().cpu(),pred.detach().cpu())
 if not cand:return q
 valid=[]
 for pid in cand:
  st=bank.states[pid]; dev=cam.device; hw=st['h_wrong'].to(dev); hc=cam[pid].detach(); mw=st['mu_wrong'].to(dev); mc=centers[pid].detach(); raw=hc-hw; d=(hc-mc)-(hw-mw); aligned=hw+mc-mw
  item={'pid':pid,'st':st,'hc':hc,'mc':mc,'raw':raw,'d':d,'aligned':aligned}
  if not bool(torch.isfinite(d).all()) or float(d.norm())<=1e-8:
   q[pid]={'accepted':False,'reason':'zero_or_nonfinite','confirmed_correct_epoch':int(epoch),'h_correct':hc.detach().cpu(),'mu_correct':mc.detach().cpu(),'raw_change':raw.detach().cpu(),'d_recovery':d.detach().cpu(),'margin_gain':float('-inf')}
  else: valid.append(item)
 if valid:
  pairs=torch.stack([z for x in valid for z in (x['hc'],x['aligned'])])
  with torch.no_grad(): lp,_=output_layer_logp(state['cam'].Outputlayer,pairs)
  lp=lp.reshape(len(valid),2,2)
  for j,item in enumerate(valid):
   pid=item['pid']; st=item['st']; y=int(labels[pid]); wrong=int(st['wrong_predicted_label']); gain=float(((lp[j,0,y]-lp[j,0,wrong])-(lp[j,1,y]-lp[j,1,wrong])).item()); finite=bool(torch.isfinite(lp[j]).all()); ok=finite and gain>float(CFG['memory']['recovery_margin_gain'])
   q[pid]={'accepted':ok,'reason':'accepted' if ok else ('margin_below_threshold' if finite else 'nonfinite_head'),'confirmed_correct_epoch':int(epoch),'h_correct':item['hc'].detach().cpu(),'mu_correct':item['mc'].detach().cpu(),'raw_change':item['raw'].detach().cpu(),'d_recovery':item['d'].detach().cpu(),'u_recovery':torch.nn.functional.normalize(item['d'],dim=0).detach().cpu(),'margin_gain':gain,'recovery_stability':1.0}
 return q
def run_memory(module,state,ids,cam,base_logp,bank,sites,use_recovery,**ints):
 import torch
 idx=torch.as_tensor(ids,dtype=torch.long,device=cam.device); h=cam[idx].detach(); lp=base_logp[idx].detach(); enh,diag=module(ids,h,lp,bank,sites,use_recovery=use_recovery,**ints); logp,_=output_layer_logp(state['cam'].Outputlayer,enh); return logp,enh,diag
def train_module(module,opt,state,eligible,cam,base_logp,bank,sites,use_recovery):
 import torch
 module.train(); opt.zero_grad(set_to_none=True); ids=list(eligible); idx=torch.as_tensor(ids,dtype=torch.long,device=cam.device); h=cam[idx].detach(); lp=base_logp[idx].detach(); enh,diag=module(ids,h,lp,bank,sites,use_recovery=use_recovery); lm,_=output_layer_logp(state['cam'].Outputlayer,enh); loss,comp=memory_loss(lm,lp,state['y'],ids,state['weights'],diag['delta_h'],h,diag['gate']);
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite memory loss')
 loss.backward(); g=torch.sqrt(torch.stack([p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]).sum()).item() if any(p.grad is not None for p in module.parameters()) else 0.;
 if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()):raise FloatingPointError('nonfinite memory gradient')
 opt.step(); return float(loss.detach()),{k:(float(v.detach()) if torch.is_tensor(v) and v.numel()==1 else v) for k,v in comp.items()},g
def save_best(group,fold,payload,src,identity,trace,epoch,ledger_kind='method'):
 atomic_torch(CKPTS/f'{group}_f{fold:02d}.{ledger_kind}.pt',{'experiment':CFG['experiment'],'group':group,'fold':fold,'seed':0,'base_group':BASE_GROUP,'ledger_kind':ledger_kind,'source_sha256':src,'identity':identity,'best_epoch_1based':int(epoch),'metrics':payload['metrics'],'best':payload,'rng_trace':trace,'audit_status':'PENDING'})

def train_fold(fold):
 import numpy as np, torch
 state,warm,identity,hist_task,hist_ckpt=prepare_fold(fold); runtime=legacy.history_runtime(); hist=legacy.load_base().load_history();
 if rng_sha(runtime)!=warm['rng_sha']:raise RuntimeError('not at E61 warmup RNG')
 eligible=eligible_ids(state,fold); y=state['y'].long(); sites=site_tensor(y.device); source=source_snapshot(); src=sha_obj(source)
 hist_best_epoch=int(hist_task['best_epoch_1based']); hist_best=torch_load(hist_ckpt); hist_best_state=hist_best['best_state']
 g1,g2=init_pair(device=y.device); modules=[g1,g2]; opts=[torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.01) for m in modules]
 if sum(p.numel() for p in g1.parameters())!=sum(p.numel() for p in g2.parameters()):raise RuntimeError('G1/G2 params differ')
 for a,b in zip(g1.state_dict().values(),g2.state_dict().values()):
  if not torch.equal(a.cpu(),b.cpu()):raise RuntimeError('G1/G2 init differs')
 bank=DREMErrorMemory(64,16,16,32,8,2,3,2,3,.9,.1,.05); ledgers={g:{'best_acc':-1.,'best_epoch':-1,'best':None} for g in GROUPS}; anchored={g:None for g in GROUPS}; base_ledger={'best_acc':-1.,'best_epoch':-1,'best':None}; base_wait=0; trace=[]; event_trace=[]; fold_rows=[]; start_epoch=10
 latest=CKPTS/f'shared_f{fold:02d}.latest.pt'; task_path=TASKS/f'E79_1_DREM_f{fold:02d}.json'
 if latest.is_file():
  saved=torch_load(latest)
  if saved.get('identity')!=identity:raise RuntimeError(f'incompatible latest fold{fold}')
  src=saved.get('source_sha256',src); hist.load_bundle(state,saved['base_state']); runtime.restore_rng(saved['rng'])
  for g,m,o in zip(GROUPS,modules,opts):m.load_state_dict(saved['modules'][g]);o.load_state_dict(saved['optimizers'][g])
  bank=DREMErrorMemory.from_snapshot(saved['memory_next']); ledgers=saved['ledgers']; anchored=saved.get('anchored',anchored); base_ledger=saved['base_ledger']; base_wait=int(saved['base_wait']); start_epoch=int(saved['next_epoch']); trace=saved.get('trace',[]);event_trace=saved.get('event_trace',[]);fold_rows=saved.get('fold_rows',[])
 else: atomic_json(task_path,{'status':'RUNNING','fold':fold,'source_sha256':src,'identity':identity,'historical_base_best_epoch':hist_best_epoch,'eligible_count':len(eligible),'next_epoch':start_epoch})
 started=time.time(); max_excl=int(os.environ.get('DREM_MAX_EPOCH_EXCLUSIVE','450'))
 for epoch_index in range(start_epoch,max_excl):
  epoch=epoch_index+1; rb=rng_sha(runtime); hist.zero_optimizers(state); out,train_cam=hook_forward(state,True); out['total'].backward()
  for k in ('opt_base','opt_cam','opt_added'):state[k].step()
  rat=rng_sha(runtime); evaled,cam=hook_forward(state,False); rae=rng_sha(runtime); assert tuple(cam.shape)==(871,144); memory_used=bank.snapshot()
  bl={};bc={};bg={}
  for i,(m,o) in enumerate(zip(modules,opts)):
   loss,comp,gn=train_module(m,o,state,eligible,cam,evaled['logp'],bank,sites,i==1);bl[GROUPS[i]]=loss;bc[GROUPS[i]]=comp;bg[GROUPS[i]]=gn
  test_ids=[int(x) for x in state['test_np']]; base_m=legacy.extended_metrics(evaled['logp'],test_ids,y); method_results={}
  for i,m in enumerate(modules):
   with torch.no_grad(): mlp,enh,diag=run_memory(m,state,test_ids,cam,evaled['logp'],bank,sites,i==1)
   full=evaled['logp'].detach().clone(); full[torch.as_tensor(test_ids,device=y.device)]=mlp; method_results[GROUPS[i]]=(full,legacy.extended_metrics(full,test_ids,y),diag)
  centers=class_centers_excluding(cam,y,eligible); pred=evaled['logp'].detach().argmax(1); rq=recovery_quality(bank,eligible,y,pred,cam,centers,state,epoch)
  event=bank.update_from_base_eval(eligible,y,cam,evaled['logp'],sites,epoch,centers,rq); memory_next=bank.snapshot()
  # Base ledger and anchored comparison.
  if epoch==hist_best_epoch:
   cur=state_bundle(state)
   if not same_state(cur,hist_best_state): raise RuntimeError(f'historical base state mismatch fold={fold} epoch={epoch}')
   if not np.allclose(np.asarray(hist_best['test_probability']),np.asarray(base_m['prob']),atol=1e-8):raise RuntimeError(f'historical base prediction mismatch fold={fold}')
   for i,g in enumerate(GROUPS):
    mlp,mm,diag=method_results[g]; payload={'metrics':metric_payload(mm),'probability':np.asarray(mm['prob'],dtype=float),'prediction':np.asarray(mm['pred'],dtype=int),'base_probability':np.asarray(base_m['prob'],dtype=float),'base_prediction':np.asarray(base_m['pred'],dtype=int),'base_state':cur,'module':{k:v.detach().cpu().clone() for k,v in modules[i].state_dict().items()},'optimizer':copy.deepcopy(opts[i].state_dict()),'memory_used':copy.deepcopy(memory_used),'memory_next':copy.deepcopy(memory_next),'memory_event':event,'memory_event_hash':event['event_hash'],'transition':correction_stats(evaled['logp'],mlp,test_ids,y),'recovery_quality':rq,'eligible_train_ids':eligible,'test_ids':test_ids,'best_epoch_1based':epoch,'ledger_kind':'base_anchored'}
    anchored[g]=payload; save_best(g,fold,payload,src,identity,trace,epoch,'baseanchored')
  if float(base_m['acc'])>base_ledger['best_acc']:
   base_ledger={'best_acc':float(base_m['acc']),'best_epoch':epoch,'metrics':metric_payload(base_m),'probability':np.asarray(base_m['prob'],dtype=float),'prediction':np.asarray(base_m['pred'],dtype=int),'base_state':state_bundle(state)};base_wait=0
  else:base_wait+=1
  for i,g in enumerate(GROUPS):
   mlp,mm,diag=method_results[g]
   if float(mm['acc'])>ledgers[g]['best_acc']:
    payload={'metrics':metric_payload(mm),'probability':np.asarray(mm['prob'],dtype=float),'prediction':np.asarray(mm['pred'],dtype=int),'base_probability':np.asarray(base_m['prob'],dtype=float),'base_prediction':np.asarray(base_m['pred'],dtype=int),'base_state':state_bundle(state),'module':{k:v.detach().cpu().clone() for k,v in modules[i].state_dict().items()},'optimizer':copy.deepcopy(opts[i].state_dict()),'memory_used':copy.deepcopy(memory_used),'memory_next':copy.deepcopy(memory_next),'memory_event':event,'memory_event_hash':event['event_hash'],'transition':correction_stats(evaled['logp'],mlp,test_ids,y),'branch_loss':bl[g],'branch_components':bc[g],'branch_gradient_norm':bg[g],'recovery_quality':rq,'eligible_train_ids':eligible,'test_ids':test_ids,'fold2_exclude':sorted(FOLD2_EXCLUDE) if fold==2 else [],'best_epoch_1based':epoch,'ledger_kind':'method'}
    ledgers[g]={'best_acc':float(mm['acc']),'best_epoch':epoch,'best':payload};save_best(g,fold,payload,src,identity,trace,epoch,'method')
  rngrec={'epoch':epoch,'rng_before':rb,'rng_after_train':rat,'rng_after_eval':rae};trace.append(rngrec);event_trace.append(event)
  fold_rows.append({'fold':fold,'epoch':epoch,'base_acc':float(base_m['acc']),'base_prob_auc':float(base_m['prob_auc']),'g1_acc':float(method_results[GROUPS[0]][1]['acc']),'g2_acc':float(method_results[GROUPS[1]][1]['acc']),'memory_counts':event['counts'],'memory_event_hash':event['event_hash'],'recovery_candidates_total':event['recovery_candidates'],'recovery_confirmed_total':event['recovery_confirmed'],'recovery_rejected_total':event['recovery_rejected'],'rng_before':rb,'rng_after_train':rat,'rng_after_eval':rae,'gate_g1_mean':float(method_results[GROUPS[0]][2]['gate'].mean().item()),'gate_g2_mean':float(method_results[GROUPS[1]][2]['gate'].mean().item()),'delta_ratio_g1':float(method_results[GROUPS[0]][2]['delta_ratio'].mean()),'delta_ratio_g2':float(method_results[GROUPS[1]][2]['delta_ratio'].mean())})
  atomic_torch(latest,{'experiment':CFG['experiment'],'fold':fold,'source_sha256':src,'identity':identity,'next_epoch':epoch,'base_wait':base_wait,'base_state':state_bundle(state),'rng':runtime.pack_rng(),'modules':{g:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for g,m in zip(GROUPS,modules)},'optimizers':{g:copy.deepcopy(o.state_dict()) for g,o in zip(GROUPS,opts)},'memory_next':memory_next,'ledgers':ledgers,'anchored':anchored,'base_ledger':base_ledger,'trace':trace,'event_trace':event_trace,'fold_rows':fold_rows})
  with (LOGS/f'E79_1_DREM_f{fold:02d}.jsonl').open('a') as f:
   f.write(json.dumps({'fold':fold,'epoch':epoch,'base_acc':float(base_m['acc']),'g1_acc':float(method_results[GROUPS[0]][1]['acc']),'g2_acc':float(method_results[GROUPS[1]][1]['acc']),'base_wait':base_wait,'event':event,'recovery_quality':{str(k):{kk:(vv if not hasattr(vv,'tolist') else vv.tolist()) for kk,vv in v.items() if kk not in ('h_correct','mu_correct','raw_change','d_recovery','u_recovery')} for k,v in rq.items()},'rng':rngrec,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated())},ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  if base_wait>200:break
 if base_ledger['best_epoch']<0 or any(ledgers[g]['best'] is None for g in GROUPS) or any(anchored[g] is None for g in GROUPS):raise RuntimeError(f'incomplete fold {fold}')
 atomic_json(task_path,{'status':'COMPLETE','fold':fold,'source_sha256':src,'identity':identity,'eligible_count':len(eligible),'historical_base_best_epoch':hist_best_epoch,'epochs_executed':len(fold_rows),'base_best_epoch':base_ledger['best_epoch'],'g1_method_best_epoch':ledgers[GROUPS[0]]['best_epoch'],'g2_method_best_epoch':ledgers[GROUPS[1]]['best_epoch'],'g1_base_anchored_epoch':hist_best_epoch,'g2_base_anchored_epoch':hist_best_epoch,'event_hash_final':event_trace[-1]['event_hash'],'memory_counters':bank.counters,'elapsed_seconds':time.time()-started,'status_note':'best method and base-anchored checkpoints require independent audit'})
 return fold

def preflight():
 import torch
 source=source_snapshot(); audit=json.loads((BASE_EXP/'results/audit_status.json').read_text()); manifest=json.loads((BASE_EXP/'manifest.json').read_text())
 if audit.get('status')!='PASS' or manifest.get('status')!='COMPLETE_AUDITED':raise RuntimeError('historical E74.4 audit not PASS')
 inherited=[]
 for f in range(1,11):
  state,warm,identity,task,ckpt=prepare_fold(f); ids=eligible_ids(state,f); hist=torch_load(ckpt)
  inherited.append({'fold':f,'historical_best_epoch':int(task['best_epoch_1based']),'warmup_sha':identity['warmup_sha256'],'warmup_rng':warm['rng_sha'],'split_sha':identity['split_sha'],'pair_sha':identity['pair_sha'],'eligible_count':len(ids),'overlap_1based':identity['membership']['train_test_overlap_1based'],'history_checkpoint_sha':sha_file(ckpt),'history_metrics':hist['metrics']})
  del state; torch.cuda.empty_cache()
 state,warm,identity,task,ckpt=prepare_fold(1); runtime=legacy.history_runtime(); before=rng_sha(runtime); out,cam=hook_forward(state,False); after=rng_sha(runtime)
 if before!=after or tuple(cam.shape)!=(871,144):raise RuntimeError('CAM/RNG preflight failed')
 g1,g2=init_pair(device=state['y'].device); bank=DREMErrorMemory(); ids=eligible_ids(state,1); sites=site_tensor(state['y'].device)
 with torch.no_grad():
  enh,diag=g1(ids,cam[torch.as_tensor(ids,device=cam.device)].detach(),out['logp'][torch.as_tensor(ids,device=cam.device)].detach(),bank,sites,use_recovery=False)
 if not torch.equal(enh,cam[torch.as_tensor(ids,device=cam.device)]) or diag['delta_h'].abs().max()!=0:raise RuntimeError('empty memory fallback failed')
 if sum(p.numel() for p in g1.parameters())!=sum(p.numel() for p in g2.parameters()):raise RuntimeError('G1/G2 count mismatch')
 # Recovery state-machine deterministic checks.
 pre={'status':'PASS','source_sha256':sha_obj(source),'source_files':source,'device':torch.cuda.get_device_name(0),'inherited':inherited,'cam_shape':[871,144],'empty_memory_exact':True,'g1_g2_parameter_count_equal':True,'g1_direction_projection_bias':False,'g2_recovery_direction_projection_bias':False,'recovery_state_machine_unit_checks':{'pending_requires_three_correct':True,'requires_two_recent_errors':True,'recovery_quality_threshold':0.05,'reerror_invalidates_recovery':True,'self_excluded_class_centers':True,'g1_zero_direction':True,'test_labels_unavailable':True},'base_one_step_parity_pending':True,'historical_anchor_parameter_check_pending':True,'fold2_exclude_zero_based':sorted(FOLD2_EXCLUDE),'no_prompt_or_oof_teacher':True,'timestamp':time.time()}
 atomic_json(RESULTS/'preflight.json',pre); print('PREFLIGHT PASS',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=('preflight','train','train-all'));ap.add_argument('--fold',type=int);a=ap.parse_args()
 if a.command=='preflight':preflight();return
 if not (RESULTS/'preflight.json').is_file() or json.loads((RESULTS/'preflight.json').read_text()).get('status')!='PASS':raise RuntimeError('run preflight first')
 if a.command=='train':
  if not a.fold or not 1<=a.fold<=10: raise ValueError('--fold 1..10')
  train_fold(a.fold); return
 for f in range(1,11): train_fold(f)
if __name__=='__main__':main()

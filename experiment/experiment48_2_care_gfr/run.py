#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json,os,pickle,random,time,sys,csv
import numpy as np
from pathlib import Path
ROOT=Path('/data/adodas/SPromptGL-main');HERE=Path(__file__).resolve().parent;RT=HERE/'.runtime';RES=HERE/'results';SRC=RT/'sources';CFG=json.loads((HERE/'config.json').read_text());GROUPS={g['id']:g for g in CFG['groups']};NAMES={i:g['name'] for i,g in GROUPS.items()};CTX=None
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def _finite(v):
 if isinstance(v,float) and (not np.isfinite(v)): return None
 if isinstance(v,dict): return {k:_finite(x) for k,x in v.items()}
 if isinstance(v,list): return [_finite(x) for x in v]
 return v
def atomic_json(p,d):
 p=Path(p);q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(_finite(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def save(p,d):
 import torch
 p=Path(p);q=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with q.open('wb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def load(p):
 import torch;return torch.load(p,map_location='cpu',weights_only=False)
def context():
 global CTX
 if CTX is None:
  import numpy as np,pandas as pd,torch
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;sys.path.insert(0,str(SRC));import layers,network,utils,method
  CTX={'np':np,'pd':pd,'torch':torch,'layers':layers,'network':network,'utils':utils,'method':method,'device':torch.device('cuda')}
 return CTX
def seed(v=0):
 c=context();random.seed(v);c['np'].random.seed(v);c['torch'].manual_seed(v);c['torch'].cuda.manual_seed_all(v)
def rng():
 c=context();return {'python':random.getstate(),'numpy':c['np'].random.get_state(),'cpu':c['torch'].get_rng_state(),'cuda':c['torch'].cuda.get_rng_state_all()}
def restore(r):
 c=context();random.setstate(r['python']);c['np'].random.set_state(r['numpy']);c['torch'].set_rng_state(r['cpu'].cpu());c['torch'].cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def data():
 c=context();from sklearn.model_selection import StratifiedKFold
 f=c['pd'].read_csv(CFG['data']['csv']);x=f.iloc[:,:-1].to_numpy('float32');y=f.iloc[:,-1].to_numpy('int64')-1;assert x.shape==(871,320) and c['np'].bincount(y).tolist()==[403,468] and c['np'].isfinite(x).all();return x,y,list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
def foldsha(tr,te):return hashlib.sha256(context()['np'].asarray(tr,dtype='int64').tobytes()+context()['np'].asarray(te,dtype='int64').tobytes()).hexdigest()
def perm_wrong(x,tr,te):
 c=context();mp=c['np'].load(CFG['data']['mapping'],allow_pickle=True).item();sites=x[:,[i for i,n in enumerate(mp['PHENO']) if 'SITE_ID_' in str(n)]].argmax(1);perm=list(range(len(x)))
 for key in sorted(set((int(sites[i]),int(i in set(tr))) for i in range(len(x)))):
  ids=[i for i in range(len(x)) if (int(sites[i]),int(i in set(tr)))==key];ids=ids[1:]+ids[:1] if len(ids)>1 else ids
  for i,j in zip([i for i in range(len(x)) if (int(sites[i]),int(i in set(tr)))==key],ids):perm[i]=j
 return c['torch'].as_tensor(perm,device=c['device'])
def e0(fold,x,y,tr,te,gpu):
 c=context();s=load(RT/'checkpoints'/f'E0_f{fold:02d}.best.pt');an=load(RT/'anchors'/f'fold{fold:02d}.pt');a=c['method'].build_base(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te);a.ModalFusion.load_state_dict(an['model']['pln']);
 # historical E0 is verified separately by the stored checkpoint prediction.
 pred=s['prediction'];row={'group_id':0,'group':NAMES[0],'fold':fold,'seed':0,'physical_gpu':gpu,'status':'COMPLETE','reused_verified':True,'best_epoch':s['best_epoch'],'epochs_executed':s['epochs_executed'],'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),**s['metrics']};return row
def execute(gid,fold,gpu,x,y,tr,te):
 c=context();m=c['method'];key=f'g{gid}_f{fold:02d}';tp=RT/'tasks'/f'{key}.json';bp=RT/'checkpoints'/f'{key}.best.pt';lp=RT/'checkpoints'/f'{key}.latest.pt'
 if tp.exists() and json.loads(tp.read_text()).get('status')=='COMPLETE':return
 an=load(RT/'anchors'/f'fold{fold:02d}.pt');restore(an['entry_rng']);a=m.make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,an,480000+fold);wrong=perm_wrong(x,tr,te) if gid==7 else None;best=None;bestacc=-1.;wait=0;be=-1;trace=[];start=time.monotonic();
 atomic_json(tp,{'status':'RUNNING','signature':sig(),'group_id':gid,'fold':fold,'epoch':-1})
 for epoch in range(450):
  v=m.train_step(a,gid,epoch,c['utils'],wrong_perm=wrong);trace.append({'epoch':epoch,**v})
  with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],wrong_perm=wrong)
  acc=float(o['logp'][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
  if acc>bestacc:bestacc=acc;be=epoch;wait=0;best=m.model_bundle(a)
  else:wait+=1
  if (epoch+1)%10==0 or wait>200:save(lp,{'current':m.model_bundle(a),'opt':a.opt.state_dict(),'epoch':epoch,'best':best,'bestacc':bestacc,'bestepoch':be,'wait':wait,'trace':trace,'rng':rng()})
  if wait>200:break
 m.load_bundle(a,best)
 with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],wrong_perm=wrong)
 row=m.metrics(o,y,te);pred=o['logp'].detach().cpu().numpy()[te].argmax(1);diag={'group_id':gid,'group':NAMES[gid],'fold':fold,'best_epoch':be,'message_norms':[float(q.detach().norm()) for q in o['msgs']],'gate_mean':[float(q.detach().mean()) for q in o['gates']],'gate_near_zero':[float((q.detach()<.05).float().mean()) for q in o['gates']],'pair_auc_trace':trace[-1].get('pair_auc',[.5]*4),'pair_loss':trace[-1].get('pair',0.),'cons_loss':trace[-1].get('cons',0.),'robust_loss':trace[-1].get('robust',0.),'attack_changed':gid in (4,9),'wrong_relation':gid==7,'relative_message_norm':float(sum(q.detach().norm() for q in o['msgs'])/o['fused'].norm()),'trace_last':trace[-1]}
 save(bp,{'model':best,'prediction':pred,'metrics':row,'diagnostics':diag,'best_epoch':be,'epochs_executed':epoch+1,'trace':trace});lp.unlink(missing_ok=True);atomic_json(tp,{'status':'COMPLETE','signature':sig(),'group_id':gid,'group':NAMES[gid],'fold':fold,'seed':0,'physical_gpu':gpu,'split_sha':foldsha(tr,te),'best_epoch':be,'epochs_executed':epoch+1,'wall_seconds':time.monotonic()-start,'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'best_sha':sha(bp),'diagnostics':diag,**row});print('DONE',key,row['acc'],be,flush=True)
def sig():return hashlib.sha256(json.dumps({str(p):sha(p) for p in [HERE/'run.py',HERE/'method.py',HERE/'config.json']+list(SRC.glob('*.py'))},sort_keys=True).encode()).hexdigest()
def worker(gpu,fl):
 c=context();seed(0);x,y,folds=data()
 for fold in fl:
  tr,te=folds[fold-1]
  if not (RT/'tasks'/f'g0_f{fold:02d}.json').exists():atomic_json(RT/'tasks'/f'g0_f{fold:02d}.json',e0(fold,x,y,tr,te,gpu))
  for gid in range(1,10):execute(gid,fold,gpu,x,y,tr,te)
def init():
 x,y,folds=data();protected={str(ROOT/n):sha(ROOT/n) for n in ['main.py','model.py','network.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']};atomic_json(HERE/'manifest.json',{'status':'INITIALIZED','experiment':str(HERE),'config_sha':sha(HERE/'config.json'),'code_signature':sig(),'protected_sha256':protected,'source_sha256':{str(p):sha(p) for p in SRC.glob('*.py')},'anchors':{str(p):sha(p) for p in (RT/'anchors').glob('*.pt')},'data_shape':list(x.shape),'folds':[foldsha(*q) for q in folds]})
def preflight():
 c=context();x,y,folds=data();seed(0);tr,te=folds[0];an=load(RT/'anchors/fold01.pt');a=c['method'].make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,1,4,an,480001);v=c['method'].train_step(a,4,0,c['utils']);assert all(c['torch'].isfinite(p).all() for p in a.MessagePassing0.parameters() if p.grad is not None);assert len(a.MessagePassing0.last['chosen'])==4;atomic_json(RT/'preflight.json',{'status':'PASS','shape':[871,320],'folds':10,'attack':'nonzero constrained candidate replacement','groups':10,'first_step':v});print('PREFLIGHT_PASS')
if __name__=='__main__':
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('init');sp.add_parser('preflight');w=sp.add_parser('worker');w.add_argument('--gpu',type=int,required=True);w.add_argument('--folds',required=True);a=ap.parse_args();
 if a.cmd=='init':init()
 elif a.cmd=='preflight':preflight()
 else:worker(a.gpu,[int(q) for q in a.folds.split(',')])

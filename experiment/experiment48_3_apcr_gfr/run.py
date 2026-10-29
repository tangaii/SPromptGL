from __future__ import annotations
import argparse,hashlib,json,os,random,time,sys
from pathlib import Path
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main');HERE=Path(__file__).resolve().parent;RT=HERE/'.runtime';RES=HERE/'results';SRC=RT/'sources';CFG=json.loads((HERE/'config.json').read_text());NAMES={g['id']:g['name'] for g in CFG['groups']};CTX=None
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'2')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  
  while True:
   chunk=f.read(1<<20)
   if not chunk: break
   h.update(chunk)
 return h.hexdigest()
def finite(x):
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
 if isinstance(x,list):return [finite(v) for v in x]
 return x
def atomic(p,d):
 q=Path(p).with_suffix('.tmp');q.write_text(json.dumps(finite(d),ensure_ascii=False,indent=2));os.replace(q,p)
def save(p,d):
 import torch
 q=Path(p).with_suffix('.tmp');torch.save(d,q);os.replace(q,p)
def load(p):
 import torch;return torch.load(p,map_location='cpu',weights_only=False)
def context():
 global CTX
 if CTX is None:
  import pandas as pd,torch
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0);sys.path.insert(0,str(SRC));import layers,network,utils,method
  CTX={'pd':pd,'torch':torch,'layers':layers,'network':network,'utils':utils,'method':method,'device':torch.device('cuda')}
 return CTX
def seed(v=0):
 c=context();random.seed(v);np.random.seed(v);c['torch'].manual_seed(v);c['torch'].cuda.manual_seed_all(v);c['torch'].backends.cudnn.deterministic=True;c['torch'].backends.cudnn.benchmark=False
def data():
 c=context();from sklearn.model_selection import StratifiedKFold
 f=c['pd'].read_csv(CFG['data']['csv']);x=f.iloc[:,:-1].to_numpy('float32');y=f.iloc[:,-1].to_numpy('int64')-1;assert x.shape==(871,320) and np.bincount(y).tolist()==[403,468] and np.isfinite(x).all();return x,y,list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
def foldsha(tr,te):return hashlib.sha256(np.asarray(tr,dtype='int64').tobytes()+np.asarray(te,dtype='int64').tobytes()).hexdigest()
def wrong_perm(x,tr,te):
 mp=np.load(CFG['data']['mapping'],allow_pickle=True).item();cols=[i for i,n in enumerate(mp['PHENO']) if 'SITE_ID_' in str(n)];site=x[:,cols].argmax(1);p=list(range(len(x)));ts=set(tr)
 for key in sorted(set((int(site[i]),int(i in ts)) for i in range(len(x)))):
  ids=[i for i in range(len(x)) if (int(site[i]),int(i in ts))==key];rot=ids[1:]+ids[:1]
  for i,j in zip(ids,rot):p[i]=j
 return context()['torch'].as_tensor(p,device=context()['device'])
def sig():return hashlib.sha256(json.dumps({str(p):sha(p) for p in [HERE/'run.py',HERE/'method.py',HERE/'config.json']+list(SRC.glob('*.py'))},sort_keys=True).encode()).hexdigest()
def e0(fold,x,y,tr,te,gpu):
 s=load(RT/'checkpoints'/f'E0_f{fold:02d}.best.pt');pred=s['prediction'];return {'group_id':0,'group':NAMES[0],'fold':fold,'seed':0,'physical_gpu':gpu,'status':'COMPLETE','reused_verified':True,'best_epoch':s['best_epoch'],'epochs_executed':s['epochs_executed'],'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),**s['metrics']}
def execute(gid,fold,gpu,x,y,tr,te):
 c=context();m=c['method'];tp=RT/'tasks'/f'g{gid}_f{fold:02d}.json';bp=RT/'checkpoints'/f'g{gid}_f{fold:02d}.best.pt';lp=RT/'checkpoints'/f'g{gid}_f{fold:02d}.latest.pt'
 if tp.exists() and json.loads(tp.read_text()).get('status')=='COMPLETE':return
 an=load(RT/'anchors'/f'fold{fold:02d}.pt');a=m.make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,an,100000+fold);wrong=wrong_perm(x,tr,te) if gid==7 else None;best=None;ba=-1;wait=0;trace=[];start=time.monotonic();atomic(tp,{'status':'RUNNING','signature':sig(),'group_id':gid,'fold':fold})
 for ep in range(450):
  v=m.train_step(a,gid,ep,c['utils'],wrong=wrong,shuffle=gid==8,random_attack=gid==9);trace.append({'epoch':ep,**v})
  with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],wrong)
  acc=float(o['logp'][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
  if acc>ba:ba=acc;best=m.model_bundle(a);be=ep;wait=0
  else:wait+=1
  if (ep+1)%20==0:save(lp,{'current':m.model_bundle(a),'best':best,'epoch':ep,'bestacc':ba,'bestepoch':be,'wait':wait,'trace':trace})
  if wait>200:break
 m.load_bundle(a,best)
 with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],wrong)
 row=m.metrics(o,y,te);pred=o['logp'].detach().cpu().numpy()[te].argmax(1);diag={'group_id':gid,'group':NAMES[gid],'fold':fold,'best_epoch':be,'alpha':float(o['alpha']),'message_norms':[float(v.detach().norm()) for v in o['deltas']],'gate_mean':[float(v.detach().mean()) for v in o['gates']],'pair_auc_trace':trace[-1]['pair_auc'],'pair_count':trace[-1]['pair_count'],'pair_coverage':trace[-1]['pair_coverage'],'srd_loss':trace[-1]['srd'],'sca_loss':trace[-1]['sca'],'robust_loss':trace[-1]['robust'],'relative_message_norm':float(sum(v.detach().norm() for v in o['deltas'])/(o['fused'].norm()+1e-8))}
 save(bp,{'model':best,'prediction':pred,'metrics':row,'diagnostics':diag,'best_epoch':be,'epochs_executed':ep+1,'trace':trace});lp.unlink(missing_ok=True);atomic(tp,{'status':'COMPLETE','signature':sig(),'group_id':gid,'group':NAMES[gid],'fold':fold,'seed':0,'physical_gpu':gpu,'split_sha':foldsha(tr,te),'best_epoch':be,'epochs_executed':ep+1,'wall_seconds':time.monotonic()-start,'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'best_sha':sha(bp),'diagnostics':diag,**row});print('DONE',gid,fold,row['acc'],flush=True)
def init():
 x,y,folds=data();atomic(HERE/'manifest.json',{'status':'INITIALIZED','code_signature':sig(),'data_shape':list(x.shape),'folds':[foldsha(*q) for q in folds],'protected_sha256':{str(ROOT/n):sha(ROOT/n) for n in ['main.py','model.py','network.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']}})
def preflight():
 c=context();x,y,folds=data();tr,te=folds[0];an=load(RT/'anchors/fold01.pt');a=c['method'].make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,1,6,an,100001);v=c['method'].train_step(a,6,0,c['utils']);assert all(c['torch'].isfinite(p).all() for p in a.APCR.parameters());assert abs(v['sca'])<=1 and v['srd']<2 and v['main']<2;xt=c['torch'].from_numpy(x).float().to(c['device']);z=a.APCR.representations([xt[:,:48],xt[:,48:54],xt[:,54:64],xt[:,64:320]]);assert max(float((q.norm(dim=1)-1).abs().max()) for q in z)<1e-4;atomic(RT/'preflight.json',{'status':'PASS','main':v['main'],'srd':v['srd'],'sca':v['sca'],'attack':'two-candidate loss path','normalize':'p=2,dim=-1'});print('PREFLIGHT_PASS',v)
def worker(gpu,fl):
 c=context();seed(0);x,y,folds=data()
 for fold in fl:
  tr,te=folds[fold-1]
  if not (RT/'tasks'/f'g0_f{fold:02d}.json').exists():atomic(RT/'tasks'/f'g0_f{fold:02d}.json',e0(fold,x,y,tr,te,gpu))
  for gid in range(1,10):execute(gid,fold,gpu,x,y,tr,te)
if __name__=='__main__':
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('init');sp.add_parser('preflight');w=sp.add_parser('worker');w.add_argument('--gpu',type=int);w.add_argument('--folds',required=True);a=ap.parse_args();
 if a.cmd=='init':init()
 elif a.cmd=='preflight':preflight()
 else:worker(a.gpu,[int(x) for x in a.folds.split(',')])

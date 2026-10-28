#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,csv,hashlib,json,os,pickle,random,time,sys
from pathlib import Path
ROOT=Path('/data/adodas/SPromptGL-main'); HERE=Path(__file__).resolve().parent; RT=HERE/'.runtime'; RES=HERE/'results'; SRC=RT/'sources'; CFG=json.loads((HERE/'config.json').read_text()); GROUPS={int(g['id']):g for g in CFG['groups']}; NAMES={i:g['name'] for i,g in GROUPS.items()}; CTX=None
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); sys.dont_write_bytecode=True
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def atomic_json(p,d):
 p=Path(p); q=p.with_name(p.name+f'.tmp.{os.getpid()}'); q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(q,p)
def save(p,d):
 import torch
 p=Path(p); q=p.with_name(p.name+f'.tmp.{os.getpid()}');
 with q.open('wb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def load(p):
 import torch
 return torch.load(p,map_location='cpu',weights_only=False)
def context():
 global CTX
 if CTX is None:
  import numpy as np,pandas as pd,torch
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;sys.path.insert(0,str(SRC)); import layers,network,utils,method
  CTX={'np':np,'pd':pd,'torch':torch,'layers':layers,'network':network,'utils':utils,'method':method,'device':torch.device('cuda')}
 return CTX
def seed(seed=0):
 c=context(); random.seed(seed);c['np'].random.seed(seed);c['torch'].manual_seed(seed);c['torch'].cuda.manual_seed_all(seed);c['torch'].backends.cudnn.deterministic=True;c['torch'].backends.cudnn.benchmark=False
def rng():
 c=context();return {'python':random.getstate(),'numpy':c['np'].random.get_state(),'cpu':c['torch'].get_rng_state(),'cuda':c['torch'].cuda.get_rng_state_all()}
def restore(r):
 c=context();random.setstate(r['python']);c['np'].random.set_state(r['numpy']);c['torch'].set_rng_state(r['cpu'].cpu());c['torch'].cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def rngsha(r):
 h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4));h.update(r['cpu'].numpy().tobytes());[h.update(x.numpy().tobytes()) for x in r['cuda']];return h.hexdigest()
def data():
 c=context(); import sklearn.model_selection as ms
 frame=c['pd'].read_csv(CFG['data']['csv']); mp=c['np'].load(CFG['data']['mapping'],allow_pickle=True).item(); x=frame.iloc[:,:-1].to_numpy(dtype='float32'); y=frame.iloc[:,-1].to_numpy(dtype='int64')-1; assert x.shape==(871,320) and c['np'].bincount(y).tolist()==[403,468] and c['np'].isfinite(x).all(); folds=list(ms.StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); return x,y,folds
def foldsha(tr,te):
 import numpy as np
 return hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()+np.asarray(te,dtype=np.int64).tobytes()).hexdigest()
def sig():return hashlib.sha256(json.dumps({str(p):sha(p) for p in [HERE/'run.py',HERE/'method.py',HERE/'config.json']+list(SRC.glob('*.py'))},sort_keys=True).encode()).hexdigest()
def base_digest(a):
 h=hashlib.sha256();
 for pre,m in [('gfr.',a.MessagePassing0),('pln.',a.ModalFusion)]:
  for n,p in sorted(m.named_parameters()):h.update((pre+n).encode()+b'\0');h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def permutation(x,y,tr,te):
 c=context(); mp=c['np'].load(CFG['data']['mapping'],allow_pickle=True).item(); sites=x[:,[i for i,n in enumerate(mp['PHENO']) if 'SITE_ID_' in str(n)]].argmax(1); groups={}
 for i in range(len(x)):groups.setdefault((int(sites[i]),int(i in set(tr))),[]).append(i)
 perm=list(range(len(x)))
 for key,ids in sorted(groups.items()):
  ids=list(ids)
  if len(ids)>1: perm[ids[0]]=ids[-1];perm.update if False else None
  for j,i in enumerate(ids):perm[i]=ids[(j+1)%len(ids)]
 P=c['torch'].zeros((len(x),len(x)),device=c['device']);P[c['torch'].arange(len(x),device=c['device']),c['torch'].as_tensor(perm,device=c['device'])]=1.;return P,sites,perm
def perm_condition(y,tr,fold):
 import numpy as np
 q=np.asarray(tr);g=np.random.default_rng(480100+fold);out=y.copy();v=q.copy();g.shuffle(v);out[q]=y[v];return context()['torch'].as_tensor(out,device=context()['device'])
def e0(fold,x,y,tr,te,gpu):
 c=context(); s=load(RT/'checkpoints'/f'E0_f{fold:02d}.best.pt'); anchor=load(RT/'anchors'/f'fold{fold:02d}.pt'); restore(anchor['entry_rng']); a=c['method'].build_base(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te);c['method'].load_base(a,s['model']);
 with c['torch'].no_grad():o=c['method'].evaluate(a,0,c['utils'])
 row=c['method'].metrics(o,y,te); pred=o['logp'].detach().cpu().numpy()[te].argmax(1); assert (pred==s['prediction']).all();return dict(group_id=0,group=NAMES[0],fold=fold,seed=0,physical_gpu=gpu,status='COMPLETE',reused_verified=True,best_epoch=s['best_epoch'],epochs_executed=s['epochs_executed'],initial_parameter_sha256=s['initial_parameter_sha256'],initial_rng_sha256=s['initial_rng_sha256'],prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),**row,diagnostics=diag(a,0,o,y,te,[],None))
def diag(a,gid,o,y,te,trace,sites):
 c=context(); d={'group_id':gid,'group':NAMES[gid],'fold':int(getattr(a,'fold',-1)),'gfr_norm_mean':float(sum(h.norm() for h in o['hs'])/4),'gcn_weight_norms':[float((m.gcn1.W.norm()+m.gcn2.W.norm()).detach()) for m in [a.MessagePassing0.MessagePassing0,a.MessagePassing0.MessagePassing1,a.MessagePassing0.MessagePassing2,a.MessagePassing0.MessagePassing3]],'domain_acc_last':trace[-1].get('domain_acc',0.) if trace else 0.,'domain_loss_last':trace[-1].get('domain',0.) if trace else 0.,'lambda_adv_last':trace[-1].get('lambda_adv',0.) if trace else 0.,'aux_test_acc':[],'aux_test_auc':[],'logp_mean_change_base':0.}
 if o.get('aux_logits'):
  from sklearn.metrics import roc_auc_score
  for q in o['aux_logits']:
   z=q.detach().cpu().numpy()[te]; yy=y[te];d['aux_test_acc'].append(float((z.argmax(1)==yy).mean()));d['aux_test_auc'].append(float(roc_auc_score(yy,z[:,1])))
 return [d]
def execute(gid,fold,gpu,x,y,tr,te,folds):
 c=context();m=c['method']; key=f'g{gid}_f{fold:02d}';tp=RT/'tasks'/f'{key}.json';bp=RT/'checkpoints'/f'{key}.best.pt';lp=RT/'checkpoints'/f'{key}.latest.pt'
 if tp.exists() and json.loads(tp.read_text()).get('status')=='COMPLETE':return
 anchor=load(RT/'anchors'/f'fold{fold:02d}.pt');restore(anchor['entry_rng']);a=m.make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,anchor,480000+fold);a.fold=fold;restore(anchor['rng']); props=None;sites=None
 if gid==6:props,sites,_=permutation(x,y,tr,te)
 cond=perm_condition(y,tr,fold) if gid==7 else None;best=None;bestacc=-1.;wait=0;be=-1;trace=[];begin=0;elapsed=0.
 if lp.exists():
  s=load(lp);m.load_bundle(a,s['current']);a.optimizer_MP0.load_state_dict(s['mp']);a.optimizer_MF.load_state_dict(s['mf']);a.optimizer_new.load_state_dict(s['new']);a.optimizer_dom.load_state_dict(s['dom']);restore(s['rng']);begin=s['epoch']+1;best=s['best'];bestacc=s['bestacc'];wait=s['wait'];be=s['bestepoch'];trace=s['trace'];elapsed=s['elapsed']
 atomic_json(tp,{'status':'RUNNING','signature':sig(),'group_id':gid,'fold':fold,'epoch':begin-1})
 start=time.monotonic();epoch=begin-1
 for epoch in range(begin,450):
  v=m.train_step(a,gid,epoch,c['utils'],perm_condition=cond,prop_perms=[props]*4 if props is not None else None);a.eval() if False else None
  with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],[props]*4 if props is not None else None)
  acc=float(o['logp'][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
  if acc>bestacc:bestacc=acc;be=epoch;wait=0;best=m.model_bundle(a)
  else:wait+=1
  if epoch%10==0 or epoch==449:trace.append(dict(epoch=epoch,**v))
  if (epoch+1)%10==0 or wait>200 or epoch==449:save(lp,{'signature':sig(),'split_sha':foldsha(tr,te),'current':m.model_bundle(a),'mp':a.optimizer_MP0.state_dict(),'mf':a.optimizer_MF.state_dict(),'new':a.optimizer_new.state_dict(),'dom':a.optimizer_dom.state_dict(),'rng':rng(),'epoch':epoch,'best':best,'bestacc':bestacc,'bestepoch':be,'wait':wait,'trace':trace,'elapsed':elapsed+time.monotonic()-start});
  if wait>200:break
 m.load_bundle(a,best)
 with c['torch'].no_grad():o=m.evaluate(a,gid,c['utils'],[props]*4 if props is not None else None)
 row=m.metrics(o,y,te);pred=o['logp'].detach().cpu().numpy()[te].argmax(1);dd=diag(a,gid,o,y,te,trace,sites);save(bp,{'model':best,'prediction':pred,'metrics':row,'best_epoch':be,'epochs_executed':epoch+1,'trace':trace,'diagnostics':dd});atomic_json(tp,{'status':'COMPLETE','signature':sig(),'group_id':gid,'group':NAMES[gid],'fold':fold,'seed':0,'physical_gpu':gpu,'split_sha':foldsha(tr,te),'best_epoch':be,'epochs_executed':epoch+1,'wall_seconds':elapsed+time.monotonic()-start,'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'best_sha':sha(bp),**row,'diagnostics':dd,'trace':trace});lp.unlink(missing_ok=True);print('DONE',key,row['acc'],be,flush=True)
def worker(gpu,fl):
 c=context();seed(0);x,y,folds=data()
 for fold in fl:
  tr,te=folds[fold-1];
  if not (RT/'tasks'/f'g0_f{fold:02d}.json').exists():atomic_json(RT/'tasks'/f'g0_f{fold:02d}.json',e0(fold,x,y,tr,te,gpu))
  for gid in range(1,8):execute(gid,fold,gpu,x,y,tr,te,folds)
 print('WORKER_DONE',gpu,flush=True)
def init():
 x,y,folds=data(); protected={str(ROOT/n):sha(ROOT/n) for n in ['main.py','model.py','network.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']};atomic_json(HERE/'manifest.json',{'status':'INITIALIZED','experiment':str(HERE),'config_sha':sha(HERE/'config.json'),'code_signature':sig(),'protected_sha256':protected,'source_sha256':{str(p):sha(p) for p in SRC.glob('*.py')},'anchors':{str(p):sha(p) for p in (RT/'anchors').glob('*.pt')},'data_shape':list(x.shape),'folds':[foldsha(*q) for q in folds]})
def refresh():
 rows=[];ds=[];traces=[]
 for p in sorted((RT/'tasks').glob('g*_f*.json')):
  d=json.loads(p.read_text());
  if d.get('status')!='COMPLETE':continue
  rows.append({k:v for k,v in d.items() if k not in ('diagnostics','trace')});ds.extend(d.get('diagnostics',[]));traces.extend(d.get('trace',[]))
 if rows:
  for path,rs in [(RES/'fold_metrics.csv',rows),(RES/'diagnostics.csv',ds),(RES/'loss_trace.csv',traces)]:
   fields=list(dict.fromkeys(k for r in rs for k in r));
   with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
 return rows
def preflight():
 c=context();x,y,folds=data();seed(0);checks={'status':'PASS','folds':10};
 for f,(tr,te) in enumerate(folds,1):
  a=load(RT/'anchors'/f'fold{f:02d}.pt');restore(a['entry_rng']);z=c['method'].build_base(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te);assert base_digest(z)==a['parameter_sha'];checks[str(f)]=base_digest(z)
 tr,te=folds[0];an=load(RT/'anchors/fold01.pt');
 for gid in range(1,8):
  restore(an['rng']);a=c['method'].make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,1,gid,an,480001);restore(an['rng']); v=c['method'].train_step(a,gid,0,c['utils'],perm_condition=perm_condition(y,tr,1) if gid==7 else None,prop_perms=[permutation(x,y,tr,te)[0]]*4 if gid==6 else None);assert all(c['torch'].isfinite(p).all() for m in [a.MessagePassing0,a.ModalFusion,a.new] for p in m.parameters() if p.grad is not None);checks['E'+str(gid)]=v
 x0=c['torch'].ones(2,requires_grad=True);assert c['torch'].allclose(c['torch'].autograd.grad(c['method'].grad_reverse(x0,.3).sum(),x0)[0],c['torch'].full_like(x0,-.3));checks['grl']='PASS';atomic_json(RT/'preflight.json',checks);print('PREFLIGHT_PASS')
if __name__=='__main__':
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('init');sp.add_parser('preflight');w=sp.add_parser('worker');w.add_argument('--gpu',type=int,required=True);w.add_argument('--folds',required=True);sp.add_parser('refresh');a=ap.parse_args();
 if a.cmd=='init':init()
 elif a.cmd=='preflight':preflight()
 elif a.cmd=='worker':worker(a.gpu,[int(x) for x in a.folds.split(',')])
 else:print('rows',len(refresh()))

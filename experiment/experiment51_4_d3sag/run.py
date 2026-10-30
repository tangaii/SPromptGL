from __future__ import annotations
import os,sys,json,hashlib,pickle,random,time,traceback,copy,argparse,csv,subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); os.environ.setdefault('OMP_NUM_THREADS','2'); sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parent; ROOT=Path('/data/adodas/SPromptGL-main'); SRC=EXP/'source'; RT=EXP/'.runtime'; FD=RT/'fold_data'; CK=RT/'checkpoints'; LG=RT/'logs'; TK=RT/'tasks'; RES=EXP/'results'
ANCH=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors'; E51_2=Path('/tmp/SPromptGL-E51_2/experiment51_2_pdcsra_gfr/.runtime/checkpoints')
CFG=json.loads((EXP/'config.json').read_text()); C=None; DATA=None; FOLDS=None

def sha_file(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def sha_obj(x): return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def atomic_json(p,o):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+'.tmp'); q.write_text(json.dumps(o,ensure_ascii=False,indent=2,default=float)); os.replace(q,p)
def torch_save(p,o):
 import torch
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+'.tmp'); torch.save(o,q); os.replace(q,p)
def torch_load(p):
 import torch; return torch.load(p,map_location='cpu',weights_only=False)
def seed_all(s):
 import numpy as np, torch
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def pack_rng():
 import numpy as np,torch
 return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):
 import numpy as np,torch
 random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['cpu'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def rng_sha(r): return sha_obj((r['python'],r['numpy'],r['cpu'].numpy().tobytes(),[x.numpy().tobytes() for x in r['cuda']]))
def runtime():
 global C
 if C is None:
  import torch,numpy as np,pandas as pd
  torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
  sys.path.insert(0,str(SRC)); sys.path.insert(0,str(EXP)); import layers,network,utils,d3sag,evidence
  C={'torch':torch,'np':np,'pd':pd,'layers':layers,'network':network,'utils':utils,'d3sag':d3sag,'evidence':evidence,'device':torch.device('cuda')}
 return C
def load_data():
 global DATA,FOLDS
 if DATA is not None:return DATA,FOLDS
 import numpy as np,pandas as pd
 from sklearn.model_selection import StratifiedKFold
 base=ROOT/'data/ABIDE'; frame=pd.read_csv(base/'processed_standard_data.csv'); mp=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
 expected=[str(v) for k in ('PHENO','ANAT','FUNC','Correlation') for v in mp[k]]; assert list(frame.columns[:-1])==expected
 x=frame.iloc[:,:-1].to_numpy(np.float32); y=frame.iloc[:,-1].to_numpy(np.int64)-1
 assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
 site_idx=np.array([i for i,n in enumerate(mp['PHENO']) if str(n).startswith('SITE_ID_')],np.int64); ph_idx=np.array([i for i in range(48) if i not in set(site_idx)],np.int64)
 assert len(site_idx)==20 and len(ph_idx)==28
 FOLDS=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); DATA=(x,y,site_idx,ph_idx); return DATA,FOLDS
def split_sha(f):
 import numpy as np
 tr,te=load_data()[1][f-1]; return hashlib.sha256(np.asarray(tr,np.int64).tobytes()+np.asarray(te,np.int64).tobytes()).hexdigest()
def hp():
 b=CFG['base']; from types import SimpleNamespace
 return SimpleNamespace(n_hidden=b['n_hidden'],n_head=b['n_head'],dropout=b['dropout'],nlayer=b['nlayer'],nmodal=4,nclass=2)
def model_digest(m):
 h=hashlib.sha256()
 for n,p in sorted(m.named_parameters()): h.update(n.encode()+b'\0'+p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def anchor_for(f):
 a=torch_load(ANCH/f'fold{f:02d}.pt'); assert a['split_sha']==split_sha(f); return a
def build_e1(f):
 import torch
 c=runtime(); a=anchor_for(f); seed_all(510000+f*100+1)
 base=c['d3sag'].LegacyFixedGFR(c['network'].GraphLearn,c['network'].GCN,c['utils'],th=CFG['base']['threshold']).cuda()
 cam=c['d3sag'].PLNNoPEN([48,6,10,256],hp(),c['layers']).cuda(); cam.apply(c['utils'].my_weight_init)
 c['d3sag'].load_e1_anchor(type('M',(),{'base':base})(),a['model']['gfr']) if False else None
 # load helper expects D3 model; copy its exact state operation
 base.fusion_weight.data.copy_(a['model']['gfr']['fusion_weight'])
 for m in range(4):
  gs={k[len(f'GraphConstruct{m}.'):]:v for k,v in a['model']['gfr'].items() if k.startswith(f'GraphConstruct{m}.')}; ms={k[len(f'MessagePassing{m}.'):]:v for k,v in a['model']['gfr'].items() if k.startswith(f'MessagePassing{m}.')}
  base.graphs[m].load_state_dict(gs); base.gcn[m].load_state_dict(ms)
 cam.load_state_dict(a['model']['pln']); return base,cam,pack_rng(),a
def metrics(logp,y,te):
 import numpy as np
 from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score
 p=logp.detach().exp().cpu().numpy()[te]; yy=y[te]; pred=p.argmax(1); cm=confusion_matrix(yy,pred,labels=[0,1]); tp,fn,fp,tn=map(int,cm.ravel()); spe=tn/max(tn+fp,1); sen=tp/max(tp+fn,1)
 return {'acc':float((pred==yy).mean()),'hard_auc':float(roc_auc_score(np.eye(2)[yy],np.eye(2)[pred])),'prob_auc':float(roc_auc_score(yy,p[:,1])),'spe':float(spe),'sen':float(sen),'ba':float((spe+sen)/2),'asd_f1':float(f1_score(yy,pred,pos_label=0,zero_division=0)),'tp':tp,'fn':fn,'fp':fp,'tn':tn}
def historical_rows(which):
 import numpy as np,torch
 c=runtime(); x,y,si,pi=load_data()[0]; rows=[]
 if which=='E1': src=E51_2; 
 for f,(tr,te) in enumerate(FOLDS,1):
  if which=='E1':
   d=torch_load(src/f'E1_f{f:02d}.best.pt'); base=c['d3sag'].LegacyFixedGFR(c['network'].GraphLearn,c['network'].GCN,c['utils'],th=.9).cuda(); cam=c['d3sag'].PLNNoPEN([48,6,10,256],hp(),c['layers']).cuda(); base.load_state_dict(d['model']); cam.load_state_dict(d['cam'])
  else:
   d=torch_load(ROOT/f'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints/g0_f{f:02d}.best.pt'); base=c['network'].GFR([48,6,10,256],[48,6,10,256],th=.9,mode='weighted-cosine').cuda(); cam=c['d3sag'].PLNNoPEN([48,6,10,256],hp(),c['layers']).cuda(); base.load_state_dict(d['model']['gfr']); cam.load_state_dict(d['model']['pln'])
  base.eval(); cam.eval(); feat=torch.from_numpy(x).float().cuda(); targ=torch.from_numpy(y).long().cuda(); _,parts=c['layers'].InitialDataLayer([48,6,10,256])(feat)
  with torch.no_grad():
   if which=='E1': fused,_,_=base(parts)
   else: fused,_,_=base(parts,4,targ,torch.as_tensor(tr,device='cuda'))
   lp=cam(fused)[0]
  ckbase = src if which=='E1' else ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints'
  ckname = ('E1' if which=='E1' else 'g0') + f'_f{f:02d}.best.pt'
  row=metrics(lp,y,te); row.update(group=which,fold=f,seed=0,status='REFERENCE',best_epoch=int(d.get('best_epoch',0)),epochs_executed=int(d.get('epochs_executed',0)),split_sha=split_sha(f),best_sha=sha_file(ckbase/ckname)); rows.append(row)
 return rows
def prepare_fold(f):
 import numpy as np,torch
 p=FD/f'fold{f:02d}.npz'; j=FD/f'fold{f:02d}.json';
 if p.exists() and j.exists(): return
 c=runtime(); x,y,si,pi=load_data()[0]; tr,te=FOLDS[f-1]; _,_,_,_=build_e1(f)
 # continuous E1 GraphLearn scores before threshold, fixed and detached
 base,_,_,_=build_e1(f); feat=torch.from_numpy(x).float().cuda(); _,parts=c['layers'].InitialDataLayer([48,6,10,256])(feat)
 with torch.no_grad():
  _,adjs,_=base(parts); cont=[g.score431.detach().cpu().numpy() for g in base.graphs]
 k=(np.clip((cont[0]+1)/2,0,1)+np.clip((cont[3]+1)/2,0,1))/2; k=np.asarray(k,np.float32); np.fill_diagonal(k,0)
 ev=c['evidence'].compute_evidence(x,y,tr,pi,si,x[:,48:64],f,FD/f'evidence{f:02d}.npz')
 rng=np.random.default_rng(514900+f); a=np.asarray(tr)[y[tr]==0]; b=np.asarray(tr)[y[tr]==1]; n=min(256//2,len(a),len(b)); ids=np.concatenate([np.sort(rng.choice(a,n,False)),np.sort(rng.choice(b,n,False))]); blocks=np.array_split(ids,3)
 # fixed within-site reliability permutation and PHENO auxiliary permutation
 sid=ev['site']; qperm=np.arange(len(x));
 for s in np.unique(sid[tr]):
  z=np.asarray(tr)[sid[tr]==s]; sh=z.copy(); rng.shuffle(sh); qperm[z]=sh
 phperm=qperm.copy()
 np.savez_compressed(p,train=tr,test=te,align_ids=ids,k=k, qrel=ev['qrel'],site_onehot=x[:,si],qperm=qperm,phperm=phperm,pheno_nosite=x[:,pi], blocks=np.array(blocks,dtype=object))
 meta={k:v for k,v in ev.items() if k not in ('train','site','qrel','oof','null_cka')}; meta.update(fold=f,split_sha=split_sha(f),align_ids=ids.tolist(),block_sizes=[len(v) for v in blocks],site_idx=si.tolist(),pheno_idx=pi.tolist(),q_sha=ev['sha_q']); atomic_json(j,meta)
def load_fd(f):
 import numpy as np,torch
 z=np.load(FD/f'fold{f:02d}.npz',allow_pickle=True); out={k:z[k] for k in z.files}; out['blocks']=[torch.as_tensor(a,device='cuda') for a in out['blocks']]; out['qrel']=torch.from_numpy(out['qrel']).float().cuda(); out['site_onehot']=torch.from_numpy(out['site_onehot']).float().cuda(); out['pheno_nosite']=torch.from_numpy(out['pheno_nosite']).float().cuda(); out['k']=torch.from_numpy(out['k']).float().cuda(); out['qperm']=torch.from_numpy(out['qperm']).long().cuda(); out['phperm']=torch.from_numpy(out['phperm']).long().cuda(); return out
def build_state(g,f):
 import torch
 c=runtime(); x,y,si,pi=load_data()[0]; tr,te=FOLDS[f-1]; base,cam,r,a=build_e1(f); seed_all(510000+f*100+1)
 with torch.random.fork_rng(devices=[0]): torch.manual_seed(514000+f); model=c['d3sag'].D3SAG(c['network'].GraphLearn,c['network'].GCN,c['layers'],c['utils'],th=.9,mode=g).cuda()
 # copy E1 anchor into nested base without touching global RNG
 c['d3sag'].load_e1_anchor(model,a['model']['gfr']); cam.load_state_dict(a['model']['pln'])
 init_rng=pack_rng(); params_base=list(model.base.parameters()); ids={id(p) for p in params_base}; new=[p for p in model.parameters() if id(p) not in ids]
 ob=torch.optim.Adam(params_base,lr=CFG['base']['lr'],weight_decay=CFG['base']['weight_decay']); on=torch.optim.Adam(new,lr=.001,weight_decay=.001); oc=torch.optim.Adam(cam.parameters(),lr=CFG['base']['lr'],weight_decay=CFG['base']['weight_decay'])
 feat=torch.from_numpy(x).float().cuda(); targ=torch.from_numpy(y).long().cuda(); return {'group':g,'fold':f,'model':model,'cam':cam,'ob':ob,'on':on,'oc':oc,'feat':feat,'targ':targ,'train':torch.as_tensor(tr,device='cuda'),'test':te,'parts':c['layers'].InitialDataLayer([48,6,10,256]),'si':si,'pi':pi,'init_rng_sha':rng_sha(init_rng),'anchor_rng_sha':rng_sha(a.get('rng',r)),'param_sha':a['parameter_sha']}
def forward(st,fd,train=True,intervention=None,epoch=0):
 import torch
 c=runtime(); st['model'].train(train); st['cam'].train(train); _,parts=st['parts'](st['feat']); ph=fd['pheno_nosite']; ev={'qrel':fd['qrel'],'site_onehot':fd['site_onehot'],'pheno_local_idx':torch.as_tensor(st['pi'],device='cuda'),'rS':float(json.loads((FD/f'fold{st["fold"]:02d}.json').read_text())['rS']),'rC':float(json.loads((FD/f'fold{st["fold"]:02d}.json').read_text())['rC'])}
 # wrong quality gate and wrong PHENO are fixed, label-free negative controls
 if st['group']=='E10': ev['qrel']=fd['qrel'][fd['qperm']]; ev['rS_wrong']=ev['rS']; ev['rC_wrong']=ev['rC']
 if st['group']=='E11': ev['qrel']=torch.ones_like(fd['qrel'])
 if st['group']=='E12' or intervention=='aux_perm': ph=ph[fd['phperm']]
 if intervention=='wrong_gate': ev['qrel']=fd['qrel'][fd['qperm']]
 block=fd['blocks'][epoch%len(fd['blocks'])]
 ev['geometry_perm']=torch.arange(len(block)-1,-1,-1,device='cuda')
 fused,adjs,z=st['model'](parts,ev,ph,mode=st['group'],align_block=block,wrong_gate=(st['group']=='E10'),quality_off=(st['group']=='E11'),wrong_geometry=(st['group']=='E9'))
 lp=st['cam'](fused)[0]; tr=st['train']; counts=torch.stack([(st['targ'][tr]==0).sum(),(st['targ'][tr]==1).sum()]).float(); wt=len(tr)/(2*counts); cls=c['utils'].ClsLoss(lp,st['targ'],tr,wt)
 graph=sum(c['utils'].GraphConstructLoss(parts[m],adjs[m],1.0,.5,0.) for m in range(4))/4
 core,short,dec,site,align,n=st['model'].losses(st['targ'],tr,ev,block,st['group'],wrong_geometry=(st['group']=='E9'),uniform=(st['group']=='E6'),enable_align=st['group'] in {'E3','E5','E6','E7','E8','E9','E10','E11','E12'},enable_short=st['group'] not in {'E2','E3','E4'},enable_site=st['group'] not in {'E2','E3'})
 # E2 ordinary dual view; E3 fixed strong alignment; E4 route no align; E5 full
 total=cls+graph+.02*core+.001*short+.001*dec+.001*site+.01*align
 if st['group']=='E2': total=cls+graph
 if st['group']=='E3': total=cls+graph+.02*core+.001*short+.001*dec+.001*site+.01*align
 if st['group']=='E4': total=cls+graph+.02*core+.001*short+.001*dec+.001*site
 if st['group']=='E8': total=cls+graph+.02*core+.001*short+.001*dec+.001*site+.01*align
 if st['group']=='E12': total=cls+graph+.02*core+.001*short+.001*dec+.001*site+.01*align
 if intervention=='base_only': lp=st['cam'](z['base'])[0]
 return {'lp':lp,'cls':cls,'graph':graph,'core':core,'short':short,'dec':dec,'site':site,'align':align,'total':total,'z':z,'fused':fused,'parts':parts,'adjs':adjs}
def ckpt(g,f): return CK/f'{g}_f{f:02d}.best.pt'
def task(g,f): return TK/f'{g}_f{f:02d}.json'
def train_one(g,f):
 import torch
 prepare_fold(f); tp=task(g,f)
 if tp.exists() and json.loads(tp.read_text()).get('status')=='COMPLETE': return
 st=build_state(g,f); fd=load_fd(f); best=-1; be=-1; wait=0; bestm=bestc=bestob=beston=bestoc=None; diag={}; start=0; t=time.time()
 try:
  for ep in range(CFG['base']['epochs']):
   st['ob'].zero_grad();st['on'].zero_grad();st['oc'].zero_grad(); out=forward(st,fd,True,epoch=ep); out['total'].backward()
   for m in (st['model'],st['cam']):
    for p in m.parameters():
     if p.grad is not None: assert torch.isfinite(p.grad).all()
   st['ob'].step();st['on'].step();st['oc'].step()
   with torch.no_grad(): ev=forward(st,fd,False,epoch=ep); met=metrics(ev['lp'],st['targ'].detach().cpu().numpy(),st['test'])
   if met['acc']>best:
    best=met['acc'];be=ep;wait=0;bestm={k:v.detach().cpu().clone() for k,v in st['model'].state_dict().items()};bestc={k:v.detach().cpu().clone() for k,v in st['cam'].state_dict().items()};bestob=copy.deepcopy(st['ob'].state_dict());beston=copy.deepcopy(st['on'].state_dict());bestoc=copy.deepcopy(st['oc'].state_dict());diag={'core':float(out['core'].detach()),'align':float(out['align'].detach()),'short':float(out['short'].detach()),'dec':float(out['dec'].detach()),'site':float(out['site'].detach()),'res_ratio':float(ev['z']['add'].norm()/ev['z']['base'].norm().clamp_min(1e-8))}; torch_save(ckpt(g,f),{'group':g,'fold':f,'best_epoch':be,'epochs_executed':ep+1,'metrics':met,'model':bestm,'cam':bestc,'ob':bestob,'on':beston,'oc':bestoc,'diagnostics':diag,'split_sha':split_sha(f),'anchor_parameter_sha':st['param_sha'],'anchor_rng_sha':st['init_rng_sha'],'historical_anchor_rng_sha':st['anchor_rng_sha'],'source_sha':sha_file(EXP/'d3sag.py')})
   else: wait+=1
   if wait>CFG['base']['early']: break
  assert bestm is not None; st['model'].load_state_dict(bestm);st['cam'].load_state_dict(bestc)
  with torch.no_grad(): chk=forward(st,fd,False,epoch=be); met=metrics(chk['lp'],st['targ'].detach().cpu().numpy(),st['test'])
  assert abs(met['acc']-best)<1e-8
  row={'group':g,'fold':f,'seed':0,'gpu':0,'status':'COMPLETE','best_epoch':be,'epochs_executed':ep+1,'wall_seconds':time.time()-t,'split_sha':split_sha(f),'best_sha':sha_file(ckpt(g,f)),'anchor_parameter_sha':st['param_sha'],'anchor_rng_sha':st['init_rng_sha'],'historical_anchor_rng_sha':st['anchor_rng_sha'],**met,**diag}; atomic_json(tp,row); print('DONE',g,f,met['acc'],flush=True)
 except Exception as e:
  atomic_json(tp,{'group':g,'fold':f,'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}); raise
def prepare():
 import numpy as np
 for p in (RT,FD,CK,LG,TK,RES):p.mkdir(parents=True,exist_ok=True)
 load_data(); [prepare_fold(f) for f in range(1,11)]
 e0=historical_rows('E0'); e1=historical_rows('E1'); rows=e0+e1
 with (RES/'fold_metrics.csv').open('w',newline='') as f:
  fields=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 man={'experiment':'Experiment51.4_D3SAG','status':'PREPARED','config_sha':sha_file(EXP/'config.json'),'data_shape':[871,320],'label_counts':[403,468],'folds':{str(f):split_sha(f) for f in range(1,11)},'groups':13,'planned_new_folds':110,'e1_seed_rule':'510000 + fold*100 + 1','new_module_seed_rule':'isolated torch fork 514000 + fold','experiment52':'EXPERIMENT52_NOT_FOUND','protected_sha256':{n:sha_file(ROOT/n) for n in ['network.py','model.py','main.py','layers.py','utils.py','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']},'runtime_excluded_from_delivery':True};atomic_json(EXP/'manifest.json',man);print('PREPARE_PASS')
def controller():
 gs=[f'E{i}' for i in range(2,13)]; jobs=[(g,f) for g in gs for f in range(1,11)]; pro=[]; workers=min(5,int(os.environ.get('D3SAG_WORKERS',CFG['gpu_plan']['workers']))); env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0'
 while jobs or pro:
  alive=sum(p.poll() is None for _,_,p,_ in pro)
  while alive<workers and jobs:
   g,f=jobs.pop(0);log=(LG/f'{g}_f{f:02d}.log').open('ab');p=subprocess.Popen([sys.executable,'-B',str(EXP/'run.py'),'worker','--group',g,'--fold',str(f)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT);pro.append((g,f,p,log));alive+=1
  for it in list(pro):
   if it[2].poll() is not None:it[3].close();pro.remove(it)
  time.sleep(.5)
 status=[json.loads(task(g,f).read_text()).get('status','MISSING') if task(g,f).exists() else 'MISSING' for g,f in [(g,f) for g in gs for f in range(1,11)]];m=json.loads((EXP/'manifest.json').read_text());m.update(status='TRAINING_COMPLETE' if status.count('COMPLETE')==110 else 'TRAINING_PARTIAL',completed_new_folds=status.count('COMPLETE'),failed_tasks=status.count('FAILED'));atomic_json(EXP/'manifest.json',m);print('CONTROLLER_DONE',status.count('COMPLETE'),status.count('FAILED'))
def aggregate():
 import numpy as np
 rows=historical_rows('E0')+historical_rows('E1')
 for g in [f'E{i}' for i in range(2,13)]:
  for f in range(1,11): rows.append(json.loads(task(g,f).read_text()))
 with (RES/'fold_metrics.csv').open('w',newline='') as f:
  fields=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 sums=[]
 for g in [f'E{i}' for i in range(13)]:
  rr=[r for r in rows if r['group']==g]; z={'group':g,'folds':len(rr)}
  for k in ('acc','hard_auc','prob_auc','spe','sen','ba','asd_f1'):
   a=np.array([float(r[k]) for r in rr]);z[k+'_mean']=float(a.mean());z[k+'_sd']=float(a.std())
  sums.append(z)
 for name,rs in [('summary',sums)]:
  fields=sorted({k for r in rs for k in r});
  with (RES/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
 by={(r['group'],int(r['fold'])):r for r in rows}; ps=[]
 for other in ['E0','E1','E2','E3','E4','E6','E7','E8','E9','E10','E11','E12']:
  for k in ('acc','hard_auc','prob_auc','spe','sen'):
   d=np.array([(float(by[('E5',f)][k])-float(by[(other,f)][k]))*100 for f in range(1,11)]);ps.append({'comparison':f'E5_vs_{other}','metric':k,'delta_pp':float(d.mean()),'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum())})
 fields=sorted({k for r in ps for k in r});
 with (RES/'paired_comparisons.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(ps)
 m=json.loads((EXP/'manifest.json').read_text());m.update(status='AGGREGATED',summary_sha=sha_file(RES/'summary.csv'),fold_metrics_sha=sha_file(RES/'fold_metrics.csv'));atomic_json(EXP/'manifest.json',m);print('AGGREGATE_PASS')
def main():
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('prepare');sp.add_parser('controller');sp.add_parser('aggregate');w=sp.add_parser('worker');w.add_argument('--group',required=True);w.add_argument('--fold',type=int,required=True);a=ap.parse_args();
 if a.cmd=='prepare':prepare()
 elif a.cmd=='controller':controller()
 elif a.cmd=='aggregate':aggregate()
 else:train_one(a.group,a.fold)
if __name__=='__main__':main()

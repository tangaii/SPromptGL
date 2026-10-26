#!/usr/bin/env python3
"""E42 audit/preflight/anchor/q1/q2/q3/q4/finalize/resume. No historical Python imports."""
from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import argparse,contextlib,copy,csv,datetime,fcntl,hashlib,io,json,pickle,random,shutil,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr,rankdata
import mechanism as m
HERE=Path(__file__).resolve().parent;RT=HERE/'runtime';RES=HERE/'results';ROOT=Path('/data/adodas/SPromptGL-main')
CFG=json.loads((HERE/'config.json').read_text());OLD=Path(CFG['historical_anchor']);EDGES=CFG['canonical_edges'];COLS=np.array(EDGES)+64
_CTX=None;_DATA=None

def native(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,Path):return str(v)
 if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
 raise TypeError(type(v).__name__)
def jw(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=native,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def jr(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def cw(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r)) or ['fold','status'];tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:(json.dumps(v,default=native,ensure_ascii=False) if isinstance(v,(dict,list,np.ndarray)) else v) for k,v in r.items()} for r in rows]);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def cr(p):return list(csv.DictReader(Path(p).open())) if Path(p).exists() else []
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def event(kind,**values):
 with (RT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=stamp(),kind=kind,**values),default=native)+'\n')
def data():
 global _DATA
 if _DATA is None:
  f=pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv');modes=np.load(ROOT/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item();x=f.iloc[:,:-1].to_numpy('float32');y=f.iloc[:,-1].to_numpy('int64')-1
  assert x.shape==(871,320) and np.bincount(y).tolist()==[403,468] and [len(v) for v in modes.values()]==[48,6,10,256]
  assert list(f.columns[:-1])==[str(v) for a in modes.values() for v in a];assert np.isfinite(x).all()
  subjects=pd.read_csv(ROOT/'experiment/experiment11_roi_mapping_recovery/results/subject_row_mapping.csv')
  _DATA=dict(x=x,y=y,folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)),columns=list(f.columns),modes=modes,subjects=subjects,site=x[:,26:46].argmax(1),age=x[:,:26].argmax(1),sex=x[:,46:48].argmax(1))
 return _DATA

def library():return jr(HERE/'prompt_candidates.json')['legacy_library']
def hypotheses():return [h for h in library() if h['candidate']]
def ref(fold):
 p=RT/f'cache/reference_{fold:02}.pt'
 if p.exists():return load(p)
 d=data();tr,te=d['folds'][fold-1];r=m.reference(d['x'],d['y'],tr,te,EDGES,fold)
 r.update(train=tr,test=te,input_sha=m.sha(ROOT/'data/ABIDE/processed_standard_data.csv'),split_sha=m.hjson([tr.tolist(),te.tolist()]))
 r['fit_subject_ID']=[data()['subjects']['subject_SUB_ID'].to_numpy()[idx].tolist() for idx in r['fits']];r['final_fit_subject_ID']=data()['subjects']['subject_SUB_ID'].to_numpy()[r['final_fit']].tolist()
 save(p,r);return r

def runtime():
 global _CTX
 if _CTX is None:
  torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.device_count()==1
  sys.path.insert(0,str(RT/'sources'));import model,network,utils
  _CTX=dict(model=model,network=network,utils=utils)
 return _CTX

def agent(fold,initial=False):
 c=runtime();d=data();tr,te=d['folds'][fold-1];a=load(RT/f'anchors/anchor_fold{fold:02}.pt');before=m.rng()
 with torch.random.fork_rng(devices=[0]):
  m.restore(a['entry_rng'])
  with contextlib.redirect_stdout(io.StringIO()):v=c['model'].EvalHelper([48,6,10,256],d['x'],d['y'],SimpleNamespace(**CFG['baseline']),tr,te)
  v.constructed_sha42=m.trunksha(v);v.constructed_rng42=m.rngsha(m.rng())
 m.restore(before)
 v.MF_sav.close();v.GCMP_sav.close()
 s=a['model'] if initial else load(RT/f'anchors/best_fold{fold:02}.pt')['model']
 v.MessagePassing0.load_state_dict(s['gfr']);v.ModalFusion.load_state_dict(s['pln'])
 if initial:v.optimizer_MP0.load_state_dict(a['mp']);v.optimizer_MF.load_state_dict(a['mf'])
 v.MessagePassing0.eval();v.ModalFusion.eval()
 return v

def forward(v,raw=None,fixed=None,loss=False):
 raw=v.feat if raw is None else raw
 old=getattr(v.MessagePassing0,'fixed_graph42',None);v.MessagePassing0.fixed_graph42=fixed
 try:
  n,parts=v.dataLayer(raw);feat,graphs,aloss=v.MessagePassing0(parts,n,v.targ,v.trn_idx);logp,_,_=v.ModalFusion(feat)
  gl=sum(runtime()['utils'].GraphConstructLoss(parts[k],graphs[k],1.,.5,0.) for k in range(4))/4 if loss else None
 finally:v.MessagePassing0.fixed_graph42=old
 return logp,graphs,gl

def frozen(fold):
 v=agent(fold)
 for mod in (v.MessagePassing0,v.ModalFusion):mod.requires_grad_(False)
 with torch.no_grad():p,g,_=forward(v)
 return v,p,[a.detach() for a in g]
def metric(v,p):return m.metrics(data()['y'][v.tst_idx],p.detach().cpu().numpy()[v.tst_idx].argmax(1))

def read_audit():
 sources=['README.md','main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy','data/ABIDE/data_processing.ipynb','data/ABIDE/phenotyic_processing.ipynb','experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv','experiment/experiment11_roi_mapping_recovery/results/subject_row_mapping.csv','experiment/experiment32_disease_contrastive_evidence_routing/integration.py','experiment/experiment32_disease_contrastive_evidence_routing/run.py','experiment/experiment32_disease_contrastive_evidence_routing/results/report.md','experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv','experiment/experiment32_disease_contrastive_evidence_routing/manifest.json']
 paths=[ROOT/p for p in sources]+[OLD/p for p in ['README.md','config.json','prompts.json','method.py','run.py','audit.py','integration.py','manifest.json','results/report.md','results/diagnostics.csv','results/mechanism_audit_E5.csv']]
 e38=Path('/tmp/SPromptGL-E38-B0A-stage/experiment/experiment38_lcher_b0a_anchor_rerun')
 paths +=[e38/p for p in ['method.py','lcher_run.py','results/report.md','results/diagnostics.csv','manifest.json']]
 table=[]
 for p in paths:
  if p.exists():
   b=p.read_bytes();table.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='READ'))
  else:table.append(dict(path=str(p),status='UNAVAILABLE'))
 jw(RT/'source_audit.json',table)
 return table

def audit():
 RT.mkdir(exist_ok=True);RES.mkdir(exist_ok=True)
 for q in ('cache','anchors','tasks','logs','checkpoints'): (RT/q).mkdir(exist_ok=True)
 sources=read_audit();d=data();original=jr(OLD/'prompts.json');hyp=original['hypotheses'];assert len(hyp)==16
 main=[h for h in hyp if h['candidate']];assert len(main)==13 and sorted(e for h in main for e in h['edges'])==EDGES
 mapping=pd.read_csv(ROOT/'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv').set_index('correlation_feature_index')
 for e in EDGES:
  row=mapping.loc[e];assert row.connection_anatomical_status=='VERIFIED_BOTH_ENDPOINTS' and row.verification_status=='VERIFIED_NUMERIC_UNIQUE'
  assert row.unique_match and row.roi_i_name_verified and row.roi_j_name_verified
 assert not set(EDGES)&{195,225,232,236}
 nets=['thalamic-temporal','thalamic-frontal','anterior-posterior default-mode-related','frontal-parietal associative','temporal-frontal social-cognitive','precuneus-parietal associative','insula-opercular attention-related','insula-posterior-cingulate','inferior-frontal-temporal language-related','temporal auditory-association','corticostriatal','sensorimotor-parietal','temporal-parietal social-information']
 candidates=[]
 for i,h in enumerate(main):
  direction={'hyper_candidate':'increased','hypo_candidate':'reduced'}.get(h['direction'],'direction-uncertain');net=nets[i]
  common=dict(hypothesis_id=h['id'],edge_indices=h['edges'],ROI_names=h['connection_names'],direction_status=h['direction'],uncertainty='network candidate; exact-edge direction unverified',source_ids=h['sources'],edge_specific_verified=False)
  texts={
   'P0':(h['asd'],h['td']),
   'P1':(f'ASD candidate: {direction} {net} functional coupling relative to a matched reference. This network-level hypothesis varies across individuals; exact atlas-edge effects remain unverified.',f'TD comparison: reference-compatible {net} functional coupling relative to a matched reference. This network-level hypothesis varies across individuals; exact atlas-edge effects remain unverified.'),
   'P2':([f'ASD network candidate: {direction} coupling between {name.replace(" — "," and ")}. Exact-edge direction is unverified; compare with a matched reference.' for name in h['connection_names']],[f'TD comparison: coupling between {name.replace(" — "," and ")}, compatible with a matched reference. Exact-edge evidence is unverified; typical variation remains possible.' for name in h['connection_names']]),
   'P3':(f'ASD supporting scenario: {direction} {net} deviation versus matched reference. Abstain for conflicting FC or unreliable quality. Age, sex and site calibration are exploratory applicability conditions, not disease evidence.',f'TD-consistent scenario: {net} coupling compatible with matched reference. Abstain for conflicting FC or unreliable quality. Age, sex and site calibration are exploratory applicability conditions, not disease evidence.')}
  for fam,(a,b) in texts.items():candidates.append(dict(**common,family=fam,text_ASD=a,text_TD=b,encoding_rule='separately encode edge clauses then mean' if fam=='P2' else 'single sentence'))
 registry=[dict(hypothesis_id=h['id'],edge_id=e,connection_name=str(mapping.loc[e].connection_name),in_hypothesis=e in h['edges'],network_supported=e in h['edges'],edge_specific_verified=False,direction_status=('direction_conflicted' if h['strength']=='NETWORK_DIRECTION_CONFLICT' else 'network_supported') if e in h['edges'] else 'unknown',source_ids=h['sources'] if e in h['edges'] else [],claim='network hypothesis only; no exact-edge causal validation') for h in main for e in EDGES]
 obj=dict(hypothesis_evidence_registry=registry,source_prompt_path=str(OLD/'prompts.json'),source_prompt_sha=m.sha(OLD/'prompts.json'),provenance='deterministic E42 templates derived from supplied literature synthesis; no GPT-4 API call',legacy_library=hyp,candidates=candidates)
 p=HERE/'prompt_candidates.json'
 if p.exists():
  prior=jr(p)
  for row in prior['candidates']:row.pop('token_count',None)
  if 'hypothesis_evidence_registry' not in prior:prior['hypothesis_evidence_registry']=registry
  assert prior==obj
  if 'hypothesis_evidence_registry' not in jr(p):jw(p,obj)
 else:jw(p,obj)
 m.prepare_sources(ROOT,RT/'sources')
 git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 info=dict(experiment=CFG['experiment'],version=CFG['version'],started=stamp(),started_unix=time.time(),scientific='AUDIT',delivery='TMP_ONLY',stage=str(HERE),target=str(ROOT/'experiment'/HERE.name),git_head=git,source_audit=sources,config_sha=m.sha(HERE/'config.json'),input_sha=m.sha(ROOT/'data/ABIDE/processed_standard_data.csv'),prompt_sha=m.sha(p),shape=list(d['x'].shape),label_counts=np.bincount(d['y']),hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True).strip(),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version),phases={},storage=jr(RT/'storage_initial.json'))
 jw(RES/'manifest.json',info);print('AUDIT PASS',flush=True)

def encode():
 sys.path.insert(0,str(ROOT));from CLIP.clip import clip
 obj=jr(HERE/'prompt_candidates.json');texts=[];slots=[]
 for row in obj['candidates']:
  for key in ('text_ASD','text_TD'):
   val=row[key] if isinstance(row[key],list) else [row[key]];ii=list(range(len(texts),len(texts)+len(val)));texts.extend(val);slots.append((row['family'],row['hypothesis_id'],key,ii))
 hyp=hypotheses();anatomy=[]
 for e in EDGES:
  hh=next(h for h in hyp if e in h['edges']);name=hh['connection_names'][hh['edges'].index(e)];anatomy.append('Resting-state functional connection between '+name.replace(' — ',' and ')+'.')
 ai=list(range(len(texts),len(texts)+39));texts+=anatomy;neutral=[]
 for h in hyp:
  name=h['connection_names'][0].replace(' — ',' and ')
  pair=[f'Acquisition record: resting-state coupling between {name} and related parcels. The record describes atlas locations and measurement processing without a diagnostic interpretation.',f'Reference record: resting-state coupling between {name} and related parcels. The record describes atlas locations and measurement processing without a diagnostic interpretation.']
  neutral.append(list(range(len(texts),len(texts)+2)));texts+=pair
 tokens=clip.tokenize(texts,context_length=77,truncate=False);lengths=(tokens!=0).sum(1);weights=Path.home()/'.cache/clip/RN50.pt';assert m.sha(weights)==CFG['text']['weights_sha']
 before=m.rng();
 with torch.random.fork_rng(devices=[0]):
  model,_=clip.load('RN50','cuda');model.eval().requires_grad_(False)
  with torch.no_grad():
   emb=torch.cat([model.encode_text(t.cuda()).float().cpu() for t in tokens.split(32)]);assert torch.equal(emb[:32],model.encode_text(tokens[:32].cuda()).float().cpu())
 assert m.rngsha(before)==m.rngsha(m.rng())
 bank={fam:torch.zeros(13,2,1024) for fam in ('P0','P1','P2','P3')}
 for fam,hid,key,ii in slots:bank[fam][int(hid[1:])-1,int(key=='text_TD')]=emb[ii].mean(0)
 bank['anatomy']=emb[ai];bank['neutral']=torch.stack([emb[ii] for ii in neutral]);bank['metadata']=dict(texts=texts,token_lengths=lengths.tolist(),tokens_sha=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),embedding_sha=hashlib.sha256(emb.numpy().tobytes()).hexdigest(),weights_sha=m.sha(weights),clip_script_sha=m.sha(ROOT/'CLIP/clip/clip.py'),tokenizer_sha=m.sha(ROOT/'CLIP/clip/simple_tokenizer.py'))
 for row in obj['candidates']:
  row['token_count']={key:[int(lengths[ii]) for fam,hid,kk,ids in slots if fam==row['family'] and hid==row['hypothesis_id'] and kk==key for ii in ids] for key in ('text_ASD','text_TD')}
 # Token metadata is deterministic, saved before any model performance is inspected.
 jw(HERE/'prompt_candidates.json',obj);save(RT/'cache/text_bank.pt',bank)
 pairs=[]
 for fam in ('P0','P1','P2','P3'):
  t=m.norm(bank[fam].numpy());cross=t.reshape(26,1024)@t.reshape(26,1024).T
  for i,h in enumerate(hyp):pairs.append(dict(family=fam,hypothesis=h['id'],pair_cos=float(t[i,0]@t[i,1]),pair_distance=float(np.linalg.norm(t[i,0]-t[i,1])),cross_hypothesis_cos_mean=float((cross.sum()-np.trace(cross))/(26*25)),source_ids=h['sources'],edge_specific_verified=False,network_evidence=h['strength'],token_count=next(r['token_count'] for r in obj['candidates'] if r['family']==fam and r['hypothesis_id']==h['id'])))
 cw(RES/'prompt_audit.csv',pairs);info=jr(RES/'manifest.json');info['text']=bank['metadata'];info['prompt_sha']=m.sha(HERE/'prompt_candidates.json');jw(RES/'manifest.json',info)
 print('TEXT PASS',len(texts),'max tokens',int(lengths.max()),flush=True)

def anchor():
 # Existing immutable signatures are admissible; no historical metric is called newly trained.
 history={int(r['fold']):r for r in cr(ROOT/'experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv') if r['config']=='B0'}
 oldrows={int(r['fold']):r for r in cr(OLD/'results/fold_metrics.csv') if r['group']=='B0'};oldman=jr(OLD/'manifest.json');assert all(not r['differences'] for r in oldman['anchor_audit'])
 for name in ('main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh'):
  assert m.sha(ROOT/name)==oldman['source_audit']['read_only_sources'][str(ROOT/name)]
 records=[]
 runtime()['network']._prompt_text_features=None
 for fold in range(1,11):
  a=OLD/f'.runtime/anchors/anchor_fold{fold:02}.pt';b=Path(oldrows[fold]['best_checkpoint']);assert m.sha(a)==oldrows[fold]['anchor_sha'] and m.sha(b)==oldrows[fold]['best_checkpoint_sha256']
  aa=load(a);bb=load(b);tr,te=data()['folds'][fold-1];assert aa['fold_hash']==bb['fold_hash']==hashlib.sha256(np.concatenate([tr,te]).tobytes()).hexdigest();assert bb['input_sha']==m.sha(ROOT/'data/ABIDE/processed_standard_data.csv')
  assert aa['initial_trunk_sha256']==history[fold]['initial_trunk_sha256'];assert aa['initial_rng_sha256']==history[fold]['initial_rng_sha256']==m.rngsha(aa['entry_rng'])
  for k in ['asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed','initial_trunk_sha256','initial_rng_sha256']:assert str(oldrows[fold][k])==str(history[fold][k]),(fold,k)
  for k,kk in [('acc','acc'),('hard_auc','auc'),('spe_td','spe_td'),('sen_asd','sen_asd')]:assert abs(float(oldrows[fold][k])-float(history[fold][kk]))<1e-14
  for src,dst in [(a,RT/f'anchors/anchor_fold{fold:02}.pt'),(b,RT/f'anchors/best_fold{fold:02}.pt')]:
   if not dst.exists():shutil.copyfile(src,dst)
   assert m.sha(src)==m.sha(dst)
  v=agent(fold,True);assert v.constructed_sha42==aa['initial_trunk_sha256'],('constructor',fold);assert v.constructed_rng42==m.rngsha(aa['post_rng']),('post_constructor_rng',fold);assert m.trunksha(v)==aa['initial_trunk_sha256'];v.optimizer_MP0.load_state_dict(aa['mp']);v.optimizer_MF.load_state_dict(aa['mf'])
  v.MessagePassing0.load_state_dict(bb['model']['gfr']);v.ModalFusion.load_state_dict(bb['model']['pln'])
  with torch.no_grad():p,_,_=forward(v)
  r=metric(v,p)
  for k,val in r.items():assert abs(val-float(oldrows[fold][k]))<1e-14,(fold,k,val)
  records.append(dict(group='B0A_VALIDATED_REUSE',seed=0,fold=fold,physical_gpu=0,origin='E40 immutable exact E32 replay, re-evaluated in E42; NOT retrained',**r,best_epoch=int(history[fold]['best_epoch']),epochs_executed=int(history[fold]['epochs_executed']),initial_trunk_sha256=aa['initial_trunk_sha256'],initial_rng_sha256=aa['initial_rng_sha256'],anchor_sha=m.sha(a),best_sha=m.sha(b),best_checkpoint=str(RT/f'anchors/best_fold{fold:02}.pt'),replay_status='EXACT_REUSE_VALIDATED'))
  print('ANCHOR',fold,'EXACT',flush=True);del v;torch.cuda.empty_cache()
 cw(RES/'fold_metrics.csv',records);info=jr(RES/'manifest.json');info['anchor']=dict(status='ANCHOR_EXACT_REUSE_PASS',folds=10,new_training_folds=0,validation_gpu=0,entries=records);jw(RES/'manifest.json',info)

def correlation(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
 if np.std(a)<1e-10 or np.std(b)<1e-10:return 0.
 return float(spearmanr(a,b).statistic)
def direction_vector():
 d=np.zeros(39,'float32')
 for h in hypotheses():
  for e in h['edges']:d[EDGES.index(e)]={'hyper_candidate':1,'hypo_candidate':-1}.get(h['direction'],0)
 return d

def q1(fold):
 d=data();tr,te=d['folds'][fold-1];r=ref(fold);hyp=hypotheses();bank=load(RT/'cache/text_bank.pt');stats=[]
 adj=m.partial(r['z'],d['x'][:,:48],tr)
 for h in hyp:
  jj=[EDGES.index(e) for e in h['edges']];sign={'hyper_candidate':1,'hypo_candidate':-1}.get(h['direction'],0)
  signal=(adj[:,jj].mean(1)*sign if sign else abs(adj[:,jj]).mean(1));effect=float(m.smd(signal[tr[d['y'][tr]==0]],signal[tr[d['y'][tr]==1]]));inner=[];sites=[]
  for k in range(5):
   idx=tr[r['assignment'][tr]==k];inner.append(float(m.smd(signal[idx[d['y'][idx]==0]],signal[idx[d['y'][idx]==1]])))
  for s in sorted(set(d['site'][tr])):
   idx=tr[d['site'][tr]==s];a=idx[d['y'][idx]==0];b=idx[d['y'][idx]==1]
   if min(len(a),len(b))>=5:sites.append(dict(site=int(s),n_asd=len(a),n_td=len(b),effect=float(m.smd(signal[a],signal[b]))))
  stability=float(np.mean(np.array(inner)>0));site_consistency=float(np.mean([s['effect']>0 for s in sites])) if sites else 0
  status='PASS' if effect>=.1 and stability>=.8 and site_consistency>=.6 else 'MIXED' if abs(effect)>=.1 else 'UNSUPPORTED'
  if not sites:status='UNTESTABLE'
  stats.append(dict(kind='TRAIN_EFFECT',fold=fold,hypothesis=h['id'],direction=h['direction'],effect_smd=effect,inner_direction_stability=stability,site_direction_consistency=site_consistency,site_effects=sites,inner_effects=inner,status=status,edge_specific_verified=False,source_ids=h['sources'],n_asd=int(sum(d['y'][tr]==0)),n_td=int(sum(d['y'][tr]==1)),signal='signed candidate-direction z' if sign else 'absolute deviation only; no sign inferred'))
 variants=[]
 for fam in ('P0','P1','P2','P3'):
  for var in ('REAL','NEUTRAL','SWAP_ASD_TD','DERANGED_HYPOTHESIS','GRAM_MATCHED_ANONYMOUS'):
   t=bank['neutral'] if var=='NEUTRAL' else bank[fam];w=m.text_weights(t.numpy(),bank['anatomy'].numpy(),hyp,EDGES,var);variants.append((fam,var,w))
 for var in ('NUMERIC_ONLY','STRUCTURED_DIAGNOSTIC'):variants.append(('ALL',var,m.text_weights(bank['P0'].numpy(),bank['anatomy'].numpy(),hyp,EDGES,var)))
 oof={f'{fam}:{var}':np.full(len(tr),-1,int) for fam,var,_ in variants};rows=[];models={}
 inner=list(StratifiedKFold(5,shuffle=True,random_state=42000+fold).split(tr,d['y'][tr]))
 for k,(aa,bb) in enumerate(inner):
  a,b=tr[aa],tr[bb];rr=m.reference(d['x'],d['y'],a,b,EDGES,42000+fold*10+k)
  assert not np.intersect1d(np.concatenate(rr['fits']),b).size
  for fam,var,w in variants:
   f=m.probe_features(rr['z'],w,hyp,EDGES,structured=(var!='NUMERIC_ONLY'));pred,coef,bias=m.probe_fit(f[a],d['y'][a],f[b],420000+fold)
   oof[f'{fam}:{var}'][bb]=pred
   rows.append(dict(fold=fold,inner_fold=k,family=fam,control=var,scope='OUTER_TRAIN_INNER_HELDOUT',**m.metrics(d['y'][b],pred),parameters=31,steps=150,train_n=len(a),validation_n=len(b)))
 for fam,var,w in variants:
  pred=oof[f'{fam}:{var}'];assert np.all(pred>=0)
  mm=m.metrics(d['y'][tr],pred);rows.append(dict(fold=fold,inner_fold='OOF_ALL',family=fam,control=var,scope='OUTER_TRAIN_OOF',**mm,parameters=31,steps=150,train_n=len(tr),validation_n=len(tr)))
  if var=='REAL':
   f=m.probe_features(r['z'],w,hyp,EDGES);_,coef,bias=m.probe_fit(f[tr],d['y'][tr],f[tr],420000+fold);models[fam]=dict(weights=w,coef=coef,bias=bias,oof=mm)
 selected=sorted(models,key=lambda fam:(-models[fam]['oof']['acc'],-models[fam]['oof']['hard_auc'],fam))[0]
 # Freeze before outer-test labels or counterfactual outputs are read.
 save(RT/f'cache/q1_{fold:02}.pt',dict(models=models,selected_family=selected,train=tr,train_hash=m.hjson(tr.tolist()),selection_used_test=False))
 cw(RT/f'tasks/q1_{fold:02}_probe.csv',rows);cw(RT/f'tasks/q1_{fold:02}_audit.csv',stats)
 print('Q1',fold,'TRAIN_ONLY_FROZEN',selected,flush=True)

def choose_teacher(fold):
 d=data();tr,_=d['folds'][fold-1];ids=d['subjects']['subject_SUB_ID'].to_numpy();out=[]
 for label in (0,1):
  buckets={s:sorted([int(i) for i in tr if d['y'][i]==label and d['site'][i]==s],key=lambda i:hashlib.sha256(f'42000:{ids[i]}'.encode()).hexdigest()) for s in sorted(set(d['site'][tr]))}
  buckets={s:v for s,v in buckets.items() if v};take=[]
  while len(take)<8:
   for s in buckets:
    if buckets[s] and len(take)<8:take.append(buckets[s].pop(0))
  out+=take
 assert len(set(out))==16 and not np.intersect1d(out,d['folds'][fold-1][1]).size
 return np.array(out)

def discrim(fold):
 d=data();tr,te=d['folds'][fold-1];r=ref(fold);cov=d['x'][:,:48];z=m.partial(r['z'],cov,tr);est=[]
 for k in range(5):
  fit=tr[r['assignment'][tr]!=k];est.append(m.smd(z[fit[d['y'][fit]==0]],z[fit[d['y'][fit]==1]]))
 est=np.array(est);mean=est.mean(0);stability=np.mean(np.sign(est)==np.sign(mean),axis=0)
 return (mean*stability).astype('float32')

def prompt_score(fold,family):
 r=ref(fold);p=load(RT/f'cache/q1_{fold:02}.pt')['models'][family];w=p['weights'];coef=p['coef'];z=r['z'];di=direction_vector();candidate=np.where(di[None]!=0,z*di,abs(z))
 # Decomposed signed linear disease logit contribution, then absolute influence.
 # No evaluation label chooses the disease side; all people receive both channels.
 c=np.einsum('he,h->e',w[:,0],coef[:13])[None]*candidate+np.einsum('he,h->e',w[:,1],coef[13:26])[None]*(-abs(z))
 return abs(c).astype('float32')

def intervention_row(v,base,graphs,mu,person,mask,keep,strength,mode,replace=None):
 cols=torch.tensor(COLS,device='cuda');mt=torch.as_tensor(mask,device='cuda',dtype=torch.float32)
 with torch.no_grad():
  raw=m.intervene(v.feat,person,cols,mu,mt,keep,strength) if replace is None else replace
  out,_,_=forward(v,raw,graphs if mode=='fixed' else None)
  pred=int(base[person].argmax());y=int(data()['y'][person]);before=base[person];after=out[person]
  return dict(drop_true=float(before[y]-after[y]),drop_predicted=float(before[pred]-after[pred]),prediction_flip=int(after.argmax()!=pred),probability_change=float((before.exp()-after.exp()).abs().mean()),other_subject_max_logp_change=float((out[torch.arange(871,device='cuda')!=person]-base[torch.arange(871,device='cuda')!=person]).abs().max()),original_correct=bool(pred==y),original_confidence=float(before.exp().max()),n_replaced=int(mask.sum() if not keep else 39-mask.sum()))

def teacher(fold,v,base,graphs):
 path=RT/f'cache/teacher_{fold:02}.pt'
 if path.exists():return load(path)
 r=ref(fold);people=choose_teacher(fold);mu=torch.tensor(r['mu'],device='cuda');values=np.zeros((16,39),'float32');truth=np.zeros_like(values);single_graph=np.zeros_like(values)
 start=time.monotonic()
 for j,person in enumerate(people):
  for e in range(39):
   mask=np.zeros(39,'float32');mask[e]=1
   a=intervention_row(v,base,graphs,mu,int(person),mask,False,1,'live');values[j,e]=a['drop_predicted'];truth[j,e]=a['drop_true']
  if j in (7,15):print('TEACHER',fold,j+1,'/16',round(time.monotonic()-start,2),flush=True)
 value=dict(people=people,impact=values,true_impact=truth,subject_ids=data()['subjects']['subject_SUB_ID'].to_numpy()[people],seconds=time.monotonic()-start,eval_disjoint_outer_test=True)
 save(path,value);return value

def q2(fold):
 d=data();tr,te=d['folds'][fold-1];r=ref(fold);v,base,graphs=frozen(fold);before=m.trunksha(v);tt=teacher(fold,v,base,graphs);mu=torch.tensor(r['mu'],device='cuda');disc=discrim(fold);ef=m.edge_features(r['z'],d['x'],EDGES,direction_vector(),disc)
 rr=Ridge(alpha=10,solver='svd').fit(ef[tt['people']].reshape(-1,46),tt['impact'].reshape(-1));s3=rr.predict(ef.reshape(-1,46)).reshape(871,39)
 scores={'S1_ABSZ':abs(r['z']),'S2_TRAIN_DISCRIM':abs(r['z'])*abs(disc),'S3_TRAIN_B0_TEACHER':s3}
 for fam in ('P0','P1','P2','P3'):scores['S4_'+fam]=prompt_score(fold,fam)
 q1cache=load(RT/f'cache/q1_{fold:02}.pt');selected='S4_'+q1cache['selected_family']
 # Exact same random masks are reused in every comparator, independent of labels.
 for seed in CFG['q2']['random_mask_seeds']:scores[f'S0_RANDOM_{seed}']=np.random.default_rng(seed+fold).random((871,39))
 statepath=RT/f'checkpoints/q2_{fold:02}.pt'
 if statepath.exists():state=load(statepath);rows=state['rows'];next_subject=state['next_subject']
 else:rows=[];next_subject=0
 for j in range(next_subject,len(te)):
  person=int(te[j])
  for name,score in scores.items():
   for k in (4,8):
    mask=np.zeros(39,'float32');idx=np.argsort(-score[person],kind='stable')[:k];mask[idx]=1
    # Primary all-node interventions, then two fixed sensitivity operators.
    for mode in ('live','fixed'):
     a=intervention_row(v,base,graphs,mu,person,mask,False,1,mode)
     rows.append(dict(fold=fold,subject_row=person,score=name,k=k,strength=1.,graph=mode,operation='delete',replacement='TD_MU',selected_style=selected,mask_indices=[EDGES[i] for i in idx],score_absz_spearman=correlation(score[person],abs(r['z'][person])),**a))
    # Half-strength and keep are essential for every score, not just the winner.
    for strength,keep in ((.5,False),(1.,True)):
     a=intervention_row(v,base,graphs,mu,person,mask,keep,strength,'live')
     rows.append(dict(fold=fold,subject_row=person,score=name,k=k,strength=strength,graph='live',operation='keep' if keep else 'delete',replacement='TD_MU',selected_style=selected,mask_indices=[EDGES[i] for i in idx],**a))
    if k==8 and (name in ('S1_ABSZ','S2_TRAIN_DISCRIM',selected) or name.startswith('S0_')):
     td=tr[d['y'][tr]==1];same=td[d['site'][td]==d['site'][person]];pool=same if len(same) else td
     donor=sorted(pool,key=lambda p:(abs(int(d['age'][p])-int(d['age'][person]))+int(d['sex'][p]!=d['sex'][person]),int(p)))[0]
     raw=v.feat.clone();col=torch.tensor(COLS[idx],device='cuda');raw[person,col]=v.feat[int(donor),col]
     a=intervention_row(v,base,graphs,mu,person,mask,False,1,'live',raw)
     rows.append(dict(fold=fold,subject_row=person,score=name,k=8,strength=1.,graph='live',operation='delete',replacement='MATCHED_TRAIN_TD_DONOR',donor_row=int(donor),**a))
  if (j+1)%8==0 or j+1==len(te):
   save(statepath,dict(rows=rows,next_subject=j+1,rng=m.rng(),config_sha=m.sha(HERE/'config.json')))
   print('Q2',fold,j+1,'/',len(te),flush=True)
 assert m.trunksha(v)==before
 save(RT/f'cache/scores_{fold:02}.pt',dict(scores=scores,discrimination=disc,edge_features=ef,teacher_coef=rr.coef_,teacher_intercept=rr.intercept_,selected=selected))
 cw(RT/f'tasks/q2_{fold:02}.csv',rows)
 print('Q2',fold,'COMPLETE',len(rows),flush=True)

def gradnorm(gg):return float(torch.sqrt(sum((g.detach()**2).sum() for g in gg)))
def gradcos(a,b):
 den=gradnorm(a)*gradnorm(b)
 return float(sum((u.detach()*v.detach()).sum() for u,v in zip(a,b))/den) if den>1e-14 else 0.
def q3(fold):
 d=data();tr,te=d['folds'][fold-1];r=ref(fold);v,base,graphs=frozen(fold);trunk_before=m.trunksha(v);tt=load(RT/f'cache/teacher_{fold:02}.pt');people=tt['people'];mu=torch.tensor(r['mu'],device='cuda');cols=torch.tensor(COLS,device='cuda');features=m.edge_features(r['z'],d['x'],EDGES,direction_vector(),np.ones(39,'float32'))
 teacher_y=tt['impact'].copy();correct=(base.argmax(1).cpu().numpy()==d['y']);teacher_y[~correct[people]]=0.
 parts=list(StratifiedKFold(5,shuffle=True,random_state=42420+fold).split(people,d['y'][people]));rows=[];gradrows=[]
 refdrop={}
 for person in people:
  pi=int(person);pred=int(base[pi].argmax());zmask=np.zeros(39,'float32');zmask[np.argsort(-abs(r['z'][pi]),kind='stable')[:8]]=1
  rng=np.random.default_rng(42011+fold*1000+pi);rmask=np.zeros(39,'float32');rmask[rng.choice(39,8,replace=False)]=1
  refdrop[pi]=dict(random=intervention_row(v,base,graphs,mu,pi,rmask,False,1,'live')['drop_predicted'],topz=intervention_row(v,base,graphs,mu,pi,zmask,False,1,'live')['drop_predicted'])
 for group in CFG['q3']['groups']:
  for inner,(aa,bb) in enumerate(parts):
   path=RT/f'checkpoints/q3_{fold:02}_{group}_{inner}.pt';fit=people[aa];valid=people[bb];center=features[fit].mean((0,1));scale=np.maximum(features[fit].std((0,1)),.1);ft=torch.tensor((features-center)/scale,device='cuda');target=teacher_y[aa];tscale=max(float(target.std()),.01);tar=torch.tensor(target/tscale,device='cuda')
   with torch.random.fork_rng(devices=[0]):torch.manual_seed(420300+fold*10+inner);sel=m.Selector().cuda()
   opt=torch.optim.Adam(sel.parameters(),lr=.01,weight_decay=0);params=list(sel.parameters());start=0;logs=[];probe_ids=[]
   if path.exists():
    state=load(path);assert state['config_sha']==m.sha(HERE/'config.json');sel.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];logs=state['logs'];probe_ids=state['probe_ids'];m.restore(state['rng'])
   for step in range(start,30):
    opt.zero_grad();score=sel(ft[fit]);tloss=torch.nn.functional.mse_loss(score,tar);gt=torch.autograd.grad(tloss,params,retain_graph=True)
    cf=torch.tensor(0.,device='cuda');cfcount=0;rawmask_grad=0.
    if group!='C0' and (step+1)%5==0:
     for count in range(2):
      ix=((step//5)*2+count)%len(fit);person=int(fit[ix]);probe_ids.append(person)
      if not correct[person] or float(base[person].exp().max())<.6:continue
      scoreone=sel(ft[person]);mask=m.topmask(scoreone,8,True);mask.retain_grad();yp=int(d['y'][person]);keep='C1'==group or group in ('C3','C3_FIXED_DIAGNOSTIC');delete='C2'==group or group in ('C3','C3_FIXED_DIAGNOSTIC')
      if keep:
       kept=m.intervene(v.feat,person,cols,mu,mask,True,1.);pk,_,_=forward(v,kept);cf=cf+.10*torch.nn.functional.kl_div(pk[person],base[person].exp(),reduction='sum')
      if delete:
       removed=m.intervene(v.feat,person,cols,mu,mask,False,1.);pd,_,_=forward(v,removed);drop=base[person,int(base[person].argmax())]-pd[person,int(base[person].argmax())];cf=cf+.05*torch.nn.functional.softplus(torch.as_tensor(.05+max(refdrop[person].values()),device='cuda')-drop)
      cfcount+=1
     if cfcount:cf=cf/cfcount
    if cfcount:gc=torch.autograd.grad(cf,params,retain_graph=False)
    else:gc=tuple(torch.zeros_like(p) for p in params)
    nt,nc=gradnorm(gt),gradnorm(gc);factor=min(1.,.25*nt/max(nc,1e-12)) if group!='C3_FIXED_DIAGNOSTIC' else 1.
    if not cfcount:factor=0.
    cosine=gradcos(gt,gc);weightnorm=float(torch.sqrt(sum((p.detach()**2).sum() for p in params)))
    beforew=[p.detach().clone() for p in params]
    for p,a,b in zip(params,gt,gc):p.grad=a+factor*b
    assert all(torch.isfinite(p.grad).all() for p in params);opt.step();update=float(torch.sqrt(sum(((p.detach()-b)**2).sum() for p,b in zip(params,beforew))))
    logs.append(dict(fold=fold,group=group,inner_fold=inner,step=step+1,kind='GRADIENT',teacher_loss=float(tloss.detach()),cf_loss=float(cf.detach()),teacher_gradient=nt,cf_gradient=nc,cf_to_teacher_ratio=nc/max(nt,1e-12),gradient_cosine=cosine,cf_scale=factor,applied_ratio=factor*nc/max(nt,1e-12),update_norm=update,weight_norm=weightnorm,update_to_weight=update/max(weightnorm,1e-6),probes=2 if group!='C0' and (step+1)%5==0 else 0,valid_probes=cfcount))
    if (step+1)%5==0:
     save(path,dict(model=sel.state_dict(),optimizer=opt.state_dict(),step=step+1,logs=logs,probe_ids=probe_ids,rng=m.rng(),config_sha=m.sha(HERE/'config.json'),fit=fit,valid=valid))
   gradrows+=logs
   sel.eval()
   with torch.no_grad():sc=sel(ft[valid]).cpu().numpy()
   for j,person in enumerate(valid):
    mask=np.zeros(39,'float32');ix=np.argsort(-sc[j],kind='stable')[:8];mask[ix]=1;truthrank=correlation(sc[j],tt['true_impact'][bb[j]])
    for strength,mode,keep in ((1.,'live',False),(.5,'live',False),(1.,'fixed',False),(1.,'live',True)):
     out=intervention_row(v,base,graphs,mu,int(person),mask,keep,strength,mode)
     rows.append(dict(fold=fold,inner_fold=inner,group=group,kind='OOF_FIDELITY',subject_row=int(person),scope='OUTER_TRAIN_TEACHER_HELDOUT',k=8,strength=strength,graph=mode,operation='keep' if keep else 'delete',direction_teacher_rank_correlation=truthrank,selected_mean_single_edge_true_impact=float(tt['true_impact'][bb[j],ix].mean()),random_drop=refdrop[int(person)]['random'],topz_drop=refdrop[int(person)]['topz'],unique_cf_people=len(set(probe_ids)),cf_probe_total=len(probe_ids),mask_indices=[EDGES[i] for i in ix],**out))
   print('Q3',fold,group,'inner',inner+1,'/5',flush=True)
 assert m.trunksha(v)==trunk_before
 cw(RT/f'tasks/q3_{fold:02}.csv',rows+gradrows)
 print('Q3',fold,'COMPLETE',len(rows),'heldout interventions',flush=True)

def qc_fields():
 d=data();source='https://preprocessed-connectomes-project.org/abide/quality_assessment.html';meanings={'anat_cnr':'gray-white contrast divided by background noise','anat_efc':'entropy focus criterion; historical definition ambiguous for monotone interpretation','anat_fber':'foreground/background energy ratio','anat_fwhm':'spatial smoothness, no universal monotone quality ordering','anat_qi1':'fraction of artifact-contaminated background voxels','anat_snr':'gray-matter mean/background-noise standard deviation','func_efc':'entropy of mean functional image; excluded ambiguity','func_fber':'mean functional foreground/background energy; not selected','func_fwhm':'mean functional spatial smoothness; excluded','func_dvars':'standardized temporal difference RMS','func_outlier':'mean outlier fraction per volume','func_quality':'distance to median volume; not selected to avoid correlated duplicated QC','func_mean_fd':'mean framewise displacement','func_num_fd':'number of frames above FD threshold; duration dependent','func_perc_fd':'percent frames above FD threshold; redundant with mean FD','func_gsr':'ghost-to-signal ratio, NOT global-signal-regression on/off'}
 directions={**CFG['q4']['anat_include'],'func_dvars':1,'func_outlier':1,'func_mean_fd':1}
 rows=[]
 for j,name in enumerate(d['columns'][48:64],48):rows.append(dict(kind='QC_DICTIONARY',column=j,name=name,meaning=meanings[name],risk_direction=directions.get(name,0),use='INCLUDE' if name in directions else 'EXCLUDE',scale='historical global StandardScaler; sign verified by positive affine fit to matched raw CSV',source=source))
 return rows

def reliability(fold):
 d=data();tr,_=d['folds'][fold-1];x=d['x'];risk={}
 for role,names in [('R2',dict.fromkeys(CFG['q4']['func_include'],1)),('R3',CFG['q4']['anat_include'])]:
  vv=[]
  for name,direction in names.items():
   col=d['columns'].index(name);a=x[:,col]*direction;cen=np.median(a[tr]);scale=max(1.4826*np.median(abs(a[tr]-cen)),.1);vv.append((a-cen)/scale)
  v=np.mean(vv,axis=0);risk[role]=v
 td=tr[d['y'][tr]==1];unc=np.zeros(871)
 for i in range(871):
  # A sparse-cell precision proxy, not an ASD support score.
  nsite=np.sum(d['site'][td]==d['site'][i]);njoint=np.sum((d['site'][td]==d['site'][i])&(d['age'][td]==d['age'][i])&(d['sex'][td]==d['sex'][i]));unc[i]=1/np.sqrt(1+njoint+.1*nsite)
 risk['R1']=unc
 rel={k:1-.5*np.searchsorted(np.sort(v[tr]),v,side='right')/len(tr) for k,v in risk.items()};rel['R0']=np.ones(871)
 return risk,rel

def q4(fold):
 d=data();tr,te=d['folds'][fold-1];r=ref(fold);v,base,graphs=frozen(fold);tt=load(RT/f'cache/teacher_{fold:02}.pt');people=tt['people'];mu=torch.tensor(r['mu'],device='cuda');risk,rels=reliability(fold);rows=[]
 # Independent proxy: model sensitivity to deterministic zero-mean FC noise;
 # labels play no role in either risk calibration or noise generation.
 instability=np.zeros(871);fcnoise=np.zeros(871);stress=[]
 for person in people:
  person=int(person);g=np.random.default_rng(424000+person+fold*1000);noise=g.normal(size=39).astype('float32');noise/=max(float(np.std(noise)),1e-6)
  for level in (0.,.25,.5,1.):
   raw=v.feat.clone();raw[person,torch.tensor(COLS,device='cuda')]+=torch.tensor(level*r['sigma']*noise,device='cuda')
   with torch.no_grad():out,_,_=forward(v,raw)
   change=float((out[person]-base[person]).abs().mean());confidence_gain=float(out[person].exp().max()-base[person].exp().max());stress.append(dict(fold=fold,kind='FC_ONLY_STRESS',subject_row=person,level=level,abs_logp_change=change,confidence_gain=confidence_gain,quality_metrics_changed=False,interpretation='computational pressure; QC gate cannot infer physical motion from unchanged QC'))
   if level==.5:instability[person]=change
  fcnoise[person]=float(np.mean(abs(r['z'][person])))
 cov=d['x'][:,:48];pinst=m.partial(instability,cov,people);pz=m.partial(abs(r['z']).mean(1),cov,tr)
 for role in ('R1','R2','R3'):
  prisk=m.partial(risk[role],cov,tr);rho_fc=correlation(prisk[tr],pz[tr]);rho_inst=correlation(prisk[people],pinst[people]);random=np.random.default_rng(42420+fold).permutation(prisk[tr]);null_fc=correlation(random,pz[tr])
  rows.append(dict(fold=fold,role=role,kind='PROXY',partial_risk_absz_rho=rho_fc,partial_risk_instability_rho=rho_inst,reversed_absz_rho=-rho_fc,reversed_instability_rho=-rho_inst,random_absz_rho=null_fc,n_proxy_people=len(people),n_train=len(tr),support=bool(rho_fc>.05 and rho_inst>.05 and rho_fc>null_fc),scope='TRAIN_ONLY',site_confound_caution=True))
 for role in ('R0','R1','R2','R3'):
  rel=rels[role];threshold=float(np.quantile(rel[tr],.1));rng=np.random.default_rng(424200+fold);perm=tr[rng.permutation(len(tr))];random_rel=rel.copy();random_rel[tr]=rel[perm]
  if role=='R1':
   shuffled_sites=d['site'].copy();shuffled_sites[tr]=d['site'][perm];perturb=rel.copy();td=tr[d['y'][tr]==1]
   for person in tr:
    ns=np.sum(shuffled_sites[td]==shuffled_sites[person]);nj=np.sum((shuffled_sites[td]==shuffled_sites[person])&(d['age'][td]==d['age'][person])&(d['sex'][td]==d['sex'][person]));u=1/np.sqrt(1+nj+.1*ns);perturb[person]=1-.5*np.searchsorted(np.sort(risk[role][tr]),u,side='right')/len(tr)
  else:perturb=random_rel
  for control,rr in [('positive',rel),('constant_one',np.ones(871)),('reversed',1.5-rel),('random_rank',random_rel),('site_permuted',perturb)]:
   for person in people:
    person=int(person);k=min(8,max(0,int(np.floor(8*rr[person]))));abstain=bool(rr[person]<threshold)
    if abstain:k=0
    score=abs(r['z'][person]);idx=np.argsort(-score,kind='stable')[:k];mask=np.zeros(39,'float32');mask[idx]=1
    # No new classifier/residual: decision-path diagnostic uses a gated keep.
    if k:
     a=intervention_row(v,base,graphs,mu,person,mask,True,1,'live')
    else:a=dict(drop_true=0.,drop_predicted=0.,prediction_flip=0,probability_change=0.,original_correct=bool(base[person].argmax()==d['y'][person]))
    rows.append(dict(fold=fold,role=role,control=control,kind='GATE_DEPENDENCE',subject_row=person,reliability=float(rr[person]),selected_count=k,abstain=abstain,scope='TRAIN_ONLY',empty_policy='abstain evidence, original B0 prediction untouched',**a))
 rows.append(dict(fold=fold,role='R4',kind='COMBINATION_GATE',status='DEFERRED_UNTIL_GLOBAL_TRAIN_ROLE_GATE'))
 cw(RT/f'tasks/q4_{fold:02}.csv',rows+stress)
 print('Q4',fold,'COMPLETE',flush=True)

def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(u,v) for u,v in zip(a,b))
 return a==b

def preflight():
 start=time.monotonic();d=data();r=ref(1);tr,te=d['folds'][0];tests={}
 v=agent(1,True);c=runtime()
 # Construct an independent reference using E32's exact source patch function.
 import ast,types
 tree=ast.parse((ROOT/'experiment/experiment32_disease_contrastive_evidence_routing/integration.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='patch_network');env={'ast':ast};exec(compile(ast.Module(body=[node],type_ignores=[]),'<E32_patch_network>','exec'),env)
 src=env['patch_network']((ROOT/'network.py').read_text());refmod=types.ModuleType('network_reference42');exec(compile(src,'<E32_exact_reference>','exec'),refmod.__dict__);refmod._prompt_text_features=c['network']._prompt_text_features
 oldpln,oldgfr=c['model'].PLN,c['model'].GFR
 try:
  c['model'].PLN=refmod.PLN;c['model'].GFR=refmod.GFR
  other=agent(1,True)
 finally:c['model'].PLN=oldpln;c['model'].GFR=oldgfr
 other.ModalFusion.dcer32_config='B0'
 def step(a,state):
  m.restore(state);a.MessagePassing0.train();a.ModalFusion.train();a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad();p,gs,gl=forward(a,loss=True);cl=c['utils'].ClsLoss(p,a.targ,a.trn_idx,a.weight);(cl+gl).backward()
  grads={(key,n):None if par.grad is None else par.grad.detach().clone() for key,mod in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)] for n,par in mod.named_parameters()};a.optimizer_MP0.step();a.optimizer_MF.step()
  return p.detach(),[g.detach() for g in gs],cl.detach(),gl.detach(),grads,m.trunksha(a),copy.deepcopy(a.optimizer_MP0.state_dict()),copy.deepcopy(a.optimizer_MF.state_dict()),m.rng()
 for k in range(2):
  state=m.rng();a=step(v,state);b=step(other,state);assert equal(a,b),'E32 source parity update mismatch'
 tests['B0_E32_forward_graphloss_classloss_gradients_two_Adam_steps']='BITWISE_EXACT'
 # Precise mid-fold restart (two previous steps, third update), both Adam states.
 ss=dict(gfr=copy.deepcopy(v.MessagePassing0.state_dict()),pln=copy.deepcopy(v.ModalFusion.state_dict()),mp=copy.deepcopy(v.optimizer_MP0.state_dict()),mf=copy.deepcopy(v.optimizer_MF.state_dict()),rng=m.rng());save(RT/'checkpoints/preflight_resume.pt',ss);left=step(v,ss['rng']);ss=load(RT/'checkpoints/preflight_resume.pt');v.MessagePassing0.load_state_dict(ss['gfr']);v.ModalFusion.load_state_dict(ss['pln']);v.optimizer_MP0.load_state_dict(ss['mp']);v.optimizer_MF.load_state_dict(ss['mf']);right=step(v,ss['rng']);assert equal(left,right);tests['resume_third_update']='BITWISE_EXACT'
 del v,other;v,base,graphs=frozen(1);trunk_before=m.trunksha(v)
 yp=v.targ.clone();v.targ[te]=v.targ[torch.tensor(te[::-1].copy(),device='cuda')]
 with torch.no_grad():p2,_,_=forward(v)
 assert torch.equal(base,p2);v.targ=yp;tests['test_label_inference']='BITWISE_INVARIANT'
 yy=d['y'].copy();yy[te]=1-yy[te];r2=m.reference(d['x'],yy,tr,te,EDGES,1,assignment=r['assignment']);assert np.array_equal(r['mu'],r2['mu']) and np.array_equal(r['z'],r2['z']);tests['test_label_reference']='BITWISE_INVARIANT'
 person=int(tr[0]);yy=d['y'].copy();yy[person]=1-yy[person];r3=m.reference(d['x'],yy,tr,te,EDGES,1,assignment=r['assignment']);assert np.array_equal(r['mu'][person],r3['mu'][person]);k=r['assignment'][person];assert np.array_equal(r['fits'][k],r3['fits'][k]);tests['own_training_label_reference_path']='OWN_FIT_IDS_AND_MU_IDENTICAL; TD scale population may change as intended training supervision'
 assert all(not set(fit)&set(te) for fit in r['fits']);assert all(person not in r['fits'][r['assignment'][person]] for person in tr);tests['crossfit_exclusion']='ALL_ASD_TD_TRAIN_SELF_EXCLUDED'
 # Canonical identity and wrong-association isolation.
 hyp=hypotheses();before=[r[k].copy() for k in ('mu','z','sigma')];wrong=m.wrong_groups(hyp);assert all(set(hyp[i]['edges']).isdisjoint(hyp[j]['edges']) and len(hyp[i]['edges'])==len(hyp[j]['edges']) for i,j in enumerate(wrong));assert all(np.array_equal(a,r[k]) for a,k in zip(before,('mu','z','sigma')));tests['wrong_pair_only_membership']='PASS'
 bank=load(RT/'cache/text_bank.pt');geom={}
 for fam in ('P0','P1','P2','P3'):
  tx=m.norm(bank[fam].numpy());assert np.isfinite(tx).all();geom[fam]=dict(mean_pair_cos=float(np.sum(tx[:,0]*tx[:,1],-1).mean()),min_pair_distance=float(np.linalg.norm(tx[:,0]-tx[:,1],axis=-1).min()))
  assert geom[fam]['min_pair_distance']>.01
 w=m.text_weights(bank['P0'].numpy(),bank['anatomy'].numpy(),hyp,EDGES);sw=m.text_weights(bank['P0'].numpy(),bank['anatomy'].numpy(),hyp,EDGES,'SWAP_ASD_TD');assert np.max(abs(w-sw))>1e-5
 real=bank['P0'].numpy();g=np.random.default_rng(420008);anon=real[...,g.permutation(1024)]*g.choice([-1.,1.],1024);assert np.max(abs(m.norm(real.reshape(26,-1))@m.norm(real.reshape(26,-1)).T-m.norm(anon.reshape(26,-1))@m.norm(anon.reshape(26,-1)).T))<2e-6;tests['prompt_geometry']=geom
 mask=torch.zeros(39,device='cuda');mask[[0,1,2,3,4,5,6,7]]=1;mu=torch.tensor(r['mu'],device='cuda');cols=torch.tensor(COLS,device='cuda');raw=m.intervene(v.feat,person,cols,mu,mask,False,1)
 changed=(raw!=v.feat);assert not changed[torch.arange(871,device='cuda')!=person].any();allowed=torch.zeros_like(changed);allowed[person,cols[:8]]=True;assert not (changed&~allowed).any();assert torch.equal(raw[person,cols[:8]],mu[person,:8]);assert torch.equal(m.intervene(v.feat,person,cols,mu,torch.zeros_like(mask),False),v.feat)
 with torch.no_grad():changedp,_,_=forward(v,raw);samep,_,_=forward(v,fixed=graphs)
 assert torch.equal(base,samep) and not torch.equal(base[person],changedp[person]);tests['single_raw_FC_and_fixed_graph']='EXACT_SCOPE_AND_LIVE_RESPONSE'
 # Finite difference is checked on the CONTINUOUS relaxation. The hard mask
 # is piecewise constant; its true finite difference must not be mislabeled STE.
 scores=torch.linspace(-.2,.2,39,device='cuda',requires_grad=True);soft=torch.sigmoid(scores);rawsoft=v.feat.clone();rawsoft[person,cols]=mu[person]+(v.feat[person,cols]-mu[person])*soft;lp,_,_=forward(v,rawsoft,graphs);loss=lp[person,int(base[person].argmax())];gs=torch.autograd.grad(loss,scores)[0];e=int(gs.abs().argmax());eps=.001
 finite=[]
 for delta in (-eps,eps):
  ss=scores.detach().clone();ss[e]+=delta;xx=v.feat.clone();xx[person,cols]=mu[person]+(v.feat[person,cols]-mu[person])*torch.sigmoid(ss)
  with torch.no_grad():qq,_,_=forward(v,xx,graphs);finite.append(float(qq[person,int(base[person].argmax())]))
 fd=(finite[1]-finite[0])/(2*eps);aut=float(gs[e]);assert abs(fd-aut)<max(.003,.15*abs(aut)) and abs(aut)>1e-8
 tests['continuous_mask_finite_difference']=dict(autograd=aut,finite_difference=fd,relative_error=abs(fd-aut)/max(abs(aut),1e-8),hard_STE_is_surrogate=True)
 ss=torch.linspace(-1,1,39,device='cuda',requires_grad=True);mask=m.topmask(ss,8,True);raw=m.intervene(v.feat,person,cols,mu,mask,False,1);pp,_,_=forward(v,raw);g=torch.autograd.grad(pp[person,int(base[person].argmax())],ss)[0];assert torch.isfinite(g).all() and g.norm()>0;assert all(par.grad is None for mod in (v.MessagePassing0,v.ModalFusion) for par in mod.parameters());tests['STE_CF_mask_gradient']=float(g.norm());assert m.trunksha(v)==trunk_before
 assert torch.equal((torch.randn(39,4,device='cuda')*torch.zeros(39,1,device='cuda')).sum(0),torch.zeros(4,device='cuda'));tests['empty_selection_summary']='EXACT_ZERO_NO_CLASSIFIER_RESIDUAL_EXISTS'
 # Raw QC columns must be positively affine-related to their measured CSV values.
 rawcsv=ROOT/'dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv';frame=pd.read_csv(rawcsv,low_memory=False).set_index('SUB_ID');ids=d['subjects']['subject_SUB_ID'].to_numpy();assert len(set(ids))==871 and set(ids)<=set(frame.index);qc=[]
 for col in range(48,64):
  name=d['columns'][col];a=pd.to_numeric(frame.loc[ids,name],errors='coerce').to_numpy();b=d['x'][:,col].astype('float64');valid=np.isfinite(a)&np.isfinite(b)&(a!=-9999);coef=np.polyfit(a[valid],b[valid],1);error=float(np.max(abs(np.polyval(coef,a[valid])-b[valid])));rho=correlation(a[valid],b[valid]);assert coef[0]>0 and error<1e-5 and rho>.99999;qc.append(dict(name=name,column=col,slope=float(coef[0]),max_error=error,rho=rho))
 tests['QC_direction_and_identity']=qc;tests['root_protected']='UNCHANGED'
 startcf=time.monotonic()
 for _ in range(4):
  with torch.no_grad():forward(v,raw)
 tests['benchmark_seconds_per_forward']=(time.monotonic()-startcf)/4;tests['peak_allocated_bytes']=torch.cuda.max_memory_allocated();tests['elapsed_seconds']=time.monotonic()-start
 jw(RT/'preflight.json',dict(status='PASS',tests=tests));print('PREFLIGHT PASS',json.dumps({k:tests[k] for k in ('benchmark_seconds_per_forward','peak_allocated_bytes','elapsed_seconds')}),flush=True)

def artifact_signature():return m.hjson({k:m.sha(HERE/k) for k in ('config.json','prompt_candidates.json','mechanism.py','run.py','runtime/cache/text_bank.pt')})
def verify_protected():
 for row in jr(RT/'source_audit.json'):
  if row['status']=='READ':assert m.sha(row['path'])==row['sha256'],'Protected input changed: '+row['path']

def worker(phase,fold):
 lock=(RT/f'tasks/{phase}_{fold:02}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);statepath=RT/f'tasks/{phase}_{fold:02}.json';signature=artifact_signature()
 if statepath.exists():
  s=jr(statepath);assert s['signature']==signature,'Code/config changed after registration'
  if s['status']=='COMPLETE':
   for p,sh in s['outputs'].items():assert m.sha(p)==sh
   print('SKIP',phase,fold,flush=True);return
 started=time.time();jw(statepath,dict(status='RUNNING',signature=signature,phase=phase,fold=fold,gpu=int(os.environ.get('E42_PHYSICAL_GPU','0')),started=stamp()))
 try:
  globals()[phase](fold)
  patterns={'q1':[f'q1_{fold:02}_probe.csv',f'q1_{fold:02}_audit.csv'],'q2':[f'q2_{fold:02}.csv'],'q3':[f'q3_{fold:02}.csv'],'q4':[f'q4_{fold:02}.csv']}[phase]
  files={str(RT/'tasks'/p):m.sha(RT/'tasks'/p) for p in patterns}
  jw(statepath,dict(status='COMPLETE',signature=signature,phase=phase,fold=fold,gpu=int(os.environ.get('E42_PHYSICAL_GPU','0')),seconds=time.time()-started,completed=stamp(),outputs=files,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
 except BaseException as e:
  jw(statepath,dict(status='FAIL',signature=signature,phase=phase,fold=fold,error=repr(e),traceback=traceback.format_exc()));raise

def run_stage(phase,folds=None):
 pending=list(range(1,11)) if folds is None else list(folds);running=[];logs={};started=time.monotonic()
 while pending or running:
  for gpu in (0,1):
   active=sum(g==gpu for _,g,_,_ in running)
   for fold in [f for f in pending if int(f>5)==gpu][:max(0,2-active)]:
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),E42_PHYSICAL_GPU=str(gpu),PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(RT/'cache'/f'mpl_{gpu}_{fold}'))
    log=(RT/f'logs/{phase}_{fold:02}.log').open('a');p=subprocess.Popen([sys.executable,str(HERE/'run.py'),'worker','--phase',phase,'--fold',str(fold)],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT);running.append((p,gpu,fold,log));pending.remove(fold)
  completed=[]
  for p,g,f,log in running:
   if p.poll() is not None:
    log.close();completed.append((p,g,f,log));print(phase,'fold',f,'GPU',g,'exit',p.returncode,flush=True)
    if p.returncode:
     for pp,_,_,_ in running:
      if pp.poll() is None:pp.terminate()
     raise RuntimeError(f'{phase} fold {f} failed; see runtime/logs/{phase}_{f:02}.log')
  running=[x for x in running if x not in completed]
  if pending or running:time.sleep(1)
 event('STAGE_COMPLETE',phase=phase,seconds=time.monotonic()-started)

def aggregate():
 for phase,name in [('q1','prompt_probe'),('q2','score_fidelity'),('q3','cf_validation'),('q4','reliability_audit')]:
  pattern='q1_*_probe.csv' if phase=='q1' else f'{phase}_??.csv';rows=[]
  for p in sorted((RT/'tasks').glob(pattern)):rows+=cr(p)
  if phase=='q4':rows=qc_fields()+rows
  cw(RES/f'{name}.csv',rows)
 rows=cr(RES/'prompt_audit.csv');rows=[r for r in rows if r.get('kind','')!='TRAIN_EFFECT']
 for p in sorted((RT/'tasks').glob('q1_*_audit.csv')):rows+=cr(p)
 cw(RES/'prompt_audit.csv',rows)

def summary_decisions():
 a=pd.read_csv(RES/'prompt_audit.csv');probe=pd.read_csv(RES/'prompt_probe.csv');f=pd.read_csv(RES/'score_fidelity.csv');c=pd.read_csv(RES/'cf_validation.csv');q=pd.read_csv(RES/'reliability_audit.csv');result={};summ={}
 aa=a[a.kind=='TRAIN_EFFECT'];hypstat=aa.groupby('hypothesis').agg(pass_folds=('status',lambda s:int((s=='PASS').sum())),effect=('effect_smd','mean'),stability=('inner_direction_stability','mean')).reset_index();credible=int((hypstat.pass_folds>=7).sum());summ['q1_hypotheses']=hypstat.to_dict('records')
 oof=probe[probe.inner_fold=='OOF_ALL'];rows=[]
 for fam in ('P0','P1','P2','P3'):
  real=oof[(oof.family==fam)&(oof.control=='REAL')].sort_values('fold');controls=[];passes=True
  for var in ('NEUTRAL','SWAP_ASD_TD','DERANGED_HYPOTHESIS','GRAM_MATCHED_ANONYMOUS','NUMERIC_ONLY'):
   ctrl=oof[(oof.family==('ALL' if var=='NUMERIC_ONLY' else fam))&(oof.control==var)].sort_values('fold');assert len(real)==len(ctrl)==10
   da=100*(real.acc.to_numpy()-ctrl.acc.to_numpy());du=100*(real.hard_auc.to_numpy()-ctrl.hard_auc.to_numpy());controls.append(dict(control=var,delta_acc_pp=float(da.mean()),delta_auc_pp=float(du.mean()),acc_wtl=[int(sum(da>1e-9)),int(sum(abs(da)<=1e-9)),int(sum(da<-1e-9))],auc_wtl=[int(sum(du>1e-9)),int(sum(abs(du)<=1e-9)),int(sum(du<-1e-9))]))
   passes &= bool(da.mean()>=.25 and du.mean()>=.25 and sum((da>0)&(du>0))>=7)
  rows.append(dict(family=fam,acc=float(real.acc.mean()*100),auc=float(real.hard_auc.mean()*100),spe=float(real.spe_td.mean()*100),sen=float(real.sen_asd.mean()*100),passes_controls=passes,controls=controls))
 summ['q1_probes']=rows;result['Q1']='GO' if any(r['passes_controls'] for r in rows) and credible>=7 else 'NO-GO';summ['q1_credible_hypotheses']=credible
 # Q2 uses each fold's style frozen solely from training OOF.
 ff=f[(f.k==8)&(f.strength==1)&(f.operation=='delete')&(f.replacement=='TD_MU')]
 score_rows=[];q2passes=[]
 for mode in ('live','fixed'):
  part=ff[ff.graph==mode];sel=part[part.score==part.selected_style];group=part.groupby(['fold','score']).drop_true.mean().unstack();chosen=sel.groupby('fold').drop_true.mean();random=part[part.score.str.startswith('S0_')].groupby('fold').drop_true.mean();topz=group['S1_ABSZ'];d1=chosen-random;d2=chosen-topz
  score_rows.append(dict(graph=mode,selected=float(chosen.mean()),random=float(random.mean()),topz=float(topz.mean()),adv_random=float(d1.mean()),adv_topz=float(d2.mean()),wins_random=int((d1>0).sum()),wins_topz=int((d2>0).sum()),per_fold={str(i):dict(selected=float(chosen.loc[i]),random=float(random.loc[i]),topz=float(topz.loc[i])) for i in chosen.index}))
  q2passes.append(bool(d1.mean()>0 and d2.mean()>0 and (d1>0).sum()>=7 and (d2>0).sum()>=7))
 summ['q2_scores']=score_rows;result['Q2']='GO' if all(q2passes) else 'NO-GO'
 # Teacher validation is on held-out training participants, not on their fit scores.
 cf=c[(c.kind=='OOF_FIDELITY')&(c.operation=='delete')&(c.strength==1)&(c.graph=='live')];base=cf[cf.group=='C0'].groupby('fold')[['drop_true','direction_teacher_rank_correlation']].mean();crow=[]
 for group in CFG['q3']['groups']:
  df=cf[cf.group==group].groupby('fold')[['drop_true','direction_teacher_rank_correlation']].mean();diff=df-base;crow.append(dict(group=group,drop_true=float(df.drop_true.mean()),direction_rho=float(df.direction_teacher_rank_correlation.mean()),delta_drop_vs_C0=float(diff.drop_true.mean()),delta_direction_vs_C0=float(diff.direction_teacher_rank_correlation.mean()),wins_drop=int((diff.drop_true>0).sum()),wins_direction=int((diff.direction_teacher_rank_correlation>0).sum())))
 summ['q3_cf']=crow;result['Q3']='GO' if any(r['group'] in ('C1','C2','C3') and r['delta_drop_vs_C0']>0 and r['delta_direction_vs_C0']>0 and r['wins_drop']>=7 and r['wins_direction']>=7 for r in crow) else 'NO-GO'
 gr=c[c.kind=='GRADIENT'];summ['q3_gradients']=[]
 for group,g in gr.groupby('group'):
  vv=g[g.valid_probes>0];summ['q3_gradients'].append(dict(group=group,steps=len(g),probe_count=int(g.probes.sum()),valid_probes=int(g.valid_probes.sum()),mean_raw_cf_teacher_ratio=float(vv.cf_to_teacher_ratio.mean()) if len(vv) else 0.,mean_applied_ratio=float(vv.applied_ratio.mean()) if len(vv) else 0.,max_applied_ratio=float(vv.applied_ratio.max()) if len(vv) else 0.,mean_gradient_cosine=float(vv.gradient_cosine.mean()) if len(vv) else 0.))
 qp=q[q.kind=='PROXY'];qr=[]
 for role,g in qp.groupby('role'):
  pos=(g.partial_risk_absz_rho>.05)&(g.partial_risk_instability_rho>.05)&(g.partial_risk_absz_rho>g.random_absz_rho)
  dep=q[(q.kind=='GATE_DEPENDENCE')&(q.role==role)&(q.control=='positive')]
  action=bool((dep.selected_count!=8).any() or dep.abstain.astype(str).str.lower().eq('true').any());passed=bool(pos.sum()>=7 and action)
  qr.append(dict(role=role,partial_absz_rho=float(g.partial_risk_absz_rho.mean()),partial_instability_rho=float(g.partial_risk_instability_rho.mean()),passing_folds=int(pos.sum()),gate_action=action,mean_selected_count=float(dep.selected_count.mean()),abstain_fraction=float(dep.abstain.astype(str).str.lower().eq('true').mean()),gate_keep_flip=float(dep.prediction_flip.mean()),go=passed))
 summ['q4_roles']=qr;result['Q4']='GO' if any(r['go'] for r in qr) else 'NO-GO'
 return result,summ

def q4combine(roles):
 rows=cr(RES/'reliability_audit.csv')
 for fold in range(1,11):
  risks,rel=reliability(fold);combined=np.prod([rel[k] for k in roles],axis=0)**(1/len(roles));v,p,g=frozen(fold);r=ref(fold);mu=torch.tensor(r['mu'],device='cuda');train=data()['folds'][fold-1][0];cut=float(np.quantile(combined[train],.1))
  for person in choose_teacher(fold):
   person=int(person);k=0 if combined[person]<cut else min(8,int(np.floor(8*combined[person])));mask=np.zeros(39,'float32');mask[np.argsort(-abs(r['z'][person]))[:k]]=1
   a=intervention_row(v,p,g,mu,person,mask,True,1,'live') if k else dict(drop_true=0.,prediction_flip=0)
   rows.append(dict(fold=fold,role='R4',kind='VERIFIED_COMBINATION',subject_row=person,components=roles,reliability=float(combined[person]),selected_count=k,**a))
 cw(RES/'reliability_audit.csv',rows)

def finalize():
 aggregate();verify_protected();dec,summ=summary_decisions();info=jr(RES/'manifest.json');tasks=[jr(p) for p in sorted((RT/'tasks').glob('q?_??.json'))];assert len(tasks)==40 and all(t['status']=='COMPLETE' for t in tasks)
 roles=[r['role'] for r in summ['q4_roles'] if r['go']]
 if roles:q4combine(roles)
 rows=cr(RES/'fold_metrics.csv');assert len(rows)==10 and len({(r['group'],r['seed'],r['fold']) for r in rows})==10
 scientific='DIAGNOSTIC_COMPLETE_NO_GO' if any(v!='GO' for v in dec.values()) else 'ALL_FOUR_LINKS_GO_INTEGRATION_NOT_YET_REGISTERED'
 info.update(scientific=scientific,decisions=dec,summary=summ,completed_diagnostic_tasks=40,new_baseline_training_folds=0,validated_reused_baseline_folds=10,integrated_classifier_training_folds=0,gpu_tasks={str(g):sum(t['gpu']==g for t in tasks) for g in (0,1)},task_seconds={str(g):sum(t['seconds'] for t in tasks if t['gpu']==g) for g in (0,1)},wall_seconds=time.time()-info['started_unix'],tasks=tasks,finished=stamp(),protected_sources='UNCHANGED')
 jw(RES/'manifest.json',info);write_report(info)
 bounded_delivery()
 print(json.dumps(dict(scientific=scientific,decisions=dec,directory=str(HERE),baseline='10 validated reused / 0 newly trained',integrated=0),ensure_ascii=False),flush=True)

def write_report(info):
 hist=pd.read_csv(ROOT/'experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv');hist=hist[hist.config=='B0'];actual=pd.read_csv(RES/'fold_metrics.csv');s=info['summary'];dec=info['decisions']
 text=['# Experiment 42 — 四环节独立可证伪机制验证','',f"科研状态：**{info['scientific']}**。交付状态：**{info.get('delivery','TMP_ONLY')}**。",'','本轮没有训练HCER-v3或新的集成分类器。按授权验证并复用E40不可变B0-A十折初始状态及最佳Checkpoint，逐折重算指标；新B0训练0折，验证复用10/10。Q1–Q4各10/10独立fold任务完成，共40个诊断任务。训练内探针及选择器不冒充外层分类实验。','', '|链|结论|主要对照|下一步|','|---|---|---|---|']
 for q,name,controls in [('Q1','文献Prompt','P0–P3/数值/中性/交换/错配/Gram匿名'),('Q2','FC评分','随机/Top-abs(z)/训练判别/冻结B0导师'),('Q3','反事实约束','C0/C1/C2/C3与固定CF权重诊断'),('Q4','适用性和QC','正向/反向/随机/站点置乱/恒一')]:text.append(f'|{q} {name}|{dec[q]}|{controls}|'+('可独立继续验证' if dec[q]=='GO' else '该链尚不足以支持集成创新')+'|')
 text+=['','## 正式四指标：仅完整十折B0-A','', '单位%，mean ± population SD；没有以micro替代折均值。Hard-AUC为原始one-hot硬标签ROC-AUC。ASD=0，TD=1。','', '|来源|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---:|---:|---:|---:|']
 for name,frame in [('E32历史训练（只读参照）',hist.rename(columns={'auc':'hard_auc'})),('E42验证复用E40最佳模型（非新训练）',actual)]:
  text.append('|'+name+'|'+'|'.join(f'{frame[k].mean()*100:.4f} ± {frame[k].std(ddof=0)*100:.4f}' for k in ('acc','hard_auc','spe_td','sen_asd'))+'|')
 text+=['','逐fold混淆、best epoch、executed epochs、初始化和入口RNG SHA均与E32 B0一致；详见fold_metrics.csv。原PEN完整构造但前向跳过SPGS+RES；原始未调用参数无梯度。','', '## Q1：文献到FC与四种文本','',f"13个假说中，训练条件效应在至少7折通过预注册方向/站点/幅值规则的数量：{s['q1_credible_hypotheses']}/13。网络级来源不等于指定HO边的直接证据；全部39条的edge_specific_verified仍为false。",'','|假说|PASS折数|平均条件效应SMD|inner方向稳定率|','|---|---:|---:|---:|']
 for r in s['q1_hypotheses']:text.append(f"|{r['hypothesis']}|{r['pass_folds']}|{r['effect']:+.4f}|{r['stability']:.4f}|")
 text+=['','P0是旧文本逐字冻结；P1平衡句式；P2逐条完整HO双端短句分别编码后按假说平均，未截断；P3可反驳条件及exploratory适用背景。四族均有26个概念文本槽，P2每槽含数条边级短句。CLIP RN50固定、eval/no_grad、只编码一次；不是GPT-4 API生成记录。','', '下面是outer-train内嵌套5折OOF探针，**不是外层正式分类结果**。所有探针相同30维输入/31参数/150 Adam步；共同数值来源和软方向metadata相同。','', '|风格|train OOF ACC|train OOF Hard-AUC|SPE|SEN|胜过全部匹配控制|','|---|---:|---:|---:|---:|---|']
 for r in s['q1_probes']:text.append(f"|{r['family']}|{r['acc']:.4f}|{r['auc']:.4f}|{r['spe']:.4f}|{r['sen']:.4f}|{r['passes_controls']}|")
 text+=['','每族对控制的同fold差（pp）及ACC/AUC W/T/L：','', '|族|控制|ΔACC|ΔHard-AUC|ACC W/T/L|AUC W/T/L|','|---|---|---:|---:|---|---|']
 for r in s['q1_probes']:
  for z in r['controls']:text.append(f"|{r['family']}|{z['control']}|{z['delta_acc_pp']:+.4f}|{z['delta_auc_pp']:+.4f}|{'/'.join(map(str,z['acc_wtl']))}|{'/'.join(map(str,z['auc_wtl']))}|")
 text+=['','可识别性限制：对允许自由线性系数的低容量读出，静态交换ASD/TD特征槽可被系数重参数化吸收；交换控制接近不等于医学类别正确。四风格只是冻结候选，训练内选择不支持绝对“最优Prompt”。结合文献连接级不确定性，不能把可编码/有梯度当作疾病信息。','', '## Q2：冻结原始分类器的FC依赖','', '|图算子|训练内选定风格删除|随机删除|Top-abs(z)删除|相对随机差/胜折|相对Top-abs(z)差/胜折|','|---|---:|---:|---:|---|---|']
 for r in s['q2_scores']:text.append(f"|{r['graph']}|{r['selected']:.6f}|{r['random']:.6f}|{r['topz']:.6f}|{r['adv_random']:+.6f} / {r['wins_random']}/10|{r['adv_topz']:+.6f} / {r['wins_topz']}/10|")
 text+=['','表为K=8、完整TD条件mu替换的真实类logp下降。score_fidelity.csv保留K=4/8、三随机掩码均值来源、四种S4风格、S1/S2/S3、逐患者原预测正确/错误、预测类/真实类效应、半强度及匹配训练TD替换。S3只用16名训练者逐边探测，评价在不重叠的outer-test人群；不是个人oracle的自评。','', '每次只修改目标人的raw_X[person,64+edge_id]，其余人的输入和其它281维保持不变，完整重过GFR→CAM→OutputLayer。固定邻接关开用于分解图重连，关闭时严格相同。selected-keep只保留39候选中的选中边；其它217条FC和其它模态始终存在，不能说“只用8条FC预测”。','', '## Q3：选择器反事实学习，B0始终冻结','', '|组|heldout真实类删除影响|与C0差|方向导师rho|与C0差|删除/方向胜折|','|---|---:|---:|---:|---:|---|']
 for r in s['q3_cf']:text.append(f"|{r['group']}|{r['drop_true']:.6f}|{r['delta_drop_vs_C0']:+.6f}|{r['direction_rho']:.5f}|{r['delta_direction_vs_C0']:+.5f}|{r['wins_drop']}/10 / {r['wins_direction']}/10|")
 text+=['','C0拟合训练B0单边依赖导师；C1加keep，C2加delete，C3两者。错误/低置信训练预测不参加必要性CF；原分类器所有参数无优化步骤，无法用破坏分类器换删除下降。5折OOF按训练导师患者划分，未用同人拟合后在同人报告选择定位。','', 'C1–C3预注册将纯CF梯度最多缩到teacher范数的0.25；C3_FIXED_DIAGNOSTIC保留原.10/.05固定权重、相同步数和患者预算，用于量级比较，不据外层test选权重。hard TopK的真实有限差分是分段常数，STE是代理梯度；数值有限差分验收针对连续松弛与固定图，而非伪称hard导数等于STE。','', '|组|步骤|探测次数/有效|原CF/teacher平均比|应用后平均比|梯度cos均值|','|---|---:|---|---:|---:|---:|']
 for r in s['q3_gradients']:text.append(f"|{r['group']}|{r['steps']}|{r['probe_count']}/{r['valid_probes']}|{r['mean_raw_cf_teacher_ratio']:.5f}|{r['mean_applied_ratio']:.5f}|{r['mean_gradient_cosine']:.5f}|")
 text+=['','## Q4：背景适用性与质量角色','', '|角色|控制后风险/FC偏离rho|风险/计算不稳定rho|满足代理方向折数|平均选择数|弃权率|支持|','|---|---:|---:|---:|---:|---:|---|']
 for r in s['q4_roles']:text.append(f"|{r['role']}|{r['partial_absz_rho']:+.5f}|{r['partial_instability_rho']:+.5f}|{r['passing_folds']}/10|{r['mean_selected_count']:.3f}|{r['abstain_fraction']:.4f}|{r['go']}|")
 text+=['','R1只以训练TD背景支持度调节参考可比性/弃权，不产生疾病支持；R2只用已验证方向的DVARS/outlier/mean FD；R3只用CNR/FBER/QI1/SNR，未将EFC/FWHM/GSR统一当坏质量。原始16列QC与871人CSV匹配后的正仿射关系在preflight逐列验收。FUNC GSR是ghost-to-signal ratio，不是是否做global signal regression。','', '门控仅进入可解释的证据keep/弃权诊断，没有新分类残差；其影响可从选边数、拒绝及冻结模型keep输出核验，不被单位范数抵消。它不改变原B0常规预测，不能据此宣称分类增益。FC-only噪声不改变真实QC，可靠性不能假装感知物理头动；压力测试只评价计算稳定和过度置信风险。R4只在训练内角色通过全十折门槛后组合，未通过的角色不纳入。','', '## 审计、资源、限制与判断','',f"设备：{info['hardware']}。诊断任务GPU0/GPU1={info['gpu_tasks']}；四阶段各卡各5fold；B0额外验证10fold在GPU0，原训练复用未计新训练。各卡诊断任务墙钟之和（并发，不能相加当总耗时）={info['task_seconds']}秒。本次总墙钟{info['wall_seconds']:.2f}秒。",'','源码/数据/Prompt/CLIP/token/split/anchor SHA、每任务日志、锁、原子检查点、resume记录见manifest及runtime。每个训练选择器保存optimizer/RNG/step/fit/valid/config SHA；已完成任务按唯一(phase,fold)跳过。根模型、E32/E38/E39/E40和数据只读，最终SHA不变。','', '局限：原val=test选模、CV前RFE和标准化、全队列transductive图均保留；诊断模型已受历史测试选模影响。10fold训练集重叠，W/T/L是筛选描述，不是10个独立数据集的显著性。参考是统计TD条件估计，遮蔽/替换有分布偏移，不是临床正常线或脑区因果试验。','', '逐环节结果优先于创新叙述；任何一环NO-GO时不训练集成分类模型。局部模型依赖、语言几何或质量相关均不足以证明正确医学语义带来独立诊断收益。','', '文献用于定义检验边界：[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[ERASER](https://aclanthology.org/2020.acl-main.408/)、[GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html)、[ROAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)、[ABIDE多站点](https://pmc.ncbi.nlm.nih.gov/articles/PMC3782703/)、[FC可复现性](https://pubmed.ncbi.nlm.nih.gov/31285817/)、[非复制](https://pubmed.ncbi.nlm.nih.gov/31916675/)、[QC字段定义](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)。PMC6591952和PMC7268009页面遇验证码，实际采用对应PubMed身份/摘要核验，不伪称已读取被拦截全文。','',f"实际工作目录：`{HERE}`。/tmp不是持久存储保障，交付状态与科研完成分开记录。"]
 (RES/'report.md').write_text('\n'.join(text)+'\n')

def bounded_delivery():
 info=jr(RES/'manifest.json');target=ROOT/'experiment'/HERE.name;delivery=dict(status='PARTIAL_TMP_ONLY');pending=HERE/'handoff_append_pending.md'
 body=(RES/'report.md').read_text();append='\n\n## Experiment 42 — Four-link Evidence Validation\n\n'+body.replace('# Experiment 42 — 四环节独立可证伪机制验证\n','',1)
 # Never truncate or rewrite handoff. Recheck all historical bytes after append.
 old=jr(RT/'storage_initial.json')['handoff'];handoff=ROOT/'handoff.md'
 try:
  prefix=handoff.read_bytes();assert len(prefix)==old['bytes'] and hashlib.sha256(prefix).hexdigest()==old['sha256'],'handoff changed concurrently; unsafe to append against old prefix'
  with handoff.open('ab') as stream:
   fcntl.flock(stream,fcntl.LOCK_EX);stream.write(append.encode());stream.flush();os.fsync(stream.fileno())
  after=handoff.read_bytes();assert after[:len(prefix)]==prefix and after[len(prefix):]==append.encode();info['handoff']=dict(status='APPENDED_PREFIX_VERIFIED',old=old,new_sha=m.sha(handoff))
 except (OSError,AssertionError) as e:
  pending.write_text(append+'\n\nHandoff append blocked: '+repr(e)+'\n');info['handoff']=dict(status='PENDING_APPEND',error=repr(e),protected_prefix=old,pending_sha=m.sha(pending))
 try:
  target.mkdir(parents=True,exist_ok=False)
  # One bounded copy attempt, exact file checks. Never remove the stage.
  for p in HERE.rglob('*'):
   rel=p.relative_to(HERE);dest=target/rel
   if p.is_symlink():continue
   if p.is_dir():dest.mkdir(exist_ok=True)
   elif p.is_file():shutil.copyfile(p,dest);assert m.sha(p)==m.sha(dest)
  delivery=dict(status='COPIED_AND_HASH_VERIFIED',target=str(target))
 except OSError as e:delivery=dict(status='PARTIAL_TMP_ONLY',error=repr(e),stage=str(HERE),target=str(target))
 info['delivery']=delivery['status'];info['delivery_details']=delivery;info['result_sha']={p.name:m.sha(p) for p in RES.glob('*.csv')};jw(RES/'manifest.json',info)
 with (RES/'report.md').open('a') as f:f.write('\n实际交付：'+json.dumps(delivery,ensure_ascii=False)+'\n\nhandoff：'+json.dumps(info['handoff'],ensure_ascii=False)+'\n')
 if pending.exists():pending.write_text(append+'\n\nDelivery: '+json.dumps(delivery,ensure_ascii=False)+'\nHandoff: '+json.dumps(info['handoff'],ensure_ascii=False)+'\n')

def prepare_anchor_files():
 rows={int(r['fold']):r for r in cr(OLD/'results/fold_metrics.csv') if r['group']=='B0'}
 for fold,row in rows.items():
  for src,dst,expected in [(OLD/f'.runtime/anchors/anchor_fold{fold:02}.pt',RT/f'anchors/anchor_fold{fold:02}.pt',row['anchor_sha']),(Path(row['best_checkpoint']),RT/f'anchors/best_fold{fold:02}.pt',row['best_checkpoint_sha256'])]:
   assert m.sha(src)==expected
   if not dst.exists():shutil.copyfile(src,dst)
   assert m.sha(dst)==expected

def all_stages():
 if not (RES/'manifest.json').exists():audit()
 if not (RT/'cache/text_bank.pt').exists():encode()
 prepare_anchor_files()
 if not (RT/'preflight.json').exists():preflight()
 info=jr(RES/'manifest.json')
 if info.get('anchor',{}).get('status')!='ANCHOR_EXACT_REUSE_PASS':anchor()
 # All design/config/source hashes frozen before prompt training or test readout.
 info=jr(RES/'manifest.json');sig=artifact_signature()
 if 'frozen_signature' in info:assert info['frozen_signature']==sig
 else:info['frozen_signature']=sig;info['frozen_at']=stamp();jw(RES/'manifest.json',info)
 run_stage('q1');aggregate()
 # A bounded two-fold compute audit, then the unchanged remaining folds.
 run_stage('q2',[1,6]);run_stage('q2',[2,3,4,5,7,8,9,10]);aggregate()
 run_stage('q3');aggregate();run_stage('q4');aggregate();finalize()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['audit','preflight','anchor','encode','q1','q2','q3','q4','worker','all','resume','finalize']);ap.add_argument('--phase',choices=['q1','q2','q3','q4']);ap.add_argument('--fold',type=int);args=ap.parse_args();torch.set_num_threads(2)
 if args.command in ('all','resume'):all_stages()
 elif args.command=='worker':worker(args.phase,args.fold)
 elif args.command in ('q1','q2','q3','q4'):
  if args.fold:worker(args.command,args.fold)
  else:run_stage(args.command)
 elif args.command=='audit':audit()
 elif args.command=='encode':encode()
 elif args.command=='preflight':prepare_anchor_files();preflight()
 else:globals()[args.command]()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if RT.exists():jw(RT/'failure.json',dict(time=stamp(),error=repr(e),traceback=traceback.format_exc()))
  raise

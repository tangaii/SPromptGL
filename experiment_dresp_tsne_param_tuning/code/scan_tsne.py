from pathlib import Path
import csv, json, hashlib, time, os
import numpy as np
import inspect
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_param_tuning')
INP=ROOT/'inputs'; OUT=ROOT/'results'
MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']
FOLDS=[7,8,9]
CONFIGS={
 'P0':dict(perplexity=40,early_exaggeration=12,learning_rate='auto',max_iter=2000),
 'P1':dict(perplexity=20,early_exaggeration=12,learning_rate='auto',max_iter=3000),
 'P2':dict(perplexity=30,early_exaggeration=12,learning_rate='auto',max_iter=3000),
 'P3':dict(perplexity=60,early_exaggeration=12,learning_rate='auto',max_iter=3000),
 'P4':dict(perplexity=80,early_exaggeration=12,learning_rate='auto',max_iter=3000),
 'P5':dict(perplexity=100,early_exaggeration=12,learning_rate='auto',max_iter=3000),
 'P6':dict(perplexity=60,early_exaggeration=8,learning_rate='auto',max_iter=3000),
 'P7':dict(perplexity=80,early_exaggeration=16,learning_rate='auto',max_iter=3000),
}

def task(model,fold,cid):
 p=INP/f'{model}_fold{fold:02d}_features.npz'; z=np.load(p,allow_pickle=True); x=z['features'].astype(np.float64); y=z['labels'].astype(int)
 t0=time.time(); x=StandardScaler().fit_transform(x); x=PCA(n_components=50,random_state=0).fit_transform(x)
 c=CONFIGS[cid]
 kwargs=dict(n_components=2,perplexity=c['perplexity'],early_exaggeration=c['early_exaggeration'],learning_rate=c['learning_rate'],init='pca',metric='euclidean',random_state=0,max_iter=c['max_iter'])
 emb=TSNE(**kwargs).fit_transform(x)
 dur=time.time()-t0
 tr15=trustworthiness(x,emb,n_neighbors=15,metric='euclidean'); tr30=trustworthiness(x,emb,n_neighbors=30,metric='euclidean')
 sil=silhouette_score(emb,y)
 digest=hashlib.sha256(z['features'].tobytes()+z['labels'].tobytes()+z['patient_ids'].tobytes()).hexdigest()
 out=OUT/'scan_coordinates'/cid; out.mkdir(parents=True,exist_ok=True)
 np.save(out/f'{model}_fold{fold:02d}.npy',emb)
 return {'config':cid,'model':model,'fold':fold,'perplexity':c['perplexity'],'early_exaggeration':c['early_exaggeration'],'learning_rate':c['learning_rate'],'max_iter':c['max_iter'],'kl_divergence':float(getattr(TSNE(**kwargs),'kl_divergence_',np.nan)) if False else None,'trustworthiness_k15':float(tr15),'trustworthiness_k30':float(tr30),'silhouette_2d':float(sil),'x_min':float(emb[:,0].min()),'x_max':float(emb[:,0].max()),'y_min':float(emb[:,1].min()),'y_max':float(emb[:,1].max()),'nonfinite':int((~np.isfinite(emb)).sum()),'runtime_sec':dur,'feature_identity_sha256':digest,'coordinate_sha256':hashlib.sha256(emb.tobytes()).hexdigest()}

def run_one(args):
 # fit once here to expose KL without retaining estimator in parent
 model,fold,cid=args; p=INP/f'{model}_fold{fold:02d}_features.npz'; z=np.load(p,allow_pickle=True); raw=z['features'].astype(np.float64); y=z['labels'].astype(int)
 t0=time.time(); x=StandardScaler().fit_transform(raw); x=PCA(n_components=50,random_state=0).fit_transform(x); c=CONFIGS[cid]
 iter_kw='max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
 est=TSNE(n_components=2,perplexity=c['perplexity'],early_exaggeration=c['early_exaggeration'],learning_rate=c['learning_rate'],init='pca',metric='euclidean',random_state=0,verbose=0,**{iter_kw:c['max_iter']})
 emb=est.fit_transform(x); tr15=trustworthiness(x,emb,n_neighbors=15); tr30=trustworthiness(x,emb,n_neighbors=30); sil=silhouette_score(emb,y)
 out=OUT/'scan_coordinates'/cid; out.mkdir(parents=True,exist_ok=True); np.save(out/f'{model}_fold{fold:02d}.npy',emb)
 return {'config':cid,'model':model,'fold':fold,'perplexity':c['perplexity'],'early_exaggeration':c['early_exaggeration'],'learning_rate':str(c['learning_rate']),'max_iter':c['max_iter'],'kl_divergence':float(est.kl_divergence_),'trustworthiness_k15':float(tr15),'trustworthiness_k30':float(tr30),'silhouette_2d':float(sil),'x_min':float(emb[:,0].min()),'x_max':float(emb[:,0].max()),'y_min':float(emb[:,1].min()),'y_max':float(emb[:,1].max()),'nonfinite':int((~np.isfinite(emb)).sum()),'runtime_sec':time.time()-t0,'feature_identity_sha256':hashlib.sha256(z['features'].tobytes()+z['labels'].tobytes()+z['patient_ids'].tobytes()).hexdigest(),'coordinate_sha256':hashlib.sha256(emb.tobytes()).hexdigest()}

if __name__=='__main__':
 OUT.joinpath('scan_coordinates').mkdir(exist_ok=True)
 args=[(m,f,c) for c in CONFIGS for f in FOLDS for m in MODELS]
 rows=[]
 with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as ex:
  futs=[ex.submit(run_one,a) for a in args]
  for i,f in enumerate(as_completed(futs),1):
   row=f.result(); rows.append(row); print(f'[{i}/{len(args)}] {row["config"]} {row["model"]} fold{row["fold"]} kl={row["kl_divergence"]:.5g}',flush=True)
 rows.sort(key=lambda r:(r['config'],r['fold'],MODELS.index(r['model'])))
 fields=list(rows[0].keys());
 with open(OUT/'parameter_comparison.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 json.dump({'configs':CONFIGS,'models':MODELS,'folds':FOLDS,'rows':len(rows)},open(OUT/'scan_manifest.json','w'),indent=2)
 print('DONE',len(rows))

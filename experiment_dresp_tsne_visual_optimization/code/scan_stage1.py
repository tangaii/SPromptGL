from pathlib import Path
import json, hashlib, time, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization'); INP=ROOT/'inputs'; OUT=ROOT/'results'
MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']
CONFIGS={
 'B0':dict(perplexity=40,early_exaggeration=12,learning_rate='auto',metric='euclidean',max_iter=2000),
 'B1':dict(perplexity=60,early_exaggeration=12,learning_rate='auto',metric='euclidean',max_iter=3000),
 'B2':dict(perplexity=5,early_exaggeration=4,learning_rate=100,metric='euclidean',max_iter=3000),
 'B3':dict(perplexity=10,early_exaggeration=8,learning_rate=200,metric='euclidean',max_iter=3000),
 'B4':dict(perplexity=20,early_exaggeration=12,learning_rate=500,metric='euclidean',max_iter=3000),
 'B5':dict(perplexity=100,early_exaggeration=16,learning_rate=200,metric='euclidean',max_iter=3000),
 'B6':dict(perplexity=150,early_exaggeration=24,learning_rate=500,metric='euclidean',max_iter=3000),
 'B7':dict(perplexity=200,early_exaggeration=32,learning_rate=800,metric='euclidean',max_iter=3000),
 'B8':dict(perplexity=200,early_exaggeration=64,learning_rate=800,metric='euclidean',max_iter=3000),
 'B9':dict(perplexity=40,early_exaggeration=12,learning_rate=10,metric='euclidean',max_iter=3000),
 'B10':dict(perplexity=40,early_exaggeration=12,learning_rate=800,metric='euclidean',max_iter=3000),
 'B11':dict(perplexity=100,early_exaggeration=12,learning_rate=800,metric='euclidean',max_iter=3000),
 'B12':dict(perplexity=5,early_exaggeration=4,learning_rate=100,metric='cosine',max_iter=3000),
 'B13':dict(perplexity=10,early_exaggeration=8,learning_rate=200,metric='cosine',max_iter=3000),
 'B14':dict(perplexity=20,early_exaggeration=12,learning_rate=500,metric='cosine',max_iter=3000),
 'B15':dict(perplexity=40,early_exaggeration=12,learning_rate='auto',metric='cosine',max_iter=3000),
 'B16':dict(perplexity=100,early_exaggeration=16,learning_rate=200,metric='cosine',max_iter=3000),
 'B17':dict(perplexity=150,early_exaggeration=24,learning_rate=500,metric='cosine',max_iter=3000),
 'B18':dict(perplexity=200,early_exaggeration=32,learning_rate=800,metric='cosine',max_iter=3000),
 'B19':dict(perplexity=200,early_exaggeration=64,learning_rate=800,metric='cosine',max_iter=3000),
 'B20':dict(perplexity=60,early_exaggeration=40,learning_rate='auto',metric='euclidean',max_iter=3000),
 'B21':dict(perplexity=80,early_exaggeration=64,learning_rate=200,metric='euclidean',max_iter=3000),
 'B22':dict(perplexity=150,early_exaggeration=64,learning_rate='auto',metric='cosine',max_iter=3000),
 'B23':dict(perplexity=100,early_exaggeration=24,learning_rate=50,metric='euclidean',max_iter=3000),
}
import inspect
ITER_KW='max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
def one(a):
 cid,m,f=a; f=int(f)
 p=INP/f'{m}_fold{f:02d}_features.npz'; z=np.load(p,allow_pickle=True)
 raw=z['features'].astype(np.float64); y=z['labels'].astype(int)
 t0=time.time(); x=PCA(50,random_state=0).fit_transform(StandardScaler().fit_transform(raw))
 c=CONFIGS[cid]
 est=TSNE(n_components=2,perplexity=c['perplexity'],early_exaggeration=c['early_exaggeration'],learning_rate=c['learning_rate'],init='pca',metric=c['metric'],random_state=0,verbose=0,**{ITER_KW:c['max_iter']})
 emb=est.fit_transform(x)
 tr15=trustworthiness(x,emb,n_neighbors=15,metric='euclidean'); tr30=trustworthiness(x,emb,n_neighbors=30,metric='euclidean')
 sil=silhouette_score(emb,y)
 out=OUT/'stage1_coordinates'/cid; out.mkdir(parents=True,exist_ok=True); np.save(out/f'{m}_fold{f:02d}.npy',emb)
 return {'config':cid,'model':m,'fold':f,'perplexity':c['perplexity'],'early_exaggeration':c['early_exaggeration'],'learning_rate':str(c['learning_rate']),'metric':c['metric'],'max_iter':c['max_iter'],'kl_divergence':float(est.kl_divergence_),'trustworthiness_k15':float(tr15),'trustworthiness_k30':float(tr30),'silhouette_2d':float(sil),'x_min':float(emb[:,0].min()),'x_max':float(emb[:,0].max()),'y_min':float(emb[:,1].min()),'y_max':float(emb[:,1].max()),'nonfinite':int((~np.isfinite(emb)).sum()),'runtime_sec':time.time()-t0,'feature_identity_sha256':hashlib.sha256(z['features'].tobytes()+z['labels'].tobytes()+z['patient_ids'].tobytes()).hexdigest(),'coordinate_sha256':hashlib.sha256(emb.tobytes()).hexdigest()}
if __name__=='__main__':
 args=[(c,m,int(f)) for c in CONFIGS for f in [4,10] for m in MODELS]
 rows=[]
 with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as ex:
  fs=[ex.submit(one,a) for a in args]
  for i,f in enumerate(as_completed(fs),1):
   r=f.result(); rows.append(r); print(i,'/',len(args),r['config'],r['model'],'f'+str(r['fold']),'kl={:.4g} tr15={:.4f}'.format(r['kl_divergence'],r['trustworthiness_k15']),flush=True)
 import csv
 rows.sort(key=lambda r:(r['config'],r['fold'],MODELS.index(r['model'])))
 with open(OUT/'parameter_comparison.csv','w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
 json.dump({'configs':CONFIGS,'folds':[4,10],'models':MODELS,'runs':len(rows)},open(OUT/'stage1_manifest.json','w'),indent=2)
 print('DONE',len(rows))

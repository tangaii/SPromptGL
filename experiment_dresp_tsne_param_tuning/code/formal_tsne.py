from pathlib import Path
import json, hashlib, inspect, time, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_param_tuning'); INP=ROOT/'inputs'; OUT=ROOT/'results/formal_coordinates'; OUT.mkdir(parents=True,exist_ok=True)
models=['SPromptGL','DARG','DARG_JSL','DReSP']; folds=range(1,11)
cfg={'id':'P3','perplexity':60,'early_exaggeration':12,'learning_rate':'auto','max_iter':3000,'random_state':0,'n_components':2,'metric':'euclidean','init':'pca','preprocessing':'StandardScaler then PCA(50)'}
iter_kw='max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
def one(a):
 m,f=a; p=INP/f'{m}_fold{f:02d}_features.npz'; z=np.load(p,allow_pickle=True); x=PCA(50,random_state=0).fit_transform(StandardScaler().fit_transform(z['features'].astype(np.float64))); t=time.time(); est=TSNE(n_components=2,perplexity=60,early_exaggeration=12,learning_rate='auto',init='pca',metric='euclidean',random_state=0,**{iter_kw:3000}); emb=est.fit_transform(x); np.save(OUT/f'{m}_fold{f:02d}.npy',emb); return {'model':m,'fold':f,'coordinate_sha256':hashlib.sha256(emb.tobytes()).hexdigest(),'kl_divergence':float(est.kl_divergence_),'runtime_sec':time.time()-t,'nonfinite':int((~np.isfinite(emb)).sum())}
if __name__=='__main__':
 args=[(m,f) for m in models for f in folds if not (OUT/f'{m}_fold{f:02d}.npy').exists()]; rows=[]
 with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as ex:
  fs=[ex.submit(one,a) for a in args]
  for i,f in enumerate(as_completed(fs),1):
   r=f.result(); rows.append(r); print(i,'/',len(args),r,flush=True)
 for m in models:
  for f in folds:
   p=OUT/f'{m}_fold{f:02d}.npy'; assert p.exists()
   e=np.load(p); rows.append({'model':m,'fold':f,'coordinate_sha256':hashlib.sha256(e.tobytes()).hexdigest(),'nonfinite':int((~np.isfinite(e)).sum())})
 json.dump({'config':cfg,'rows':rows,'sklearn_version':__import__('sklearn').__version__,'iter_keyword_used':iter_kw},open(ROOT/'results/formal_tsne.json','w'),indent=2)
 print('formal complete',len(list(OUT.glob('*.npy'))))

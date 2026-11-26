from pathlib import Path
import numpy as np, json, hashlib, inspect, time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization'); INP=ROOT/'inputs'; OUT=ROOT/'results/stage2_coordinates'; OUT.mkdir(parents=True,exist_ok=True)
models=['SPromptGL','DARG','DARG_JSL','DReSP']; cids=['B0','B1','B5','B6','B9','B15']; folds=[7,8]
configs={x: {'B0':(40,12,'auto','euclidean',2000),'B1':(60,12,'auto','euclidean',3000),'B5':(100,16,200,'euclidean',3000),'B6':(150,24,500,'euclidean',3000),'B9':(40,12,10,'euclidean',3000),'B15':(40,12,'auto','cosine',3000)}[x] for x in cids}
kw='max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'; rows=[]
for cid in cids:
 for f in folds:
  for m in models:
   z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); x=PCA(50,random_state=0).fit_transform(StandardScaler().fit_transform(z['features'].astype(float))); per,ee,lr,metric,it=configs[cid]; t=time.time(); e=TSNE(n_components=2,perplexity=per,early_exaggeration=ee,learning_rate=lr,metric=metric,init='pca',random_state=0,**{kw:it}); xy=e.fit_transform(x); np.save(OUT/f'{cid}_{m}_fold{f:02d}.npy',xy); rows.append({'config':cid,'model':m,'fold':f,'kl':float(e.kl_divergence_),'trust15':float(trustworthiness(x,xy,n_neighbors=15)),'trust30':float(trustworthiness(x,xy,n_neighbors=30)),'silhouette':float(silhouette_score(xy,z['labels'])),'runtime':time.time()-t})
print('rows',len(rows)); json.dump(rows,open(ROOT/'results/stage2_comparison.json','w'),indent=2)

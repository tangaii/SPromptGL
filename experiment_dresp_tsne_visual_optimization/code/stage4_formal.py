from pathlib import Path
import numpy as np, json, hashlib, inspect, time
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, silhouette_samples
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization')
INP=ROOT/'inputs'; CO=ROOT/'results/formal_coordinates'; CO.mkdir(exist_ok=True)
MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']; FOLDS=list(range(1,11))
PARAM=dict(perplexity=100, early_exaggeration=16, learning_rate=200, metric='euclidean', init='pca', random_state=0, max_iter=3000)
kw='max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
CONFIG_ID='B5'
def job(args):
 m,f=args
 t=time.time()
 z=np.load(INP/f'{m}_fold{f:02d}_features.npz')
 x=PCA(50,random_state=0).fit_transform(StandardScaler().fit_transform(z['features'].astype(float)))
 e=TSNE(n_components=2,perplexity=100,early_exaggeration=16,learning_rate=200,metric='euclidean',init='pca',random_state=0,n_iter=3000)
 xy=e.fit_transform(x)
 np.save(CO/f'{m}_fold{f:02d}.npy',xy)
 sha=hashlib.sha256((INP/f'{m}_fold{f:02d}_features.npz').read_bytes()).hexdigest()
 return dict(model=m,fold=f,kl=float(e.kl_divergence_),runtime=time.time()-t,feature_sha=sha,coord_sha=hashlib.sha256((CO/f'{m}_fold{f:02d}.npy').read_bytes()).hexdigest())
jobs=[(m,f) for m in MODELS for f in FOLDS]
with ProcessPoolExecutor(max_workers=10) as ex:
 res=list(ex.map(job,jobs))
json.dump(dict(config=CONFIG_ID,param={k:str(v) for k,v in PARAM.items()},runs=res),open(ROOT/'results/stage4_run_manifest.json','w'),indent=2)
print('done',len(res))

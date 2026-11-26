from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization'); INP=ROOT/'inputs'; CO=ROOT/'results/formal_coordinates'; FIG=ROOT/'results/figures'
MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']; colors={0:'#D98282',1:'#4C9A91'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
def scatter(ax,xy,y,title):
 for label,name in [(0,'ASD'),(1,'TD')]:
  q=y==label; ax.scatter(xy[q,0],xy[q,1],s=12,alpha=.70,c=colors[label],label=name,linewidths=0,rasterized=True)
 ax.set_title(title,fontsize=11); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
 for spine in ax.spines.values(): spine.set_visible(False)
for m in MODELS:
 for f in range(1,11):
  z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); xy=np.load(CO/f'{m}_fold{f:02d}.npy')
  fig,ax=plt.subplots(figsize=(5,4),facecolor='white'); scatter(ax,xy,z['labels'],f'{m} · Fold {f:02d}'); ax.legend(frameon=False,loc='best',markerscale=1.5); fig.tight_layout(pad=.5); fig.savefig(FIG/'individual'/m/f'fold{f:02d}.png',dpi=600,facecolor='white'); plt.close(fig)
for f in range(1,11):
 fig,axs=plt.subplots(2,2,figsize=(9,7),facecolor='white'); axs=axs.ravel()
 for i,m in enumerate(MODELS):
  z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); xy=np.load(CO/f'{m}_fold{f:02d}.npy'); scatter(axs[i],xy,z['labels'],f'({chr(97+i)}) {m}')
  if i==0: axs[i].legend(frameon=False,loc='best',markerscale=1.5)
 fig.tight_layout(pad=.8,w_pad=.4,h_pad=.5); fig.savefig(FIG/'comparison'/f'fold{f:02d}_comparison.png',dpi=600,facecolor='white'); plt.close(fig)
# metrics
rows=[]
for m in MODELS:
 for f in range(1,11):
  z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); X=z['features'].astype(float); y=z['labels'].astype(int); xy=np.load(CO/f'{m}_fold{f:02d}.npy')
  x50=PCA(50,random_state=0).fit_transform(StandardScaler().fit_transform(X))
  # Euclidean geometry in 2D and raw feature dimensions
  c0=xy[y==0]; c1=xy[y==1]; center_dist=float(np.linalg.norm(c0.mean(0)-c1.mean(0)))
  within=np.mean([np.linalg.norm(c0-c0.mean(0),axis=1).mean(),np.linalg.norm(c1-c1.mean(0),axis=1).mean()])
  raw0=X[y==0]; raw1=X[y==1]; raw_center=float(np.linalg.norm(raw0.mean(0)-raw1.mean(0)))
  raw_within=np.mean([np.linalg.norm(raw0-raw0.mean(0),axis=1).mean(),np.linalg.norm(raw1-raw1.mean(0),axis=1).mean()])
  rows.append({'model':m,'fold':f,'silhouette_2d':silhouette_score(xy,y),'silhouette_raw144':silhouette_score(X,y),'trustworthiness_k15':trustworthiness(x50,xy,n_neighbors=15),'trustworthiness_k30':trustworthiness(x50,xy,n_neighbors=30),'center_distance_2d':center_dist,'within_mean_distance_2d':within,'between_within_ratio_2d':center_dist/max(within,1e-12),'center_distance_raw144':raw_center,'within_mean_distance_raw144':raw_within,'between_within_ratio_raw144':raw_center/max(raw_within,1e-12),'x_min':float(xy[:,0].min()),'x_max':float(xy[:,0].max()),'y_min':float(xy[:,1].min()),'y_max':float(xy[:,1].max()),'nonfinite':int((~np.isfinite(xy)).sum()),'n_patients':len(y),'feature_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'coordinate_sha256':hashlib.sha256(xy.tobytes()).hexdigest()})
with open(ROOT/'results/final_metrics.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# coordinates
with open(ROOT/'results/tsne_coordinates.csv','w',newline='') as f:
 fields=['model','fold','patient_id','true_label','tsne_x','tsne_y']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
 for m in MODELS:
  for fold in range(1,11):
   z=np.load(INP/f'{m}_fold{fold:02d}_features.npz'); xy=np.load(CO/f'{m}_fold{fold:02d}.npy')
   for i in range(871): w.writerow({'model':m,'fold':fold,'patient_id':int(z['patient_ids'][i]),'true_label':int(z['labels'][i]),'tsne_x':xy[i,0],'tsne_y':xy[i,1]})
print('figures, metrics, coordinates done')

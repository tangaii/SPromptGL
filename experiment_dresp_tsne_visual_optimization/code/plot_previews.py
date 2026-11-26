from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization'); INP=ROOT/'inputs'; OUT=ROOT/'results'; MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']; C={0:'#D98282',1:'#4C9A91'}
def draw(ax,xy,y,title):
 for l in [0,1]:
  q=y==l; ax.scatter(xy[q,0],xy[q,1],s=8,alpha=.72,c=C[l],label=['ASD','TD'][l],linewidths=0)
 ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
for cid in ['B'+str(i) for i in range(24)]:
 for fold in [4,10]:
  fig,axs=plt.subplots(2,4,figsize=(12,6),facecolor='white')
  for i,m in enumerate(MODELS):
   z=np.load(INP/f'{m}_fold{fold:02d}_features.npz'); draw(axs[0,i],np.load(OUT/'stage1_coordinates'/cid/f'{m}_fold{fold:02d}.npy'),z['labels'],m)
   z=np.load(INP/f'{m}_fold{fold:02d}_features.npz'); draw(axs[1,i],np.load(OUT/'stage1_coordinates'/cid/f'{m}_fold{fold:02d}.npy'),z['labels'],f'Fold {fold:02d}')
  fig.suptitle(f'{cid} | Fold {fold:02d}',fontsize=14); fig.tight_layout(); fig.savefig(OUT/'figures/previews'/f'{cid}_fold{fold:02d}.png',dpi=180,facecolor='white'); plt.close(fig)
print('previews done')

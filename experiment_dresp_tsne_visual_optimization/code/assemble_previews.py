from pathlib import Path
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=Path('.'); IN=ROOT/'inputs'; R=ROOT/'results'; MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']; COLORS={0:'#D98282',1:'#4C9A91'}
def draw(ax,xy,y,title):
 for lab,name in [(0,'ASD'),(1,'TD')]:
  q=y==lab; ax.scatter(xy[q,0],xy[q,1],s=5,alpha=.7,c=COLORS[lab],label=name,linewidths=0)
 ax.set_title(title,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
for cid in [f'B{i}' for i in range(24)]:
 fig,axs=plt.subplots(2,4,figsize=(12,6),facecolor='white')
 for ri,fold in enumerate([4,10]):
  for ci,m in enumerate(MODELS):
   z=np.load(IN/f'{m}_fold{fold:02d}_features.npz'); xy=np.load(R/'stage1_coordinates'/cid/f'{m}_fold{fold:02d}.npy'); draw(axs[ri,ci],xy,z['labels'],f'Fold {fold} · {m}')
   if ri==0 and ci==0: axs[ri,ci].legend(frameon=False,fontsize=7,markerscale=1.3)
 cfg=json.load(open(R/'stage1_manifest.json'))['configs'][cid]
 fig.suptitle(f"{cid} | perplexity={cfg['perplexity']} | early_exaggeration={cfg['early_exaggeration']} | learning_rate={cfg['learning_rate']} | metric={cfg['metric']}",fontsize=11)
 fig.tight_layout(); fig.savefig(R/'figures/previews'/f'{cid}.png',dpi=180,facecolor='white'); plt.close(fig)
# six stage2 preview grids, 2 folds by 4 models
stage2=json.load(open(R/'stage2_comparison.json'))
for cid in ['B0','B1','B5','B6','B9','B15']:
 cfg=json.load(open(R/'stage1_manifest.json'))['configs'][cid]
 fig,axs=plt.subplots(2,4,figsize=(12,6),facecolor='white')
 for ri,fold in enumerate([7,8]):
  for ci,m in enumerate(MODELS):
   z=np.load(IN/f'{m}_fold{fold:02d}_features.npz'); xy=np.load(R/'stage2_coordinates'/f'{cid}_{m}_fold{fold:02d}.npy'); draw(axs[ri,ci],xy,z['labels'],f'Fold {fold} · {m}')
 fig.suptitle(f"Stage 2 {cid} | p={cfg['perplexity']} ee={cfg['early_exaggeration']} lr={cfg['learning_rate']} metric={cfg['metric']}",fontsize=11); fig.tight_layout(); fig.savefig(R/'figures/stage2_previews'/f'{cid}.png',dpi=180,facecolor='white'); plt.close(fig)
print('preview assembly complete')

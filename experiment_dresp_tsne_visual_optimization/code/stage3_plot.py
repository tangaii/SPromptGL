from pathlib import Path
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization'); OUT=ROOT/'results/stage3_figures'; OUT.mkdir(parents=True,exist_ok=True)
models=['SPromptGL','DARG','DARG_JSL','DReSP']; cids=['B0','B1','B5','B6','B9','B15']; folds=[7,8]
for cid in cids:
 fig,axs=plt.subplots(2,4,figsize=(16,8))
 for i,f in enumerate(folds):
  for j,m in enumerate(models):
   z=np.load(ROOT/f'inputs/{m}_fold{f:02d}_features.npz'); xy=np.load(ROOT/f'results/stage2_coordinates/{cid}_{m}_fold{f:02d}.npy')
   ax=axs[i,j]; ax.scatter(xy[:,0],xy[:,1],c=z['labels'],s=6,cmap='tab10'); ax.set_title(f'{m} f{f}',fontsize=9); ax.set_xticks([]); ax.set_yticks([])
 fig.suptitle(cid); fig.tight_layout(); fig.savefig(OUT/f'{cid}_grid.png',dpi=150); plt.close(fig)
# summary table
rows=json.load(open(ROOT/'results/stage2_comparison.json'))
import collections
agg=collections.defaultdict(list)
for r_ in rows: agg[(r_['config'],r_['model'])].append(r_)
lines=['config model kl trust15 trust30 silhouette']
for k in sorted(agg):
 v=agg[k]; lines.append(' '.join(map(str,[k[0],k[1],round(np.mean([x['kl'] for x in v]),4),round(np.mean([x['trust15'] for x in v]),4),round(np.mean([x['trust30'] for x in v]),4),round(np.mean([x['silhouette'] for x in v]),4)])))
(ROOT/'results/stage3_summary.txt').write_text('\n'.join(lines))
print('\n'.join(lines))

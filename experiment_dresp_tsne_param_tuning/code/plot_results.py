from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path('/model/spromptgl/experiment/experiment_dresp_tsne_param_tuning'); INP=ROOT/'inputs'; OUT=ROOT/'results'; CO=OUT/'formal_coordinates'
MODELS=['SPromptGL','DARG','DARG_JSL','DReSP']; COLORS={0:'#D98282',1:'#4C9A91'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
def panel(ax, xy,y,title):
 for lab in [0,1]:
  q=y==lab; ax.scatter(xy[q,0],xy[q,1],s=7,alpha=.72,c=COLORS[lab],label='ASD' if lab==0 else 'TD',linewidths=0)
 ax.set_title(title,fontsize=12); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for m in MODELS:
 d=OUT/'figures/individual'/m; d.mkdir(parents=True,exist_ok=True)
 for f in range(1,11):
  z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); xy=np.load(CO/f'{m}_fold{f:02d}.npy'); fig,ax=plt.subplots(figsize=(5,4),facecolor='white'); panel(ax,xy,z['labels'],m+f' fold {f:02d}'); ax.legend(frameon=False,loc='best',markerscale=2); fig.tight_layout(); fig.savefig(d/f'fold{f:02d}.png',dpi=600,facecolor='white'); plt.close(fig)
for f in range(1,11):
 fig,axs=plt.subplots(2,2,figsize=(10,8),facecolor='white'); axs=axs.ravel()
 for i,m in enumerate(MODELS):
  z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); panel(axs[i],np.load(CO/f'{m}_fold{f:02d}.npy'),z['labels'],f'({chr(97+i)}) {m}')
  if i==0: axs[i].legend(frameon=False,loc='best',markerscale=2)
 fig.suptitle(f'ABIDE Fold {f:02d}',fontsize=14); fig.tight_layout(); fig.savefig(OUT/'figures/comparison'/f'fold{f:02d}_comparison.png',dpi=600,facecolor='white'); fig.savefig(OUT/'figures/comparison'/f'fold{f:02d}_comparison.pdf',facecolor='white'); plt.close(fig)
# parameter previews
rows=list(csv.DictReader(open(OUT/'parameter_comparison.csv')))
for f in [7,8,9]:
 for cid in ['P0','P1','P2','P3','P4','P5','P6','P7']:
  fig,axs=plt.subplots(2,2,figsize=(10,8),facecolor='white'); axs=axs.ravel()
  for i,m in enumerate(MODELS):
   z=np.load(INP/f'{m}_fold{f:02d}_features.npz'); panel(axs[i],np.load(OUT/'scan_coordinates'/cid/f'{m}_fold{f:02d}.npy'),z['labels'],f'({chr(97+i)}) {m}')
   if i==0: axs[i].legend(frameon=False,loc='best',markerscale=2)
  fig.suptitle(f'Fold {f:02d} · {cid}',fontsize=14); fig.tight_layout(); fig.savefig(OUT/'figures/parameter_previews'/f'fold{f:02d}_{cid}.png',dpi=220,facecolor='white'); plt.close(fig)
# coordinate csv
with open(OUT/'tsne_coordinates.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['model','fold','patient_id','true_label','tsne_x','tsne_y','parameter']); w.writeheader()
 for m in MODELS:
  for fold in range(1,11):
   z=np.load(INP/f'{m}_fold{fold:02d}_features.npz'); xy=np.load(CO/f'{m}_fold{fold:02d}.npy')
   for i in range(871): w.writerow({'model':m,'fold':fold,'patient_id':int(z['patient_ids'][i]),'true_label':int(z['labels'][i]),'tsne_x':float(xy[i,0]),'tsne_y':float(xy[i,1]),'parameter':'P3'})
print('plots complete')

import os,sys,json
from pathlib import Path
ROOT=Path('/model/spromptgl/experiment/experiment_darg_jsl')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'runtime/python'),str(ROOT/'code/history_base')]
import jsl_runtime as rt
import torch, numpy as np
ctx=rt.setup(1)
payload=torch.load(ROOT/'checkpoints/fold01.best.pt',map_location=ctx['state']['x'].device,weights_only=False);rt.restore(ctx,payload)
res,cap=rt.forward(ctx,False);print(cap['enhanced'].shape,cap['cam'].shape,ctx['state']['labels_np'].shape)
np.savez('/model/spromptgl/experiment/experiment_dresp_tsne_40folds/results/testjsl.npz',features=cap['enhanced'].detach().cpu().numpy(),labels=ctx['state']['labels_np'])

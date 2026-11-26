import sys,torch,numpy as np
from pathlib import Path
root=Path('/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr')
sys.path.insert(0,str(root)); import run
state,warm,ident=run.prepare(1); state['fold']=1; state['group']='E74.4-G2-Positive-DCR'; p=run.torch_load(root/'.runtime/checkpoints/E74.4-G2-Positive-DCR_f01.best.pt'); print(p.keys());run.load_history().load_bundle(state,p['best_state']);out=run.forward(state,state['group'],training=False,collect=True); print(out.keys(),out['fused'].shape); q=state['cam'](out['fused']); print(type(q),len(q),[x.shape if hasattr(x,'shape') else type(x) for x in q])

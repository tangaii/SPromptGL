import json,random,sys,time
from pathlib import Path
import numpy as np, torch
E=Path('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison')
sys.path.insert(0,str(E/'common'))
from tadpole_model import E83Model
random.seed(0); np.random.seed(0); torch.manual_seed(0)
data=np.loadtxt(E/'preprocessing/processed_standard_data.csv',delimiter=',',skiprows=1)
X=torch.tensor(data[:,:-1],dtype=torch.float32)
y=torch.tensor(data[:,-1].astype(int)-1,dtype=torch.long)
s=json.loads((E/'data_audit/split_indices.json').read_text())['splits'][0]
tr=torch.tensor(s['train_idx'],dtype=torch.long); te=torch.tensor(s['test_idx'],dtype=torch.long)
rng=np.random.default_rng(532108)
tr_np=np.asarray(s['train_idx']); pairs=[]
for target in [1,0]:
 mask=[]
 for i in range(len(tr_np)):
  for j in range(i+1,len(tr_np)):
   same=int(y[tr_np[i]]==y[tr_np[j]])
   if same==target: mask.append((tr_np[i],tr_np[j]))
 rng.shuffle(mask); pairs.extend([(a,b,target) for a,b in mask[:2048]])
pi=torch.tensor([p[0] for p in pairs]); pj=torch.tensor([p[1] for p in pairs]); pt=torch.tensor([p[2] for p in pairs],dtype=torch.float32)
for group in ['G1_E28_E0','G2_E43_1_E00','G3_E74_4_G2','G4_E82_4_6_10_G1']:
 torch.manual_seed(0)
 model=E83Model([145,148,3,36,24,9],3,10,4,2,0.0,group,str(E/'sources/E82_dependencies/prompts.pt')).cuda()
 opt=torch.optim.Adam(model.parameters(),lr=.012,weight_decay=.0036)
 xx=X.cuda(); yy=y.cuda(); trc=tr.cuda(); pd=(pi.cuda(),pj.cuda(),pt.cuda())
 t=time.time(); lp,aux=model(xx,yy,trc,pd if group.startswith(('G3','G4')) else None)
 loss=torch.nn.functional.nll_loss(lp[trc],yy[trc])+aux['graph_loss']*.25+aux['pair_loss']*.05+aux['preserve_loss']*.01+aux['dpsl_loss']
 loss.backward(); opt.step();
 print(group,'loss',float(loss),'lp',tuple(lp.shape),'cam',tuple(aux['cam'].shape),'graph',float(aux['graph_loss']),'pair',float(aux['pair_loss']),'dpsl',float(aux['dpsl_loss']),'sec',time.time()-t)
 print('finite',bool(torch.isfinite(loss)),'grad',float(sum((p.grad.norm() for p in model.parameters() if p.grad is not None))))
 del model,opt,lp,aux
 torch.cuda.empty_cache()

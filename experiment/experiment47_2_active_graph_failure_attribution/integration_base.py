"""Read-only SPromptGL B0-B integration for Experiment 47.1.

The source snapshot under ``.runtime/sources`` is copied from Experiment43.1
and is loaded in isolation.  This module contains no new network structure;
the only train-time changes used later are the two pre-registered optimizer or
loss interventions in ``train.py``.
"""
from __future__ import annotations
import copy, hashlib, pickle, random, sys, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# Match the historical runner before importing torch/CUDA kernels.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

ROOT=Path('/data/adodas/SPromptGL-main')
EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime'
B0=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest'
DIMS=[48,6,10,256]
SLICES=[(0,48),(48,54),(54,64),(64,320)]
MODALITIES=['PHENO','ANAT','FUNC','CORR']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/'data/ABIDE'
    frame=pd.read_csv(base/'processed_standard_data.csv')
    mapping=np.load(base/'modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert [len(mapping[k]) for k in mapping]==DIMS
    x=frame.iloc[:,:-1].to_numpy(dtype=np.float32)
    y=frame.iloc[:,-1].to_numpy(dtype=np.int64)-1
    assert x.shape==(871,320) and np.isfinite(x).all()
    assert np.bincount(y).tolist()==[403,468]
    folds=list(StratifiedKFold(n_splits=10,shuffle=True,random_state=10).split(x,y))
    return x,y,folds,mapping

def hp():
    return SimpleNamespace(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
        nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,
        n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',
        alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2',seed=0)

_CTX=None
def modules():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        src=str(RT/'sources')
        if src not in sys.path: sys.path.insert(0,src)
        import layers,network,utils
        _CTX=dict(torch=torch,layers=layers,network=network,utils=utils)
        assert not hasattr(network,'PLN')
    return _CTX

def seed_all(seed=0):
    c=modules(); t=c['torch']; random.seed(seed);np.random.seed(seed);t.manual_seed(seed);t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark=False;t.backends.cudnn.deterministic=True

def rng():
    t=modules()['torch']
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':t.get_rng_state(),'cuda':t.cuda.get_rng_state_all()}
def restore(r):
    t=modules()['torch'];random.setstate(r['python']);np.random.set_state(r['numpy']);t.set_rng_state(r['cpu']);t.cuda.set_rng_state_all([q.cpu() for q in r['cuda']])
def rng_sha(r):
    h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4))
    h.update(r['cpu'].cpu().numpy().tobytes())
    for q in r['cuda']: h.update(q.cpu().numpy().tobytes())
    return h.hexdigest()

def model_state(agent):
    return {'gfr':{k:v.detach().cpu().clone() for k,v in agent.gfr.state_dict().items()},
            'cam':{k:v.detach().cpu().clone() for k,v in agent.cam.state_dict().items()}}
def set_model(agent,state):
    agent.gfr.load_state_dict(state['gfr']);agent.cam.load_state_dict(state.get('cam',state.get('pln')))
def equal(a,b):
    t=modules()['torch']
    if isinstance(a,t.Tensor): return t.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def make_cam():
    c=modules();t=c['torch'];nn=t.nn;h=hp();layers=c['layers']
    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__();self.hyperpm=h;self.input_data_dims=DIMS;self.d_q=self.d_k=self.d_v=h.n_hidden
            self.n_head=h.n_head;self.dropout=h.dropout;self.n_layer=h.nlayer;self.modal_num=h.nmodal;self.n_class=h.nclass
            self.d_out=self.d_v*self.n_head*self.modal_num;self.fusion_weight=nn.Parameter(t.ones([6,1]))
            self.InputLayer=layers.VariLengthInputLayer(DIMS,self.d_k,self.d_v,self.n_head,self.dropout)
            self.Encoder=[];self.FeedForward=[]
            for i in range(self.n_layer):
                e=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout)
                f=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout)
                self.add_module('encode_%d'%i,e);self.add_module('feed_%d'%i,f);self.Encoder.append(e);self.FeedForward.append(f)
            self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*self.modal_num,self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
        def forward(self,x):
            bs=x.size(0);x,a0=self.InputLayer(x);a=a0.mean(dim=1)
            for i in range(self.n_layer):
                x,al=self.Encoder[i](q=x,k=x,v=x,modal_num=self.modal_num);a=al.mean(dim=1);x=self.FeedForward[i](x)
            out,hid=self.Outputlayer(x.view(bs,-1),a.view(bs,-1));return out,hid
    return PLNNoPEN()

class Agent: pass
def make_agent(fold,device='cuda:0'):
    c=modules();t=c['torch'];h=hp();x,y,folds,_=data();tr,te=folds[fold-1]
    a=Agent();a.dev=t.device(device);a.hp=h;a.x_np=x;a.y_np=y;a.train_idx=np.asarray(tr);a.test_idx=np.asarray(te);a.val_idx=a.test_idx
    a.x=t.from_numpy(x).float().to(a.dev);a.y=t.from_numpy(y).long().to(a.dev);a.parts_layer=c['layers'].InitialDataLayer(DIMS)
    count=np.bincount(y[tr],minlength=2);a.class_weight=t.from_numpy((len(tr)/(count*2)).astype(np.float32)).to(a.dev)
    a.gfr=c['network'].GFR(DIMS,DIMS,th=h.th,mode=h.GC_mode).to(a.dev);a.cam=make_cam().to(a.dev)
    a.opt_gfr=t.optim.Adam(a.gfr.parameters(),lr=h.lr,weight_decay=h.reg)
    a.opt_cam=t.optim.Adam(a.cam.parameters(),lr=h.lr,weight_decay=h.reg)
    a.cam.apply(c['utils'].my_weight_init)
    return a

def anchor(fold):
    return modules()['torch'].load(RT/'anchors'/f'fold{fold:02}.pt',map_location='cpu',weights_only=False)
def best_checkpoint(fold):
    return modules()['torch'].load(RT/'checkpoints'/f'g0_f{fold:02}.best.pt',map_location='cpu',weights_only=False)

def load_anchor_agent(fold,device='cuda:0',group='T0'):
    a=make_agent(fold,device);s=anchor(fold)
    set_model(a,{'gfr':s['model']['gfr'],'cam':s['model']['pln']})
    # Rebuild optimizer only after the exact anchor model has been installed.
    a.opt_gfr=make_gfr_optimizer(a,group);a.opt_cam=modules()['torch'].optim.Adam(a.cam.parameters(),lr=a.hp.lr,weight_decay=a.hp.reg)
    if group!='T1':
        a.opt_gfr.load_state_dict(copy.deepcopy(s['mp']))
    else:
        # The historical anchor has one GFR parameter group.  T1 has two
        # groups solely to set weight_decay=0 for the six effective GCN W
        # tensors.  Transfer the exact per-parameter Adam moments by mapping
        # old parameter ids to the new optimizer's state-dict ids.
        old=copy.deepcopy(s['mp']);new=a.opt_gfr.state_dict();params=list(a.gfr.parameters())
        old_ids=list(old['param_groups'][0]['params']); assert len(old_ids)==len(params)
        old_pid_to_obj={pid:p for pid,p in zip(old_ids,params)}
        new_obj_to_pid={p:pid for gg in new['param_groups'] for pid,p in zip(gg['params'],[q for q in a.opt_gfr.param_groups[new['param_groups'].index(gg)]['params']])}
        remap={oldpid:new_obj_to_pid[obj] for oldpid,obj in old_pid_to_obj.items()}
        new['state']={remap[k]:v for k,v in old['state'].items()}
        # Keep the anchor's scalar hyperparameters while retaining T1's
        # frozen group assignment and decay values.
        for gg in new['param_groups']:
            for key in ['lr','betas','eps','amsgrad','maximize','foreach','capturable','differentiable','fused','decoupled_weight_decay']:
                if key in old['param_groups'][0]: gg[key]=old['param_groups'][0][key]
        a.opt_gfr.load_state_dict(new)
    a.opt_cam.load_state_dict(copy.deepcopy(s['mf']))
    restore(s['rng'])
    return a,s

def make_gfr_optimizer(a,group):
    t=modules()['torch'];h=a.hp
    if group!='T1': return t.optim.Adam(a.gfr.parameters(),lr=h.lr,weight_decay=h.reg)
    targets=[];others=[]
    for n,p in a.gfr.named_parameters():
        if n.startswith(('MessagePassing1.gcn1.W','MessagePassing1.gcn2.W','MessagePassing2.gcn1.W','MessagePassing2.gcn2.W','MessagePassing3.gcn1.W','MessagePassing3.gcn2.W')):targets.append(p)
        else:others.append(p)
    return t.optim.Adam([{'params':others,'weight_decay':h.reg},{'params':targets,'weight_decay':0.0}],lr=h.lr)

def parts(a): return a.parts_layer(a.x)[1]
def raw_x(a): return modules()['torch'].cat(parts(a),dim=1)
def graph_loss(a,adjs,parts_value=None):
    c=modules();u=c['utils'];ps=parts_value if parts_value is not None else parts(a)
    return sum(u.GraphConstructLoss(ps[i],adjs[i],1.,.5,0.) for i in range(4))/4
def classification_loss(a,logp):
    return modules()['utils'].ClsLoss(logp,a.y,a.train_idx,a.class_weight)

def forward_components(a,training=False,compute_cam=True):
    """Call the unmodified GFR/CAM forward while recording GCN internals.

    ``compute_cam=False`` is used by the training path so that the operation
    order remains exactly the historical simple-2 order: GFR, graph loss,
    then CAM.  Hooks are observational and do not alter RNG consumption.
    """
    c=modules();t=c['torch'];g=a.gfr
    captured={'gcn1':{},'gcn2':{},'graph':{}}
    hooks=[]
    for i in range(4):
        gc=getattr(g,f'GraphConstruct{i}')
        hooks.append(gc.register_forward_hook(lambda m,inp,out,i=i:captured['graph'].__setitem__(i,out)))
        mp=getattr(g,f'MessagePassing{i}')
        hooks.append(mp.gcn1.register_forward_hook(lambda m,inp,out,i=i:captured['gcn1'].__setitem__(i,out)))
        hooks.append(mp.gcn2.register_forward_hook(lambda m,inp,out,i=i:captured['gcn2'].__setitem__(i,out)))
    try:
        model_num,ps=a.parts_layer(a.x)
        feat,adjs,aloss=g(ps,model_num,a.y,a.train_idx)
    finally:
        for h in hooks:h.remove()
    logp=hid=None
    if compute_cam:
        logp,hid=a.cam(feat)
    return dict(feat=feat,parts=ps,raw=t.cat(ps,1),adjs=adjs,aloss=aloss,logp=logp,hidden=hid,
                gcn1=captured['gcn1'],gcn2=captured['gcn2'],raw_scores={i:getattr(g,f'GraphConstruct{i}').score431 for i in range(4)})

def eval_components(a):
    was_gfr=a.gfr.training;was_cam=a.cam.training;a.gfr.eval();a.cam.eval()
    with modules()['torch'].no_grad():o=forward_components(a)
    if was_gfr:a.gfr.train()
    if was_cam:a.cam.train()
    return o

def metric(logp,a):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    p=logp.detach().cpu().numpy()[a.test_idx].argmax(1);y=a.y_np[a.test_idx];tp,fn,fp,tn=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    return {'acc':float((p==y).mean()),'hard_auc':float(roc_auc_score(np.eye(2)[y],np.eye(2)[p])),'spe':float(tn/(tn+fp)),'sen':float(tp/(tp+fn)), 'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,'pred':p.astype(int)}

def train_loss(a,group):
    # Keep the original operation sequence: GFR -> GraphConstructLoss -> CAM.
    o=forward_components(a,compute_cam=False);gl=graph_loss(a,o['adjs'],o['parts']);o['logp'],o['hidden']=a.cam(o['feat']);cl=classification_loss(a,o['logp']);
    total=cl if group=='T2' else cl+gl
    return o,cl,gl,total

def state_digest(a):
    h=hashlib.sha256()
    for pref,m in [('gfr',a.gfr),('cam',a.cam)]:
        for n,p in sorted(m.state_dict().items()):h.update((pref+n).encode()+p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def grad_norms(loss,params):
    t=modules()['torch'];gs=t.autograd.grad(loss,params,retain_graph=True,allow_unused=True);out=[]
    for p,g in zip(params,gs):out.append(0. if g is None else float(g.detach().norm().cpu()))
    return gs,out

def layer_manual(x,adj,mp,fw):
    """Manual exact GraphConv factorization using supplied W and adjacency."""
    t=modules()['torch'];u=modules()['utils'];eye=t.eye(adj.size(0),device=adj.device,dtype=adj.dtype);an=u.normalize_adj(adj+eye)
    w1=mp.gcn1.W if hasattr(mp,'gcn1') else mp['w1']
    w2=mp.gcn2.W if hasattr(mp,'gcn2') else mp['w2']
    z1=x@w1; q1=an@z1; h=t.relu(q1); z2=h@w2; q2=an@z2; scaled=t.sigmoid(fw)*q2
    return dict(an=an,xw1=z1,agg1=q1,relu=h,x2w2=z2,agg2=q2,scaled=scaled)

def manual_fusion(raw,branches):
    # Historical GFR overwrite is part of the diagnostic baseline.
    c=branches[0]['scaled']+raw;c=branches[1]['scaled']+raw;c=branches[2]['scaled']+c;c=branches[3]['scaled']+c
    return c

def norm(x): return float(x.detach().norm().cpu())
def spectral(x): return float(modules()['torch'].linalg.matrix_norm(x,ord=2).detach().cpu())
def rank(x):
    return int(np.linalg.matrix_rank(x.detach().cpu().numpy()))

def optimizer_one_step_delta(param,grad,state,group,weight_decay):
    """Run an isolated one-param Adam step with a copied state."""
    t=modules()['torch'];q=t.nn.Parameter(param.detach().clone());opt=t.optim.Adam([q],lr=group['lr'],betas=group['betas'],eps=group['eps'],weight_decay=weight_decay,amsgrad=group.get('amsgrad',False));
    if state: opt.state[q]=copy.deepcopy(state)
    q.grad=None if grad is None else grad.detach().clone();before=q.detach().clone();opt.step();return q.detach()-before

def optimizer_group_for(a,param):
    for g in a.opt_gfr.param_groups:
        if any(p is param for p in g['params']):return g
    raise KeyError('parameter not in optimizer')


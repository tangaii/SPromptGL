"""B0-B reconstruction plus the single PID-DGAA auxiliary alignment path."""
from __future__ import annotations
import copy, hashlib, math, random
from types import SimpleNamespace
import numpy as np

def make_pln(torch, nn, layers, hp, dims):
    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyperpm=hp; self.input_data_dims=dims
            self.d_q=self.d_k=self.d_v=hp.n_hidden; self.n_head=hp.n_head
            self.dropout=hp.dropout; self.n_layer=hp.nlayer; self.modal_num=hp.nmodal; self.n_class=hp.nclass
            self.d_out=self.d_v*self.n_head*self.modal_num
            self.fusion_weight=nn.Parameter(torch.ones([6,1]))
            self.InputLayer=layers.VariLengthInputLayer(dims,self.d_k,self.d_v,self.n_head,self.dropout)
            self.Encoder=[]; self.FeedForward=[]
            for i in range(self.n_layer):
                e=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout)
                f=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout)
                self.add_module('encode_%d'%i,e); self.add_module('feed_%d'%i,f)
                self.Encoder.append(e); self.FeedForward.append(f)
            self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*self.modal_num,self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
        def forward(self,x):
            bs=x.size(0); x,input_attention=self.InputLayer(x); attn=input_attention.mean(dim=1)
            for i in range(self.n_layer):
                x,layer_attention=self.Encoder[i](q=x,k=x,v=x,modal_num=self.modal_num)
                attn=layer_attention.mean(dim=1); x=self.FeedForward[i](x)
            return self.Outputlayer(x.view(bs,-1),attn.view(bs,-1))
    return PLNNoPEN()

class GradientReverse:
    @staticmethod
    def apply(x, strength):
        import torch
        class _GR(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, scale):
                ctx.scale=float(scale); return inp.view_as(inp)
            @staticmethod
            def backward(ctx, grad): return -ctx.scale*grad, None
        return _GR.apply(x, strength)

def build_augment(torch, nn, hp, fold):
    """Construct all E1-E6 added modules under an isolated RNG scope."""
    class Augment(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj=nn.Sequential(nn.Linear(320,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64))
            self.aux=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
            self.domain=nn.ModuleList([nn.Sequential(nn.Linear(128+2,64),nn.GELU(),nn.Linear(64,2)) for _ in range(3)])
    # Device-local fork preserves the exact B0-B RNG stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(45045 + int(fold)); torch.cuda.manual_seed_all(45045 + int(fold))
        out=Augment().cuda()
    return out

def state_hash(torch, state):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy(); h.update(str((a.dtype,a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr): h.update(repr(k).encode()); add(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(state); return h.hexdigest()

def make_agent(torch, np, layers, network, utils, hp, x, y, train, test, anchor, fold, group, cfg):
    from collections import Counter
    a=SimpleNamespace(); a.dev=torch.device('cuda'); a.hp=hp; a.group=group; a.fold=fold
    a.feat=torch.from_numpy(x).float().cuda(); a.targ=torch.from_numpy(y).long().cuda()
    a.trn_idx=np.asarray(train); a.tst_idx=np.asarray(test); a.val_idx=np.asarray(test)
    c=Counter(y[a.trn_idx]); a.weight=torch.from_numpy((len(a.trn_idx)/np.asarray(list(c.values()))/2).astype(np.float32)).cuda()
    a.dataLayer=layers.InitialDataLayer(list(cfg['dims']))
    a.gfr=network.GFR(list(cfg['dims']),list(cfg['dims']),th=hp.th,mode=hp.GC_mode).cuda()
    a.pln=make_pln(torch,torch.nn,layers,hp,list(cfg['dims'])).cuda(); a.pln.apply(utils.my_weight_init)
    a.mp=torch.optim.Adam(a.gfr.parameters(),lr=hp.lr,weight_decay=hp.reg)
    a.mf=torch.optim.Adam(a.pln.parameters(),lr=hp.lr,weight_decay=hp.reg)
    a.augment=build_augment(torch,torch.nn,hp,fold)
    a.newopt=torch.optim.Adam(a.augment.parameters(),lr=cfg['protocol']['new_lr'],weight_decay=cfg['protocol']['new_weight_decay'])
    a.gfr.load_state_dict(copy.deepcopy(anchor['model']['gfr'])); a.pln.load_state_dict(copy.deepcopy(anchor['model']['pln']))
    a.mp.load_state_dict(copy.deepcopy(anchor['mp'])); a.mf.load_state_dict(copy.deepcopy(anchor['mf']))
    a.capture={}; a.hooks=[]
    for idx in (1,2,3):
        mod=getattr(a.gfr,f'MessagePassing{idx}')
        a.hooks.append(mod.register_forward_hook(lambda module,inputs,out,idx=idx:a.capture.__setitem__(idx,out)))
    return a

def close_agent(a):
    for h in getattr(a,'hooks',[]): h.remove()

def original_gfr_step(torch, utils, a):
    a.gfr.train(); a.pln.train(); a.mp.zero_grad(); a.mf.zero_grad()
    n,parts=a.dataLayer(a.feat); feat,adjs,_=a.gfr(parts,n,a.targ,a.trn_idx)
    gl=sum(utils.GraphConstructLoss(parts[i],adjs[i],a.hp.theta_smooth,a.hp.theta_degree,a.hp.theta_sparsity) for i in range(n))/4
    out=a.pln(feat)[0]; cl=utils.ClsLoss(out,a.targ,a.trn_idx,a.weight); loss=cl+gl
    loss.backward(); a.mp.step(); a.mf.step(); return loss,cl,gl

def grl_domain(torch, z, labels, module, scale, condition):
    if scale <= 0: return z.new_zeros(())
    if condition:
        onehot=torch.nn.functional.one_hot(labels.long(),num_classes=2).float()
        inp=torch.cat([GradientReverse.apply(z,scale),onehot],dim=1)
    else: inp=torch.cat([GradientReverse.apply(z,scale),z.new_zeros((z.shape[0],2))],dim=1)
    logits=module(inp)
    src=torch.cat([torch.zeros(z.shape[0]//2,dtype=torch.long,device=z.device),torch.ones(z.shape[0]-z.shape[0]//2,dtype=torch.long,device=z.device)])
    return torch.nn.functional.cross_entropy(logits,src)

def lambda_schedule(cfg, epoch, values):
    lo,mid,hi=cfg['protocol']['adv_warmup']
    if epoch<mid:return [0.0]*3
    ramp=min(1.0,(epoch-mid)/max(1,hi-mid))
    return [float(v)*ramp for v in values]

def forward_train(torch, utils, a, cfg, pid_lambdas, epoch):
    a.capture.clear(); a.gfr.train(); a.pln.train(); a.augment.train()
    a.mp.zero_grad(); a.mf.zero_grad(); a.newopt.zero_grad()
    n,parts=a.dataLayer(a.feat); feat,adjs,aloss=a.gfr(parts,n,a.targ,a.trn_idx)
    gl=sum(utils.GraphConstructLoss(parts[i],adjs[i],a.hp.theta_smooth,a.hp.theta_degree,a.hp.theta_sparsity) for i in range(n))/4
    main_input=a.feat if a.group==6 else feat
    out=a.pln(main_input)[0]; cl=utils.ClsLoss(out,a.targ,a.trn_idx,a.weight)
    idx=torch.as_tensor(a.trn_idx,device='cuda',dtype=torch.long)
    z={m:a.augment.proj(a.capture[m]) for m in (1,2,3)}
    aux=sum(torch.nn.functional.cross_entropy(a.augment.aux(z[m][idx]),a.targ[idx]) for m in (1,2,3))/3
    modes={2:(False,pid_lambdas),3:(True,[pid_lambdas[0],pid_lambdas[1],pid_lambdas[2]]),4:(True,pid_lambdas),5:(True,[pid_lambdas[1],pid_lambdas[2],pid_lambdas[0]])}
    domain=torch.zeros((),device='cuda')
    if a.group in modes:
        cond,vals=modes[a.group]; ls=lambda_schedule(cfg,epoch,vals)
        for k,(i,j) in enumerate(((1,2),(1,3),(2,3))):
            pair=torch.cat([z[i][idx],z[j][idx]],dim=1)
            # The domain head receives the pair representation; GRL is applied to it.
            labels=a.targ[idx]; onehot=torch.nn.functional.one_hot(labels,num_classes=2).float() if cond else pair.new_zeros((pair.shape[0],2))
            din=torch.cat([GradientReverse.apply(pair,ls[k]),onehot],dim=1)
            logits=a.augment.domain[k](din)
            src=torch.cat([torch.zeros(pair.shape[0]//2,dtype=torch.long,device='cuda'),torch.ones(pair.shape[0]-pair.shape[0]//2,dtype=torch.long,device='cuda')])
            # Balance source labels deterministically by equal halves of each pair.
            domain=domain+torch.nn.functional.cross_entropy(logits,src)
        domain=domain/3
    total=cl+gl+cfg['protocol']['lambda_aux']*aux+domain
    return total,{'cls':cl,'graph':gl,'aux':aux,'domain':domain,'out':out,'feat':feat,'adjs':adjs,'z':z}

def train_step(torch, utils, a, cfg, pid_lambdas, epoch):
    total,vals=forward_train(torch,utils,a,cfg,pid_lambdas,epoch); assert torch.isfinite(total)
    total.backward()
    for module in (a.gfr,a.pln,a.augment):
        for p in module.parameters():
            if p.grad is not None: assert torch.isfinite(p.grad).all()
    a.mp.step(); a.mf.step(); a.newopt.step()
    return float(total.detach()),{k:float(v.detach()) for k,v in vals.items() if k in ('cls','graph','aux','domain')}

@__import__('contextlib').contextmanager
def no_grad():
    import torch
    with torch.no_grad(): yield

def evaluate(torch, a, use_raw=False):
    a.gfr.eval(); a.pln.eval(); a.augment.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat); feat,adjs,aloss=a.gfr(parts,n,a.targ,a.trn_idx)
        return a.pln(a.feat if use_raw else feat)[0], feat, adjs

def metric_row(out, y, test):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    pred=out.detach().cpu().numpy()[test].argmax(1); target=np.asarray(y)[test]
    tp,fn,fp,tn=map(int,confusion_matrix(target,pred,labels=[0,1]).ravel())
    return dict(acc=float((pred==target).mean()),hard_auc=float(roc_auc_score(np.eye(2)[target],np.eye(2)[pred])),spe=float(tn/(tn+fp)),sen=float(tp/(tp+fn)),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn,prediction=pred.tolist(),test_indices=np.asarray(test).tolist())

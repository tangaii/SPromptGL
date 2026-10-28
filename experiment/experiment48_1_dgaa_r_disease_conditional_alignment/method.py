from __future__ import annotations
import copy, math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIMS=[48,6,10,256]; MODALITIES=['PHENO','ANAT','FUNC','CORR']; PAIRS={'CP':(3,0),'CA':(3,1),'CF':(3,2)}

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,c): ctx.c=float(c); return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.c*g,None
def grad_reverse(x,c): return GRL.apply(x,float(c))

class PLNNoPEN(nn.Module):
    def __init__(self,layers,hp):
        super().__init__(); self.hyperpm=hp; self.input_data_dims=INPUT_DIMS; self.d_q=self.d_k=self.d_v=hp.n_hidden; self.n_head=hp.n_head; self.dropout=hp.dropout; self.n_layer=hp.nlayer; self.modal_num=4; self.n_class=2; self.d_out=self.d_v*self.n_head*4
        self.fusion_weight=nn.Parameter(torch.ones([6,1])); self.InputLayer=layers.VariLengthInputLayer(INPUT_DIMS,self.d_k,self.d_v,self.n_head,self.dropout); self.Encoder=[]; self.FeedForward=[]
        for i in range(self.n_layer):
            e=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout); self.add_module('encode_%d'%i,e); self.Encoder.append(e)
            f=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout); self.add_module('feed_%d'%i,f); self.FeedForward.append(f)
        self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*4,self.d_v*self.n_head,2,4,self.dropout)
    def forward(self,x):
        bs=x.size(0); x,ia=self.InputLayer(x); attn=ia.mean(1)
        for i in range(self.n_layer): x,la=self.Encoder[i](q=x,k=x,v=x,modal_num=4); attn=la.mean(1); x=self.FeedForward[i](x)
        hidden=self.Outputlayer.mlp_head(x.view(bs,-1)); raw=self.Outputlayer.classifier(hidden); return F.log_softmax(raw,1),raw,hidden

class SharedDiseaseHead(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,z): return self.net(z)
class DomainHead(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,z): return self.net(z)
class NewModules(nn.Module):
    def __init__(self):
        super().__init__(); self.proj=nn.Sequential(nn.LayerNorm(320),nn.Linear(320,128),nn.GELU(),nn.Linear(128,64),nn.LayerNorm(64)); self.cls=SharedDiseaseHead(); self.domains=nn.ModuleDict({k:DomainHead() for k in PAIRS})
    def project(self,hs): return [self.proj(h) for h in hs]

def hp(): return SimpleNamespace(n_hidden=18,n_head=2,dropout=.35,nlayer=1,lr=.0038,reg=.11,th=.9,GC_mode='weighted-cosine')
def build_base(torch_mod,layers,network,utils,device,x,y,tr,te):
    h=hp(); g=network.GFR(INPUT_DIMS,INPUT_DIMS,th=h.th,mode=h.GC_mode).to(device); p=PLNNoPEN(layers,h).to(device); p.apply(utils.my_weight_init); feat=torch_mod.from_numpy(x).float().to(device); targ=torch_mod.from_numpy(y).long().to(device); idx=torch_mod.as_tensor(tr,device=device); counts=torch.bincount(targ[idx],minlength=2).float(); weight=(len(tr)/counts/2).to(device)
    return SimpleNamespace(dev=device,hyperpm=h,feat=feat,targ=targ,trn_idx=tr,tst_idx=te,val_idx=te,weight=weight,dataLayer=layers.InitialDataLayer(INPUT_DIMS),MessagePassing0=g,ModalFusion=p)
def base_state(a): return {'gfr':{k:v.detach().cpu().clone() for k,v in a.MessagePassing0.state_dict().items()},'pln':{k:v.detach().cpu().clone() for k,v in a.ModalFusion.state_dict().items()}}
def load_base(a,s): a.MessagePassing0.load_state_dict(s['gfr']); a.ModalFusion.load_state_dict(s['pln'])
def make_agent(torch_mod,layers,network,utils,device,x,y,tr,te,fold,gid,anchor,seed):
    a=build_base(torch_mod,layers,network,utils,device,x,y,tr,te); load_base(a,anchor['model']); a.optimizer_MP0=torch_mod.optim.Adam(a.MessagePassing0.parameters(),lr=.0038,weight_decay=.11); a.optimizer_MF=torch_mod.optim.Adam(a.ModalFusion.parameters(),lr=.0038,weight_decay=.11); a.optimizer_MP0.load_state_dict(copy.deepcopy(anchor['mp'])); a.optimizer_MF.load_state_dict(copy.deepcopy(anchor['mf']))
    cr=torch_mod.get_rng_state(); cg=torch_mod.cuda.get_rng_state_all(); torch_mod.manual_seed(seed); torch_mod.cuda.manual_seed_all(seed); a.new=NewModules().to(device); torch_mod.set_rng_state(cr); torch_mod.cuda.set_rng_state_all(cg); a.optimizer_new=torch_mod.optim.Adam([p for n,p in a.new.named_parameters() if not n.startswith('domains.')],lr=.00038,weight_decay=0.); a.optimizer_dom=torch_mod.optim.Adam(a.new.domains.parameters(),lr=.00038,weight_decay=0.); return a
def lambda_adv(epoch): return 0. if epoch<45 else (.03*(epoch-45)/45 if epoch<90 else .03)
def graph_loss(a,utils,prop_perms=None):
    n,parts=a.dataLayer(a.feat); combined,adjs,aloss=a.MessagePassing0(parts,n,a.targ,a.trn_idx,prop_perms=prop_perms); gl=sum(utils.GraphConstructLoss(parts[i],adjs[i],1.,.5,0.) for i in range(4))/4; lp,raw,hid=a.ModalFusion(combined); return parts,combined,adjs,gl,lp,raw,hid
def domain_loss(a,z,enabled,conditional,lam,condition_labels):
    if not enabled or lam<=0:return None,0.,0.,[]
    idx=torch.as_tensor(a.trn_idx,device=z[0].device); losses=[]; acc=[]
    for pair in enabled:
        ia,ib=PAIRS[pair]
        groups=[torch.ones(len(idx),dtype=torch.bool,device=idx.device)] if not conditional else [condition_labels[idx]==0,condition_labels[idx]==1]
        for mask in groups:
            if int(mask.sum())<2: continue
            za,zb=z[ia][idx][mask],z[ib][idx][mask]; inp=torch.cat([za,zb]); lab=torch.cat([torch.zeros(len(za),dtype=torch.long,device=inp.device),torch.ones(len(zb),dtype=torch.long,device=inp.device)]); logits=a.new.domains[pair](grad_reverse(inp,lam)); losses.append(F.cross_entropy(logits,lab)); acc.append(float(logits.detach().argmax(1).eq(lab).float().mean()))
    if not losses:return None,0.,0.,[]
    return torch.stack(losses).mean(),float(torch.stack(losses).mean().detach()),sum(acc)/len(acc),losses
def model_bundle(a): return {'base':base_state(a),'new':{k:v.detach().cpu().clone() for k,v in a.new.state_dict().items()}}
def load_bundle(a,s): load_base(a,s['base']); a.new.load_state_dict(s['new'])
def forward(a,utils,prop_perms=None):
    parts,combined,adjs,gl,lp,raw,hid=graph_loss(a,utils,prop_perms); hs=list(a.MessagePassing0.raw_gcn_outputs); return parts,combined,adjs,gl,lp,raw,hid,hs
def aux_weights(gid): return [1/6,1/6,1/6,.5] if gid>=5 else [.25]*4
def train_step(a,gid,epoch,utils,perm_condition=None,prop_perms=None):
    # evaluate() deliberately switches the modules to eval mode; every new
    # optimization step must restore the original training-mode dropout and
    # normalization behavior before the forward pass.
    a.MessagePassing0.train(); a.ModalFusion.train(); a.new.train()
    for o in [a.optimizer_MP0,a.optimizer_MF,a.optimizer_new,a.optimizer_dom]:o.zero_grad(set_to_none=True)
    parts,comb,adjs,gl,lp,raw,hid,hs=forward(a,utils,prop_perms); z=a.new.project(hs); tr=torch.as_tensor(a.trn_idx,device=z[0].device); ws=aux_weights(gid); ce=[F.cross_entropy(a.new.cls(z[i])[tr],a.targ[tr],weight=a.weight) for i in range(4)]; aux=.1*sum(w*l for w,l in zip(ws,ce)); main=utils.ClsLoss(lp,a.targ,a.trn_idx,a.weight); task=main+gl+aux
    conditional=gid in (3,4,5,6,7); enabled=['CP','CA','CF'] if gid in (2,3) else (['CF'] if gid in (4,5,6,7) else []); labels=a.targ if perm_condition is None else perm_condition; dl,domain,da,_=domain_loss(a,z,enabled,conditional,lambda_adv(epoch),labels)
    loss=task+(dl if dl is not None else task.new_zeros(())); loss.backward();
    assert all(torch.isfinite(p.grad).all() for m in [a.MessagePassing0,a.ModalFusion,a.new] for p in m.parameters() if p.grad is not None)
    a.optimizer_MP0.step();a.optimizer_MF.step();a.optimizer_new.step();
    if dl is not None:a.optimizer_dom.step()
    return {'total':float(loss.detach()),'main':float(main.detach()),'graph':float(gl.detach()),'aux':float(aux.detach()),'domain':domain,'domain_acc':da,'lambda_adv':lambda_adv(epoch),'gfr_norms':[float(h.norm().detach()) for h in hs],'gc_weight_norms':[float((m.gcn1.W.norm()+m.gcn2.W.norm()).detach()) for m in [a.MessagePassing0.MessagePassing0,a.MessagePassing0.MessagePassing1,a.MessagePassing0.MessagePassing2,a.MessagePassing0.MessagePassing3]]}
@torch.no_grad()
def evaluate(a,gid,utils,prop_perms=None):
    a.MessagePassing0.eval(); a.ModalFusion.eval()
    parts,comb,adjs,gl,lp,raw,hid,hs=forward(a,utils,prop_perms); z=a.new.project(hs) if gid else None; return {'logp':lp,'raw':raw,'parts':parts,'combined':comb,'adjs':adjs,'graph':gl,'hs':hs,'z':z,'aux_logits':[a.new.cls(q) for q in z] if z else None}
def metrics(out,y,te):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    p=out['logp'].detach().cpu().numpy()[te].argmax(1); tar=y[te]; tp,fn,fp,tn=map(int,confusion_matrix(tar,p,labels=[0,1]).ravel()); one=torch.eye(2).numpy(); return {'acc':float((p==tar).mean()),'hard_auc':float(roc_auc_score(one[tar],one[p])),'spe':float(tn/(tn+fp)),'sen':float(tp/(tp+fn)),'asd_tp':tp,'asd_fn':fn,'td_fp':fp,'td_tn':tn,'probability_auc':float(roc_auc_score(tar,out['logp'].detach().cpu().numpy()[te,1]))}

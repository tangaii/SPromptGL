"""D3-SAG: evidence-constrained four-component, selective alignment.

The E1 GFR is kept as an untouched submodule.  The added branch is patient
wise and does not construct a second patient graph.  It receives CORR and the
non-site PHENO view separately, and writes only bounded, explicitly routed
S/C residuals back to the original 320-dimensional CAM input.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from d3sag_base import LegacyFixedGFR, PLNNoPEN, DIMS

def cov(x,y):
    x=x-x.mean(0,keepdim=True); y=y-y.mean(0,keepdim=True)
    return (x.T@y)/max(1,x.shape[0])

def rms(x): return torch.sqrt(x.pow(2).mean(1,keepdim=True).clamp_min(1e-8))

class FourHead(nn.Module):
    def __init__(self, d):
        super().__init__(); self.enc=nn.Sequential(nn.Linear(d,64),nn.GELU())
        self.out=nn.Linear(64,64)
    def forward(self,x): return self.out(self.enc(x)).view(x.shape[0],4,16)

class D3SAG(nn.Module):
    def __init__(self, graphlearn_cls, gcn_cls, layers_module, utils_module, th=.9, mode='E5'):
        super().__init__(); self.mode=mode; self.utils=utils_module
        self.base=LegacyFixedGFR(graphlearn_cls,gcn_cls,utils_module,th=th)
        self.fc_head=FourHead(256); self.ph_head=FourHead(28)
        self.c_joint=nn.Sequential(nn.Linear(48,32),nn.GELU(),nn.Linear(32,16,bias=False))
        self.core=nn.Linear(32,2)
        self.write_s_fc=nn.Linear(16,256,bias=False); self.write_s_ph=nn.Linear(16,28,bias=False)
        self.write_c_fc=nn.Linear(16,256,bias=False); self.write_c_ph=nn.Linear(16,28,bias=False)
        # 1% initial, 5% maximum.  No bias can bypass a zero route.
        self.rho_s=nn.Parameter(torch.full((2,),-math.log(4.0)))
        self.rho_c=nn.Parameter(torch.full((2,),-math.log(4.0)))
        self.last={}

    def _views(self,parts,pheno_perm=None):
        corr=parts[3]
        p=parts[0]
        if pheno_perm is not None: p=p[pheno_perm]
        # Non-site indices are fixed by the audited feature-name mapping and
        # stored as a buffer-like tensor by the runner.
        return corr,p

    def forward(self,parts,evidence,pheno_nosite,mode=None,align_block=None,wrong_gate=False,quality_off=False,wrong_geometry=False):
        mode=mode or self.mode
        base,adjs,bd=self.base(parts)
        corr,p=self._views(parts,evidence.get('pheno_perm'))
        zf=self.fc_head(corr); zp=self.ph_head(pheno_nosite)
        S_F,C_F,I_F,N_F=[zf[:,k] for k in range(4)]; S_P,C_P,I_P,N_P=[zp[:,k] for k in range(4)]
        # D3 routes are fold-local and fixed before training; no test labels.
        rs=float(evidence.get('rS',0.0)); rc=float(evidence.get('rC',0.0))
        if mode=='E2': rs=rc=1.0
        elif mode=='E3': rs=rc=1.0
        if mode=='E6_UNIFORM': pass
        if mode=='E7_NO_C': rc=0.0
        if mode=='E10_WRONG_GATE':
            rs,rc=float(evidence.get('rS_wrong',rs)),float(evidence.get('rC_wrong',rc))
        q=evidence['qrel'].to(base.device).float()
        if quality_off: q=torch.ones_like(q)
        if wrong_gate: q=evidence.get('q_wrong',q).to(base.device).float()
        # Quality enters only the relation weighting, never the disease view.
        core_c=self.c_joint(torch.cat([C_F,C_P,C_F*C_P],1))
        core=torch.cat([rs*S_F,rc*core_c],1)
        # Bounded residuals, with exact zero when the evidence route is zero.
        rs_scale=.05*torch.sigmoid(self.rho_s[0]); rc_scale=.05*torch.sigmoid(self.rho_c[0])
        sfc=rs*rs_scale*rms(parts[3]).detach()*torch.tanh(self.write_s_fc(S_F))
        spv=torch.zeros((len(parts[0]),48),device=base.device,dtype=base.dtype); spv[:,evidence['pheno_local_idx'].long()]=rs*rs_scale*torch.tanh(self.write_s_ph(S_P))
        cfc=rc*rc_scale*rms(parts[3]).detach()*torch.tanh(self.write_c_fc(core_c))
        cpv=torch.zeros((len(parts[0]),48),device=base.device,dtype=base.dtype); cpv[:,evidence['pheno_local_idx'].long()]=rc*rc_scale*torch.tanh(self.write_c_ph(core_c))
        # S/C residuals are deliberately written to PHENO non-site and CORR;
        # site columns remain untouched by the added branch.
        add=torch.cat([spv+cpv,torch.zeros((len(base),6+10),device=base.device),sfc+cfc],1)
        fused=base+add
        self.last={'base':base,'fused':fused,'zf':zf,'zp':zp,'S_F':S_F,'S_P':S_P,'C_F':C_F,'C_P':C_P,
                   'I_F':I_F,'I_P':I_P,'N_F':N_F,'N_P':N_P,'core':core,'core_c':core_c,'q':q,
                   'rs':rs,'rc':rc,'res_s':sfc,'res_c':cfc,'adjs':adjs,'base_details':bd,'add':add}
        return fused,adjs,self.last

    def losses(self, y, tr, evidence, align_block=None, mode=None, wrong_geometry=False, uniform=False, enable_align=True, enable_short=True, enable_site=True):
        z=self.last; idx=tr.long(); yy=y[idx]
        zero=z['fused'].new_zeros(())
        core_loss=F.cross_entropy(self.core(z['core'][idx]),yy)
        # I/N are diagnostic shortcuts only in the sense of finite covariance
        # proxies; these penalties do not claim mutual-information removal.
        short=(cov(torch.cat([z['I_F'][idx],z['I_P'][idx],z['N_F'][idx],z['N_P'][idx]],1),F.one_hot(yy,2).float())**2).mean() if enable_short else zero
        dec=(cov(z['C_F'][idx],z['S_F'][idx])**2).mean()+(cov(z['C_P'][idx],z['S_P'][idx])**2).mean()
        site_loss=zero
        if enable_site and 'site_onehot' in evidence:
            site=evidence['site_onehot'].to(z['fused'].device)[idx].float()
            for yyv in (0,1):
                sel=yy==yyv
                if sel.sum()>3: site_loss=site_loss+(cov(z['core'][idx][sel],site[sel])**2).mean()
        align=zero; pair_count=0
        if enable_align and align_block is not None and float(z['rs'])>0:
            ids=align_block.long().to(z['fused'].device); sf=z['S_F'][ids]; sp=z['S_P'][ids]
            df=torch.cdist(sf,sf); dp=torch.cdist(sp,sp)
            off=~torch.eye(len(ids),dtype=torch.bool,device=ids.device)
            df=df/(df[off].pow(2).mean().sqrt().detach()+1e-6); dp=dp/(dp[off].pow(2).mean().sqrt().detach()+1e-6)
            q=z['q'][ids]; w=(float(z['rs'])*q[:,None]*q[None,:])
            if 'k_block' in evidence: w=w*evidence['k_block'].to(w.device)
            if uniform:
                # Match E5's total budget on this same fixed pair set.
                old=w[off].mean().detach(); w=torch.zeros_like(w)+old
            if wrong_geometry:
                perm=evidence['geometry_perm'].to(dp.device); dp=dp[perm][:,perm]
            den=w[off].sum().detach()+1e-6; align=(w[off].detach()*(df[off]-dp[off]).pow(2)).sum()/den; pair_count=int(off.sum())
        return core_loss,short,dec,site_loss,align,pair_count

def load_e1_anchor(model, anchor_gfr):
    model.base.fusion_weight.data.copy_(anchor_gfr['fusion_weight'])
    for m in range(4):
        gsrc={k[len(f'GraphConstruct{m}.'):]:v for k,v in anchor_gfr.items() if k.startswith(f'GraphConstruct{m}.')}
        msrc={k[len(f'MessagePassing{m}.'):]:v for k,v in anchor_gfr.items() if k.startswith(f'MessagePassing{m}.')}
        model.base.graphs[m].load_state_dict(gsrc,strict=True); model.base.gcn[m].load_state_dict(msrc,strict=True)

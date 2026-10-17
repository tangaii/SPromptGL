"""Experiment 22: signed, fold-supervised, concept-conditional FC evidence.

Strictly a WEAKLY SUPERVISED disease-knowledge prior: the original dataset has
no verified patient-level symptom supervision. No patient/test labels are read.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CONDITIONS = ('REAL', 'NUMERIC', 'NEUTRAL', 'WRONG_ANATOMY', 'ISOMETRIC')
VARIANTS = ('SIGNED', 'UNSIGNED', 'SHARED')
INTERVENTIONS = ('REAL', 'ADAPT_OFF', 'GRAPH_OFF', 'GRAPH_SEMANTIC_OFF',
                 'GRAPH_STRENGTH_OFF', 'BRANCH_OFF')


def _masked_softmax(logits, mask):
    assert mask.ndim == 2 and logits.shape[-2:] == mask.shape
    return F.softmax(logits.masked_fill(~mask[None], -1e4), dim=-1)


class SignedConceptEvidence22(nn.Module):
    """Drop-in PLN adapter, operating only on CORR latent slot (index 3).

    P: static CLIP difference prior; mask: verified anatomical candidate set.
    Signed reliability derives ONLY from outer-training labels and FC values.
    All patient effects have bounded per-concept parameters; no free text Key.
    """
    def __init__(self, *, condition, use_graph, cache, raw_corr,
                 train_idx, train_labels, complement=False, variant='SIGNED',
                 kl_weight=0.002):
        super().__init__()
        if condition not in CONDITIONS or variant not in VARIANTS:
            raise ValueError((condition, variant))
        if raw_corr.ndim != 2 or raw_corr.shape[1] != 256:
            raise AssertionError('CORR must be [N,256]')
        n = raw_corr.shape[0]
        ids = torch.as_tensor(train_idx, dtype=torch.long, device=raw_corr.device)
        labels = torch.as_tensor(train_labels, dtype=torch.long, device=raw_corr.device)
        if ids.ndim != 1 or labels.shape != ids.shape:
            raise AssertionError('Only outer-training labels may be passed')
        if len(ids.unique()) != len(ids) or len(ids) >= n or (ids < 0).any() or (ids >= n).any():
            raise AssertionError('Bad training indices')
        if set(labels.cpu().tolist()) != {0, 1}:
            raise AssertionError('ASD=0/TD=1 expected in outer train')
        self.condition = condition
        self.variant = variant
        self.use_graph = bool(use_graph)
        self.complement = bool(complement)
        self.intervention = 'REAL'
        self.kl_weight = float(kl_weight)
        self.register_buffer('valid_idx', torch.as_tensor(cache['valid_idx'], dtype=torch.long))
        self.register_buffer('mask', torch.as_tensor(cache['masks_wrong' if condition == 'WRONG_ANATOMY' else 'masks'], dtype=torch.bool))
        prior_key = {'REAL':'prior_real', 'NUMERIC':None, 'NEUTRAL':'prior_neutral',
                     'WRONG_ANATOMY':'prior_real', 'ISOMETRIC':'prior_random'}[condition]
        prior = torch.zeros(4,252) if prior_key is None else torch.as_tensor(cache[prior_key], dtype=torch.float32)
        self.register_buffer('prior', prior)
        support = torch.as_tensor(cache['line_graph'],dtype=torch.bool)
        if support.shape != (252,252) or not torch.equal(support,support.T) or not bool(support.diag().all()):
            raise AssertionError('Invalid verified FC edge line graph')
        if self.prior.shape != (4,252) or self.mask.shape != (4,252) or not bool(self.mask.any(dim=1).all()):
            raise AssertionError('Incorrect prior/masks')
        self.register_buffer('support',support)
        links=support.clone();links.fill_diagonal_(False)
        dst,src=links.nonzero(as_tuple=True)
        self.register_buffer('dst',dst)
        self.register_buffer('src',src)
        self.register_buffer('train_idx',ids.cpu())
        with torch.no_grad():
            train_z=raw_corr.index_select(1,self.valid_idx.to(raw_corr.device)).index_select(0,ids)
            center=train_z.mean(0);scale=train_z.std(0,unbiased=False).clamp_min(1e-3)
            z0=train_z[labels==0];z1=train_z[labels==1]
            # Balanced label-dependent effect size; folds carry their own estimates.
            var=(z0.var(0,unbiased=False)+z1.var(0,unbiased=False))/2
            contrast=(z0.mean(0)-z1.mean(0))/(var+1e-3).sqrt()
            # Shrink extreme effects; no tuning with held-out labels.
            reliability=torch.tanh(contrast/0.4)
        self.register_buffer('center',center.float().cpu())
        self.register_buffer('scale',scale.float().cpu())
        self.register_buffer('train_contrast',contrast.float().cpu())
        self.register_buffer('reliability',reliability.float().cpu())
        self.encoder=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.LayerNorm(32))
        self.edge_id=nn.Embedding(252,32)
        self.edge_norm=nn.LayerNorm(32)
        if self.use_graph:
            self.graph_proj=nn.Linear(32,32,bias=False)
            self.graph_norm=nn.LayerNorm(32)
        else:
            self.graph_proj=None;self.graph_norm=None
        self.readout=nn.Sequential(nn.Linear(136,64),nn.GELU(),nn.Linear(64,36))
        self.alpha=nn.Parameter(torch.zeros(()))
        # Four domain-specific positive/negative channel intensities, starting .1.
        init=math.log(.10/.25)
        self.pos_logits=nn.Parameter(torch.full((4,),init))
        self.neg_logits=nn.Parameter(torch.full((4,),init))
        self._loss=None
        self.last={}

    def _read(self, raw):
        z=raw.index_select(1,self.valid_idx)
        d=((z-self.center)/self.scale).clamp(-6,6)
        gate=torch.sigmoid(2*(d.abs()-1)).clamp_min(1e-5)
        return z,d,gate

    def _attention(self,d):
        # Correlation direction is a training-label association, not pathology.
        signed=torch.tanh(d/2)*self.reliability[None]
        plus=0.35*self.pos_logits.sigmoid()
        minus=0.35*self.neg_logits.sigmoid()
        if self.variant=='SHARED':
            plus=plus.mean().expand(4);minus=minus.mean().expand(4)
        if self.variant=='UNSIGNED':
            correction=.5*(plus+minus)[None,:,None]*signed.abs()[:,None,:]
        else:
            correction=plus[None,:,None]*signed.clamp_min(0)[:,None,:] - \
                       minus[None,:,None]*(-signed).clamp_min(0)[:,None,:]
        # A content-specific, fixed prior also controls signed patient correction.
        correction=correction*(1+0.5*self.prior)[None]
        if self.intervention=='ADAPT_OFF':correction=correction*0.
        base_logits=self.prior/0.8
        att0=_masked_softmax(base_logits[None],self.mask)
        att=_masked_softmax(base_logits[None]+correction,self.mask)
        p0=att0.detach()
        kl=(att*(att.clamp_min(1e-9).log()-p0.clamp_min(1e-9).log())).sum(-1)
        self._loss=self.kl_weight*kl.index_select(0,self.train_idx).mean()
        return att,p0,kl,signed,correction

    def _message(self,h,att,gate):
        src,dst=self.src,self.dst
        w=(gate[:,src]*gate[:,dst]).clamp_min(1e-8)
        # Avoid the E21 normalized-domain-profile amplitude collapse.
        semantic_profile=F.normalize(att.transpose(1,2),dim=-1,eps=1e-8)
        cos=(semantic_profile[:,src]*semantic_profile[:,dst]).sum(-1)
        # Sum-k attention retains absolute importance even for single-domain edges.
        strength=att.sum(1)
        # Mean over non-empty anatomical candidate union, per patient.
        union=self.mask.any(0)
        scale=strength[:,union].mean(-1,keepdim=True).clamp_min(1e-8)
        logstrength=(strength/scale).clamp_min(1e-8).log().clamp(-2,2)
        if self.condition!='NUMERIC' and self.intervention!='GRAPH_SEMANTIC_OFF':
            w=w*torch.exp(0.4*(cos-0.5))
            if self.intervention!='GRAPH_STRENGTH_OFF':
                w=w*torch.exp(0.45*(logstrength[:,src]+logstrength[:,dst])*0.5)
        denom=h.new_ones((len(h),252)).index_add(1,dst,w)
        neighbour=torch.zeros_like(h).index_add(1,dst,h[:,src]*w.unsqueeze(-1))
        mixed=(h+neighbour)/denom.unsqueeze(-1)
        updated=self.graph_norm(h+F.gelu(self.graph_proj(mixed)))
        return updated,dict(cos=cos.detach(),strength=strength.detach(),logstrength=logstrength.detach(),
                            mean_abs_change=(updated-h).norm(dim=-1).mean().detach())

    def forward(self,base,raw_modalities):
        if base.ndim!=3 or tuple(base.shape[1:])!=(4,36):
            raise AssertionError('Attach after PLN Encoder+FF, x=[N,4,36]')
        if not isinstance(raw_modalities,(list,tuple)) or len(raw_modalities)!=4:
            raise AssertionError('Original 4 raw modalities required')
        corr=raw_modalities[3]
        if tuple(corr.shape)!=(base.shape[0],256):raise AssertionError('CORR shape')
        z,d,gate=self._read(corr)
        x=torch.stack([torch.tanh(d/3),torch.tanh(d.abs()/3),z.tanh()],dim=-1)
        h=self.edge_norm(self.encoder(x)+self.edge_id.weight[None])
        att,p0,kl,signed,correction=self._attention(d)
        ginfo={}
        if self.use_graph and self.intervention!='GRAPH_OFF':h,ginfo=self._message(h,att,gate)
        values=torch.einsum('nke,ned->nkd',att,h)
        signed_feature=torch.einsum('nke,ne->nk',att,signed)
        coverage=torch.einsum('nke,ne->nk',att,gate)
        features=torch.cat([values.flatten(1),signed_feature,coverage],dim=-1)
        out=self.readout(features)
        if self.complement:
            # Orthogonal to current CORR latent token, not a guarantee of E9 novelty.
            vec=base[:,3];out=out-(out*vec).sum(-1,keepdim=True)/(vec.square().sum(-1,keepdim=True)+1e-8)*vec
        direction=F.normalize(out,dim=-1,eps=1e-8)
        delta=.1*self.alpha.tanh()*base[:,3].norm(dim=-1,keepdim=True).detach()*direction
        if self.intervention=='BRANCH_OFF':delta=delta*0.
        pad=torch.zeros_like(delta)
        self.last={'att':att.detach(),'p0':p0.detach(),'kl':kl.detach(),
                   'eta_plus':(0.35*self.pos_logits.sigmoid()).detach(),
                   'eta_minus':(0.35*self.neg_logits.sigmoid()).detach(),
                   'correction':correction.detach(),'signed':signed.detach(),
                   'reliability':self.reliability.detach(),
                   'delta':delta.detach(),'graph':ginfo}
        return base+torch.stack([pad,pad,pad,delta],dim=1)

    def auxiliary_loss(self):
        if self._loss is None:raise RuntimeError('Run forward before auxiliary_loss')
        return self._loss


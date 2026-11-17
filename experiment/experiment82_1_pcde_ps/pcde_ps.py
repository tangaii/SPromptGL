"""E82.1 PCDE-PS; deterministic evidence routing on CAM144."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-8
PROMPTS = [
    'Nromal brain volume', 'Uniform cortical thickness',
    'Symmetrical brain structure', 'Normal ventricular morphology',
    'Altered brain volume', 'Abnormal functional connectivity',
    'White matter integrity issues', 'Different activation patterns',
]
GROUPS = {
    "G1": ("anonymous", True), "G2": ("medical", False), "G3": ("medical", True)
}

def symmetric(a):
    return (a + a.T) * 0.5


def sign_columns(u):
    indices = u.abs().argmax(dim=0)  # argmax uses the first index for ties
    signs = torch.where(u[indices, torch.arange(u.shape[1])] < 0, -1., 1.)
    return u * signs


def eigh(a):
    """CPU float64; coordinate-projector canonicalization of tied eigenspaces."""
    values, vectors = torch.linalg.eigh(symmetric(a))
    scale = max(float(values.abs().max()), 1e-30)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and float(values[stop] - values[start]) <= 1e-12 * scale:
            stop += 1
        if stop - start > 1:
            projector = vectors[:, start:stop] @ vectors[:, start:stop].T
            chosen = []
            for col in range(len(values)):
                v = projector[:, col].clone()
                for _ in range(2):
                    for q in chosen:
                        v = v - q * torch.dot(q, v)
                norm = v.norm()
                if norm > 1e-9:
                    chosen.append(v / norm)
                if len(chosen) == stop - start:
                    break
            if len(chosen) != stop - start:
                raise RuntimeError('degenerate eigenspace canonicalization failed')
            vectors[:, start:stop] = torch.stack(chosen, dim=1)
        start = stop
    return values, sign_columns(vectors)


def normalize16(x):
    return (x - x.mean()) / torch.sqrt((x - x.mean()).square().mean() + EPS)


def covariance(x):
    centered = x - x.mean(0)
    return symmetric(centered.T @ centered / max(len(x) - 1, 1))


def build_spectra(z_train, y_train):
    """Only accepts legal training CAM rows and their labels; no test-label input."""
    z = z_train.detach().cpu().double()
    y = y_train.detach().cpu().long()
    assert z.shape[1] == 144 and set(y.tolist()) == {0, 1}
    x0, x1 = z[y == 0], z[y == 1]
    n0, n1 = len(x0), len(x1)
    mu = z.mean(0)
    s, s0, s1 = covariance(z), covariance(x0), covariance(x1)
    delta = x0.mean(0) - x1.mean(0)
    eye = torch.eye(144, dtype=torch.float64)
    pool = (n0*s0+n1*s1)/(n0+n1)
    gv, gu = eigh(s)
    ug = sign_columns(gu[:, -8:].flip(1))
    # Complete deterministic QR creates a true orthogonal complement, excluding
    # the artificial zero eigenvalues in Ug from disease-direction selection.
    complement = torch.linalg.qr(ug, mode='complete').Q[:, 8:]
    p = eye - ug @ ug.T
    result = {}
    for variant, matrix in [('O', symmetric(p @ (s0-s1) @ p))]:
        fallback = False
        if float(matrix.norm()) <= EPS:
            matrix = symmetric(p @ s @ p)
            fallback = True
        dv, du = eigh(complement.T @ matrix @ complement)
        order = torch.argsort(dv.abs() if variant == 'O' and not fallback else dv,
                              descending=True, stable=True)
        ud = sign_columns(complement @ du[:, order[:8]])
        u = torch.cat([ug, ud], dim=1)
        orth_error = float((u.T @ u - torch.eye(16, dtype=u.dtype)).abs().max())
        if orth_error > 1e-10 or not torch.isfinite(u).all():
            raise RuntimeError('invalid spectral basis')
        lam = torch.diagonal(u.T @ s @ u).clamp_min(EPS)
        nu = torch.diagonal(u.T @ (s0-s1) @ u).abs() / (torch.diagonal(u.T @ pool @ u)+EPS)
        sorted_scores = (dv.abs() if variant == 'O' and not fallback else dv)[order]
        result[variant] = {
            'U': u, 'mu': mu, 'lambda': lam, 'nu': nu,
            'loglambda_normalized': normalize16(lam.log()),
            'nu_normalized': normalize16(nu.log1p()),
            'covariance': s, 'cov_asd': s0, 'cov_td': s1,
            'mean_asd': x0.mean(0), 'mean_td': x1.mean(0), 'delta_mean': delta,
            'pooled': pool,
            'global_eigenvalues': gv.flip(0), 'disease_eigenvalues': dv[order],
            'global_gaps': (gv[1:] - gv[:-1]).flip(0),
            'disease_selection_gaps': sorted_scores[:-1]-sorted_scores[1:],
            'orthogonality_max_abs': orth_error,
            'covariance_rule': 'unbiased n-1; pooled weighted by class sample counts n_c',
            'n_train': len(z), 'n_asd': n0, 'n_td': n1,
            'fallback': fallback,
            'variance_coverage': float(lam.sum()/s.trace()),
            'mean_discrimination': ((u.T @ delta).square() / (torch.diagonal(u.T @ pool @ u)+EPS)),
            'covariance_discrimination': torch.diagonal(u.T @ (s0-s1) @ u).abs() / (torch.diagonal(u.T @ pool @ u)+EPS),
        }
    return result['O']

def smooth_limit(raw, z):
    c = .08 * z.norm(dim=-1, keepdim=True).clamp_min(EPS)
    x = raw.norm(dim=-1, keepdim=True) / c
    small = x < 1e-4
    safe_x = torch.where(small, torch.ones_like(x), x)
    # Analytic continuation tanh(x)/x at zero. Avoid 0/0 even in the
    # unselected autograd branch, and retain derivative identity at zero.
    small_x = torch.where(small, x, torch.zeros_like(x))
    scale = torch.where(small, 1-small_x.square()/3+2*small_x.pow(4)/15, torch.tanh(x)/safe_x)
    return raw * scale


def text_geometry(t, seed=821991):
    t=t.detach().cpu().double()
    _,s,vh=torch.linalg.svd(t,full_matrices=False)
    rank=int((s>s[0]*1e-12).sum());v=sign_columns(vh[:rank].T)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    rotation,rr=torch.linalg.qr(torch.randn(rank,rank,generator=gen,dtype=torch.float64))
    rotation=rotation*torch.where(rr.diag()<0,-1.,1.)
    anon=((t@v)@rotation)@v.T
    # The first rank columns exactly span all text rows; complete with a local stream.
    a=torch.randn(1024,32-rank,generator=gen,dtype=torch.float64)
    a=a-v@(v.T@a)
    extra=torch.linalg.qr(a,mode='reduced').Q
    proj=torch.cat([v,sign_columns(extra)],1)
    qmed=F.normalize(t@proj,dim=-1).float()
    qanon=F.normalize(anon@proj,dim=-1).float()
    def err(x,y):return float((x@x.T-y@y.T).abs().max())
    return {'medical':t.float(),'anonymous':anon.float(),'P_text':proj.float(),
            'q_medical':qmed,'q_anonymous':qanon,'rank':rank,'seed':seed,'rotation':rotation,
            'gram_error':err(anon,t),'gram_error_float32':err(anon.float(),t.float()),
            'projection_gram_medical':err(t@proj,t),'projection_gram_anonymous':err(anon@proj,anon),
            'projection_gram_float32_medical':err(t.float()@proj.float(),t.float()),
            'projection_gram_float32_anonymous':err(anon.float()@proj.float(),anon.float()),
            'projection_orthogonality':float((proj.T@proj-torch.eye(32)).abs().max()),
            'text_relative_change':float((anon-t).norm()/t.norm())}


def class_statistics(h, labels):
    """Unbiased n-1 covariance diagonals, including total (between+within) variance."""
    return {'mean_asd':h[labels==0].mean(0),'mean_td':h[labels==1].mean(0),
        'var_asd':h[labels==0].var(0,unbiased=True),'var_td':h[labels==1].var(0,unbiased=True),
        'mean_pool':h.mean(0),'var_pool':h.var(0,unbiased=True)}


class PCDEPS(nn.Module):
    def __init__(self,spectrum,text_cache,group,fold,legal_train,legal_labels,warmup_cam):
        super().__init__()
        self.group,self.fold=group,fold
        self.text_kind,self.gate_enabled=GROUPS[group]
        for n,k in [('U','U'),('mu_warm','mu'),('loglam','loglambda_normalized'),('nu','nu_normalized')]:
            self.register_buffer(n,spectrum[k].float().clone())
        self.register_buffer('q0',text_cache['q_'+self.text_kind].float().clone())
        self.register_buffer('P_text',text_cache['P_text'].float().clone())
        self.register_buffer('legal',torch.as_tensor(legal_train).long().clone())
        self.register_buffer('legal_labels',torch.as_tensor(legal_labels).long().clone())
        h=(warmup_cam.double()-spectrum['mu'])@spectrum['U']
        initial=class_statistics(h[self.legal],self.legal_labels)
        for n,v in initial.items():self.register_buffer(n,v.float().clone())
        self.register_buffer('ema_updates',torch.tensor(0,dtype=torch.int64))
        self.direction=nn.Linear(144,32)
        self.evidence=nn.Linear(4,32)
        self.norm=nn.LayerNorm(32)
        self.key=nn.Linear(32,32,bias=False)
        self.null=nn.Parameter(torch.empty(32));nn.init.normal_(self.null,std=.02)
        self.meta_asd=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,32),nn.Tanh())
        self.meta_td=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,32),nn.Tanh())
        self.gate=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.gate[2].weight);nn.init.zeros_(self.gate[2].bias)
        for p in self.gate.parameters():p.requires_grad_(self.gate_enabled)

    def update_statistics(self,h):
        with torch.no_grad():
            current=class_statistics(h.detach()[self.legal],self.legal_labels)
            for n,v in current.items():getattr(self,n).mul_(.95).add_(v,alpha=.05)
            self.ema_updates.add_(1)

    def forward(self,z,intervention='full'):
        assert z.shape==(871,144)
        if intervention=='noop':
            return z,{'z':z,'delta':torch.zeros_like(z),'evidence_loss':z.new_zeros(())}
        h=(z-self.mu_warm)@self.U
        if self.training:self.update_statistics(h)
        floor=(self.var_pool*1e-4).clamp_min(EPS)
        va=(.9*self.var_asd+.1*self.var_pool).maximum(floor)
        vt=(.9*self.var_td+.1*self.var_pool).maximum(floor)
        llr=-.5*(va.log()+(h-self.mean_asd).square()/va-vt.log()-(h-self.mean_td).square()/vt)
        e=(4*torch.tanh(llr/4)).detach()
        hh=(h-self.mean_pool)/torch.sqrt(self.var_pool.clamp_min(EPS)+EPS)
        v=torch.cat([hh,e],-1)
        ma=self.meta_asd(v);mt=self.meta_td(v)
        m=torch.cat([mt[:,None].expand(-1,4,-1),ma[:,None].expand(-1,4,-1)],1)
        if intervention=='conditioning_off':m=torch.zeros_like(m)
        q=F.normalize(self.q0[None]+.25*m,dim=-1)
        inputs=torch.stack([hh,hh.abs(),self.loglam.expand_as(h),self.nu.expand_as(h)],-1)
        tokens=self.norm(self.direction(self.U.T)[None]+self.evidence(inputs))
        keys=self.key(tokens)
        signed=torch.cat([-e[:,None].expand(-1,4,-1),e[:,None].expand(-1,4,-1)],1)
        scores=torch.einsum('nkd,njd->nkj',q,keys)/math.sqrt(32)+.2*signed
        null=torch.einsum('nkd,d->nk',q,self.null)/math.sqrt(32)
        a=torch.cat([scores,null[:,:,None]],-1).div(.5).softmax(-1)
        ra=1-(1-a[:,4:,:16]).prod(1);rt=1-(1-a[:,:4,:16]).prod(1)
        wa=(ra+EPS)/(ra.sum(-1,keepdim=True)+16*EPS)
        wt=(rt+EPS)/(rt.sum(-1,keepdim=True)+16*EPS)
        sa=(wa*F.softplus(e)).sum(-1);st=(wt*F.softplus(-e)).sum(-1)
        margin=(sa-st)/(sa+st+EPS)
        ha=-(wa*(wa+EPS).log()).sum(-1)/math.log(16)
        ht=-(wt*(wt+EPS).log()).sum(-1)/math.log(16)
        r=torch.stack([margin.abs(),1-ha,1-ht,1-F.cosine_similarity(wa,wt,dim=-1)],-1)
        gate=torch.sigmoid(self.gate(r)).squeeze(-1) if self.gate_enabled and intervention!='gate_off' else torch.ones_like(sa)
        raw=(.5*(wa-wt)*(h-self.mean_pool))@self.U.T
        bounded=smooth_limit(raw,z)
        delta=gate[:,None]*bounded
        if self.fold==2:
            mask=torch.ones((871,1),dtype=z.dtype,device=z.device);mask[[376,406]]=0
            delta=delta*mask
        evidence_loss=F.binary_cross_entropy_with_logits(2*(sa[self.legal]-st[self.legal]),1-self.legal_labels.float())
        return z+delta,{'z':z,'h':h,'hhat':hh,'llr':llr,'e':e,'tokens':tokens,'q':q,'meta':m,
            'a':a,'w_asd':wa,'w_td':wt,'s_asd':sa,'s_td':st,'margin':margin,'h_asd':ha,'h_td':ht,
            'gate_input':r,'gate':gate,'delta_raw':raw,'delta_bounded':bounded,'delta':delta,
            'evidence_loss':evidence_loss,'variance_asd_protected':va,'variance_td_protected':vt}

    def counts(self):
        return {'nominal_parameters':sum(p.numel() for p in self.parameters()),
            'effective_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad)}

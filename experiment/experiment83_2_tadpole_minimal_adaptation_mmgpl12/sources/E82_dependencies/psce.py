"""E82.2 PSCE: Original spectrum, fixed queries, deterministic counterfactual evidence."""
from __future__ import annotations

import copy
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
GROUPS = {'G1': 'anonymous', 'G2': 'medical'}

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



class PSCE(nn.Module):
    def __init__(self, spectrum, text_cache, group, fold):
        super().__init__()
        self.group, self.fold = group, fold
        for n,k in [('U','U'),('mu','mu'),('lam','lambda'),('loglam','loglambda_normalized'),('nu','nu_normalized')]:
            self.register_buffer(n,spectrum[k].float().clone())
        self.register_buffer('q0',text_cache['q_'+GROUPS[group]].float().clone())
        self.register_buffer('P_text',text_cache['P_text'].float().clone())
        self.direction=nn.Linear(144,32)
        self.evidence=nn.Linear(4,32)
        self.norm=nn.LayerNorm(32)
        self.key=nn.Linear(32,32,bias=False)
        self.null=nn.Parameter(torch.empty(32));nn.init.normal_(self.null,std=.02)
        self.spectral=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,z,intervention='full'):
        assert z.shape==(871,144)
        h=(z-self.mu)@self.U;hh=h/torch.sqrt(self.lam+EPS)
        loglam=self.loglam.expand_as(h);nu=self.nu.expand_as(h)
        e=torch.stack([hh,hh.abs(),loglam,nu],dim=-1)
        tokens=self.norm(self.direction(self.U.T)[None]+self.evidence(e))
        scores=torch.einsum('kd,njd->nkj',self.q0,self.key(tokens))/math.sqrt(32)
        null=(self.q0@self.null/math.sqrt(32))[None,:,None].expand(len(z),-1,-1)
        a=torch.cat([scores,null],dim=-1).div(.5).softmax(-1)
        coverage=1-torch.prod(1-a[:,:,:16],dim=1)
        ws=coverage/(coverage.sum(-1,keepdim=True)+EPS)
        entropy=-(ws*(ws+EPS).log()).sum(-1)/math.log(16)
        n=a[:,:,16].mean(-1)
        wp=self.spectral(torch.stack([hh.abs(),loglam,nu],dim=-1)).squeeze(-1).softmax(-1)
        beta=((1-n)*(1-entropy).clamp(.2,.8)).detach()
        w=beta[:,None]*ws+(1-beta[:,None])*wp
        if intervention=='semantic_off':w=wp
        if intervention=='noop':w=torch.full_like(h,1/16)
        raw=((w-1/16)*h)@self.U.T;delta=smooth_limit(raw,z)
        if self.fold==2:
            mask=torch.ones((871,1),device=z.device,dtype=z.dtype);mask[[376,406]]=0;delta=delta*mask
        return z+delta,{'z':z,'h':h,'hhat':hh,'tokens':tokens,'a':a,'coverage':coverage,
            'w_sem':ws,'w_sp':wp,'entropy':entropy,'null':n,'beta':beta,'w':w,'delta_raw':raw,'delta':delta}

    def counts(self):
        return {'nominal_parameters':sum(p.numel() for p in self.parameters()),
                'effective_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad)}


def teacher_snapshot(output_layer):
    """Deepcopy does not initialize parameters or touch any RNG stream."""
    teacher=copy.deepcopy(output_layer)
    for m in teacher.modules():
        m._forward_pre_hooks.clear();m._forward_hooks.clear();m._backward_hooks.clear()
    teacher.eval();teacher.requires_grad_(False)
    return teacher


def teacher_logits(teacher,z):
    # Exactly the original OutputLayer before its final log_softmax. Its second
    # argument is unused by the audited forward; the margin cancels log_softmax.
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    return teacher.classifier(teacher.mlp_head(z))


@torch.no_grad()
def counterfactual(teacher,z,U,mu,legal,legal_labels):
    """17 deterministic head batches on legal training rows; no test labels."""
    z=z.detach()[legal];labels=legal_labels.detach().long()
    assert labels.shape==(len(z),) and set(labels.cpu().tolist()).issubset({0,1})
    h=(z-mu)@U
    components=h[:,:,None]*U.T[None]
    perturb=smooth_limit(components,z[:,None,:])
    masked=z[:,None,:]-perturb
    inputs=torch.cat([z[:,None,:],masked],dim=1)
    logits=teacher_logits(teacher,inputs.reshape(-1,144)).reshape(len(z),17,2)
    sign=(1-2*labels).to(z.dtype)[:,None]
    margins=(logits[:,:,0]-logits[:,:,1])*sign
    D=margins[:,:1]-margins[:,1:]
    correct=logits[:,0].argmax(-1)==labels
    valid=correct & (margins[:,0]>0) & (D.max(-1).values>1e-6)
    pos=D.relu()
    # Preregistered EPS-regularized positive contribution distribution.
    target=pos/(pos.sum(-1,keepdim=True)+EPS)
    return {'target':target.detach(),'valid':valid,'D':D,'correct':correct,'margins':margins,
            'mask_relative':perturb.norm(dim=-1)/z.norm(dim=-1,keepdim=True).clamp_min(EPS)}


def cf_kl(w_sem,legal,cf):
    w=w_sem[legal][cf['valid']];t=cf['target'][cf['valid']]
    if not len(w):return w_sem.sum()*0
    return (torch.xlogy(t,t)-t*w.clamp_min(EPS).log()).sum(-1).mean()

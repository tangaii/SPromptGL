"""Frozen E82 patient spectral evidence selection; no stochastic forward ops."""
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
SPECS = {
    'G1': ('O', 'spectral', 'medical'), 'G2': ('O', 'full', 'medical'),
    'G3': ('I', 'spectral', 'medical'), 'G4': ('I', 'semantic', 'medical'),
    'G5': ('I', 'full', 'medical'), 'G6': ('I', 'full', 'anonymous'),
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
    s0s, s1s = .9*s0 + .1*s0.trace()/144*eye, .9*s1 + .1*s1.trace()/144*eye
    pool, pools = (n0*s0+n1*s1)/(n0+n1), (n0*s0s+n1*s1s)/(n0+n1)
    gv, gu = eigh(s)
    ug = sign_columns(gu[:, -8:].flip(1))
    # Complete deterministic QR creates a true orthogonal complement, excluding
    # the artificial zero eigenvalues in Ug from disease-direction selection.
    complement = torch.linalg.qr(ug, mode='complete').Q[:, 8:]
    p = eye - ug @ ug.T
    d = p @ delta
    c = symmetric(p @ (s0s - s1s) @ p)
    dd, cc = torch.outer(d, d), symmetric(c @ c)
    tm, tc = float(dd.trace()), float(cc.trace())
    dmean = dd/(tm+EPS) if tm > EPS else torch.zeros_like(dd)
    dcov = cc/(tc+EPS) if tc > EPS else torch.zeros_like(cc)
    di = symmetric(.5*dmean + .5*dcov)
    result = {}
    for variant, matrix in [('O', symmetric(p @ (s0-s1) @ p)), ('I', di)]:
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
        if variant == 'O':
            nu = torch.diagonal(u.T @ (s0-s1) @ u).abs() / (torch.diagonal(u.T @ pool @ u)+EPS)
        else:
            denom = torch.diagonal(u.T @ pools @ u) + EPS
            nu = .5*(u.T @ delta).square()/denom + .5*torch.diagonal(u.T @ (s0s-s1s) @ u).abs()/denom
        sorted_scores = (dv.abs() if variant == 'O' and not fallback else dv)[order]
        result[variant] = {
            'U': u, 'mu': mu, 'lambda': lam, 'nu': nu,
            'loglambda_normalized': normalize16(lam.log()),
            'nu_normalized': normalize16(nu.log1p()),
            'covariance': s, 'cov_asd': s0, 'cov_td': s1,
            'cov_asd_shrunk': s0s, 'cov_td_shrunk': s1s,
            'mean_asd': x0.mean(0), 'mean_td': x1.mean(0), 'delta_mean': delta,
            'pooled': pool, 'pooled_shrunk': pools,
            'global_eigenvalues': gv.flip(0), 'disease_eigenvalues': dv[order],
            'global_gaps': (gv[1:] - gv[:-1]).flip(0),
            'disease_selection_gaps': sorted_scores[:-1]-sorted_scores[1:],
            'orthogonality_max_abs': orth_error,
            'covariance_rule': 'unbiased n-1; pooled weighted by class sample counts n_c',
            'n_train': len(z), 'n_asd': n0, 'n_td': n1,
            'mean_term_zero': tm <= EPS, 'cov_term_zero': tc <= EPS,
            'mean_term_trace': tm, 'cov_term_trace': tc, 'fallback': fallback,
            'variance_coverage': float(lam.sum()/s.trace()),
            'mean_discrimination': ((u.T @ delta).square() / (torch.diagonal(u.T @ pool @ u)+EPS)),
            'covariance_discrimination': torch.diagonal(u.T @ (s0-s1) @ u).abs() / (torch.diagonal(u.T @ pool @ u)+EPS),
        }
    singular = torch.linalg.svdvals(result['O']['U'].T @ result['I']['U']).clamp(0, 1)
    result['comparison'] = {'principal_angles_degrees': torch.acos(singular)*180/math.pi,
        'projector_frobenius': float((result['O']['U'] @ result['O']['U'].T - result['I']['U'] @ result['I']['U'].T).norm())}
    return result


def anonymous_text(t, seed=829991):
    t = t.cpu().double()
    _, singular, vh = torch.linalg.svd(t, full_matrices=False)
    rank = int((singular > singular[0]*1e-12).sum())
    basis = vh[:rank].T
    generator = torch.Generator(device='cpu').manual_seed(seed)
    a = torch.randn((rank, rank), generator=generator, dtype=torch.float64)
    rotation, r = torch.linalg.qr(a)
    rotation = rotation * torch.where(r.diag() < 0, -1., 1.)
    anon = ((t @ basis) @ rotation) @ basis.T
    return anon.float(), {'rank': rank, 'rotation': rotation, 'basis': basis,
        'gram_error_float64': float((anon @ anon.T-t @ t.T).abs().max()),
        'gram_error_float32': float((anon.float() @ anon.float().T-t.float() @ t.float().T).abs().max()),
        'relative_text_change': float((anon-t).norm()/t.norm()), 'seed': seed}


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


class PSSEC(nn.Module):
    def __init__(self, spectrum, text, group, fold):
        super().__init__()
        self.group, self.fold = group, fold
        _, self.mode, _ = SPECS[group]
        for name, key in [('U', 'U'), ('mu', 'mu'), ('lam', 'lambda'),
                          ('loglam', 'loglambda_normalized'), ('nu', 'nu_normalized')]:
            self.register_buffer(name, spectrum[key].float().clone())
        self.register_buffer('text', text.float().clone())
        self.direction = nn.Linear(144, 32)
        self.evidence = nn.Linear(4, 32)
        self.norm = nn.LayerNorm(32)
        self.text_proj = nn.Linear(1024, 32, bias=False)
        self.key = nn.Linear(32, 32, bias=False)
        self.null = nn.Parameter(torch.empty(32))
        nn.init.normal_(self.null, std=.02)
        self.spectral = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        for name, parameter in self.named_parameters():
            active = name.startswith('spectral.') if self.mode == 'spectral' else not name.startswith('spectral.') if self.mode == 'semantic' else True
            parameter.requires_grad_(active)

    def forward(self, z, intervention='full'):
        assert z.shape == (871, 144)
        h = (z - self.mu) @ self.U
        hh = h / torch.sqrt(self.lam + EPS)
        loglam = self.loglam.expand_as(h)
        nu = self.nu.expand_as(h)
        semantic_on = self.mode != 'spectral' and intervention != 'semantic_off'
        spectral_on = self.mode != 'semantic' or intervention == 'semantic_off'
        details = {'z': z, 'h': h, 'hhat': hh}
        if semantic_on:
            e = torch.stack([hh, hh.abs(), loglam, nu], dim=-1)
            tokens = self.norm(self.direction(self.U.T)[None] + self.evidence(e))
            q = F.normalize(self.text_proj(self.text), dim=-1)
            k = self.key(tokens)
            scores = torch.einsum('kd,njd->nkj', q, k) / math.sqrt(32)
            # q@null has shape [8], explicitly broadcast as [N,8,1].
            null = (q @ self.null / math.sqrt(32))[None, :, None].expand(len(z), -1, -1)
            a = torch.cat([scores, null], dim=-1).div(.5).softmax(-1)
            coverage = 1 - torch.prod(1-a[:, :, :16], dim=1)
            ws = coverage / (coverage.sum(-1, keepdim=True) + EPS)
            entropy = -(ws * (ws+EPS).log()).sum(-1) / math.log(16)
            n = a[:, :, 16].mean(-1)
            details.update(tokens=tokens, a=a, coverage=coverage, w_sem=ws, entropy=entropy, null=n)
        if spectral_on:
            v = torch.stack([hh.abs(), loglam, nu], dim=-1)
            wp = self.spectral(v).squeeze(-1).softmax(-1)
            details['w_sp'] = wp
        if intervention == 'noop':
            w = torch.full_like(h, 1/16)
        elif self.mode == 'spectral' or intervention == 'semantic_off':
            w = wp
        elif self.mode == 'semantic':
            w = ws
        else:
            beta = ((1-n) * (1-entropy).clamp(.2, .8)).detach()
            w = beta[:, None]*ws + (1-beta[:, None])*wp
            details['beta'] = beta
        raw = ((w-1/16)*h) @ self.U.T
        delta = smooth_limit(raw, z)
        if self.fold == 2:
            mask = torch.ones((871, 1), device=z.device, dtype=z.dtype)
            mask[[376, 406]] = 0
            delta = delta * mask
        details.update(w=w, delta_raw=raw, delta=delta)
        return z + delta, details

    def counts(self):
        return {'nominal_parameters': sum(p.numel() for p in self.parameters()),
                'effective_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)}

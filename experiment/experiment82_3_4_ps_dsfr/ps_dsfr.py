"""Class-conditional spectral feature residual for the historical CAM144."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-8
FOLD2_ROWS = (376, 406)
PROMPTS = [
    'Nromal brain volume', 'Uniform cortical thickness',
    'Symmetrical brain structure', 'Normal ventricular morphology',
    'Altered brain volume', 'Abnormal functional connectivity',
    'White matter integrity issues', 'Different activation patterns',
]
EMA_BUFFERS = ('ema_mean', 'ema_second', 'ema_counts', 'ema_updates', 'class_mean', 'class_var')
SPECTRUM_BUFFERS = {
    'U': 'U', 'mu': 'mu', 'lam': 'lambda',
    'loglam': 'loglambda_normalized', 'nu': 'nu_normalized',
}


def smooth_limit(raw, z, bound):
    """E82.2 radial tanh limit with a true relative bound for tiny CAM rows."""
    length = torch.linalg.vector_norm(z.double(), dim=-1, keepdim=True)
    safe_length = length.clamp_min(EPS)
    raw64 = raw.double()
    x = torch.linalg.vector_norm(raw64, dim=-1, keepdim=True) / (bound * safe_length)
    small = x < 1e-4
    x_small = torch.where(small, x, torch.zeros_like(x))
    x_right = torch.where(small, torch.ones_like(x), x)
    scale = torch.where(small, 1 - x_small.square() / 3 + 2 * x_small.pow(4) / 15,
                        x.tanh() / x_right)
    relative = length / safe_length
    representable = length / (length + torch.finfo(z.dtype).tiny)
    return (raw64 * scale * relative * representable).to(z.dtype)


def class_evidence(attention, llr, reliability):
    # Prompt cache is TD, ASD. The class axis returned here is ASD=0, TD=1.
    by_class = torch.stack((attention[:, 4:8], attention[:, 0:4]), dim=1)
    coverage = 1 - (1 - by_class[..., :16]).prod(dim=2)
    null = by_class[..., 16].mean(dim=2)
    semantic = (1 - null[..., None]) * coverage / (coverage.sum(dim=-1, keepdim=True) + EPS)
    semantic_difference = semantic[:, 0] - semantic[:, 1]
    modifier = reliability * (llr * (16 * semantic_difference).tanh()).sigmoid()
    evidence = semantic * modifier[:, None]
    difference = evidence[:, 0] - evidence[:, 1]
    return coverage, null, semantic, evidence, difference


class PSDSFR(nn.Module):
    def __init__(self, spectrum, query, stats, fold, config):
        super().__init__()
        self.fold = fold
        self.cfg = config
        stat = config['class_statistics']
        for name, key in SPECTRUM_BUFFERS.items():
            self.register_buffer(name, spectrum[key].float().clone())
        self.register_buffer('q0', query.float().clone())
        self.register_buffer('class_mean', stats['mean'].float().clone())
        self.register_buffer('class_var', stats['stable'].float().clone())
        self.register_buffer('ema_mean', stats['mean'].clone())
        counts = stats['counts'].clone()
        fraction = (counts[:, None] - stat['variance_ddof']) / counts[:, None].double()
        self.register_buffer('ema_second', stats['variance'] * fraction + stats['mean'].square())
        self.register_buffer('ema_counts', counts)
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))

        # Retain the 6081 E82.3.2 parameters and their construction order.
        self.direction = nn.Linear(144, 32)
        self.evidence = nn.Linear(4, 32)
        self.norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, 32, bias=False)
        self.null = nn.Parameter(torch.empty(32))
        nn.init.normal_(self.null, std=.02)
        self.spectral = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def initialize_gate(self):
        gate = self.cfg['signed_gate']
        self.signed_gate = nn.Sequential(nn.Linear(gate['input'], gate['hidden']),
                                         nn.GELU(), nn.Linear(gate['hidden'], gate['output']))
        nn.init.zeros_(self.signed_gate[-1].weight)
        nn.init.zeros_(self.signed_gate[-1].bias)

    def parameter_groups(self):
        groups = {'prompt': [], 'spectral': [], 'signed_gate': []}
        for name, param in self.named_parameters():
            prefix = name.split('.')[0]
            groups[prefix if prefix in groups else 'prompt'].append(param)
        return groups

    @torch.no_grad()
    def update_calibration(self, obs):
        from history import stabilize_variance

        counts = self.ema_counts.cpu()
        assert torch.equal(counts, obs['counts'])
        alpha = self.cfg['ema']['alpha']
        mean = (1 - alpha) * self.ema_mean.cpu() + alpha * obs['mean']
        second = (1 - alpha) * self.ema_second.cpu() + alpha * obs['second']
        stat = self.cfg['class_statistics']
        unbias = counts[:, None].double() / (counts[:, None] - stat['variance_ddof'])
        variance = (second - mean.square()).clamp_min(0) * unbias
        _, stable = stabilize_variance(variance, counts, stat)
        assert torch.isfinite(mean).all() and torch.isfinite(second).all()
        self.ema_mean.copy_(mean)
        self.ema_second.copy_(second)
        self.class_mean.copy_(mean.float())
        self.class_var.copy_(stable.float())
        self.ema_updates.add_(1)

    def forward(self, z, intervention='full'):
        assert z.shape == (self.cfg['subjects'], 144)
        assert intervention in ('full', 'residual_off', 'gate_off', 'semantic_off')
        h = z @ self.U
        hh = ((z - self.mu) @ self.U) / (self.lam + EPS).sqrt()
        loglam, nu = self.loglam.expand_as(hh), self.nu.expand_as(hh)
        tokens = self.norm(self.direction(self.U.T)[None] +
                           self.evidence(torch.stack((hh, hh.abs(), loglam, nu), dim=-1)))
        temperature = self.cfg['temperature']
        dot = torch.einsum('kd,njd->nkj', self.q0, self.key(tokens)) / math.sqrt(32)
        null_scores = (self.q0 @ self.null / math.sqrt(32))[None, :, None].expand(len(z), -1, -1)
        attention = torch.cat((dot, null_scores), dim=-1).div(temperature).softmax(-1)

        asd = (hh - self.class_mean[0]).square() / self.class_var[0] + self.class_var[0].log()
        td = (hh - self.class_mean[1]).square() / self.class_var[1] + self.class_var[1].log()
        raw_llr = .5 * (td - asd)
        llr_limit = self.cfg['class_statistics']['llr_bound']
        llr = llr_limit * (raw_llr / llr_limit).tanh()
        w_sp = self.spectral(torch.stack((hh.abs(), loglam, nu), dim=-1)).squeeze(-1).softmax(-1)
        reliability = .5 + .5 * (16 * w_sp).tanh()
        coverage, null, semantic, evidence, difference = class_evidence(attention, llr, reliability)
        if intervention == 'semantic_off':
            difference = torch.zeros_like(difference)
            null = torch.zeros_like(null)
        normalized = F.layer_norm(z, (144,), eps=self.cfg['signed_gate']['layer_norm_eps'])
        gate_input = torch.cat((normalized, difference, reliability, null), dim=-1)
        gate = self.signed_gate(gate_input).tanh()
        if intervention == 'gate_off':
            gate = torch.zeros_like(gate)
        spectral_residual = gate * difference * h
        raw = spectral_residual @ self.U.T
        delta = smooth_limit(raw, z, self.cfg['residual_limit'])
        if self.fold == 2:
            delta = delta.index_fill(0, torch.tensor(FOLD2_ROWS, device=z.device), 0)
        if intervention == 'residual_off':
            delta = torch.zeros_like(delta)
        enhanced = z + delta
        return enhanced, dict(z=z, h=h, hh=hh, tokens=tokens, attention=attention, coverage=coverage,
                              null=null, semantic=semantic, raw_llr=raw_llr, llr=llr, w_sp=w_sp,
                              reliability=reliability, evidence=evidence, difference=difference,
                              gate_input=gate_input, gate=gate, spectral_residual=spectral_residual,
                              raw=raw, delta=delta, z_enhanced=enhanced)

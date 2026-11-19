"""Differentiable 16D CAM144 subspace and medical prompt residual."""
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
EMA_KEYS = ('ema_mean', 'ema_second', 'ema_counts', 'ema_updates')
INTERVENTIONS = ('full', 'residual_off', 'spectral_guidance_off', 'semantic_evidence_off')


@torch.no_grad()
def observed_moments(legal_cam, legal_targets):
    x = legal_cam.detach().cpu().double()
    y = legal_targets.detach().cpu().long()
    assert x.shape == (len(y), 144)
    parts = [x[y == label] for label in (0, 1)]
    counts = torch.tensor([len(rows) for rows in parts], dtype=torch.long)
    assert counts.min() > 1 and torch.isfinite(x).all()
    return dict(ema_mean=torch.stack([rows.mean(0) for rows in parts]),
                ema_second=torch.stack([rows.T @ rows / len(rows) for rows in parts]),
                ema_counts=counts)


def covariance_moments(mean, second, counts):
    count = counts.double()
    n = count.sum()
    center = (count[:, None] * mean).sum(0) / n
    total_second = (count[:, None, None] * second).sum(0) / n
    within = (second - mean[:, :, None] * mean[:, None, :]) * (count / (count - 1))[:, None, None]
    total = (total_second - torch.outer(center, center)) * (n / (n - 1))
    return center, (within + within.mT) * 0.5, (total + total.T) * 0.5


def normalize_directions(x):
    deviation = x - x.mean()
    return deviation / (deviation.square().mean() + EPS).sqrt()


def smooth_limit(raw, z, bound):
    """Radial tanh applied in FP64, including near-zero CAM rows."""
    length = z.double().norm(dim=-1, keepdim=True)
    stable_length = length.clamp_min(EPS)
    raw64 = raw.double()
    x = raw64.norm(dim=-1, keepdim=True) / (bound * stable_length)
    near_zero = x < 1e-4
    tiny_x = torch.where(near_zero, x, torch.zeros_like(x))
    divisor = torch.where(near_zero, torch.ones_like(x), x)
    scale = torch.where(near_zero, 1 - tiny_x.square() / 3 + 2 * tiny_x.pow(4) / 15, x.tanh() / divisor)
    return (raw64 * scale * (length / stable_length) *
            (length / (length + torch.finfo(z.dtype).tiny))).to(z.dtype)


def deterministic_logits(head, z):
    ln, affine, gelu, dropout = head.mlp_head
    assert isinstance(dropout, nn.Dropout) and isinstance(gelu, nn.GELU)
    hidden = F.layer_norm(z, ln.normalized_shape, ln.weight.detach(), ln.bias.detach(), ln.eps)
    hidden = F.gelu(F.linear(hidden, affine.weight.detach(), affine.bias.detach()), approximate=gelu.approximate)
    return F.linear(hidden, head.classifier.weight.detach(), head.classifier.bias.detach())


def head_jacobian(head, z):
    with torch.enable_grad():
        features = z.detach().requires_grad_(True)
        logits = deterministic_logits(head, features)
        gap = logits[:, 0] - logits[:, 1]
        jacobian, = torch.autograd.grad(gap.sum(), features, create_graph=False)
    return jacobian.detach()


def class_prompt_evidence(attention):
    ordered = torch.stack((attention[:, 4:8], attention[:, :4]), dim=1)
    coverage = 1 - (1 - ordered[..., :16]).prod(dim=2)
    null = ordered[..., 16].mean(dim=2)
    class_response = (1 - null[..., None]) * coverage / (coverage.sum(-1, keepdim=True) + EPS)
    return class_response[:, 0] - class_response[:, 1], null


def true_margin(logp, targets):
    return (logp[:, 0] - logp[:, 1]) * (1 - 2 * targets)


def balanced_mean(values, targets):
    return (values[targets == 0].mean() + values[targets == 1].mean()) * 0.5


def gaussian_separation(basis, stats, config):
    mean = stats['ema_mean'].to(basis)
    second = stats['ema_second'].to(basis)
    counts = stats['ema_counts'].to(basis.device)
    _, within, total = covariance_moments(mean, second, counts)
    stat = config['class_statistics']
    floor = (stat['relative_floor'] * total.trace() / total.size(0)).clamp_min(stat['variance_floor'])
    eye = torch.eye(basis.size(1), device=basis.device, dtype=basis.dtype)
    covariance = basis.T @ (stat['own_variance'] * within + stat['pooled_variance'] * total) @ basis
    covariance = (covariance + covariance.mT) * 0.5 + floor * eye
    middle = covariance.mean(0)
    factors = torch.linalg.cholesky(torch.cat((covariance, middle[None]), dim=0))
    diff = (mean[0] - mean[1]) @ basis
    solved = torch.linalg.solve_triangular(factors[2], diff[:, None], upper=False)
    mean_part = solved.square().sum() / 8
    logdets = 2 * factors.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    cov_part = ((logdets[2] - logdets[:2].mean()) * 0.5).clamp_min(0)
    return mean_part + cov_part, mean_part, cov_part


def difficulty_matrix(per_patient_grad, margin, targets, config):
    hp = config['loss']
    gradient = per_patient_grad.detach().double()
    difficulty = (hp['g2_weight_floor'] + (1 - hp['g2_weight_floor']) *
                  (hp['g2_margin_center'] - margin.detach().double()).sigmoid())
    parts = []
    for label in (0, 1):
        selected = targets == label
        g, w = gradient[selected], difficulty[selected]
        parts.append(g.T @ (w[:, None] * g) / w.sum())
    covariance = torch.stack(parts).mean(0)
    return (covariance + covariance.T) * 0.5


def gradient_coverage(basis, covariance):
    return ((basis * (covariance @ basis)).sum() / covariance.trace().clamp_min(1e-12))


class PSDSO(nn.Module):
    def __init__(self, u0, query, moments, fold, config):
        super().__init__()
        self.cfg = config
        self.fold = fold
        self.register_buffer('U0', u0.double().clone())
        self.register_buffer('eye16', torch.eye(16, dtype=torch.float64))
        self.register_buffer('q0', query.float().clone())
        self.register_buffer('tau_s', torch.ones(()))
        self.register_buffer('tau_l', torch.ones(()))
        for name, value in moments.items():
            self.register_buffer(name, value.clone())
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))
        self.offset = nn.Parameter(torch.zeros_like(self.U0))
        self.direction = nn.Linear(144, 32)
        self.evidence = nn.Linear(4, 32)
        self.norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, 32, bias=False)
        self.null_key = nn.Parameter(torch.empty(32))
        nn.init.normal_(self.null_key, std=0.02)
        self.spectral = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def initialize_fusion(self):
        self.fusion_net = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1))
        nn.init.zeros_(self.fusion_net[-1].weight)
        nn.init.zeros_(self.fusion_net[-1].bias)

    def initialize_gate(self):
        self.gate_net = nn.Sequential(nn.Linear(178, 16), nn.GELU(), nn.Linear(16, 16))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def parameter_groups(self):
        groups = {key: [] for key in ('offset', 'prompt', 'spectral', 'fusion_net', 'gate_net')}
        for name, parameter in self.named_parameters():
            prefix = name.split('.')[0]
            groups[prefix if prefix in groups else 'prompt'].append(parameter)
        return groups

    def basis(self, fixed=False):
        if fixed:
            return self.U0
        tangent = self.offset - self.U0 @ (self.U0.T @ self.offset)
        radius = self.cfg['stiefel']['offset_frobenius_radius']
        offset = tangent / (1 + tangent.square().sum() / radius**2).sqrt()
        gram = self.eye16 + offset.T @ offset
        lower = torch.linalg.cholesky(gram)
        return torch.linalg.solve_triangular(lower, (self.U0 + offset).T, upper=False).T

    def ema_state(self):
        return {name: getattr(self, name).detach().cpu().clone() for name in EMA_KEYS}

    @torch.no_grad()
    def update_ema(self, observation):
        assert torch.equal(self.ema_counts.cpu(), observation['ema_counts'])
        alpha = self.cfg['ema']['alpha']
        for key in ('ema_mean', 'ema_second'):
            field = getattr(self, key)
            field.copy_((1 - alpha) * field + alpha * observation[key].to(field))
        self.ema_updates.add_(1)

    def projection(self, z, basis):
        center, within, total = covariance_moments(self.ema_mean, self.ema_second, self.ema_counts)
        variation = (basis * (total @ basis)).sum(0).clamp_min(EPS)
        scale = (variation + EPS).sqrt()
        projected = (z.double() - center) @ basis
        hh = projected / scale
        loc = ((self.ema_mean - center) @ basis) / scale
        projected_cov = basis.T @ within @ basis
        directional_cov = projected_cov.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        w = self.ema_counts.double()[:, None] / self.ema_counts.sum()
        within_pooled = (w * directional_cov).sum(0)
        variance = directional_cov / scale.square()
        pooled = (w * variance).sum(0)
        hp = self.cfg['class_statistics']
        floor = (hp['relative_floor'] * pooled).clamp_min(hp['variance_floor'])
        stable = hp['own_variance'] * variance + hp['pooled_variance'] * pooled + floor
        loglam = normalize_directions(variation.log()).to(z.dtype)
        contrast = (directional_cov[0] - directional_cov[1]).abs() / (within_pooled + EPS)
        nu = normalize_directions(contrast.log1p()).to(z.dtype)
        return dict(hh=hh.to(z.dtype), projected_cov=projected_cov, class_mean=loc, class_var=stable,
                    hh_double=hh,
                    loglam=loglam, nu=nu, variation=variation)

    def evidence_parts(self, z, basis):
        d = self.projection(z, basis)
        hh, loglam, nu = d['hh'], d['loglam'], d['nu']
        loglam, nu = loglam.expand_as(hh), nu.expand_as(hh)
        basis_tokens = self.direction(basis.T.to(z.dtype))[None]
        tokens = self.norm(basis_tokens + self.evidence(torch.stack((hh, hh.abs(), loglam, nu), dim=-1)))
        dot = torch.einsum('kd,njd->nkj', self.q0, self.key(tokens)) / math.sqrt(32)
        null_scores = (self.q0 @ self.null_key / math.sqrt(32))[None, :, None].expand(len(z), -1, -1)
        attention = torch.cat((dot, null_scores), dim=-1).div(self.cfg['fusion']['temperature']).softmax(-1)
        s, null = class_prompt_evidence(attention)
        norm2 = (d['hh_double'][:, None, :] - d['class_mean']).square() / d['class_var'] + d['class_var'].log()
        raw_llr = 0.5 * (norm2[:, 1] - norm2[:, 0])
        bound = self.cfg['class_statistics']['llr_bound']
        llr = bound * (raw_llr / bound).tanh()
        spec = self.spectral(torch.stack((hh.abs(), loglam, nu), dim=-1)).squeeze(-1).softmax(-1)
        d.update(attention=attention, raw_s=s, null=null, raw_llr=raw_llr.to(z.dtype), llr=llr.to(z.dtype),
                 reliability=0.5 + 0.5 * (16 * spec).tanh())
        return d

    @torch.no_grad()
    def calibrate(self, legal_warmup_cam):
        parts = self.evidence_parts(legal_warmup_cam, self.U0)
        hp = self.cfg['fusion']
        for destination, key, floor in ((self.tau_s, 'raw_s', hp['semantic_floor']),
                                        (self.tau_l, 'llr', hp['llr_floor'])):
            destination.copy_(torch.quantile(parts[key].abs().flatten().cpu(), hp['quantile']).clamp_min(floor))

    def forward(self, z, jacobian, intervention='full'):
        assert z.shape == jacobian.shape == (self.cfg['subjects'], 144)
        assert intervention in INTERVENTIONS
        basis = self.basis(fixed=intervention == 'spectral_guidance_off')
        d = self.evidence_parts(z, basis)
        s, statistical = (d['raw_s'] / self.tau_s).tanh(), (d['llr'] / self.tau_l).tanh()
        null, reliability = d['null'], d['reliability']
        if intervention == 'semantic_evidence_off':
            s, null = torch.zeros_like(s), torch.zeros_like(null)
        n0, n1 = null[:, :1].expand_as(s), null[:, 1:].expand_as(s)
        fusion_features = torch.stack((s, statistical, s.abs(), statistical.abs(),
                                       s * statistical, n0, n1, reliability), dim=-1)
        hp = self.cfg['fusion']
        alpha = hp['alpha_min'] + (hp['alpha_max'] - hp['alpha_min']) * self.fusion_net(fusion_features).squeeze(-1).sigmoid()
        difference = reliability * (1 - null.mean(1, keepdim=True)) * (alpha * s + (1 - alpha) * statistical)
        gate_features = torch.cat((F.layer_norm(z, (144,)), difference, reliability, null), dim=-1)
        gate = self.gate_net(gate_features).tanh()
        theta = gate * difference
        projected = jacobian.detach().double() @ basis
        norm_sq = projected.square().sum(-1, keepdim=True)
        stabilized = norm_sq + self.cfg['jacobian_epsilon']**2
        unit = (projected / stabilized.sqrt()).to(z.dtype)
        first_order_score = (theta.double() * projected.square()).sum(-1) / stabilized.squeeze(-1)
        raw = (theta * unit) @ basis.T.to(z.dtype)
        delta = smooth_limit(raw, z, self.cfg['residual_limit'])
        if self.fold == 2:
            delta = delta.index_fill(0, torch.as_tensor(FOLD2_ROWS, device=z.device), 0)
        if intervention == 'residual_off':
            delta = torch.zeros_like(delta)
        d.update(basis=basis, z=z, jacobian=jacobian, jacobian_projected=projected,
                 signed_semantic=s, signed_statistical=statistical, active_null=null, alpha=alpha,
                 difference=difference, gate=gate, theta=theta, first_order_score=first_order_score,
                 raw=raw, delta=delta, enhanced=z + delta)
        return d

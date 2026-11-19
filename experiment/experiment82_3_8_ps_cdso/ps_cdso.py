"""Full-covariance Gaussian evidence and gain on a trainable CAM144 subspace."""
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
RISK_KEYS = ('margin_memory', 'risk_updates')
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


def head_sensitivity(head, z):
    with torch.enable_grad():
        features = z.detach().requires_grad_(True)
        logits = deterministic_logits(head, features)
        gap = logits[:, 0] - logits[:, 1]
        jacobian, = torch.autograd.grad(gap.sum(), features, create_graph=False)
    return jacobian.detach(), gap.detach()


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


def gaussian_separation(distribution):
    """Bhattacharyya separation of the same two Gaussians used by the residual."""
    covariance = distribution['class_cov']
    factor = torch.linalg.cholesky(covariance.mean(0))
    diff = distribution['class_mean'][0] - distribution['class_mean'][1]
    solved = torch.linalg.solve_triangular(factor, diff[:, None], upper=False)
    mean_part = solved.square().sum() / 8
    logdet_middle = 2 * factor.diagonal().log().sum()
    cov_part = ((logdet_middle - distribution['class_logdet'].mean()) * 0.5).clamp_min(0)
    return mean_part + cov_part, mean_part, cov_part


def gain_terms(on, off, targets, config):
    hp = config['loss']
    margin_off = true_margin(off, targets).detach()
    margin_on = true_margin(on, targets)
    improvement = margin_on - margin_off
    weight = hp['g2_weight_floor'] + (1 - hp['g2_weight_floor']) * (
        hp['g2_margin_center'] - margin_off).sigmoid()
    temp, target = hp['g2_gain_temperature'], hp['g2_gain_target']
    gain = weight * temp * (F.softplus((target - margin_on) / temp) -
                            F.softplus((target - margin_off) / temp))
    easy = (margin_off - hp['protection_margin_center']).sigmoid()
    return (balanced_mean(gain, targets), balanced_mean(easy * improvement.neg().relu().square(), targets),
            improvement, weight)


class PSCDSO(nn.Module):
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
        self.register_buffer('margin_memory', torch.zeros(int(moments['ema_counts'].sum())))
        self.register_buffer('risk_updates', torch.zeros((), dtype=torch.long))
        self.offset = nn.Parameter(torch.zeros_like(self.U0))
        self.direction = nn.Linear(144, 32)
        self.evidence = nn.Linear(5, 32)
        self.norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, 32, bias=False)
        self.null_key = nn.Parameter(torch.empty(32))
        nn.init.normal_(self.null_key, std=0.02)
        self.spectral = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def initialize_fusion(self):
        self.fusion_net = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1))
        nn.init.zeros_(self.fusion_net[-1].weight)
        nn.init.zeros_(self.fusion_net[-1].bias)

    def initialize_gate(self):
        self.gate_net = nn.ModuleDict({
            'direction': nn.Sequential(nn.Linear(180, 16), nn.GELU(), nn.Linear(16, 16)),
            'credibility': nn.Sequential(nn.Linear(9, 8), nn.GELU(), nn.Linear(8, 1)),
        })
        nn.init.zeros_(self.gate_net['direction'][-1].weight)
        nn.init.zeros_(self.gate_net['direction'][-1].bias)

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

    def risk_state(self):
        return {name: getattr(self, name).detach().cpu().clone() for name in RISK_KEYS}

    @torch.no_grad()
    def update_risk(self, margins):
        margins = margins.to(self.margin_memory)
        assert margins.shape == self.margin_memory.shape and torch.isfinite(margins).all()
        if int(self.risk_updates) == 0:
            self.margin_memory.copy_(margins)
        else:
            hp = self.cfg['long_protection']
            alpha = torch.where(margins >= self.margin_memory, hp['rise_alpha'], hp['fall_alpha'])
            self.margin_memory.copy_(self.margin_memory + alpha * (margins - self.margin_memory))
        self.risk_updates.add_(1)

    def protection_loss(self, margin, targets):
        hp = self.cfg['long_protection']
        reference = self.margin_memory.detach()
        reliability = ((reference - hp['reliable_margin']) / hp['reliability_temperature']).sigmoid()
        deficit = (reference - hp['floor_slack'] - margin).relu()
        return balanced_mean(reliability * deficit.square(), targets), reliability, deficit

    def projection(self, z, basis):
        center, within, total = covariance_moments(self.ema_mean, self.ema_second, self.ema_counts)
        variation = (basis * (total @ basis)).sum(0).clamp_min(EPS)
        projected = (z.double() - center) @ basis
        loc = (self.ema_mean - center) @ basis
        directional_cov = (basis.T @ within @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(0)
        w = self.ema_counts.double()[:, None] / self.ema_counts.sum()
        within_pooled = (w * directional_cov).sum(0)
        hp = self.cfg['class_statistics']
        floor = (hp['relative_floor'] * total.trace() / total.size(0)).clamp_min(hp['variance_floor'])
        covariance = basis.T @ (hp['own_variance'] * within + hp['pooled_variance'] * total) @ basis
        covariance = (covariance + covariance.mT) * 0.5 + floor * self.eye16
        lower = torch.linalg.cholesky(covariance)
        logdet = 2 * lower.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        centered = (projected[None] - loc[:, None]).transpose(1, 2)
        whitened = torch.linalg.solve_triangular(lower, centered, upper=False)
        precision_times_x = torch.linalg.solve_triangular(lower.mT, whitened, upper=True)
        quadratic = centered * precision_times_x
        raw_llr = (0.5 * (quadratic[1] - quadratic[0]) +
                   0.5 * (logdet[1] - logdet[0]) / basis.shape[1]).T
        llr_total = 0.5 * (whitened[1].square().sum(0) - whitened[0].square().sum(0) + logdet[1] - logdet[0])
        inverse_lower = torch.linalg.solve_triangular(lower, self.eye16.expand_as(lower), upper=False)
        precision_diag = inverse_lower.square().sum(dim=-2)
        stability = (covariance.diagonal(dim1=-2, dim2=-1) * precision_diag).reciprocal().clamp(0, 1).mean(0)
        loglam = normalize_directions(variation.log()).to(z.dtype)
        contrast = (directional_cov[0] - directional_cov[1]).abs() / (within_pooled + EPS)
        nu = normalize_directions(contrast.log1p()).to(z.dtype)
        return dict(hh=(projected / (variation + EPS).sqrt()).to(z.dtype),
                    projected=projected, class_mean=loc, class_cov=covariance, class_logdet=logdet,
                    loglam=loglam, nu=nu, variation=variation, stability=stability.to(z.dtype),
                    raw_llr=raw_llr, llr_total=llr_total)

    def evidence_parts(self, z, basis):
        d = self.projection(z, basis)
        hh, loglam, nu, stability = (d[key].expand_as(d['hh']) for key in ('hh', 'loglam', 'nu', 'stability'))
        bound = self.cfg['class_statistics']['llr_bound']
        llr = (d['raw_llr'] / bound).tanh().to(z.dtype)
        basis_tokens = self.direction(basis.T.to(z.dtype))[None]
        tokens = self.norm(basis_tokens + self.evidence(torch.stack((hh, hh.abs(), loglam, nu, llr), dim=-1)))
        dot = torch.einsum('kd,njd->nkj', self.q0, self.key(tokens)) / math.sqrt(32)
        null_scores = (self.q0 @ self.null_key / math.sqrt(32))[None, :, None].expand(len(z), -1, -1)
        attention = torch.cat((dot, null_scores), dim=-1).div(self.cfg['fusion']['temperature']).softmax(-1)
        s, null = class_prompt_evidence(attention)
        spec = self.spectral(torch.stack((hh.abs(), loglam, nu, stability), dim=-1)).squeeze(-1).softmax(-1)
        reliability = (0.5 + 0.5 * (16 * spec).tanh()) * (0.5 + 0.5 * stability.sqrt())
        d.update(attention=attention, raw_s=s, null=null, llr=bound * llr, reliability=reliability)
        return d

    @torch.no_grad()
    def calibrate(self, legal_warmup_cam):
        parts = self.evidence_parts(legal_warmup_cam, self.U0)
        hp = self.cfg['fusion']
        for destination, key, floor in ((self.tau_s, 'raw_s', hp['semantic_floor']),
                                        (self.tau_l, 'llr', hp['llr_floor'])):
            destination.copy_(torch.quantile(parts[key].abs().flatten().cpu(), hp['quantile']).clamp_min(floor))

    def forward(self, z, jacobian, gap, intervention='full'):
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
        head_gap = (gap.detach() / 2).tanh()[:, None]
        total = (d['llr_total'] / (4 * self.tau_l)).tanh().to(z.dtype)[:, None]
        head_direction = head_gap.expand_as(s)
        credibility_input = torch.stack((head_direction, head_direction.abs(), s, statistical,
                                         s * statistical, n0, n1, d['stability'].expand_as(s),
                                         total.expand_as(s)), dim=-1)
        credibility = self.gate_net['credibility'](credibility_input).squeeze(-1).sigmoid()
        gate_features = torch.cat((F.layer_norm(z, (144,)), difference, reliability, null, head_gap, total), dim=-1)
        gate = credibility * self.gate_net['direction'](gate_features).tanh()
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
                 raw=raw, delta=delta, enhanced=z + delta, credibility=credibility)
        return d

"""Patient-level hypothesis correction in a single trainable CAM144 subspace."""
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
    return class_response, null


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


class PSCHC(nn.Module):
    def __init__(self, u0, query, moments, fold, group, config):
        super().__init__()
        self.cfg = config
        self.fold = fold
        self.group = group
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
        self.credibility = nn.Sequential(nn.Linear(9, 8), nn.GELU(), nn.Linear(8, 1))
        if self.group == 'G1':
            self.gate_net = nn.Sequential(nn.Linear(180, 16), nn.GELU(), nn.Linear(16, 16))
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)

    def initialize_candidates(self):
        self.candidate_net = nn.ModuleList(
            nn.Sequential(nn.Linear(9, 12), nn.GELU(), nn.Linear(12, 1)) for _ in range(2))
        self.calibrator = nn.Sequential(nn.Linear(12, 8), nn.GELU(), nn.Linear(8, 2))
        self.router = nn.Sequential(nn.Linear(14, 12), nn.GELU(), nn.Linear(12, 3))
        for network in (*self.candidate_net, self.calibrator, self.router):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def parameter_groups(self):
        specific = ('gate_net',) if self.group == 'G1' else ('candidate_net', 'router', 'calibrator')
        groups = {key: [] for key in ('offset', 'prompt', 'spectral', 'fusion_net', 'credibility', *specific)}
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
        projected_total = basis.T @ total @ basis
        projected_origin = self.U0.T @ total @ self.U0
        variation = projected_total.diagonal().clamp_min(EPS)
        projected = (z.double() - center) @ basis
        loc = (self.ema_mean - center) @ basis
        directional_cov = (basis.T @ within @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(0)
        w = self.ema_counts.double()[:, None] / self.ema_counts.sum()
        within_pooled = (w * directional_cov).sum(0)
        hp = self.cfg['class_statistics']
        floor = (hp['relative_floor'] * total.trace() / total.size(0)).clamp_min(hp['variance_floor'])
        covariance = basis.T @ (hp['own_variance'] * within + hp['pooled_variance'] * total) @ basis
        covariance = (covariance + covariance.mT) * 0.5
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        ridge = floor + trace.clamp_min(0) / (hp['max_condition'] - 1)
        covariance = covariance + ridge[:, None, None] * self.eye16
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
                    projected=projected, projected_total=projected_total, projected_origin=projected_origin,
                    class_mean=loc, class_cov=covariance, class_logdet=logdet,
                    dispersion=covariance.diagonal(dim1=-2, dim2=-1).mean(-1) * precision_diag.mean(-1),
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
        responses, null = class_prompt_evidence(attention)
        s = responses[:, 0] - responses[:, 1]
        spec = self.spectral(torch.stack((hh.abs(), loglam, nu, stability), dim=-1)).squeeze(-1).softmax(-1)
        reliability = (0.5 + 0.5 * (16 * spec).tanh()) * (0.5 + 0.5 * stability.sqrt())
        d.update(attention=attention, raw_s=s, responses=responses, null=null, llr=bound * llr, reliability=reliability)
        return d

    @torch.no_grad()
    def calibrate(self, legal_warmup_cam):
        parts = self.evidence_parts(legal_warmup_cam, self.U0)
        hp = self.cfg['fusion']
        for destination, key, floor in ((self.tau_s, 'raw_s', hp['semantic_floor']),
                                        (self.tau_l, 'llr', hp['llr_floor'])):
            destination.copy_(torch.quantile(parts[key].abs().flatten().cpu(), hp['quantile']).clamp_min(floor))

    def protect_rows(self, values):
        if self.fold == 2:
            rows = torch.as_tensor(FOLD2_ROWS, device=values.device)
            return values.index_fill(0, rows, 0)
        return values

    def correct_candidates(self, z, head, d, basis, unit, head_gap, total, s, statistical, null,
                           reliability, credibility, difference, responses, w_sem, w_stat, agreement):
        n = len(z)
        sign = z.new_tensor([1., -1.]).view(1, 2, 1)
        category_shape = (n, 2, 16)

        def both(value):
            return value[:, None].expand(category_shape)

        candidate_features = torch.stack(
            (both(s)*sign, both(statistical)*sign, both(difference)*sign,
             both(head_gap)*sign, (16*responses).tanh(), both(reliability),
             d['stability'].expand(category_shape), null[:, :, None].expand(category_shape), both(total)*sign),
            dim=-1)
        # clamp_min has a unit subgradient at zero. Both arms start at zero
        # while their first-layer outputs can receive the classification signal.
        amplitude = torch.stack([network(candidate_features[:, c]).squeeze(-1).tanh().clamp_min(0)
                                 for c, network in enumerate(self.candidate_net)], dim=1)
        support = both(agreement) * (both(w_sem)*(sign*both(s)).relu() +
                                     both(w_stat)*(sign*both(statistical)).relu())
        directional = sign * self.cfg['counterfactual']['candidate_scale'] * amplitude * (
            support * both(credibility * reliability))
        raw_candidates = (directional * unit[:, None, :]) @ basis.T.to(z.dtype)
        features = z[:, None].expand_as(raw_candidates)
        candidates = self.protect_rows(smooth_limit(raw_candidates, features, self.cfg['residual_limit']))
        logits = deterministic_logits(head, (features+candidates).reshape(-1, 144)).reshape(n, 2, 2)
        gaps = logits[..., 0]-logits[..., 1]
        off_head = deterministic_logits(head, z)
        shift = gaps-(off_head[:, 0]-off_head[:, 1])[:, None]
        decision_features = torch.cat((
            head_gap, total, s.mean(-1, keepdim=True), statistical.mean(-1, keepdim=True),
            (s*statistical).mean(-1, keepdim=True), null, reliability.mean(-1, keepdim=True),
            d['stability'].mean().expand(n, 1), difference.mean(-1, keepdim=True),
            (shift/2).tanh()), dim=-1)
        calibrated = shift * (0.5 + self.calibrator(decision_features).sigmoid())
        route_input = torch.cat((decision_features, (calibrated/2).tanh()), dim=-1)
        scores = self.router(route_input) / self.cfg['counterfactual']['selection_temperature']
        route = scores.softmax(-1)
        choice = scores.detach().argmax(-1)
        rows = torch.arange(n, device=z.device)

        def with_keep(value):
            return torch.cat((torch.zeros_like(value[:, :1]), value), dim=1)

        actions = with_keep(candidates)
        expected = (route[:, :, None]*actions).sum(1)
        # Evaluate just the selected intervention. The zero-valued expression
        # in parentheses supplies a soft selection gradient during training.
        delta = actions[rows, choice].detach() + (expected-expected.detach())
        raw = with_keep(raw_candidates)[rows, choice]
        theta = with_keep(directional)[rows, choice]
        expected_norm = (route[:, :, None]*actions.double().norm(dim=-1, keepdim=True)).sum(1).squeeze(1)
        soft_norm = expected.double().norm(dim=-1)
        cancellation = torch.where(expected_norm > EPS, (1-soft_norm/expected_norm.clamp_min(EPS)).clamp(0, 1), 0.)
        d.update(candidates=candidates, candidate_theta=directional, candidate_gap_effect=shift,
                 route=route, route_scores=scores, route_choice=choice,
                 calibration_inputs=decision_features, calibrated_gap=calibrated,
                 soft_cancellation=cancellation, soft_residual=expected)
        return delta, raw, theta, amplitude.mean(1)

    def forward(self, z, jacobian, gap, head, intervention='full'):
        assert z.shape == jacobian.shape == (self.cfg['subjects'], 144)
        assert intervention in INTERVENTIONS
        basis = self.basis(fixed=intervention == 'spectral_guidance_off')
        d = self.evidence_parts(z, basis)
        s = (d['raw_s'] / self.tau_s).tanh()
        statistical = (d['llr'] / self.tau_l).tanh()
        null, reliability, responses = d['null'], d['reliability'], d['responses']
        if intervention == 'semantic_evidence_off':
            s, null, responses = torch.zeros_like(s), torch.zeros_like(null), torch.zeros_like(responses)
        n0, n1 = null[:, :1].expand_as(s), null[:, 1:].expand_as(s)
        hp = self.cfg['fusion']
        availability = 1 - null.mean(1, keepdim=True)
        semantic_strength = availability * s.abs()
        statistical_strength = d['stability'] * statistical.abs()
        agreement = 1 - hp['conflict_attenuation'] * (-s * statistical).relu() * availability
        fusion_features = torch.stack((s, statistical, semantic_strength, statistical_strength,
                                       s * statistical, n0, n1, reliability), dim=-1)
        alpha = hp['alpha_min'] + (hp['alpha_max'] - hp['alpha_min']) * self.fusion_net(fusion_features).squeeze(-1).sigmoid()
        w_sem, w_stat = alpha * semantic_strength, (1-alpha) * statistical_strength
        difference = reliability * agreement * (w_sem*s + w_stat*statistical)
        head_gap = (gap.detach() / 2).tanh()[:, None]
        total = (d['llr_total'] / (4 * self.tau_l)).tanh().to(z.dtype)[:, None]
        head_direction = head_gap.expand_as(s)
        credibility_input = torch.stack((head_direction, head_direction.abs(), s, statistical,
                                         s * statistical, n0, n1, d['stability'].expand_as(s),
                                         total.expand_as(s)), dim=-1)
        credibility = self.credibility(credibility_input).squeeze(-1).sigmoid()
        projected = jacobian.detach().double() @ basis
        stabilized = projected.square().sum(-1, keepdim=True) + self.cfg['jacobian_epsilon']**2
        unit = (projected / stabilized.sqrt()).to(z.dtype)
        if self.group == 'G1':
            gate_features = torch.cat((F.layer_norm(z, (144,)), difference, reliability, null, head_gap, total), dim=-1)
            gate = credibility * self.gate_net(gate_features).tanh()
            theta = gate * difference
            raw = (theta * unit) @ basis.T.to(z.dtype)
            delta = smooth_limit(raw, z, self.cfg['residual_limit'])
        else:
            delta, raw, theta, gate = self.correct_candidates(
                z, head, d, basis, unit, head_gap, total, s, statistical, null,
                reliability, credibility, difference, responses, w_sem, w_stat, agreement)
        delta = self.protect_rows(delta)
        if intervention == 'residual_off':
            delta = torch.zeros_like(delta)
        first_order_score = (theta.double() * projected.square()).sum(-1) / stabilized.squeeze(-1)
        d.update(basis=basis, z=z, jacobian=jacobian, jacobian_projected=projected,
                 signed_semantic=s, signed_statistical=statistical, active_null=null, alpha=alpha,
                 difference=difference, gate=gate, theta=theta, first_order_score=first_order_score,
                 raw=raw, delta=delta, enhanced=z + delta, credibility=credibility,
                 w_sem=w_sem, w_stat=w_stat, evidence_agreement=agreement)
        return d

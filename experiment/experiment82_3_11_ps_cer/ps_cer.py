"""Conflict-aware feature actions on authentic CAM tokens and one Stiefel subspace."""
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


def anonymous_queries(medical, seed):
    """Use no embedding direction or disease information from the medical queries."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    sample = torch.randn(medical.shape, generator=generator, dtype=torch.float64)
    return (F.normalize(sample, dim=-1)*medical.double().norm(dim=-1, keepdim=True)).float()


class PSCER(nn.Module):
    def __init__(self, u0, query, moments, fold, config):
        super().__init__()
        self.cfg = config
        self.fold = fold
        for name, value in moments.items():
            self.register_buffer(name, value.clone())
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('U0', u0.double().clone())
        self.register_buffer('eye16', torch.eye(16, dtype=torch.float64))
        self.register_buffer('q0', query.float().clone())
        self.register_buffer('tau_l', torch.ones(()))
        self.offset = nn.Parameter(torch.zeros_like(self.U0))
        self.local_map = nn.Sequential(nn.Linear(36, 32), nn.GELU(), nn.Linear(32, 32))
        self.context_map = nn.Sequential(nn.Linear(36, 32), nn.GELU(), nn.Linear(32, 32))
        self.null_key = nn.Parameter(torch.empty(2, 32))
        nn.init.normal_(self.null_key, std=0.02)
        self.direction_gain = nn.Parameter(torch.zeros(2, 16))
        self.candidate_net = nn.Sequential(nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 1))
        self.router = nn.Sequential(nn.Linear(13, 16), nn.GELU(), nn.Linear(16, 3))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def parameter_groups(self):
        groups = {key: [] for key in ('offset', 'prompt', 'direction_gain', 'candidate_net', 'router')}
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
                    raw_llr=raw_llr, llr_total=llr_total, mahalanobis=whitened.square().sum(1).T)

    @torch.no_grad()
    def calibrate(self, legal_warmup_cam):
        llr = self.projection(legal_warmup_cam, self.U0)['llr_total']
        hp = self.cfg['routing']
        self.tau_l.copy_(torch.quantile(llr.abs().cpu(), hp['llr_scale_quantile']).clamp_min(hp['llr_scale_floor']))

    @torch.no_grad()
    def project_strength(self):
        self.direction_gain.clamp_(0, 1)

    def prompt(self, tokens, projection, null_key):
        features = F.normalize(projection(F.layer_norm(tokens, (36,))), dim=-1)
        q = F.normalize(self.q0, dim=-1)
        temp = self.cfg['routing']['token_temperature']
        token_scores = temp*(features @ q.T)
        categories = torch.stack((token_scores[..., 4:8], token_scores[..., :4]), dim=2)
        class_scores = categories.logsumexp(-1) - math.log(4)
        null_scores = temp*(features @ F.normalize(null_key, dim=0))
        joint = torch.cat((class_scores, null_scores.unsqueeze(-1)), dim=-1).softmax(-1).mean(1)
        active = joint[:, :2]
        conditional = active / active.sum(-1, keepdim=True).clamp_min(EPS)
        entropy = -(categories.softmax(-1)*categories.log_softmax(-1)).sum(-1).mean((1, 2))
        concentration = (1 - entropy/math.log(4)).clamp(0, 1)
        return conditional, joint[:, 2], concentration

    def forward(self, z, local, context, jacobian, gap, head, intervention='full'):
        assert local.shape == context.shape == (len(z), 4, 36)
        assert intervention in INTERVENTIONS
        basis = self.basis(fixed=intervention == 'spectral_guidance_off')
        d = self.projection(z, basis)
        p_local, null_local, concentration_local = self.prompt(local, self.local_map, self.null_key[0])
        p_context, null_context, concentration_context = self.prompt(context, self.context_map, self.null_key[1])
        s_local, s_context = p_local[:, 0]-p_local[:, 1], p_context[:, 0]-p_context[:, 1]
        semantic = 0.5*(s_local+s_context)
        null = 0.5*(null_local+null_context)
        concentration = 0.5*(concentration_local+concentration_context)
        consistency = 1-0.5*(s_local-s_context).abs()
        sem_rel = (1-null)*consistency*(0.5+0.5*concentration)
        statistical = (d['llr_total'].to(z.dtype)/self.tau_l).tanh()
        mahal = d['mahalanobis'].min(-1).values.to(z.dtype)
        stat_rel = (0.5+0.5*d['stability'].mean())/(1+(mahal/16-1).relu())
        hp = self.cfg['routing']
        sign = z.new_tensor([1., -1.])
        sem_support = semantic[:, None]*sign
        stat_support = statistical[:, None]*sign
        sem_strength = (sem_rel[:, None]*sem_support-hp['semantic_floor']).relu()
        stat_strength = (stat_rel[:, None]*stat_support-hp['statistical_floor']).relu()
        # Detect a genuine directional conflict, including confident base decisions.
        base_conflicts = gap[:, None]*sign < 0
        agreement = (sem_strength*stat_strength + EPS).sqrt()-math.sqrt(EPS)
        if intervention == 'semantic_evidence_off':
            # Remove every semantic input and use the remaining statistical evidence.
            semantic = torch.zeros_like(semantic)
            sem_support = torch.zeros_like(sem_support)
            sem_rel = torch.zeros_like(sem_rel)
            consistency = torch.zeros_like(consistency)
            null = torch.ones_like(null)
            concentration = torch.zeros_like(concentration)
            agreement = stat_strength
        prior = 2*agreement*base_conflicts

        def by_direction(value):
            return value.expand(-1, 2, 16)

        def by_patient(value):
            return value[:, None, None].expand(-1, 2, 16)

        directional = (d['raw_llr'].to(z.dtype)/self.cfg['class_statistics']['llr_bound']).tanh()
        x = torch.stack((
            by_direction(d['hh'][:, None, :]), by_direction(d['hh'].abs()[:, None, :]),
            by_direction(directional[:, None, :])*sign[None, :, None],
            by_direction(sem_support[:, :, None]), by_direction(stat_support[:, :, None]),
            by_direction(d['stability'][None, None, :].expand(len(z), -1, -1)),
            by_patient(stat_rel), by_patient(sem_rel), by_direction(prior[:, :, None])), dim=-1)
        magnitude = self.candidate_net(x).squeeze(-1).sigmoid()
        j_spec = jacobian @ basis.to(z.dtype)
        unit = j_spec / (j_spec.square().sum(-1, keepdim=True)+self.cfg['jacobian_epsilon']**2).sqrt()
        coeffs = sign[None, :, None]*self.direction_gain[None]*magnitude*unit[:, None, :]
        coeffs = coeffs*(0.25+0.75*d['stability'])*(0.25+0.75*stat_rel[:, None, None])
        raw_candidates = coeffs @ basis.T.to(z.dtype)
        candidates = smooth_limit(raw_candidates, z[:, None, :], self.cfg['residual_limit'])
        if self.fold == 2:
            rows = torch.as_tensor(FOLD2_ROWS, device=z.device)
            candidates = candidates.index_fill(0, rows, 0)
            prior = prior.index_fill(0, rows, 0)
        if intervention == 'residual_off':
            candidates = torch.zeros_like(candidates)
            prior = torch.zeros_like(prior)

        # This deterministic view informs routing at training and inference. The
        # train objective separately evaluates the three true same-mask actions.
        det_gaps = []
        for index in range(2):
            logits = deterministic_logits(head, z+candidates[:, index])
            det_gaps.append(logits[:, 0] - logits[:, 1])
        shift = torch.stack(det_gaps, 1)-gap[:, None]
        scale = hp['gap_scale']
        router_x = torch.cat((
            torch.stack((semantic, statistical, sem_rel, stat_rel, consistency, null,
                         concentration, (gap/scale).tanh(), d['stability'].mean().expand_as(gap)), dim=1),
            prior, (shift*sign/scale).tanh()), dim=1)
        alpha = hp['logit_bound']
        scores = alpha*(self.router(router_x)/alpha).tanh()
        weights = scores.exp()*torch.cat((torch.ones_like(prior[:, :1]), prior), dim=1)
        route = weights/weights.sum(1, keepdim=True)
        choice = route.argmax(1)
        arms = torch.cat((torch.zeros_like(candidates[:, :1]), candidates), dim=1)
        delta = arms[torch.arange(len(z), device=z.device), choice]
        d.update(z=z, basis=basis, j_spec=j_spec, jacobian=jacobian, gap=gap,
                 p_local=p_local, p_context=p_context, null=null, semantic=semantic,
                 statistical=statistical, sem_rel=sem_rel, stat_rel=stat_rel,
                 consistency=consistency, concentration=concentration,
                 prior=prior, candidates=candidates, candidate_effect=shift, route=route,
                 route_scores=scores, choice=choice, delta=delta, enhanced=z+delta)
        return d

"""Adaptive class-calibrated Prompt selection at the historical CAM144 boundary."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-8
TEMPERATURE = 0.5
RESIDUAL_LIMIT = 0.08
PROMPTS = [
    'Nromal brain volume', 'Uniform cortical thickness',
    'Symmetrical brain structure', 'Normal ventricular morphology',
    'Altered brain volume', 'Abnormal functional connectivity',
    'White matter integrity issues', 'Different activation patterns',
]
QUERIES = {'G1': 'q_anonymous', 'G2': 'q_medical'}
FOLD2_ROWS = (376, 406)
SPECTRUM_BUFFERS = {
    'U': 'U', 'mu': 'mu', 'lam': 'lambda',
    'loglam': 'loglambda_normalized', 'nu': 'nu_normalized',
}
EMA_BUFFERS = ('ema_mean', 'ema_second', 'ema_counts', 'ema_updates', 'class_mean', 'class_var')


def legal_spectral_features(z_train, labels, spectrum):
    z_train = z_train.detach().cpu().double()
    labels = labels.detach().cpu().long()
    assert z_train.shape == (len(labels), 144) and set(labels.tolist()) == {0, 1}
    hh = ((z_train - spectrum['mu']) @ spectrum['U']) / (spectrum['lambda'] + EPS).sqrt()
    return hh, [hh[labels == label] for label in (0, 1)]


def stabilize_variance(variance, counts, config):
    pooled = (counts[:, None] * variance).sum(0) / counts.sum()
    floor = (config['relative_floor'] * pooled).clamp_min(config['variance_floor'])
    stable = config['own_variance'] * variance + config['pooled_variance'] * pooled + floor
    assert torch.isfinite(stable).all() and (stable > 0).all()
    return pooled, stable


def class_statistics(z_train, labels, spectrum, config):
    """Estimate once from legal training rows in fixed ASD/TD order."""
    hh, by_class = legal_spectral_features(z_train, labels, spectrum)
    counts = torch.tensor([len(rows) for rows in by_class], dtype=torch.long)
    assert counts.min() > config['variance_ddof']
    mean = torch.stack([rows.mean(0) for rows in by_class])
    variance = torch.stack([rows.var(0, correction=config['variance_ddof']) for rows in by_class])
    pooled, stable = stabilize_variance(variance, counts, config)
    return {'mean': mean, 'variance': variance, 'pooled': pooled, 'stable': stable, 'counts': counts}, hh


@torch.no_grad()
def observed_moments(z_train, labels, spectrum):
    _, by_class = legal_spectral_features(z_train, labels, spectrum)
    mean = torch.stack([rows.mean(0) for rows in by_class])
    second = torch.stack([rows.square().mean(0) for rows in by_class])
    counts = torch.tensor([len(rows) for rows in by_class], dtype=torch.long)
    assert torch.isfinite(mean).all() and torch.isfinite(second).all()
    return {'mean': mean, 'second': second, 'counts': counts}


def log_likelihood_ratio(hh, mean, variance, bound):
    # The common log(2*pi) term cancels between ASD and TD.
    asd = (hh - mean[0]).square() / variance[0] + variance[0].log()
    td = (hh - mean[1]).square() / variance[1] + variance[1].log()
    raw = 0.5 * (td - asd)
    return raw, bound * torch.tanh(raw / bound)


def normalize_weights(weights):
    return weights / (weights.sum(-1, keepdim=True) + EPS)


def select_prompt_evidence(attention, llr, *, direction_gate=True):
    # Each product is a symmetric continuous union over the four Prompt rows.
    c_n = 1 - torch.prod(1 - attention[:, :4, :16], dim=1)
    c_a = 1 - torch.prod(1 - attention[:, 4:, :16], dim=1)
    p_n, p_a = normalize_weights(c_n), normalize_weights(c_a)
    g_asd, g_td = F.softplus(llr), F.softplus(-llr)
    s_asd = (p_a * g_asd).sum(-1)
    s_td = (p_n * g_td).sum(-1)
    scores = torch.stack([s_asd, s_td], dim=-1)
    pi = scores.softmax(-1)
    w_n, w_a = p_n, p_a
    if direction_gate:
        w_n = normalize_weights(c_n * g_td)
        w_a = normalize_weights(c_a * g_asd)
    w_sem = pi[:, :1] * w_a + pi[:, 1:] * w_n
    return {'c_n': c_n, 'c_a': c_a, 'p_n': p_n, 'p_a': p_a, 'g_asd': g_asd, 'g_td': g_td,
            'w_n': w_n, 'w_a': w_a, 'scores': scores, 'pi': pi, 'w_sem': w_sem}


def balanced_ce(scores, labels):
    per_patient = F.cross_entropy(scores, labels, reduction='none')
    return 0.5 * (per_patient[labels == 0].mean() + per_patient[labels == 1].mean())


def smooth_limit(raw, z):
    c = .08 * z.norm(dim=-1, keepdim=True).clamp_min(EPS)
    x = raw.norm(dim=-1, keepdim=True) / c
    small = x < 1e-4
    safe_x = torch.where(small, torch.ones_like(x), x)
    # E82.2 analytic continuation of tanh(x)/x at zero.
    small_x = torch.where(small, x, torch.zeros_like(x))
    scale = torch.where(small, 1-small_x.square()/3+2*small_x.pow(4)/15, torch.tanh(x)/safe_x)
    return raw * scale


class ACCSPE(nn.Module):
    def __init__(self, spectrum, text_cache, stats, group, fold, statistics_cfg, ema_alpha):
        super().__init__()
        self.fold = fold
        self.statistics_cfg = statistics_cfg
        self.llr_bound = statistics_cfg['llr_bound']
        self.ema_alpha = ema_alpha
        for name, key in SPECTRUM_BUFFERS.items():
            self.register_buffer(name, spectrum[key].float().clone())
        self.register_buffer('q0', text_cache[QUERIES[group]].float().clone())
        self.register_buffer('P_text', text_cache['P_text'].float().clone())
        self.register_buffer('class_mean', stats['mean'].float().clone())
        self.register_buffer('class_var', stats['stable'].float().clone())
        self.register_buffer('ema_mean', stats['mean'].clone())
        counts = stats['counts'].clone()
        correction = statistics_cfg['variance_ddof']
        biased_variance = stats['variance'] * ((counts[:, None] - correction) / counts[:, None].double())
        self.register_buffer('ema_second', biased_variance + self.ema_mean.square())
        self.register_buffer('ema_counts', counts)
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))
        # Preserve both layer definitions and E82.2's initialization order.
        self.direction = nn.Linear(144, 32)
        self.evidence = nn.Linear(4, 32)
        self.norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, 32, bias=False)
        self.null = nn.Parameter(torch.empty(32))
        nn.init.normal_(self.null, std=.02)
        self.spectral = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    @torch.no_grad()
    def update_calibration(self, observation):
        """Advance once after the caller has finished evaluating and freezing Best."""
        counts = self.ema_counts.cpu()
        assert torch.equal(counts, observation['counts'])
        alpha = self.ema_alpha
        mean = (1 - alpha) * self.ema_mean.cpu() + alpha * observation['mean']
        second = (1 - alpha) * self.ema_second.cpu() + alpha * observation['second']
        correction = self.statistics_cfg['variance_ddof']
        variance = (second - mean.square()).clamp_min(0) * (counts[:, None].double() / (counts[:, None] - correction))
        _, stable = stabilize_variance(variance, counts, self.statistics_cfg)
        assert torch.isfinite(mean).all() and torch.isfinite(second).all()
        self.ema_mean.copy_(mean)
        self.ema_second.copy_(second)
        self.class_mean.copy_(mean.float())
        self.class_var.copy_(stable.float())
        self.ema_updates.add_(1)

    def _project(self, z):
        h = (z - self.mu) @ self.U
        return h, h / torch.sqrt(self.lam + EPS)

    def _prompt(self, hh, loglam, nu, *, direction_gate=True):
        e = torch.stack([hh, hh.abs(), loglam, nu], dim=-1)
        tokens = self.norm(self.direction(self.U.T)[None] + self.evidence(e))
        scores = torch.einsum('kd,njd->nkj', self.q0, self.key(tokens)) / math.sqrt(32)
        null = (self.q0 @ self.null / math.sqrt(32))[None, :, None].expand(len(hh), -1, -1)
        a = torch.cat([scores, null], dim=-1).div(TEMPERATURE).softmax(-1)
        raw_llr, bounded_llr = log_likelihood_ratio(hh, self.class_mean, self.class_var, self.llr_bound)
        return {'tokens': tokens, 'a': a, 'raw_llr': raw_llr, 'bounded_llr': bounded_llr,
                **select_prompt_evidence(a, bounded_llr, direction_gate=direction_gate)}

    def prompt_aux(self, detached_z):
        assert not detached_z.requires_grad
        _, hh = self._project(detached_z)
        return self._prompt(hh, self.loglam.expand_as(hh), self.nu.expand_as(hh), direction_gate=False)['scores']

    def forward(self, z, intervention='full'):
        assert z.shape == (871, 144) and intervention in ('full', 'semantic_off', 'residual_off', 'gate_off')
        h, hh = self._project(z)
        loglam, nu = self.loglam.expand_as(h), self.nu.expand_as(h)
        details = self._prompt(hh, loglam, nu, direction_gate=intervention != 'gate_off')
        ws = details['w_sem']
        entropy = -(ws * (ws + EPS).log()).sum(-1) / math.log(16)
        n = details['a'][:, :, 16].mean(-1)
        wp = self.spectral(torch.stack([hh.abs(), loglam, nu], dim=-1)).squeeze(-1).softmax(-1)
        beta = ((1 - n) * (1 - entropy).clamp(.2, .8)).detach()
        w = beta[:, None] * ws + (1 - beta[:, None]) * wp
        if intervention == 'semantic_off':
            w = wp
        if intervention == 'residual_off':
            w = torch.full_like(h, 1/16)
        raw = ((w - 1/16) * h) @ self.U.T
        delta = smooth_limit(raw, z)
        if self.fold == 2:
            mask = torch.ones((871, 1), device=z.device, dtype=z.dtype)
            mask[list(FOLD2_ROWS)] = 0
            delta = delta * mask
        details.update(z=z, h=h, hhat=hh, w_sp=wp, entropy=entropy, null=n,
                       beta=beta, w=w, delta_raw=raw, delta=delta)
        return z + delta, details

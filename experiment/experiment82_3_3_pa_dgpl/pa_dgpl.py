"""Patient-adaptive spectral residual and the first-order decision-gain loss."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PREVIOUS = Path('/model/spromptgl/experiment/experiment82_3_2_da_srpe')
sys.path.insert(0, str(PREVIOUS))
try:
    from da_srpe import DASRPE, EMA_BUFFERS, EPS, FOLD2_ROWS, PROMPTS, SPECTRUM_BUFFERS, smooth_limit
finally:
    sys.path.pop(0)


def zero_head(inputs, hidden):
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class PADGPL(DASRPE):
    """Keep the initialized DA-SRPE evidence branch and its EMA buffers."""

    def initialize_signed_gate(self, config):
        self.gate_norm_eps = config['layer_norm_eps']
        self.signed_gate = zero_head(config['input'], config['hidden'])

    def initialize_beta(self, config):
        self.beta_eps = config['logit_eps']
        self.beta_change_limit = config['max_logit_change']
        self.beta_net = zero_head(config['input'], config['hidden'])

    def parameter_groups(self):
        groups = dict(prompt=[], spectral=[], beta=[], signed_gate=[])
        for name, parameter in self.named_parameters():
            part = name.split('.')[0]
            group = {'spectral': 'spectral', 'beta_net': 'beta', 'signed_gate': 'signed_gate'}.get(part, 'prompt')
            groups[group].append(parameter)
        return groups

    def forward(self, z, intervention='full'):
        assert z.shape == (871, 144) and intervention in ('full', 'semantic_off', 'gate_off', 'residual_off')
        h, hh = self._project(z)
        loglam, nu = self.loglam.expand_as(h), self.nu.expand_as(h)
        wp = self.spectral(torch.stack([hh.abs(), loglam, nu], dim=-1)).squeeze(-1).softmax(-1)
        details = {}
        if intervention == 'semantic_off':
            w, gap = wp, torch.zeros_like(wp)
            beta = null = torch.zeros_like(wp[:, 0])
        else:
            details = self._prompt(hh, loglam, nu)
            ws = details['w_sem']
            entropy = -(ws * (ws + EPS).log()).sum(-1) / math.log(16)
            null = details['a'][:, :, 16].mean(-1)
            gap = ws - wp
            scores = details['scores']
            beta_input = torch.stack((1 - entropy, null, scores[:, 0] - scores[:, 1], gap.abs().sum(-1)), dim=-1)
            beta_old = (1 - null) * (1 - entropy).clamp(.2, .8)
            beta_adjustment = self.beta_change_limit * self.beta_net(beta_input).squeeze(-1).tanh()
            beta = (torch.logit(beta_old.clamp(self.beta_eps, 1 - self.beta_eps)) + beta_adjustment).sigmoid()
            w = beta[:, None] * ws + (1 - beta[:, None]) * wp
            details.update(entropy=entropy, beta_old=beta_old, beta_input=beta_input, beta_adjustment=beta_adjustment)

        raw = ((w - 1 / 16) * h) @ self.U.T
        base = smooth_limit(raw, z)
        # The historical denominator is floored at EPS. Restore the literal
        # relative bound even at a zero or sub-EPS CAM norm.
        base = base * (z.norm(dim=-1, keepdim=True) / EPS).clamp(max=1)
        gate_input = torch.cat((F.layer_norm(z, (144,), eps=self.gate_norm_eps), gap,
                                beta[:, None], null[:, None]), dim=-1)
        gate = self.signed_gate(gate_input).squeeze(-1).tanh()
        if intervention == 'gate_off':
            gate = torch.zeros_like(gate)
        delta = gate[:, None] * base
        if self.fold == 2:
            delta = delta.index_fill(0, torch.as_tensor(FOLD2_ROWS, device=z.device), 0)
        if intervention == 'residual_off':
            delta = torch.zeros_like(delta)
        enhanced = z + delta
        details.update(z=z, h=h, hhat=hh, w_sp=wp, null=null, beta=beta, w=w, delta_raw=raw,
                       delta_base=base, gate_input=gate_input, gate=gate, delta=delta, z_enhanced=enhanced)
        return enhanced, details


def decision_gain(logp, enhanced, delta, rows, all_labels, config):
    """Differentiate the existing row-wise classifier once, using legal labels."""
    labels = all_labels.index_select(0, rows)
    selected = logp.index_select(0, rows)
    margin = (selected[:, 0] - selected[:, 1]) * (1 - 2 * labels)
    derivative, = torch.autograd.grad(margin.sum(), enhanced, retain_graph=True, create_graph=False)
    v = derivative.index_select(0, rows).detach()
    gain = (v * delta.index_select(0, rows)).sum(-1)
    difficulty = ((config['difficulty_center'] - margin.detach()) / config['difficulty_temperature']).sigmoid()
    weight = difficulty.clamp(config['difficulty_min'], config['difficulty_max'])
    per_subject = weight * F.softplus(-gain / config['softplus_temperature'])
    loss = .5 * (per_subject[labels == 0].mean() + per_subject[labels == 1].mean())
    return {'loss': loss, 'gain': gain, 'margin': margin, 'v': v, 'weight': weight, 'labels': labels}


class RestrictedGain(torch.autograd.Function):
    """Carry the real gain value and deliver its VJP only to new parameters."""

    @staticmethod
    def forward(ctx, value, gradients, *parameters):
        ctx.save_for_backward(*gradients)
        return value.clone()

    @staticmethod
    def backward(ctx, upstream):
        return None, None, *(upstream * gradient for gradient in ctx.saved_tensors)


def auxiliary_coefficient(norms, config):
    ratios = [c / (g + config['grad_epsilon']) for c, g in norms if g > 0]
    return min(config['lambda_max'], config['rho'] * min(ratios)) if ratios else 0.


def gain_gradients(cls, benefit, groups, config):
    parameters = [p for group in groups.values() for p in group]
    primary = torch.autograd.grad(cls, parameters, retain_graph=True, create_graph=False)
    auxiliary = torch.autograd.grad(benefit, parameters, retain_graph=True, create_graph=False)
    diagnostics = []
    offset = 0
    for group in groups.values():
        stop = offset + len(group)
        u, v = (torch.cat([g.detach().flatten() for g in grads[offset:stop]]).double()
                for grads in (primary, auxiliary))
        diagnostics.append(torch.stack((u.norm(), v.norm(), u.dot(v))))
        offset = stop
    measurements = torch.stack(diagnostics).cpu().tolist()
    if not all(math.isfinite(x) for row in measurements for x in row):
        raise FloatingPointError('nonfinite classification or gain gradient')
    coefficient = auxiliary_coefficient(((c, g) for c, g, _ in measurements), config)
    audit = {
        name: {'cls_norm': c, 'gain_norm': g, 'weighted_ratio': coefficient * g / c if c else (0. if g else None),
               'cosine': min(1., max(-1., dot / (c * g))) if c and g else None}
        for name, (c, g, dot) in zip(groups, measurements)
    }
    restricted = RestrictedGain.apply(benefit.detach(), auxiliary, *parameters)
    return restricted, coefficient, audit

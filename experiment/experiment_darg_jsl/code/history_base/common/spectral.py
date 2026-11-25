"""Small, DSE-free spectral helpers used by the JSL controller."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-8


def margin(logits):
    return logits[..., 0] - logits[..., 1]


def fixed_logits(head, cam):
    norm, linear, gelu, _ = head.mlp_head
    x = F.layer_norm(cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    x = F.linear(x, linear.weight.detach(), linear.bias.detach())
    x = F.gelu(x, approximate=gelu.approximate)
    return F.linear(x, head.classifier.weight.detach(), head.classifier.bias.detach())


def head_jacobian(head, cam):
    with torch.enable_grad():
        leaf = cam.detach().requires_grad_()
        value = margin(fixed_logits(head, leaf))
        jacobian, = torch.autograd.grad(value.sum(), leaf)
    return value.detach() + ((cam - cam.detach()) * jacobian).sum(-1), jacobian.detach()


def split_jacobian(jacobian, basis):
    axes = basis.to(jacobian)
    coefficients = jacobian @ axes
    parallel = coefficients @ axes.T
    return coefficients, parallel, jacobian - parallel


def bounded(raw, cam_norm, fraction=0.08):
    cap = fraction * cam_norm
    ratio = raw.norm(dim=-1) / cap.clamp_min(EPS)
    scale = torch.tanh(ratio) / ratio.clamp_min(EPS)
    scale = torch.where(ratio < 1e-6, torch.ones_like(scale), scale)
    return raw * scale[:, None]


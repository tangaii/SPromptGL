"""Deterministic native head, medical spectral tokens and shared CAM144 geometry."""
from __future__ import annotations

import math

from .history import CFG, require

import torch
from torch import nn
from torch.nn import functional as F

EPS = CFG["spectrum"]["eps"]
R = CFG["residual"]
B = CFG["boundary"]
P = CFG["prompt"]


def native_logits(head, cam):
    """The current original head, with Dropout omitted and weights differentiable."""
    norm, linear, gelu, dropout = head.mlp_head
    require(isinstance(dropout, nn.Dropout), "Original head sequence differs")
    return head.classifier(gelu(linear(norm(cam))))


def fixed_logits(head, cam):
    """Same deterministic function with only CAM differentiation."""
    norm, linear, gelu, _ = head.mlp_head
    x = F.layer_norm(cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    x = F.linear(x, linear.weight.detach(), linear.bias.detach())
    x = F.gelu(x, approximate=gelu.approximate)
    return F.linear(x, head.classifier.weight.detach(), head.classifier.bias.detach())


def margin(logits):
    return logits[..., 0] - logits[..., 1]


def head_jacobian(head, cam):
    with torch.enable_grad():
        leaf = cam.detach().requires_grad_()
        value = margin(fixed_logits(head, leaf))
        jacobian, = torch.autograd.grad(value.sum(), leaf)
    # Keep the value and the exact first-order derivative in the residual path.
    return value.detach() + ((cam - cam.detach()) * jacobian).sum(-1), jacobian.detach()


def split_jacobian(jacobian, basis):
    axes = basis.to(jacobian)
    coefficients = jacobian @ axes
    parallel = coefficients @ axes.T
    return coefficients, parallel, jacobian - parallel


def feasibility(cam_norm, jacobian_norm, margin_value):
    capacity = R["fraction"] * cam_norm * jacobian_norm
    feasible = torch.sigmoid((capacity - margin_value.abs()) / (B["b_fraction"] * capacity + B["b_offset"]))
    return capacity, feasible


def l2_bound(raw, cam_norm):
    cap = R["fraction"] * cam_norm
    safe_cap = cap.clamp_min(EPS)
    ratio = raw.norm(dim=-1) / safe_cap
    small = ratio < 1e-4
    local = torch.where(small, ratio, 0.0)
    regular = torch.where(small, 1.0, ratio)
    shrink = torch.where(small, 1 - local.square() / 3 + 2 * local.pow(4) / 15,
                         regular.tanh() / regular)
    return shrink * (cap / safe_cap)


def unit(x, floor=R["jacobian_floor"]):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(floor)


def normalized_weights(x):
    return x / x.sum(-1, keepdim=True).clamp_min(EPS)


def small_controller(inputs, outputs, bias=0.0, *, initial_sd=None):
    cfg = CFG["controllers"]
    net = nn.Sequential(nn.Linear(inputs, cfg["width"]), nn.GELU(), nn.Linear(cfg["width"], outputs))
    nn.init.normal_(net[-1].weight, std=cfg["action_initial_sd"] if initial_sd is None else initial_sd)
    nn.init.constant_(net[-1].bias, bias)
    return net


class MedicalSpectrum(nn.Module):
    """A fixed basis and an E82.2-style direction/evidence token encoder."""

    def __init__(self, original, text, excluded, *, differential=False):
        super().__init__()
        for key, source in (("basis", "U"), ("center", "mu"), ("variance", "lambda"),
                            ("logvar", "loglambda_normalized"), ("disease_covariance", "nu_normalized")):
            self.register_buffer(key, original[source].double().clone())
        for key in ("medical", "anonymous"):
            self.register_buffer("query_" + key, text["q_" + key].float().clone())
        allowed = torch.ones(CFG["subjects"], dtype=torch.bool)
        allowed[excluded] = False
        self.register_buffer("allowed", allowed)
        width = P["width"]
        self.direction = nn.Linear(144, width)
        self.descriptor = nn.Linear(4, width)
        self.norm = nn.LayerNorm(width)
        self.key = nn.Linear(width, width, bias=False)
        self.null = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.null, std=0.02)
        p = (P["temperature"] - P["min_temperature"]) / (P["max_temperature"] - P["min_temperature"])
        self.temperature_logit = nn.Parameter(torch.tensor(math.log(p / (1 - p))))
        if differential:
            eta = P["common_suppression_init"] / P["common_suppression_cap"]
            self.suppression_logit = nn.Parameter(torch.tensor(math.log(eta / (1 - eta))))
        else:
            self.register_parameter("suppression_logit", None)

    def encode(self, cam, intervention):
        require(tuple(cam.shape) == (CFG["subjects"], 144), "Correction input is not CAM144")
        branch = cam
        basis = self.basis.to(cam)
        z = (branch - self.center.to(cam)) @ basis
        white = z / (self.variance.to(cam) + EPS).sqrt()
        logvar = self.logvar.to(cam).expand_as(white)
        disease = self.disease_covariance.to(cam).expand_as(white)
        tokens = self.norm(self.direction(basis.T)[None] +
                           self.descriptor(torch.stack((white, white.abs(), logvar, disease), -1)))
        query = self.query_anonymous if intervention == "anonymous_query" else self.query_medical
        q = F.normalize(query, dim=-1, eps=EPS)
        k = F.normalize(self.key(tokens), dim=-1, eps=EPS)
        null = F.normalize(self.null, dim=-1, eps=EPS)
        temperature = P["min_temperature"] + (P["max_temperature"] - P["min_temperature"]) * self.temperature_logit.sigmoid()

        def attend(words):
            scores = torch.einsum("nkd,qd->nqk", k, words)
            null_score = (words @ null)[None, :, None].expand(len(cam), -1, -1)
            return (torch.cat((scores, null_score), -1) / temperature).softmax(-1)

        n = P["spectral_tokens"]
        original_attention = attend(q)
        original_semantic = original_attention[:, :, :n]
        attention = original_attention
        context_numerator = torch.einsum("nqk,qd->nkd", original_semantic, q)
        dual = {}
        if self.suppression_logit is not None:
            suppression = P["common_suppression_cap"] * self.suppression_logit.sigmoid()
            if intervention == "differential_prompt_off":
                suppression = torch.zeros_like(suppression)
                differential, differential_attention = q, original_attention
            else:
                common = F.normalize(q.mean(0), dim=-1, eps=EPS)
                differential = F.normalize(q - suppression * (q @ common)[:, None] * common, dim=-1, eps=EPS)
                differential_attention = attend(differential)
                attention = 0.5 * (original_attention + differential_attention)
                context_numerator = 0.5 * (context_numerator + torch.einsum(
                    "nqk,qd->nkd", differential_attention[:, :, :n], differential))
            dual = dict(original_attention=original_attention, differential_attention=differential_attention,
                        common_suppression=suppression, differential_query=differential)
        semantic = attention[:, :, :n]
        context = context_numerator / semantic.sum(1).clamp_min(EPS).unsqueeze(-1)
        coverage = 1 - (1 - semantic).prod(1)
        null_mass = 1 - semantic.sum(-1).mean(-1)
        return {
            "branch": branch, "z": z, "white": white, "descriptors": torch.stack((white.abs(), logvar, disease), -1),
            "tokens": tokens, "attention": attention, "semantic_attention": semantic,
            "coverage": coverage, "context": context, "null_mass": null_mass, "temperature": temperature,
            "cam_norm": branch.norm(dim=-1), **dual,
        }

    def correct(self, cam, raw_parallel, raw_perp, details, intervention):
        allowed = self.allowed[:, None]
        raw_parallel = raw_parallel.masked_fill(~allowed, 0)
        raw_perp = raw_perp.masked_fill(~allowed, 0)
        if intervention == "residual_off":
            raw_parallel, raw_perp = torch.zeros_like(raw_parallel), torch.zeros_like(raw_perp)
        raw = raw_parallel + raw_perp
        scale = l2_bound(raw, details["cam_norm"])[:, None]
        delta = raw * scale
        return cam + delta, {
            **details, "raw_parallel": raw_parallel, "raw_perp": raw_perp, "delta": delta,
            "scale": scale, "delta_parallel": raw_parallel * scale, "delta_perp": raw_perp * scale,
            "raw_ratio": raw.norm(dim=-1) / details["cam_norm"].clamp_min(EPS),
            "ratio": delta.norm(dim=-1) / details["cam_norm"].clamp_min(EPS),
        }

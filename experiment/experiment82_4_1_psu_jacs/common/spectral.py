"""Fixed Original O, signed Prompt control, and the CAM144 residual boundary."""
from __future__ import annotations

import math

import torch
from torch import nn

DIM, EPS = 16, 1e-8
QUERY_RANK = 8
NEUTRAL_COVERAGE = 1 - (16 / 17) ** 8


class ScaleBranchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cam, scale):
        ctx.scale = scale
        return cam.view_as(cam)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def smooth_limit(raw, cam, fraction):
    cap = fraction * cam.norm(dim=-1, keepdim=True)
    ratio = raw.norm(dim=-1, keepdim=True) / cap.clamp_min(EPS)
    small = ratio < 1e-4
    near_zero = torch.where(small, ratio, 0.0)
    safe_ratio = torch.where(small, 1.0, ratio)
    scale = torch.where(small, 1 - near_zero.square() / 3 + 2 * near_zero.pow(4) / 15,
                        ratio.tanh() / safe_ratio)
    return raw * scale * (cap / cap.clamp_min(EPS))


class SpectralPrompt(nn.Module):
    def __init__(self, spectrum, text, exclusions, cfg, method, descriptor_width):
        super().__init__()
        self.subjects = cfg["subjects"]
        self.temperature = cfg["attention_temperature"]
        self.norm_limit = cfg["residual"]["norm_limit"]
        self.branch_gradient = cfg["branch_gradient"]
        self.branch_scale = cfg["branch_scale"]
        for name, key in (("basis", "U"), ("center", "mu"), ("variance", "lambda"),
                          ("logvar", "loglambda_normalized"), ("nu", "nu_normalized")):
            self.register_buffer(name, spectrum[key].float().clone())
        medical = text["q_medical"].clone()
        track, _ = torch.linalg.qr(medical.double().T, mode="reduced")
        for query in (medical, text["q_anonymous"]):
            recovered = (query.double() @ track) @ track.T
            if not torch.allclose(recovered, query.double(), atol=1e-6, rtol=0):
                raise ValueError("An archived Query leaves the fixed RN50 support")
        self.register_buffer("query", medical)
        self.register_buffer("query_support", track.float())
        allowed = torch.ones(self.subjects, dtype=torch.bool)
        allowed[exclusions] = False
        self.register_buffer("allowed", allowed)

        self.direction = nn.Linear(144, 32, bias=False)
        self.evidence = nn.Linear(4, 32)
        self.norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, QUERY_RANK, bias=False)
        self.null = nn.Parameter(torch.empty(QUERY_RANK))
        nn.init.normal_(self.null, std=0.02)

        width = method["signed_width"]
        self.signed = nn.Sequential(nn.Linear(32 + QUERY_RANK + descriptor_width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.normal_(self.signed[-1].weight, std=method["initial_activation_sd"])
        nn.init.zeros_(self.signed[-1].bias)
        self.strength = nn.Linear(34, 1)
        nn.init.normal_(self.strength.weight, std=0.025)
        nn.init.constant_(self.strength.bias, method["strength_bias"])

    def gradient_groups(self):
        prompt_modules = (self.direction, self.evidence, self.norm, self.key)
        return {
            "prompt": (self.null, *(p for module in prompt_modules for p in module.parameters())),
            "residual": (*self.signed.parameters(), *self.strength.parameters()),
        }

    def tokens(self, cam, query, semantic_off):
        if tuple(cam.shape) != (self.subjects, 144):
            raise ValueError(f"Historical classifier requires CAM144, got {tuple(cam.shape)}")
        branch_cam = ScaleBranchGradient.apply(cam, self.branch_gradient)
        z = (branch_cam - self.center) @ self.basis
        white = z / (self.variance + EPS).sqrt()
        features = torch.stack((white, white.abs(), self.logvar.expand_as(z), self.nu.expand_as(z)), -1)
        tokens = self.norm(self.direction(self.basis.T) + self.evidence(features))
        q8 = (self.query if query is None else query) @ self.query_support
        keys = self.key(tokens)
        scores = (keys @ q8.T).transpose(-2, -1) / math.sqrt(32)
        null = (q8 @ self.null / math.sqrt(32))[None, :, None].expand(len(cam), -1, -1)
        attention = (torch.cat((scores, null), -1) / self.temperature).softmax(-1)
        semantic = attention[:, :, :DIM]
        context = semantic.transpose(-2, -1) @ q8
        context = context / semantic.sum(1).clamp_min(EPS).unsqueeze(-1)
        coverage = 1 - (1 - semantic).prod(dim=1)
        if semantic_off:
            coverage = torch.full_like(coverage, NEUTRAL_COVERAGE)
            context = torch.zeros_like(context)
        importance = coverage / NEUTRAL_COVERAGE
        eta_input = torch.cat((tokens.mean(1), white.abs().mean(-1, keepdim=True),
                              coverage.mean(-1, keepdim=True)), -1)
        eta = self.strength(eta_input).sigmoid()
        return {"branch_cam": branch_cam, "z": z, "white": white, "tokens": tokens,
                "attention": attention, "context": context, "coverage": coverage,
                "importance": importance, "eta": eta}

    def signed_logits(self, details, descriptors):
        encoded = torch.cat((details["tokens"], details["context"], torch.stack(descriptors, dim=-1)), dim=-1)
        return self.signed(encoded).squeeze(-1)

    def finish(self, cam, direction, details):
        branch = details["branch_cam"]
        cam_norm = branch.norm(dim=-1, keepdim=True)
        raw_z = self.branch_scale * cam_norm * details["eta"] * direction
        raw = (raw_z @ self.basis.T).masked_fill(~self.allowed[:, None], 0)
        delta = smooth_limit(raw, branch, self.norm_limit)
        return cam + delta, {**details, "raw_z": raw_z, "delta": delta,
                             "relative": delta.norm(dim=-1) / cam_norm.flatten().clamp_min(EPS)}

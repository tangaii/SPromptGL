"""Shared Original O tokens, medical attention and CAM144 residual boundary."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from common.references import GaussianEvidence

DIM = 16
EPS = 1e-8
NEUTRAL_COVERAGE = 1 - (16 / 17) ** 8


def smooth_limit(raw, cam, fraction):
    cap = fraction * cam.norm(dim=-1, keepdim=True)
    radius = raw.norm(dim=-1, keepdim=True) / cap.clamp_min(EPS)
    small = radius < 1e-4
    near_zero = torch.where(small, radius, 0.0)
    safe_ratio = torch.where(small, 1.0, radius)
    scale = torch.where(small, 1 - near_zero.square() / 3 + 2 * near_zero.pow(4) / 15,
                        radius.tanh() / safe_ratio)
    return raw * scale * (cap / cap.clamp_min(EPS))


class SpectralPrompt(nn.Module):
    def __init__(self, spectrum, text, exclusions, cfg):
        super().__init__()
        self.subjects = cfg["subjects"]
        self.temperature = cfg["attention_temperature"]
        self.norm_limit = cfg["residual"]["norm_limit"]
        for name, key in (
            ("basis", "U"), ("center", "mu"), ("variance", "lambda"),
            ("logvar", "loglambda_normalized"), ("nu", "nu_normalized"),
        ):
            self.register_buffer(name, spectrum[key].float().clone())
        self.register_buffer("query", text["q_medical"].float().clone())
        allowed = torch.ones(self.subjects, dtype=torch.bool)
        allowed[exclusions] = False
        self.register_buffer("allowed", allowed)

        self.direction = nn.Linear(144, 32)
        self.evidence = nn.Linear(4, 32)
        self.norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, 32, bias=False)
        self.null = nn.Parameter(torch.empty(32))
        nn.init.normal_(self.null, std=0.02)

    def gradient_groups(self):
        attention_modules = (self.direction, self.evidence, self.norm, self.key)
        prompt = (self.null, *(p for module in attention_modules for p in module.parameters()))
        return {"prompt": prompt, "residual": tuple(p for p in self.parameters() if all(p is not q for q in prompt))}

    def project(self, cam):
        if tuple(cam.shape) != (self.subjects, 144):
            raise ValueError(f"Historical classifier requires CAM144, got {tuple(cam.shape)}")
        return (cam - self.center) @ self.basis

    def attention(self, z_white, directions, logvar, nu, query, semantic_off):
        features = torch.stack((z_white, z_white.abs(), logvar.expand_as(z_white), nu.expand_as(z_white)), -1)
        tokens = self.norm(self.direction(directions) + self.evidence(features))
        keys = self.key(tokens)
        scores = (keys @ query.T).transpose(-2, -1) / math.sqrt(32)
        null = (query @ self.null / math.sqrt(32))[None, :, None].expand(len(z_white), -1, -1)
        attention = (torch.cat((scores, null), -1) / self.temperature).softmax(-1)
        if semantic_off:
            attention = torch.full_like(attention, 1 / 17)
        coverage = 1 - (1 - attention[:, :, :DIM]).prod(dim=1)
        weights = coverage / coverage.sum(-1, keepdim=True).clamp_min(EPS)
        return {"tokens": tokens, "attention": attention, "coverage": coverage, "weights": weights}

    def base_tokens(self, z, query, semantic_off):
        scale = (self.variance + EPS).sqrt()
        details = self.attention(z / scale, self.basis.T, self.logvar, self.nu, query, semantic_off)
        return scale, details

    def finish(self, cam, z, delta_z, details):
        raw = (delta_z @ self.basis.T).masked_fill(~self.allowed[:, None], 0)
        delta = smooth_limit(raw, cam, self.norm_limit)
        cam_norm = cam.norm(dim=-1).clamp_min(EPS)
        return cam + delta, {
            **details, "z": z, "delta_z": delta_z, "raw": raw, "delta": delta,
            "relative": delta.norm(dim=-1) / cam_norm,
        }


class DiseaseSpectralPrompt(SpectralPrompt):
    def __init__(self, spectrum, text, reference, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg)
        self.statistics = GaussianEvidence(reference, self.subjects, cfg["crossfit"])
        self.method = method

    def inputs(self, cam, crossfit, query, semantic_off, statistics_off=False):
        z = self.project(cam)
        scale, details = self.base_tokens(z, self.query if query is None else query, semantic_off)
        llr, asd, directions, evidence, reliability, support = self.statistics.class_evidence(z, crossfit)
        if statistics_off:
            evidence = torch.zeros_like(evidence)
            reliability = torch.zeros_like(reliability)
        # Express both score terms in units of the reference scale.
        mixture = (evidence * scale / self.method["gradient_scale"]).tanh()
        contrast = (((2 * asd - 1)[:, None] * (directions[:, 0] - directions[:, 1]))
                    * scale / self.method["gradient_scale"]).tanh() * reliability[:, None]
        selection = (details["coverage"] / NEUTRAL_COVERAGE) * (
            details["weights"] * DIM).clamp_min(EPS).pow(self.method["selection_power"])
        details.update(llr=llr, p_asd=asd, class_directions=directions, evidence=evidence,
                       reliability=reliability, support=support, mixture=mixture, contrast=contrast,
                       selection=selection)
        return z, scale, details

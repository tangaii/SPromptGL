"""Prompt and reliable cross-fitted statistics control the projected head Jacobian."""
from __future__ import annotations

import math

import torch

from common.references import ClassMoments, head_jacobian
from common.spectral import DIM, SpectralPrompt


class JacsDaSrpe(SpectralPrompt):
    def __init__(self, spectrum, text, reference, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg, method, descriptor_width=6)
        self.method = method
        self.statistics = ClassMoments(reference, self.subjects, self.variance, method["stats"])

    def forward(self, cam, *, head, training, query=None, semantic_off=False, variant=None):
        d = self.tokens(cam, query, semantic_off)
        logits, jac = head_jacobian(head, cam)
        projected = jac @ self.basis
        bounded = projected / (projected.norm(dim=-1, keepdim=True) + self.method["jacobian_floor"])
        carrier = bounded
        descriptor = bounded
        if variant == "jacobian_neutral":
            descriptor = torch.zeros_like(bounded)
            carrier = bounded.norm(dim=-1, keepdim=True).expand_as(bounded) / math.sqrt(DIM)
        stats = self.statistics.evidence(d["z"], training, reliability_off=variant == "reliability_off")
        white = d["white"].tanh()
        features = (white, d["importance"] - 1, descriptor, stats["class_evidence"],
                    stats["reliability"], stats["effect"].tanh() * stats["reliability"])
        signed = self.signed_logits(d, features) + self.method["spectral_skip_scale"] * white
        signed = signed + self.method["evidence_skip_scale"] * stats["class_evidence"]
        action = (d["importance"] * signed).tanh()
        return self.finish(cam, action * carrier, {
            **d, **stats, "action": action, "jacobian": jac, "projected_jacobian": projected,
            "bounded_jacobian": bounded, "frozen_logits": logits, "carrier": carrier,
        })

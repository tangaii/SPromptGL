"""Prompt-controlled signed residuals along spectral and complementary Jacobians."""
from __future__ import annotations

import torch
from torch import nn

from common.geometry import decompose_jacobian, normalize_component
from common.references import ClassMoments, head_jacobian
from common.spectral import PromptCorrection


class PdjcDaSrpe(PromptCorrection):
    def __init__(self, spectrum, text, reference, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg)
        self.method = method
        self.statistics = ClassMoments(reference, self.subjects, self.variance, method["stats"])
        self.controller = nn.Sequential(nn.Linear(74, method["signed_width"]), nn.GELU(),
                                        nn.Linear(method["signed_width"], 4))
        nn.init.normal_(self.controller[-1].weight, std=method["initial_action_weight_sd"])
        nn.init.zeros_(self.controller[-1].bias)

    def gradient_groups(self):
        prompt = [self.temperature_logit, self.null, *self.u_projection.parameters(), self.u_position,
                  self.key.weight]
        residual = list(self.controller.parameters())
        return {"prompt": prompt, "residual": residual}

    def forward(self, cam, *, head, training, query=None, semantic_off=False, variant=None):
        details = self.prepare(cam, query=query, semantic_off=semantic_off)
        logits, jacobian = head_jacobian(head, cam)
        components = decompose_jacobian(jacobian, self.basis)
        carrier_parallel, norm_parallel = normalize_component(components["parallel"], self.method["jacobian_floor"])
        carrier_perpendicular, norm_perpendicular = normalize_component(components["perpendicular"], self.method["jacobian_floor"])
        evidence = self.statistics.evidence(details["coordinates_u"], training,
                                            reliability_off=variant == "reliability_off")

        probability = logits.softmax(-1)
        uncertainty = -(probability * probability.clamp_min(1e-8).log()).sum(-1) / torch.log(
            torch.tensor(2.0, device=cam.device, dtype=cam.dtype)
        )
        absolute_gap = (logits[:, 0] - logits[:, 1]).abs()
        coverage = details["coverage"]
        prompt_context = (details["context"] * coverage.unsqueeze(-1)).sum(1) / coverage.sum(1, keepdim=True).clamp_min(1e-8)
        scalar_features = torch.stack((details["white_u"].mean(-1), details["white_u"].abs().mean(-1),
                                       details["white_u"].square().mean(-1).sqrt(), coverage.mean(-1),
                                       evidence["reliability"].mean(-1), evidence["class_evidence"].mean(-1),
                                       evidence["effect"].abs().mean(-1), evidence["class_support"].mean(-1),
                                       uncertainty, absolute_gap), dim=-1)
        pooled = torch.cat((details["tokens_u"].mean(1), prompt_context, scalar_features), dim=-1)
        output = self.controller(pooled)
        signed = output[:, :2].tanh()
        strength = output[:, 2:].sigmoid()

        scale = self.branch_scale * details["branch_cam"].norm(dim=-1, keepdim=True)
        raw_parallel = scale * strength[:, :1] * signed[:, :1] * carrier_parallel
        raw_perpendicular = scale * strength[:, 1:] * signed[:, 1:] * carrier_perpendicular
        if variant == "spectral_only":
            raw_perpendicular = torch.zeros_like(raw_perpendicular)
        elif variant == "complement_only":
            raw_parallel = torch.zeros_like(raw_parallel)

        entropy = -(details["attention"] * details["attention"].clamp_min(1e-8).log()).sum(-1)
        details.update({
            **evidence,
            **components,
            "jacobian": jacobian,
            "frozen_logits": logits,
            "carrier_parallel": carrier_parallel,
            "carrier_perpendicular": carrier_perpendicular,
            "jacobian_parallel_norm": norm_parallel,
            "jacobian_perpendicular_norm": norm_perpendicular,
            "action_parallel": signed[:, :1],
            "action_perpendicular": signed[:, 1:],
            "strength_parallel": strength[:, :1],
            "strength_perpendicular": strength[:, 1:],
            "uncertainty": uncertainty,
            "attention_entropy": entropy / torch.log(torch.tensor(details["attention"].shape[-1], device=cam.device)),
            "null_weight": details["attention"][..., -1],
            "temperature": details["temperature"],
        })
        return self.finish(cam, raw_parallel, raw_perpendicular, details, residual_off=variant == "residual_off")

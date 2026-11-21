"""EC-PSCE with calibrated query contrast and magnitude-preserving spectral actions."""
from __future__ import annotations

import math

from common.history import CFG
from common.spectral import (EPS, P, R, MedicalSpectrum, feasibility, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn


class ECPSCE(nn.Module):
    def __init__(self, original, text, excluded):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        width = CFG["controllers"]["width"]
        self.spectral_weights = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, 1, bias=False))
        self.action = small_controller(2 * P["width"] + 11, 1)
        self.gate = small_controller(2 * P["width"] + 8, 1, CFG["controllers"]["gate_initial_bias"])

    def forward(self, cam, head, training, intervention="full"):
        d = self.spectrum.encode(cam, intervention)
        w_sem = normalized_weights(d["coverage"])
        w_sp = self.spectral_weights(d["descriptors"]).squeeze(-1).softmax(-1)
        n = P["spectral_tokens"]
        entropy = -(w_sem * w_sem.clamp_min(EPS).log()).sum(-1) / math.log(n)
        beta = (1 - d["null_mass"]) * (0.20 + 0.60 * (1 - entropy))
        semantic_off = intervention == "semantic_off"
        if semantic_off:
            beta = torch.zeros_like(beta)
        w = torch.lerp(w_sp, w_sem, beta[:, None])

        # The archived order is normal prompts followed by abnormal prompts.
        attention = d["semantic_attention"]
        contrast = attention[:, 4:].mean(1) - attention[:, :4].mean(1)
        if intervention == "prompt_contrast_off":
            contrast = torch.zeros_like(contrast)
        calibrated = (contrast / P["contrast_temperature"]).tanh()
        native_margin, jacobian = head_jacobian(head, d["branch"])
        j_axis, parallel, _ = split_jacobian(jacobian, self.spectrum.basis)
        capacity, feasible = feasibility(d["cam_norm"], parallel.norm(dim=-1), native_margin)

        sem = torch.zeros_like(w_sem) if semantic_off else n * w_sem - 1
        scalars = torch.stack((d["white"].tanh(), sem, n * w_sp - 1, n * w - 1, unit(j_axis),
                               native_margin.tanh()[:, None].expand_as(w), beta[:, None].expand_as(w), calibrated), -1)
        query_tokens = d["context"] * d["coverage"].unsqueeze(-1)
        tokens = d["tokens"] * (n * w).unsqueeze(-1)
        inputs = torch.cat((tokens, query_tokens, scalars, d["descriptors"].tanh()), -1)
        action = self.action(inputs).squeeze(-1).tanh()
        if intervention == "signed_action_off":
            action = action.abs()

        spectrum_summary = (d["tokens"] * w.unsqueeze(-1)).sum(1)
        prompt_summary = (query_tokens * w.unsqueeze(-1)).sum(1)
        condition = torch.stack((native_margin.tanh(), capacity.log1p(), feasible, beta, d["null_mass"],
                                 d["white"].abs().mean(-1).log1p(), d["coverage"].mean(-1),
                                 (w * calibrated).sum(-1)), -1)
        gate = self.gate(torch.cat((spectrum_summary, prompt_summary, condition), -1)).squeeze(-1).sigmoid()

        coordinates = R["spectral_action_multiplier"] * w * action
        raw = (R["fraction"] * d["cam_norm"] * gate)[:, None] * (coordinates @ self.spectrum.basis.to(cam).T)
        d.update(w_sem=w_sem, w_sp=w_sp, beta=beta, importance=w, entropy=entropy, action=action,
                 prompt_contrast=contrast, calibrated_contrast=calibrated, direction_coordinates=coordinates,
                 margin=native_margin, jacobian=jacobian, jacobian_axis=j_axis, capacity=capacity,
                 feasibility=feasible, gate=gate)
        return self.spectrum.correct(cam, raw, torch.zeros_like(raw), d, intervention)

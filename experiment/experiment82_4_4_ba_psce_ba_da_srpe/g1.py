"""BA-PSCE: medical allocation and patient-specific signed Original O actions."""
from __future__ import annotations

import math

from common.history import CFG
from common.spectral import (EPS, P, R, MedicalSpectrum, feasibility, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn


class BAPSCE(nn.Module):
    def __init__(self, original, text, excluded):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        self.spectral_weights = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1, bias=False))
        self.action = small_controller(2 * P["width"] + 9, 1)
        self.gate = small_controller(2 * P["width"] + 7, 1, CFG["controllers"]["gate_initial_bias"])

    def forward(self, cam, head, training, intervention="full"):
        d = self.spectrum.encode(cam, intervention)
        w_sem = normalized_weights(d["coverage"])
        w_sp = self.spectral_weights(d["descriptors"]).squeeze(-1).softmax(-1)
        entropy = -(w_sem * w_sem.clamp_min(EPS).log()).sum(-1) / math.log(P["spectral_tokens"])
        beta = (1 - d["null_mass"]) * (0.20 + 0.60 * (1 - entropy))
        if intervention == "semantic_off":
            beta = torch.zeros_like(beta)
        w = torch.lerp(w_sp, w_sem, beta[:, None])
        native_margin, jacobian = head_jacobian(head, d["branch"])
        j_axis, parallel, _ = split_jacobian(jacobian, self.spectrum.basis)
        capacity, feasible = feasibility(d["cam_norm"], parallel.norm(dim=-1), native_margin)

        n = P["spectral_tokens"]
        sem = torch.zeros_like(w_sem) if intervention == "semantic_off" else n * w_sem - 1
        scalar = torch.stack((d["white"].tanh(), sem, n * w_sp - 1, n * w - 1,
                              unit(j_axis), native_margin.tanh()[:, None].expand_as(w)), -1)
        query_tokens = d["context"] * d["coverage"].unsqueeze(-1)
        tokens = d["tokens"] * (n * w).unsqueeze(-1)
        inputs = torch.cat((tokens, query_tokens, scalar, d["descriptors"].tanh()), -1)
        action = self.action(inputs).squeeze(-1).tanh()

        spectrum_summary = (d["tokens"] * w.unsqueeze(-1)).sum(1)
        prompt_summary = (query_tokens * w.unsqueeze(-1)).sum(1)
        scalars = torch.stack((native_margin.tanh(), capacity.log1p(), feasible, beta, d["null_mass"],
                               d["white"].abs().mean(-1).log1p(), d["coverage"].mean(-1)), -1)
        condition = torch.cat((spectrum_summary, prompt_summary, scalars), -1)
        gate = self.gate(condition).squeeze(-1).sigmoid()

        axes = self.spectrum.basis.to(cam)
        coordinate_direction = unit(w * action, R["direction_floor"])
        size = R["fraction"] * d["cam_norm"] * gate
        raw = size[:, None] * (coordinate_direction @ axes.T)
        d.update(w_sem=w_sem, w_sp=w_sp, beta=beta, importance=w, entropy=entropy, action=action,
                 direction_coordinates=coordinate_direction, margin=native_margin, jacobian=jacobian,
                 jacobian_axis=j_axis, capacity=capacity, feasibility=feasible, gate=gate)
        return self.spectrum.correct(cam, raw, torch.zeros_like(raw), d, intervention)

"""PS-BRSC. Prompt allocation and boundary-conditioned signed Original O actions."""
from __future__ import annotations

import math

from common.history import CFG
from common.spectral import (EPS, P, R, MedicalSpectrum, feasibility, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn


class PSBRSC(nn.Module):
    def __init__(self, original, text, excluded):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        self.spectral_weights = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.action = small_controller(P["width"] * 2 + 5, 1)
        self.gate = small_controller(P["width"] + 6, 1, CFG["controllers"]["gate_initial_bias"])

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
        bound, feasible = feasibility(d["cam_norm"], parallel.norm(dim=-1), native_margin)
        p_asd = native_margin.sigmoid()
        weighted_tokens = d["tokens"] * (P["spectral_tokens"] * w).unsqueeze(-1)
        inputs = torch.stack((d["white"].tanh(), 16 * w - 1, unit(j_axis),
                              native_margin.tanh()[:, None].expand_as(w), d["coverage"]), -1)
        action = self.action(torch.cat((weighted_tokens, d["context"] * d["coverage"].unsqueeze(-1), inputs), -1))
        action = action.squeeze(-1).tanh()
        if intervention == "signed_action_unsigned":
            action = action.abs()
        summary = torch.cat(((d["tokens"] * w.unsqueeze(-1)).sum(1),
                             torch.stack((bound.log1p(), native_margin.tanh(), 4 * p_asd * (1 - p_asd),
                                          beta, d["white"].square().mean(-1).log1p(), d["coverage"].mean(-1)), -1)), -1)
        learned_gate = self.gate(summary).squeeze(-1).sigmoid()
        gate = learned_gate if intervention == "correctability_gate_off" else learned_gate * feasible
        dz = unit(w) * action
        raw = (R["branch_scale"] * d["cam_norm"] * gate)[:, None] * (dz @ self.spectrum.basis.to(cam).T)
        d.update(w_sem=w_sem, w_sp=w_sp, beta=beta, importance=w, entropy=entropy, action=action,
                 margin=native_margin, jacobian=jacobian, jacobian_axis=j_axis, capacity=bound,
                 feasibility=feasible, learned_gate=learned_gate, gate=gate)
        return self.spectrum.correct(cam, raw, torch.zeros_like(raw), d, intervention)

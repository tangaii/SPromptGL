"""BA-DA-SRPE: bounded disease evidence and independently signed Jacobian displacements."""
from __future__ import annotations

from common.evidence import ClassEvidence, E
from common.history import CFG
from common.spectral import (R, P, MedicalSpectrum, feasibility, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn


class BADASRPE(nn.Module):
    def __init__(self, original, text, excluded, reference):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        self.statistics = ClassEvidence(reference, original)
        self.evidence_tokens = nn.Linear(6, P["width"])
        width = 2 * P["width"] + 11
        self.actions = small_controller(width, 2)
        self.gate = small_controller(width, 1, CFG["controllers"]["gate_initial_bias"])

    def forward(self, cam, head, training, intervention="full"):
        d = self.spectrum.encode(cam, intervention)
        raw_llr, llr, reliability = self.statistics(d["white"], training)
        attention = d["semantic_attention"]
        normal = 1 - (1 - attention[:, :4]).prod(1)
        abnormal = 1 - (1 - attention[:, 4:]).prod(1)
        preference = (llr / E["prompt_evidence_temperature"]).sigmoid()
        weights = normalized_weights(torch.lerp(normal, abnormal, preference))
        support = (weights * reliability).sum(-1)
        evidence = (weights * reliability * llr).sum(-1)
        evidence_direction = (evidence / E["evidence_temperature"]).tanh()

        native_margin, jacobian = head_jacobian(head, d["branch"])
        j_axis, parallel, perp = split_jacobian(jacobian, self.spectrum.basis)
        parallel_norm = parallel.norm(dim=-1)
        perp_norm = perp.norm(dim=-1)
        capacity, feasible = feasibility(d["cam_norm"], jacobian.norm(dim=-1), native_margin)
        conflict = (-native_margin * evidence_direction / E["conflict_temperature"]).sigmoid()
        evidence_features = torch.stack((d["white"].tanh(), (llr / 2).tanh(), reliability,
                                         normal, abnormal, d["coverage"]), -1)
        tokens = d["tokens"] + self.evidence_tokens(evidence_features).tanh()
        spectrum_summary = (tokens * weights.unsqueeze(-1)).sum(1)
        prompt_summary = (d["context"] * weights.unsqueeze(-1)).sum(1)
        scalars = torch.stack((native_margin.tanh(), capacity.log1p(), feasible, conflict, support,
                               evidence_direction, d["null_mass"], parallel_norm.log1p(), perp_norm.log1p(),
                               d["white"].abs().mean(-1).log1p(), d["white"].tanh().mean(-1)), -1)
        condition = torch.cat((spectrum_summary, prompt_summary, scalars), -1)
        gate = self.gate(condition).squeeze(-1).sigmoid()
        action_logits = self.actions(condition)
        signed_prior = E["signed_prior_scale"] * evidence_direction
        if intervention == "evidence_prior_off":
            signed_prior = torch.zeros_like(signed_prior)
        signs = (action_logits + signed_prior[:, None]).tanh()
        par = signs[:, :1] * unit(parallel)
        rem = signs[:, 1:] * unit(perp)
        length = (par + rem).norm(dim=-1).clamp_min(R["direction_floor"])
        scale = R["fraction"] * d["cam_norm"] * gate / length
        raw_parallel, raw_perp = scale[:, None] * par, scale[:, None] * rem
        if intervention == "spectral_only":
            raw_perp = torch.zeros_like(raw_perp)
        if intervention == "complement_only":
            raw_parallel = torch.zeros_like(raw_parallel)

        d.update(raw_llr=raw_llr, llr=llr, reliability=reliability, importance=weights, support=support,
                 normal=normal, abnormal=abnormal, evidence=evidence, direction=evidence_direction,
                 signed_prior=signed_prior, action_logits=action_logits, action=signs, margin=native_margin,
                 jacobian=jacobian, jacobian_axis=j_axis, parallel=parallel, perp=perp, capacity=capacity,
                 feasibility=feasible, conflict=conflict, gate=gate,
                 strength_parallel=signs[:, 0], strength_perp=signs[:, 1])
        return self.spectrum.correct(cam, raw_parallel, raw_perp, d, intervention)

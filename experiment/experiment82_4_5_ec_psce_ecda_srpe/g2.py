"""ECDA-SRPE with independent amplitude-bearing actions and a spectrum-only prior."""
from __future__ import annotations

from common.evidence import ClassEvidence, E
from common.history import CFG
from common.spectral import (P, R, MedicalSpectrum, feasibility, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn


class ECDASRPE(nn.Module):
    def __init__(self, original, text, excluded, reference):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        self.statistics = ClassEvidence(reference, original)
        self.evidence_tokens = nn.Linear(6, P["width"])
        width = 2 * P["width"] + 11
        self.parallel_action = small_controller(width, 1)
        self.complement_action = small_controller(width, 1, initial_sd=CFG["controllers"]["complement_initial_sd"])
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
        capacity, feasible = feasibility(d["cam_norm"], jacobian.norm(dim=-1), native_margin)
        conflict = (-native_margin * evidence_direction / E["conflict_temperature"]).sigmoid()
        descriptors = torch.stack((d["white"].tanh(), (llr / 2).tanh(), reliability,
                                   normal, abnormal, d["coverage"]), -1)
        tokens = d["tokens"] + self.evidence_tokens(descriptors).tanh()
        spectrum_summary = (tokens * weights.unsqueeze(-1)).sum(1)
        prompt_summary = (d["context"] * weights.unsqueeze(-1)).sum(1)
        scalars = torch.stack((native_margin.tanh(), capacity.log1p(), feasible, conflict, support,
                               evidence_direction, d["null_mass"], parallel.norm(dim=-1).log1p(),
                               perp.norm(dim=-1).log1p(), d["white"].abs().mean(-1).log1p(),
                               d["white"].tanh().mean(-1)), -1)
        condition = torch.cat((spectrum_summary, prompt_summary, scalars), -1)
        gate = self.gate(condition).squeeze(-1).sigmoid()
        parallel_logit = self.parallel_action(condition).squeeze(-1)
        complement_logit = self.complement_action(condition).squeeze(-1)
        prior = E["signed_prior_scale"] * evidence_direction
        if intervention == "evidence_prior_off":
            prior = torch.zeros_like(prior)
        strength_parallel = (parallel_logit + prior).tanh()
        strength_perp = complement_logit.tanh()
        scale = (R["fraction"] * d["cam_norm"] * gate)[:, None]
        raw_parallel = scale * strength_parallel[:, None] * unit(parallel)
        raw_perp = scale * strength_perp[:, None] * unit(perp)
        if intervention == "spectral_only":
            raw_perp = torch.zeros_like(raw_perp)
        if intervention == "complement_only":
            raw_parallel = torch.zeros_like(raw_parallel)

        d.update(raw_llr=raw_llr, llr=llr, reliability=reliability, importance=weights, support=support,
                 normal=normal, abnormal=abnormal, evidence=evidence, direction=evidence_direction, signed_prior=prior,
                 action_logits=torch.stack((parallel_logit, complement_logit), -1),
                 action=torch.stack((strength_parallel, strength_perp), -1), margin=native_margin,
                 jacobian=jacobian, jacobian_axis=j_axis, parallel=parallel, perp=perp, capacity=capacity,
                 feasibility=feasible, conflict=conflict, gate=gate, strength_parallel=strength_parallel,
                 strength_perp=strength_perp)
        return self.spectrum.correct(cam, raw_parallel, raw_perp, d, intervention)

"""PE-DJSC. Signed continuous Gaussian evidence drives two native Jacobian spaces."""
from __future__ import annotations

from common.evidence import ClassEvidence, E
from common.history import CFG
from common.spectral import (R, P, MedicalSpectrum, feasibility, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn


class PEDJSC(nn.Module):
    def __init__(self, original, text, excluded, reference):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        self.statistics = ClassEvidence(reference, original)
        self.gate = small_controller(P["width"] * 2 + 8, 1, CFG["controllers"]["gate_initial_bias"])
        bias = CFG["controllers"]["strength_initial_bias"]
        self.parallel_strength = small_controller(P["width"] * 2 + 8, 1, bias)
        self.perp_strength = small_controller(P["width"] * 2 + 8, 1, bias)

    def forward(self, cam, head, training, intervention="full"):
        d = self.spectrum.encode(cam, intervention)
        raw_llr, llr, reliability = self.statistics(d["white"], training, intervention == "reliability_off")
        att = d["semantic_attention"]
        normal = 1 - (1 - att[:, :4]).prod(1)
        abnormal = 1 - (1 - att[:, 4:]).prod(1)
        preference = (llr / E["prompt_evidence_temperature"]).sigmoid()
        weights = normalized_weights(torch.lerp(normal, abnormal, preference))
        support = (weights * reliability).sum(-1)
        evidence = (weights * reliability * llr).sum(-1)
        direction = (evidence / E["evidence_temperature"]).tanh()
        if intervention == "evidence_direction_off":
            direction = direction.abs()

        native_margin, jacobian = head_jacobian(head, d["branch"])
        j_axis, parallel, perp = split_jacobian(jacobian, self.spectrum.basis)
        bound, feasible = feasibility(d["cam_norm"], jacobian.norm(dim=-1), native_margin)
        # Direction ablation preserves the magnitude and leaves the conflict computation intact.
        conflict = (-native_margin * (evidence / E["evidence_temperature"]).tanh() /
                    E["conflict_temperature"]).sigmoid()
        summary = torch.cat(((d["tokens"] * weights[:, :, None]).sum(1),
                             (d["context"] * (weights * reliability)[:, :, None]).sum(1),
                             torch.stack((native_margin.tanh(), feasible, conflict, support, evidence.tanh(),
                                          d["null_mass"], parallel.norm(dim=-1).log1p(), perp.norm(dim=-1).log1p()), -1)), -1)
        learned_gate = self.gate(summary).squeeze(-1).sigmoid()
        reliability_scale = E["reliability_gate_floor"] + (1 - E["reliability_gate_floor"]) * support
        gate = feasible * conflict * reliability_scale * learned_gate
        strength_parallel = self.parallel_strength(summary).squeeze(-1).sigmoid()
        strength_perp = self.perp_strength(summary).squeeze(-1).sigmoid()
        raw_amplitude = R["branch_scale"] * d["cam_norm"] * gate * direction
        raw_parallel = (raw_amplitude * strength_parallel)[:, None] * unit(parallel)
        raw_perp = (raw_amplitude * strength_perp)[:, None] * unit(perp)
        if intervention == "spectral_only":
            raw_perp = torch.zeros_like(raw_perp)
        if intervention == "complement_only":
            raw_parallel = torch.zeros_like(raw_parallel)
        d.update(raw_llr=raw_llr, llr=llr, reliability=reliability, importance=weights, support=support,
                 evidence=evidence, direction=direction, margin=native_margin, jacobian=jacobian,
                 jacobian_axis=j_axis, parallel=parallel, perp=perp, capacity=bound, feasibility=feasible,
                 conflict=conflict, gate=gate, learned_gate=learned_gate,
                 strength_parallel=strength_parallel, strength_perp=strength_perp)
        return self.spectrum.correct(cam, raw_parallel, raw_perp, d, intervention)

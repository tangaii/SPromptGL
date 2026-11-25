"""Evidence-calibrated common-margin repair in two orthogonal spaces."""
from __future__ import annotations

from common.evidence import AgreementEvidence, E
from common.history import CFG
from common.spectral import P, R, MedicalSpectrum, head_jacobian, normalized_weights, split_jacobian

import torch
from torch import nn

M = CFG["margin_calibration"]


class SCMRDASRPE(nn.Module):
    def __init__(self, original, text, excluded, reference):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded)
        self.statistics = AgreementEvidence(reference, original)
        self.evidence_tokens = nn.Linear(8, P["width"])
        width = M["width"]
        self.calibrator = nn.Sequential(nn.Linear(2 * P["width"] + 12, width), nn.GELU(), nn.Linear(width, 3))
        layer = self.calibrator[-1]
        nn.init.normal_(layer.weight, std=CFG["controllers"]["action_initial_sd"])
        nn.init.zeros_(layer.bias)
        with torch.no_grad():
            layer.weight[0].zero_()
            layer.bias[1] = M["reliability_initial_bias"]

    def forward(self, cam, head, training, intervention="full"):
        d = self.spectrum.encode(cam, intervention)
        raw_llr, llr, reliability = self.statistics(d["white"], training)
        disagreement_axis = self.statistics.disagreement(d["white"], training)
        attention = d["semantic_attention"]
        # Coverage is defined over the prompt axis in the historical SCMR
        # implementation.  Keep its original 4+4 route for an eight-query
        # regression and use fixed 4+4+4 disease groups for MMGPL-12.
        if attention.shape[1] == 8:
            groups = (1 - (1 - attention[:, :4]).prod(1),
                      1 - (1 - attention[:, 4:]).prod(1))
            low = attention.new_zeros(attention.shape[0], dtype=torch.long)
            high = attention.new_ones(attention.shape[0], dtype=torch.long)
        else:
            groups = (1 - (1 - attention[:, :4]).prod(1),
                      1 - (1 - attention[:, 4:8]).prod(1),
                      1 - (1 - attention[:, 8:12]).prod(1))
            # The native head determines a fixed Top-2 pair.  Each member
            # uses its own four MMGPL queries and the lower disease stage is
            # routed to the historical normal slot.
            norm, linear, gelu, _ = head.mlp_head
            logits_for_route = head.classifier(gelu(linear(norm(cam))))
            top = logits_for_route.topk(2, dim=-1).indices
            low = torch.minimum(top[:, 0], top[:, 1])
            high = torch.maximum(top[:, 0], top[:, 1])
        stacked = torch.stack(groups, 1)
        if stacked.ndim == 2:
            normal = stacked.gather(1, low[:, None]).squeeze(1)
            abnormal = stacked.gather(1, high[:, None]).squeeze(1)
        else:
            route_index_low = low[:, None, None].expand(-1, 1, *stacked.shape[2:])
            route_index_high = high[:, None, None].expand(-1, 1, *stacked.shape[2:])
            normal = stacked.gather(1, route_index_low).squeeze(1)
            abnormal = stacked.gather(1, route_index_high).squeeze(1)
        contrast = abnormal - normal
        preference = (llr / E["prompt_evidence_temperature"]).sigmoid()
        weights = normalized_weights(torch.lerp(normal, abnormal, preference))
        support = (weights * reliability).sum(-1)
        evidence = (weights * reliability * llr).sum(-1)
        disagreement = (weights * disagreement_axis).sum(-1)
        evidence_direction = (evidence / E["evidence_temperature"]).tanh()

        native_margin, jacobian = head_jacobian(head, cam)
        j_axis, parallel, perp = split_jacobian(jacobian, self.spectrum.basis)
        jp2, jq2 = parallel.square().sum(-1), perp.square().sum(-1)
        jp, jq = parallel.norm(dim=-1), perp.norm(dim=-1)
        conflict = (-native_margin * evidence_direction / E["conflict_temperature"]).sigmoid()
        if intervention == "evidence_conflict_off":
            conflict = torch.full_like(conflict, .5)
        white = d["white"].tanh()
        descriptors = torch.stack((white, (llr / 2).tanh(), reliability, normal, abnormal,
                                   d["coverage"], contrast, disagreement_axis.log1p()), -1)
        tokens = d["tokens"] + self.evidence_tokens(descriptors).tanh()
        spectrum_summary = (tokens * weights.unsqueeze(-1)).sum(1)
        prompt_summary = (d["context"] * weights.unsqueeze(-1)).sum(1)
        cap = R["fraction"] * d["cam_norm"]
        scalars = torch.stack((native_margin.tanh(), native_margin.abs().log1p(), conflict, support,
                               evidence_direction, d["null_mass"], (cap * jp).log1p(), (cap * jq).log1p(),
                               white.mean(-1), disagreement.log1p(), (weights * contrast).sum(-1),
                               (weights * llr).sum(-1).tanh()), -1)
        logits = self.calibrator(torch.cat((spectrum_summary, prompt_summary, scalars), -1))
        shift = M["shift_cap"] * logits[:, 0].tanh()
        if intervention == "calibrated_evidence_off":
            shift = shift.abs() * evidence_direction
        trust, allocation = logits[:, 1:].sigmoid().unbind(-1)
        target = native_margin + shift
        desired = trust * shift
        strength_parallel = desired * allocation
        strength_perp = desired * (1 - allocation)
        raw_parallel = (strength_parallel / (jp2 + M["jacobian_ridge"]))[:, None] * parallel
        raw_perp = (strength_perp / (jq2 + M["jacobian_ridge"]))[:, None] * perp
        if intervention == "spectral_only":
            raw_perp = torch.zeros_like(raw_perp)
        if intervention == "complement_only":
            raw_parallel = torch.zeros_like(raw_parallel)

        d.update(raw_llr=raw_llr, llr=llr, reliability=reliability, importance=weights, support=support,
                 normal=normal, abnormal=abnormal, prompt_contrast=contrast, evidence=evidence,
                 direction=evidence_direction, disagreement_axis=disagreement_axis, disagreement=disagreement,
                 calibration_logits=logits, shift=shift, target_margin=target, desired_change=desired,
                 correction_reliability=trust, allocation=allocation, action=torch.stack((strength_parallel, strength_perp), -1),
                 margin=native_margin, jacobian=jacobian, jacobian_axis=j_axis, parallel=parallel, perp=perp,
                 capacity=cap * jacobian.norm(dim=-1), conflict=conflict, gate=trust,
                 strength_parallel=strength_parallel, strength_perp=strength_perp)
        return self.spectrum.correct(cam, raw_parallel, raw_perp, d, intervention)

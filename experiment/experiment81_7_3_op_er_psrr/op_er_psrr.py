"""Optimizer-aware PSRR variants and the local SPGS-to-RES compatibility path."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tp_psrr import TPPSRR


class ERPSRR(TPPSRR):
    """Node-conditioned PSRR followed by the original local SPGS/RES equations."""

    def __init__(self, semantic_basis, prompt_features, **kwargs):
        super().__init__(semantic_basis, selection_mode="node", **kwargs)
        self.register_buffer("prompt_features", prompt_features.detach().float().clone())
        self.spgs_mlp = nn.Sequential(nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, 9), nn.ReLU())
        self.reliability_net = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))

    def relation_reliability(self, prompt_scores, visual_real, beta_td, beta_asd,
                             semantic_td, semantic_asd):
        entropy_td = -(beta_td * beta_td.clamp_min(1e-12).log()).sum(dim=-1) / math.log(4.0)
        entropy_asd = -(beta_asd * beta_asd.clamp_min(1e-12).log()).sum(dim=-1) / math.log(4.0)
        uncertainty = 0.5 * (entropy_td + entropy_asd)
        uncertainty = uncertainty.unsqueeze(-1).expand_as(semantic_asd)
        features = torch.stack((
            0.5 * (semantic_td.abs() + semantic_asd.abs()),
            (semantic_asd - semantic_td).abs(),
            visual_real,
            uncertainty,
        ), dim=-1)
        gate = torch.sigmoid(self.reliability_net(features).squeeze(-1))
        return gate, {"gate_features": features, "gate_values": gate}

    def _spgs(self, x):
        n = x.shape[0]
        local = x.reshape(n, 4, 4, 9)
        text = self.prompt_features.to(device=x.device, dtype=x.dtype)
        text_groups = torch.stack((
            text[:4].reshape(-1),
            text[4:].reshape(-1),
        ), dim=0)
        text_features = F.normalize(self.spgs_mlp(text_groups), p=2, dim=-1)
        local_norm = F.normalize(local, p=2, dim=-1)
        similarity = torch.einsum("nqgd,cd->nqgc", local_norm, text_features)
        similarity = similarity.clamp(min=-5.0, max=5.0)
        class_weights = torch.softmax(similarity, dim=-1).amax(dim=-1, keepdim=True)
        return local * class_weights

    @staticmethod
    def _res(x, local, global_features, global_interaction_off=False):
        n = x.shape[0]
        pieces = []
        for group in range(4):
            part = local[:, :, group, :]
            if global_interaction_off:
                weights = torch.full_like(part, 1.0 / part.shape[-1])
            else:
                weights = torch.softmax(part * global_features, dim=-1)
            pieces.append(weights * part)
        return x + torch.cat(pieces, dim=2).reshape(n, 144)

    def forward_cam(self, z_cam, *, grouping="medical", intervention="full",
                    blocked_rows=(), return_diagnostics=False):
        if z_cam.ndim != 2 or z_cam.shape[-1] != 144:
            raise ValueError(f"ER-PSRR expects CAM144, got {tuple(z_cam.shape)}")
        x = z_cam
        n = x.shape[0]
        local = self._spgs(x)
        global_features = F.avg_pool1d(x.reshape(n * 4, 1, 36), kernel_size=4, stride=4).reshape(n, 4, 9)
        projected_delta, details = self.project_relation_delta(
            local.reshape(n, 144), grouping=grouping, intervention=intervention,
            return_diagnostics=return_diagnostics,
        )
        raw_relation_delta = projected_delta.reshape(n, 144)
        local_reference_norm = local.reshape(n, 144).detach().norm(dim=-1, keepdim=True)
        local_bound = self.rho * local_reference_norm
        local_delta = local_bound * raw_relation_delta / torch.sqrt(
            local_bound.square() + raw_relation_delta.square().sum(dim=-1, keepdim=True) + self.eps
        )
        if blocked_rows:
            row_mask = torch.ones((n, 1), dtype=local_delta.dtype, device=local_delta.device)
            row_mask[list(blocked_rows)] = 0.0
            local_delta = local_delta * row_mask
        local_delta = local_delta.reshape(n, 4, 4, 9)
        base_res = self._res(x, local, global_features, intervention == "global_interaction_off")
        full_res = self._res(x, local + local_delta, global_features, intervention == "global_interaction_off")
        raw_delta_cam = full_res - base_res
        reference_norm = x.detach().norm(dim=-1, keepdim=True)
        bound = self.rho * reference_norm
        delta_cam = bound * raw_delta_cam / torch.sqrt(
            bound.square() + raw_delta_cam.square().sum(dim=-1, keepdim=True) + self.eps
        )
        if blocked_rows:
            row_mask = torch.ones((n, 1), dtype=delta_cam.dtype, device=delta_cam.device)
            row_mask[list(blocked_rows)] = 0.0
            delta_cam = delta_cam * row_mask
        details["relation_delta"] = local_delta
        details["relation_raw_delta"] = raw_relation_delta
        details["local_delta"] = local_delta
        details["raw_delta"] = raw_relation_delta
        details["reference_norm"] = reference_norm
        details["delta"] = delta_cam
        details["residual_ratio"] = delta_cam.norm(dim=-1) / (reference_norm[:, 0] + self.eps)
        details["raw_ratio"] = raw_relation_delta.norm(dim=-1) / (reference_norm[:, 0] + self.eps)
        details["spgs_local"] = local
        details["global_features"] = global_features
        details["evidence_margin"] = details["evidence_asd"] - details["evidence_td"]
        details["res_path_residual"] = raw_delta_cam
        return x + delta_cam, details


def evidence_loss(details, labels, legal_rows, margin=0.1):
    rows = torch.as_tensor(legal_rows, dtype=torch.long, device=labels.device)
    if rows.numel() == 0:
        raise ValueError("ER-PSRR evidence loss needs legal outer-training rows")
    y = labels.index_select(0, rows)
    target = torch.where(y == 0, torch.ones_like(y), -torch.ones_like(y)).to(details["evidence_margin"].dtype)
    evidence = details["evidence_margin"].index_select(0, rows)
    return F.softplus(float(margin) - target * evidence).mean()

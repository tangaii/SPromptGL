"""Trajectory-preserved prompt-guided sparse relational reasoning."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Entmax15(torch.autograd.Function):
    """Exact sparse alpha=1.5 entmax on the last dimension."""

    @staticmethod
    def forward(ctx, scores):
        values = scores - scores.amax(dim=-1, keepdim=True)
        values = values / 2.0
        ordered = values.sort(dim=-1, descending=True).values
        count = ordered.shape[-1]
        rho = torch.arange(1, count + 1, dtype=values.dtype, device=values.device)
        rho = rho.view(*([1] * (values.ndim - 1)), count)
        mean = ordered.cumsum(dim=-1) / rho
        mean_sq = ordered.square().cumsum(dim=-1) / rho
        ss = rho * (mean_sq - mean.square())
        delta = (1.0 - ss).clamp_min(0.0) / rho
        tau = mean - delta.sqrt()
        support = tau <= ordered
        support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
        tau_star = tau.gather(-1, support_size - 1)
        probs = (values - tau_star).clamp_min(0.0).square()
        ctx.save_for_backward(probs)
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        (probs,) = ctx.saved_tensors
        root = probs.sqrt()
        denom = root.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(root.dtype).tiny)
        centered = grad_output - (grad_output * root).sum(dim=-1, keepdim=True) / denom
        return root * centered


def entmax15(scores):
    return Entmax15.apply(scores)


def prompt_groups(grouping):
    if grouping == "medical":
        return (0, 1, 2, 3), (4, 5, 6, 7)
    if grouping == "scrambled":
        return (0, 1, 4, 5), (2, 3, 6, 7)
    raise ValueError(f"unknown Prompt grouping: {grouping}")


class TPPSRR(nn.Module):
    """Shared TP-PSRR parameters with node- or edge-conditioned Prompt selection."""

    def __init__(self, semantic_basis, rho=0.1, relation_scale=1.0, temperature=1.0,
                 selection_mode="node", eps=1e-12):
        super().__init__()
        if tuple(semantic_basis.shape) != (8, 8):
            raise ValueError("fixed semantic basis must have shape [8,8]")
        basis = semantic_basis.detach().float().clone()
        center = basis - basis.mean(dim=0, keepdim=True)
        center_norms = center.norm(dim=-1, keepdim=True)
        if not torch.isfinite(center).all() or (center_norms <= 1e-8).any():
            raise ValueError("centered fixed Prompt basis contains a non-finite or zero row")
        center = center / center_norms
        if not torch.isfinite(center).all() or not torch.allclose(
            center.norm(dim=-1), torch.ones(8, dtype=center.dtype, device=center.device), atol=1e-6, rtol=0
        ):
            raise ValueError("centered fixed Prompt basis failed normalization")

        self.register_buffer("semantic_basis", basis)
        self.register_buffer("semantic_center", center)
        self.register_buffer("neighbors", self._make_neighbors(), persistent=True)
        self.local_projection = nn.Linear(9, 32)
        self.position = nn.Parameter(torch.empty(16, 32))
        self.local_norm = nn.LayerNorm(32)
        self.query = nn.Linear(32, 32)
        self.query_norm = nn.LayerNorm(32)
        self.visual_key = nn.Linear(64, 32)
        self.semantic_reduce = nn.Linear(64, 4, bias=False)
        self.semantic_expand = nn.Linear(4, 8, bias=False)
        self.value = nn.Linear(96, 32)
        self.null_key = nn.Parameter(torch.empty(32))
        self.output = nn.Linear(32, 9, bias=False)
        self.rho = float(rho)
        self.relation_scale = float(relation_scale)
        self.temperature = float(temperature)
        if selection_mode not in ("node", "edge"):
            raise ValueError(f"unknown Prompt selection mode: {selection_mode}")
        self.selection_mode = selection_mode
        self.eps = float(eps)
        self.audit_edge_overrides = ()
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.normal_(self.null_key, mean=0.0, std=1.0 / math.sqrt(32.0))
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _make_neighbors():
        rows = [[sender for sender in range(16) if sender != receiver] for receiver in range(16)]
        return torch.tensor(rows, dtype=torch.long)

    def forward(self, z_cam, grouping="medical", intervention="full", blocked_rows=(), return_diagnostics=False):
        if z_cam.ndim != 2 or z_cam.shape[-1] != 144:
            raise ValueError(f"PSRR-LCR expects [N,144], got {tuple(z_cam.shape)}")
        if intervention not in (
            "full", "prompt_off", "relation_off", "uniform_beta", "global_interaction_off",
            "dua_score_off", "second_order_off", "high_frequency_off", "multiscale_off",
        ):
            raise ValueError(f"unknown PSRR-LCR intervention: {intervention}")

        n = z_cam.shape[0]
        groups = prompt_groups(grouping)
        local_9 = z_cam.reshape(n, 4, 4, 9).reshape(n, 16, 9)
        local_tokens = self.local_norm(self.local_projection(local_9) + self.position.unsqueeze(0))
        contrast, diagnostics = self._relations(
            local_tokens, groups, intervention, return_diagnostics, relation_features=local_9
        )
        if intervention == "relation_off":
            contrast = torch.zeros_like(contrast)

        local_delta = self.output(contrast).reshape(n, 4, 4, 9)
        global_features = F.avg_pool1d(z_cam.reshape(n * 4, 1, 36), kernel_size=4, stride=4)
        global_features = global_features.reshape(n, 4, 9)
        if intervention == "global_interaction_off":
            matching = torch.full_like(local_delta, 1.0 / 9.0)
        else:
            matching = torch.softmax(local_delta * global_features.unsqueeze(2), dim=-1)
        raw_delta = (matching * local_delta).reshape(n, 144)

        reference_norm = z_cam.detach().norm(dim=-1, keepdim=True)
        bound = self.rho * reference_norm
        delta = bound * raw_delta / torch.sqrt(bound.square() + raw_delta.square().sum(dim=-1, keepdim=True) + self.eps)
        if blocked_rows:
            row_mask = torch.ones((n, 1), dtype=delta.dtype, device=delta.device)
            row_mask[list(blocked_rows)] = 0.0
            delta = delta * row_mask
        z_final = z_cam + delta

        if return_diagnostics:
            details = {
                **diagnostics,
                "z_final": z_final,
                "local_tokens": local_tokens,
                "relation_contrast": contrast,
                "local_delta": local_delta,
                "raw_delta": raw_delta,
                "delta": delta,
                "global_features": global_features,
                "matching": matching,
                "reference_norm": reference_norm,
                "residual_ratio": delta.norm(dim=-1) / (reference_norm[:, 0] + self.eps),
                "raw_ratio": raw_delta.norm(dim=-1) / (reference_norm[:, 0] + self.eps),
            }
        else:
            details = {
                "raw_delta": raw_delta,
                "delta": delta,
                "reference_norm": reference_norm,
                "residual_ratio": delta.norm(dim=-1) / (reference_norm[:, 0] + self.eps),
                "z_final": z_final,
            }
        return z_final, details

    def relation_distributions(self, z_cam, grouping="medical", intervention="full"):
        """Run only the local relation path, without CAM or OutputLayer calls."""
        n = z_cam.shape[0]
        local_9 = z_cam.reshape(n, 16, 9)
        local_tokens = self.local_norm(self.local_projection(local_9) + self.position.unsqueeze(0))
        _, details = self._relations(
            local_tokens, prompt_groups(grouping), intervention, True, relation_features=local_9
        )
        return details["attention_td"], details["attention_asd"]

    def apply_audit_edge_overrides(self, attention_td, attention_asd):
        """Move selected real-edge mass to Null for deterministic checkpoint audits."""
        if not self.audit_edge_overrides:
            return attention_td, attention_asd
        attention_td = attention_td.clone()
        attention_asd = attention_asd.clone()
        neighbors = self.neighbors.to(device=attention_td.device)
        for patient, receiver, sender, class_index in self.audit_edge_overrides:
            positions = torch.nonzero(neighbors[receiver] == sender, as_tuple=False).flatten()
            if positions.numel() != 1:
                raise ValueError(f"audit edge is not one of the 15 real neighbors: {receiver}->{sender}")
            position = int(positions.item())
            attention = attention_asd if class_index == 0 else attention_td
            mass = attention[patient, receiver, position].clone()
            attention[patient, receiver, position] = 0.0
            attention[patient, receiver, 15] = attention[patient, receiver, 15] + mass
        return attention_td, attention_asd

    def _relations(self, tokens, groups, intervention, return_diagnostics, relation_features=None):
        n = tokens.shape[0]
        neighbor_ids = self.neighbors.to(device=tokens.device)
        senders = tokens[:, neighbor_ids]
        receivers = tokens.unsqueeze(2).expand_as(senders)
        product = receivers * senders
        difference = senders - receivers
        key_input = torch.cat((product, difference.abs()), dim=-1)
        value_input = torch.cat((senders, product, difference), dim=-1)

        query = self.query_norm(self.query(tokens))
        visual_real = (query.unsqueeze(2) * self.visual_key(key_input)).sum(-1) / math.sqrt(32.0)
        visual_null = (query * self.null_key).sum(-1, keepdim=True) / math.sqrt(32.0)
        visual_scores = torch.cat((visual_real, visual_null), dim=-1)

        semantic_key = F.normalize(self.semantic_expand(self.semantic_reduce(key_input)), dim=-1)
        prompt_scores = torch.einsum("kd,nijd->nijk", self.semantic_center, semantic_key)
        if intervention == "prompt_off":
            prompt_scores_for_selection = torch.zeros_like(prompt_scores)
        else:
            prompt_scores_for_selection = prompt_scores

        if self.selection_mode == "node":
            prompt_evidence = (
                torch.logsumexp(visual_real.unsqueeze(-1) + prompt_scores_for_selection, dim=2)
                - torch.logsumexp(visual_real, dim=2).unsqueeze(-1)
            )
            if intervention == "uniform_beta":
                beta_td = torch.full_like(prompt_evidence[..., list(groups[0])], 0.25)
                beta_asd = torch.full_like(prompt_evidence[..., list(groups[1])], 0.25)
            else:
                beta_td = torch.softmax(prompt_evidence[..., list(groups[0])] / self.temperature, dim=-1)
                beta_asd = torch.softmax(prompt_evidence[..., list(groups[1])] / self.temperature, dim=-1)
            semantic_td = (prompt_scores_for_selection[..., list(groups[0])] * beta_td.unsqueeze(2)).sum(-1)
            semantic_asd = (prompt_scores_for_selection[..., list(groups[1])] * beta_asd.unsqueeze(2)).sum(-1)
        else:
            # Edge mode keeps every receiver-candidate relation separate. The
            # Null candidate has no semantic edge score and is handled below.
            prompt_evidence = prompt_scores_for_selection
            if intervention == "uniform_beta":
                beta_td = torch.full_like(prompt_evidence[..., list(groups[0])], 0.25)
                beta_asd = torch.full_like(prompt_evidence[..., list(groups[1])], 0.25)
            else:
                beta_td = torch.softmax(prompt_evidence[..., list(groups[0])] / self.temperature, dim=-1)
                beta_asd = torch.softmax(prompt_evidence[..., list(groups[1])] / self.temperature, dim=-1)
            semantic_td = (prompt_scores_for_selection[..., list(groups[0])] * beta_td).sum(-1)
            semantic_asd = (prompt_scores_for_selection[..., list(groups[1])] * beta_asd).sum(-1)
        disease_difference = semantic_asd - semantic_td
        scores_td = torch.cat((visual_real - 0.5 * self.relation_scale * disease_difference, visual_null), dim=-1)
        scores_asd = torch.cat((visual_real + 0.5 * self.relation_scale * disease_difference, visual_null), dim=-1)

        attention_visual = entmax15(visual_scores)
        if intervention == "relation_off":
            attention_td = attention_visual
            attention_asd = attention_visual
        else:
            attention_td = entmax15(scores_td)
            attention_asd = entmax15(scores_asd)

        values = self.value(value_input)
        contrast = ((attention_asd[..., :15] - attention_td[..., :15]).unsqueeze(-1) * values).sum(dim=2)
        if intervention in ("prompt_off", "relation_off"):
            contrast = torch.zeros_like(contrast)

        details = {"beta_td": beta_td, "beta_asd": beta_asd, "selection_mode": self.selection_mode}
        if return_diagnostics:
            details.update({
                "prompt_evidence": prompt_evidence,
                "prompt_scores": prompt_scores,
                "prompt_scores_raw_basis": torch.einsum("kd,nijd->nijk", self.semantic_basis, semantic_key),
                "prompt_scores_used": prompt_scores_for_selection,
                "visual_scores": visual_scores,
                "scores_td": scores_td,
                "scores_asd": scores_asd,
                "disease_relation_difference": disease_difference,
                "attention_visual": attention_visual,
                "attention_td": attention_td,
                "attention_asd": attention_asd,
                "relation_support_difference": (attention_td[..., :15] > 0) ^ (attention_asd[..., :15] > 0),
                "message_contrast": contrast,
            })
        return contrast, details


def residual_penalty(details, train_indices):
    raw = details["raw_delta"]
    reference = details.get("reference_norm")
    if reference is None:
        raise ValueError("raw residual penalty needs reference CAM norms")
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=raw.device)
    if indices.numel() == 0:
        raise ValueError("no legal training rows for PSRR-LCR residual regularization")
    ratio = raw[indices].norm(dim=-1) / (reference[indices, 0] + 1e-12)
    return ratio.square().mean()

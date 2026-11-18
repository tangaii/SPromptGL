"""Prompt-guided sparse within-patient relational reasoning."""
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


def _prompt_groups(grouping):
    if grouping == "medical":
        return (0, 1, 2, 3), (4, 5, 6, 7)
    if grouping == "scrambled":
        return (0, 1, 4, 5), (2, 3, 6, 7)
    if grouping == "swapped_medical":
        return (4, 5, 6, 7), (0, 1, 2, 3)
    if grouping == "swapped_scrambled":
        return (2, 3, 6, 7), (0, 1, 4, 5)
    raise ValueError(f"unknown Prompt grouping: {grouping}")


class PSRR(nn.Module):
    def __init__(self, semantic_basis, rho=0.1, eps=1e-12):
        super().__init__()
        if tuple(semantic_basis.shape) != (8, 8):
            raise ValueError("fixed semantic basis must have shape [8,8]")
        self.register_buffer("semantic_basis", semantic_basis.detach().float().clone())
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
        self.eps = float(eps)
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.normal_(self.null_key, mean=0.0, std=1.0 / math.sqrt(32.0))
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _make_neighbors():
        rows = []
        for receiver in range(16):
            rows.append([sender for sender in range(16) if sender != receiver])
        return torch.tensor(rows, dtype=torch.long)

    def forward(self, z_cam, grouping="medical", mechanism="pairwise", intervention="full",
                blocked_rows=(), return_diagnostics=False):
        if z_cam.ndim != 2 or z_cam.shape[-1] != 144:
            raise ValueError(f"PSRR expects [N,144], got {tuple(z_cam.shape)}")
        if mechanism not in ("pairwise", "unary"):
            raise ValueError(f"unknown PSRR mechanism: {mechanism}")
        if intervention not in ("full", "prompt_off", "relation_off", "null_only", "global_off"):
            raise ValueError(f"unknown PSRR intervention: {intervention}")

        n = z_cam.shape[0]
        groups = _prompt_groups(grouping)
        local_9 = z_cam.reshape(n, 4, 4, 9).reshape(n, 16, 9)
        u = self.local_norm(self.local_projection(local_9) + self.position.unsqueeze(0))
        if mechanism == "pairwise":
            relation_contrast, diagnostics = self._pairwise(u, groups, intervention, return_diagnostics)
        else:
            relation_contrast, diagnostics = self._unary(u, groups, intervention, return_diagnostics)

        if intervention in ("prompt_off", "relation_off", "null_only"):
            relation_contrast = torch.zeros_like(relation_contrast)

        local_delta = self.output(relation_contrast).reshape(n, 4, 4, 9)
        global_features = F.avg_pool1d(z_cam.reshape(n * 4, 1, 36), kernel_size=4, stride=4)
        global_features = global_features.reshape(n, 4, 9)
        if intervention == "global_off":
            matching = torch.full_like(local_delta, 1.0 / 9.0)
        else:
            matching = torch.softmax(local_delta * global_features.unsqueeze(2), dim=-1)
        raw_delta = (matching * local_delta).reshape(n, 144)

        reference = z_cam.detach()
        reference_norm = reference.norm(dim=-1, keepdim=True)
        bound = self.rho * reference_norm
        delta = bound * raw_delta / torch.sqrt(bound.square() + raw_delta.square().sum(dim=-1, keepdim=True) + self.eps)
        if blocked_rows:
            row_mask = torch.ones((n, 1), dtype=delta.dtype, device=delta.device)
            row_mask[list(blocked_rows)] = 0.0
            delta = delta * row_mask

        if return_diagnostics:
            details = {
                **diagnostics,
                "local_tokens": u,
                "relation_contrast": relation_contrast,
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
            }
        return z_cam + delta, details

    def _pairwise(self, u, groups, intervention, return_diagnostics):
        n = u.shape[0]
        neighbor_ids = self.neighbors.to(device=u.device)
        senders = u[:, neighbor_ids]
        receivers = u.unsqueeze(2).expand_as(senders)
        product = receivers * senders
        difference = senders - receivers
        key_input = torch.cat((product, difference.abs()), dim=-1)
        value_input = torch.cat((senders, product, difference), dim=-1)

        query = self.query_norm(self.query(u))
        visual_real = (query.unsqueeze(2) * self.visual_key(key_input)).sum(-1) / math.sqrt(32.0)
        visual_null = (query * self.null_key).sum(-1, keepdim=True) / math.sqrt(32.0)
        visual_scores = torch.cat((visual_real, visual_null), dim=-1)

        semantic_key = F.normalize(self.semantic_expand(self.semantic_reduce(key_input)), dim=-1)
        prompt_real = torch.einsum("kd,nipd->nkip", self.semantic_basis, semantic_key)
        if intervention in ("prompt_off", "relation_off"):
            prompt_real = torch.zeros_like(prompt_real)

        prompt_evidence = (
            torch.logsumexp(visual_real.unsqueeze(1) + prompt_real, dim=-1)
            - torch.logsumexp(visual_real, dim=-1).unsqueeze(1)
        ).mean(dim=-1)
        beta_normal = torch.softmax(prompt_evidence[:, list(groups[0])], dim=-1)
        beta_asd = torch.softmax(prompt_evidence[:, list(groups[1])], dim=-1)
        semantic_normal = (prompt_real[:, list(groups[0])] * beta_normal[:, :, None, None]).sum(dim=1)
        semantic_asd = (prompt_real[:, list(groups[1])] * beta_asd[:, :, None, None]).sum(dim=1)
        scores_normal = torch.cat((visual_real + semantic_normal, visual_null), dim=-1)
        scores_asd = torch.cat((visual_real + semantic_asd, visual_null), dim=-1)

        if intervention == "null_only":
            visual_adj = torch.zeros_like(visual_scores)
            visual_adj[..., -1] = 1.0
            normal_adj = visual_adj
            asd_adj = visual_adj
        else:
            visual_adj = entmax15(visual_scores)
            if intervention in ("prompt_off", "relation_off"):
                normal_adj = visual_adj
                asd_adj = visual_adj
            else:
                normal_adj = entmax15(scores_normal)
                asd_adj = entmax15(scores_asd)

        values = self.value(value_input)
        message_visual = (visual_adj[..., :15, None] * values).sum(dim=2)
        message_normal = ((normal_adj[..., :15] - visual_adj[..., :15]).unsqueeze(-1) * values).sum(dim=2)
        message_asd = ((asd_adj[..., :15] - visual_adj[..., :15]).unsqueeze(-1) * values).sum(dim=2)
        contrast = message_asd - message_normal
        if intervention in ("prompt_off", "relation_off", "null_only"):
            contrast = torch.zeros_like(contrast)

        details = {"beta_normal": beta_normal, "beta_asd": beta_asd}
        if return_diagnostics:
            details.update({
                "prompt_evidence": prompt_evidence,
                "prompt_scores": prompt_real,
                "visual_scores": visual_scores,
                "scores_normal": scores_normal,
                "scores_asd": scores_asd,
                "attention_visual": visual_adj,
                "attention_normal": normal_adj,
                "attention_asd": asd_adj,
                "message_visual": message_visual,
                "message_normal": message_normal,
                "message_asd": message_asd,
                "relation_support_difference": (normal_adj[..., :15] > 0) ^ (asd_adj[..., :15] > 0),
            })
        return contrast, details

    def _unary(self, u, groups, intervention, return_diagnostics):
        n = u.shape[0]
        zero = torch.zeros_like(u)
        key_input = torch.cat((u, u), dim=-1)
        value_input = torch.cat((u, u.square(), zero), dim=-1)
        query = self.query_norm(self.query(u))
        visual_key = self.visual_key(key_input)
        visual_score = (query * visual_key).sum(-1) / math.sqrt(32.0)
        semantic_key = F.normalize(self.semantic_expand(self.semantic_reduce(key_input)), dim=-1)
        prompt_score = torch.einsum("kd,npd->nkp", self.semantic_basis, semantic_key)
        prompt_score = prompt_score * (1.0 + torch.tanh(visual_score).unsqueeze(1))
        if intervention == "prompt_off":
            prompt_score = torch.zeros_like(prompt_score)

        evidence = prompt_score.mean(dim=-1)
        beta_normal = torch.softmax(evidence[:, list(groups[0])], dim=-1)
        beta_asd = torch.softmax(evidence[:, list(groups[1])], dim=-1)
        response_normal = (prompt_score[:, list(groups[0])] * beta_normal[:, :, None]).sum(dim=1)
        response_asd = (prompt_score[:, list(groups[1])] * beta_asd[:, :, None]).sum(dim=1)
        values = self.value(value_input)
        message_normal = response_normal.unsqueeze(-1) * values
        message_asd = response_asd.unsqueeze(-1) * values
        contrast = message_asd - message_normal
        if intervention in ("prompt_off", "relation_off", "null_only"):
            contrast = torch.zeros_like(contrast)

        details = {
            "beta_normal": beta_normal,
            "beta_asd": beta_asd,
            "prompt_evidence": evidence,
        }
        if return_diagnostics:
            details.update({
                "prompt_scores": prompt_score,
                "visual_scores": visual_score,
                "message_normal": message_normal,
                "message_asd": message_asd,
                "attention_visual": None,
                "attention_normal": None,
                "attention_asd": None,
                "relation_support_difference": None,
            })
        return contrast, details


def residual_penalty(details, train_indices):
    raw = details["raw_delta"]
    reference = details.get("reference_norm")
    if reference is None:
        raise ValueError("raw residual penalty needs reference CAM norms")
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=raw.device)
    if indices.numel() == 0:
        raise ValueError("no legal training rows for PSRR residual regularization")
    ratio = raw[indices].norm(dim=-1) / (reference[indices, 0] + 1e-12)
    return ratio.square().mean()

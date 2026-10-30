#!/usr/bin/env python3
"""Experiment 53 disease relation learning and independent propagation."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


MODS = ["PHENO", "ANAT", "FUNC", "CORR"]
DIMS = [48, 6, 10, 256]


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def grad_reverse(x, scale=0.03):
    return GradientReverse.apply(x, scale)


class RelationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        return self.net(x)


class RelationScorer(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def pair_features(z, i, j):
        zn = F.normalize(z, dim=-1)
        zi, zj = zn[i], zn[j]
        return torch.cat(((zi - zj).abs(), zi * zj, (zi * zj).sum(-1, keepdim=True)), dim=-1)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class ConditionalModalityDiscriminator(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, features, pair_condition):
        cond = F.one_hot(pair_condition.long(), num_classes=2).to(features.dtype)
        return self.net(torch.cat((features, cond), dim=-1))


class RelationCore(nn.Module):
    """Four independent relation encoders/scorers and one shared discriminator."""

    def __init__(self, hidden=32, grl_scale=0.03):
        super().__init__()
        self.encoders = nn.ModuleList([RelationEncoder(d, hidden) for d in DIMS])
        self.scorers = nn.ModuleList([RelationScorer(hidden) for _ in DIMS])
        self.discriminator = ConditionalModalityDiscriminator()
        self.grl_scale = float(grl_scale)

    def encode(self, parts):
        return [encoder(x) for encoder, x in zip(self.encoders, parts)]

    def pair_forward(self, parts, pair_i, pair_j, pair_y, reliable_modalities: Sequence[int], adv_mode: str):
        z = self.encode(parts)
        features = [scorer.pair_features(z[m], pair_i, pair_j) for m, scorer in enumerate(self.scorers)]
        logits = [self.scorers[m](features[m]) for m in range(4)]
        if pair_y is None or pair_y.numel() == 0:
            zero = parts[0].new_zeros(())
            return z, features, logits, zero, zero, None, None
        pair_target = pair_y.to(logits[0].dtype)
        pair_loss = torch.stack([
            F.binary_cross_entropy_with_logits(logit, pair_target) for logit in logits
        ]).mean()
        adv_loss = parts[0].new_zeros(())
        adv_logits = None
        adv_modal = None
        reliable = list(reliable_modalities)
        if adv_mode != "none" and len(reliable) >= 2:
            rel_features = torch.cat([grad_reverse(features[m], self.grl_scale) for m in reliable], dim=0)
            if adv_mode == "conditional":
                condition = pair_y.long()
            else:
                condition = torch.zeros_like(pair_y.long())
            condition = condition.repeat(len(reliable))
            adv_modal = torch.arange(len(reliable), device=rel_features.device).repeat_interleave(pair_i.numel())
            adv_logits = self.discriminator(rel_features, condition)
            adv_loss = F.cross_entropy(adv_logits, adv_modal)
        return z, features, logits, pair_loss, adv_loss, adv_logits, adv_modal

    def optimizer_param_groups(self, lr=0.001, matrix_decay=1e-4):
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and name.endswith("weight") and "norm" not in name.lower():
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        return [
            {"params": decay, "lr": lr, "weight_decay": matrix_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]


def _top_per_source(weights, valid, src, slot_index, max_neighbors):
    """Select up to max_neighbors per receiver using detached scores."""
    n, width = slot_index.shape
    safe = slot_index.clamp_min(0)
    matrix = weights[safe].detach().view(n, width)
    valid_matrix = valid[safe].view(n, width) & (slot_index >= 0)
    matrix = matrix.masked_fill(~valid_matrix, float("-inf"))
    k = min(int(max_neighbors), width)
    if k <= 0:
        return torch.zeros_like(weights, dtype=torch.bool)
    values, positions = torch.topk(matrix, k=k, dim=1)
    selected_slots = torch.gather(slot_index, 1, positions)
    selected = torch.zeros_like(weights, dtype=torch.bool)
    good = torch.isfinite(values) & (selected_slots >= 0)
    selected[selected_slots[good]] = True
    return selected


class DiseaseRelationPropagation(nn.Module):
    """Relation learning plus bounded positive/negative patient propagation.

    The base GFR/CAM remain outside this module. This module only creates
    relation losses and the independent Delta-H propagation path.
    """

    def __init__(
        self,
        relation_core: RelationCore,
        candidate_i,
        candidate_j,
        modality_edge_indices,
        slot_index,
        reverse_dir,
        calibration_a,
        calibration_b,
        omega,
        shuffle_perm,
        uniform_pos,
        uniform_neg,
        gamma_negative=0.25,
        lambda_prop=0.02,
        max_neighbors=12,
    ):
        super().__init__()
        self.relation = relation_core
        self.register_buffer("candidate_i", candidate_i.long())
        self.register_buffer("candidate_j", candidate_j.long())
        self.register_buffer("slot_index", slot_index.long())
        self.register_buffer("reverse_dir", reverse_dir.long())
        self.register_buffer("calibration_a", calibration_a.float())
        self.register_buffer("calibration_b", calibration_b.float())
        self.register_buffer("omega", omega.float())
        self.register_buffer("shuffle_perm", shuffle_perm.long())
        self.register_buffer("uniform_pos", uniform_pos.long())
        self.register_buffer("uniform_neg", uniform_neg.long())
        for m, idx in enumerate(modality_edge_indices):
            self.register_buffer(f"modality_edge_{m}", idx.long())
        self.gamma_negative = float(gamma_negative)
        self.lambda_prop = float(lambda_prop)
        self.max_neighbors = int(max_neighbors)

    def modality_edge_indices(self, m):
        return getattr(self, f"modality_edge_{m}")

    def _candidate_q(self, z):
        e = int(self.candidate_i.numel())
        q_by_modality = []
        for m, scorer in enumerate(self.relation.scorers):
            q = self.candidate_i.new_full((e,), 0.5, dtype=torch.float32)
            idx = self.modality_edge_indices(m)
            if idx.numel():
                features = scorer.pair_features(z[m], self.candidate_i[idx], self.candidate_j[idx])
                logits = scorer(features)
                q[idx] = torch.sigmoid(self.calibration_a[m] * logits + self.calibration_b[m])
            q_by_modality.append(q)
        q_stack = torch.stack(q_by_modality, dim=0)
        present = torch.stack([
            torch.zeros(e, dtype=torch.bool, device=q_stack.device).scatter_(0, self.modality_edge_indices(m), True)
            for m in range(4)
        ], dim=0)
        usable = present & (self.omega[:, None] > 0)
        denom = (usable.to(q_stack.dtype) * self.omega[:, None]).sum(0)
        fused = (q_stack * usable.to(q_stack.dtype) * self.omega[:, None]).sum(0) / denom.clamp_min(1e-8)
        fused = torch.where(denom > 0, fused, torch.full_like(fused, 0.5))
        return fused, q_stack, denom, usable

    def _directed_weights(self, undirected_ids, weights):
        ids = undirected_ids.long()
        src = torch.cat((self.candidate_i[ids], self.candidate_j[ids]), dim=0)
        dst = torch.cat((self.candidate_j[ids], self.candidate_i[ids]), dim=0)
        w = torch.cat((weights, weights), dim=0)
        return src, dst, w

    def _select_sign(self, q, sign, threshold, continuous=False):
        """Return an undirected edge mask; selection is detached, weights remain differentiable."""
        e = q.numel()
        if continuous:
            return torch.ones(e, dtype=torch.bool, device=q.device)
        valid = q >= threshold if sign == "positive" else q <= threshold
        score = (q - 0.5).abs()
        directed_valid = torch.cat((valid, valid), dim=0)
        directed_score = torch.cat((score, score), dim=0)
        selected_dir = _top_per_source(
            directed_score,
            directed_valid,
            torch.cat((self.candidate_i, self.candidate_j), dim=0),
            self.slot_index,
            self.max_neighbors,
        )
        selected_dir = selected_dir | selected_dir[self.reverse_dir]
        return selected_dir[:e] | selected_dir[e:]

    def _uniform_message_edges(self, positive=True):
        ids = self.uniform_pos if positive else self.uniform_neg
        return self._directed_weights(ids, torch.ones_like(ids, dtype=torch.float32))

    def _shuffle_edges(self, ids, weights):
        if ids.numel() == 0:
            return self._directed_weights(ids, weights)
        mapped = self.shuffle_perm[ids]
        src = self.candidate_i[ids]
        dst = self.candidate_j[mapped]
        keep = src != dst
        src, dst, weights = src[keep], dst[keep], weights[keep]
        return torch.cat((src, dst), 0), torch.cat((dst, src), 0), torch.cat((weights, weights), 0)

    @staticmethod
    def _message(raw, src, dst, weight, dims):
        out = raw.new_zeros(raw.shape)
        if src.numel() == 0:
            return out
        for start, dim in zip([0, 48, 54, 64], dims):
            x = raw[:, start:start + dim]
            diff = x[dst] - x[src]
            accum = x.new_zeros(x.shape)
            accum.index_add_(0, src, weight[:, None] * diff)
            degree = x.new_zeros((x.shape[0], 1))
            degree.index_add_(0, src, weight[:, None])
            msg = accum / degree.clamp_min(1e-6)
            scale = x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            out[:, start:start + dim] = scale * torch.tanh(msg / (scale + 1e-6))
        return out

    def _propagate(self, raw, q, group, override=None):
        continuous = group == "E6"
        if override == "off":
            zero = raw.new_zeros(raw.shape)
            return zero, {"positive_edges": 0, "negative_edges": 0, "positive_norm": 0.0, "negative_norm": 0.0}
        if group == "E8" or override == "uniform":
            src_p, dst_p, w_p = self._uniform_message_edges(True)
            src_n, dst_n, w_n = self._uniform_message_edges(False)
        else:
            pos_mask = self._select_sign(q, "positive", 0.55, continuous=continuous)
            neg_mask = self._select_sign(q, "negative", 0.45, continuous=continuous)
            pos_ids = torch.where(pos_mask)[0]
            neg_ids = torch.where(neg_mask)[0]
            pos_weights = ((q[pos_ids] - 0.5) * 2.0).clamp_min(0.0)
            neg_weights = ((0.5 - q[neg_ids]) * 2.0).clamp_min(0.0)
            if group == "E7" or override == "shuffle":
                src_p, dst_p, w_p = self._shuffle_edges(pos_ids, pos_weights)
                src_n, dst_n, w_n = self._shuffle_edges(neg_ids, neg_weights)
            else:
                src_p, dst_p, w_p = self._directed_weights(pos_ids, pos_weights)
                src_n, dst_n, w_n = self._directed_weights(neg_ids, neg_weights)
        if override == "positive_off":
            src_p, dst_p, w_p = raw.new_empty((0,), dtype=torch.long), raw.new_empty((0,), dtype=torch.long), raw.new_empty((0,))
        if override == "negative_off":
            src_n, dst_n, w_n = raw.new_empty((0,), dtype=torch.long), raw.new_empty((0,), dtype=torch.long), raw.new_empty((0,))
        pos_msg = self._message(raw, src_p, dst_p, w_p, DIMS)
        neg_msg = self._message(raw, src_n, dst_n, w_n, DIMS)
        delta = pos_msg - self.gamma_negative * neg_msg
        return self.lambda_prop * delta, {
            "positive_edges": int(src_p.numel()),
            "negative_edges": int(src_n.numel()),
            "positive_norm": float(pos_msg.detach().norm().cpu()),
            "negative_norm": float(neg_msg.detach().norm().cpu()),
            "delta_norm": float(delta.detach().norm().cpu()),
        }

    def forward(
        self,
        parts,
        pair_i,
        pair_j,
        pair_y,
        group,
        reliable_modalities,
        adv_mode,
        propagation_override=None,
    ):
        z, pair_features, pair_logits, pair_loss, adv_loss, adv_logits, adv_modal = self.relation.pair_forward(
            parts, pair_i, pair_j, pair_y, reliable_modalities, adv_mode
        )
        q, q_by_modality, denom, usable = self._candidate_q(z)
        raw = torch.cat(parts, dim=1)
        delta, prop_stats = self._propagate(raw, q, group, override=propagation_override)
        return delta, {
            "z": z,
            "pair_features": pair_features,
            "pair_logits": pair_logits,
            "pair_loss": pair_loss,
            "adv_loss": adv_loss,
            "adv_logits": adv_logits,
            "adv_modal": adv_modal,
            "candidate_q": q,
            "candidate_q_by_modality": q_by_modality,
            "candidate_reliable_denominator": denom,
            "candidate_usable": usable,
            **prop_stats,
        }

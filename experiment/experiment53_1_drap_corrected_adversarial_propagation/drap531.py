#!/usr/bin/env python3
"""Corrected Experiment 53.1 relation adversary and deterministic propagation."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


MODS = ("PHENO", "ANAT", "FUNC", "CORR")
DIMS = (48, 6, 10, 256)
STARTS = (0, 48, 54, 64)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 0.03) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


class RelationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, x):
        return self.net(x)


class RelationScorer(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def pair_features(z, pair_i, pair_j):
        zn = F.normalize(z, dim=-1)
        zi, zj = zn[pair_i], zn[pair_j]
        return torch.cat(((zi - zj).abs(), zi * zj, (zi * zj).sum(-1, keepdim=True)), dim=-1)

    def forward(self, pair_features):
        return self.net(pair_features).squeeze(-1)


class ModalityDiscriminator(nn.Module):
    def __init__(self, pair_dim: int = 65, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(pair_dim + 2, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, pair_features, pair_condition):
        cond = pair_condition.to(dtype=pair_features.dtype)
        return self.net(torch.cat((pair_features, cond), dim=-1))


class RelationCore(nn.Module):
    """State-dict-compatible relation encoders/scorers and a four-source head."""

    def __init__(self, hidden: int = 32, grl_scale: float = 0.03):
        super().__init__()
        self.encoders = nn.ModuleList([RelationEncoder(d, hidden) for d in DIMS])
        self.scorers = nn.ModuleList([RelationScorer(hidden) for _ in DIMS])
        self.discriminator = ModalityDiscriminator(hidden * 2 + 1)
        self.grl_scale = float(grl_scale)

    def encode(self, parts):
        return [enc(x) for enc, x in zip(self.encoders, parts)]

    def pair_forward(self, parts, pair_i, pair_j, pair_y, adv_mode: str, adv_pair_weight):
        z = self.encode(parts)
        features = [scorer.pair_features(z[m], pair_i, pair_j) for m, scorer in enumerate(self.scorers)]
        logits = [self.scorers[m](features[m]) for m in range(4)]
        if pair_y is None or pair_y.numel() == 0:
            zero = parts[0].new_zeros(())
            return z, features, logits, zero, zero, None, None, None

        target = pair_y.to(dtype=logits[0].dtype)
        pair_loss = torch.stack([
            F.binary_cross_entropy_with_logits(lg, target) for lg in logits
        ]).mean()
        if adv_mode == "none":
            zero = parts[0].new_zeros(())
            return z, features, logits, pair_loss, zero, None, None, None

        if adv_mode not in ("conditional", "unconditional"):
            raise ValueError(f"invalid adversarial mode: {adv_mode}")
        n_pair = pair_y.numel()
        rel = torch.cat([grad_reverse(feat, self.grl_scale) for feat in features], dim=0)
        if adv_mode == "conditional":
            condition = F.one_hot(pair_y.long(), num_classes=2).to(dtype=rel.dtype)
        else:
            condition = rel.new_zeros((n_pair, 2))
        condition = condition.repeat(4, 1)
        modality = torch.arange(4, device=rel.device).repeat_interleave(n_pair)
        adv_logits = self.discriminator(rel, condition)
        ce = F.cross_entropy(adv_logits, modality, reduction="none")
        weights = adv_pair_weight.detach().to(dtype=ce.dtype).repeat(4)
        adv_loss = (ce * weights).sum() / weights.sum().clamp_min(1e-8)
        return z, features, logits, pair_loss, adv_loss, adv_logits, modality, condition

    def optimizer_param_groups(self, lr=0.001, matrix_decay=1e-4):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and name.endswith("weight") and "norm" not in name.lower():
                decay.append(p)
            else:
                no_decay.append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": matrix_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]


class DiseaseRelationPropagation(nn.Module):
    """OOF-calibrated disease relation fusion and bounded signed propagation."""

    def __init__(
        self,
        relation: RelationCore,
        candidate_i,
        candidate_j,
        modality_edge_ids,
        slots,
        reverse_dir,
        calibration_a,
        calibration_b,
        omega,
        corr_edge_ids,
        corr_permutation,
        shuffled_pos_ids,
        shuffled_neg_ids,
        shuffled_pos_budget,
        shuffled_neg_budget,
        ordinary_pos_ids,
        ordinary_neg_ids,
        ordinary_pos_weights,
        ordinary_neg_weights,
        gamma_negative=0.25,
        lambda_prop=0.02,
        max_neighbors=12,
        thresholds=(0.55, 0.45),
    ):
        super().__init__()
        self.relation = relation
        for name, value in {
            "candidate_i": candidate_i,
            "candidate_j": candidate_j,
            "slots": slots,
            "reverse_dir": reverse_dir,
            "calibration_a": calibration_a,
            "calibration_b": calibration_b,
            "omega": omega,
            "corr_edge_ids": corr_edge_ids,
            "corr_permutation": corr_permutation,
            "shuffled_pos_ids": shuffled_pos_ids,
            "shuffled_neg_ids": shuffled_neg_ids,
            "shuffled_pos_budget": shuffled_pos_budget,
            "shuffled_neg_budget": shuffled_neg_budget,
            "ordinary_pos_ids": ordinary_pos_ids,
            "ordinary_neg_ids": ordinary_neg_ids,
            "ordinary_pos_weights": ordinary_pos_weights,
            "ordinary_neg_weights": ordinary_neg_weights,
        }.items():
            dtype = torch.long if name.endswith(("_i", "_j", "slots", "reverse_dir", "edge_ids", "permutation", "_ids")) else torch.float32
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        for m, idx in enumerate(modality_edge_ids):
            self.register_buffer(f"modality_edge_{m}", torch.as_tensor(idx, dtype=torch.long))
        self.gamma_negative = float(gamma_negative)
        self.lambda_prop = float(lambda_prop)
        self.max_neighbors = int(max_neighbors)
        self.positive_threshold = float(thresholds[0])
        self.negative_threshold = float(thresholds[1])
        self.shuffled_pos_target_edges = int(len(shuffled_pos_ids))
        self.shuffled_neg_target_edges = int(len(shuffled_neg_ids))

    def modality_edge_ids(self, modality):
        return getattr(self, f"modality_edge_{modality}")

    def candidate_scores(self, z):
        n_edge = self.candidate_i.numel()
        by_mod = []
        present = []
        for m, scorer in enumerate(self.relation.scorers):
            ids = self.modality_edge_ids(m)
            vals = self.candidate_i.new_full((n_edge,), 0.5, dtype=torch.float32)
            mask = torch.zeros(n_edge, dtype=torch.bool, device=self.candidate_i.device)
            if ids.numel():
                feats = scorer.pair_features(z[m], self.candidate_i[ids], self.candidate_j[ids])
                logits = scorer(feats)
                calibrated = torch.sigmoid(self.calibration_a[m] * logits + self.calibration_b[m])
                vals = vals.index_copy(0, ids, calibrated)
                mask[ids] = True
            by_mod.append(vals)
            present.append(mask)
        stack = torch.stack(by_mod, dim=0)
        present_t = torch.stack(present, dim=0)
        usable = present_t & (self.omega[:, None] > 0)
        weights = usable.to(stack.dtype) * self.omega[:, None]
        denom = weights.sum(0)
        fused = (stack * weights).sum(0) / denom.clamp_min(1e-8)
        fused = torch.where(denom > 0, fused, torch.full_like(fused, 0.5))
        return fused, stack, denom, usable

    def _stable_top_per_node(self, q, sign, node_budget=None):
        e = q.numel()
        if not e or self.slots.size(1) == 0:
            return torch.zeros(e, dtype=torch.bool, device=q.device)
        if sign == "positive":
            valid = q.detach() >= self.positive_threshold
        else:
            valid = q.detach() <= self.negative_threshold
        score = (q.detach() - 0.5).abs()
        directed_valid = torch.cat((valid, valid), dim=0)
        directed_score = torch.cat((score, score), dim=0)
        safe = self.slots.clamp_min(0)
        values = directed_score[safe].masked_fill(~(self.slots >= 0), float("-inf"))
        values = values.masked_fill(~directed_valid[safe], float("-inf"))
        order = torch.argsort(values, dim=1, descending=True, stable=True)
        k = min(self.max_neighbors, values.size(1))
        chosen = torch.gather(safe, 1, order[:, :k])
        chosen_values = torch.gather(values, 1, order[:, :k])
        picked = torch.zeros(2 * e, dtype=torch.bool, device=q.device)
        rank = torch.arange(k, device=q.device).view(1, k)
        allowed = torch.ones((self.slots.size(0), k), dtype=torch.bool, device=q.device) if node_budget is None else rank < node_budget.to(device=q.device, dtype=torch.long).clamp(0, k).view(-1, 1)
        good = torch.isfinite(chosen_values) & (chosen >= 0) & allowed
        picked[chosen[good]] = True
        # Mutual top-k keeps the graph symmetric while enforcing the per-node
        # degree budget at both endpoints.
        result = picked[:e] & picked[e:]
        target = self.shuffled_pos_target_edges if sign == "positive" else self.shuffled_neg_target_edges
        current = int(result.sum().item())
        if current > target:
            edge_ids = torch.where(result)[0]
            ranked = torch.argsort(score[edge_ids], descending=True, stable=True)
            keep = edge_ids[ranked[:target]]
            result = torch.zeros_like(result)
            result[keep] = True
        return result

    def _direct(self, ids, weights):
        ids = ids.long()
        if ids.numel() == 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=self.candidate_i.device)
            return empty_idx, empty_idx, weights.new_empty((0,))
        a, b = self.candidate_i[ids], self.candidate_j[ids]
        src = torch.cat((a, b), dim=0)
        dst = torch.cat((b, a), dim=0)
        return src, dst, torch.cat((weights, weights), dim=0)

    def _weight_matrix_message(self, raw, src, dst, weight):
        n = raw.size(0)
        if src.numel() == 0 or not bool((weight.detach() > 0).any()):
            return torch.zeros_like(raw), {
                "effective_directed_edges": 0, "single_neighbor_fraction": 0.0,
                "zero_neighbor_fraction": 1.0, "multi_neighbor_fraction": 0.0,
                "adjacency": raw.new_zeros((n, n)),
            }
        # Candidate selectors create unique directed entries; this avoids atomic scatter reductions.
        adjacency = raw.new_zeros((n, n)).index_put((src, dst), weight, accumulate=False)
        degree = adjacency.sum(dim=1, keepdim=True)
        neighbor_sum = adjacency @ raw
        msg = (neighbor_sum - degree * raw) / degree.clamp_min(1e-8)
        degree_count = (adjacency.detach() > 0).sum(dim=1)
        positive_degree = degree_count > 0
        one = degree_count == 1
        block_msg = torch.zeros_like(raw)
        for start, dim in zip(STARTS, DIMS):
            x = raw[:, start:start + dim]
            m = msg[:, start:start + dim]
            scale = x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            block_msg[:, start:start + dim] = scale * torch.tanh(m / (scale + 1e-6))
        return block_msg, {
            "effective_directed_edges": int((weight.detach() > 0).sum().item()),
            "single_neighbor_fraction": float(one.float().mean().cpu()),
            "zero_neighbor_fraction": float((~positive_degree).float().mean().cpu()),
            "multi_neighbor_fraction": float((degree_count >= 2).float().mean().cpu()),
            "adjacency": adjacency,
        }

    def _propagate(self, raw, q, cfg: Dict, enabled: bool, override: Optional[str] = None):
        zero = torch.zeros_like(raw)
        if not enabled or override == "off":
            return zero, zero, zero, {"positive_edges": 0, "negative_edges": 0, "positive_norm": 0.0, "negative_norm": 0.0, "delta_norm": 0.0}

        q_prop = q
        identity = cfg["identity_mode"]
        shuffled = override == "shuffle" or identity == "per_forward_score_permutation_within_CORR_candidates"
        if shuffled:
            if self.corr_edge_ids.numel():
                permuted = q[self.corr_edge_ids[self.corr_permutation]]
                q_prop = q.index_copy(0, self.corr_edge_ids, permuted)

        if identity == "ordinary_similarity_with_fixed_degree_budgets" or override == "ordinary":
            p_ids, n_ids = self.ordinary_pos_ids, self.ordinary_neg_ids
            wp, wn = self.ordinary_pos_weights, self.ordinary_neg_weights
        elif shuffled:
            p_mask = self._stable_top_per_node(q_prop, "positive", self.shuffled_pos_budget)
            n_mask = self._stable_top_per_node(q_prop, "negative", self.shuffled_neg_budget)
            pw_all = (2.0 * (q_prop - 0.5)).clamp_min(0.0)
            nw_all = (2.0 * (0.5 - q_prop)).clamp_min(0.0)
            p_ids = torch.where(p_mask & (pw_all.detach() > 0))[0]
            n_ids = torch.where(n_mask & (nw_all.detach() > 0))[0]
            wp, wn = pw_all[p_ids], nw_all[n_ids]
        else:
            continuous = cfg["uncertainty_mode"] == "continuous"
            if continuous:
                p_mask = torch.ones_like(q_prop, dtype=torch.bool)
                n_mask = torch.ones_like(q_prop, dtype=torch.bool)
                pw_all = (2.0 * (q_prop - 0.5)).clamp_min(0.0)
                nw_all = (2.0 * (0.5 - q_prop)).clamp_min(0.0)
            else:
                p_mask = self._stable_top_per_node(q_prop, "positive")
                n_mask = self._stable_top_per_node(q_prop, "negative")
                pw_all = (2.0 * (q_prop - 0.5)).clamp_min(0.0)
                nw_all = (2.0 * (0.5 - q_prop)).clamp_min(0.0)
            p_ids = torch.where(p_mask & (pw_all.detach() > 0))[0]
            n_ids = torch.where(n_mask & (nw_all.detach() > 0))[0]
            wp, wn = pw_all[p_ids], nw_all[n_ids]

        if override == "positive_off":
            p_ids, wp = p_ids[:0], wp[:0]
        if override == "negative_off":
            n_ids, wn = n_ids[:0], wn[:0]
        if not cfg["positive_enabled"]:
            p_ids, wp = p_ids[:0], wp[:0]
        if not cfg["negative_enabled"]:
            n_ids, wn = n_ids[:0], wn[:0]

        ps, pd, pw = self._direct(p_ids, wp)
        ns, nd, nw = self._direct(n_ids, wn)
        pmsg, pstat = self._weight_matrix_message(raw, ps, pd, pw)
        nmsg, nstat = self._weight_matrix_message(raw, ns, nd, nw)
        delta = self.lambda_prop * (pmsg - self.gamma_negative * nmsg)
        return delta, pmsg, nmsg, {
            "positive_edges": int((pw.detach() > 0).sum().item()),
            "negative_edges": int((nw.detach() > 0).sum().item()),
            "positive_norm": float(pmsg.detach().norm().cpu()),
            "negative_norm": float(nmsg.detach().norm().cpu()),
            "delta_norm": float(delta.detach().norm().cpu()),
            "positive_single_neighbor_fraction": pstat.get("single_neighbor_fraction", 1.0),
            "negative_single_neighbor_fraction": nstat.get("single_neighbor_fraction", 1.0),
            "positive_zero_neighbor_fraction": pstat.get("zero_neighbor_fraction", 1.0),
            "negative_zero_neighbor_fraction": nstat.get("zero_neighbor_fraction", 1.0),
            "positive_multi_neighbor_fraction": pstat.get("multi_neighbor_fraction", 0.0),
            "negative_multi_neighbor_fraction": nstat.get("multi_neighbor_fraction", 0.0),
            "positive_adjacency": pstat.get("adjacency"),
            "negative_adjacency": nstat.get("adjacency"),
            "q_prop": q_prop,
            "shuffled_target_positive_edges": self.shuffled_pos_target_edges,
            "shuffled_target_negative_edges": self.shuffled_neg_target_edges,
        }

    def forward(self, parts, pair_i, pair_j, pair_y, adv_pair_weight, group_cfg, propagation_enabled=True, override=None):
        z, pair_features, pair_logits, pair_loss, adv_loss, adv_logits, adv_modal, adv_condition = self.relation.pair_forward(
            parts, pair_i, pair_j, pair_y, group_cfg["adversarial_mode"], adv_pair_weight
        )
        q, q_by_modality, denom, usable = self.candidate_scores(z)
        raw = torch.cat(parts, dim=1)
        delta, pos_msg, neg_msg, prop = self._propagate(raw, q, group_cfg, propagation_enabled, override)
        return delta, {
            "z": z,
            "pair_features": pair_features,
            "pair_logits": pair_logits,
            "pair_loss": pair_loss,
            "adv_loss": adv_loss,
            "adv_logits": adv_logits,
            "adv_modal": adv_modal,
            "adv_condition": adv_condition,
            "candidate_q": q,
            "candidate_q_by_modality": q_by_modality,
            "candidate_reliable_denominator": denom,
            "candidate_usable": usable,
            "positive_message": pos_msg,
            "negative_message": neg_msg,
            **prop,
        }

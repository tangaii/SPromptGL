"""CORR-anchored disease-relational alignment integrated inside the E1 GFR."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F


DIMS = (48, 6, 10, 256)
MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def grad_reverse(value: torch.Tensor, scale: float = 0.03) -> torch.Tensor:
    return _GradientReverse.apply(value, scale)


def pair_features(zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
    """Symmetric 65-D representation for an unordered patient pair."""
    cosine = F.cosine_similarity(zi, zj, dim=-1, eps=1e-8).unsqueeze(-1)
    return torch.cat(((zi - zj).abs(), zi * zj, cosine), dim=-1)


class RelationEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1, eps=1e-8)


class RelationScorer(nn.Module):
    def __init__(self, pair_dim: int = 65, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        return self.net(pair_repr).squeeze(-1)


class ConditionalModalityDiscriminator(nn.Module):
    def __init__(self, pair_dim: int = 65, hidden: int = 32, n_modalities: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_modalities),
        )

    def forward(self, pair_repr: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((pair_repr, condition), dim=-1))


def conditional_adversarial_loss(
    pair_reprs: List[torch.Tensor],
    pair_same: torch.Tensor,
    discriminator: ConditionalModalityDiscriminator,
    grl_scale: float = 0.03,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """CORR anchors the source classifier; only the three student encoders see GRL."""
    cond = F.one_hot(pair_same.long(), num_classes=2).to(pair_reprs[0].dtype)
    modality_logits = []
    modality_targets = []
    for modality, relation in enumerate(pair_reprs):
        adversarial_repr = relation.detach() if modality == 3 else grad_reverse(relation, grl_scale)
        modality_logits.append(discriminator(adversarial_repr, cond))
        modality_targets.append(torch.full(
            (relation.shape[0],), modality, dtype=torch.long, device=relation.device
        ))
    logits = torch.cat(modality_logits, dim=0)
    targets = torch.cat(modality_targets, dim=0)
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, {"logits": logits, "targets": targets, "accuracy": accuracy}


def relation_preservation_loss(
    pair_logits: List[torch.Tensor],
    pair_same: torch.Tensor,
    margin: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    same = pair_same.bool()
    gaps = torch.stack([
        logits[same].mean() - logits[~same].mean()
        for logits in pair_logits
    ])
    return F.relu(float(margin) - gaps).mean(), gaps


def _mutual_topk(mask: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    n = mask.shape[0]
    if n <= 1:
        return torch.zeros_like(mask, dtype=torch.bool)
    k = min(int(k), n - 1)
    score = scores.detach().masked_fill(~mask, -torch.inf)
    score = score.clone()
    score.fill_diagonal_(-torch.inf)
    idx = torch.topk(score, k=k, dim=1, largest=True, sorted=True).indices
    directed = torch.zeros_like(mask, dtype=torch.bool)
    directed.scatter_(1, idx, True)
    return directed & directed.T & mask & ~torch.eye(n, device=mask.device, dtype=torch.bool)


def _symmetric_adjacency(
    n: int, row: torch.Tensor, col: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    upper = weight.new_zeros((n, n)).index_put((row, col), weight)
    return upper + upper.T


def _bounded_block_delta(delta: torch.Tensor, raw_block: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(raw_block.square().mean(dim=1, keepdim=True).clamp_min(1e-8))
    return rms * torch.tanh(delta / (rms + 1e-8))


def _weighted_neighbor_difference(adjacency: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    row_sum = adjacency.sum(dim=1, keepdim=True)
    return (adjacency @ raw - row_sum * raw) / row_sum.clamp_min(1.0)


class CADRGFR(nn.Module):
    """One GFR containing E1 GraphLearn/GCNs and disease-relational graph paths.

    `legacy` is an already constructed and anchor-loaded Experiment 51
    LegacyFixedGFR. Its modules are moved into this module without rebuilding
    them, preserving the original construction RNG and parameter names.
    """

    def __init__(
        self,
        legacy: nn.Module,
        reliability: torch.Tensor,
        calibration: torch.Tensor,
        fallback_modal: torch.Tensor,
        init_seed: int,
        grl_scale: float = 0.03,
        edge_topk: int = 12,
        positive_threshold: float = 0.55,
        negative_threshold: float = 0.45,
        negative_scale: float = 0.25,
        direct_scale: float = 0.02,
    ):
        super().__init__()
        self.graphs = legacy.graphs
        self.gcn = legacy.gcn
        self.fusion_weight = legacy.fusion_weight
        self.utils = legacy.utils
        self.grl_scale = float(grl_scale)
        self.edge_topk = int(edge_topk)
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)
        self.negative_scale = float(negative_scale)
        self.direct_scale = float(direct_scale)
        self.register_buffer("reliability", reliability.detach().float().clone())
        self.register_buffer("calibration", calibration.detach().float().clone())
        self.register_buffer("fallback_modal", fallback_modal.detach().bool().clone())

        param_device = next(legacy.parameters()).device
        cuda_devices = [param_device.index if param_device.index is not None else torch.cuda.current_device()] if param_device.type == "cuda" else []
        init_scope = torch.random.fork_rng(devices=cuda_devices, enabled=True) if torch.cuda.is_available() else nullcontext()
        with init_scope:
            torch.manual_seed(int(init_seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(init_seed))
            self.encoders = nn.ModuleList([
                RelationEncoder(dim, 32) for dim in DIMS
            ])
            self.scorers = nn.ModuleList([RelationScorer(65, 32) for _ in DIMS])
            self.discriminator = ConditionalModalityDiscriminator(65, 32, 4)
            self.edge_gate_logits = nn.Parameter(torch.zeros(4))
            gate_start = torch.logit(reliability.detach().float().clamp(0.02, 0.98))
            self.reliability_gate_logits = nn.Parameter(gate_start.clone())
        self.to(param_device)
        self.last: Dict[str, object] = {}

    def base_parameters(self):
        yield from self.graphs.parameters()
        yield from self.gcn.parameters()
        yield self.fusion_weight

    def added_named_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith(("graphs.", "gcn.", "fusion_weight")):
                yield name, parameter

    def _encode_pairs(
        self,
        parts: List[torch.Tensor],
        pair_index: Optional[torch.Tensor],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        embeddings = [encoder(x) for encoder, x in zip(self.encoders, parts)]
        if pair_index is None:
            return embeddings, [], []
        left = pair_index[:, 0].long()
        right = pair_index[:, 1].long()
        pair_reprs = [pair_features(z[left], z[right]) for z in embeddings]
        pair_logits = [scorer(r) for scorer, r in zip(self.scorers, pair_reprs)]
        return embeddings, pair_reprs, pair_logits

    def _candidate_support(
        self,
        modality: int,
        raw_modality: torch.Tensor,
        original_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = original_adj.shape[0]
        original_support = (original_adj > 0).detach()
        original_support.fill_diagonal_(False)
        raw_norm = F.normalize(raw_modality, dim=-1, eps=1e-8)
        raw_cosine = raw_norm @ raw_norm.T
        raw_similarity = ((raw_cosine + 1.0) * 0.5).clamp(0.0, 1.0)

        support = original_support
        selection_scores = original_adj
        if bool(self.fallback_modal[modality].item()):
            fallback = _mutual_topk(
                torch.ones_like(original_support, dtype=torch.bool), raw_cosine, self.edge_topk
            )
            support = original_support | fallback
            selection_scores = torch.where(original_support, original_adj, raw_cosine)

        selected = _mutual_topk(support, selection_scores, self.edge_topk)
        row, col = torch.triu(selected, diagonal=1).nonzero(as_tuple=True)
        is_original = original_support[row, col]
        similarity = torch.where(is_original, original_adj[row, col], raw_similarity[row, col])
        return selected, row, col, similarity

    def forward(
        self,
        parts: List[torch.Tensor],
        pair_index: Optional[torch.Tensor] = None,
        use_cadr: bool = True,
        intervention: str = "full",
    ) -> tuple[torch.Tensor, List[torch.Tensor], Dict[str, object]]:
        raw = torch.cat(parts, dim=1)
        original_adjs = [graph(x) for graph, x in zip(self.graphs, parts)]
        embeddings, pair_reprs, pair_logits = self._encode_pairs(parts, pair_index)

        if (not use_cadr) or intervention == "disease_off":
            fused = raw
            messages = []
            for modality, adjacency in enumerate(original_adjs):
                identity = torch.eye(adjacency.shape[0], device=raw.device, dtype=raw.dtype)
                normalized = self.utils.normalize_adj(adjacency + identity)
                message = self.gcn[modality](raw, normalized)
                fused = fused + torch.sigmoid(self.fusion_weight[modality]) * message
                messages.append(message)
            info = {
                "original_adjs": original_adjs,
                "positive_adjs": [torch.zeros_like(a) for a in original_adjs],
                "negative_adjs": [torch.zeros_like(a) for a in original_adjs],
                "positive_messages": messages,
                "negative_messages": [torch.zeros_like(raw) for _ in original_adjs],
                "direct_messages": [torch.zeros_like(raw) for _ in original_adjs],
                "messages": messages,
                "fusion_gates": torch.zeros(4, device=raw.device),
                "edge_q": [raw.new_empty((0,)) for _ in original_adjs],
                "edge_rows": [torch.empty(0, dtype=torch.long, device=raw.device) for _ in original_adjs],
                "edge_cols": [torch.empty(0, dtype=torch.long, device=raw.device) for _ in original_adjs],
                "edge_similarity": [raw.new_empty((0,)) for _ in original_adjs],
                "candidate_masks": [torch.zeros_like(a, dtype=torch.bool) for a in original_adjs],
                "pair_reprs": pair_reprs,
                "pair_logits": pair_logits,
                "active": False,
            }
            self.last = info
            return fused, original_adjs, info

        n = raw.shape[0]
        identity = torch.eye(n, device=raw.device, dtype=raw.dtype)
        positive_adjs, negative_adjs = [], []
        positive_messages, negative_messages, direct_messages, messages = [], [], [], []
        edge_q, edge_rows, edge_cols, edge_sims, candidate_masks = [], [], [], [], []
        edge_diagnostics = []
        fused = raw
        fusion_gates = []
        for modality, (x_mod, z, scorer, original_adj) in enumerate(
            zip(parts, embeddings, self.scorers, original_adjs)
        ):
            candidate, row, col, similarity = self._candidate_support(modality, x_mod, original_adj)
            repr_edges = pair_features(z[row], z[col])
            raw_logits = scorer(repr_edges)
            cal_a, cal_b = self.calibration[modality]
            q = torch.sigmoid(cal_a * raw_logits + cal_b)
            pos_mask = (q >= self.positive_threshold).detach()
            neg_mask = (q <= self.negative_threshold).detach()
            pos_conf = (2.0 * (q - 0.5)).clamp(0.0, 1.0)
            neg_conf = (2.0 * (0.5 - q)).clamp(0.0, 1.0)
            rel = self.reliability[modality]
            edge_gate = torch.sigmoid(self.edge_gate_logits[modality])
            pos_weight = similarity * pos_conf * pos_mask.to(q.dtype) * rel * edge_gate
            neg_weight = similarity * neg_conf * neg_mask.to(q.dtype) * rel * edge_gate
            a_pos = _symmetric_adjacency(n, row, col, pos_weight)
            a_neg = _symmetric_adjacency(n, row, col, neg_weight)

            if intervention == "positive_off":
                a_pos = torch.zeros_like(a_pos)
            if intervention == "negative_off":
                a_neg = torch.zeros_like(a_neg)
            if intervention == "modality_off:" + MODALITIES[modality]:
                a_pos = torch.zeros_like(a_pos)
                a_neg = torch.zeros_like(a_neg)

            norm_pos = self.utils.normalize_adj(a_pos + identity)
            norm_neg = self.utils.normalize_adj(a_neg + identity)
            h_pos = self.gcn[modality](raw, norm_pos)
            h_self = self.gcn[modality](raw, identity)
            h_neg = self.gcn[modality](raw, norm_neg) - h_self

            pos_diff = _weighted_neighbor_difference(a_pos, raw)
            neg_diff = _weighted_neighbor_difference(a_neg, raw)
            direct_blocks = []
            offset = 0
            for dim in DIMS:
                block = raw[:, offset:offset + dim]
                direct_blocks.append(
                    _bounded_block_delta(pos_diff[:, offset:offset + dim], block)
                    - self.negative_scale * _bounded_block_delta(
                        neg_diff[:, offset:offset + dim], block
                    )
                )
                offset += dim
            direct = torch.cat(direct_blocks, dim=1)
            if intervention == "direct_off":
                direct = torch.zeros_like(direct)

            message = h_pos - self.negative_scale * h_neg + self.direct_scale * direct
            alpha = torch.sigmoid(self.reliability_gate_logits[modality])
            if intervention == "modality_off:" + MODALITIES[modality]:
                alpha = alpha * 0.0
            fused = fused + torch.sigmoid(self.fusion_weight[modality]) * alpha * message

            positive_adjs.append(a_pos)
            negative_adjs.append(a_neg)
            positive_messages.append(h_pos)
            negative_messages.append(h_neg)
            direct_messages.append(direct)
            messages.append(message)
            edge_q.append(q)
            edge_rows.append(row)
            edge_cols.append(col)
            edge_sims.append(similarity)
            candidate_masks.append(candidate)
            edge_diagnostics.append({
                "candidate_edges": int(candidate.sum().item() // 2),
                "scored_edges": int(q.numel()),
                "positive_eligible_edges": int(pos_mask.sum().item()),
                "negative_eligible_edges": int(neg_mask.sum().item()),
                "positive_edges": int((pos_weight.detach() > 0).sum().item()),
                "negative_edges": int((neg_weight.detach() > 0).sum().item()),
                "uncertain_edges": int((~pos_mask & ~neg_mask).sum().item()),
                "degree_mean_pos": float(a_pos.detach().sum(1).mean().item()),
                "degree_mean_neg": float(a_neg.detach().sum(1).mean().item()),
                "alpha": float(alpha.detach().item()),
                "edge_gate": float(edge_gate.detach().item()),
                "q_mean": float(q.detach().mean().item()) if q.numel() else float("nan"),
                "q_std": float(q.detach().std(unbiased=False).item()) if q.numel() else float("nan"),
            })
            fusion_gates.append(alpha)

        info = {
            "original_adjs": original_adjs,
            "positive_adjs": positive_adjs,
            "negative_adjs": negative_adjs,
            "positive_messages": positive_messages,
            "negative_messages": negative_messages,
            "direct_messages": direct_messages,
            "messages": messages,
            "fusion_gates": torch.stack(fusion_gates),
            "edge_q": edge_q,
            "edge_rows": edge_rows,
            "edge_cols": edge_cols,
            "edge_similarity": edge_sims,
            "candidate_masks": candidate_masks,
            "edge_diagnostics": edge_diagnostics,
            "pair_reprs": pair_reprs,
            "pair_logits": pair_logits,
            "active": True,
        }
        self.last = info
        return fused, original_adjs, info

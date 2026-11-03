"""Experiment 61 adaptive propagation path, isolated from historical sources."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cadr_gfr import (
    MODALITIES,
    DIMS,
    _bounded_block_delta,
    _symmetric_adjacency,
    _weighted_neighbor_difference,
)


def e2e_adaptive_forward(
    model,
    parts: List[torch.Tensor],
    pair_index: Optional[torch.Tensor],
    intervention: str = "full",
    permute_scores: bool = False,
    permutation_seed: int = 0,
):
    """E2 graph path with a differentiable scalar blend and hard edge support."""
    raw = torch.cat(parts, dim=1)
    n = raw.shape[0]
    identity = torch.eye(n, dtype=raw.dtype, device=raw.device)
    original_adjs = [graph(x) for graph, x in zip(model.graphs, parts)]
    embeddings, pair_reprs, pair_logits = model._encode_pairs(parts, pair_index)
    messages_original, messages_pos, messages_neg, messages_direct, messages_cadr = [], [], [], [], []
    positive_adjs, negative_adjs, edge_q, edge_rows, edge_cols = [], [], [], [], []
    edge_sims, candidate_masks, diagnostics = [], [], []
    fused = raw
    gate_values = torch.sigmoid(model.adaptive_gate_logits)

    for modality, (x_mod, z, scorer, original_adj) in enumerate(
        zip(parts, embeddings, model.scorers, original_adjs)
    ):
        candidate, row, col, similarity = model._candidate_support(modality, x_mod, original_adj)
        from cadr_gfr import pair_features

        relation_repr = pair_features(z[row], z[col])
        logits = scorer(relation_repr)
        if permute_scores and logits.numel() > 1:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(int(permutation_seed) + modality)
            logits = logits[torch.randperm(logits.numel(), generator=generator, device=logits.device)]
        q = torch.sigmoid(logits)
        pos_mask = (q >= model.positive_threshold).detach()
        neg_mask = (q <= model.negative_threshold).detach()
        pos_conf = (2.0 * (q - 0.5)).clamp(0.0, 1.0)
        neg_conf = (2.0 * (0.5 - q)).clamp(0.0, 1.0)
        edge_gate = torch.sigmoid(model.edge_gate_logits[modality])
        pos_weight = similarity * pos_conf * pos_mask.to(q.dtype) * edge_gate
        neg_weight = similarity * neg_conf * neg_mask.to(q.dtype) * edge_gate
        a_pos = _symmetric_adjacency(n, row, col, pos_weight)
        a_neg = _symmetric_adjacency(n, row, col, neg_weight)
        if intervention == "positive_off":
            a_pos = torch.zeros_like(a_pos)
        if intervention == "negative_off":
            a_neg = torch.zeros_like(a_neg)
        if intervention == "modality_off:" + MODALITIES[modality]:
            a_pos = torch.zeros_like(a_pos)
            a_neg = torch.zeros_like(a_neg)

        norm_original = model.utils.normalize_adj(original_adj + identity)
        m_original = model.gcn[modality](raw, norm_original)
        norm_pos = model.utils.normalize_adj(a_pos + identity)
        norm_neg = model.utils.normalize_adj(a_neg + identity)
        m_pos = model.gcn[modality](raw, norm_pos)
        m_neg = model.gcn[modality](raw, norm_neg) - model.gcn[modality](raw, identity)

        pos_diff = _weighted_neighbor_difference(a_pos, raw)
        neg_diff = _weighted_neighbor_difference(a_neg, raw)
        direct_blocks = []
        offset = 0
        for dim in DIMS:
            block = raw[:, offset:offset + dim]
            direct_blocks.append(
                _bounded_block_delta(pos_diff[:, offset:offset + dim], block)
                - model.negative_scale * _bounded_block_delta(
                    neg_diff[:, offset:offset + dim], block
                )
            )
            offset += dim
        m_direct = torch.cat(direct_blocks, dim=1)
        if intervention == "direct_off":
            m_direct = torch.zeros_like(m_direct)
        m_cadr = m_pos - model.negative_scale * m_neg + model.direct_scale * m_direct
        if intervention == "disease_off":
            gate = gate_values[modality] * 0.0
        elif intervention == "modality_off:" + MODALITIES[modality]:
            gate = gate_values[modality] * 0.0
        else:
            gate = gate_values[modality]
        mixed = (1.0 - gate) * m_original + gate * m_cadr
        fused = fused + torch.sigmoid(model.fusion_weight[modality]) * mixed

        messages_original.append(m_original)
        messages_pos.append(m_pos)
        messages_neg.append(m_neg)
        messages_direct.append(m_direct)
        messages_cadr.append(m_cadr)
        positive_adjs.append(a_pos)
        negative_adjs.append(a_neg)
        edge_q.append(q)
        edge_rows.append(row)
        edge_cols.append(col)
        edge_sims.append(similarity)
        candidate_masks.append(candidate)
        diagnostics.append({
            "candidate_edges": int(candidate.sum().item() // 2),
            "scored_edges": int(q.numel()),
            "positive_eligible_edges": int(pos_mask.sum().item()),
            "negative_eligible_edges": int(neg_mask.sum().item()),
            "positive_edges": int((pos_weight.detach() > 0).sum().item()),
            "negative_edges": int((neg_weight.detach() > 0).sum().item()),
            "uncertain_edges": int((~pos_mask & ~neg_mask).sum().item()),
            "q_mean": float(q.detach().mean().item()) if q.numel() else None,
            "q_std": float(q.detach().std(unbiased=False).item()) if q.numel() else None,
            "adaptive_gate": float(gate.detach().item()),
            "edge_gate": float(edge_gate.detach().item()),
            "propagation_mode": "E2E_ADAPTIVE_BLEND",
        })

    info: Dict[str, object] = {
        "original_adjs": original_adjs,
        "positive_adjs": positive_adjs,
        "negative_adjs": negative_adjs,
        "messages_original": messages_original,
        "positive_messages": messages_pos,
        "negative_messages": messages_neg,
        "direct_messages": messages_direct,
        "messages_cadr": messages_cadr,
        "messages": messages_cadr,
        "fusion_gates": gate_values,
        "edge_q": edge_q,
        "edge_rows": edge_rows,
        "edge_cols": edge_cols,
        "edge_similarity": edge_sims,
        "candidate_masks": candidate_masks,
        "pair_reprs": pair_reprs,
        "pair_logits": pair_logits,
        "propagation_modes": ["E2E_ADAPTIVE_BLEND"] * 4,
        "edge_diagnostics": diagnostics,
        "active": True,
    }
    model.last = info
    return fused, original_adjs, info

"""Reliability-guided unified disease-aware graph propagation for E74.1."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
DIMS = (48, 6, 10, 256)
GATE_SCALES = (0.01, 0.01, 0.01, 1.0)
REWEIGHT_STRENGTH = 0.8
DIRECT_SCALE = 0.02
CHUNK_SIZE = 16384


def _pair_score_chunks(embedding, row, col, scorer, chunk_size=CHUNK_SIZE):
    from cadr_gfr import pair_features

    scores = []
    for start in range(0, row.numel(), chunk_size):
        stop = min(start + chunk_size, row.numel())
        features = pair_features(embedding[row[start:stop]], embedding[col[start:stop]])
        scores.append(scorer(features))
    if scores:
        return torch.cat(scores, dim=0)
    return embedding.new_empty((0,))


def _original_mutual_topk(model, modality, features, original, k=12):
    """Use the historical candidate rule, then discard every fallback-only edge."""
    selected, _, _, _ = model._candidate_support(modality, features, original)
    effective = (original > 0).detach()
    effective.fill_diagonal_(False)
    selected_effective = selected & effective
    row, col = torch.triu(selected_effective, diagonal=1).nonzero(as_tuple=True)
    return selected, row, col


def _symmetric_delta(n, row, col, delta):
    upper = delta.new_zeros((n, n)).index_put((row, col), delta)
    return upper + upper.transpose(0, 1)


def _score_transform(q, modality, intervention, fold, permutation_seed_base):
    if intervention == "disease_neutral":
        return torch.full_like(q, 0.5)
    if intervention == "corr_neutral" and modality == 3:
        return torch.full_like(q, 0.5)
    if intervention == "score_permuted" and q.numel() > 1:
        generator = torch.Generator(device=q.device)
        generator.manual_seed(int(permutation_seed_base) + int(fold) * 100 + modality)
        return q[torch.randperm(q.numel(), generator=generator, device=q.device)]
    return q


def _edge_delta(original, row, col, q, gate, modality):
    if modality == 3:
        relation = 2.0 * q - 1.0
    else:
        eligible = ((q >= 0.55) | (q <= 0.45)).detach()
        relation = torch.where(eligible, 2.0 * q - 1.0, torch.zeros_like(q))
    old = original[row, col]
    delta = old * (REWEIGHT_STRENGTH * gate * relation)
    return delta, relation


def _bounded_neighbor_delta(adjacency, raw):
    from cadr_gfr import _bounded_block_delta

    n = adjacency.shape[0]
    off_diagonal = adjacency * (1.0 - torch.eye(n, dtype=adjacency.dtype, device=adjacency.device))
    degree = off_diagonal.sum(dim=1, keepdim=True)
    difference = (off_diagonal @ raw - degree * raw) / degree.clamp_min(1.0)
    blocks = []
    start = 0
    for width in DIMS:
        stop = start + width
        blocks.append(_bounded_block_delta(difference[:, start:stop], raw[:, start:stop]))
        start = stop
    return difference, torch.cat(blocks, dim=1)


def unified_adjacency(original, row, col, q, gate, modality, intervention="full",
                      fold=0, permutation_seed_base=741001):
    """Reweight original support only and preserve the GraphLearn diagonal."""
    q_used = _score_transform(q, modality, intervention, fold, permutation_seed_base)
    if intervention == "disease_neutral" or (intervention == "corr_neutral" and modality == 3):
        gate_used = gate.new_zeros(())
    else:
        gate_used = gate
    delta, relation = _edge_delta(original, row, col, q_used, gate_used, modality)
    adjacency = original + _symmetric_delta(original.shape[0], row, col, delta)
    return adjacency, {
        "q_raw": q,
        "q_used": q_used,
        "relation": relation,
        "delta_values": delta,
        "row": row,
        "col": col,
        "gate": gate,
    }


def rudag_forward(model, parts, pair_index: Optional[torch.Tensor], group,
                  intervention="full", fold=0, collect_diagnostics=False,
                  permutation_seed_base=741001):
    """One GraphLearn, one unified graph and one GCN call per modality."""
    from cadr_gfr import MODALITIES as BASE_MODALITIES

    if tuple(BASE_MODALITIES) != MODALITIES:
        raise RuntimeError(f"unexpected DARG modality order: {BASE_MODALITIES}")
    direct_enabled = group == "G2-RUDAG-Full" and intervention != "direct_off"
    raw = torch.cat(parts, dim=1)
    n = raw.shape[0]
    identity = torch.eye(n, dtype=raw.dtype, device=raw.device)
    original_adjs = [graph(features) for graph, features in zip(model.graphs, parts)]
    embeddings, pair_reprs, pair_logits = model._encode_pairs(parts, pair_index)

    unified_adjs, messages, direct_messages, differences = [], [], [], []
    scored_q, q_used_all, edge_rows, edge_cols = [], [], [], []
    edge_diagnostics = []
    fused = raw
    gate_logits = model.adaptive_gate_logits

    for modality, (features, embedding, scorer, original) in enumerate(
        zip(parts, embeddings, model.scorers, original_adjs)
    ):
        effective = (original > 0).detach()
        effective.fill_diagonal_(False)
        if modality == 3:
            selected = effective
            row, col = torch.triu(selected, diagonal=1).nonzero(as_tuple=True)
            candidate_all = selected
            candidate_effective = selected
            fallback_candidates = 0
        else:
            candidate_all, _, _ = _original_mutual_topk(model, modality, features, original)
            candidate_effective = candidate_all & effective
            row, col = torch.triu(candidate_effective, diagonal=1).nonzero(as_tuple=True)
            fallback_candidates = int(torch.triu(candidate_all & ~effective, diagonal=1).sum().item())

        logits = _pair_score_chunks(embedding, row, col, scorer)
        q = torch.sigmoid(logits)
        gate = GATE_SCALES[modality] * torch.sigmoid(gate_logits[modality])
        adjacency, edge_info = unified_adjacency(
            original, row, col, q, gate, modality, intervention,
            fold=fold, permutation_seed_base=permutation_seed_base,
        )
        normalized = model.utils.normalize_adj(adjacency + identity)
        message = model.gcn[modality](raw, normalized)
        if direct_enabled:
            difference, bounded = _bounded_neighbor_delta(adjacency, raw)
            direct = DIRECT_SCALE * gate * bounded
        else:
            difference = torch.zeros_like(raw)
            direct = torch.zeros_like(raw)
        fused = fused + torch.sigmoid(model.fusion_weight[modality]) * (message + direct)

        unified_adjs.append(adjacency)
        messages.append(message)
        direct_messages.append(direct)
        differences.append(difference)
        scored_q.append(q)
        q_used_all.append(edge_info["q_used"])
        edge_rows.append(row)
        edge_cols.append(col)

        if collect_diagnostics:
            with torch.no_grad():
                delta_matrix = adjacency - original
                changed = delta_matrix.abs() > 0
                positive = edge_info["relation"] > 0
                negative = edge_info["relation"] < 0
                qv = edge_info["q_used"]
                offdiag = adjacency * (1.0 - torch.eye(n, dtype=adjacency.dtype, device=adjacency.device))
                degree = offdiag.sum(1)
                edge_count = int(torch.triu(effective, diagonal=1).sum().item())
                selected_count = int(row.numel())
                edge_diagnostics.append({
                    "modality": MODALITIES[modality],
                    "original_effective_edges": edge_count,
                    "candidate_edges_before_support_intersection": int(torch.triu(candidate_all, diagonal=1).sum().item()),
                    "fallback_candidates_excluded": fallback_candidates,
                    "scored_edges": selected_count,
                    "modified_edges": int(torch.triu(changed, diagonal=1).sum().item()),
                    "modified_edge_ratio": float(torch.triu(changed, diagonal=1).sum().item() / max(edge_count, 1)),
                    "corr_effective_edge_coverage": float(selected_count / max(edge_count, 1)) if modality == 3 else None,
                    "positive_enhancement_edges": int(((edge_info["delta_values"] > 0) & positive).sum().item()),
                    "negative_suppression_edges": int(((edge_info["delta_values"] < 0) & negative).sum().item()),
                    "positive_score_edges": int(positive.sum().item()),
                    "negative_score_edges": int(negative.sum().item()),
                    "uncertain_score_edges": int((~positive & ~negative).sum().item()),
                    "q_mean": float(qv.mean().item()) if qv.numel() else 0.0,
                    "q_std": float(qv.std(unbiased=False).item()) if qv.numel() else 0.0,
                    "q_positive_fraction": float((qv >= 0.55).float().mean().item()) if qv.numel() else 0.0,
                    "q_negative_fraction": float((qv <= 0.45).float().mean().item()) if qv.numel() else 0.0,
                    "q_uncertain_fraction": float(((qv > 0.45) & (qv < 0.55)).float().mean().item()) if qv.numel() else 0.0,
                    "gate_effective": float(gate.detach().item()),
                    "gate_theta": float(gate_logits[modality].detach().item()),
                    "adjacency_change_frobenius": float(delta_matrix.norm().item()),
                    "adjacency_change_relative": float((delta_matrix.norm() / original.norm().clamp_min(1e-12)).item()),
                    "adjacency_symmetry_max_abs": float((adjacency - adjacency.T).abs().max().item()),
                    "minimum_edge_weight": float(adjacency.min().item()),
                    "mean_unweighted_degree": float((adjacency.gt(0).sum(1) - adjacency.diag().gt(0).long()).float().mean().item()),
                    "mean_weighted_degree_offdiag": float(degree.mean().item()),
                    "normalized_graph_finite": bool(torch.isfinite(normalized).all().item()),
                    "gcn_message_raw_l2_ratio": float((message.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "direct_difference_raw_l2_ratio": float((difference.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "weighted_direct_message_raw_l2_ratio": float((direct.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "total_message_raw_l2_ratio": float(((message + direct).norm() / raw.norm().clamp_min(1e-12)).item()),
                    "gcn_calls": 1,
                })

    info = {
        "original_adjs": original_adjs,
        "unified_adjs": unified_adjs,
        "messages": messages,
        "direct_messages": direct_messages,
        "neighbor_differences": differences,
        "pair_reprs": pair_reprs,
        "pair_logits": pair_logits,
        "edge_q": scored_q,
        "edge_q_used": q_used_all,
        "edge_rows": edge_rows,
        "edge_cols": edge_cols,
        "edge_diagnostics": edge_diagnostics,
        "direct_enabled": direct_enabled,
        "gates": [GATE_SCALES[i] * torch.sigmoid(gate_logits[i]) for i in range(4)],
        "propagation_modes": ["RUDAG_UNIFIED_GRAPH"] * 4,
        "intervention": intervention,
    }
    model.last = info
    return fused, original_adjs, info


def forward_state(state, e61, group, training=False, intervention="full",
                  fold=0, collect_diagnostics=False, permutation_seed_base=741001):
    from cadr_gfr import conditional_adversarial_loss, relation_preservation_loss

    import torch.nn.functional as F

    state["model"].train(training)
    state["cam"].train(training)
    fused, original_adjs, info = rudag_forward(
        state["model"], state["parts"], state["train_pairs"], group,
        intervention=intervention, fold=fold,
        collect_diagnostics=collect_diagnostics,
        permutation_seed_base=permutation_seed_base,
    )
    logp, hidden, attention = state["cam"](fused)
    cls = e61.load_base().runtime()["utils"].ClsLoss(
        logp, state["y"], state["train"], state["weights"]
    )
    graph, graph_parts = e61.load_base()._graph_loss(state["parts"], original_adjs)
    pair = torch.stack([
        F.binary_cross_entropy_with_logits(logits, state["train_same"])
        for logits in info["pair_logits"]
    ]).mean()
    adv, adv_info = conditional_adversarial_loss(
        info["pair_reprs"], state["train_same"],
        state["model"].discriminator, state["model"].grl_scale,
    )
    preserve, preserve_gaps = relation_preservation_loss(
        info["pair_logits"], state["train_same"], margin=0.02
    )
    total = cls + graph + 0.05 * pair + 0.01 * adv + 0.01 * preserve
    return {
        "fused": fused,
        "logp": logp,
        "hidden": hidden,
        "attention": attention,
        "original_adjs": original_adjs,
        "unified_adjs": info["unified_adjs"],
        "info": info,
        "cls": cls,
        "graph": graph,
        "graph_parts": graph_parts,
        "pair": pair,
        "adv": adv,
        "adv_info": adv_info,
        "preserve": preserve,
        "preserve_gaps": preserve_gaps,
        "total": total,
    }

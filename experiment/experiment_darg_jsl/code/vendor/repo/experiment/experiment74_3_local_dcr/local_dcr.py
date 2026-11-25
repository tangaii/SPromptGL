"""Modality-local disease-contrastive residual propagation for Experiment 74.3."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
DIMS = (48, 6, 10, 256)
GATE_SCALES = (0.01, 0.01, 0.01, 1.0)
REWEIGHT_STRENGTH = 0.8
CHUNK_SIZE = 16384
CADR_PATH = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py")
CADR_MODULE_NAME = "cadr_gfr_e53_localdcr743"
BLOCK_SLICES = ((0, 48), (48, 54), (54, 64), (64, 320))
GROUP_SPECS = {
    "E74.3-G1-Local-DCR": {"direct_scale": 0.02, "positive_only": False},
    "E74.3-G2-Local-DCR-3x": {"direct_scale": 0.06, "positive_only": False},
    "E74.3-G3-Positive-Local-DCR": {"direct_scale": 0.06, "positive_only": True},
}


def _cadr_module():
    module = sys.modules.get(CADR_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(CADR_MODULE_NAME, CADR_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[CADR_MODULE_NAME] = module
        spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != CADR_PATH.resolve():
        raise RuntimeError("Local-DCR loaded an unexpected CADR relation module")
    return module


def _pair_score_chunks(embedding, row, col, scorer, chunk_size=CHUNK_SIZE):
    pair_features = _cadr_module().pair_features
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


def _bounded_disease_residual(base_adjacency, delta_adjacency, raw):
    bounded_block_delta = _cadr_module()._bounded_block_delta
    n = raw.shape[0]
    identity = torch.eye(n, dtype=raw.dtype, device=raw.device)
    off_diagonal_mask = 1.0 - identity
    delta_off = delta_adjacency * off_diagonal_mask
    base_degree = (base_adjacency * off_diagonal_mask).sum(dim=1, keepdim=True)
    residual = (delta_off @ raw - delta_off.sum(dim=1, keepdim=True) * raw) / base_degree.clamp_min(1.0)
    blocks = []
    start = 0
    for width in DIMS:
        stop = start + width
        blocks.append(bounded_block_delta(residual[:, start:stop], raw[:, start:stop]))
        start = stop
    return residual, torch.cat(blocks, dim=1)


def direct_delta_for_group(delta_adjacency, group):
    if group not in GROUP_SPECS:
        raise ValueError(f"unknown Local-DCR group: {group}")
    if GROUP_SPECS[group]["positive_only"]:
        return delta_adjacency.clamp_min(0.0)
    return delta_adjacency


def localize_residual(bounded_residual, modality):
    if modality not in range(4):
        raise ValueError(f"invalid modality index: {modality}")
    start, stop = BLOCK_SLICES[modality]
    feature_mask = bounded_residual.new_zeros((bounded_residual.shape[1],))
    feature_mask[start:stop] = 1.0
    return bounded_residual * feature_mask.unsqueeze(0)


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
    BASE_MODALITIES = _cadr_module().MODALITIES
    if tuple(BASE_MODALITIES) != MODALITIES:
        raise RuntimeError(f"unexpected DARG modality order: {BASE_MODALITIES}")
    if group not in GROUP_SPECS:
        raise ValueError(f"unknown Local-DCR group: {group}")
    group_spec = GROUP_SPECS[group]
    direct_enabled = intervention != "direct_off"
    raw = torch.cat(parts, dim=1)
    n = raw.shape[0]
    identity = torch.eye(n, dtype=raw.dtype, device=raw.device)
    original_adjs = [graph(features) for graph, features in zip(model.graphs, parts)]
    embeddings, pair_reprs, pair_logits = model._encode_pairs(parts, pair_index)

    unified_adjs, messages, direct_messages = [], [], []
    weighted_messages, weighted_direct_messages = [], []
    disease_delta_adjs, direct_delta_adjs, raw_residuals = [], [], []
    bounded_unmasked_residuals, bounded_residuals = [], []
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
        delta_adjacency = adjacency - original
        delta_adjacency = delta_adjacency * (1.0 - identity)
        direct_delta = direct_delta_for_group(delta_adjacency, group)
        if direct_enabled:
            residual, bounded_unmasked = _bounded_disease_residual(original, direct_delta, raw)
            bounded = localize_residual(bounded_unmasked, modality)
            direct = group_spec["direct_scale"] * gate * bounded
        else:
            residual = torch.zeros_like(raw)
            bounded_unmasked = torch.zeros_like(raw)
            bounded = torch.zeros_like(raw)
            direct = torch.zeros_like(raw)
        fusion_weight = torch.sigmoid(model.fusion_weight[modality])
        weighted_message = fusion_weight * message
        weighted_direct = fusion_weight * direct
        fused = fused + fusion_weight * (message + direct)

        unified_adjs.append(adjacency)
        messages.append(message)
        direct_messages.append(direct)
        weighted_messages.append(weighted_message)
        weighted_direct_messages.append(weighted_direct)
        disease_delta_adjs.append(delta_adjacency)
        direct_delta_adjs.append(direct_delta)
        raw_residuals.append(residual)
        bounded_unmasked_residuals.append(bounded_unmasked)
        bounded_residuals.append(bounded)
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
                    "deltaA_relative_to_base": float((delta_adjacency.norm() / original.norm().clamp_min(1e-12)).item()),
                    "direct_delta_relative_to_base": float((direct_delta.norm() / original.norm().clamp_min(1e-12)).item()),
                    "direct_positive_increment_edges": int(torch.triu(direct_delta > 0, diagonal=1).sum().item()),
                    "direct_negative_increment_edges": int(torch.triu(direct_delta < 0, diagonal=1).sum().item()),
                    "disease_residual_raw_l2_ratio": float((residual.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "disease_residual_bounded_unmasked_l2_ratio": float((bounded_unmasked.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "disease_residual_bounded_local_l2_ratio": float((bounded.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "direct_scale": group_spec["direct_scale"],
                    "positive_only_direct": group_spec["positive_only"],
                    "fusion_weighted_gcn_message_raw_l2_ratio": float((weighted_message.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "fusion_weighted_dcr_message_raw_l2_ratio": float((weighted_direct.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "fusion_weighted_gcn_to_dcr_norm_ratio": float((weighted_message.norm() / weighted_direct.norm().clamp_min(1e-12)).item()),
                    "adjacency_symmetry_max_abs": float((adjacency - adjacency.T).abs().max().item()),
                    "minimum_edge_weight": float(adjacency.min().item()),
                    "mean_unweighted_degree": float((adjacency.gt(0).sum(1) - adjacency.diag().gt(0).long()).float().mean().item()),
                    "mean_weighted_degree_offdiag": float(degree.mean().item()),
                    "normalized_graph_finite": bool(torch.isfinite(normalized).all().item()),
                    "gcn_message_raw_l2_ratio": float((message.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "disease_residual_l2_ratio": float((residual.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "weighted_direct_message_raw_l2_ratio": float((direct.norm() / raw.norm().clamp_min(1e-12)).item()),
                    "total_message_raw_l2_ratio": float(((message + direct).norm() / raw.norm().clamp_min(1e-12)).item()),
                    "gcn_calls": 1,
                })

    info = {
        "original_adjs": original_adjs,
        "unified_adjs": unified_adjs,
        "messages": messages,
        "direct_messages": direct_messages,
        "weighted_messages": weighted_messages,
        "weighted_direct_messages": weighted_direct_messages,
        "disease_delta_adjs": disease_delta_adjs,
        "direct_delta_adjs": direct_delta_adjs,
        "disease_residuals_raw": raw_residuals,
        "disease_residuals_bounded_unmasked": bounded_unmasked_residuals,
        "disease_residuals_bounded": bounded_residuals,
        "pair_reprs": pair_reprs,
        "pair_logits": pair_logits,
        "edge_q": scored_q,
        "edge_q_used": q_used_all,
        "edge_rows": edge_rows,
        "edge_cols": edge_cols,
        "edge_diagnostics": edge_diagnostics,
        "direct_enabled": direct_enabled,
        "direct_scale": group_spec["direct_scale"],
        "positive_only_direct": group_spec["positive_only"],
        "gates": [GATE_SCALES[i] * torch.sigmoid(gate_logits[i]) for i in range(4)],
        "propagation_modes": ["RUDAG_UNIFIED_GRAPH"] * 4,
        "intervention": intervention,
        "fused_delta_relative_l2": float(((fused - raw).norm() / raw.norm().clamp_min(1e-12)).detach().item()),
    }
    model.last = info
    return fused, original_adjs, info


def forward_state(state, e61, group, training=False, intervention="full",
                  fold=0, collect_diagnostics=False, permutation_seed_base=741001):
    cadr = _cadr_module()
    conditional_adversarial_loss = cadr.conditional_adversarial_loss
    relation_preservation_loss = cadr.relation_preservation_loss

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

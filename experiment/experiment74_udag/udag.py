"""UDAG adjacency replacement. No new modules, randomness, or classifier."""
from __future__ import annotations
import torch


def unified_adjacency(original, row, col, similarity, q, gate,
                      positive_threshold=.55, negative_threshold=.45,
                      intervention="full"):
    """Only selected off-diagonal edges change; scatter an undirected delta."""
    from cadr_gfr import _symmetric_adjacency
    old = original[row, col]
    existing = (old > 0).detach()
    positive = (q >= positive_threshold).detach()
    negative = (q <= negative_threshold).detach()
    enabled_pos = positive if intervention != "positive_off" else torch.zeros_like(positive)
    enabled_neg = negative if intervention != "negative_off" else torch.zeros_like(negative)
    if intervention == "disease_off":
        enabled_pos = torch.zeros_like(positive)
        enabled_neg = torch.zeros_like(negative)
    pos_delta = torch.where(existing, old, similarity) * (2 * (q - .5)).clamp(0, 1) * gate * enabled_pos.to(q.dtype)
    neg_delta = old * (2 * (.5 - q)).clamp(0, 1) * gate * (enabled_neg & existing).to(q.dtype)
    delta = pos_delta - neg_delta
    updated = original + _symmetric_adjacency(original.shape[0], row, col, delta)
    return updated, {"existing": existing, "positive": positive, "negative": negative,
                     "positive_delta": pos_delta, "negative_delta": neg_delta,
                     "delta": delta}


def udag_forward(model, parts, pair_index=None, intervention="full", collect_reference=False):
    """Four native GraphLearn calls and exactly one GCN per modality normally.

    collect_reference adds original-GCN calls ONLY in isolated eval audits.
    Graph construction losses always receive the unmodified original_adjs.
    """
    from cadr_gfr import MODALITIES, pair_features
    raw = torch.cat(parts, dim=1)
    eye = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
    original_adjs = [graph(x) for graph, x in zip(model.graphs, parts)]
    embeddings, pair_reprs, pair_logits = model._encode_pairs(parts, pair_index)
    unified, messages, originals, diagnostics, qs = [], [], [], [], []
    fused = raw
    for m, (x, z, scorer, original) in enumerate(zip(parts, embeddings, model.scorers, original_adjs)):
        candidate, row, col, similarity = model._candidate_support(m, x, original)
        logits = scorer(pair_features(z[row], z[col]))
        q = torch.sigmoid(logits)
        gate = torch.sigmoid(model.edge_gate_logits[m])
        adj, change = unified_adjacency(original, row, col, similarity, q, gate,
                                       model.positive_threshold, model.negative_threshold, intervention)
        normalized = model.utils.normalize_adj(adj + eye)
        message = model.gcn[m](raw, normalized)
        fused = fused + torch.sigmoid(model.fusion_weight[m]) * message
        if collect_reference:
            originals.append(model.gcn[m](raw, model.utils.normalize_adj(original + eye)))
        unified.append(adj); messages.append(message); qs.append(q)
        ex, pos, neg = change["existing"], change["positive"], change["negative"]
        od, ad = original.detach(), adj.detach()
        selected_delta = change["delta"].detach()
        diagnostics.append({
            "modality": MODALITIES[m], "original_edges": int(torch.triu(od > 0, diagonal=1).sum()),
            "candidate_edges": int(q.numel()), "original_candidate_edges": int(ex.sum()),
            "fallback_candidate_edges": int((~ex).sum()), "positive_edges": int(pos.sum()),
            "negative_edges": int(neg.sum()), "uncertain_edges": int((~pos & ~neg).sum()),
            "fallback_positive_added": int(((~ex) & (change["positive_delta"].detach() > 0)).sum()),
            "original_positive_enhanced": int((ex & (change["positive_delta"].detach() > 0)).sum()),
            "original_negative_suppressed": int((ex & (change["negative_delta"].detach() > 0)).sum()),
            "mean_selected_weight_change": float(selected_delta.mean()) if q.numel() else 0.,
            "mean_absolute_selected_weight_change": float(selected_delta.abs().mean()) if q.numel() else 0.,
            "mean_positive_existing_enhancement_ratio": float((change["positive_delta"][ex & pos] / original[row[ex & pos], col[ex & pos]]).detach().mean()) if (ex & pos).any() else 0.,
            "mean_negative_existing_suppression_ratio": float((change["negative_delta"][ex & neg] / original[row[ex & neg], col[ex & neg]]).detach().mean()) if (ex & neg).any() else 0.,
            "final_edges": int(torch.triu(ad > 0, diagonal=1).sum()),
            "final_mean_degree": float(((ad > 0).sum(1) - (ad.diag() > 0).long()).float().mean()),
            "final_mean_weighted_degree_offdiag": float((ad.sum(1) - ad.diag()).mean()),
            "adjacency_change_frobenius": float((ad-od).norm()),
            "adjacency_change_relative": float((ad-od).norm()/od.norm().clamp_min(1e-12)),
            "q_mean": float(q.detach().mean()) if q.numel() else 0.,
            "q_std": float(q.detach().std(unbiased=False)) if q.numel() else 0.,
            "edge_gate": float(gate.detach()), "minimum_edge_weight": float(ad.min()),
            "symmetry_max_abs": float((ad-ad.T).abs().max()),
            "normalized_finite": bool(torch.isfinite(normalized).all()),
            "gcn_calls_formal": 1,
        })
    info = {"original_adjs": original_adjs, "unified_adjs": unified,
            "messages": messages, "messages_original": originals,
            "edge_q": qs, "edge_diagnostics": diagnostics,
            "pair_reprs": pair_reprs, "pair_logits": pair_logits,
            "active": intervention != "disease_off", "propagation_modes": ["UDAG_SINGLE_GRAPH"]*4}
    model.last = info
    return fused, original_adjs, info


def forward_state(state, e61, training=False, intervention="full", collect_reference=False):
    import torch.nn.functional as F
    from cadr_gfr import conditional_adversarial_loss, relation_preservation_loss
    state["model"].train(training); state["cam"].train(training)
    fused, adjs, info = udag_forward(state["model"], state["parts"], state["train_pairs"], intervention, collect_reference)
    logp, hidden, attention = state["cam"](fused)
    cls = e61.load_base().runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    graph, graph_parts = e61.load_base()._graph_loss(state["parts"], adjs)
    pair = torch.stack([F.binary_cross_entropy_with_logits(logits, state["train_same"]) for logits in info["pair_logits"]]).mean()
    adv, adv_info = conditional_adversarial_loss(info["pair_reprs"], state["train_same"], state["model"].discriminator, state["model"].grl_scale)
    preserve, gaps = relation_preservation_loss(info["pair_logits"], state["train_same"], margin=.02)
    return {"fused":fused,"logp":logp,"hidden":hidden,"attention":attention,"adjs":adjs,"info":info,
            "cls":cls,"graph":graph,"graph_parts":graph_parts,"pair":pair,"adv":adv,"adv_info":adv_info,
            "preserve":preserve,"preserve_gaps":gaps,"total":cls+graph+.05*pair+.01*adv+.01*preserve}

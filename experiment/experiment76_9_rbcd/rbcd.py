"""Risk-aware Boundary-Conditioned Disease Correction (E76.9).

The implementation keeps E76.8's legal patient-prototype retrieval path but changes
what boundary prototypes are allowed to do: reliable prototypes define disease
orientation, while boundary prototypes enter the risk estimator as confusion
features.  No boundary-only positive class score is added.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from path_adapter import legacy_module


MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
SLOTS = 4
CAM_DIM = 144
SLOT_DIM = 36
TOP_K = 8
TEMPERATURE = 0.20


def correct_logp(base_logp: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    shifted = base_logp + torch.stack((delta / 2.0, -delta / 2.0), dim=-1)
    normalized = F.log_softmax(shifted, dim=-1)
    # Preserve bit-exact base probabilities at the zero-residual initializer while
    # retaining the derivative through delta for the first branch update.
    if bool((delta.detach() == 0).all()):
        return normalized + (base_logp - normalized).detach()
    return normalized


def _safe_mean(x: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim) if x.numel() else x.new_zeros(x.shape[:-1])


def _boundary_eta(attention_asd: torch.Tensor, attention_td: torch.Tensor,
                  reliability_asd: torch.Tensor, reliability_td: torch.Tensor,
                  coverage_asd: torch.Tensor, coverage_td: torch.Tensor) -> torch.Tensor:
    reliability = torch.sqrt((reliability_asd * reliability_td).clamp_min(0.0))
    coverage = torch.minimum(coverage_asd, coverage_td)
    return (((attention_asd + attention_td) * 0.5) * reliability * coverage).clamp(0.0, 1.0)


class RBCD(nn.Module):
    """Patient-conditioned risk and directional residual branch.

    Boundary prototypes are deliberately excluded from the disease direction.  The
    risk vector is assembled from current base probabilities, prototype evidence,
    support quality, disagreement, and frozen semantic attention.
    """

    _site_diverse_indices = staticmethod(legacy_module.BCSPC._site_diverse_indices)
    _confusion_weights = legacy_module.BCSPC._confusion_weights
    semantic_slots = legacy_module.BCSPC.semantic_slots

    RISK_DIM = 38
    CORRECTION_DIM = 28

    def __init__(self, brainprompt: torch.Tensor, spromptgl: torch.Tensor):
        super().__init__()
        if tuple(brainprompt.shape) != (2, 1024) or tuple(spromptgl.shape) != (8, 1024):
            raise ValueError("frozen prompt vectors must have shapes (2,1024) and (8,1024)")
        self.register_buffer("brainprompt_vectors", F.normalize(brainprompt.detach().float(), dim=-1))
        self.register_buffer("spromptgl_vectors", F.normalize(spromptgl.detach().float(), dim=-1))
        self.semantic_projection = nn.Sequential(nn.Linear(2048, 16), nn.GELU(), nn.Linear(16, 144))
        self.query_projection = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.prototype_projection = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.reliable_text_projection = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.boundary_text_projection = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.modality_gate = nn.Sequential(nn.Linear(12, 16), nn.GELU(), nn.Linear(16, 4))
        self.risk_mlp = nn.Sequential(nn.Linear(self.RISK_DIM, 16), nn.GELU(), nn.Linear(16, 1))
        self.correction_mlp = nn.Sequential(nn.Linear(self.CORRECTION_DIM, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.constant_(self.risk_mlp[-1].bias, -2.0)
        nn.init.zeros_(self.correction_mlp[-1].weight)
        nn.init.zeros_(self.correction_mlp[-1].bias)

    def _pool(self, q: torch.Tensor, current: torch.Tensor, memory: dict,
              ids: torch.Tensor, weights: torch.Tensor, sites: torch.Tensor,
              slot: int, query_ids: torch.Tensor):
        batch = q.shape[0]
        if ids.numel() == 0:
            return (q.new_zeros((batch, SLOT_DIM)), q.new_zeros(batch),
                    torch.full((batch, TOP_K), -1, dtype=torch.long, device=q.device),
                    q.new_zeros((batch, TOP_K)), 0, 0)
        with torch.no_grad():
            mem_slots = F.normalize(memory["features"][ids, slot], dim=-1)
            index_query = q.detach().clone()
            query_ids = torch.as_tensor(query_ids, device=q.device, dtype=torch.long)
            observed = memory["observed"][query_ids]
            if observed.any():
                index_query[observed] = F.normalize(
                    memory["features"][query_ids[observed], slot], dim=-1
                )
            sim_index = index_query @ mem_slots.T
            positions, fallback = self._site_diverse_indices(sim_index, ids, sites, TOP_K)
            chosen = ids[positions]
            if (chosen == query_ids[:, None]).any():
                raise RuntimeError("illegal prototype self retrieval")
            chosen_sites = sites[chosen]
            site_counts = (chosen_sites[:, :, None] == chosen_sites[:, None, :]).sum(-1).max(-1).values
            violations = int(((site_counts > 2) & (fallback == 0)).sum().item())
        support_values = current[chosen, slot]  # live features from this CAM forward
        similarities = (q[:, None, :] * support_values).sum(-1)
        reliabilities = weights[chosen].clamp_min(1e-8)
        attention = torch.softmax(similarities / TEMPERATURE + reliabilities.log(), dim=-1)
        prototype = F.normalize((attention[..., None] * support_values).sum(1), dim=-1)
        selected = torch.full((batch, TOP_K), -1, dtype=torch.long, device=q.device)
        pooling = q.new_zeros((batch, TOP_K))
        selected[:, :chosen.shape[1]] = chosen
        pooling[:, :attention.shape[1]] = attention
        return prototype, (attention * reliabilities).sum(-1), selected, pooling, int(fallback.sum()), violations

    def _prototypes(self, query: torch.Tensor, current: torch.Tensor, memory: dict,
                    support: torch.Tensor, sites: torch.Tensor, query_ids: torch.Tensor,
                    boundary_off: bool, reliable_only: bool):
        proto, sim, counts, reliability, selected, pooling, candidates = [], [], [], [], [], [], []
        fallbacks = violations = invalid = 0
        for cls in (0, 1):
            ids = support[memory["labels"][support] == cls]
            predicted = memory["probabilities"][ids].argmax(-1)
            weight_all = query.new_zeros(871)
            weight_all[ids] = (
                (predicted == cls).to(query.dtype)
                * ((memory["probabilities"][ids, cls] - 0.5) / 0.5).clamp(0, 1)
            )
            rel_ids = ids[weight_all[ids] > 0]
            class_proto, class_sim, class_count, class_rel = [], [], [], []
            class_ids, class_pool, class_candidates = [], [], []
            for slot in range(SLOTS):
                if boundary_off or reliable_only:
                    bnd_ids, bnd_all = rel_ids, weight_all
                else:
                    bnd_ids, bnd_weights, bad = self._confusion_weights(
                        memory, support, ids, cls, slot
                    )
                    invalid += int(bad)
                    good = bnd_weights > 0
                    bnd_ids, bnd_weights = bnd_ids[good], bnd_weights[good]
                    bnd_all = query.new_zeros(871)
                    bnd_all[bnd_ids] = bnd_weights
                role_proto, role_sim, role_count, role_rel = [], [], [], []
                role_ids, role_pool, role_candidate = [], [], []
                for ids_role, weights in ((rel_ids, weight_all), (bnd_ids, bnd_all)):
                    p, r, chosen, pw, fb, vi = self._pool(
                        query[:, slot], current, memory, ids_role, weights, sites, slot, query_ids
                    )
                    fallbacks += fb
                    violations += vi
                    role_proto.append(p)
                    role_sim.append((query[:, slot] * p).sum(-1))
                    role_count.append((chosen >= 0).sum(-1).to(query.dtype))
                    role_rel.append(r)
                    role_ids.append(chosen)
                    role_pool.append(pw)
                    role_candidate.append(query.new_full((query.shape[0],), len(ids_role)))
                class_proto.append(torch.stack(role_proto, 1))
                class_sim.append(torch.stack(role_sim, -1))
                class_count.append(torch.stack(role_count, -1))
                class_rel.append(torch.stack(role_rel, -1))
                class_ids.append(torch.stack(role_ids, 1))
                class_pool.append(torch.stack(role_pool, 1))
                class_candidates.append(torch.stack(role_candidate, -1))
            proto.append(torch.stack(class_proto, 1))
            sim.append(torch.stack(class_sim, 1))
            counts.append(torch.stack(class_count, 1))
            reliability.append(torch.stack(class_rel, 1))
            selected.append(torch.stack(class_ids, 1))
            pooling.append(torch.stack(class_pool, 1))
            candidates.append(torch.stack(class_candidates, 1))
        return (torch.stack(proto, 1), torch.stack(sim, 1), torch.stack(counts, 1),
                torch.stack(reliability, 1), torch.stack(selected, 1), torch.stack(pooling, 1),
                torch.stack(candidates, 1), fallbacks, violations, invalid)

    def forward(self, cam144: torch.Tensor, query_ids, memory: dict, support_ids,
                sites: torch.Tensor, base_logp: torch.Tensor, *, residual_off=False,
                boundary_off=False, semantic_swap=False, no_semantic_key=False,
                reliable_only=False, risk_gate_off=False):
        if tuple(cam144.shape) != (871, CAM_DIM):
            raise ValueError(f"RBCD expects CAM144 shape (871,144), got {tuple(cam144.shape)}")
        query_ids = torch.as_tensor(query_ids, device=cam144.device, dtype=torch.long)
        support = torch.as_tensor(support_ids, device=cam144.device, dtype=torch.long)
        support = support[memory["observed"][support] & (memory["labels"][support] >= 0)]
        current = F.normalize(cam144.reshape(871, SLOTS, SLOT_DIM), dim=-1)
        query = current[query_ids]
        semantic = self.semantic_slots()
        if semantic_swap:
            semantic = semantic.flip(0)
        if no_semantic_key:
            semantic = torch.zeros_like(semantic)
        prototypes, similarities, counts, reliability, selected, pooling, candidates, fallbacks, violations, invalid = self._prototypes(
            query, current, memory, support, sites, query_ids, boundary_off, reliable_only
        )
        coverage = (counts > 0).to(query.dtype)
        qk = self.query_projection(query)[:, None, :, None, :]
        role_text = torch.stack((self.reliable_text_projection(semantic), self.boundary_text_projection(semantic)), 2)
        keys = self.prototype_projection(prototypes) + role_text[None]
        role_logits = (qk * keys).sum(-1) / math.sqrt(SLOT_DIM)
        attention = torch.softmax(role_logits.masked_fill(coverage == 0, -1e4), dim=-1)
        attention = attention * coverage
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        if reliable_only:
            attention = torch.stack((coverage[..., 0], torch.zeros_like(coverage[..., 0])), -1)
            coverage = coverage.clone(); coverage[..., 1] = 0
            counts = counts.clone(); counts[..., 1] = 0
            reliability = reliability.clone(); reliability[..., 1] = 0
        reliable_difference = similarities[:, 0, :, 0] - similarities[:, 1, :, 0]
        boundary_difference = similarities[:, 0, :, 1] - similarities[:, 1, :, 1]
        if boundary_off or reliable_only:
            boundary_difference = torch.zeros_like(boundary_difference)
        modality_input = torch.cat((reliable_difference, boundary_difference,
                                    reliable_difference - boundary_difference), -1)
        modality_weights = torch.softmax(self.modality_gate(modality_input), -1)
        d_rel = (modality_weights * reliable_difference).sum(-1)
        d_boundary = (modality_weights * boundary_difference).sum(-1)
        base = base_logp[query_ids, 0] - base_logp[query_ids, 1]
        probability = base_logp[query_ids].exp()
        entropy = -(probability * base_logp[query_ids]).sum(-1)
        rel_quality = reliability[:, :, 0].mean(-1)
        bnd_quality = reliability[:, :, 1].mean(-1)
        rel_coverage = coverage[:, :, 0].mean(-1)
        bnd_coverage = coverage[:, :, 1].mean(-1)
        boundary_competition = similarities[:, :, :, 1].mean(-1)
        semantic_attention = attention[:, :, :, 0].mean(1)
        disagreement = reliable_difference - boundary_difference
        opposition = torch.sigmoid(-base.detach() * d_rel / 0.20)
        risk_features = torch.cat((
            probability, base.detach().abs()[:, None], entropy[:, None],
            reliable_difference, boundary_difference, disagreement,
            rel_quality, bnd_quality, rel_coverage, bnd_coverage,
            boundary_competition, (-base.detach() * d_rel)[:, None], opposition[:, None],
            modality_weights, semantic_attention, d_rel[:, None], d_boundary[:, None],
        ), -1)
        if risk_features.shape[-1] != self.RISK_DIM:
            raise RuntimeError(f"RBCD risk feature width changed: {risk_features.shape[-1]} != {self.RISK_DIM}")
        risk_logit = self.risk_mlp(risk_features).squeeze(-1)
        risk = torch.sigmoid(risk_logit)
        if risk_gate_off:
            risk = torch.zeros_like(risk)
        disease_direction = torch.tanh(d_rel / 0.20)
        correction_features = torch.cat((
            base.detach()[:, None], base.detach().abs()[:, None], risk[:, None], opposition[:, None],
            d_rel[:, None], d_boundary[:, None], disagreement, modality_weights, semantic_attention,
            rel_quality, bnd_quality, rel_coverage, bnd_coverage, boundary_competition,
        ), -1)
        if correction_features.shape[-1] != self.CORRECTION_DIM:
            raise RuntimeError(f"RBCD correction feature width changed: {correction_features.shape[-1]} != {self.CORRECTION_DIM}")
        activation = self.correction_mlp(correction_features).squeeze(-1)
        delta = (base.detach().abs() + 0.5) * risk * opposition * torch.tanh(activation) * disease_direction
        if residual_off:
            delta = torch.zeros_like(delta)
        final_logp = correct_logp(base_logp[query_ids], delta)
        return final_logp, {
            "delta": delta, "activation": activation, "base_score": base,
            "final_score": base + delta, "risk_logit": risk_logit, "risk": risk,
            "opposition": opposition, "disease_direction": disease_direction,
            "d_rel": d_rel, "d_boundary": d_boundary,
            "reliable_difference": reliable_difference, "boundary_difference": boundary_difference,
            "disagreement": disagreement, "modality_weights": modality_weights,
            "semantic_attention": semantic_attention, "attention": attention,
            "support_topk": counts, "support_coverage": coverage, "support_reliability": reliability,
            "candidate_count": candidates, "selected_ids": selected, "pool_weights": pooling,
            "reliable_similarity": similarities[..., 0], "boundary_similarity": similarities[..., 1],
            "boundary_competition": boundary_competition,
            "retrieval_fallbacks": int(fallbacks), "site_cap_violations": int(violations),
            "invalid_confusion_candidates": int(invalid),
        }


def init_isolated(brainprompt: torch.Tensor, spromptgl: torch.Tensor, fold: int):
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(769101 + int(fold))
        torch.cuda.manual_seed_all(769101 + int(fold))
        return RBCD(brainprompt, spromptgl)


def weighted_nll(logp: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor,
                 sample_weight: torch.Tensor) -> torch.Tensor:
    nll = -logp[torch.arange(labels.numel(), device=labels.device), labels]
    denom = (class_weights[labels] * sample_weight).sum().clamp_min(1e-12)
    return (nll * class_weights[labels] * sample_weight).sum() / denom


def branch_losses(final_logp: torch.Tensor, diagnostics: dict, labels: torch.Tensor,
                  class_weights: torch.Tensor, risk_soft: torch.Tensor,
                  risk_hard: torch.Tensor, base_logp: torch.Tensor):
    sample_weight = 1.0 + 2.0 * risk_hard.to(final_logp.dtype)
    cls_final = weighted_nll(final_logp, labels, class_weights, sample_weight)
    risk = F.binary_cross_entropy_with_logits(diagnostics["risk_logit"], risk_soft, reduction="mean")
    true_direction = torch.where(labels == 0, torch.ones_like(labels, dtype=final_logp.dtype),
                                 -torch.ones_like(labels, dtype=final_logp.dtype))
    base_pred = base_logp.argmax(-1)
    hard_focus = (risk_hard > 0.5) & (base_pred != labels)
    if bool(hard_focus.any()):
        direction = F.softplus(0.20 - true_direction[hard_focus] * diagnostics["delta"][hard_focus]).mean()
    else:
        direction = final_logp.new_zeros(())
    base_correct = base_pred == labels
    keep = (base_correct.to(final_logp.dtype) * diagnostics["delta"].square()).mean()
    residual = diagnostics["delta"].square().mean()
    total = cls_final + 0.20 * risk + 0.10 * direction + 0.10 * keep + 0.02 * residual
    return total, {"cls_final": cls_final, "risk": risk, "direction": direction,
                   "keep": keep, "residual": residual, "sample_weight_mean": sample_weight.mean()}

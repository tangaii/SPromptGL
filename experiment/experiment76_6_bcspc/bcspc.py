"""Boundary-Consistent Semantic Prototype Correction for Experiment 76.6."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


PROMPTS_ASD_TD = (
    "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
)


def split_support(train_ids, labels):
    """Stable class-stratified split; IDs are sorted before alternating A/B."""
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    a, b = [], []
    for cls in (0, 1):
        members = sorted(int(i) for i in train_ids if int(labels_np[int(i)]) == cls)
        a.extend(members[::2])
        b.extend(members[1::2])
    return sorted(a), sorted(b)


def init_isolated(brainprompt_vectors, sprompt_vectors, seed=766101):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return BCSPC(brainprompt_vectors, sprompt_vectors)


class BCSPC(nn.Module):
    def __init__(self, brainprompt_vectors: torch.Tensor, sprompt_vectors: torch.Tensor):
        super().__init__()
        if brainprompt_vectors.ndim != 2 or brainprompt_vectors.shape[0] != 2:
            raise ValueError("BrainPrompt vectors must have ASD=0, TD=1 rows")
        if sprompt_vectors.ndim != 2 or sprompt_vectors.shape[0] != 8:
            raise ValueError("SPromptGL vectors must contain the original eight prompts")
        if brainprompt_vectors.shape[1] != sprompt_vectors.shape[1]:
            raise ValueError("BrainPrompt and SPromptGL CLIP dimensions differ")
        self.register_buffer("brainprompt_vectors", F.normalize(brainprompt_vectors.detach().float(), dim=-1))
        self.register_buffer("spromptgl_vectors", F.normalize(sprompt_vectors.detach().float(), dim=-1))

        self.semantic_projection = nn.Sequential(
            nn.Linear(brainprompt_vectors.shape[-1] + sprompt_vectors.shape[-1], 16),
            nn.GELU(),
            nn.Linear(16, 144),
        )
        self.query_projection = nn.Linear(36, 36, bias=False)
        self.prototype_projection = nn.Linear(36, 36, bias=False)
        self.reliable_text_projection = nn.Linear(36, 36, bias=False)
        self.boundary_text_projection = nn.Linear(36, 36, bias=False)
        self.modality_gate = nn.Sequential(nn.Linear(144 + 12, 16), nn.GELU(), nn.Linear(16, 4))
        self.residual_gate = nn.Sequential(nn.Linear(50, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.residual_gate[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(()))

    def semantic_slots(self):
        sprompt_groups = torch.stack((self.spromptgl_vectors[4:8].mean(0),
                                      self.spromptgl_vectors[0:4].mean(0)))
        paired = torch.cat((self.brainprompt_vectors, sprompt_groups), dim=-1)
        return self.semantic_projection(paired).reshape(2, 4, 36)

    @staticmethod
    def _site_diverse_indices(sim: torch.Tensor, ids: torch.Tensor, sites: torch.Tensor, k: int):
        """Select a stable Top-K with a strict two-per-site cap and explicit fallback."""
        batch, count = sim.shape
        if count == 0:
            empty = torch.empty((batch, 0), dtype=torch.long, device=sim.device)
            return empty, torch.zeros(batch, dtype=torch.long, device=sim.device)
        ids = ids.to(sim.device)
        site_ids = sites.to(sim.device)[ids]
        per_site_scores, per_site_positions = [], []
        for site in torch.unique(site_ids, sorted=True):
            positions = torch.nonzero(site_ids == site, as_tuple=False).flatten()
            local = sim[:, positions]
            take = min(2, positions.numel())
            order = torch.argsort(local, dim=-1, descending=True, stable=True)[:, :take]
            per_site_scores.append(local.gather(1, order))
            per_site_positions.append(positions[order])
        candidates = torch.cat(per_site_positions, dim=1)
        candidate_scores = torch.cat(per_site_scores, dim=1)
        wanted = min(k, count)
        diverse_order = torch.argsort(candidate_scores, dim=-1, descending=True, stable=True)
        diverse_count = candidates.shape[1]
        take_diverse = min(wanted, diverse_count)
        selected = candidates.gather(1, diverse_order[:, :take_diverse])
        fallback_counts = torch.zeros(batch, dtype=torch.long, device=sim.device)
        if take_diverse < wanted:
            global_order = torch.argsort(sim, dim=-1, descending=True, stable=True)
            rows = []
            for row in range(batch):
                chosen = selected[row].tolist()
                seen = set(chosen)
                fallback = 0
                for position in global_order[row].tolist():
                    if position in seen:
                        continue
                    chosen.append(position)
                    seen.add(position)
                    fallback += 1
                    if len(chosen) == wanted:
                        break
                rows.append(torch.tensor(chosen, dtype=torch.long, device=sim.device))
                fallback_counts[row] = fallback
            selected = torch.stack(rows)
        return selected, fallback_counts

    def _confusion_weights(self, memory: Dict[str, torch.Tensor], support: torch.Tensor,
                           candidate_ids: torch.Tensor, cls: int, slot: int):
        labels = memory["labels"]
        probabilities = memory["probabilities"]
        features = F.normalize(memory["features"][:, slot], dim=-1)
        if support.numel() == 0 or candidate_ids.numel() == 0:
            return candidate_ids, probabilities.new_zeros(candidate_ids.numel()), 0
        support_labels = labels[support]
        same_pool = support[support_labels == cls]
        other_pool = support[support_labels != cls]
        if same_pool.numel() < 2 or other_pool.numel() == 0:
            return candidate_ids[:0], probabilities.new_zeros(0), int(candidate_ids.numel())
        similarity = features[candidate_ids] @ features[support].T
        same_mask = (support_labels == cls)[None, :].expand(candidate_ids.numel(), -1).clone()
        same_mask &= candidate_ids[:, None] != support[None, :]
        other_mask = (support_labels != cls)[None, :].expand(candidate_ids.numel(), -1)
        sim_same = similarity.masked_fill(~same_mask, -torch.inf).max(-1).values
        sim_other = similarity.masked_fill(~other_mask, -torch.inf).max(-1).values
        valid = torch.isfinite(sim_same) & torch.isfinite(sim_other)
        candidates = candidate_ids[valid]
        confusion = sim_other[valid] - sim_same[valid]
        p_y = probabilities[candidates, cls]
        weights = torch.sigmoid(confusion / 0.10) * (0.5 + 0.5 * (1.0 - p_y))
        return candidates, weights, int((~valid).sum().item())

    def _retrieve(self, query: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                  memory: Dict[str, torch.Tensor], sites: torch.Tensor, slot: int, top_k: int):
        if ids.numel() == 0:
            zeros = query.new_zeros((query.shape[0], 36))
            empty = torch.empty((query.shape[0], 0), dtype=torch.long, device=query.device)
            return zeros, query.new_zeros(query.shape[0]), query.new_zeros(query.shape[0]), 0, empty, 0
        ids = ids.to(query.device)
        features = F.normalize(memory["features"][ids, slot], dim=-1)
        similarities = F.normalize(query, dim=-1) @ features.T
        positions, fallbacks = self._site_diverse_indices(similarities, ids, sites, min(top_k, ids.numel()))
        selected_ids = ids[positions]
        selected_sim = similarities.gather(1, positions)
        selected_features = memory["features"][selected_ids, slot]
        reliability = weights[selected_ids].clamp_min(1e-8)
        pool_weights = torch.softmax(selected_sim / 0.20 + reliability.log(), dim=-1)
        prototype = F.normalize((pool_weights[..., None] * selected_features).sum(1), dim=-1)
        support_count = torch.full((query.shape[0],), positions.shape[1], dtype=query.dtype, device=query.device)
        selected_sites = sites.to(query.device)[selected_ids]
        site_counts = (selected_sites[:, :, None] == selected_sites[:, None, :]).sum(-1)
        violations = int(((site_counts.max(-1).values > 2) & (fallbacks == 0)).sum().item())
        return (prototype, selected_sim.max(-1).values, support_count,
                int(fallbacks.sum().item()), positions, violations)

    def forward(self, cam144: torch.Tensor, memory: Dict[str, torch.Tensor], support_ids: torch.Tensor,
                sites: torch.Tensor, base_score: torch.Tensor, *, no_boundary: bool = False,
                residual_off: bool = False, boundary_off: bool = False, semantic_swap: bool = False):
        if cam144.ndim != 2 or cam144.shape[-1] != 144:
            raise ValueError("BCSPC requires the captured CAM144 representation")
        query = F.normalize(cam144.reshape(-1, 4, 36), dim=-1)
        support = support_ids.to(query.device)
        support = support[memory["observed"][support] & (memory["labels"][support] >= 0)]
        semantic = self.semantic_slots()
        if semantic_swap:
            semantic = semantic.flip(0)
        prototypes_rel, prototypes_bnd = [], []
        rel_similarity, bnd_similarity = [], []
        supports = torch.zeros((query.shape[0], 2, 4, 2), dtype=query.dtype, device=query.device)
        coverage = torch.zeros_like(supports)
        fallback_total = 0
        invalid_confusion_total = 0
        selected_site_violations = 0

        for cls in (0, 1):
            class_ids = support[memory["labels"][support] == cls]
            predicted = memory["probabilities"][class_ids].argmax(-1)
            class_probability = memory["probabilities"][class_ids, cls]
            reliable_weights_all = torch.zeros(memory["labels"].shape[0], dtype=query.dtype, device=query.device)
            rel_mask = predicted == cls
            rel_ids = class_ids[rel_mask]
            rel_weights = ((class_probability[rel_mask] - 0.5) / 0.5).clamp(0, 1)
            reliable_weights_all[rel_ids] = rel_weights
            class_rel, class_bnd, class_rel_sim, class_bnd_sim = [], [], [], []
            for slot in range(4):
                rel_ids_slot = rel_ids[reliable_weights_all[rel_ids] > 0]
                p_rel, s_rel, n_rel, falls, rel_pos, site_violations = self._retrieve(
                    query[:, slot], rel_ids_slot, reliable_weights_all, memory, sites, slot, 8)
                fallback_total += falls
                selected_site_violations += site_violations
                if no_boundary or boundary_off:
                    bnd_ids, bnd_weights = rel_ids_slot, reliable_weights_all
                    invalid_count = 0
                else:
                    bnd_ids, bnd_weights, invalid_count = self._confusion_weights(
                        memory, support, class_ids, cls, slot)
                    bnd_ids = bnd_ids[bnd_weights > 0]
                    bnd_weights_all = torch.zeros_like(reliable_weights_all)
                    bnd_weights_all[bnd_ids] = bnd_weights[bnd_weights > 0]
                    bnd_weights = bnd_weights_all
                    invalid_confusion_total += invalid_count
                p_bnd, s_bnd, n_bnd, falls, bnd_pos, site_violations = self._retrieve(
                    query[:, slot], bnd_ids, bnd_weights, memory, sites, slot, 8)
                fallback_total += falls
                selected_site_violations += site_violations
                for role, ids_role, positions in ((0, rel_ids_slot, rel_pos), (1, bnd_ids, bnd_pos)):
                    supports[:, cls, slot, role] = positions.shape[1] if positions is not None else 0
                    coverage[:, cls, slot, role] = float(ids_role.numel() > 0)
                class_rel.append(p_rel)
                class_bnd.append(p_bnd)
                class_rel_sim.append((query[:, slot] * p_rel).sum(-1))
                class_bnd_sim.append((query[:, slot] * p_bnd).sum(-1))
            prototypes_rel.append(torch.stack(class_rel, dim=1))
            prototypes_bnd.append(torch.stack(class_bnd, dim=1))
            rel_similarity.append(torch.stack(class_rel_sim, dim=-1))
            bnd_similarity.append(torch.stack(class_bnd_sim, dim=-1))

        p_rel = torch.stack(prototypes_rel, dim=1)
        p_bnd = torch.stack(prototypes_bnd, dim=1)
        e_rel = torch.stack(rel_similarity, dim=1)
        e_bnd = torch.stack(bnd_similarity, dim=1)
        query_key = self.query_projection(query)
        rel_key = self.prototype_projection(p_rel) + self.reliable_text_projection(semantic)[None]
        bnd_key = self.prototype_projection(p_bnd) + self.boundary_text_projection(semantic)[None]
        query_key = query_key[:, None, :, :]
        attention_logits = torch.stack(((query_key * rel_key).sum(-1),
                                        (query_key * bnd_key).sum(-1)), dim=-1) / math.sqrt(36)
        attention = torch.softmax(attention_logits, dim=-1)
        evidence = attention[..., 0] * e_rel + attention[..., 1] * e_bnd
        difference = evidence[:, 0] - evidence[:, 1]
        reliable_difference = e_rel[:, 0] - e_rel[:, 1]
        boundary_difference = e_bnd[:, 0] - e_bnd[:, 1]
        gate_features = torch.cat((cam144,
                                   reliable_difference,
                                   boundary_difference,
                                   difference), dim=-1)
        modality_weights = torch.softmax(self.modality_gate(gate_features), dim=-1)
        d = (modality_weights * difference).sum(-1)
        disagreement = difference - d[:, None]
        attention_features = attention.flatten(1)
        support_features = coverage.flatten(1)
        gate_input = torch.cat((base_score.detach()[:, None], base_score.detach().abs()[:, None],
                                reliable_difference, boundary_difference, difference, attention_features,
                                support_features, disagreement), dim=-1)
        learned_gate = torch.sigmoid(self.residual_gate(gate_input).squeeze(-1) - 2.0)
        margin_gate = torch.sigmoid((1.5 - base_score.detach().abs()) / 0.5)
        coverage_factor = coverage.mean(dim=(1, 2, 3))
        gate = learned_gate * margin_gate * coverage_factor
        delta = 0.8 * torch.tanh(self.alpha) * gate * torch.tanh(d / 0.20)
        if residual_off:
            delta = torch.zeros_like(delta)
        final_score = base_score.detach() + delta
        return final_score, {
            "delta": delta,
            "gate": gate,
            "learned_gate": learned_gate,
            "margin_gate": margin_gate,
            "coverage_factor": coverage_factor,
            "evidence": evidence,
            "reliable_similarity": e_rel,
            "boundary_similarity": e_bnd,
            "reliable_difference": reliable_difference,
            "boundary_difference": boundary_difference,
            "slot_difference": difference,
            "modality_weights": modality_weights,
            "attention": attention,
            "support_topk": supports,
            "support_coverage": coverage,
            "final_boundary_evidence": d,
            "retrieval_fallbacks": fallback_total,
            "invalid_confusion_candidates": invalid_confusion_total,
            "site_cap_violations": selected_site_violations,
        }


def branch_losses(scores: torch.Tensor, diagnostics: Dict[str, torch.Tensor], base_score: torch.Tensor,
                  labels: torch.Tensor):
    y = (labels == 0).to(scores.dtype)
    margin_weight = 1.0 + torch.exp(-base_score.detach().abs())
    correction = (F.binary_cross_entropy_with_logits(scores, y, reduction="none") * margin_weight).mean()
    direction = torch.where(labels == 0, 1.0, -1.0).to(scores.dtype)
    boundary = F.softplus(0.15 - direction * diagnostics["final_boundary_evidence"]).mean()
    pred = torch.where(base_score.detach() >= 0, 0, 1)
    keep = ((pred == labels) & (base_score.detach().abs() > 1.5)).to(scores.dtype)
    keep_loss = (keep * diagnostics["delta"].square()).mean()
    delta_loss = diagnostics["delta"].square().mean()
    budget_loss = F.relu(diagnostics["gate"].mean() - 0.25).square()
    total = correction + 0.03 * boundary + 0.10 * keep_loss + 0.02 * delta_loss + 0.05 * budget_loss
    return total, {
        "correction": correction,
        "boundary": boundary,
        "keep": keep_loss,
        "delta": delta_loss,
        "budget": budget_loss,
        "mean_margin_weight": margin_weight.mean(),
        "keep_count": keep.sum(),
    }


def final_log_probabilities(base_logp: torch.Tensor, base_score: torch.Tensor,
                            final_score: torch.Tensor, module: BCSPC):
    if float(module.alpha.detach()) == 0.0:
        return base_logp
    values = torch.stack((final_score * 0.5, -final_score * 0.5), dim=-1)
    return F.log_softmax(values, dim=-1)

"""Semantic-conditioned prototype evidence and persistent error memory."""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def _unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=1e-8)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def deterministic_centers(vectors: torch.Tensor, count: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = _unit(vectors.detach())
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return vectors.new_zeros((count, vectors.shape[-1])), torch.zeros(count, dtype=torch.long, device=vectors.device)
    n = vectors.shape[0]
    k = min(count, n)
    centers = [vectors[0]]
    nearest = vectors @ centers[0]
    for _ in range(1, k):
        index = int(torch.argmin(nearest).item())
        centers.append(vectors[index])
        nearest = torch.maximum(nearest, vectors @ vectors[index])
    centers_t = torch.stack(centers)
    for _ in range(iterations):
        assignments = (vectors @ centers_t.T).argmax(dim=1)
        updated = []
        for slot in range(k):
            members = vectors[assignments == slot]
            updated.append(_unit(members.mean(0)) if members.numel() else centers_t[slot])
        new_centers = torch.stack(updated)
        if torch.equal(new_centers, centers_t):
            break
        centers_t = new_centers
    assignments = (vectors @ centers_t.T).argmax(dim=1)
    padded = vectors.new_zeros((count, vectors.shape[-1]))
    counts = torch.zeros(count, dtype=torch.long, device=vectors.device)
    padded[:k] = centers_t
    counts[:k] = torch.bincount(assignments, minlength=k)
    return padded, counts


class SC_PERM(nn.Module):
    def __init__(self, text_embeddings: torch.Tensor, cfg: dict, text_source: str):
        super().__init__()
        if text_embeddings.shape != (2, 8, 1024):
            raise ValueError(f"expected text embedding bank [2,8,1024], got {tuple(text_embeddings.shape)}")
        self.cfg = cfg
        self.text_source = text_source
        self.classes = 2
        self.modalities = 4
        self.feature_dim = 36
        self.k_class = int(cfg["class_prototypes_per_class_modality"])
        self.k_hard = int(cfg["hard_prototypes_per_direction_modality"])
        self.top_k = int(cfg["top_k"])
        self.semantic_temperature = float(cfg["feature_temperature"])
        self.cluster_iterations = int(cfg["cluster_iterations"])

        # One semantic projection and one prototype projection are shared across classes/modalities.
        self.text_projection = nn.Linear(1024, int(cfg["semantic_projection_dim"]), bias=False)
        self.feature_projection = nn.Linear(36, int(cfg["semantic_projection_dim"]), bias=False)
        self.semantic_beta_raw = nn.Parameter(torch.tensor(_logit(cfg["semantic_beta_init"] / cfg["semantic_beta_max"])))
        self.evidence_lambda_raw = nn.Parameter(torch.tensor(_logit(cfg["evidence_lambda_init"] / cfg["evidence_lambda_max"])))
        self.register_buffer("text_embeddings", text_embeddings.detach().float())
        self.register_buffer("class_features", torch.zeros(871, 4, 36))
        self.register_buffer("class_labels", torch.full((871,), -1, dtype=torch.long))
        self.register_buffer("class_valid", torch.zeros(871, dtype=torch.bool))
        self.register_buffer("class_assignment", torch.full((4, 2, 871), -1, dtype=torch.long))
        self.register_buffer("class_sums", torch.zeros(4, 2, self.k_class, 36))
        self.register_buffer("class_centers", torch.zeros(4, 2, self.k_class, 36))
        self.register_buffer("class_counts", torch.zeros(4, 2, self.k_class, dtype=torch.long))
        self.register_buffer("class_usage", torch.zeros(4, 2, self.k_class, dtype=torch.long))
        self.register_buffer("class_queries", torch.zeros(4, 2, dtype=torch.long))

        pool = int(cfg["hard_patient_pool_capacity_per_direction"])
        self.pool_size = pool
        self.register_buffer("hard_features", torch.zeros(2, pool, 4, 36))
        self.register_buffer("hard_subject_ids", torch.full((2, pool), -1, dtype=torch.long))
        self.register_buffer("hard_first_error", torch.full((2, pool), -1, dtype=torch.long))
        self.register_buffer("hard_last_error", torch.full((2, pool), -1, dtype=torch.long))
        self.register_buffer("hard_frequency", torch.zeros(2, pool, dtype=torch.long))
        self.register_buffer("hard_votes", torch.zeros(2, pool, dtype=torch.long))
        self.register_buffer("hard_confidence", torch.zeros(2, pool))
        self.register_buffer("hard_novelty", torch.zeros(2, pool))
        self.register_buffer("hard_assignment", torch.full((2, 4, pool), -1, dtype=torch.long))
        self.register_buffer("hard_sums", torch.zeros(2, 4, self.k_hard, 36))
        self.register_buffer("hard_centers", torch.zeros(2, 4, self.k_hard, 36))
        self.register_buffer("hard_counts", torch.zeros(2, 4, self.k_hard, dtype=torch.long))
        self.register_buffer("hard_usage", torch.zeros(2, 4, self.k_hard, dtype=torch.long))
        self.register_buffer("hard_queries", torch.zeros(2, 4, dtype=torch.long))
        self.register_buffer("epoch_updates", torch.zeros(1, dtype=torch.long))

    def _class_semantics(self, class_order: tuple[int, int] = (0, 1), source_index: int = 1) -> torch.Tensor:
        # Text prompts are stored in published order: TD 4, then ASD 4.
        prompt_bank = self.text_embeddings[source_index].reshape(8, 1024)
        if source_index == 1:
            shared_measurement_context = prompt_bank[[3, 7]].mean(0)
            asd_anchor = prompt_bank[4:7].mean(0) + shared_measurement_context
            td_anchor = prompt_bank[0:3].mean(0) + shared_measurement_context
            raw = torch.stack((asd_anchor, td_anchor))
        else:
            raw = torch.stack((prompt_bank[4:8].mean(0), prompt_bank[0:4].mean(0)))
        projected = _unit(self.text_projection(raw))
        return projected[torch.as_tensor(class_order, device=projected.device)]

    def _semantic_relations(self, centers: torch.Tensor, class_order: tuple[int, int], source_index: int) -> torch.Tensor:
        semantics = self._class_semantics(class_order, source_index)
        projected_centers = _unit(self.feature_projection(centers))
        relation = torch.einsum("mckd,cd->mck", projected_centers, semantics)
        # Only CORR is a disease-specific semantic measurement. Other modality text
        # conditioning is attenuated to avoid calling quality/context a biomarker.
        modality_scale = relation.new_tensor((0.15, 0.10, 0.10, 1.0))
        beta = self.cfg["semantic_beta_max"] * torch.sigmoid(self.semantic_beta_raw)
        return beta * modality_scale[:, None, None] * relation

    @torch.no_grad()
    def update_class_memory(self, all_features: torch.Tensor, labels: torch.Tensor,
                            allowed_ids: torch.Tensor) -> None:
        ids = allowed_ids.to(device=self.class_features.device, dtype=torch.long).unique(sorted=True)
        if ids.numel() == 0:
            raise RuntimeError("class prototype bank cannot be initialized from an empty train fold")
        features = all_features.detach().float().reshape(871, 4, 36)
        current = _unit(features[ids])
        same = self.class_valid[ids] & (self.class_labels[ids] == labels[ids])
        previous = self.class_features[ids]
        momentum = float(self.cfg["prototype_ema"])
        blended = _unit(momentum * previous + (1.0 - momentum) * current)
        updated = torch.where(same[:, None, None], blended, current)
        self.class_features.zero_(); self.class_labels.fill_(-1); self.class_valid.zero_()
        self.class_assignment.fill_(-1); self.class_sums.zero_(); self.class_centers.zero_(); self.class_counts.zero_()
        self.class_features[ids] = updated
        self.class_labels[ids] = labels[ids]
        self.class_valid[ids] = True
        for modality in range(4):
            for cls in (0, 1):
                member_ids = ids[labels[ids] == cls]
                vectors = self.class_features[member_ids, modality]
                centers, counts = deterministic_centers(vectors, self.k_class, self.cluster_iterations)
                active = torch.nonzero(counts > 0, as_tuple=False).flatten()
                assignments = (active[(vectors @ centers[active].T).argmax(dim=1)] if active.numel()
                               else torch.full((vectors.shape[0],), -1, dtype=torch.long, device=vectors.device))
                self.class_assignment[modality, cls, member_ids] = assignments
                self.class_centers[modality, cls] = centers
                for slot in range(self.k_class):
                    selected = assignments == slot
                    if selected.any():
                        self.class_sums[modality, cls, slot] = vectors[selected].sum(0)
                        self.class_counts[modality, cls, slot] = selected.sum()
                    else:
                        self.class_centers[modality, cls, slot].zero_()
        self.epoch_updates += 1

    @torch.no_grad()
    def update_hard_memory(self, all_features: torch.Tensor, labels: torch.Tensor,
                           vote_counts: torch.Tensor, prediction_confidence: torch.Tensor,
                           allowed_ids: torch.Tensor, epoch: int) -> dict:
        query_ids = allowed_ids.to(device=self.hard_features.device, dtype=torch.long).unique(sorted=True)
        y = labels.to(device=query_ids.device, dtype=torch.long)[query_ids]
        votes = vote_counts.to(device=query_ids.device, dtype=torch.long)[query_ids]
        confidence = prediction_confidence.to(device=query_ids.device, dtype=torch.float32)[query_ids]
        stable = votes >= int(self.cfg["hard_error_min_votes"])
        fp_ids = query_ids[stable & (y == 1)]
        fn_ids = query_ids[stable & (y == 0)]
        incoming = (fp_ids, fn_ids)
        features = _unit(all_features.detach().float().reshape(871, 4, 36))
        max_age = int(self.cfg["hard_memory_max_age_epochs"])
        ema = float(self.cfg["hard_memory_ema"])
        retained_counts = []
        for direction in range(2):
            live = self.hard_subject_ids[direction] >= 0
            stale = live & ((epoch - self.hard_last_error[direction]) > max_age)
            self.hard_subject_ids[direction, stale] = -1
            self.hard_first_error[direction, stale] = -1
            self.hard_last_error[direction, stale] = -1
            self.hard_frequency[direction, stale] = 0
            self.hard_votes[direction, stale] = 0
            self.hard_confidence[direction, stale] = 0
            self.hard_novelty[direction, stale] = 0
            self.hard_features[direction, stale] = 0
            incoming_ids = incoming[direction].unique(sorted=True)
            for subject in incoming_ids.tolist():
                query_position = int(torch.searchsorted(
                    query_ids, torch.tensor(subject, device=query_ids.device)
                ).item())
                vote = int(votes[query_position].item())
                latest_confidence = float(confidence[query_position].item())
                slots = torch.nonzero(self.hard_subject_ids[direction] == subject, as_tuple=False).flatten()
                if slots.numel():
                    slot = int(slots[0].item())
                    self.hard_features[direction, slot] = _unit(
                        ema * self.hard_features[direction, slot] + (1.0 - ema) * features[subject]
                    )
                    self.hard_frequency[direction, slot] += 1
                    self.hard_votes[direction, slot] = min(
                        max(int(self.hard_votes[direction, slot].item()), vote), 3
                    )
                    self.hard_last_error[direction, slot] = epoch
                    all_slots = torch.arange(self.pool_size, device=query_ids.device)
                    peers = torch.nonzero((self.hard_subject_ids[direction] >= 0) & (all_slots != slot),
                                          as_tuple=False).flatten()
                    if peers.numel():
                        peer_center = _unit(self.hard_features[direction, peers].mean(0))
                        novelty = 1.0 - (self.hard_features[direction, slot] * peer_center).sum(-1).mean()
                    else:
                        novelty = self.hard_novelty.new_tensor(1.0)
                    self.hard_novelty[direction, slot] = novelty
                    self.hard_confidence[direction, slot] = latest_confidence
                else:
                    live_slots = torch.nonzero(self.hard_subject_ids[direction] >= 0, as_tuple=False).flatten()
                    if live_slots.numel():
                        center = _unit(self.hard_features[direction, live_slots].mean(0))
                        new_novelty = 1.0 - (features[subject] * center).sum(-1).mean()
                    else:
                        new_novelty = self.hard_novelty.new_tensor(1.0)
                    free = torch.nonzero(self.hard_subject_ids[direction] < 0, as_tuple=False).flatten()
                    if free.numel():
                        slot = int(free[0].item())
                    else:
                        existing = self.hard_features[direction, live_slots]
                        center = _unit(existing.mean(0))
                        novelty = 1.0 - (existing * center.unsqueeze(0)).sum(-1).mean(-1)
                        age = (epoch - self.hard_last_error[direction, live_slots]).float()
                        score = self.hard_votes[direction, live_slots].float() + 0.5 * novelty - 0.02 * age
                        # Lowest support/novelty and oldest recent error is evicted first.
                        worst = int(torch.argmin(score).item())
                        new_score = min(float(vote), 3.0) + 0.5 * float(new_novelty.item())
                        if new_score <= float(score[worst].item()):
                            continue
                        slot = int(live_slots[worst].item())
                    self.hard_subject_ids[direction, slot] = subject
                    self.hard_features[direction, slot] = features[subject]
                    self.hard_first_error[direction, slot] = epoch
                    self.hard_last_error[direction, slot] = epoch
                    self.hard_frequency[direction, slot] = 1
                    self.hard_votes[direction, slot] = min(vote, 3)
                    self.hard_confidence[direction, slot] = latest_confidence
                    self.hard_novelty[direction, slot] = new_novelty
            retained_counts.append(int((self.hard_subject_ids[direction] >= 0).sum().item()))
        self._recluster_hard()
        return {"fp_ids": fp_ids, "fn_ids": fn_ids, "pool_fp": retained_counts[0], "pool_fn": retained_counts[1]}

    @torch.no_grad()
    def _recluster_hard(self) -> None:
        self.hard_assignment.fill_(-1); self.hard_sums.zero_(); self.hard_centers.zero_(); self.hard_counts.zero_()
        for direction in range(2):
            slots = torch.nonzero(self.hard_subject_ids[direction] >= 0, as_tuple=False).flatten()
            if slots.numel() == 0:
                continue
            for modality in range(4):
                vectors = self.hard_features[direction, slots, modality]
                centers, counts = deterministic_centers(vectors, self.k_hard, self.cluster_iterations)
                active = torch.nonzero(counts > 0, as_tuple=False).flatten()
                assignments = (active[(vectors @ centers[active].T).argmax(dim=1)] if active.numel()
                               else torch.full((vectors.shape[0],), -1, dtype=torch.long, device=vectors.device))
                self.hard_centers[direction, modality] = centers
                self.hard_assignment[direction, modality, slots] = assignments
                for proto in range(self.k_hard):
                    selected = assignments == proto
                    if selected.any():
                        self.hard_sums[direction, modality, proto] = vectors[selected].sum(0)
                        self.hard_counts[direction, modality, proto] = selected.sum()
                    else:
                        self.hard_centers[direction, modality, proto].zero_()

    def _leave_one_out_centers(self, subject_ids: torch.Tensor, *, hard_direction: int | None = None,
                               modality: int, disease_class: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        subject_ids = subject_ids.to(dtype=torch.long, device=self.class_features.device)
        if hard_direction is None:
            sums = self.class_sums[modality, disease_class].unsqueeze(0).expand(subject_ids.numel(), -1, -1).clone()
            counts = self.class_counts[modality, disease_class].unsqueeze(0).expand(subject_ids.numel(), -1).clone()
            member = self.class_valid[subject_ids] & (self.class_labels[subject_ids] == disease_class)
            assignments = self.class_assignment[modality, disease_class, subject_ids]
            rows = torch.nonzero(member & (assignments >= 0), as_tuple=False).flatten()
            slots = assignments[rows]
            sums[rows, slots] -= self.class_features[subject_ids[rows], modality]
            counts[rows, slots] -= 1
        else:
            sums = self.hard_sums[hard_direction, modality].unsqueeze(0).expand(subject_ids.numel(), -1, -1).clone()
            counts = self.hard_counts[hard_direction, modality].unsqueeze(0).expand(subject_ids.numel(), -1).clone()
            where = self.hard_subject_ids[hard_direction]
            matches = subject_ids.unsqueeze(1) == where.unsqueeze(0)
            member = matches.any(dim=1)
            pool_slots = matches.to(torch.long).argmax(dim=1)
            rows = torch.nonzero(member, as_tuple=False).flatten()
            prototypes = self.hard_assignment[hard_direction, modality, pool_slots[rows]]
            valid_rows = prototypes >= 0
            rows, prototypes = rows[valid_rows], prototypes[valid_rows]
            selected_pool_slots = pool_slots[rows]
            sums[rows, prototypes] -= self.hard_features[hard_direction, selected_pool_slots, modality]
            counts[rows, prototypes] -= 1
        valid = counts > 0
        centers = _unit(sums / counts.clamp_min(1).unsqueeze(-1))
        return centers, valid

    def _retrieve(self, queries: torch.Tensor, centers: torch.Tensor, valid: torch.Tensor,
                  semantic_relation: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        similarities = torch.einsum("nd,nkd->nk", _unit(queries), centers)
        similarities = similarities.masked_fill(~valid, -1e4)
        if semantic_relation is not None:
            similarities = similarities + semantic_relation
        k = min(self.top_k, similarities.shape[-1])
        values, indices = similarities.topk(k, dim=-1)
        selected_valid = valid.gather(1, indices)
        weights = F.softmax(values / self.semantic_temperature, dim=-1) * selected_valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        selected_sim = torch.einsum("nk,nk->n", weights, values * selected_valid.float())
        selected_sim = torch.where(selected_valid.any(-1), selected_sim, torch.zeros_like(selected_sim))
        return selected_sim, weights, indices

    def compute_evidence(self, h: torch.Tensor, query_ids: torch.Tensor, *, hard_enabled: bool,
                         prototype_off: bool = False, hard_off: bool = False,
                         semantic_swap: bool = False, text_source: str | None = None) -> dict:
        n = h.shape[0]
        features = h.reshape(n, 4, 36)
        ids = query_ids.to(device=h.device, dtype=torch.long)
        class_order = (1, 0) if semantic_swap else (0, 1)
        class_centers_by_class, class_valid_by_class = [], []
        source_index = (0 if text_source == "original" else 1) if text_source is not None else (0 if self.text_source == "original" else 1)
        sem_rel = self._semantic_relations(self.class_centers, class_order, source_index)
        class_scores = h.new_zeros((n, 2))
        typical_gate_scores = h.new_zeros((n, 2))
        class_relations = h.new_zeros((n, 4, 2, self.k_class))
        class_weights = h.new_zeros((n, 4, 2, min(self.top_k, self.k_class)))
        class_indices = torch.zeros((n, 4, 2, min(self.top_k, self.k_class)), dtype=torch.long, device=h.device)
        for modality in range(4):
            for cls in (0, 1):
                centers, valid = self._leave_one_out_centers(ids, modality=modality, disease_class=cls)
                relation = sem_rel[modality, cls].unsqueeze(0).expand(n, -1)
                scores, weights, indices = self._retrieve(features[:, modality], centers, valid, relation)
                typical_gate_scores[:, cls] += scores * (0.15, 0.10, 0.10, 1.0)[modality]
                if prototype_off:
                    scores = torch.zeros_like(scores)
                class_scores[:, cls] += scores * (0.15, 0.10, 0.10, 1.0)[modality]
                class_relations[:, modality, cls] = relation
                class_weights[:, modality, cls, :weights.shape[-1]] = weights
                class_indices[:, modality, cls, :indices.shape[-1]] = indices
                if not prototype_off:
                    with torch.no_grad():
                        if indices.numel():
                            self.class_usage[modality, cls].scatter_add_(
                                0, indices.reshape(-1), (weights.detach().reshape(-1) > 0).long()
                            )
                        self.class_queries[modality, cls] += n

        hard_scores = h.new_zeros((n, 2))
        hard_similarity = h.new_zeros((n, 2, 4))
        hard_weights = h.new_zeros((n, 2, 4, min(self.top_k, self.k_hard)))
        if hard_enabled and not hard_off:
            hard_class_order = (0, 1) if semantic_swap else (1, 0)  # FP is TD-anchored; FN is ASD-anchored.
            hard_sem = self._semantic_relations(
                self.hard_centers.permute(1, 0, 2, 3), class_order=hard_class_order, source_index=source_index
            )
            for modality in range(4):
                for direction in (0, 1):
                    centers, valid = self._leave_one_out_centers(ids, hard_direction=direction, modality=modality)
                    semantic_relation = hard_sem[modality, direction].unsqueeze(0).expand(n, -1)
                    sim, weights, indices = self._retrieve(features[:, modality], centers, valid, semantic_relation)
                    target_class = 1 if direction == 0 else 0
                    competing_class = 1 - target_class
                    competing_typical = typical_gate_scores[:, competing_class] / 1.35
                    contrast = sim - competing_typical - float(self.cfg["hard_contrast_margin"])
                    gate = torch.sigmoid(contrast / 0.2) * valid.any(-1).float()
                    value = (0.5 * torch.tanh(2.0 * F.relu(contrast))) * gate
                    hard_scores[:, target_class] += value * (0.15, 0.10, 0.10, 1.0)[modality]
                    hard_similarity[:, direction, modality] = sim
                    hard_weights[:, direction, modality, :weights.shape[-1]] = weights
                    with torch.no_grad():
                        if indices.numel():
                            self.hard_usage[direction, modality].scatter_add_(
                                0, indices.reshape(-1), (weights.detach().reshape(-1) > 0).long()
                            )
                        self.hard_queries[direction, modality] += n
            cap = float(self.cfg["hard_evidence_max"])
            hard_scores = hard_scores.clamp(min=0.0, max=cap)
        evidence = class_scores + hard_scores
        centered = (evidence[:, 0] - evidence[:, 1]).clamp(-float(self.cfg["evidence_center_clip"]),
                                                                  float(self.cfg["evidence_center_clip"]))
        scale = float(self.cfg["evidence_lambda_max"]) * torch.sigmoid(self.evidence_lambda_raw)
        correction = torch.stack((0.5 * scale * centered, -0.5 * scale * centered), dim=1)
        return {
            "class_scores": class_scores, "hard_scores": hard_scores,
            "hard_similarity": hard_similarity, "evidence": evidence,
            "correction": correction, "lambda": scale,
            "semantic_beta": float(self.cfg["semantic_beta_max"]) * torch.sigmoid(self.semantic_beta_raw),
            "class_relations": class_relations, "class_weights": class_weights,
            "class_indices": class_indices, "hard_weights": hard_weights,
        }

    def prototype_aux_loss(self, scores: torch.Tensor, labels: torch.Tensor, train_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(scores[train_ids], labels[train_ids])

    def hard_ranking_loss(self, info: dict, labels: torch.Tensor, fp_ids: torch.Tensor,
                          fn_ids: torch.Tensor) -> torch.Tensor:
        terms = []
        margin = float(self.cfg["hard_ranking_margin"])
        if fp_ids.numel():
            # FP patients (TD mistaken for ASD) must receive TD-directed hard evidence.
            terms.append(F.relu(margin - (info["hard_scores"][fp_ids, 1] - info["hard_scores"][fp_ids, 0])).mean())
        if fn_ids.numel():
            # FN patients (ASD mistaken for TD) must receive ASD-directed hard evidence.
            terms.append(F.relu(margin - (info["hard_scores"][fn_ids, 0] - info["hard_scores"][fn_ids, 1])).mean())
        return torch.stack(terms).mean() if terms else info["hard_scores"].sum() * 0.0

    @torch.no_grad()
    def diagnostics(self) -> dict:
        def collapse(centers: torch.Tensor, counts: torch.Tensor) -> dict:
            similarities = []
            collapsed = 0
            pairs = 0
            for index in range(centers.shape[0]):
                for category in range(centers.shape[1]):
                    active = torch.nonzero(counts[index, category] > 0, as_tuple=False).flatten()
                    if active.numel() < 2:
                        continue
                    points = _unit(centers[index, category, active])
                    matrix = points @ points.T
                    upper = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
                    values = matrix[upper]
                    similarities.extend(values.detach().cpu().tolist())
                    collapsed += int((values > 0.98).sum().item())
                    pairs += int(values.numel())
            return {
                "active_pair_count": pairs,
                "max_cosine": max(similarities) if similarities else 0.0,
                "fraction_cosine_above_0p98": collapsed / max(pairs, 1),
            }

        class_usage_denominator = (self.class_queries.unsqueeze(-1) * self.top_k).clamp_min(1)
        hard_usage_denominator = (self.hard_queries.unsqueeze(-1) * self.top_k).clamp_min(1)
        return {
            "class_active": int((self.class_counts > 0).sum().item()),
            "class_slots": int(self.class_counts.numel()),
            "class_occupancy_counts": self.class_counts.detach().cpu().tolist(),
            "class_usage_rate": (self.class_usage.float() / class_usage_denominator).detach().cpu().tolist(),
            "class_collapse": collapse(self.class_centers, self.class_counts),
            "hard_active": int((self.hard_counts > 0).sum().item()),
            "hard_slots": int(self.hard_counts.numel()),
            "hard_occupancy_counts": self.hard_counts.detach().cpu().tolist(),
            "hard_usage_rate": (self.hard_usage.float() / hard_usage_denominator).detach().cpu().tolist(),
            "hard_collapse": collapse(self.hard_centers, self.hard_counts),
            "hard_patient_pool": self.hard_subject_ids.detach().cpu().tolist(),
            "hard_first_error": self.hard_first_error.detach().cpu().tolist(),
            "hard_last_error": self.hard_last_error.detach().cpu().tolist(),
            "hard_frequency": self.hard_frequency.detach().cpu().tolist(),
            "hard_votes": self.hard_votes.detach().cpu().tolist(),
            "hard_confidence": self.hard_confidence.detach().cpu().tolist(),
            "hard_novelty": self.hard_novelty.detach().cpu().tolist(),
            "class_usage": self.class_usage.detach().cpu().tolist(),
            "hard_usage": self.hard_usage.detach().cpu().tolist(),
            "class_centers": self.class_centers.detach().cpu().tolist(),
            "hard_centers": self.hard_centers.detach().cpu().tolist(),
            "semantic_beta": float(self.cfg["semantic_beta_max"] * torch.sigmoid(self.semantic_beta_raw).item()),
            "evidence_lambda": float(self.cfg["evidence_lambda_max"] * torch.sigmoid(self.evidence_lambda_raw).item()),
        }


class TemporalErrorTracker(nn.Module):
    def __init__(self, window: int = 3):
        super().__init__()
        self.window = int(window)
        self.register_buffer("prediction_history", torch.full((window, 871), -1, dtype=torch.long))
        self.register_buffer("history_position", torch.zeros(1, dtype=torch.long))
        self.register_buffer("vote_counts", torch.zeros(871, dtype=torch.long))
        self.register_buffer("stable_fp_mask", torch.zeros(871, dtype=torch.bool))
        self.register_buffer("stable_fn_mask", torch.zeros(871, dtype=torch.bool))
        self.register_buffer("seen_stable_fp_mask", torch.zeros(871, dtype=torch.bool))
        self.register_buffer("seen_stable_fn_mask", torch.zeros(871, dtype=torch.bool))
        self.register_buffer("stable_previous", torch.zeros(2, 871, dtype=torch.bool))
        self.register_buffer("stable_age", torch.zeros(2, 871, dtype=torch.long))

    @torch.no_grad()
    def update(self, logp: torch.Tensor, labels: torch.Tensor, allowed_ids: torch.Tensor,
               min_votes: int = 2) -> dict:
        ids = allowed_ids.to(device=logp.device, dtype=torch.long).unique(sorted=True)
        probs = logp.detach().argmax(dim=1)
        pos = int(self.history_position.item() % self.window)
        self.prediction_history[pos].fill_(-1)
        self.prediction_history[pos, ids] = probs[ids]
        self.history_position += 1
        history = self.prediction_history[:, ids]
        current_labels = labels[ids]
        fp_votes = ((history == 0) & (current_labels.unsqueeze(0) == 1)).sum(0)
        fn_votes = ((history == 1) & (current_labels.unsqueeze(0) == 0)).sum(0)
        current_votes = torch.maximum(fp_votes, fn_votes).clamp_max(self.window)
        self.vote_counts.zero_()
        self.vote_counts[ids] = current_votes
        fp = ids[(current_labels == 1) & (fp_votes >= min_votes)]
        fn = ids[(current_labels == 0) & (fn_votes >= min_votes)]
        self.stable_fp_mask.zero_(); self.stable_fn_mask.zero_()
        self.stable_fp_mask[fp] = True; self.stable_fn_mask[fn] = True
        self.seen_stable_fp_mask[fp] = True; self.seen_stable_fn_mask[fn] = True
        current = torch.zeros_like(self.stable_previous, device=logp.device)
        current[0, fp] = True
        current[1, fn] = True
        prior = self.stable_previous.to(logp.device)
        prior_age = self.stable_age.to(logp.device)
        age = torch.where(current, torch.where(prior, prior_age + 1, torch.ones_like(prior_age)),
                          torch.zeros_like(prior_age))
        self.stable_previous.copy_(current.to(self.stable_previous.device))
        self.stable_age.copy_(age.to(self.stable_age.device))
        asd_probability = logp.detach().float().exp()[ids, 0]
        prediction_confidence = logp.detach().float().exp().max(dim=1).values
        fp_mask = (current_labels == 1) & (fp_votes >= min_votes)
        fn_mask = (current_labels == 0) & (fn_votes >= min_votes)
        return {"fp_ids": fp, "fn_ids": fn, "vote_counts": self.vote_counts.clone(),
                "prediction_confidence": prediction_confidence,
                "unique_stable_fp_patients": int(self.seen_stable_fp_mask.sum().item()),
                "unique_stable_fn_patients": int(self.seen_stable_fn_mask.sum().item()),
                "near_fp_ids": ids[(current_labels == 1) & (fp_votes < min_votes)
                                   & (asd_probability >= 0.40) & (asd_probability < 0.50)],
                "near_fn_ids": ids[(current_labels == 0) & (fn_votes < min_votes)
                                   & (asd_probability > 0.50) & (asd_probability <= 0.60)],
                "fp_confidence": float(asd_probability[fp_mask].mean().item()) if fp_mask.any() else 0.0,
                "fn_confidence": float((1.0 - asd_probability[fn_mask]).mean().item()) if fn_mask.any() else 0.0,
                "fp_age_mean": float(age[0, fp].float().mean().item()) if fp.numel() else 0.0,
                "fn_age_mean": float(age[1, fn].float().mean().item()) if fn.numel() else 0.0,
                "observed": int((history >= 0).any(0).sum().item())}

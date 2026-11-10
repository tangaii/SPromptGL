"""Deterministic class and directional-error prototype memory for E77."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def _cluster(features: torch.Tensor, k_requested: int, iterations: int,
             old_centers: torch.Tensor | None = None,
             old_counts: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-order farthest-first initialization followed by deterministic Lloyd updates."""
    values = _unit(features.detach().float()).cpu().numpy().astype(np.float64, copy=False)
    n, dim = values.shape
    if n == 0:
        return (features.new_zeros((k_requested, dim)),
                torch.full((k_requested,), 0, dtype=torch.long, device=features.device),
                torch.empty((0,), dtype=torch.long, device=features.device))
    k = min(int(k_requested), n)
    previous = []
    if old_centers is not None and old_counts is not None:
        old = old_centers.detach().float().cpu().numpy()
        counts = old_counts.detach().cpu().numpy()
        previous = [old[i].astype(np.float64, copy=False) for i in range(min(k, len(counts))) if counts[i] > 0]
    centers = list(previous)
    if not centers:
        centers.append(values[0])
    while len(centers) < k:
        c = np.stack(centers)
        similarity = values @ c.T
        distance = 1.0 - similarity.max(axis=1)
        candidate = int(np.argmax(distance))
        if distance[candidate] <= 1e-12:
            break
        centers.append(values[candidate])
    center_array = np.stack(centers[:k])
    assignments = np.zeros(n, dtype=np.int64)
    active = np.ones(len(center_array), dtype=np.bool_)
    for _ in range(iterations):
        scores = values @ center_array.T
        assignments = scores.argmax(axis=1)
        updated = center_array.copy()
        active[:] = False
        for index in range(len(center_array)):
            members = values[assignments == index]
            if members.shape[0]:
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 1e-12:
                    updated[index] = mean / norm
                    active[index] = True
        if np.allclose(updated, center_array, rtol=0, atol=1e-12):
            center_array = updated
            break
        center_array = updated

    out_centers = features.new_zeros((k_requested, dim))
    out_counts = torch.zeros(k_requested, dtype=torch.long, device=features.device)
    out_assignments = torch.full((n,), -1, dtype=torch.long, device=features.device)
    valid_count = min(k, len(center_array))
    if valid_count:
        out_centers[:valid_count] = torch.as_tensor(center_array[:valid_count], dtype=features.dtype, device=features.device)
        labels = torch.as_tensor(assignments, dtype=torch.long, device=features.device)
        counts = torch.bincount(labels, minlength=valid_count)
        active_tensor = torch.as_tensor(active[:valid_count], dtype=torch.bool, device=features.device)
        counts = counts * active_tensor.to(counts.dtype)
        out_counts[:valid_count] = counts
        out_assignments = torch.where(counts[labels] > 0, labels, torch.full_like(labels, -1))
    return out_centers, out_counts, out_assignments


class TemporalErrorTracker(nn.Module):
    """Keeps eval-mode training predictions only; tests are never read for memory updates."""

    def __init__(self, n_subjects: int = 871, window: int = 3):
        super().__init__()
        self.n_subjects = int(n_subjects)
        self.window = int(window)
        self.register_buffer("pred_history", torch.full((window, n_subjects), -1, dtype=torch.long))
        self.register_buffer("prob_history", torch.full((window, n_subjects), float("nan")))
        self.register_buffer("history_valid", torch.zeros(window, dtype=torch.bool))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("stable_previous", torch.zeros(2, n_subjects, dtype=torch.bool))
        self.register_buffer("stable_age", torch.zeros(2, n_subjects, dtype=torch.long))

    @torch.no_grad()
    def update(self, logp: torch.Tensor, labels: torch.Tensor, allowed_ids: torch.Tensor,
               min_votes: int = 2) -> Dict[str, torch.Tensor | int | float]:
        device = logp.device
        ids = allowed_ids.to(device=device, dtype=torch.long).unique(sorted=True)
        probabilities = logp.detach().float().exp()[:, 0]
        prediction = logp.detach().argmax(dim=1)
        slot = int(self.cursor.item())
        self.pred_history[slot, ids] = prediction[ids].to(self.pred_history.device)
        self.prob_history[slot, ids] = probabilities[ids].to(self.prob_history.device)
        self.history_valid[slot] = True
        self.cursor.fill_((slot + 1) % self.window)
        if int(self.history_valid.sum().item()) < self.window:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"fp_ids": empty, "fn_ids": empty, "near_fp_ids": empty, "near_fn_ids": empty,
                    "fp_age_mean": 0.0, "fn_age_mean": 0.0, "fp_confidence": 0.0,
                    "fn_confidence": 0.0, "stable_window_ready": False}

        hist_pred = self.pred_history[self.history_valid].to(device)
        hist_prob = self.prob_history[self.history_valid].to(device)
        votes_asd = (hist_pred[:, ids] == 0).sum(dim=0)
        votes_td = (hist_pred[:, ids] == 1).sum(dim=0)
        mean_asd = hist_prob[:, ids].mean(dim=0)
        train_labels = labels.to(device=device, dtype=torch.long)[ids]
        current_fp = (train_labels == 1) & (votes_asd >= min_votes)
        current_fn = (train_labels == 0) & (votes_td >= min_votes)
        near_fp = (train_labels == 1) & (votes_asd < min_votes) & (mean_asd >= 0.40) & (mean_asd < 0.50)
        near_fn = (train_labels == 0) & (votes_td < min_votes) & (mean_asd > 0.50) & (mean_asd <= 0.60)

        current = torch.zeros_like(self.stable_previous, device=device)
        current[0, ids] = current_fp
        current[1, ids] = current_fn
        prior = self.stable_previous.to(device)
        prior_age = self.stable_age.to(device)
        ages = torch.where(current, torch.where(prior, prior_age + 1, torch.ones_like(prior_age)),
                           torch.zeros_like(prior_age))
        self.stable_previous.copy_(current.to(self.stable_previous.device))
        self.stable_age.copy_(ages.to(self.stable_age.device))
        fp_ids, fn_ids = ids[current_fp], ids[current_fn]
        return {
            "fp_ids": fp_ids,
            "fn_ids": fn_ids,
            "near_fp_ids": ids[near_fp],
            "near_fn_ids": ids[near_fn],
            "fp_age_mean": float(ages[0, fp_ids].float().mean().item()) if fp_ids.numel() else 0.0,
            "fn_age_mean": float(ages[1, fn_ids].float().mean().item()) if fn_ids.numel() else 0.0,
            "fp_confidence": float(mean_asd[current_fp].mean().item()) if current_fp.any() else 0.0,
            "fn_confidence": float((1.0 - mean_asd[current_fn]).mean().item()) if current_fn.any() else 0.0,
            "stable_window_ready": True,
        }


class ECPSM(nn.Module):
    """Feature-boundary memory that leaves the inherited classifier head unchanged."""

    def __init__(self, text_embeddings: torch.Tensor, n_subjects: int = 871,
                 class_prototypes: int = 4, hard_prototypes: int = 2,
                 top_k: int = 2, temperature: float = 0.2,
                 semantic_temperature: float = 0.2, ema_momentum: float = 0.9,
                 kmeans_iterations: int = 25):
        super().__init__()
        if text_embeddings.shape != (8, 1024):
            raise ValueError(f"expected the audited SPromptGL RN50 text cache [8,1024], got {tuple(text_embeddings.shape)}")
        if n_subjects != 871:
            raise ValueError(f"E77 fold-bound memory expects 871 aligned rows, got {n_subjects}")
        normalized = _unit(text_embeddings.detach().float())
        td_anchor = _unit(normalized[:4].mean(dim=0))
        asd_anchor = _unit(normalized[4:].mean(dim=0))
        self.register_buffer("text_bank", torch.stack((asd_anchor, td_anchor)))
        self.n_subjects = int(n_subjects)
        self.class_prototypes = int(class_prototypes)
        self.hard_prototypes = int(hard_prototypes)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.semantic_temperature = float(semantic_temperature)
        self.ema_momentum = float(ema_momentum)
        self.kmeans_iterations = int(kmeans_iterations)

        self.text_projection = nn.Linear(1024, 9)
        self.prompt_condition = nn.Linear(81, 9, bias=False)
        self.evidence_residual = nn.Linear(91, 36, bias=False)

        n, m, d = self.n_subjects, 4, 36
        kc, kh = self.class_prototypes, self.hard_prototypes
        self.register_buffer("class_features", torch.zeros(n, m, d))
        self.register_buffer("class_labels", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("class_valid", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("class_assignment", torch.full((m, 2, n), -1, dtype=torch.long))
        self.register_buffer("class_sums", torch.zeros(m, 2, kc, d))
        self.register_buffer("class_centers", torch.zeros(m, 2, kc, d))
        self.register_buffer("class_counts", torch.zeros(m, 2, kc, dtype=torch.long))

        self.register_buffer("hard_features", torch.zeros(2, n, m, d))
        self.register_buffer("hard_valid", torch.zeros(2, n, dtype=torch.bool))
        self.register_buffer("hard_assignment", torch.full((2, m, n), -1, dtype=torch.long))
        self.register_buffer("hard_sums", torch.zeros(2, m, kh, d))
        self.register_buffer("hard_centers", torch.zeros(2, m, kh, d))
        self.register_buffer("hard_counts", torch.zeros(2, m, kh, dtype=torch.long))
        self.register_buffer("class_usage", torch.zeros(m, 2, kc, dtype=torch.long))
        self.register_buffer("hard_usage", torch.zeros(2, m, kh, dtype=torch.long))
        self.register_buffer("class_query_count", torch.zeros(m, 2, dtype=torch.long))
        self.register_buffer("hard_query_count", torch.zeros(2, m, dtype=torch.long))

    @torch.no_grad()
    def update_class_memory(self, all_features: torch.Tensor, labels: torch.Tensor,
                            allowed_ids: torch.Tensor) -> None:
        device = self.class_features.device
        ids = allowed_ids.to(device=device, dtype=torch.long).unique(sorted=True)
        labels = labels.to(device=device, dtype=torch.long)
        raw_features = all_features.detach().to(device=device, dtype=torch.float32).reshape(self.n_subjects, 4, 36)
        current = _unit(raw_features[ids])
        previous_valid = self.class_valid[ids]
        same_class = previous_valid & (self.class_labels[ids] == labels[ids])
        previous = self.class_features[ids]
        blended = _unit(self.ema_momentum * previous + (1.0 - self.ema_momentum) * current)
        updated = torch.where(same_class[:, None, None], blended, current)

        prior_centers = self.class_centers.clone()
        prior_counts = self.class_counts.clone()

        self.class_features.zero_()
        self.class_labels.fill_(-1)
        self.class_valid.zero_()
        self.class_assignment.fill_(-1)
        self.class_sums.zero_()
        self.class_centers.zero_()
        self.class_counts.zero_()
        self.class_features[ids] = updated
        self.class_labels[ids] = labels[ids]
        self.class_valid[ids] = True

        for modality in range(4):
            for disease_class in (0, 1):
                member_ids = ids[labels[ids] == disease_class]
                if member_ids.numel() == 0:
                    continue
                vectors = self.class_features[member_ids, modality]
                centers, _, assignment = _cluster(
                    vectors, self.class_prototypes, self.kmeans_iterations,
                    prior_centers[modality, disease_class], prior_counts[modality, disease_class],
                )
                self.class_assignment[modality, disease_class, member_ids] = assignment
                for cluster_id in range(self.class_prototypes):
                    selected = assignment == cluster_id
                    if selected.any():
                        members = vectors[selected]
                        self.class_sums[modality, disease_class, cluster_id] = members.sum(dim=0)
                        self.class_counts[modality, disease_class, cluster_id] = int(selected.sum().item())
                        self.class_centers[modality, disease_class, cluster_id] = centers[cluster_id]

    @torch.no_grad()
    def update_hard_memory(self, all_features: torch.Tensor, fp_ids: torch.Tensor,
                           fn_ids: torch.Tensor, allowed_ids: torch.Tensor) -> None:
        device = self.hard_features.device
        allowed = allowed_ids.to(device=device, dtype=torch.long).unique(sorted=True)
        raw_features = all_features.detach().to(device=device, dtype=torch.float32).reshape(self.n_subjects, 4, 36)
        requested = (fp_ids.to(device=device, dtype=torch.long).unique(sorted=True),
                     fn_ids.to(device=device, dtype=torch.long).unique(sorted=True))
        prior_centers = self.hard_centers.clone()
        prior_counts = self.hard_counts.clone()
        all_allowed = torch.zeros(self.n_subjects, dtype=torch.bool, device=device)
        all_allowed[allowed] = True
        self.hard_assignment.fill_(-1)
        self.hard_sums.zero_()
        self.hard_centers.zero_()
        self.hard_counts.zero_()
        for direction in range(2):
            ids = requested[direction]
            if ids.numel() and not bool(all_allowed[ids].all()):
                raise RuntimeError("hard memory received an ID outside effective train minus explicit test")
            previous_valid = self.hard_valid[direction, ids]
            previous = self.hard_features[direction, ids]
            current = _unit(raw_features[ids])
            blended = _unit(self.ema_momentum * previous + (1.0 - self.ema_momentum) * current)
            updated = torch.where(previous_valid[:, None, None], blended, current)
            self.hard_valid[direction].zero_()
            self.hard_features[direction].zero_()
            if ids.numel():
                self.hard_valid[direction, ids] = True
                self.hard_features[direction, ids] = updated
            for modality in range(4):
                if ids.numel() == 0:
                    continue
                vectors = self.hard_features[direction, ids, modality]
                centers, _, assignment = _cluster(
                    vectors, self.hard_prototypes, self.kmeans_iterations,
                    prior_centers[direction, modality], prior_counts[direction, modality],
                )
                self.hard_assignment[direction, modality, ids] = assignment
                for cluster_id in range(self.hard_prototypes):
                    selected = assignment == cluster_id
                    if selected.any():
                        members = vectors[selected]
                        self.hard_sums[direction, modality, cluster_id] = members.sum(dim=0)
                        self.hard_counts[direction, modality, cluster_id] = int(selected.sum().item())
                        self.hard_centers[direction, modality, cluster_id] = centers[cluster_id]

    def _retrieve(self, query: torch.Tensor, query_ids: torch.Tensor, member_features: torch.Tensor,
                  member_valid: torch.Tensor, assignment: torch.Tensor, sums: torch.Tensor,
                  counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, dim = query.shape
        k = counts.shape[0]
        ids = query_ids.to(device=query.device, dtype=torch.long)
        q_valid = member_valid[ids] & (assignment[ids] >= 0)
        q_assignment = assignment[ids].clamp_min(0)
        self_feature = member_features[ids]
        onehot = F.one_hot(q_assignment, num_classes=k).to(query.dtype) * q_valid[:, None].to(query.dtype)
        adjusted_sums = sums[None, :, :].expand(batch, -1, -1) - onehot[:, :, None] * self_feature[:, None, :]
        adjusted_counts = counts[None, :].expand(batch, -1) - onehot.to(counts.dtype)
        centers = _unit(adjusted_sums / adjusted_counts.clamp_min(1).to(query.dtype)[:, :, None])
        valid_centers = adjusted_counts > 0
        scores = (_unit(query)[:, None, :] * centers).sum(dim=-1)
        scores = scores.masked_fill(~valid_centers, -1e9)
        top_n = min(self.top_k, k)
        top_scores, top_indices = torch.topk(scores, k=top_n, dim=-1, largest=True, sorted=True)
        top_valid = torch.gather(valid_centers, 1, top_indices)
        weights = torch.softmax(top_scores / self.temperature, dim=-1) * top_valid.to(query.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected_centers = torch.gather(centers, 1, top_indices[:, :, None].expand(-1, -1, dim))
        retrieved = (weights[:, :, None] * selected_centers).sum(dim=1)
        return retrieved, top_indices, top_valid, weights

    def _record_retrieval_usage(self, info: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for modality in range(4):
                for disease_class in (0, 1):
                    valid = info["class_valid"][modality][disease_class]
                    ids = info["class_indices"][modality][disease_class][valid]
                    if ids.numel():
                        self.class_usage[modality, disease_class] += torch.bincount(
                            ids, minlength=self.class_prototypes
                        ).to(self.class_usage.device)
                    self.class_query_count[modality, disease_class] += valid.sum().to(self.class_query_count.device)
                for direction in range(2):
                    valid = info["hard_valid"][direction][modality]
                    ids = info["hard_indices"][direction][modality][valid]
                    if ids.numel():
                        self.hard_usage[direction, modality] += torch.bincount(
                            ids, minlength=self.hard_prototypes
                        ).to(self.hard_usage.device)
                    self.hard_query_count[direction, modality] += valid.sum().to(self.hard_query_count.device)

    def forward(self, h: torch.Tensor, query_ids: torch.Tensor, *, use_hard: bool = False,
                prototype_off: bool = False, semantic_swap: bool = False,
                hard_off: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.ndim != 2 or tuple(h.shape[1:]) != (144,):
            raise ValueError(f"EC-PSM expects the inherited CAM [N,144] boundary, got {tuple(h.shape)}")
        if prototype_off:
            empty = torch.empty(0, device=h.device)
            return h, {"residual": torch.zeros_like(h), "class_indices": [], "class_valid": [],
                       "hard_indices": [], "hard_valid": [], "class_evidence": empty,
                       "fp_similarity": empty, "fn_similarity": empty, "prompt": empty}

        n = h.shape[0]
        x = h.reshape(n, 4, 36)
        ids = query_ids.to(device=h.device, dtype=torch.long)
        retrieved_class = [[None, None] for _ in range(4)]
        class_indices, class_valid = [[None, None] for _ in range(4)], [[None, None] for _ in range(4)]
        fp_retrieved, fn_retrieved = [None] * 4, [None] * 4
        hard_indices, hard_valid = [[None for _ in range(4)] for _ in range(2)], [[None for _ in range(4)] for _ in range(2)]
        for modality in range(4):
            for disease_class in (0, 1):
                result = self._retrieve(
                    x[:, modality], ids, self.class_features[:, modality],
                    self.class_valid & (self.class_labels == disease_class),
                    self.class_assignment[modality, disease_class],
                    self.class_sums[modality, disease_class], self.class_counts[modality, disease_class],
                )
                retrieved_class[modality][disease_class] = result[0]
                class_indices[modality][disease_class] = result[1]
                class_valid[modality][disease_class] = result[2]
            for direction in range(2):
                if use_hard and not hard_off:
                    result = self._retrieve(
                        x[:, modality], ids, self.hard_features[direction, :, modality],
                        self.hard_valid[direction], self.hard_assignment[direction, modality],
                        self.hard_sums[direction, modality], self.hard_counts[direction, modality],
                    )
                    hard_indices[direction][modality] = result[1]
                    hard_valid[direction][modality] = result[2]
                    if direction == 0:
                        fp_retrieved[modality] = (result[0], result[2].any(dim=1))
                    else:
                        fn_retrieved[modality] = (result[0], result[2].any(dim=1))
                else:
                    hard_indices[direction][modality] = torch.zeros((n, min(self.top_k, self.hard_prototypes)), dtype=torch.long, device=h.device)
                    hard_valid[direction][modality] = torch.zeros((n, min(self.top_k, self.hard_prototypes)), dtype=torch.bool, device=h.device)
                    empty = torch.zeros_like(x[:, modality])
                    if direction == 0:
                        fp_retrieved[modality] = (empty, torch.zeros(n, dtype=torch.bool, device=h.device))
                    else:
                        fn_retrieved[modality] = (empty, torch.zeros(n, dtype=torch.bool, device=h.device))

        anchors = _unit(self.text_projection(self.text_bank))
        if semantic_swap:
            anchors = anchors.flip(0)
        summaries = [[None, None] for _ in range(4)]
        evidence = h.new_zeros((n, 4, 2))
        prompts = h.new_zeros((n, 4, 2, 9))
        fp_similarity, fn_similarity = h.new_zeros((n, 4)), h.new_zeros((n, 4))
        for modality in range(4):
            blocks = x[:, modality].reshape(n, 4, 9)
            normalized_blocks = _unit(blocks)
            for disease_class in (0, 1):
                other = 1 - disease_class
                retrieved = retrieved_class[modality][disease_class]
                class_difference = retrieved - retrieved_class[modality][other]
                anchor = anchors[disease_class].expand(n, -1)
                condition = torch.cat((retrieved, class_difference, anchor), dim=-1)
                prompt = _unit(anchor + 0.1 * torch.tanh(self.prompt_condition(condition)))
                prompt_blocks = prompt[:, None, :].expand(-1, 4, -1)
                block_scores = (normalized_blocks * _unit(prompt_blocks)).sum(dim=-1)
                attention = torch.softmax(block_scores / self.semantic_temperature, dim=-1)
                summaries[modality][disease_class] = (attention[:, :, None] * blocks).sum(dim=1)
                evidence[:, modality, disease_class] = (attention * block_scores).sum(dim=-1)
                prompts[:, modality, disease_class] = prompt
            if use_hard and not hard_off:
                fp_vector, fp_mask = fp_retrieved[modality]
                fn_vector, fn_mask = fn_retrieved[modality]
                sim_fp = (_unit(x[:, modality]) * _unit(fp_vector)).sum(dim=-1)
                sim_fn = (_unit(x[:, modality]) * _unit(fn_vector)).sum(dim=-1)
                fp_similarity[:, modality] = torch.where(fp_mask, sim_fp, torch.zeros_like(sim_fp))
                fn_similarity[:, modality] = torch.where(fn_mask, sim_fn, torch.zeros_like(sim_fn))

        class_direction = evidence[:, :, 0] - evidence[:, :, 1]
        adjusted_direction = class_direction + 0.1 * fn_similarity - 0.1 * fp_similarity
        updates = []
        for modality in range(4):
            evidence_input = torch.cat((
                x[:, modality],
                retrieved_class[modality][0] - retrieved_class[modality][1],
                summaries[modality][0],
                summaries[modality][1],
                adjusted_direction[:, modality:modality + 1],
            ), dim=-1)
            if evidence_input.shape[-1] != 91:
                raise RuntimeError(f"EC-PSM evidence shape changed: {tuple(evidence_input.shape)}")
            updates.append(0.05 * torch.tanh(self.evidence_residual(evidence_input)))
        residual = torch.stack(updates, dim=1).reshape(n, 144)
        transformed = h + residual
        return transformed, {
            "residual": residual,
            "class_indices": class_indices,
            "class_valid": class_valid,
            "hard_indices": hard_indices,
            "hard_valid": hard_valid,
            "class_evidence": class_direction.detach(),
            "fp_similarity": fp_similarity.detach(),
            "fn_similarity": fn_similarity.detach(),
            "prompt": prompts.detach(),
        }

    @torch.no_grad()
    def record_usage(self, info: Dict[str, torch.Tensor]) -> None:
        if info.get("class_valid"):
            self._record_retrieval_usage(info)

    @torch.no_grad()
    def usage_summary(self) -> Dict[str, object]:
        classes = {}
        for modality in range(4):
            for disease_class, name in ((0, "ASD"), (1, "TD")):
                counts = self.class_counts[modality, disease_class].cpu().numpy()
                active = counts[counts > 0]
                cosine = None
                active_centers = self.class_centers[modality, disease_class][self.class_counts[modality, disease_class] > 0]
                if active_centers.shape[0] > 1:
                    sim = _unit(active_centers) @ _unit(active_centers).T
                    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
                    cosine = float(upper.mean().item())
                classes[f"{modality}:{name}"] = {
                    "valid_subjects": int(self.class_valid.logical_and(self.class_labels == disease_class).sum().item()),
                    "active_prototypes": int((self.class_counts[modality, disease_class] > 0).sum().item()),
                    "occupancy": [int(v) for v in counts.tolist()],
                    "retrieval_counts": [int(v) for v in self.class_usage[modality, disease_class].cpu().tolist()],
                    "query_count": int(self.class_query_count[modality, disease_class].item()),
                    "coverage_rate": float((self.class_usage[modality, disease_class] > 0).float().mean().item()),
                    "mean_pairwise_cosine": cosine,
                    "collapse_rate_cos_gt_0_98": float(
                        (torch.triu(_unit(active_centers) @ _unit(active_centers).T, diagonal=1) > 0.98).sum().item()
                        / max(active_centers.shape[0] * (active_centers.shape[0] - 1) / 2, 1)
                    ) if active_centers.shape[0] > 1 else 0.0,
                }
        hard = {}
        for direction, name in ((0, "FP_TD_to_ASD"), (1, "FN_ASD_to_TD")):
            for modality in range(4):
                counts = self.hard_counts[direction, modality].cpu().tolist()
                n_subjects = int(self.hard_valid[direction].sum().item())
                active = self.hard_centers[direction, modality][self.hard_counts[direction, modality] > 0]
                cosine = None
                if active.shape[0] > 1:
                    sim = _unit(active) @ _unit(active).T
                    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
                    cosine = float(upper.mean().item())
                hard[f"{name}:{modality}"] = {
                    "true_error_subjects": n_subjects,
                    "active_prototypes": int((self.hard_counts[direction, modality] > 0).sum().item()),
                    "occupancy": [int(v) for v in counts],
                    "retrieval_counts": [int(v) for v in self.hard_usage[direction, modality].cpu().tolist()],
                    "query_count": int(self.hard_query_count[direction, modality].item()),
                    "retrieval_coverage": float(self.hard_query_count[direction, modality].item() > 0),
                    "mean_pairwise_cosine": cosine,
                }
        return {"class": classes, "hard": hard}


from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def deterministic_farthest_kmeans(x: np.ndarray, k: int, iterations: int = 20) -> np.ndarray:
    """Deterministic farthest-first initialization followed by deterministic Lloyd updates."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < k:
        raise ValueError(f"need at least K support points, got {x.shape} for K={k}")
    mean = x.mean(0, keepdims=True)
    first = int(np.argmax(((x - mean) ** 2).sum(1)))
    centers = [x[first].copy()]
    nearest = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        index = int(np.argmax(nearest))
        centers.append(x[index].copy())
        nearest = np.minimum(nearest, ((x - centers[-1]) ** 2).sum(1))
    centers = np.stack(centers)
    for _ in range(iterations):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = distances.argmin(1)
        updated = centers.copy()
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(0)
            else:
                updated[cluster] = x[int(np.argmax(distances.min(1)))]
        if np.array_equal(updated, centers):
            centers = updated
            break
        centers = updated
    return centers.astype(np.float32)


def initialize_anchors(module: "DRAPS", warmup_cam: torch.Tensor, labels: torch.Tensor,
                       support_ids, site_ids: np.ndarray) -> dict:
    """Build train-fold-only deterministic anchors from E61 epoch-10 CAM144."""
    support = np.asarray(support_ids, dtype=np.int64)
    y = labels.detach().cpu().numpy().astype(np.int64)
    h = warmup_cam.detach().cpu().double().numpy()
    sites = np.asarray(site_ids, dtype=np.int64)
    if h.shape != (871, 144) or y.shape != (871,) or sites.shape != (871,):
        raise ValueError("warmup CAM, labels, and site identity do not match the 871-row study")
    anchors = []
    counts_by_class = {}
    for cls in (0, 1):
        ids = support[y[support] == cls]
        class_k = module.k_by_class[cls]
        if len(ids) < max(20, class_k * 4):
            raise RuntimeError(f"insufficient legal class-{cls} support: {len(ids)}")
        residual = h[ids] - h[ids].mean(0, keepdims=True)
        robust = residual.copy()
        site_counts = {}
        for site in np.unique(sites[ids]):
            local = np.flatnonzero(sites[ids] == site)
            n = int(len(local))
            site_counts[str(int(site))] = n
            shrink = n / (n + 10.0)
            robust[local] -= shrink * residual[local].mean(0, keepdims=True)
        with torch.no_grad():
            tensor = torch.as_tensor(
                robust, dtype=module.heterogeneity_projection.weight.dtype,
                device=module.heterogeneity_projection.weight.device,
            )
            projected = unit(module.heterogeneity_norm(module.heterogeneity_projection(tensor)))
        centers = deterministic_farthest_kmeans(projected.detach().cpu().double().numpy(), class_k)
        if class_k < module.k:
            pad = np.repeat(centers.mean(0, keepdims=True), module.k - class_k, axis=0)
            centers = np.concatenate([centers, pad], axis=0)
        anchors.append(centers)
        counts_by_class[str(cls)] = site_counts
    anchor_tensor = unit(torch.as_tensor(np.stack(anchors), dtype=module.anchors.dtype,
                                         device=module.anchors.device))
    module.anchors.copy_(anchor_tensor)
    return {
        "support_count": int(len(support)),
        "class_support_count": {str(c): int((y[support] == c).sum()) for c in (0, 1)},
        "site_counts_by_class": counts_by_class,
        "anchor_sha256_tensor": module.anchor_digest(),
        "source_identity": "E61 epoch-10 CAM144, legal outer-train support only; labels used only on support rows",
    }


class DRAPS(nn.Module):
    """Disease-discriminative reliability-aware prototype semantic fusion.

    The module owns the two trainable patient projections, a shared semantic
    residual map, and a six-statistic low-capacity patient gate. Prototypes and
    their EMA anchors are fold-local buffers/statistics; no classifier bypass is
    introduced.
    """

    def __init__(self, text_coords: torch.Tensor, prompt_classes, k: int, prompt_source: str):
        super().__init__()
        if isinstance(k, (tuple, list)):
            self.k_by_class = (int(k[0]), int(k[1]))
        else:
            self.k_by_class = (int(k), int(k))
        if min(self.k_by_class) < 1:
            raise ValueError("each disease class needs at least one candidate prototype")
        self.k = max(self.k_by_class)
        self.prompt_source = str(prompt_source)
        self.semantic_projection = nn.Linear(144, 16, bias=False)
        self.heterogeneity_projection = nn.Linear(144, 32, bias=False)
        self.heterogeneity_norm = nn.LayerNorm(32)
        self.feature_projection = nn.Linear(16, 144, bias=True)
        self.gate_linear = nn.Linear(6, 1, bias=True)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, math.log(0.2 / 0.8))
        self.register_buffer("text_coords", text_coords.detach().float().clone())
        self.register_buffer("prompt_classes", torch.as_tensor(prompt_classes, dtype=torch.long))
        self.register_buffer("anchors", torch.zeros(2, self.k, 32))
        candidate_mask = torch.zeros(2, self.k, dtype=torch.bool)
        for cls, class_k in enumerate(self.k_by_class):
            candidate_mask[cls, :class_k] = True
        self.register_buffer("candidate_mask", candidate_mask)
        self.cluster_temperature = 0.25
        self.route_temperature = 0.25
        self.prompt_temperature = 0.5
        self.proto_temperature = 0.25
        self.ema_decay = 0.9
        self.ema_weight = 0.1
        self.eps = 1e-8

    def anchor_digest(self) -> str:
        data = self.anchors.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    @torch.no_grad()
    def update_ema(self, dynamic_proto_z: torch.Tensor) -> str:
        """Update non-gradient anchors exactly once per epoch without RNG use."""
        if tuple(dynamic_proto_z.shape) != tuple(self.anchors.shape):
            raise RuntimeError(f"EMA source shape {tuple(dynamic_proto_z.shape)} != {tuple(self.anchors.shape)}")
        source = unit(dynamic_proto_z.detach())
        updated = unit(self.ema_decay * self.anchors + self.ema_weight * source)
        self.anchors.copy_(updated)
        return self.anchor_digest()

    def _support_site_weights(self, support_ids: torch.Tensor, site_ids: torch.Tensor,
                              support_labels: torch.Tensor) -> torch.Tensor:
        """Within-class inverse-sqrt site-size weights, used for every K including K=1."""
        weights = torch.ones(len(support_ids), dtype=self.anchors.dtype, device=self.anchors.device)
        for cls in (0, 1):
            mask = support_labels == cls
            if not bool(mask.any()):
                raise RuntimeError(f"class {cls} has no legal training support")
            _, inverse, counts = torch.unique(site_ids[support_ids[mask]], sorted=True,
                                              return_inverse=True, return_counts=True)
            local = counts[inverse].to(weights.dtype).rsqrt()
            weights[mask] = local / local.mean().clamp_min(self.eps)
        return weights

    def _site_reliability(self, support_mass: torch.Tensor, support_sites: torch.Tensor,
                          class_totals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return [2,K] site reliability and [2,K] number of supported sites."""
        unique_sites = torch.unique(support_sites, sorted=True)
        n_sites = int(unique_sites.numel())
        if n_sites == 0:
            return torch.zeros((2, self.k), dtype=support_mass.dtype, device=support_mass.device), \
                torch.zeros((2, self.k), dtype=support_mass.dtype, device=support_mass.device)
        # Avoid CUDA atomic index_add_: fixed-mask reductions are deterministic and
        # are part of the public two-step recovery contract.
        site_mass = torch.stack([
            torch.stack([
                support_mass[support_sites == site, cls, :].sum(dim=0)
                for site in unique_sites
            ], dim=0)
            for cls in (0, 1)
        ], dim=0)
        support_count = (site_mass > self.eps).to(support_mass.dtype).sum(dim=1)
        coverage = support_count / float(n_sites)
        probs = site_mass / class_totals[:, None, :].clamp_min(self.eps)
        entropy = -(probs.clamp_min(self.eps) * probs.clamp_min(self.eps).log()).sum(dim=1)
        if n_sites > 1:
            entropy = entropy / math.log(n_sites)
        else:
            entropy = torch.ones_like(entropy)
        site_rel = 0.5 * (coverage + entropy)
        site_rel = site_rel * self.candidate_mask.to(site_rel.dtype)
        return site_rel, support_count

    def _aggregate(self, q: torch.Tensor, z: torch.Tensor, labels: torch.Tensor,
                   support_ids: torch.Tensor, site_ids: torch.Tensor, query_ids: torch.Tensor):
        support_y = labels[support_ids]
        if torch.any((support_y < 0) | (support_y > 1)):
            raise RuntimeError("support labels must use ASD=0, TD=1")
        if len(torch.unique(support_ids)) != len(support_ids):
            raise RuntimeError("duplicate prototype support identities")
        if self.k_by_class == (1, 1):
            rho = torch.ones((len(support_ids), 1), dtype=q.dtype, device=q.device)
        else:
            rho_logits = torch.einsum("nd,nkd->nk", z[support_ids], self.anchors[support_y]) \
                / self.cluster_temperature
            rho_mask = self.candidate_mask[support_y]
            rho = torch.softmax(rho_logits.masked_fill(~rho_mask, float("-inf")), dim=-1) * rho_mask.to(q.dtype)
        site_weight = self._support_site_weights(support_ids, site_ids, support_y)
        onehot = F.one_hot(support_y, num_classes=2).to(q.dtype)
        support_mass = onehot[:, :, None] * (site_weight[:, None, None] * rho[:, None, :])
        base_counts = support_mass.sum(0)  # [2,K]
        base_sum_z = torch.einsum("nck,nd->ckd", support_mass, z[support_ids])
        base_sum_q = torch.einsum("nck,nd->ckd", support_mass, q[support_ids])
        if torch.any(base_counts[self.candidate_mask] <= self.eps):
            raise RuntimeError("dynamic prototype has non-positive support mass")
        base_proto_z = unit(base_sum_z / base_counts.clamp_min(self.eps)[..., None])
        base_proto_q = unit(base_sum_q / base_counts.clamp_min(self.eps)[..., None])
        valid_proto_mask = self.candidate_mask[..., None]
        base_proto_z = torch.where(valid_proto_mask, base_proto_z, torch.zeros_like(base_proto_z))
        base_proto_q = torch.where(valid_proto_mask[..., :16], base_proto_q, torch.zeros_like(base_proto_q))

        # Exact LOO: only a query that is itself in legal support has a removed mass.
        pos = torch.searchsorted(support_ids, query_ids)
        valid = pos < len(support_ids)
        safe = pos.clamp_max(max(len(support_ids) - 1, 0))
        is_support = valid & (support_ids[safe] == query_ids)
        qclass = support_y[safe]
        own_onehot = F.one_hot(qclass, num_classes=2).to(q.dtype) * is_support[:, None].to(q.dtype)
        removed_mass = own_onehot[:, :, None] * (site_weight[safe, None, None] * rho[safe, None, :])
        sums_z = base_sum_z[None, :, :, :] - removed_mass[..., None] * z[query_ids, None, None, :]
        sums_q = base_sum_q[None, :, :, :] - removed_mass[..., None] * q[query_ids, None, None, :]
        counts = base_counts[None, :, :] - removed_mass
        if torch.any(counts[:, self.candidate_mask] <= self.eps):
            raise RuntimeError("LOO prototype has zero effective support")
        proto_z = unit(sums_z / counts.clamp_min(self.eps)[..., None])
        proto_q = unit(sums_q / counts.clamp_min(self.eps)[..., None])
        query_valid_mask = self.candidate_mask[None, :, :, None]
        proto_z = torch.where(query_valid_mask, proto_z, torch.zeros_like(proto_z))
        proto_q = torch.where(query_valid_mask[..., :16], proto_q, torch.zeros_like(proto_q))
        class_sums_z = sums_z.sum(dim=2)
        class_sums_q = sums_q.sum(dim=2)
        class_counts = counts.sum(dim=2)
        global_loo_z = unit(class_sums_z / class_counts[..., None])
        global_loo_q = unit(class_sums_q / class_counts[..., None])

        mass_sq = support_mass.square().sum(0)
        n_eff = base_counts.square() / (mass_sq + self.eps)
        site_rel, site_support_count = self._site_reliability(
            support_mass, site_ids[support_ids], base_counts
        )
        support_rel = ((n_eff - 20.0) / 10.0).clamp(0.0, 1.0)
        reliability = support_rel * site_rel
        fallback = reliability.sum(dim=1) <= self.eps
        valid_counts = self.candidate_mask.to(q.dtype).sum(dim=1, keepdim=True)
        uniform = self.candidate_mask.to(q.dtype) / valid_counts.clamp_min(1.0)
        reliability_weights = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(self.eps)
        if bool(fallback.any()):
            reliability_weights = torch.where(fallback[:, None], uniform, reliability_weights)
            for cls in (0, 1):
                if bool(fallback[cls]):
                    proto_z[:, cls, :, :] = global_loo_z[:, cls, None, :]
                    proto_q[:, cls, :, :] = global_loo_q[:, cls, None, :]
                    base_proto_z[cls] = unit(base_sum_z[cls].sum(0) / base_counts[cls].sum().clamp_min(self.eps))
                    base_proto_q[cls] = unit(base_sum_q[cls].sum(0) / base_counts[cls].sum().clamp_min(self.eps))
        query_z = z[query_ids]
        route_logits = torch.einsum("nd,nckd->nck", query_z, proto_z) / self.route_temperature
        route_mask = self.candidate_mask[None, :, :]
        beta = torch.softmax(route_logits.masked_fill(~route_mask, float("-inf")), dim=-1)
        beta = beta * route_mask.to(q.dtype)
        beta_reliable = beta * reliability_weights[None, :, :]
        beta_reliable = beta_reliable / beta_reliable.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        class_reliability_query = torch.einsum("nck,ck->nc", beta, reliability)
        proto_similarity = torch.einsum("nd,nckd->nck", query_z, proto_z)
        log_reliability = torch.where(
            self.candidate_mask[None, :, :],
            reliability_weights[None, :, :].clamp_min(self.eps).log(),
            torch.full_like(reliability_weights[None, :, :], float("-inf")),
        )
        proto_scores = self.proto_temperature * torch.logsumexp(
            log_reliability + proto_similarity / self.proto_temperature, dim=-1
        )
        return {
            "rho_support": rho,
            "support_site_weight": site_weight,
            "support_mass": support_mass,
            "proto_z": proto_z,
            "proto_q": proto_q,
            "base_proto_z": base_proto_z,
            "base_proto_q": base_proto_q,
            "global_loo_z": global_loo_z,
            "global_loo_q": global_loo_q,
            "beta": beta,
            "beta_reliable": beta_reliable,
            "proto_similarity": proto_similarity,
            "proto_scores": proto_scores,
            "n_eff": n_eff,
            "support_reliability": support_rel,
            "site_reliability": site_rel,
            "reliability": reliability,
            "reliability_weights": reliability_weights,
            "site_support_count": site_support_count,
            "class_fallback": fallback,
            "class_reliability_query": class_reliability_query,
            "counts": counts,
            "base_counts": base_counts,
            "loo_removed_mass": removed_mass,
            "is_support_query": is_support,
            "support_position": safe,
        }

    def _site_loss(self, rho: torch.Tensor, support_y: torch.Tensor,
                   support_ids: torch.Tensor, site_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # The K=1 branch is naturally zero: no within-class prototype choice exists.
        if self.k_by_class == (1, 1):
            return rho.new_zeros(()), {"groups": 0, "max_kl": 0.0}
        terms = []
        max_kl = 0.0
        for cls in (0, 1):
            class_k = self.k_by_class[cls]
            if class_k <= 1:
                continue
            cls_mask = support_y == cls
            class_mean = rho[cls_mask, :class_k].mean(0).clamp_min(self.eps)
            for site in torch.unique(site_ids[support_ids[cls_mask]], sorted=True):
                group = cls_mask & (site_ids[support_ids] == site)
                if int(group.sum()) < 20:
                    continue
                group_mean = rho[group, :class_k].mean(0).clamp_min(self.eps)
                kl = (group_mean * (group_mean.log() - class_mean.log())).sum() / math.log(class_k)
                terms.append(kl)
                max_kl = max(max_kl, float(kl.detach().cpu()))
        if not terms:
            return rho.new_zeros(()), {"groups": 0, "max_kl": 0.0}
        return torch.stack(terms).mean(), {"groups": len(terms), "max_kl": max_kl}

    def _effective_reliability(self, agg: dict, intervention: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if intervention == "reliability_off" or intervention == "global_center":
            uniform = self.candidate_mask.to(agg["reliability_weights"].dtype)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            beta = agg["beta"]
            beta_rel = beta * uniform[None, :, :]
            beta_rel = beta_rel / beta_rel.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            return uniform, beta_rel
        return agg["reliability_weights"], agg["beta_reliable"]

    def forward_feature(self, h: torch.Tensor, labels: torch.Tensor, support_ids,
                        query_ids, site_ids, class_weights: torch.Tensor,
                        *, training: bool, intervention: str = "full"):
        if h.ndim != 2 or tuple(h.shape) != (871, 144):
            raise RuntimeError(f"DRAPS requires CAM144 [871,144], got {tuple(h.shape)}")
        device = h.device
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device).sort().values
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        site_ids = torch.as_tensor(site_ids, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        q = unit(self.semantic_projection(h))
        z = unit(self.heterogeneity_norm(self.heterogeneity_projection(h)))
        agg = self._aggregate(q, z, labels, support_ids, site_ids, query_ids)
        proto_z, proto_q = agg["proto_z"], agg["proto_q"]
        if intervention == "global_center":
            proto_z = agg["global_loo_z"][:, :, None, :].expand_as(proto_z)
            proto_q = agg["global_loo_q"][:, :, None, :].expand_as(proto_q)
        reliability_weights, beta_reliable = self._effective_reliability(agg, intervention)
        # Patient-conditioned opposing disease references; there is no index-wise ASD/TD pairing.
        vbar = torch.einsum("nck,nckd->ncd", beta_reliable, proto_q)

        prompt_bank = 1 if intervention == "anonymous" else (0 if self.prompt_source == "spromptgl" else 1)
        prompt_coords = self.text_coords[prompt_bank]
        prompt_classes = self.prompt_classes
        if intervention == "prompt_class_swap":
            prompt_classes = 1 - prompt_classes
        direct = q[query_ids] @ prompt_coords.T
        class_scores, class_attention, class_prompt_vectors = [], [], []
        prompt_indices = []
        for cls in (0, 1):
            idx = torch.nonzero(prompt_classes == cls, as_tuple=False).flatten()
            if len(idx) != 4:
                raise RuntimeError("each disease class must retain exactly four prompts")
            prompt_indices.append(idx)
            opposite = 1 - cls
            current = torch.einsum("nkd,pd->nkp", proto_q[:, cls], prompt_coords[idx])
            competing = torch.einsum("nd,pd->np", vbar[:, opposite], prompt_coords[idx])
            score = direct[:, None, idx] + current - competing[:, None, :]
            alpha = torch.softmax(score / self.prompt_temperature, dim=-1)
            class_scores.append((alpha * score).sum(-1))
            class_attention.append(alpha)
            class_prompt_vectors.append(torch.einsum("nkp,pd->nkd", alpha, prompt_coords[idx]))
        class_evidence = torch.stack([
            (beta_reliable[:, cls] * class_scores[cls]).sum(-1) for cls in (0, 1)
        ], dim=1)
        pooled_prompt = []
        for cls in (0, 1):
            opposite = 1 - cls
            enriched = class_prompt_vectors[cls] + 0.5 * (proto_q[:, cls] - vbar[:, opposite, None, :])
            pooled_prompt.append((beta_reliable[:, cls, :, None] * enriched).sum(1))
        semantic_vector = unit(pooled_prompt[0] - pooled_prompt[1])
        if intervention == "semantic_off":
            semantic_vector = torch.zeros_like(semantic_vector)
        raw_residual = self.feature_projection(semantic_vector)
        residual_norm = raw_residual.norm(dim=-1, keepdim=True)
        target_norm = h[query_ids].norm(dim=-1, keepdim=True)
        normalized_residual = raw_residual * target_norm / residual_norm.clamp_min(self.eps)
        if intervention == "semantic_off":
            normalized_residual = torch.zeros_like(normalized_residual)

        proto_scores = agg["proto_scores"]
        if intervention in {"reliability_off", "global_center"}:
            query_z = z[query_ids]
            similarity_for_scores = torch.einsum("nd,nckd->nck", query_z, proto_z)
            log_weights = torch.where(
                self.candidate_mask[None, :, :],
                reliability_weights[None, :, :].clamp_min(self.eps).log(),
                torch.full_like(reliability_weights[None, :, :], float("-inf")),
            )
            proto_scores = self.proto_temperature * torch.logsumexp(
                log_weights + similarity_for_scores / self.proto_temperature, dim=-1
            )
        raw_rel = agg["reliability"]
        if intervention == "reliability_off":
            raw_rel = torch.ones_like(raw_rel)
        rel_query = torch.einsum("nck,ck->nc", agg["beta"], raw_rel)
        match_margin = torch.tanh((proto_scores[:, 0] - proto_scores[:, 1]).detach() * 1.0)
        semantic_margin = torch.tanh((class_evidence[:, 0] - class_evidence[:, 1]).detach() * 1.0)
        conflict = (match_margin - semantic_margin).abs().clamp(0.0, 1.0)
        patient_reliability = (0.5 * rel_query.sum(-1) * (1.0 - 0.5 * conflict)).clamp(0.0, 1.0)
        gate_statistics = torch.stack([
            rel_query[:, 0], rel_query[:, 1], match_margin, semantic_margin,
            0.5 * rel_query.sum(-1), -conflict,
        ], dim=1)
        gate = 0.25 * torch.sigmoid(self.gate_linear(gate_statistics)).squeeze(-1) * patient_reliability
        fusion_off = intervention in {"feature_residual_off", "module_off"}
        effective_gate = gate if not fusion_off else gate * 0.0
        feature_residual = torch.zeros_like(h)
        fused_query = h[query_ids] + effective_gate[:, None] * normalized_residual
        feature_residual = feature_residual.index_copy(0, query_ids, normalized_residual)
        h_fused = h.index_copy(0, query_ids, fused_query) if not fusion_off else h

        if training:
            pos = torch.searchsorted(support_ids, query_ids)
            safe = pos.clamp_max(len(support_ids) - 1)
            legal = (pos < len(support_ids)) & (support_ids[safe] == query_ids)
            if not bool(legal.all()):
                raise RuntimeError("training auxiliary queries must all be legal support patients")
            query_labels = labels[query_ids]
            proto_loss = F.cross_entropy(proto_scores / self.proto_temperature, query_labels,
                                         weight=class_weights)
            semantic_loss = F.cross_entropy(class_evidence, query_labels, weight=class_weights)
            site_loss, site_info = self._site_loss(agg["rho_support"], labels[support_ids],
                                                   support_ids, site_ids)
        else:
            proto_loss = h.new_zeros(())
            semantic_loss = h.new_zeros(())
            site_loss, site_info = h.new_zeros(()), {"groups": 0, "max_kl": 0.0}
        losses = {
            "proto_raw": proto_loss,
            "sem_raw": semantic_loss,
            "site_raw": site_loss,
            "proto_weighted": 0.05 * proto_loss,
            "sem_weighted": 0.02 * semantic_loss,
            "site_weighted": 0.005 * site_loss,
        }
        return h_fused, {
            "h_original": h,
            "h_fused": h_fused,
            "feature_residual": feature_residual,
            "normalized_residual": normalized_residual,
            "semantic_vector": semantic_vector,
            "semantic_loss": semantic_loss,
            "proto_loss": proto_loss,
            "class_evidence": class_evidence,
            "proto_scores": proto_scores,
            "attention": torch.stack(class_attention, dim=1),
            "beta": agg["beta"],
            "beta_reliable": beta_reliable,
            "rho_support": agg["rho_support"],
            "support_mass": agg["support_mass"],
            "support_site_weight": agg["support_site_weight"],
            "proto_z": proto_z,
            "proto_q": proto_q,
            "base_proto_z": agg["base_proto_z"],
            "base_proto_q": agg["base_proto_q"],
            "global_loo_z": agg["global_loo_z"],
            "global_loo_q": agg["global_loo_q"],
            "prototype_counts": agg["counts"],
            "base_counts": agg["base_counts"],
            "loo_removed_mass": agg["loo_removed_mass"],
            "is_support_query": agg["is_support_query"],
            "n_eff": agg["n_eff"],
            "support_reliability": agg["support_reliability"],
            "site_reliability": agg["site_reliability"],
            "prototype_reliability": agg["reliability"],
            "reliability_weights": reliability_weights,
            "site_support_count": agg["site_support_count"],
            "class_fallback": agg["class_fallback"],
            "class_reliability_query": rel_query,
            "vbar": vbar,
            "gate": effective_gate,
            "gate_raw": gate,
            "gate_statistics": gate_statistics,
            "patient_reliability": patient_reliability,
            "conflict": conflict,
            "match_margin": match_margin,
            "semantic_margin": semantic_margin,
            "support_ids": support_ids,
            "query_ids": query_ids,
            "support_labels": labels[support_ids],
            "support_sites": site_ids[support_ids],
            "class_prompt_indices": prompt_indices,
            "prompt_coords": prompt_coords,
            "losses": losses,
            "site_info": site_info,
            "ema_source": agg["base_proto_z"].detach(),
            "intervention": intervention,
            "prompt_bank": prompt_bank,
            "raw_residual_norm": residual_norm,
            "target_feature_norm": target_norm,
        }


def initialize(text_coords: torch.Tensor, prompt_classes: torch.Tensor, k: int,
               prompt_source: str, fold: int, device: torch.device) -> DRAPS:
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(782000 + int(fold))
        module = DRAPS(text_coords, prompt_classes, k, prompt_source)
    return module.to(device)

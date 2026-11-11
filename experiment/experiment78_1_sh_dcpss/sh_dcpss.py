from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def deterministic_farthest_kmeans(x: np.ndarray, k: int, iterations: int = 20) -> np.ndarray:
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


def initialize_anchors(module: "SHDCPSS", warmup_cam: torch.Tensor, labels: torch.Tensor,
                       support_ids, site_ids: np.ndarray) -> dict:
    support = np.asarray(support_ids, dtype=np.int64)
    y = labels.detach().cpu().numpy().astype(np.int64)
    h = warmup_cam.detach().cpu().double().numpy()
    sites = np.asarray(site_ids, dtype=np.int64)
    if h.shape != (871, 144) or y.shape != (871,) or sites.shape != (871,):
        raise ValueError("warmup CAM, labels, and site identity do not match the 871-row study")
    anchors = []
    counts_by_site = {}
    robust_by_class = {}
    for cls in (0, 1):
        ids = support[y[support] == cls]
        if len(ids) < max(20, module.k * 4):
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
        robust_by_class[str(cls)] = robust
        counts_by_site[str(cls)] = site_counts
        with torch.no_grad():
            tensor = torch.as_tensor(robust, dtype=module.heterogeneity_projection.weight.dtype,
                                     device=module.heterogeneity_projection.weight.device)
            projected = unit(module.heterogeneity_norm(module.heterogeneity_projection(tensor)))
        projected_np = projected.detach().cpu().double().numpy()
        centers = deterministic_farthest_kmeans(projected_np, module.k)
        anchors.append(centers)
    anchor_tensor = unit(torch.as_tensor(np.stack(anchors), dtype=module.anchors.dtype,
                                         device=module.anchors.device))
    module.anchors.copy_(anchor_tensor)
    return {
        "support_count": int(len(support)),
        "class_support_count": {"0": int((y[support] == 0).sum()), "1": int((y[support] == 1).sum())},
        "site_counts_by_class": counts_by_site,
        "anchor_sha256_tensor": module.anchor_digest(),
        "source_identity": "E61 epoch-10 CAM144, legal outer-train only; labels used only on support rows",
    }


class SHDCPSS(nn.Module):
    """Heterogeneity-preserving, fold-support disease prototypes with CAM feature fusion."""

    def __init__(self, text_coords: torch.Tensor, prompt_classes, k: int, prompt_source: str):
        super().__init__()
        self.k = int(k)
        self.prompt_source = str(prompt_source)
        self.semantic_projection = nn.Linear(144, 16, bias=False)
        self.heterogeneity_projection = nn.Linear(144, 32, bias=False)
        self.heterogeneity_norm = nn.LayerNorm(32)
        self.feature_projection = nn.Linear(16, 144, bias=True)
        self.lambda_raw = nn.Parameter(torch.tensor(float(math.log(0.4 / 0.6))))
        self.register_buffer("text_coords", text_coords.detach().float().clone())
        self.register_buffer("prompt_classes", torch.as_tensor(prompt_classes, dtype=torch.long))
        self.register_buffer("anchors", torch.zeros(2, self.k, 32))
        self.cluster_temperature = 0.25
        self.route_temperature = 0.25
        self.prompt_temperature = 0.5

    @property
    def fusion_lambda(self):
        return 0.25 * torch.sigmoid(self.lambda_raw)

    def anchor_digest(self):
        import hashlib
        data = self.anchors.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def _support_site_weights(self, support_ids: torch.Tensor, site_ids: torch.Tensor,
                              support_labels: torch.Tensor) -> torch.Tensor:
        n = len(support_ids)
        weights = torch.ones(n, dtype=self.anchors.dtype, device=self.anchors.device)
        if self.k == 1:
            return weights
        for cls in (0, 1):
            cls_mask = support_labels == cls
            sites, inverse, counts = torch.unique(site_ids[support_ids[cls_mask]],
                                                  sorted=True, return_inverse=True,
                                                  return_counts=True)
            local = counts[inverse].to(weights.dtype).rsqrt()
            weights[cls_mask] = local / local.mean().clamp_min(1e-8)
        return weights

    def _aggregate(self, q: torch.Tensor, z: torch.Tensor, y: torch.Tensor,
                   support_ids: torch.Tensor, site_ids: torch.Tensor,
                   query_ids: torch.Tensor, prototype_off: bool):
        support_y = y[support_ids]
        if torch.any((support_y < 0) | (support_y > 1)):
            raise RuntimeError("support labels must use ASD=0, TD=1")
        if prototype_off and self.k > 1:
            rho = torch.full((len(support_ids), self.k), 1.0 / self.k,
                             dtype=q.dtype, device=q.device)
        elif self.k == 1:
            rho = torch.ones((len(support_ids), 1), dtype=q.dtype, device=q.device)
        else:
            support_anchor = self.anchors[support_y]
            rho = torch.softmax(torch.einsum("sd,skd->sk", z[support_ids], support_anchor)
                                / self.cluster_temperature, dim=-1)
        site_weight = self._support_site_weights(support_ids, site_ids, support_y)
        class_mask = F.one_hot(support_y, num_classes=2).to(q.dtype)
        support_mass = class_mask[:, :, None] * (site_weight[:, None, None] * rho[:, None, :])
        sum_z = torch.einsum("scK,sd->cKd", support_mass, z[support_ids])
        sum_q = torch.einsum("scK,sd->cKd", support_mass, q[support_ids])
        base_counts = support_mass.sum(0)

        pos = torch.searchsorted(support_ids, query_ids)
        valid_pos = pos < len(support_ids)
        safe_pos = pos.clamp_max(max(len(support_ids) - 1, 0))
        is_support = valid_pos & (support_ids[safe_pos] == query_ids)
        support_pos = safe_pos
        qclass = support_y[support_pos]
        qclass_mask = F.one_hot(qclass, num_classes=2).to(q.dtype) * is_support[:, None].to(q.dtype)
        removed_mass = qclass_mask[:, :, None] * (
            site_weight[support_pos, None, None] * rho[support_pos, None, :]
        )
        removed_z = removed_mass[..., None] * z[query_ids, None, None, :]
        removed_q = removed_mass[..., None] * q[query_ids, None, None, :]
        counts = base_counts[None, :, :] - removed_mass
        sums_z = sum_z[None, :, :, :] - removed_z
        sums_q = sum_q[None, :, :, :] - removed_q
        if torch.any(counts <= 0):
            raise RuntimeError("LOO prototype has zero effective support")
        proto_z = unit(sums_z / counts[..., None])
        proto_q = unit(sums_q / counts[..., None])
        if prototype_off and self.k > 1:
            proto_z = proto_z.mean(2, keepdim=True).expand_as(proto_z)
            proto_q = proto_q.mean(2, keepdim=True).expand_as(proto_q)
        beta = torch.softmax(torch.einsum("qd,qckd->qck", z[query_ids], proto_z)
                             / self.route_temperature, dim=-1)
        return {
            "rho_support": rho,
            "support_site_weight": site_weight,
            "proto_z": proto_z,
            "proto_q": proto_q,
            "beta": beta,
            "counts": counts,
            "base_counts": base_counts,
            "loo_removed_mass": removed_mass,
            "is_support_query": is_support,
            "support_position": support_pos,
        }

    def _heterogeneity_losses(self, z, support_ids, query_ids, support_y,
                              aggregates, rho, site_ids):
        zero = z.new_zeros(())
        if self.k == 1:
            separation = zero
            utilization = zero
        else:
            eye = torch.eye(self.k, dtype=torch.bool, device=z.device)
            pair_mask = torch.triu(~eye, diagonal=1)
            dynamic_proto = aggregates["proto_z"].mean(0)
            pair_sims = torch.einsum("ckd,cmd->ckm", dynamic_proto, dynamic_proto)
            pair_values = pair_sims[:, pair_mask]
            separation = (F.relu(pair_values - 0.95) / 0.05).mean().clamp(0.0, 1.0)
            class_utilization = []
            for cls in (0, 1):
                mean_rho = rho[support_y == cls].mean(0).clamp_min(1e-8)
                class_utilization.append(
                    (mean_rho * (mean_rho.log() + math.log(self.k))).sum() / math.log(self.k)
                )
            utilization = torch.stack(class_utilization).mean()
        class_stds = []
        for cls in (0, 1):
            values = z[support_ids[support_y == cls]]
            class_stds.append(values.std(0, unbiased=False).mean())
        variance = F.relu((0.15 - torch.stack(class_stds).mean()) / 0.15).clamp(0.0, 1.0)
        qpos = aggregates["is_support_query"].nonzero(as_tuple=False).flatten()
        if len(qpos):
            query_y = support_y[aggregates["support_position"][qpos]]
            pos_beta = aggregates["beta"][qpos, query_y]
            pos_proto = aggregates["proto_z"][qpos, query_y]
            similarities = torch.einsum("qd,qkd->qk", z[query_ids[qpos]], pos_proto)
            compactness = ((pos_beta * (1.0 - similarities) / 2.0).sum(-1)).mean().clamp(0.0, 1.0)
        else:
            compactness = zero
        total = 0.35 * variance + 0.25 * separation + 0.20 * utilization + 0.20 * compactness
        return total, {"variance": variance, "separation": separation,
                       "utilization": utilization, "compactness": compactness}

    def _site_loss(self, rho: torch.Tensor, support_y: torch.Tensor,
                   support_ids: torch.Tensor, site_ids: torch.Tensor):
        if self.k == 1:
            return rho.new_zeros(()), {"groups": 0, "max_kl": 0.0}
        terms = []
        max_kl = 0.0
        for cls in (0, 1):
            cls_mask = support_y == cls
            class_mean = rho[cls_mask].mean(0).clamp_min(1e-8)
            for site in torch.unique(site_ids[support_ids[cls_mask]], sorted=True):
                group = cls_mask & (site_ids[support_ids] == site)
                if int(group.sum()) < 20:
                    continue
                group_mean = rho[group].mean(0).clamp_min(1e-8)
                kl = (group_mean * (group_mean.log() - class_mean.log())).sum() / math.log(self.k)
                terms.append(kl)
                max_kl = max(max_kl, float(kl.detach().cpu()))
        if not terms:
            return rho.new_zeros(()), {"groups": 0, "max_kl": 0.0}
        return torch.stack(terms).mean(), {"groups": len(terms), "max_kl": max_kl}

    def forward_feature(self, h: torch.Tensor, labels: torch.Tensor, support_ids,
                        query_ids, site_ids, class_weights: torch.Tensor,
                        *, training: bool, intervention: str = "full"):
        if h.ndim != 2 or tuple(h.shape) != (871, 144):
            raise RuntimeError(f"SH-DCPSS requires CAM144 [871,144], got {tuple(h.shape)}")
        device = h.device
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device).sort().values
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        site_ids = torch.as_tensor(site_ids, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        q = unit(self.semantic_projection(h))
        z = unit(self.heterogeneity_norm(self.heterogeneity_projection(h)))
        subtype_off = intervention == "subprototype_off"
        aggregates = self._aggregate(q, z, labels, support_ids, site_ids, query_ids, subtype_off)
        support_y = labels[support_ids]
        heterogeneity, heterogeneity_parts = self._heterogeneity_losses(
            z, support_ids, query_ids, support_y, aggregates,
            aggregates["rho_support"], site_ids
        ) if training else (h.new_zeros(()), {})
        site_loss, site_info = self._site_loss(
            aggregates["rho_support"], support_y, support_ids, site_ids
        ) if training else (h.new_zeros(()), {"groups": 0, "max_kl": 0.0})

        prompt_bank = 1 if intervention == "anonymous" else (0 if self.prompt_source == "spromptgl" else 1)
        prompt_coords = self.text_coords[prompt_bank]
        prompt_classes = self.prompt_classes
        if intervention == "prompt_class_swap":
            prompt_classes = 1 - prompt_classes
        direct = q[query_ids] @ prompt_coords.T
        prompt_indices = [(prompt_classes == cls).nonzero(as_tuple=False).flatten() for cls in (0, 1)]
        if any(len(indices) != 4 for indices in prompt_indices):
            raise RuntimeError("each disease class must retain exactly four prompts")
        class_scores, class_attention, class_prompt_vectors = [], [], []
        for cls in (0, 1):
            opposite = 1 - cls
            idx = prompt_indices[cls]
            current = torch.einsum("qkd,pd->qkp", aggregates["proto_q"][:, cls], prompt_coords[idx])
            competing = torch.einsum("qkd,pd->qkp", aggregates["proto_q"][:, opposite], prompt_coords[idx])
            score = direct[:, None, idx] + current - competing
            alpha = torch.softmax(score / self.prompt_temperature, dim=-1)
            class_scores.append((alpha * score).sum(-1))
            class_attention.append(alpha)
            class_prompt_vectors.append(torch.einsum("qkp,pd->qkd", alpha, prompt_coords[idx]))
        subtype_evidence = torch.stack(class_scores, dim=1)
        attention = torch.stack(class_attention, dim=1)
        class_evidence = (aggregates["beta"] * subtype_evidence).sum(-1)
        prompt_vectors = []
        for cls in (0, 1):
            opposite = 1 - cls
            diff = aggregates["proto_q"][:, cls] - aggregates["proto_q"][:, opposite]
            enriched = class_prompt_vectors[cls] + 0.5 * diff
            prompt_vectors.append((aggregates["beta"][:, cls, :, None] * enriched).sum(1))
        semantic_vector = unit(prompt_vectors[0] - prompt_vectors[1])
        semantic_off = intervention == "semantic_off"
        if semantic_off:
            semantic_vector = torch.zeros_like(semantic_vector)
        raw_residual = self.feature_projection(semantic_vector)
        residual_norm = raw_residual.norm(dim=-1, keepdim=True)
        target_norm = h[query_ids].norm(dim=-1, keepdim=True)
        normalized_residual = raw_residual * target_norm / residual_norm.clamp_min(1e-8)
        if semantic_off:
            normalized_residual = torch.zeros_like(normalized_residual)
        feature_residual = torch.zeros_like(h).index_copy(0, query_ids, normalized_residual)
        fusion_off = intervention in {"feature_residual_off", "module_off"}
        scale = self.fusion_lambda if not fusion_off else self.fusion_lambda * 0.0
        h_fused = h if fusion_off else h + scale * feature_residual

        if training:
            query_support_pos = torch.searchsorted(support_ids, query_ids)
            safe = query_support_pos.clamp_max(len(support_ids) - 1)
            legal = (query_support_pos < len(support_ids)) & (support_ids[safe] == query_ids)
            if not bool(legal.all()):
                raise RuntimeError("training auxiliary queries must all be legal support patients")
            query_labels = labels[query_ids]
            semantic_loss = F.cross_entropy(class_evidence, query_labels, weight=class_weights)
        else:
            semantic_loss = h.new_zeros(())
        weighted = {
            "sem": 0.02 * semantic_loss,
            "het": 0.02 * heterogeneity,
            "site": 0.01 * site_loss,
        }
        losses = {
            "semantic_raw": semantic_loss,
            "heterogeneity_raw": heterogeneity,
            "site_raw": site_loss,
            **{f"heterogeneity_{key}": value for key, value in heterogeneity_parts.items()},
            "sem_weighted": weighted["sem"],
            "het_weighted": weighted["het"],
            "site_weighted": weighted["site"],
        }
        return h_fused, {
            "h_original": h,
            "h_fused": h_fused,
            "feature_residual": feature_residual,
            "normalized_residual": normalized_residual,
            "semantic_vector": semantic_vector,
            "semantic_loss": semantic_loss,
            "class_evidence": class_evidence,
            "attention": attention,
            "beta": aggregates["beta"],
            "rho_support": aggregates["rho_support"],
            "support_site_weight": aggregates["support_site_weight"],
            "proto_z": aggregates["proto_z"],
            "proto_q": aggregates["proto_q"],
            "prototype_counts": aggregates["counts"],
            "base_counts": aggregates["base_counts"],
            "loo_removed_mass": aggregates["loo_removed_mass"],
            "is_support_query": aggregates["is_support_query"],
            "support_ids": support_ids,
            "query_ids": query_ids,
            "support_labels": support_y,
            "support_sites": site_ids[support_ids],
            "class_prompt_indices": [x for x in prompt_indices],
            "prompt_coords": prompt_coords,
            "losses": losses,
            "site_info": site_info,
            "lambda": scale,
            "raw_residual_norm": residual_norm,
            "target_feature_norm": target_norm,
            "heterogeneity_parts": heterogeneity_parts,
        }


def initialize(text_coords, prompt_classes, k: int, prompt_source: str, fold: int, device):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(781000 + int(fold))
        module = SHDCPSS(text_coords, prompt_classes, k, prompt_source)
    module = module.to(device)
    return module

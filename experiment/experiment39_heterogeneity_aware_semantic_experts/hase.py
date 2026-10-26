"""HASE-MoE modules and fold-local, auditable connectivity controls."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


N_EXPERTS = 4
N_CORR = 256
CORR_OFFSET = 64
MECH_DIM = 32
ADAPTER_DIM = 64
ROUTER_DIM = 32
AGE_DIM = 26
SEX_DIM = 2
QC_DIM = 16
EPS = 1e-8
UNVERIFIED_FEATURES = (195, 225, 232, 236)


def train_median_mad(corr: np.ndarray, train_index: np.ndarray):
    """Fit robust reference statistics using only the outer-training rows."""
    train = np.asarray(corr, dtype=np.float64)[np.asarray(train_index, dtype=np.int64)]
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    scale = 1.4826 * mad + EPS
    return median.astype(np.float32), scale.astype(np.float32)


def true_masks(rows):
    masks = np.zeros((N_EXPERTS, N_CORR), dtype=np.float32)
    for row in rows:
        idx = int(row["correlation_feature_index"])
        if row["verified"].lower() != "true":
            continue
        for k, name in enumerate(("D1", "D2", "D3", "D4")):
            if row[name] == "1":
                masks[k, idx] = 1.0
    return masks


def wrong_masks(real_masks: np.ndarray, seed: int = 39):
    """Permute only anatomically verified candidate features."""
    allowed = np.setdiff1d(np.arange(N_CORR), np.asarray(UNVERIFIED_FEATURES))
    permutation = np.arange(N_CORR, dtype=np.int64)
    permutation[allowed] = np.random.default_rng(seed).permutation(allowed)
    wrong = real_masks[:, permutation].copy()
    return wrong, permutation


def data_driven_masks(corr: np.ndarray, train_index: np.ndarray, shape_masks: np.ndarray):
    """Unsupervised correlation-profile clustering with frozen mask-shape counts.

    The semantic masks contribute only the four cardinalities and their 16
    membership-pattern counts. Feature identities are reassigned by fold-local
    unsupervised clustering of training-only FC correlation profiles.
    """
    train = np.asarray(corr, dtype=np.float64)[np.asarray(train_index, dtype=np.int64)]
    allowed = np.setdiff1d(np.arange(N_CORR), np.asarray(UNVERIFIED_FEATURES))
    profile = np.corrcoef(train, rowvar=False)
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    profile = profile[np.ix_(allowed, allowed)]
    np.fill_diagonal(profile, 0.0)
    model = KMeans(n_clusters=N_EXPERTS, n_init=10, random_state=39, algorithm="lloyd")
    model.fit(profile)
    distances = model.transform(profile)
    affinity = -distances

    # Canonicalize unsupervised cluster labels so runs do not inherit KMeans' label order.
    center_keys = []
    for k, center in enumerate(model.cluster_centers_):
        center_keys.append((int(allowed[np.argmax(center)]), float(np.linalg.norm(center)), k))
    order = [item[2] for item in sorted(center_keys)]
    affinity = affinity[:, order]
    affinity = (affinity - affinity.mean(axis=0, keepdims=True)) / (
        affinity.std(axis=0, keepdims=True) + EPS
    )

    patterns = np.zeros(len(allowed), dtype=np.int64)
    for k in range(N_EXPERTS):
        patterns |= (shape_masks[k, allowed].astype(np.int64) << k)
    counts = np.bincount(patterns, minlength=1 << N_EXPERTS)

    slots = []
    for pattern, count in enumerate(counts):
        bits = np.array([(pattern >> k) & 1 for k in range(N_EXPERTS)], dtype=np.float64)
        score = affinity @ (2.0 * bits - 1.0)
        slots.extend((float(score[j]), int(j), int(pattern)) for j in range(len(allowed)))
    slots.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned = np.full(len(allowed), -1, dtype=np.int64)
    used = np.zeros_like(counts)
    for _, feature, pattern in slots:
        if assigned[feature] != -1 or used[pattern] >= counts[pattern]:
            continue
        assigned[feature] = pattern
        used[pattern] += 1
        if np.all(used == counts):
            break
    if np.any(assigned < 0) or not np.array_equal(used, counts):
        raise RuntimeError("Unable to assign fold-local data-driven FC masks")

    masks = np.zeros_like(shape_masks)
    for k in range(N_EXPERTS):
        masks[k, allowed] = ((assigned >> k) & 1).astype(np.float32)
    return masks, {
        "cluster_centroid_feature_ids": [int(item[0]) for item in sorted(center_keys)],
        "pattern_counts": counts.tolist(),
        "candidate_feature_indices": allowed.astype(int).tolist(),
        "n_train": int(len(train_index)),
    }


def no_fixed_point_derangement(indices, seed: int):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("Patient mismatch requires at least two rows per split")
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(indices)
    shift = int(rng.integers(1, len(ordered)))
    sources = np.roll(ordered, shift)
    target_to_source = dict(zip(ordered.tolist(), sources.tolist()))
    mapped = np.asarray([target_to_source[int(index)] for index in indices], dtype=np.int64)
    if np.any(mapped == indices):
        raise RuntimeError("Patient derangement unexpectedly contains a fixed point")
    return mapped


def mismatch_sources(n_subjects, train_index, test_index, fold, enabled):
    source_rows = np.arange(n_subjects, dtype=np.int64)
    if not enabled:
        return source_rows
    source_rows[train_index] = no_fixed_point_derangement(train_index, 390700 + fold)
    source_rows[test_index] = no_fixed_point_derangement(test_index, 391700 + fold)
    if not np.array_equal(np.sort(source_rows[train_index]), np.sort(train_index)):
        raise RuntimeError("Training mismatch crossed the outer split")
    if not np.array_equal(np.sort(source_rows[test_index]), np.sort(test_index)):
        raise RuntimeError("Test mismatch crossed the outer split")
    return source_rows


class HASEMoE(nn.Module):
    """Four independent FC encoders, patient-conditioned router and adapters."""

    def __init__(self, masks, median, scale, source_rows, route_flags):
        super().__init__()
        masks = torch.as_tensor(masks, dtype=torch.float32)
        self.register_buffer("masks", masks)
        self.register_buffer("median", torch.as_tensor(median, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("source_rows", torch.as_tensor(source_rows, dtype=torch.long))
        self.route_flags = dict(route_flags)

        self.fc_encoder = nn.ModuleList(
            nn.Linear(2 * N_CORR, MECH_DIM, bias=False) for _ in range(N_EXPERTS)
        )
        self.expert_code = nn.Parameter(torch.empty(N_EXPERTS, 8))
        self.router_hidden = nn.Linear(
            4 * 36 + MECH_DIM + AGE_DIM + SEX_DIM + QC_DIM + 8,
            ROUTER_DIM,
            bias=False,
        )
        self.router_output = nn.Linear(ROUTER_DIM, 1, bias=False)
        self.adapter_hidden = nn.ModuleList(
            nn.Linear(4 * 36 + MECH_DIM, ADAPTER_DIM, bias=False)
            for _ in range(N_EXPERTS)
        )
        self.adapter_output = nn.ModuleList(
            nn.Linear(ADAPTER_DIM, 4 * 36, bias=False) for _ in range(N_EXPERTS)
        )
        nn.init.normal_(self.expert_code, mean=0.0, std=0.05)
        for layer in self.adapter_output:
            nn.init.zeros_(layer.weight)

    def _responses(self, raw, mask_override=None, source_rows=None):
        corr = raw[:, CORR_OFFSET : CORR_OFFSET + N_CORR]
        rows = self.source_rows if source_rows is None else source_rows
        source_corr = corr[rows]
        z = (source_corr - self.median) / self.scale
        masks = self.masks if mask_override is None else mask_override
        response = []
        for k in range(N_EXPERTS):
            mask = masks[k].to(device=raw.device, dtype=raw.dtype)
            pair = torch.cat((source_corr * mask, z * mask), dim=-1)
            response.append(torch.tanh(self.fc_encoder[k](pair)))
        return torch.stack(response, dim=1), source_corr, z, masks

    def forward(
        self,
        hidden,
        raw,
        *,
        mask_override=None,
        route_uniform=False,
        expert_keep=None,
        route_flags=None,
        source_rows=None,
    ):
        response, source_corr, z, masks = self._responses(raw, mask_override, source_rows)
        n = hidden.shape[0]
        hflat = hidden.reshape(n, -1)
        pheno = torch.cat((raw[:, :AGE_DIM], raw[:, 46:48]), dim=-1)
        qc = raw[:, 48:64]
        flags = self.route_flags if route_flags is None else route_flags

        logits = []
        deltas = []
        for k in range(N_EXPERTS):
            code = self.expert_code[k].expand(n, -1)
            pieces = [hflat]
            pieces.append(response[:, k] if flags.get("fc", True) else response[:, k].new_zeros(n, MECH_DIM))
            pieces.append(pheno if flags.get("pheno", True) else pheno.new_zeros(n, AGE_DIM + SEX_DIM))
            pieces.append(qc if flags.get("qc", True) else qc.new_zeros(n, QC_DIM))
            router_input = torch.cat(pieces + [code], dim=-1)
            route_hidden = torch.tanh(self.router_hidden(router_input))
            logits.append(self.router_output(route_hidden).squeeze(-1))

            adapter_input = torch.cat((hflat, response[:, k]), dim=-1)
            adapted = torch.tanh(self.adapter_hidden[k](adapter_input))
            deltas.append(self.adapter_output[k](adapted).view(n, 4, 36))

        logits = torch.stack(logits, dim=-1)
        weights = torch.softmax(logits, dim=-1)
        if route_uniform:
            weights = torch.full_like(weights, 1.0 / N_EXPERTS)
        expert_delta = torch.stack(deltas, dim=1)
        if expert_keep is not None:
            keep = torch.as_tensor(expert_keep, dtype=expert_delta.dtype, device=expert_delta.device)
            expert_delta = expert_delta * keep.view(1, N_EXPERTS, 1, 1)

        fused = torch.sum(weights[:, :, None, None] * expert_delta, dim=1)
        base_norm = torch.linalg.vector_norm(hidden.reshape(n, -1), dim=-1)
        delta_norm = torch.linalg.vector_norm(fused.reshape(n, -1), dim=-1)
        scale = torch.clamp(0.1 * base_norm / (delta_norm + EPS), max=1.0)
        bounded = fused * scale[:, None, None]
        updated = hidden + bounded
        return updated, {
            "weights": weights,
            "logits": logits,
            "response": response,
            "expert_delta": expert_delta,
            "unbounded_delta": fused,
            "bounded_delta": bounded,
            "residual_ratio": torch.linalg.vector_norm(bounded.reshape(n, -1), dim=-1)
            / (base_norm + EPS),
            "source_corr": source_corr,
            "source_z": z,
            "masks": masks,
        }


class CapacityMatchedAdapter(nn.Module):
    """One unmasked adapter; width 183 matches HASE trainable capacity to <0.3%."""

    def __init__(self, width=183):
        super().__init__()
        self.hidden = nn.Linear(4 * 36 + 2 * N_CORR + AGE_DIM + SEX_DIM + QC_DIM, width, bias=False)
        self.output = nn.Linear(width, 4 * 36, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, raw, source_rows, median, scale):
        n = hidden.shape[0]
        corr = raw[:, CORR_OFFSET : CORR_OFFSET + N_CORR][source_rows]
        z = (corr - median) / scale
        pheno = torch.cat((raw[:, :AGE_DIM], raw[:, 46:48]), dim=-1)
        qc = raw[:, 48:64]
        values = torch.cat((hidden.reshape(n, -1), corr, z, pheno, qc), dim=-1)
        delta = self.output(torch.tanh(self.hidden(values))).view(n, 4, 36)
        base_norm = torch.linalg.vector_norm(hidden.reshape(n, -1), dim=-1)
        delta_norm = torch.linalg.vector_norm(delta.reshape(n, -1), dim=-1)
        factor = torch.clamp(0.1 * base_norm / (delta_norm + EPS), max=1.0)
        bounded = delta * factor[:, None, None]
        updated = hidden + bounded
        return updated, {
            "weights": torch.ones((n, 1), device=hidden.device, dtype=hidden.dtype),
            "response": values,
            "expert_delta": delta[:, None],
            "unbounded_delta": delta,
            "bounded_delta": bounded,
            "residual_ratio": torch.linalg.vector_norm(bounded.reshape(n, -1), dim=-1)
            / (base_norm + EPS),
            "source_corr": corr,
            "source_z": z,
            "masks": None,
        }


def specialization_loss(expert_delta, train_index):
    """Mean squared cosine between centered expert residuals; zero-safe."""
    if expert_delta.shape[1] < 2:
        return expert_delta.new_zeros(())
    idx = torch.as_tensor(train_index, dtype=torch.long, device=expert_delta.device)
    centered = expert_delta[idx].flatten(start_dim=2)
    centered = centered - centered.mean(dim=0, keepdim=True)
    values = []
    for k in range(centered.shape[1]):
        for j in range(k + 1, centered.shape[1]):
            left = centered[:, k].reshape(-1)
            right = centered[:, j].reshape(-1)
            cosine = F.cosine_similarity(left, right, dim=0, eps=1e-8)
            values.append(cosine.square())
    if not values:
        return expert_delta.new_zeros(())
    return torch.stack(values).mean()


def parameter_audit(module):
    registered = sum(parameter.numel() for parameter in module.parameters())
    return {"registered": registered, "tensors": sum(1 for _ in module.parameters())}

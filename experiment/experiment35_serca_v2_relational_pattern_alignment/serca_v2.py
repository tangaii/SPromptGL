"""Disease-semantic relational-pattern alignment for Experiment 35."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIGS = (
    "E0_B0A", "E1_SERCA34_CAL", "E2_NUMERIC_GLOBAL", "E3_FIRST_ORDER",
    "E4_ALL_PAIR", "E5_RANDOM_RULE", "E6_PATIENT_SHUFFLED",
    "E7_ORIGINAL_EDGES", "E8_CORR_ONLY", "E9_AUX_ONLY",
    "E10_DIRECT_MATCHED", "E11_SERCA_V2_FULL",
)
RELATIONS = ("PHENO", "ANAT", "FUNC", "CORR")
DOMAIN_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TRUE_RULES = ((1, 3), (2, 3))
RANDOM_RULES = ((0, 1), (0, 3))
SIGNATURE_DIM = 24
HIDDEN_DIM = 36
ETA = 0.05
SEMANTIC_TOPK = 8
LAMBDA_SEM = 0.25
EDGE_MODULATION = 0.25


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def make_semantic_neighbors(kernel: np.ndarray, train_indices: np.ndarray,
                            topk: int = SEMANTIC_TOPK) -> tuple[np.ndarray, dict]:
    """Build directed recipient-to-training-source top-k edges."""
    kernel = np.asarray(kernel, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    n = kernel.shape[0]
    assert kernel.shape == (n, n) and np.isfinite(kernel).all()
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_indices] = True
    rows, cols, weights = [], [], []
    candidates = train_indices
    for target in range(n):
        available = candidates[candidates != target]
        if not len(available):
            continue
        count = min(topk, len(available))
        values = kernel[target, available]
        selected = np.argpartition(values, -count)[-count:]
        selected = selected[np.argsort(values[selected])[::-1]]
        source_ids = available[selected]
        rows.extend([target] * count)
        cols.extend(source_ids.tolist())
        weights.extend(values[selected].tolist())
    adjacency = np.zeros((n, n), dtype=np.float32)
    if rows:
        adjacency[np.asarray(rows), np.asarray(cols)] = np.asarray(weights, dtype=np.float32)
    assert np.count_nonzero(adjacency[~train_mask]) <= n * topk
    assert np.count_nonzero(adjacency[:, ~train_mask]) == 0
    assert np.count_nonzero(np.diag(adjacency)) == 0
    return adjacency, dict(
        directed_edge_count=int(np.count_nonzero(adjacency)),
        train_source_only=bool(np.count_nonzero(adjacency[:, ~train_mask]) == 0),
        test_rows_with_edges=int(np.count_nonzero(adjacency[~train_mask].sum(axis=1))),
        train_rows_with_edges=int(np.count_nonzero(adjacency[train_mask].sum(axis=1))),
        topk=int(topk),
    )


class SeRCAV2(nn.Module):
    def __init__(self, signature: np.ndarray, signature_mask: np.ndarray,
                 semantic_kernel: np.ndarray, kernel_center: float,
                 semantic_neighbors: np.ndarray, relation_mask: np.ndarray,
                 direct_message: bool = False):
        super().__init__()
        signature = torch.as_tensor(signature, dtype=torch.float32)
        signature_mask = torch.as_tensor(signature_mask, dtype=torch.bool)
        semantic_kernel = torch.as_tensor(semantic_kernel, dtype=torch.float32)
        semantic_neighbors = torch.as_tensor(semantic_neighbors, dtype=torch.float32)
        relation_mask = torch.as_tensor(relation_mask, dtype=torch.bool)
        assert signature.shape == (871, SIGNATURE_DIM)
        assert signature_mask.shape == (SIGNATURE_DIM,)
        assert semantic_kernel.shape == (871, 871)
        assert semantic_neighbors.shape == (871, 871)
        assert relation_mask.shape == (4,) and relation_mask.any()
        self.register_buffer("signature", signature)
        self.register_buffer("signature_mask", signature_mask)
        self.register_buffer("semantic_kernel", semantic_kernel)
        self.kernel_center = float(kernel_center)
        self.register_buffer("semantic_neighbors", semantic_neighbors)
        self.register_buffer("relation_mask", relation_mask)
        self.value_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(width, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM))
            for width in (48, 6, 10, 256)
        ])
        self.router = nn.Linear(SIGNATURE_DIM + 4, 5)
        self.alpha = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.direct_message = bool(direct_message)
        self.source_override = None
        self.collect_diagnostics = False
        self.last_diagnostics = None

    def source_weights(self, strength: torch.Tensor) -> torch.Tensor:
        signature = self.signature * self.signature_mask.to(self.signature.dtype)
        logits = self.router(torch.cat([signature, strength], dim=-1))
        unavailable = ~self.relation_mask
        logits[:, 1:] = logits[:, 1:].masked_fill(unavailable[None, :], -torch.inf)
        weights = F.softmax(logits, dim=-1)
        if self.source_override is not None:
            disabled = torch.as_tensor(self.source_override, dtype=torch.bool, device=logits.device)
            weights = torch.cat([weights[:, :1],
                                 weights[:, 1:] * (~disabled)[None, :].to(weights.dtype)], dim=-1)
        return weights

    def forward(self, hidden: torch.Tensor, payload: dict) -> torch.Tensor:
        raw_inputs = payload["raw_inputs"]
        raw_adjs = payload["raw_adjs"]
        train_indices = payload["train_indices"]
        assert hidden.shape == (871, 4, HIDDEN_DIM)
        assert len(raw_adjs) == 4 and raw_inputs.shape == (871, 320)
        assert all(adj.shape == (871, 871) for adj in raw_adjs)
        train_mask = torch.zeros(871, dtype=torch.bool, device=hidden.device)
        train_mask[torch.as_tensor(train_indices, dtype=torch.long, device=hidden.device)] = True
        values = []
        for projection, part in zip(self.value_projections,
                                    torch.split(raw_inputs, (48, 6, 10, 256), dim=-1)):
            values.append(projection(part))
        ordinary, conditioned, deltas = [], [], []
        edge_audit = []
        kernel = self.semantic_kernel.to(device=hidden.device, dtype=hidden.dtype)
        sparse_kernel = self.semantic_neighbors.to(device=hidden.device, dtype=hidden.dtype)
        identity = torch.eye(871, device=hidden.device, dtype=hidden.dtype)
        for relation, (raw, value) in enumerate(zip(raw_adjs, values)):
            nonnegative_raw = raw.clamp_min(0)
            base_weight = nonnegative_raw + identity
            base = _row_normalize(base_weight)
            off = nonnegative_raw
            off = off - torch.diag(torch.diagonal(off))
            centered = kernel - self.kernel_center
            modulation = (1.0 + EDGE_MODULATION * centered).clamp(0.75, 1.25)
            diagonal = torch.diag(torch.diagonal(nonnegative_raw))
            semantic_weight = diagonal + off * modulation + identity + LAMBDA_SEM * sparse_kernel
            semantic = _row_normalize(semantic_weight)
            base_message = base @ value
            semantic_message = semantic @ value
            ordinary.append(base_message)
            conditioned.append(semantic_message)
            deltas.append(semantic_message if self.direct_message else semantic_message - base_message)
            raw_edges = off > 0
            overlap = (raw_edges & (sparse_kernel > 0)).sum().float() / raw_edges.sum().clamp_min(1)
            relative = (off * (modulation - 1.0)).abs().sum() / off.abs().sum().clamp_min(1e-12)
            edge_audit.append((int(raw_edges.sum().item()), float(overlap.item()), float(relative.item()),
                               float((semantic.sum(dim=1) - 1.0).abs().mean().item())))
        source_delta = torch.stack(deltas, dim=1)
        strength = source_delta.detach().norm(dim=-1)
        weights = self.source_weights(strength)
        selected_weights = weights[:, 1:]
        mixed = (source_delta * selected_weights.unsqueeze(-1)).sum(dim=1)
        train_rms_h = hidden[train_mask].square().mean().sqrt()
        train_rms_r = mixed[train_mask].square().mean().sqrt()
        scale = (train_rms_h / torch.maximum(train_rms_r, 0.1 * train_rms_h).clamp_min(1e-8)).detach()
        calibrated = scale * mixed
        raw_delta = ETA * torch.tanh(self.alpha)[None, :, None] * calibrated[:, None, :]
        hidden_norm = hidden.norm(dim=-1, keepdim=True)
        delta_norm = raw_delta.norm(dim=-1, keepdim=True)
        cap = 0.05 * hidden_norm
        ratio = torch.minimum(torch.ones_like(delta_norm), cap / delta_norm.clamp_min(1e-12))
        delta = raw_delta * ratio
        result = hidden + delta
        if self.collect_diagnostics:
            contribution = source_delta * selected_weights.unsqueeze(-1)
            self.last_diagnostics = dict(
                alpha=self.alpha.detach().cpu().tolist(),
                route_mean=weights.detach().mean(dim=0).cpu().tolist(),
                source_weights_mean=selected_weights.detach().mean(dim=0).cpu().tolist(),
                null_weight_mean=float(weights[:, 0].detach().mean().item()),
                source_message_rms=source_delta.detach().square().mean(dim=(0, 2)).sqrt().cpu().tolist(),
                source_contribution_rms=contribution.detach().square().mean(dim=(0, 2)).sqrt().cpu().tolist(),
                raw_message_rms=float(train_rms_r.detach().item()),
                target_rms=float(train_rms_h.detach().item()),
                calibration_scale=float(scale.item()),
                calibrated_message_rms=float(calibrated[train_mask].detach().square().mean().sqrt().item()),
                mean_token_residual_ratio=ratio.detach().squeeze(-1).mean(dim=0).cpu().tolist(),
                max_token_residual_ratio=ratio.detach().squeeze(-1).max(dim=0).values.cpu().tolist(),
                max_token_residual_ratio_observed=float(
                    (delta.norm(dim=-1) / hidden_norm.squeeze(-1).clamp_min(1e-12)).max().item()),
                semantic_edge_count=int(torch.count_nonzero(sparse_kernel).item()),
                raw_relation_edge_count=[item[0] for item in edge_audit],
                semantic_neighbor_overlap_fraction=[item[1] for item in edge_audit],
                existing_edge_relative_reweight=[item[2] for item in edge_audit],
                semantic_row_normalization_error=[item[3] for item in edge_audit],
            )
        return result

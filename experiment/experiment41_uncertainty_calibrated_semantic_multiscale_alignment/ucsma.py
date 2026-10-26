"""Small-cohort normative references and the UCSMA residual adapter."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def robust_scale(values: np.ndarray, axis: int = 0, floor: float = 0.05) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    scale = 1.4826 * np.median(np.abs(values - median), axis=axis)
    return np.maximum(scale, floor)


def ridge_fit(design: np.ndarray, target: np.ndarray, alpha: float = 10.0):
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    inverse = np.linalg.pinv(gram, hermitian=True)
    coef = inverse @ design.T @ target
    return coef, inverse


def predict_with_uncertainty(design: np.ndarray, coef: np.ndarray,
                             inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = design @ coef
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design, optimize=True)
    return prediction, np.maximum(leverage, 0.0)


def fold_normative_features(x: np.ndarray, train_idx: np.ndarray,
                            y: np.ndarray,
                            edge_to_roi: np.ndarray, concept_roi_masks: np.ndarray,
                            concept_edge_masks: np.ndarray,
                            family_roi_masks: np.ndarray,
                            family_edge_masks: np.ndarray,
                            mode: str = "conditional",
                            alpha: float = 10.0) -> dict[str, np.ndarray | dict]:
    """Return train-OOF and outer-test normative and multiscale evidence."""
    from sklearn.model_selection import KFold

    n = x.shape[0]
    corr = x[:, 64:320].astype(np.float64, copy=False)
    phen = x[:, :48]
    anat_func = x[:, 48:64]
    known_edge = (edge_to_roi >= 0).all(axis=1)
    mapped_edge_ids = np.flatnonzero(known_edge)

    if mode == "global":
        categorical = np.zeros((n, 0), dtype=np.float64)
        design = np.ones((n, 1), dtype=np.float64)
    else:
        age_code = np.argmax(phen[:, :26], axis=1)
        sex_code = np.argmax(phen[:, 46:48], axis=1)
        site_code = np.argmax(phen[:, 26:46], axis=1)
        age_oh = np.eye(26, dtype=np.float64)[age_code]
        sex_oh = np.eye(2, dtype=np.float64)[sex_code]
        site_oh = np.eye(20, dtype=np.float64)[site_code]
        # ANAT6 precedes FUNC10; mean_fd is FUNC index 6, global column 60.
        mean_fd = x[:, 48 + 6 + 6]
        fd_center = np.median(mean_fd[train_idx])
        fd_scale = float(robust_scale(mean_fd[train_idx, None], floor=0.05)[0])
        fd = np.clip((mean_fd - fd_center) / fd_scale, -4.0, 4.0)[:, None]
        categorical = np.concatenate([age_oh, sex_oh, site_oh], axis=1)
        design = np.concatenate([np.ones((n, 1)), categorical, fd], axis=1)

    inner = list(KFold(n_splits=5, shuffle=True, random_state=41041).split(train_idx))
    oof_mean = np.full((n, 256), np.nan, dtype=np.float64)
    oof_lev = np.full(n, np.nan, dtype=np.float64)
    oof_support = np.full(n, np.nan, dtype=np.float64)
    train_td = train_idx[y[train_idx] == 1]
    for fit_pos, held_pos in inner:
        inner_fit_idx = train_idx[fit_pos]
        held_idx = train_idx[held_pos]
        reference_idx = inner_fit_idx[y[inner_fit_idx] == 1]
        if len(reference_idx) < 80:
            raise ValueError("Inner TD reference has insufficient support")
        coef, inverse = ridge_fit(design[reference_idx], corr[reference_idx], alpha)
        pred, leverage = predict_with_uncertainty(design[held_idx], coef, inverse)
        oof_mean[held_idx] = pred
        oof_lev[held_idx] = leverage
        oof_support[held_idx] = _condition_support(
            phen, reference_idx, held_idx, mode)

    residual_oof = corr[train_td] - oof_mean[train_td]
    sigma = robust_scale(residual_oof, axis=0, floor=0.05)
    z_oof = np.clip(residual_oof / sigma[None, :], -4.0, 4.0)
    thresholds = np.quantile(np.abs(z_oof), 0.95, axis=0)
    # The edge threshold has a robust lower bound for a degenerate column.
    thresholds = np.maximum(thresholds, 0.5)

    outer_coef, outer_inverse = ridge_fit(design[train_td], corr[train_td], alpha)
    outer_test_idx = np.setdiff1d(np.arange(n), train_idx, assume_unique=False)
    outer_pred, outer_lev = predict_with_uncertainty(
        design[outer_test_idx], outer_coef, outer_inverse)
    oof_mean[outer_test_idx] = outer_pred
    oof_lev[outer_test_idx] = outer_lev
    oof_support[outer_test_idx] = _condition_support(
        phen, train_td, outer_test_idx, mode)
    if not np.isfinite(oof_mean).all() or not np.isfinite(oof_lev).all():
        raise FloatingPointError("Normative mean or leverage is non-finite")

    # Acquisition quality is a reliability proxy, not disease evidence.
    q_center = np.median(anat_func[train_idx], axis=0)
    q_scale = robust_scale(anat_func[train_idx], axis=0, floor=0.05)
    q_z = np.clip(np.abs((anat_func - q_center) / q_scale), 0.0, 4.0)
    qc_support = np.exp(-0.15 * q_z.mean(axis=1))
    reference_support = np.clip(
        oof_support / (1.0 + np.sqrt(np.maximum(oof_lev, 0.0))), 0.0, 1.0)
    ref_edge = np.broadcast_to(
        (reference_support * qc_support)[:, None], (n, 256)).copy()
    z = np.clip((corr - oof_mean) / sigma[None, :], -4.0, 4.0)
    edge_features = np.stack([
        np.maximum(z, 0.0), np.maximum(-z, 0.0),
        (z >= thresholds[None, :]).astype(np.float64),
        (z <= -thresholds[None, :]).astype(np.float64),
        np.broadcast_to(ref_edge, (n, 256)),
        np.broadcast_to((1.0 - reference_support)[:, None], (n, 256)),
    ], axis=-1)

    roi_degree = np.zeros(111, dtype=np.int64)
    incident: list[list[int]] = [[] for _ in range(111)]
    for edge_idx in np.flatnonzero(known_edge):
        left, right = map(int, edge_to_roi[edge_idx])
        incident[left].append(int(edge_idx))
        incident[right].append(int(edge_idx))
        roi_degree[left] += 1
        roi_degree[right] += 1

    # Propagate the existing degree/(degree+3) ROI reliability to semantic
    # edge pooling. Unknown endpoints have zero anatomical support.
    degree_reliability = roi_degree / (roi_degree + 3.0)
    endpoint_coverage = np.zeros(256, dtype=np.float64)
    for edge_idx in np.flatnonzero(known_edge):
        left, right = edge_to_roi[edge_idx]
        endpoint_coverage[edge_idx] = np.sqrt(
            degree_reliability[left] * degree_reliability[right])
    alignment_edge_support = ref_edge * endpoint_coverage[None, :]

    roi_features = np.zeros((n, 111, 6), dtype=np.float64)
    for roi, edges in enumerate(incident):
        if not edges:
            continue
        edge_idx = np.asarray(edges, dtype=np.int64)
        weights = ref_edge[:, edge_idx]
        denom = np.maximum(weights.sum(axis=1), 1e-12)
        roi_features[:, roi, 0] = (weights * edge_features[:, edge_idx, 0]).sum(1) / denom
        roi_features[:, roi, 1] = (weights * edge_features[:, edge_idx, 1]).sum(1) / denom
        roi_features[:, roi, 2] = (weights * edge_features[:, edge_idx, 2]).sum(1) / denom
        roi_features[:, roi, 3] = (weights * edge_features[:, edge_idx, 3]).sum(1) / denom
        coverage_adjust = roi_degree[roi] / (roi_degree[roi] + 3.0)
        roi_features[:, roi, 4] = coverage_adjust * weights.mean(1)
        roi_features[:, roi, 5] = 1.0 - weights.mean(1)

    def pool_edge_mask(edge_mask: np.ndarray) -> np.ndarray:
        edge_idx = np.flatnonzero(edge_mask & known_edge)
        if len(edge_idx) == 0:
            return np.zeros((n, 6), dtype=np.float64)
        weights = alignment_edge_support[:, edge_idx]
        denom = np.maximum(weights.sum(1), 1e-12)
        pooled = np.stack([
            (weights * edge_features[:, edge_idx, c]).sum(1) / denom
            for c in range(4)
        ], axis=1)
        degree_scale = len(edge_idx) / (len(edge_idx) + 3.0)
        return np.column_stack([pooled, degree_scale * weights.mean(1),
                                1.0 - weights.mean(1)])

    family_features = np.stack([pool_edge_mask(mask) for mask in family_edge_masks], axis=1)
    concept_features = np.zeros((n, len(concept_roi_masks), 6), dtype=np.float64)
    for concept_id, roi_mask in enumerate(concept_roi_masks):
        if concept_id < len(concept_edge_masks) and concept_edge_masks[concept_id].any():
            concept_features[:, concept_id] = pool_edge_mask(concept_edge_masks[concept_id])
            continue
        roi_idx = np.flatnonzero(roi_mask)
        if not len(roi_idx):
            continue
        selected = roi_features[:, roi_idx]
        weights = selected[:, :, 4]
        denom = np.maximum(weights.sum(1), 1e-12)
        concept_features[:, concept_id, :4] = (
            selected[:, :, :4] * weights[:, :, None]).sum(1) / denom[:, None]
        concept_features[:, concept_id, 4] = weights.mean(1)
        concept_features[:, concept_id, 5] = (
            selected[:, :, 5] * np.maximum(weights, 1e-12)).sum(1) / denom

    thresholds_by_concept = np.zeros(len(concept_roi_masks), dtype=np.float64)
    td_oof_rows = train_td
    for concept_id in range(len(concept_roi_masks)):
        signal = np.maximum(concept_features[td_oof_rows, concept_id, 2],
                            concept_features[td_oof_rows, concept_id, 3])
        thresholds_by_concept[concept_id] = max(float(np.quantile(signal, .90)), .02)

    z_clipped = np.abs((corr - oof_mean) / sigma[None, :]) > 4.0
    return {
        "z": z.astype(np.float32),
        "edge_features": edge_features.astype(np.float32),
        "roi_features": roi_features.astype(np.float32),
        "family_features": family_features.astype(np.float32),
        "concept_features": concept_features.astype(np.float32),
        "reference_support": reference_support.astype(np.float32),
        "qc_support": qc_support.astype(np.float32),
        "edge_support": ref_edge.astype(np.float32),
        "alignment_edge_support": alignment_edge_support.astype(np.float32),
        "endpoint_coverage": endpoint_coverage.astype(np.float32),
        "known_edge_mask": known_edge.astype(bool),
        "sigma": sigma.astype(np.float32),
        "edge_thresholds": thresholds.astype(np.float32),
        "concept_thresholds": thresholds_by_concept.astype(np.float32),
        "clip_fraction": float(z_clipped.mean()),
        "td_oof_abs_z_q95": float(np.quantile(np.abs(z_oof), .95)),
        "td_oof_abs_z_median": float(np.median(np.abs(z_oof))),
        "conditional_support_mean": float(reference_support[train_idx].mean()),
        "qc_support_mean": float(qc_support[train_idx].mean()),
        "max_leverage": float(np.max(oof_lev)),
        "td_oof_mse": float(np.mean(residual_oof ** 2)),
        "td_oof_mae": float(np.mean(np.abs(residual_oof))),
        "mean_fd_global_column": 60,
        "roi_degree": roi_degree,
        "known_edge_ids": mapped_edge_ids,
    }


def _condition_support(phen: np.ndarray, reference_idx: np.ndarray,
                       query_idx: np.ndarray, mode: str) -> np.ndarray:
    if mode == "global":
        return np.ones(len(query_idx), dtype=np.float64)
    age = np.argmax(phen[:, :26], axis=1)
    site = np.argmax(phen[:, 26:46], axis=1)
    sex = np.argmax(phen[:, 46:48], axis=1)
    counts_age = np.bincount(age[reference_idx], minlength=26)
    counts_site = np.bincount(site[reference_idx], minlength=20)
    counts_sex = np.bincount(sex[reference_idx], minlength=2)
    support = (np.minimum(counts_age[age[query_idx]] / 20.0, 1.0) *
               np.minimum(counts_site[site[query_idx]] / 20.0, 1.0) *
               np.minimum(counts_sex[sex[query_idx]] / 50.0, 1.0))
    return np.clip(np.cbrt(support), 0.05, 1.0)


class UCSMAResidual(nn.Module):
    """Zero-initialized residual; smooth per-subject L2 bound is 5% of CAM."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 144):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features: torch.Tensor, cam: torch.Tensor):
        raw = self.output(F.gelu(self.input(features)))
        flat_cam = cam.reshape(cam.shape[0], -1)
        norm = raw.norm(dim=1, keepdim=True)
        cap = 0.05 * flat_cam.norm(dim=1, keepdim=True).clamp_min(1e-8)
        ratio = torch.where(norm > 1e-8,
                            cap * torch.tanh(norm / cap) / norm.clamp_min(1e-8),
                            torch.ones_like(norm))
        residual = raw * ratio
        return (flat_cam + residual).reshape_as(cam), residual


def semantic_mass(concept_features: np.ndarray, eligible: np.ndarray,
                  reject: bool, thresholds: np.ndarray,
                  direction_weight: np.ndarray | None = None,
                  match_scale: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Build empirical-tail evidence mass with an explicit unmatched channel."""
    n, k, width = concept_features.shape
    # slots: positive mean, negative mean, positive tail coverage,
    # negative tail coverage, evidence support, reference reliability.
    p_cov = concept_features[:, :, 2]
    n_cov = concept_features[:, :, 3]
    support = concept_features[:, :, 4]
    if direction_weight is None:
        direction_weight = np.full(k, 0.5, dtype=np.float64)
    direction_weight = np.asarray(direction_weight, dtype=np.float64)
    if direction_weight.shape != (k,):
        raise ValueError("direction_weight must have one value per concept")
    # The literature direction is only a weak population-level prior; observed
    # patient signs remain represented symmetrically in the feature tensor.
    directional = (direction_weight[None, :] * n_cov +
                   (1.0 - direction_weight[None, :]) * p_cov)
    nondirectional = 0.5 * (p_cov + n_cov)
    signed_magnitude = 1.0 - np.exp(-0.25 * np.clip(
        concept_features[:, :, 0] + concept_features[:, :, 1], 0.0, 8.0))
    signal = 0.55 * signed_magnitude + 0.30 * nondirectional + 0.15 * directional
    strength = np.clip((signal - thresholds[None, :]) /
                       np.maximum(0.95 - thresholds[None, :], 0.05), 0.0, 1.0)
    raw = float(match_scale) * strength * support * eligible[None, :]
    if reject:
        denom = 1.0 + raw.sum(axis=1, keepdims=True)
        matched = raw / denom
        unmatched = 1.0 / denom[:, 0]
    else:
        logits = np.log(np.maximum(raw, 1e-6))
        logits[:, ~eligible] = -1e9
        logits -= logits.max(axis=1, keepdims=True)
        matched = np.exp(logits)
        matched /= matched.sum(axis=1, keepdims=True)
        unmatched = np.zeros(n, dtype=np.float64)
    return matched, unmatched


def pool_mask(edge_features: np.ndarray, edge_support: np.ndarray,
              edge_mask: np.ndarray, known_edge: np.ndarray) -> np.ndarray:
    edge_idx = np.flatnonzero(edge_mask & known_edge)
    if len(edge_idx) == 0:
        return np.zeros((edge_features.shape[0], 6), dtype=np.float64)
    weights = edge_support[:, edge_idx]
    denom = np.maximum(weights.sum(1), 1e-12)
    pooled = np.stack([
        (weights * edge_features[:, edge_idx, c]).sum(1) / denom
        for c in range(4)
    ], axis=1)
    degree_scale = len(edge_idx) / (len(edge_idx) + 3.0)
    return np.column_stack([pooled, degree_scale * weights.mean(1),
                            1.0 - weights.mean(1)])


def summarize_concepts(edge_features: np.ndarray, edge_support: np.ndarray,
                       roi_features: np.ndarray, concept_roi_masks: np.ndarray,
                       concept_edge_masks: np.ndarray,
                       known_edge: np.ndarray) -> np.ndarray:
    n = edge_features.shape[0]
    output = np.zeros((n, concept_roi_masks.shape[0], 6), dtype=np.float64)
    for k, roi_mask in enumerate(concept_roi_masks):
        if k < len(concept_edge_masks) and concept_edge_masks[k].any():
            output[:, k] = pool_mask(edge_features, edge_support,
                                     concept_edge_masks[k], known_edge)
            continue
        roi_idx = np.flatnonzero(roi_mask)
        if not len(roi_idx):
            continue
        selected = roi_features[:, roi_idx]
        weights = selected[:, :, 4]
        denom = np.maximum(weights.sum(1), 1e-12)
        output[:, k, :4] = (
            selected[:, :, :4] * weights[:, :, None]).sum(1) / denom[:, None]
        output[:, k, 4] = weights.mean(1)
        output[:, k, 5] = (
            selected[:, :, 5] * np.maximum(weights, 1e-12)).sum(1) / denom
    return output

"""SAGE-Align evidence construction and a small shared residual branch."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def robust_scale(values: np.ndarray, axis=0, floor: float = 0.05) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=axis, keepdims=True)
    scale = 1.4826 * np.median(np.abs(values - median), axis=axis)
    return np.maximum(scale, floor)


def ridge_fit(design: np.ndarray, target: np.ndarray, alpha: float = 10.0):
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    inverse = np.linalg.pinv(design.T @ design + penalty, hermitian=True)
    return inverse @ design.T @ target, inverse


def predict_with_uncertainty(design, coef, inverse):
    pred = design @ coef
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design, optimize=True)
    return pred, np.maximum(leverage, 0.0)


def _condition_support(phen, reference_idx, query_idx, mode):
    if mode == "global":
        return np.ones(len(query_idx), dtype=np.float64)
    age = np.argmax(phen[:, :26], axis=1)
    site = np.argmax(phen[:, 26:46], axis=1)
    sex = np.argmax(phen[:, 46:48], axis=1)
    ca = np.bincount(age[reference_idx], minlength=26)
    cs = np.bincount(site[reference_idx], minlength=20)
    cx = np.bincount(sex[reference_idx], minlength=2)
    support = (np.minimum(ca[age[query_idx]] / 20.0, 1.0) *
               np.minimum(cs[site[query_idx]] / 20.0, 1.0) *
               np.minimum(cx[sex[query_idx]] / 50.0, 1.0))
    return np.clip(np.cbrt(support), 0.05, 1.0)


def _pool(values, weights, mask):
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return np.zeros((values.shape[0], values.shape[-1]), dtype=np.float64)
    w = weights[:, ids]
    denom = np.maximum(w.sum(1), 1e-12)
    return (w[:, :, None] * values[:, ids]).sum(1) / denom[:, None]


def build_normative(x, train_idx, y, endpoints, concept_roi_masks,
                    concept_edge_masks, family_masks,
                    mode="conditional", use_qc=True, alpha=10.0):
    """Fit an outer-fold TD reference and return all multiscale evidence."""
    from sklearn.model_selection import KFold

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    corr = x[:, 64:320]
    phen = x[:, :48]
    quality = x[:, 48:64]
    endpoints = np.asarray(endpoints)
    known = (endpoints >= 0).all(axis=1)
    if mode == "global":
        design = np.ones((n, 1), dtype=np.float64)
    else:
        age = np.argmax(phen[:, :26], axis=1)
        site = np.argmax(phen[:, 26:46], axis=1)
        sex = np.argmax(phen[:, 46:48], axis=1)
        fd = x[:, 60]
        center = np.median(fd[train_idx])
        scale = float(robust_scale(fd[train_idx, None], floor=0.05)[0])
        fd = np.clip((fd - center) / scale, -4.0, 4.0)[:, None]
        design = np.column_stack([np.ones(n), np.eye(26)[age],
                                  np.eye(20)[site], np.eye(2)[sex], fd])

    train_idx = np.asarray(train_idx, dtype=np.int64)
    oof_mean = np.full((n, 256), np.nan)
    oof_lev = np.full(n, np.nan)
    oof_support = np.full(n, np.nan)
    for fit_pos, hold_pos in KFold(n_splits=5, shuffle=True,
                                   random_state=41041).split(train_idx):
        fit = train_idx[fit_pos]
        hold = train_idx[hold_pos]
        ref = fit[y[fit] == 1]
        if len(ref) < 80:
            raise ValueError("insufficient inner TD reference")
        coef, inv = ridge_fit(design[ref], corr[ref], alpha)
        oof_mean[hold], oof_lev[hold] = predict_with_uncertainty(
            design[hold], coef, inv)
        oof_support[hold] = _condition_support(phen, ref, hold, mode)

    td = train_idx[y[train_idx] == 1]
    coef, inv = ridge_fit(design[td], corr[td], alpha)
    test = np.setdiff1d(np.arange(n), train_idx, assume_unique=False)
    oof_mean[test], oof_lev[test] = predict_with_uncertainty(
        design[test], coef, inv)
    oof_support[test] = _condition_support(phen, td, test, mode)
    residual = corr[td] - oof_mean[td]
    sigma = robust_scale(residual, axis=0, floor=0.05)
    z = np.clip((corr - oof_mean) / sigma[None, :], -4.0, 4.0)

    # Verified FUNC quality fields only: DVARS, outlier, mean FD, percent FD.
    qc_local = np.array([9, 10, 12, 14], dtype=np.int64)
    q_center = np.median(quality[train_idx][:, qc_local], axis=0)
    q_scale = robust_scale(quality[train_idx][:, qc_local], floor=0.05)
    q_dist = np.clip(np.abs((quality[:, qc_local] - q_center) / q_scale), 0, 4)
    qc_support = np.exp(-0.20 * q_dist.mean(axis=1)) if use_qc else np.ones(n)
    reference_support = np.clip(
        oof_support / (1.0 + np.sqrt(np.maximum(oof_lev, 0.0))), 0.0, 1.0)
    edge_support = reference_support * qc_support
    edge_features = np.stack([
        np.maximum(z, 0.0), np.maximum(-z, 0.0),
        (z >= 1.0).astype(np.float64), (z <= -1.0).astype(np.float64),
        np.broadcast_to(edge_support[:, None], (n, 256)),
        np.broadcast_to(1.0 - reference_support[:, None], (n, 256)),
    ], axis=-1)

    degree = np.zeros(111, dtype=np.int64)
    incident = [[] for _ in range(111)]
    for edge, (a, b) in enumerate(endpoints):
        if a >= 0:
            incident[int(a)].append(edge)
            incident[int(b)].append(edge)
            degree[int(a)] += 1
            degree[int(b)] += 1
    edge_weight = np.broadcast_to(edge_support[:, None], (n, 256))
    roi_features = np.zeros((n, 111, 6), dtype=np.float64)
    for roi, ids in enumerate(incident):
        if not ids:
            continue
        mask = np.zeros(256, dtype=bool)
        mask[np.asarray(ids, dtype=np.int64)] = True
        pooled = _pool(edge_features, edge_weight, mask & known)
        roi_features[:, roi, :4] = pooled[:, :4]
        roi_features[:, roi, 4] = degree[roi] / (degree[roi] + 3.0) * edge_support
        roi_features[:, roi, 5] = 1.0 - edge_support
    family_edge_masks = []
    for mask in family_masks:
        mask = np.asarray(mask, dtype=bool)
        family_edge_masks.append(np.asarray([
            bool(a >= 0 and (mask[a] or mask[b])) for a, b in endpoints
        ], dtype=bool))
    family_features = np.stack([
        _pool(edge_features, edge_weight, mask & known)
        for mask in family_edge_masks
    ], axis=1)

    def concept_from_masks(edge_masks, roi_masks, family_masks_for_concepts):
        rows = []
        for k in range(len(edge_masks)):
            edge_pool = _pool(edge_features, edge_weight,
                              np.asarray(edge_masks[k], dtype=bool) & known)
            roi_ids = np.flatnonzero(roi_masks[k])
            if len(roi_ids):
                rw = np.maximum(roi_features[:, roi_ids, 4], 1e-12)
                roi_pool = (rw[:, :, None] * roi_features[:, roi_ids]).sum(1)
                roi_pool /= rw.sum(1)[:, None]
            else:
                roi_pool = np.zeros((n, 6))
            fam_ids = np.flatnonzero(family_masks_for_concepts[k])
            if len(fam_ids):
                fw = np.maximum(family_features[:, fam_ids, 4], 1e-12)
                fam_pool = (fw[:, :, None] * family_features[:, fam_ids]).sum(1)
                fam_pool /= fw.sum(1)[:, None]
            else:
                fam_pool = np.zeros((n, 6))
            rows.append(np.concatenate([edge_pool, roi_pool, fam_pool], axis=1))
        return np.stack(rows, axis=1)

    concept_family = np.zeros((4, len(family_masks)), dtype=bool)
    concept_family[0, [0, 2]] = True
    concept_family[1, [1, 3, 4]] = True
    concept_family[2, [1, 2, 5]] = True
    concept_family[3, [7]] = True
    concept_features = concept_from_masks(concept_edge_masks, concept_roi_masks,
                                          concept_family)
    signal = (0.45 * (1.0 - np.exp(-0.25 * np.clip(
        concept_features[:, :, 0] + concept_features[:, :, 1], 0, 8))) +
              0.30 * np.maximum(concept_features[:, :, 2], concept_features[:, :, 3]) +
              0.25 * np.clip(concept_features[:, :, 4], 0, 1))
    thresholds = np.maximum(np.quantile(signal[td], 0.75, axis=0), 0.05)
    return {
        "z": z.astype(np.float32), "edge_features": edge_features.astype(np.float32),
        "roi_features": roi_features.astype(np.float32),
        "family_features": family_features.astype(np.float32),
        "concept_features": concept_features.astype(np.float32),
        "signal": signal.astype(np.float32), "thresholds": thresholds.astype(np.float32),
        "reference_support": reference_support.astype(np.float32),
        "qc_support": qc_support.astype(np.float32),
        "edge_support": edge_support.astype(np.float32), "known_edge": known,
        "degree": degree, "train_idx": train_idx, "test_idx": test,
        "train_y": np.asarray(y, dtype=np.int64),
        "td_oof_mse": float(np.mean(residual ** 2)),
        "td_oof_mae": float(np.mean(np.abs(residual))),
        "z_clip_fraction": float(np.mean(np.abs(z) >= 3.999)),
        "signal_q75": np.quantile(signal[td], 0.75, axis=0).tolist(),
    }


def semantic_score(evidence, edge_masks, roi_masks, eligible, forced=False,
                   concept_family=None):
    """Use one calibrated continuous score for match and reject decisions."""
    features = recompute_concept_features(evidence, edge_masks, roi_masks,
                                          concept_family)
    signal = (0.45 * (1.0 - np.exp(-0.25 * np.clip(
        features[:, :, 0] + features[:, :, 1], 0, 8))) +
              0.30 * np.maximum(features[:, :, 2], features[:, :, 3]) +
              0.25 * np.clip(features[:, :, 4], 0, 1))
    train_idx = np.asarray(evidence["train_idx"], dtype=np.int64)
    train_y = np.asarray(evidence.get("train_y", np.ones(0, dtype=int)), dtype=np.int64)
    if len(train_y) == 0:
        td = train_idx
    else:
        td = train_idx[train_y[train_idx] == 1]
    thresholds = np.maximum(np.quantile(signal[td], 0.75, axis=0), 0.05)
    n, k = len(signal), len(edge_masks)
    strength = 1.0 / (1.0 + np.exp(-(signal - thresholds[None, :]) / 0.20))
    support = np.zeros((n, k), dtype=np.float64)
    edge_support = np.asarray(evidence["edge_support"], dtype=np.float64)
    if edge_support.ndim == 1:
        edge_support = np.broadcast_to(edge_support[:, None], (n, 256))
    for j in range(k):
        ids = np.flatnonzero(np.asarray(edge_masks[j], dtype=bool) & evidence["known_edge"])
        if len(ids):
            support[:, j] = edge_support[:, ids].mean(axis=1)
        roi_ids = np.flatnonzero(roi_masks[j])
        if len(roi_ids):
            support[:, j] *= np.clip(evidence["roi_features"][:, roi_ids, 4].mean(1), 0, 1)
    raw = 3.0 * strength * support * np.asarray(eligible, dtype=np.float64)[None, :]
    raw[:, ~np.asarray(eligible, dtype=bool)] = 0.0
    if forced:
        denom = raw.sum(axis=1, keepdims=True)
        matched = np.divide(raw, denom, out=np.zeros_like(raw), where=denom > 1e-12)
        unmatched = (denom[:, 0] <= 1e-12).astype(np.float64)
    else:
        denom = 1.0 + raw.sum(axis=1)
        unmatched = 1.0 / denom
        matched = raw / denom[:, None]
    return matched.astype(np.float32), unmatched.astype(np.float32), features.astype(np.float32), signal.astype(np.float32), thresholds.astype(np.float32)


def recompute_concept_features(evidence, edge_masks, roi_masks, concept_family=None):
    """Re-pool actual patient evidence after changing the anatomical mapping."""
    edge_features = evidence["edge_features"].astype(np.float64)
    edge_weight = evidence["edge_support"].astype(np.float64)
    if edge_weight.ndim == 1:
        edge_weight = np.broadcast_to(edge_weight[:, None], (len(edge_features), 256))
    roi_features = evidence["roi_features"].astype(np.float64)
    known = evidence["known_edge"]
    family_features = evidence.get("family_features")
    rows = []
    for k in range(len(edge_masks)):
        ids = np.flatnonzero(np.asarray(edge_masks[k], dtype=bool) & known)
        if len(ids):
            w = edge_weight[:, ids]
            edge_pool = (w[:, :, None] * edge_features[:, ids]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]
        else:
            edge_pool = np.zeros((len(edge_features), 6))
        roi_ids = np.flatnonzero(roi_masks[k])
        if len(roi_ids):
            w = np.maximum(roi_features[:, roi_ids, 4], 1e-12)
            roi_pool = (w[:, :, None] * roi_features[:, roi_ids]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]
        else:
            roi_pool = np.zeros((len(edge_features), 6))
        if concept_family is not None and family_features is not None:
            fam_ids = np.flatnonzero(concept_family[k])
            if len(fam_ids):
                w = np.maximum(family_features[:, fam_ids, 4], 1e-12)
                fam_pool = (w[:, :, None] * family_features[:, fam_ids]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]
            else:
                fam_pool = np.zeros_like(roi_pool)
        else:
            fam_pool = np.zeros_like(roi_pool)
        rows.append(np.concatenate([edge_pool, roi_pool, fam_pool], axis=1))
    return np.stack(rows, axis=1)


class SAGEAlign(nn.Module):
    """Shared low-rank evidence transform; CAM is query-only modulation."""
    def __init__(self, evidence_dim=18, concepts=4, rank=8, output_dim=144,
                 cam_width=36, cap=0.08):
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.concepts = int(concepts)
        self.rank = int(rank)
        self.cap = float(cap)
        self.evidence_proj = nn.Linear(evidence_dim, rank, bias=False)
        self.cam_query = nn.Linear(cam_width, concepts, bias=False)
        self.output_proj = nn.Linear(rank, cam_width, bias=False)
        self.residual_proj = nn.Linear(cam_width, output_dim, bias=False)
        nn.init.zeros_(self.residual_proj.weight)

    def forward(self, evidence: torch.Tensor, cam: torch.Tensor):
        if evidence.ndim != 3 or evidence.shape[1] != self.concepts:
            raise ValueError("SAGE evidence must be [batch, concepts, evidence_dim]")
        query = self.cam_query(cam.mean(dim=1))
        slots = F.gelu(self.evidence_proj(evidence))
        pooled = (torch.sigmoid(query).unsqueeze(-1) * slots).sum(dim=1)
        raw = self.residual_proj(self.output_proj(pooled))
        base = cam.reshape(cam.shape[0], -1)
        norm = raw.norm(dim=1, keepdim=True)
        cap = self.cap * base.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.where(norm > 1e-12,
                            cap * torch.tanh(norm / cap) / norm.clamp_min(1e-12),
                            torch.ones_like(norm))
        residual = raw * scale
        return (base + residual).reshape_as(cam), residual


def zero_evidence_test(branch, cam):
    zeros = torch.zeros((cam.shape[0], branch.concepts, branch.evidence_dim),
                        device=cam.device)
    _, residual = branch(zeros, cam)
    return bool(torch.count_nonzero(residual).item() == 0)

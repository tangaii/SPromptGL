"""Semantic-guided relational complementary alignment for Experiment 34."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIGS = (
    "S0_B0A", "S1_NUMERIC_GLOBAL", "S2_RANDOM_CONCEPTS", "S3_SINGLE_CONCEPT",
    "S4_CORR_REL_ONLY", "S5_AUX_REL_ONLY", "S6_UNIFORM_RELATION",
    "S7_DIRECT_MESSAGE", "S8_PATIENT_PERMUTED", "S9_SERCA_FULL",
)
RELATIONS = ("PHENO", "ANAT", "FUNC", "CORR")
CONCEPTS = ("D1", "D2", "D3", "D4")


def fit_similarity(fc: np.ndarray, support: np.ndarray, train: np.ndarray):
    """Label-free signed-FC Gaussian similarities with train-only scales."""
    fc = np.asarray(fc, dtype=np.float64)
    support = np.asarray(support, dtype=np.int64)
    train = np.asarray(train, dtype=np.int64)
    assert fc.shape == (871, 256) and len(support) > 0
    fit = fc[train][:, support]
    mean = fit.mean(axis=0, dtype=np.float64)
    scale = fit.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (fc[:, support] - mean) / scale
    gram = z @ z.T
    norm2 = np.einsum("ij,ij->i", z, z)
    distance2 = np.maximum(norm2[:, None] + norm2[None, :] - 2.0 * gram, 0.0)
    distance2 = (distance2 + distance2.T) * 0.5
    train_d = distance2[np.ix_(train, train)]
    tri = np.triu_indices(len(train), 1)
    positive = train_d[tri]
    positive = positive[positive > 1e-12]
    tau = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-distance2 / tau)
    np.fill_diagonal(similarity, 1.0)
    train_s = similarity[np.ix_(train, train)]
    center = float(train_s[tri].mean()) if len(tri[0]) else 1.0
    assert np.isfinite(similarity).all() and 0.0 <= similarity.min() <= similarity.max() <= 1.0
    return similarity.astype(np.float32), dict(
        support_count=int(len(support)), tau=tau, center=center,
        train_distance_p05=float(np.quantile(positive, .05)) if len(positive) else 0.0,
        train_distance_median=tau,
        train_distance_p95=float(np.quantile(positive, .95)) if len(positive) else 0.0,
        similarity_mean=float(similarity.mean()),
        similarity_std=float(similarity.std()),
        similarity_train_mean=float(train_s[tri].mean()) if len(tri[0]) else 1.0,
        similarity_train_std=float(train_s[tri].std()) if len(tri[0]) else 0.0,
        standardized_feature_scale_min=float(scale.min()),
        standardized_feature_scale_max=float(scale.max()),
    )


def build_conditioned_graphs(raw_adjs, base_adjs, similarities, centers,
                             normalize_adj, relation_mask=None, kappa=1.0):
    """Reweight existing off-diagonal edges and retain raw-graph gradients."""
    relation_mask = tuple(range(4)) if relation_mask is None else tuple(relation_mask)
    assert len(raw_adjs) == len(base_adjs) == 4
    assert similarities.shape == (4, 871, 871) and len(centers) == 4
    identity = torch.eye(871, device=raw_adjs[0].device, dtype=raw_adjs[0].dtype)
    conditioned, difference = [], []
    for m, raw in enumerate(raw_adjs):
        assert raw.shape == (871, 871) and torch.isfinite(raw).all()
        if m not in relation_mask:
            conditioned.append(base_adjs[m].unsqueeze(0).expand(4, -1, -1))
            difference.append(torch.zeros_like(base_adjs[m]).unsqueeze(0).expand(4, -1, -1))
            continue
        diagonal = torch.diag(torch.diagonal(raw))
        off = raw - diagonal
        row_conditioned, row_difference = [], []
        for k in range(4):
            sim = similarities[k].to(device=raw.device, dtype=raw.dtype)
            factor = torch.exp(float(kappa) * (sim - float(centers[k])))
            weighted = diagonal + off * factor
            normed = normalize_adj(weighted + identity)
            row_conditioned.append(normed)
            row_difference.append(normed - base_adjs[m])
            assert torch.isfinite(normed).all() and torch.all(normed >= -1e-7)
            assert torch.equal(torch.diagonal(weighted), torch.diagonal(raw))
            assert torch.count_nonzero((weighted - diagonal) * (off == 0)) == 0
        conditioned.append(torch.stack(row_conditioned))
        difference.append(torch.stack(row_difference))
    return torch.stack(conditioned), torch.stack(difference)


class SeRCA(nn.Module):
    def __init__(self, similarities, centers, relation_mask, concept_mask,
                 direct_message=False, uniform_relation=False):
        super().__init__()
        similarities = torch.as_tensor(similarities, dtype=torch.float32)
        centers = torch.as_tensor(centers, dtype=torch.float32)
        assert similarities.shape == (4, 871, 871) and centers.shape == (4,)
        self.register_buffer("similarities", similarities)
        self.register_buffer("centers", centers)
        self.register_buffer("relation_mask", torch.as_tensor(relation_mask, dtype=torch.bool))
        self.register_buffer("concept_mask", torch.as_tensor(concept_mask, dtype=torch.bool))
        self.alpha = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.theta = nn.Parameter(torch.zeros(4, 4, 4, dtype=torch.float32))
        self.direct_message = bool(direct_message)
        self.uniform_relation = bool(uniform_relation)
        self.collect_diagnostics = False
        self.last_diagnostics = None
        assert self.relation_mask.any() and self.concept_mask.any()

    def weights(self):
        valid = self.relation_mask[:, None] & self.concept_mask[None, :]
        if self.uniform_relation:
            weight = valid.to(self.alpha.dtype)
            return (weight / weight.sum()).unsqueeze(0).expand(4, -1, -1)
        logits = self.theta.masked_fill(~valid.unsqueeze(0), -torch.inf)
        return F.softmax(logits.flatten(1), dim=-1).view(4, 4, 4)

    def forward(self, hidden, raw_adjs, base_adjs, normalize_adj):
        assert hidden.shape == (871, 4, 36)
        conditioned, difference = build_conditioned_graphs(
            raw_adjs, base_adjs, self.similarities, self.centers,
            normalize_adj, relation_mask=torch.where(self.relation_mask)[0].tolist(),
        )
        weights = self.weights()
        source_matrices = conditioned if self.direct_message else difference
        message_matrix = torch.einsum("tmk,mkij->tij", weights, source_matrices)
        messages = torch.bmm(message_matrix, hidden.transpose(0, 1)).transpose(0, 1)
        raw_delta = (0.10 * torch.tanh(self.alpha))[None, :, None] * messages
        input_norm = hidden.norm(dim=-1, keepdim=True)
        raw_norm = raw_delta.norm(dim=-1, keepdim=True)
        cap = 0.10 * input_norm
        scale = torch.minimum(torch.ones_like(raw_norm), cap / raw_norm.clamp_min(1e-12))
        delta = raw_delta * scale
        if self.collect_diagnostics:
            by_relation = torch.einsum("tmk,mkij->tmij", weights, source_matrices)
            contribution = torch.bmm(
                by_relation.reshape(16, 871, 871),
                hidden.transpose(0, 1).repeat(4, 1, 1),
            ).view(4, 4, 871, 36).norm(dim=-1).mean(dim=-1)
            base_stack = base_adjs_tensor(base_adjs, conditioned)
            offdiag = ~torch.eye(871, device=conditioned.device, dtype=torch.bool)
            normalized_change = (conditioned - base_stack).abs() * offdiag
            base_offdiag = base_stack.abs() * offdiag
            normalized_change_fraction = (
                ((normalized_change > 1e-10) & (base_offdiag > 0)).sum(dim=(-1, -2)) /
                (base_offdiag > 0).sum(dim=(-1, -2)).clamp_min(1)
            )
            normalized_relative_change = (
                normalized_change.sum(dim=(-1, -2)) /
                base_offdiag.sum(dim=(-1, -2)).clamp_min(1e-12)
            )
            ratio = delta.norm(dim=-1) / input_norm.squeeze(-1).clamp_min(1e-12)
            self.last_diagnostics = dict(
                alpha=self.alpha.detach().cpu().tolist(),
                weights=weights.detach().cpu().numpy(),
                mean_token_residual_ratio=ratio.mean(dim=0).detach().cpu().tolist(),
                max_token_residual_ratio=ratio.max(dim=0).values.detach().cpu().tolist(),
                mean_relation_message_contribution=contribution.detach().cpu().numpy(),
                mean_conditioned_normalized_offdiag_relative_change=normalized_relative_change.detach().cpu().numpy(),
                mean_conditioned_normalized_offdiag_changed_edge_fraction=normalized_change_fraction.detach().cpu().numpy(),
                max_token_residual_ratio_observed=float(ratio.max().item()),
            )
        return hidden + delta


def base_adjs_tensor(base_adjs, conditioned):
    return torch.stack(base_adjs).unsqueeze(1).expand_as(conditioned)

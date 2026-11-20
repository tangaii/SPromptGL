"""Fold-local CPU float64 Gaussian references and deterministic original-head helpers."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F

EPS = 1e-8
DIM = 16


def fit_gaussians(z, labels, cfg):
    """Fit two equally weighted classes in the unwhitened Original O coordinates."""
    fitted = []
    for label in (0, 1):
        rows = z[labels == label]
        if len(rows) <= DIM:
            raise ValueError(f"Insufficient Gaussian training rows for class {label}")
        mean = rows.mean(axis=0)
        centered = rows - mean
        sample = centered.T @ centered / (len(rows) - 1)
        ridge = max(cfg["ridge_min"], cfg["ridge_relative"] * np.trace(sample) / DIM)
        shrink = cfg["shrinkage"]
        covariance = (1 - shrink) * sample + shrink * np.diag(np.diag(sample)) + ridge * np.eye(DIM)
        cholesky = np.linalg.cholesky(covariance)
        fitted.append({
            "mean": mean, "covariance": covariance, "cholesky": cholesky,
            "logdet": 2 * np.log(np.diag(cholesky)).sum(), "ridge": ridge,
            "condition": np.linalg.cond(covariance), "count": len(rows),
        })
    results = {key: np.asarray([entry[key] for entry in fitted]) for key in fitted[0]}
    squared = [np.square(solve_triangular(L, (z - mean).T, lower=True)).sum(axis=0)
               for mean, L in zip(results["mean"], results["cholesky"])]
    results["support"] = np.asarray(max(cfg["ridge_min"],
                                       np.quantile(np.minimum(*squared), cfg["support_quantile"])))
    return results


def reference_llr(z, stats):
    mahalanobis = []
    for mean, chol in zip(stats["mean"], stats["cholesky"]):
        transformed = solve_triangular(chol, (z - mean).T, lower=True)
        mahalanobis.append(np.square(transformed).sum(axis=0))
    a, t = mahalanobis
    return 0.5 * (t - a + stats["logdet"][1] - stats["logdet"][0])


def calibrate(raw, labels, cfg):
    """Zero-intercept, nonnegative temperature fit under equal class priors."""
    scale = max(float(np.sqrt(np.mean(np.square(raw)))), cfg["calibration_scale_min"])
    evidence = raw / scale
    target = (labels == 0).astype(np.float64)
    weights = 0.5 / np.bincount(labels, minlength=2)[labels]
    ridge = cfg["calibration_relative_l2"]
    upper = cfg["calibration_max_slope"]

    def derivative(slope):
        return np.sum(weights * (expit(slope * evidence) - target) * evidence) + ridge * slope

    if derivative(0) >= 0:
        slope = 0.0
    elif derivative(upper) <= 0:
        slope = upper
    else:
        slope = brentq(derivative, 0, upper, xtol=1e-12)
    return np.array([scale, slope], dtype=np.float64)


def fit_crossfit(z, labels, legal, fold, cfg):
    """Five held-out groups, plus pair holdouts within the same split for calibration."""
    if z.dtype != np.float64 or z.shape != (len(legal), DIM):
        raise ValueError("Expected legal frozen CPU float64 Original O features")
    seed = cfg["seed_base"] + fold
    splits = cfg["splits"]
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=np.random.RandomState(seed))
    assignment = np.full(len(legal), -1, dtype=np.int64)
    for i, (_, holdout) in enumerate(splitter.split(z, labels)):
        assignment[holdout] = i
    if (assignment < 0).any() or np.unique(legal).size != len(legal):
        raise ValueError("Malformed inner split")

    full = fit_gaussians(z, labels, cfg)
    stats = [full]
    oof = np.empty(len(legal), dtype=np.float64)
    for i in range(splits):
        fit = assignment != i
        gaussian = fit_gaussians(z[fit], labels[fit], cfg)
        stats.append(gaussian)
        oof[~fit] = reference_llr(z[~fit], gaussian)

    # When calibrating group i, all calibration labels AND the Gaussian fits
    # that produced their scores exclude group i. Share the pairwise fits.
    inner_calibration_raw = np.full((splits, len(legal)), np.nan, dtype=np.float64)
    for i, j in combinations(range(splits), 2):
        keep = (assignment != i) & (assignment != j)
        gaussian = fit_gaussians(z[keep], labels[keep], cfg)
        inner_calibration_raw[i, assignment == j] = reference_llr(z[assignment == j], gaussian)
        inner_calibration_raw[j, assignment == i] = reference_llr(z[assignment == i], gaussian)

    calibration = [calibrate(oof, labels, cfg)]
    calibrated_oof = np.empty_like(oof)
    for i, raw in enumerate(inner_calibration_raw):
        fit = assignment != i
        scaler = calibrate(raw[fit], labels[fit], cfg)
        calibration.append(scaler)
        calibrated_oof[~fit] = expit(oof[~fit] * (scaler[1] / scaler[0]))

    gaussian = {key: np.stack([entry[key] for entry in stats]) for key in full}
    if not all(np.isfinite(value).all() for value in (*gaussian.values(), oof, calibration, calibrated_oof)):
        raise FloatingPointError("Nonfinite frozen statistics or calibration")
    return {
        "seed": seed, "legal": legal.copy(), "labels": labels.copy(), "assignment": assignment,
        "gaussian": gaussian, "calibration": np.stack(calibration), "oof_llr": oof, "oof_asd": calibrated_oof,
    }


def group_rows(reference, subjects):
    """Pack rows by Gaussian fit so Cholesky solves can share matrix factors."""
    group = np.zeros(subjects, dtype=np.int64)
    group[reference["legal"]] = reference["assignment"] + 1
    rows = [np.flatnonzero(group == i) for i in range(len(reference["calibration"]))]
    width = max(map(len, rows))
    packed = np.stack([np.pad(row, (0, width - len(row)), mode="edge") for row in rows])
    unpack = np.empty(subjects, dtype=np.int64)
    for i, row in enumerate(rows):
        unpack[row] = i * width + np.arange(len(row))
    return packed, unpack


class GaussianEvidence(nn.Module):
    def __init__(self, reference, subjects, cfg):
        super().__init__()
        gaussian = reference["gaussian"]
        for name in ("mean", "cholesky", "logdet", "support"):
            self.register_buffer(name, torch.tensor(gaussian[name], dtype=torch.float32))
        calibration = reference["calibration"]
        self.register_buffer("llr_multiplier", torch.tensor(calibration[:, 1] / calibration[:, 0], dtype=torch.float32))
        pack, unpack = group_rows(reference, subjects)
        self.register_buffer("pack", torch.from_numpy(pack))
        self.register_buffer("unpack", torch.from_numpy(unpack))
        self.support_decay = cfg["support_decay"]

    def class_evidence(self, z, crossfit):
        if crossfit:
            grouped, selection = z[self.pack], slice(None)
        else:
            grouped, selection = z.unsqueeze(0), slice(0, 1)
        error = (grouped[:, None] - self.mean[selection, :, None]).transpose(-2, -1)
        cholesky = self.cholesky[selection]
        white = torch.linalg.solve_triangular(cholesky, error, upper=False)
        direction = -torch.linalg.solve_triangular(cholesky.transpose(-2, -1), white, upper=True)
        squared_distance = white.square().sum(-2)
        deviance = squared_distance + self.logdet[selection, :, None]
        llr = 0.5 * (deviance[:, 1] - deviance[:, 0])
        asd = torch.sigmoid(llr * self.llr_multiplier[selection, None])
        support = (-(squared_distance.amin(1) / self.support[selection, None] - 1).relu()
                   * self.support_decay).exp()
        reliability = (2 * asd - 1).abs() * support
        fused = asd[:, None] * direction[:, 0] + (1 - asd[:, None]) * direction[:, 1]
        values = (
            llr.flatten(), asd.flatten(), direction.permute(0, 3, 1, 2).reshape(-1, 2, DIM),
            fused.transpose(-2, -1).reshape(-1, DIM), reliability.flatten(), support.flatten(),
        )
        return tuple(value[self.unpack] for value in values) if crossfit else values


def frozen_head(head, cam):
    """Current original head with detached parameters and its dropout in eval mode."""
    norm, linear, gelu, dropout = head.mlp_head
    if not isinstance(dropout, nn.Dropout):
        raise TypeError("Historical classification head changed")
    hidden = F.layer_norm(cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    hidden = F.linear(hidden, linear.weight.detach(), linear.bias.detach())
    hidden = F.gelu(hidden, approximate=gelu.approximate)
    return F.linear(hidden, head.classifier.weight.detach(), head.classifier.bias.detach())


def margin(logits, labels):
    return (logits[:, 0] - logits[:, 1]) * (1 - 2 * labels)


def balanced_mean(values, labels):
    return 0.5 * (values[labels == 0].mean() + values[labels == 1].mean())

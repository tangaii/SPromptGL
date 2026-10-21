"""Fold-local, label-free QC-to-FC conditional statistics for Experiment 33."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.model_selection import KFold

RIDGE = 1.0
INNER_SPLITS = 3
EPS = 1e-12


def _scale_fit(values: np.ndarray):
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _ridge_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_all: np.ndarray):
    """Standardized ridge with an unpenalized intercept, fitted only on x_fit/y_fit."""
    xm, xs = _scale_fit(x_fit)
    ym, ys = _scale_fit(y_fit)
    xz = (x_fit - xm) / xs
    yz = (y_fit - ym) / ys
    gram = xz.T @ xz + RIDGE * np.eye(xz.shape[1], dtype=np.float64)
    beta = np.linalg.solve(gram, xz.T @ yz)
    pred_z = ((x_all - xm) / xs) @ beta
    return pred_z * ys + ym, dict(x_mean=xm, x_scale=xs, y_mean=ym, y_scale=ys, beta=beta)


def _oof_reliability(c: np.ndarray, q: np.ndarray, f: np.ndarray, fold: int):
    n = len(c)
    oof_c = np.zeros_like(f, dtype=np.float64)
    oof_cq = np.zeros_like(f, dtype=np.float64)
    scale_sq = np.zeros_like(f, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int8)
    splitter = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=33033 + fold)
    for fit_pos, val_pos in splitter.split(np.arange(n)):
        pred_c, model_c = _ridge_predict(c[fit_pos], f[fit_pos], c[val_pos])
        x_cq_fit = np.concatenate((c[fit_pos], q[fit_pos]), axis=1)
        x_cq_val = np.concatenate((c[val_pos], q[val_pos]), axis=1)
        pred_cq, model_cq = _ridge_predict(x_cq_fit, f[fit_pos], x_cq_val)
        oof_c[val_pos] = pred_c
        oof_cq[val_pos] = pred_cq
        scale_sq[val_pos] = model_c["y_scale"] ** 2
        assert np.allclose(model_c["y_scale"], model_cq["y_scale"], rtol=0, atol=1e-12)
        seen[val_pos] += 1
    assert np.all(seen == 1)
    mse_c = np.mean(((f - oof_c) ** 2) / scale_sq, axis=0)
    mse_cq = np.mean(((f - oof_cq) ** 2) / scale_sq, axis=0)
    gain = (mse_c - mse_cq) / (mse_c + EPS)
    reliability = np.clip(np.maximum(gain, 0.0), 0.0, 1.0)
    return gain, reliability, dict(
        mse_c=mse_c, mse_cq=mse_cq, oof_prediction_c=oof_c,
        oof_prediction_cq=oof_cq, inner_fold_assignments=seen,
    )


def _site_shuffle(values: np.ndarray, site_ids: np.ndarray, train: np.ndarray, fold: int):
    rng = np.random.default_rng(53000 + fold)
    out = values.copy()
    permutation = np.arange(len(values))
    singleton_count = 0
    train_mask = np.zeros(len(values), dtype=bool)
    train_mask[train] = True
    for partition in (train_mask, ~train_mask):
        for site in np.unique(site_ids):
            ids = np.flatnonzero((site_ids == site) & partition)
            if len(ids) <= 1:
                singleton_count += len(ids)
                continue
            permutation[ids] = rng.permutation(ids)
            out[ids] = values[permutation[ids]]
    return out, permutation, singleton_count


def fit_fold(pheno: np.ndarray, qc: np.ndarray, fc: np.ndarray,
             train_indices: np.ndarray, fold: int):
    """Fit all frozen directions using outer-train rows only; labels are not accepted."""
    c = np.asarray(pheno, dtype=np.float64)
    q = np.asarray(qc, dtype=np.float64)
    f = np.asarray(fc, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    assert c.shape == (871, 48) and q.shape == (871, 16) and f.shape == (871, 256)
    assert len(np.unique(train)) == len(train) and train.min() >= 0 and train.max() < len(c)
    assert all(np.isfinite(v).all() for v in (c, q, f))
    c_train, q_train, f_train = c[train], q[train], f[train]

    gain, reliability, oof = _oof_reliability(c_train, q_train, f_train, fold)

    site = c[:, 26:46]
    site_train = site[train]
    site_pred, site_model = _ridge_predict(site_train, f_train, site)
    site_pred -= site_pred[train].mean(axis=0, keepdims=True)

    cq = np.concatenate((c, q), axis=1)
    joint_pred, joint_model = _ridge_predict(cq[train], f_train, cq)
    joint_pred -= joint_pred[train].mean(axis=0, keepdims=True)

    q_hat, c_to_q = _ridge_predict(c_train, q_train, c)
    f_hat_c, c_to_f = _ridge_predict(c_train, f_train, c)
    q_res = q - q_hat
    f_res = f - f_hat_c
    f_hat_q, q_to_f = _ridge_predict(q_res[train], f_res[train], q_res)
    f_hat_q -= f_hat_q[train].mean(axis=0, keepdims=True)
    f_only = f_res.copy()

    site_ids = site.argmax(axis=1)
    wrong_pair, permutation, singleton_count = _site_shuffle(f_hat_q, site_ids, train, fold)
    mismatch = np.flatnonzero(permutation != np.arange(len(permutation)))

    arrays = {
        "site_only": site_pred,
        "unconditional": joint_pred,
        "qcar": f_hat_q,
        "qcar_wrong_pair": wrong_pair,
        "fc_only": f_only,
        "qcar_reverse": f_hat_q,
    }
    assert all(value.shape == (871, 256) and np.isfinite(value).all() for value in arrays.values())
    stats = dict(
        fold=int(fold), outer_train_count=int(len(train)),
        outer_train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
        n_oof_connections_nonzero=int(np.count_nonzero(reliability)),
        n_oof_connections_positive=int(np.count_nonzero(gain > 0)),
        gain_mean=float(gain.mean()), gain_median=float(np.median(gain)),
        gain_min=float(gain.min()), gain_max=float(gain.max()),
        reliability_mean=float(reliability.mean()), reliability_median=float(np.median(reliability)),
        reliability_min=float(reliability.min()), reliability_max=float(reliability.max()),
        oof_mse_c_mean=float(oof["mse_c"].mean()),
        oof_mse_cq_mean=float(oof["mse_cq"].mean()),
        q_residual_norm_train_mean=float(np.linalg.norm(q_res[train], axis=1).mean()),
        q_residual_norm_all=np.linalg.norm(q_res, axis=1),
        q_residual_norm_train_p80=float(np.quantile(np.linalg.norm(q_res[train], axis=1), .80)),
        wrong_pair_subject_fraction=float(len(mismatch) / len(permutation)),
        wrong_pair_site_singletons=int(singleton_count),
        wrong_pair_permutation_sha256=hashlib.sha256(permutation.tobytes()).hexdigest(),
        fit_sample_count=int(len(train)), fit_uses_labels=False,
        fit_uses_outer_test_fc=False,
    )
    digest = hashlib.sha256()
    digest.update(train.tobytes())
    for name in sorted(arrays):
        digest.update(name.encode() + b"\0")
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    digest.update(np.ascontiguousarray(reliability).tobytes())
    stats["statistics_sha256"] = digest.hexdigest()
    stats["reliability"] = reliability
    stats["gain"] = gain
    return arrays, stats


def json_stats(stats: dict):
    return {key: value for key, value in stats.items()
            if key not in {"reliability", "gain", "q_residual_norm_all"}}


def calibration_diagnostics(fc: np.ndarray, q_residual_norm: np.ndarray,
                            candidate: np.ndarray, reliability: np.ndarray,
                            gamma: float, train_threshold: float):
    fc = np.asarray(fc, dtype=np.float64)
    raw = float(gamma) * candidate * reliability[None, :]
    raw_norm = np.linalg.norm(raw, axis=1)
    fc_norm = np.linalg.norm(fc, axis=1)
    cap = .15 * fc_norm
    scale = np.minimum(1.0, cap / np.maximum(raw_norm, 1e-12))
    delta = raw * scale[:, None]
    ratio = np.linalg.norm(delta, axis=1) / np.maximum(fc_norm, 1e-12)
    threshold = float(train_threshold)
    high = q_residual_norm >= threshold
    return dict(
        gamma=float(gamma), mean_relative_correction=float(ratio.mean()),
        max_relative_correction=float(ratio.max()),
        n_subjects_nonzero_correction=int(np.count_nonzero(np.linalg.norm(delta, axis=1) > 0)),
        high_qc_residual_threshold=threshold,
        high_qc_mean_relative_correction=float(ratio[high].mean()),
        low_qc_mean_relative_correction=float(ratio[~high].mean()),
        correction_sha256=hashlib.sha256(np.ascontiguousarray(delta).tobytes()).hexdigest(),
    )

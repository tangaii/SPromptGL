"""Independent best/latest, RNG, mechanism, and paired patient-level audit."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

from common import history as h
import numpy as np
import torch
from scipy.linalg import cho_solve
from scipy.special import expit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F

import run

CFG, HERE, RESULTS = h.CFG, h.HERE, run.RESULTS
HISTORICAL = ("DARG", "E82.2-G2", "E82.3.2-G2")
METHODS = tuple(run.METHODS)
METRIC_NAMES = ("ACC", "Hard-AUC/BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
SCOPES = ("10_folds", "exclude_fold2")


def eq(a, b, message):
    h.require(np.array_equal(a, b), message)



def close(a, b, message, *, atol=1e-9, rtol=1e-9):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=message)



def array(tensor):
    return tensor.detach().cpu().numpy()



def metric_values(probability, predictions, labels):
    asd = labels == CFG["labels"]["ASD"]
    h.require(probability.shape == (len(labels), 2) and np.isfinite(probability).all(), "Malformed probability")
    eq(predictions, probability.argmax(-1), "Predictions differ from archived probability argmax")
    close(probability.sum(-1), 1.0, "Probabilities do not sum to one", atol=2e-6)
    ba = balanced_accuracy_score(labels, predictions)
    close(ba, roc_auc_score(asd, predictions == 0), "Binary Hard-AUC differs from BA")
    return dict(zip(h.METRICS, (
        float(accuracy_score(labels, predictions)), float(ba), float(roc_auc_score(asd, probability[:, 0])),
        float(recall_score(labels, predictions, pos_label=0)), float(recall_score(labels, predictions, pos_label=1)),
        float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
    )))



def reference_split(ctx):
    state, saved = ctx["state"], ctx["reference"]
    legal, test = state["legal_np"], state["test_np"]
    h.require(not np.intersect1d(legal, test).size and not np.intersect1d(legal, state["excluded"]).size,
              "New training features or labels include outer test rows")
    boundary = {"legal_count": len(legal), "test_count": len(test),
                "excluded_1based": [i + 1 for i in state["excluded"]]}
    if saved is None:
        return boundary
    eq(saved["legal"], legal, "Frozen Gaussian legal rows differ")
    eq(saved["labels"], state["labels_np"][legal], "Frozen Gaussian labels differ")
    h.require(not np.intersect1d(legal, test).size, "New training features or labels include test rows")
    h.require(not np.intersect1d(legal, state["excluded"]).size, "Fold 2 excluded rows enter new training")
    labels = saved["labels"]
    z = ctx["reference_z"]
    cfg = CFG["crossfit"]
    h.require(saved["seed"] == cfg["seed_base"] + state["fold"], "Cross-fit seed mismatch")
    parts = list(StratifiedKFold(cfg["splits"], shuffle=True, random_state=saved["seed"]).split(z, labels))
    assignment = np.empty(len(z), dtype=np.int64)
    for group, (train, valid) in enumerate(parts):
        assignment[valid] = group
        h.require(not np.intersect1d(train, valid).size and not np.intersect1d(legal[train], test).size,
                  "Held-out membership overlaps fit")
    eq(assignment, saved["assignment"], "Recorded stratified split is not reproducible")

    def stats(rows):
        means, covs, ridges, counts = [], [], [], []
        for label in (0, 1):
            samples = z[rows][labels[rows] == label]
            count = len(samples)
            raw = np.cov(samples, rowvar=False, ddof=1)
            ridge = max(cfg["ridge_min"], cfg["ridge_relative"] * raw.diagonal().mean())
            regularized = (1 - cfg["shrinkage"]) * raw
            regularized[np.diag_indices_from(regularized)] = raw.diagonal() + ridge
            means.append(samples.mean(axis=0))
            covs.append(regularized)
            ridges.append(ridge)
            counts.append(count)
        return np.asarray(means), np.asarray(covs), ridges, counts

    def log_ratio(points, means, covs):
        scores = []
        for mean, cov in zip(means, covs):
            diff = points - mean
            solution = cho_solve((np.linalg.cholesky(cov), True), diff.T).T
            scores.append(-0.5 * (np.sum(diff * solution, axis=-1) + np.linalg.slogdet(cov)[1]))
        return scores[0] - scores[1]

    gaussians = saved["gaussian"]
    raw_oof = np.empty(len(z), dtype=np.float64)
    groups = [(np.arange(len(z)), np.array([], dtype=np.int64)), *parts]
    for index, (fit, hold) in enumerate(groups):
        means, covs, ridges, counts = stats(fit)
        close(gaussians["mean"][index], means, "Frozen class mean")
        close(gaussians["covariance"][index], covs, "Frozen class covariance")
        close(gaussians["ridge"][index], ridges, "Frozen class ridge")
        eq(gaussians["count"][index], counts, "Frozen Gaussian class counts")
        close(gaussians["condition"][index], np.linalg.cond(covs), "Frozen Gaussian conditioning")
        factors = gaussians["cholesky"][index]
        close(factors @ factors.transpose(0, 2, 1), covs, "Frozen Cholesky reconstruction")
        close(gaussians["logdet"][index], np.linalg.slogdet(covs)[1], "Frozen log determinant")
        distances = [
            np.einsum("nd,nd->n", z[fit] - mean, cho_solve((factor, True), (z[fit] - mean).T).T)
            for mean, factor in zip(means, factors)
        ]
        support = max(cfg["ridge_min"], np.quantile(np.minimum(*distances), cfg["support_quantile"]))
        close(gaussians["support"][index], support, "Class evidence support threshold")
        raw_oof[hold] = log_ratio(z[hold], means, covs)
    close(raw_oof, saved["oof_llr"], "OOF held-out LLR differs", atol=1e-8)

    def verify_calibrator(fit_raw, fit_y, calibration):
        expected_scale = max(float(np.sqrt(np.square(fit_raw).mean())), cfg["calibration_scale_min"])
        scale, slope = calibration
        close(scale, expected_scale, "OOF calibrator scale differs")
        h.require(0 <= slope <= cfg["calibration_max_slope"], "OOF calibrator sign or range differs")
        x = fit_raw / scale
        gradient = (expit(slope * x) - (fit_y == 0)) * x
        optimality = np.mean([gradient[fit_y == c].mean() for c in (0, 1)]) + cfg["calibration_relative_l2"] * slope
        if slope == 0:
            h.require(optimality >= -1e-8, "Lower-bound calibrator violates its optimum")
        elif slope == cfg["calibration_max_slope"]:
            h.require(optimality <= 1e-8, "Upper-bound calibrator violates its optimum")
        else:
            h.require(abs(optimality) <= 1e-8, "Calibrator violates its optimum")

    verify_calibrator(raw_oof, labels, saved["calibration"][0])
    out_probability = np.empty_like(raw_oof)
    for group, (fit, hold) in enumerate(parts):
        scores = np.full(len(z), np.nan)
        for other, (_, other_hold) in enumerate(parts):
            if group == other:
                continue
            fitting_rows = np.flatnonzero((assignment != group) & (assignment != other))
            means, covs, _, _ = stats(fitting_rows)
            scores[other_hold] = log_ratio(z[other_hold], means, covs)
            h.require(not np.intersect1d(fitting_rows, hold).size, "Nested calibration feeds held-out samples into a fit")
        h.require(np.isnan(scores[hold]).all() and np.isfinite(scores[fit]).all(), "Calibration self-exclusion failed")
        calibration = saved["calibration"][group + 1]
        verify_calibrator(scores[fit], labels[fit], calibration)
        out_probability[hold] = expit(raw_oof[hold] * calibration[1] / calibration[0])
    close(out_probability, saved["oof_asd"], "OOF calibration differs", atol=1e-8)
    return {
        **boundary, "seed": saved["seed"], "support": gaussians["support"].tolist(),
        "gaussian_condition": gaussians["condition"].tolist(),
        "calibration": saved["calibration"].tolist(),
        "oof_auc": float(roc_auc_score(labels == 0, raw_oof)),
        "oof_calibrated_auc": float(roc_auc_score(labels == 0, out_probability)),
        "oof_brier": float(np.mean(np.square(out_probability - (labels == 0)))),
        "inner_counts": [np.bincount(labels[hold]).tolist() for _, hold in parts],
    }



def load_historical(fold, labels, ctx, guard):
    files = {
        "DARG": (h.runner().best_path(h.BASE, fold), h.E744 / "results/fold_metrics.csv", h.BASE, 1.0),
        "E82.2-G2": (h.E822 / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt",
                    h.E822 / "results/fold_metrics.csv", "G2", 100.0),
        "E82.3.2-G2": (h.E8232 / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt",
                      h.E8232 / "results/fold_metrics.csv", "G2", 100.0),
    }
    result = {}
    for name, (ckpt, csv_path, group, scale) in files.items():
        h.require(h.sha_file(ckpt) == guard[str(ckpt)], f"Historical {name} Best changed")
        saved = h.load_torch(ckpt)
        h.require(saved["identity"] == ctx["identity"]["identity"] and saved["fold"] == fold,
                  f"Historical {name} does not match this fold")
        if "test_indices" in saved:
            eq(saved["test_indices"], ctx["state"]["test_np"], f"{name} archived test order differs")
        else:
            archived = h.historical_task(fold)["identity"]
            h.require(saved["identity"]["split_sha"] == archived["split_sha"], f"{name} historical split SHA differs")
        if name == "DARG":
            prob, pred = saved["test_probability"], saved["test_prediction"]
            best_epoch = saved["best_epoch_1based"]
        else:
            prob, pred = saved["probability"], saved["prediction"]
            best_epoch = saved["best_epoch"]
        measured = metric_values(prob, pred, labels)
        with csv_path.open(newline="") as stream:
            historic, = [row for row in csv.DictReader(stream) if row["group"] == group and int(row["fold"]) == fold]
        for key, value in measured.items():
            close(value, float(historic[key]) / scale, f"{name} audited historical {key}", atol=1e-12, rtol=0)
        result[name] = {
            "probability": prob, "prediction": pred, "metrics": measured, "best_epoch": int(best_epoch),
            "margin": (1 - 2 * labels) * (np.log(np.maximum(prob[:, 0], 1e-38)) - np.log(np.maximum(prob[:, 1], 1e-38))),
        }
    return result



def delta_analysis(patients, source, target):
    truth = np.asarray([row["label"] for row in patients])
    previous = np.asarray([row[source + "_prediction"] for row in patients])
    current = np.asarray([row[target + "_prediction"] for row in patients])
    fixed, harmed = (previous != truth) & (current == truth), (previous == truth) & (current != truth)
    delta = np.asarray([row[target + "_p_asd"] - row[source + "_p_asd"] for row in patients])
    margin = np.asarray([row[target + "_true_margin"] - row[source + "_true_margin"] for row in patients])
    changed = previous != current
    result = {
        "flips": int(changed.sum()), "corrected": int(fixed.sum()), "harmed": int(harmed.sum()),
        "net": int(fixed.sum() - harmed.sum()),
        "p_asd_abs_mean_pp": 100 * float(np.abs(delta).mean()), "p_asd_abs_max_pp": 100 * float(np.abs(delta).max()),
        "p_asd_measurably_changed": int((np.abs(delta) > CFG["comparison"]["measurable_probability_delta"]).sum()),
        "true_margin_delta_mean": float(margin.mean()), "true_margin_delta_abs_mean": float(np.abs(margin).mean()),
        "margin_improved": int((margin > 0).sum()), "margin_declined": int((margin < 0).sum()),
        "genuine_new_corrections_vs_darg": sum(bool(c and row["DARG_prediction"] != row["label"]) for c, row in zip(fixed, patients)),
    }
    for label, prefix in enumerate(("asd", "td")):
        selected = truth == label
        result[f"{prefix}_corrected"] = int((fixed & selected).sum())
        result[f"{prefix}_harmed"] = int((harmed & selected).sum())
        result[f"{prefix}_net"] = result[f"{prefix}_corrected"] - result[f"{prefix}_harmed"]
        result[f"{prefix}_true_margin_delta_mean"] = float(margin[selected].mean())
    return result



def append_handoff(body):
    path = h.HISTORY / "handoff.md"
    marker = "## Experiment 82.4 · 四组谱–语义 Prompt 特征残差".encode()
    block = body.encode()
    h.require(block.count(marker) == 1, "Handoff block identity differs")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        old = stream.read()
        target = old.find(marker)
        if target < 0:
            h.require(not old or old.endswith(b"\n"), "Handoff ending needs verification")
            chunk = b"\n" + block
            offset = len(old)
            stream.write(chunk)
        else:
            offset = target
            chunk = block
            h.require(old[offset:offset + len(chunk)] == chunk and old.count(marker) == 1, "Handoff already contains a different E82.4 block")
        stream.flush()
        os.fsync(stream.fileno())
    observed = path.read_bytes()
    h.require(observed[:offset] == old[:offset] and observed[offset:offset + len(chunk)] == chunk
              and observed.count(marker) == 1, "Independent handoff readback differs")
    return {"status": "PASS", "path": str(path), "offset": offset,
            "append_sha": hashlib.sha256(chunk).hexdigest(), "historical_prefix_sha": hashlib.sha256(old[:offset]).hexdigest()}



def verify_gaussian(ctx, d, crossfit):
    state, reference = ctx["state"], ctx["reference"]
    z = array(d["z"]).astype(np.float64)
    assignment = np.zeros(len(z), dtype=np.int64)
    if crossfit:
        assignment[reference["legal"]] = reference["assignment"] + 1
    g = reference["gaussian"]
    expected_llr, expected_p, expected_support = (np.empty(len(z)) for _ in range(3))
    expected_direction = np.empty((len(z), 2, 16))
    for index in np.unique(assignment):
        rows = assignment == index
        distances = []
        for cls in (0, 1):
            diff = z[rows] - g["mean"][index, cls]
            direction = cho_solve((g["cholesky"][index, cls], True), -diff.T).T
            expected_direction[rows, cls] = direction
            distances.append(np.sum(-diff * direction, axis=1))
        asd, td = distances
        raw = (td - asd + g["logdet"][index, 1] - g["logdet"][index, 0]) * 0.5
        scale, slope = reference["calibration"][index]
        expected_llr[rows] = raw
        expected_p[rows] = expit(raw * slope / scale)
        expected_support[rows] = np.exp(-CFG["crossfit"]["support_decay"]
                                        * np.maximum(0, np.minimum(asd, td) / g["support"][index] - 1))
    expected_reliability = np.abs(2 * expected_p - 1) * expected_support
    expected_evidence = expected_p[:, None] * expected_direction[:, 0] + (1 - expected_p[:, None]) * expected_direction[:, 1]
    for name, expected in (("llr", expected_llr), ("p_asd", expected_p), ("class_directions", expected_direction),
                           ("evidence", expected_evidence), ("support", expected_support), ("reliability", expected_reliability)):
        close(array(d[name]), expected, f"Independent float64 Gaussian {name}", atol=4e-4, rtol=3e-4)
    if crossfit:
        legal = state["legal_np"]
        h.require((assignment[legal] > 0).all() and not assignment[state["test_np"]].any(),
                  "A formal test row selected an inner calibration model")
    return float(np.abs(array(d["llr"]) - expected_llr).max())


def float32_dot(actual, left, right, name, *, preceding_rounding=0):
    """Compare a float32 dot product to FP64 with an elementwise roundoff bound."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    operations = 2 * left.shape[-1] + preceding_rounding
    unit = np.finfo(np.float32).eps / 2
    gamma = operations * unit / (1 - operations * unit)
    error = np.abs(actual - left @ right)
    bound = gamma * (np.abs(left) @ np.abs(right)) + 2 * unit
    ratio = float(np.max(error / bound))
    h.require(ratio <= 1, f"{name} exceeds float32 dot-product error bound ({ratio:.3f}×)")
    return float(np.max(error)), ratio


def verify_branch(ctx, cam, d, crossfit):
    state, module = ctx["state"], ctx["module"]
    group = ctx["method"]["group"]
    U, center = (array(getattr(module, k)).astype(np.float64) for k in ("basis", "center"))
    C = array(cam).astype(np.float64)
    z, delta_z = (array(d[k]).astype(np.float64) for k in ("z", "delta_z"))
    attention = array(d["attention"])
    z_error, z_bound_fraction = float32_dot(z, C - center, U, "Original O coordinates", preceding_rounding=2)
    h.require(attention.shape == (len(C), 8, 17) and np.isfinite(attention).all(),
              "Invalid 16-token plus null attention")
    close(attention.sum(-1), 1, "8x17 normalized attention", atol=2e-6)
    coverage = 1 - np.prod(1 - attention[..., :16].astype(np.float64), axis=1)
    weights = coverage / coverage.sum(-1, keepdims=True)
    close(array(d["coverage"]), coverage, "Eight-prompt union coverage", atol=5e-7)
    close(array(d["weights"]), weights, "Normalized coverage direction weights", atol=3e-7)
    observed_weights = array(d["weights"]).astype(np.float64)
    raw = delta_z @ U.T
    raw[state["excluded"]] = 0
    cap = CFG["residual"]["norm_limit"] * np.linalg.norm(C, axis=1, keepdims=True)
    r = np.linalg.norm(raw, axis=1, keepdims=True) / np.maximum(cap, 1e-8)
    smooth = np.divide(np.tanh(r), r, out=np.ones_like(r), where=r > 0)
    expected_delta = raw * smooth * (cap / np.maximum(cap, 1e-8))
    delta = array(d["delta"])
    close(delta, expected_delta, "CAM144 projection, smooth limit, or Fold 2 mask", atol=5e-6)
    close(delta @ U @ U.T, delta, "Enhanced CAM residual leaves Original O", atol=5e-6)
    h.require(np.isfinite(delta).all() and float(array(d["relative"]).max()) <= CFG["residual"]["norm_limit"] + 2e-6
              and not np.count_nonzero(delta[state["excluded"]]), "Invalid applied residual")
    formula = {"status": "PASS", "independent_delta_max_abs_error": float(np.abs(delta - expected_delta).max()),
               "projection_float32_max_abs_error": z_error,
               "projection_roundoff_bound_max_fraction": z_bound_fraction}

    if group == "G1":
        R = array(d["rotation"]).astype(np.float64)
        t = np.swapaxes(R, -1, -2)
        close(t @ R, np.broadcast_to(np.eye(16), R.shape), "Patient Cayley orthogonality", atol=8e-7)
        close(array(d["coordinates"]), (R @ z[..., None])[..., 0], "Rotated coordinates", atol=7e-6)
        close(array(d["directions"]), R @ U.T, "Rotated token axes", atol=1e-6)
        close(delta_z, (t @ ((weights - 1 / 16) * array(d["h"]))[..., None])[..., 0],
              "Inverse-rotated centered Prompt residual", atol=4e-6)
        Q = array(module.query)
        Q_difference = Q - ctx["method"]["query_common_shrinkage"] * Q.mean(axis=0)
        close(array(d["effective_queries"]), Q_difference / np.linalg.norm(Q_difference, axis=1, keepdims=True),
              "Anchored Query common component shrinkage", atol=1e-6)
        formula["rotation_orthogonality_error"] = float(np.abs(t @ R - np.eye(16)).max())
    elif group == "G2":
        loadings = np.square(U.reshape(16, 9, 16)).sum(axis=1)
        close(array(module.slot_loadings), loadings, "Original contiguous CAM slot loadings", atol=2e-7)
        affinity_error, affinity_ratio = float32_dot(array(d["affinity"]) / 16, observed_weights,
                                                     array(module.slot_loadings).T, "Prompt-local affinity")
        local_z = array(d["local_z"]).astype(np.float64)
        float32_dot(local_z, array(d["local_raw"]), U, "Local candidate spectrum")
        formula.update(local_affinity_float32_max_abs_error=16 * affinity_error,
                       local_affinity_roundoff_bound_max_fraction=affinity_ratio)
        beta = array(d["beta"])
        floor = ctx["method"]["local_mix_floor"]
        h.require((beta >= floor - 1e-7).all() and (beta <= 1 - floor + 1e-7).all(), "Local/global weights out of bounds")
        global_weights = array(d["spectral_weights"]) * (1 - array(d["psce_mix"])[:, None]) + observed_weights * array(d["psce_mix"])[:, None]
        global_z = (global_weights - 1 / 16) * z
        close(array(d["global_z"]), global_z, "Historical PSCE global branch", atol=4e-6)
        close(delta_z, (1 - beta) * global_z + beta * local_z, "Local/global spectral fusion", atol=4e-6)
        formula["local_projection_energy"] = float(np.square(local_z).sum()
                                                   / max(np.square(array(d["local_raw"])).sum(), 1e-12))
    else:
        formula["gaussian_llr_max_abs_error"] = verify_gaussian(ctx, d, crossfit)
        expected_selection = array(d["coverage"]).astype(np.float64) / (1 - (16 / 17)**8) * (
            16 * observed_weights)**ctx["method"]["selection_power"]
        close(array(d["selection"]), expected_selection, "Prompt selectivity weight", atol=3e-6)
        if group == "G3":
            close(delta_z, array(d["control"]) * expected_selection * array(d["h"]),
                  "Directional control and semantic multiplication", atol=6e-6)
        else:
            V, a, e = (array(d[key]).astype(np.float64) for key in ("factors", "operator_weights", "candidate"))
            gram = V.T @ V
            close(gram, np.eye(4), "Rank-four factors orthogonality", atol=8e-7)
            float32_dot(array(d["semantic_coefficients"]), array(d["selection"]), np.square(V),
                        "Original Prompt contribution to operator weights", preceding_rounding=2)
            operator = (V[None] * a[:, None, :]) @ V.T
            close(delta_z, (operator @ e[..., None])[..., 0], "V diag(a) V^T candidate", atol=2e-6)
            formula.update(operator_gram=gram.tolist(), operator_column_rank=int(np.linalg.matrix_rank(V)))
    return formula


@torch.no_grad()
def checkpoint_mechanisms(ctx, best):
    state, module, method = ctx["state"], ctx["module"], ctx["method"]
    name = method["group"]
    model_before, rng_before = h.sha(run.state_dict(ctx)), h.rng_sha()
    rows = state["test_np"]
    labels = state["labels_np"][rows]
    original, capture = run.forward(state, False)
    cam = capture["cam"]
    head = state["cam"].Outputlayer
    h.require(not head.training and torch.equal(run.frozen_head(head, cam), capture["logits"]),
              "No-Dropout auxiliary teacher differs from native head")
    original_logits = capture["logits"]

    def summary(logp, logits):
        probability = array(logp[rows].exp())
        hard = probability.argmax(-1)
        logits = array(logits[rows])
        return {
            "probability": probability, "prediction": hard, "logits": logits,
            "margin": (logits[:, 0] - logits[:, 1]) * (1 - 2 * labels),
            "metrics": metric_values(probability, hard, labels), "best_epoch": best["best_epoch"],
        }

    result = {"residual_off": summary(original["logp"], original_logits)}
    interventions = {
        "full": {},
        "query_replaced": {"query": ctx["text"]["q_anonymous"].to(cam.device)},
        "semantic_off": {"semantic_off": True},
        "method_variant": {"variant": method["variant"]},
    }
    if name == "G1":
        interventions["query_common_kept"] = {"variant": "common_kept"}
    details = {}
    for key, flags in interventions.items():
        enhanced, view = module(cam, crossfit=False, **flags)
        logp, _ = head(enhanced, None)
        logits = head.classifier(head.mlp_head(enhanced))
        h.require(torch.equal(logp, F.log_softmax(logits, dim=-1)), "Native classifier replay differs")
        result[key] = summary(logp, logits)
        details[key] = view
        h.require(not bool(view["delta"][state["excluded"]].count_nonzero()), "Fold 2 excluded residual not exactly zero")
        if state["excluded"]:
            selected = np.isin(rows, state["excluded"])
            eq(result[key]["probability"][selected], result["residual_off"]["probability"][selected],
               "Fold 2 probability changed after residual intervention")
    for key in ("probability", "prediction", "logits"):
        eq(result["full"][key], best["evaluation"][key], f"Independent Best replay {key} differs")
    h.require(int((result["full"]["prediction"] == labels).sum()) == best["evaluation"]["correct"], "Best correct count")
    full = details["full"]
    math_checks = verify_branch(ctx, cam, full, crossfit=False)
    attention = array(full["attention"][rows])
    coverage = array(full["coverage"][rows])
    normal = attention[:, :, :16] / np.linalg.norm(attention[:, :, :16], axis=-1, keepdims=True)
    cosines = normal @ normal.transpose(0, 2, 1)
    relative = array(full["relative"][rows])
    residual = full["delta"][rows]
    margin_delta = result["full"]["margin"] - result["residual_off"]["margin"]
    response = {
        "attention_entropy": float(-(attention * np.log(np.maximum(attention, 1e-8))).sum(-1).mean() / np.log(17)),
        "attention_uniform_l1": float(np.abs(attention - 1 / 17).sum(-1).mean()),
        "prompt_pairwise_cosine": float(cosines[:, np.triu_indices(8, 1)[0], np.triu_indices(8, 1)[1]].mean()),
        "prompt_pairwise_matrix": cosines.mean(0).tolist(),
        "null_mean": float(attention[:, :, 16].mean()), "null_per_prompt": attention[:, :, 16].mean(0).tolist(),
        "coverage_mean": float(coverage.mean()), "coverage_sd": float(coverage.std()),
        "coverage_per_direction": coverage.mean(0).tolist(),
        "residual_relative_mean": float(relative.mean()), "residual_relative_max": float(relative.max()),
        "residual_relative_gt_1e_6": int((relative > 1e-6).sum()),
        "residual_relative_quantiles": np.quantile(relative, [.0, .1, .5, .9, 1.]).tolist(),
        "residual_logit_max_abs": float(np.abs(result["full"]["logits"] - result["residual_off"]["logits"]).max()),
        "residual_mean_true_margin_change": float(margin_delta.mean()),
        **{k: float(v) for k, v in run.method_diagnostics(name, full, rows).items()},
    }
    for key, intervention in details.items():
        if key == "full":
            continue
        response[key + "_max_abs_delta_cam_change"] = float((residual - intervention["delta"][rows]).abs().amax())
        response[key + "_mean_abs_delta_cam_change"] = float((residual - intervention["delta"][rows]).abs().mean())
        response[key + "_mean_l1_weights"] = float((full["weights"][rows] - intervention["weights"][rows]).abs().sum(-1).mean())
        if name == "G2":
            response[key + "_mean_l1_slot_affinity"] = float(
                (full["affinity"][rows] - intervention["affinity"][rows]).abs().sum(-1).mean())
        if name == "G4":
            response[key + "_mean_l1_operator_weights"] = float(
                (full["operator_weights"][rows] - intervention["operator_weights"][rows]).abs().sum(-1).mean())
    if name == "G2":
        affinity = array(full["affinity"][rows])
        response.update(
            slot_affinity_mean=affinity.mean(axis=0).tolist(),
            slot_affinity_population_sd=affinity.std(axis=0).tolist(),
            slot_affinity_quantiles=np.quantile(affinity, [.05, .5, .95], axis=0).tolist(),
        )
    if name in ("G3", "G4"):
        llr = array(full["llr"][rows])
        post = array(full["p_asd"][rows])
        response.update(
            llr_quantiles=np.quantile(llr, [.0, .05, .5, .95, 1.]).tolist(),
            posterior_saturation=float(((post < .01) | (post > .99)).mean()),
            controls_mean_per_direction=array(full["control"][rows]).mean(axis=0).tolist(),
        )
    if name == "G4":
        weights = array(full["operator_weights"][rows])
        response.update(operator_weight_mean=weights.mean(axis=0).tolist(),
                        operator_weight_quantiles=np.quantile(weights, [.05, .5, .95], axis=0).tolist())
        legal, y = state["legal_np"], state["labels_np"][state["legal_np"]]
        base = array(original_logits)
        half = array(run.frozen_head(head, cam + .5 * full["delta"]))
        complete = array(run.frozen_head(head, cam + full["delta"]))

        def true_margin(values, ind, ground_truth):
            return (values[ind, 0] - values[ind, 1]) * (1 - 2 * ground_truth)
        y0 = true_margin(base, legal, y)
        y1 = true_margin(half, legal, y)
        y2 = true_margin(complete, legal, y)
        protect = 1 - np.exp(-np.maximum(y0, 0) / CFG["protection"]["margin_scale"])
        tolerance = CFG["protection"]["allowed_margin_drop"]
        response.update(
            stage_half_test_margin=float((true_margin(half, rows, labels) - true_margin(base, rows, labels)).mean()),
            stage_full_minus_half_test_margin=float((true_margin(complete, rows, labels) - true_margin(half, rows, labels)).mean()),
            protected_correct_train=int((protect > 0).sum()),
            protected_train_half_declines=int(((protect > 0) & (y1 < y0 - tolerance)).sum()),
            protected_train_second_half_declines=int(((protect > 0) & (y2 < y1 - tolerance)).sum()),
            difficult_train_half_gain=float((y1 - y0)[y0 <= 0].mean()) if (y0 <= 0).any() else None,
            difficult_train_second_half_gain=float((y2 - y1)[y0 <= 0].mean()) if (y0 <= 0).any() else None,
        )
    h.require(h.rng_sha() == rng_before and h.sha(run.state_dict(ctx)) == model_before,
              "Read-only mechanism audit changed RNG or parameters")
    return result, response, math_checks, relative


def audit_journal(ctx, task, best, latest, preflight, recovery):
    name, fold = ctx["method"]["group"], ctx["state"]["fold"]
    p = run.paths(name, fold)
    raw = p["journal"].read_bytes()
    lines = raw.splitlines(keepends=True)
    h.require(len(raw) == latest["journal_size"] and hashlib.sha256(raw).hexdigest() == latest["journal_sha"],
              "Committed training journal differs")
    records = [json.loads(line) for line in lines]
    end = latest["next_epoch"]
    eq([row["epoch"] for row in records], list(range(run.START + 1, end + 1)), "Training epoch trajectory differs")
    settings = ctx["method"]["auxiliary"]
    summed_cap = sum(spec["gradient_cap"] for spec in settings.values())
    maxima = {}
    if recovery is not None:
        preceding = recovery["epoch"] - run.START
        prefix = b"".join(lines[:preceding])
        h.require(hashlib.sha256(prefix).hexdigest() == recovery["journal_sha"], "Recovered journal prefix differs")
        h.require(records[preceding - 1]["rng"]["rng_after_eval"] == recovery["rng_sha"]
                  and records[preceding]["rng"]["rng_before"] == recovery["rng_sha"],
                  "Training RNG discontinuity at automatic recovery")

    arch = {row["epoch"]: row for row in h.historical_task(fold)["rng_trace"]}
    calls = preflight["dropout_calls"]
    inputs = [
        torch.empty_strided(call["shape"], call["stride"], device=ctx["state"]["x"].device,
                            dtype=getattr(torch, call["dtype"].split(".")[-1])).fill_(1)
        for call in calls
    ]
    h.require(len(inputs) == 6 and all(t.dtype == torch.float32 for t in inputs), "Incomplete native dropout inventory")
    original_rng = h.base().pack_rng()
    count, selected, wait, archived_epochs = -1, 0, 0, 0
    try:
        h.base().restore_rng(ctx["warmup"]["rng"])
        for record in records:
            epoch, rng = record["epoch"], record["rng"]
            h.require(h.rng_sha() == rng["rng_before"], f"RNG pre-phase replay failed at {epoch}")
            for source, call in zip(inputs, calls):
                F.dropout(source, p=call["p"], training=True, inplace=call["inplace"])
            h.require(h.rng_sha() == rng["rng_after_train"] == rng["rng_after_eval"], f"RNG schedule replay failed at {epoch}")
            if epoch in arch:
                h.require(rng == arch[epoch], f"Historical RNG differs at {epoch}")
                archived_epochs += 1
            if record["correct"] > count:
                count, selected, wait = record["correct"], epoch, 0
            else:
                wait += 1
            h.require((count, selected, wait) == (record["best_correct"], record["best_epoch"], record["wait"]),
                      "Best selection or early stop counter differs")
            h.require(wait <= CFG["early_stop_wait_gt"] or epoch == end, "Training exceeded early stopping")
            diag = record["diagnostics"]
            h.require(all(math.isfinite(value) for value in diag.values()), "Nonfinite training journal diagnostics")
            h.require(diag["delta_relative_max"] <= CFG["residual"]["norm_limit"] + 2e-6, "Applied residual exceeded bound")
            for family in ("prompt", "residual"):
                watched = {f"{family}_applied_ratio": summed_cap}
                watched.update({f"{family}_{term}_applied_ratio": spec["gradient_cap"] for term, spec in settings.items()})
                for key, cap in watched.items():
                    h.require(diag[key] <= cap + 2e-6, "Logged gradient contribution exceeded budget")
                    maxima[key] = max(maxima.get(key, 0.0), diag[key])
                for term in settings:
                    key = f"{family}_{term}_raw_ratio"
                    maxima[key] = max(maxima.get(key, 0.0), diag[key])
        h.require(h.rng_sha() == latest["rng_sha"], "Terminal RNG does not match replay")
    finally:
        h.base().restore_rng(original_rng)
    h.require(selected == best["best_epoch"] == latest["best_epoch"] == task["best_epoch"]
              and count == best["evaluation"]["correct"] == latest["best_correct"], "Best selection differs")
    h.require(end == task["epoch"] and (end == CFG["max_epochs"] or wait > CFG["early_stop_wait_gt"])
              and wait == latest["wait"], "Termination differs from historical protocol")
    h.require(best["rng_sha"] == records[selected - run.START - 1]["rng"]["rng_after_eval"]
              and records[0]["diagnostics"] == preflight["epoch11_diagnostics"], "Best RNG or initial real step differs")
    final = records[-max(1, len(records) // 4):]
    watched = ("cls", "attention_entropy", "prompt_pairwise_cosine", "null_mean", "coverage_mean", "margin_change",
               "delta_relative_mean", "delta_nonzero_fraction", "prompt_primary", "residual_primary", *settings)
    return {
        "best_epoch": selected, "end_epoch": end, "epochs": len(records), "historical_rng_phases": archived_epochs,
        "dropout_replayed_epochs": len(records), "gradient_ratios": maxima,
        "final_quartile": {name: float(np.mean([row["diagnostics"][name] for row in final])) for name in watched},
        "actual_restart_epoch": recovery["epoch"] if recovery else None,
    }


def checkpoint_gradient(ctx):
    """Check actual CE on the reloaded model and the adjoint through Delta C."""
    state, branch = ctx["state"], ctx["module"]
    end_to_end = run.forward(state, True, branch)
    out, capture = end_to_end
    current = state["model"]
    native = {
        "head": tuple(state["cam"].Outputlayer.parameters()),
        "cam": tuple(p for name, p in state["cam"].named_parameters()
                     if p.requires_grad and name.startswith(("InputLayer.", "encode_", "feed_"))),
        "gfr": tuple(current.base_parameters()),
        "darg": tuple(p for name, p in current.added_named_parameters()
                      if p.requires_grad and name.startswith(("encoders.", "scorers.", "adaptive_gate_logits"))),
    }
    shared = tuple(p for values in native.values() for p in values)
    branch_params = dict(branch.named_parameters())
    cam, enhanced = capture["cam"], capture["enhanced"]
    requested = (*shared, *branch_params.values(), cam, enhanced)
    gradients = torch.autograd.grad(out["cls"], requested, retain_graph=True, allow_unused=True)
    adjoint = torch.autograd.grad(capture["details"]["delta"], (cam, *shared),
                                 grad_outputs=gradients[-1].detach(), allow_unused=True)

    def grad_norm(grads):
        pieces = [g.detach().double().square().sum() for g in grads if g is not None]
        return float(torch.stack(pieces).sum().sqrt()) if pieces else 0.0

    stats = {"cam144": grad_norm([gradients[-2]]), "enhanced": grad_norm([gradients[-1]]),
             "delta_to_cam144": grad_norm([adjoint[0]])}
    offset = 0
    for family, params in native.items():
        stats[family] = grad_norm(gradients[offset:offset + len(params)])
        stats["delta_to_" + family] = grad_norm(adjoint[1 + offset:1 + offset + len(params)])
        offset += len(params)
    # Delta C is computed before the original final head is applied.
    h.require(stats.pop("delta_to_head") == 0, "The residual unexpectedly reads the original head parameters")
    by_param = dict(zip(branch_params.values(), gradients[offset:-2]))
    for family, params in branch.gradient_groups().items():
        stats[family] = grad_norm([by_param[p] for p in params])
    h.require(all(math.isfinite(v) and v > 0 for v in stats.values()), f"CE VJP lost an actual path: {stats}")

    zeros, structural = [], []
    for name, parameter in branch_params.items():
        actual = by_param[parameter]
        h.require(actual is not None and bool(torch.isfinite(actual).all()), f"CE path disconnected: {name}")
        norm = grad_norm([actual])
        if ctx["method"]["group"] == "G2" and name == "spectral.2.bias":
            # E82.2's shared final scalar bias translates every direction score
            # by the same amount. The following softmax cancels this exactly.
            final = branch.spectral[-1]
            h.require(final.bias is parameter and parameter.numel() == 1 and final.out_features == 1
                      and norm < 1e-8, "Historical PSCE shift-invariant bias showed a functional gradient")
            structural.append({"parameter": name, "ce_norm": norm, "identity": "softmax(s + scalar) = softmax(s)"})
        elif norm == 0:
            zeros.append(name)
    h.require(not zeros, f"Functionally active branch parameters have zero CE gradient: {zeros}")
    return {**stats, "active_branch_tensors": len(branch_params) - len(structural),
            "structural_null_parameters": structural}


def replay_latest(ctx, latest, manifest):
    implementation = manifest["implementation"]
    state, model = ctx["state"], ctx["module"]
    name, fold = ctx["method"]["group"], state["fold"]
    run.restore(ctx, latest, implementation)
    h.require(h.sha(run.state_dict(ctx)) == h.sha(latest["state"]), "Latest state readback differs")
    tested = run.evaluate(ctx)
    for key in tested:
        eq(tested[key], latest["evaluation"][key], f"Latest evaluation differs: {key}")
    h.require(h.rng_sha() == latest["rng_sha"], "Latest eval advanced RNG")
    with torch.no_grad():
        _, original = run.forward(state, False)
        cam = original["cam"]
        _, d = model(cam, crossfit=True)
        training_formula = verify_branch(ctx, cam, d, crossfit=True)
    del original, cam, d

    path = checkpoint_gradient(ctx)
    run.restore(ctx, latest, implementation)

    def transition():
        diagnostic, _ = run.train_step(ctx)
        after_train = h.rng_sha()
        prediction = run.evaluate(ctx)
        h.require(after_train == h.rng_sha(), "Continuation evaluation advanced RNG")
        return h.sha((diagnostic, prediction, run.state_dict(ctx))), after_train

    first, first_rng = transition()
    # Load the committed file again and repeat the state transition, independently of its previous outputs.
    reopened = h.load_torch(run.paths(name, fold)["latest"])
    run.restore(ctx, reopened, implementation)
    second, second_rng = transition()
    h.require((first, first_rng) == (second, second_rng), "Latest next train/eval transition differs on repeat")
    if latest["next_epoch"] < CFG["max_epochs"]:
        archived = {row["epoch"]: row for row in h.historical_task(fold)["rng_trace"]}
        if (expected := archived.get(latest["next_epoch"] + 1)) is not None:
            h.require(first_rng == expected["rng_after_train"], "Latest continuation is off the archival RNG trajectory")
    run.restore(ctx, reopened, implementation)
    return {"status": "PASS", "next_transition_sha": first, "next_rng_sha": first_rng,
            "next_epoch_probe": latest["next_epoch"] + 1, "class_gradient": path, "training_formulas": training_formula}


def audit_source(manifest):
    actual = h.sha_file(HERE / "audit.py")
    score = manifest.get("audit_sha", actual)
    h.require(actual == manifest.get("audit_publish_sha", score), "Frozen audit implementation differs")
    return score, actual


def audit_fold(group, fold):
    manifest = h.read_json(run.MANIFEST)
    p = run.paths(group, fold)
    task, preflight = h.read_json(p["task"]), h.read_json(p["preflight"])
    scoring_sha, source_sha = audit_source(manifest)
    h.require(task["implementation"] == preflight["implementation"] == manifest["implementation"]
              == h.implementation_hashes(), "Frozen audit or training sources differ")
    h.require(task["status"] == "COMPLETE" and preflight["status"] == "PASS", "Fold did not complete")
    h.require(h.sha_file(p["best"]) == task["best_sha"] and h.sha_file(p["latest"]) == task["latest_sha"],
              "Best or Latest SHA differs")
    best, latest = h.load_torch(p["best"]), h.load_torch(p["latest"])
    h.require(best["protected_sha"] == latest["protected_sha"] == manifest["protected_bundle_sha"]
              and latest["best_sha"] == task["best_sha"], "Shared protected-file provenance differs")
    ctx = run.setup(group, fold, validate_spectrum=True)
    state = ctx["state"]
    h.require(ctx["identity"] == preflight["identity"], "Warmup, Prompt, spectrum or statistics identity differs")
    expected_buffers = h.sha(dict(ctx["module"].named_buffers()))
    reference = reference_split(ctx)
    labels = state["labels_np"][state["test_np"]]
    historical = load_historical(fold, labels, ctx, manifest["protected_sha256"])
    run.restore(ctx, best, manifest["implementation"])
    h.require(h.sha(run.state_dict(ctx)) == h.sha(best["state"]), "Best full state did not reload")
    h.require(h.sha(dict(ctx["module"].named_buffers())) == expected_buffers, "Best reference buffers drifted")
    results, response, formula, relative = checkpoint_mechanisms(ctx, best)
    recoveries = [wave.get("slow_context_restores", {}).get(str(fold))
                  for wave in manifest["training_runs"] if wave["group"] == group]
    recovery = next((r for r in reversed(recoveries) if r is not None), None)
    progress = audit_journal(ctx, task, best, latest, preflight, recovery)
    restored = replay_latest(ctx, latest, manifest)
    h.require(h.sha(dict(ctx["module"].named_buffers())) == expected_buffers, "Latest reference buffers drifted")
    variants = {**historical, **results}
    patients = []
    for j, row in enumerate(state["test_np"]):
        patient = {"row_1based": int(row) + 1, "label": int(labels[j]), "residual_relative": float(relative[j])}
        for key, measured in variants.items():
            patient[f"{key}_prediction"] = int(measured["prediction"][j])
            patient[f"{key}_p_asd"] = float(measured["probability"][j, 0])
            patient[f"{key}_true_margin"] = float(measured["margin"][j])
        patients.append(patient)
    report = {
        "status": "PASS", "group": group, "fold": fold, "best_exact": True, "rng_status": "PASS",
        "best_sha": task["best_sha"], "latest_sha": task["latest_sha"], "audit_sha": scoring_sha, "audit_code_sha": source_sha,
        "warmup_sha": ctx["identity"]["identity"]["warmup_sha256"], "source_and_protected_identity": "PASS",
        "membership_sha": ctx["identity"]["membership_sha"], "spectrum_sha": ctx["identity"]["spectrum_sha"],
        "prompt_sha": ctx["identity"]["query_sha"], "reference_sha": ctx["identity"]["reference_sha"],
        "parameters": ctx["identity"]["trainable_parameters"], "reference": reference, "formulas": formula,
        "mechanism": response, "progress": progress, "latest": restored,
        "metrics": {key: measured["metrics"] for key, measured in variants.items()},
        "best_epochs": {key: measured["best_epoch"] for key, measured in variants.items()}, "patients": patients,
        "initial_gradient": preflight["gradient_paths"], "initial_epoch_diagnostics": preflight["epoch11_diagnostics"],
    }
    h.atomic_json(p["audit"], report)
    print(f"PASS {group} fold {fold}: Best, Latest, gradient and {progress['epochs']} RNG phases", flush=True)


def comparisons_for(group):
    result = [(name, "full") for name in HISTORICAL]
    result += [("DARG", "residual_off"), ("residual_off", "full"), ("query_replaced", "full"),
               ("semantic_off", "full"), ("method_variant", "full")]
    if group == "G1":
        result.append(("query_common_kept", "full"))
    return result


def summarize(all_audits):
    complete_metrics, paired, per_fold_pairs = {}, {}, []
    for group, audits in all_audits.items():
        complete_metrics[group], paired[group] = {}, {}
        test_rows = sorted(p["row_1based"] for row in audits for p in row["patients"])
        eq(test_rows, list(range(1, CFG["subjects"] + 1)), "Outer folds do not cover each patient exactly once")
        for scope, selected in zip(SCOPES, (audits, [a for a in audits if a["fold"] != 2])):
            all_patients = [p for row in selected for p in row["patients"]]
            metrics, comp = {}, {}
            for method in selected[0]["metrics"]:
                measures = np.asarray([[a["metrics"][method][k] * 100 for k in h.METRICS] for a in selected])
                metrics[method] = {
                    k: {"mean": float(mean), "sd": float(sd)}
                    for k, mean, sd in zip(h.METRICS, measures.mean(0), measures.std(0, ddof=0))
                }
            for previous, current in comparisons_for(group):
                by_metric = {}
                differences = np.asarray([
                    [(a["metrics"][current][key] - a["metrics"][previous][key]) * 100 for key in h.METRICS]
                    for a in selected
                ])
                differences[np.abs(differences) < 1e-10] = 0
                for k, val in zip(h.METRICS, differences.T):
                    by_metric[k] = {"mean_pp": float(val.mean()), "sd_pp": float(val.std(ddof=0)),
                                    "wins": int((val > 0).sum()), "ties": int((val == 0).sum()), "losses": int((val < 0).sum())}
                comp[f"{previous}->{current}"] = {"metrics": by_metric, **delta_analysis(all_patients, previous, current)}
                if scope == SCOPES[0]:
                    for a, values in zip(selected, differences):
                        per_fold_pairs.append({
                            "group": group, "fold": a["fold"], "from": previous, "to": current,
                            **{key + "_delta_pp": float(value) for key, value in zip(h.METRICS, values)},
                            **delta_analysis(a["patients"], previous, current),
                        })
            complete_metrics[group][scope] = metrics
            paired[group][scope] = comp
    for group in METHODS[1:]:
        for scope in SCOPES:
            for history in HISTORICAL:
                h.require(complete_metrics[group][scope][history] == complete_metrics[METHODS[0]][scope][history],
                          "Historical references disagree across independent group audits")
    return complete_metrics, paired, per_fold_pairs


def scientific_decision(audits, paired):
    decisions = {}
    for group in METHODS:
        inputs = paired[group][SCOPES[0]]
        overall, direct, query = (inputs[key] for key in ("DARG->full", "residual_off->full", "query_replaced->full"))
        sensitivity = paired[group][SCOPES[1]]["DARG->full"]
        performance_go = {
            "acc_gain": overall["metrics"]["acc"]["mean_pp"] > 0,
            "ba_gain": overall["metrics"]["hard_auc"]["mean_pp"] > 0,
            "sen_gain": overall["metrics"]["sen"]["mean_pp"] > 0,
            "net_corrections": overall["net"] > 0,
            "exclude_fold2_acc_and_net_gain": sensitivity["metrics"]["acc"]["mean_pp"] > 0 and sensitivity["net"] > 0,
        }
        variant_max = max(row["mechanism"]["method_variant_max_abs_delta_cam_change"] for row in audits[group])
        mechanism_go = {
            "direct_positive_hard_net": direct["net"] > 0,
            "direct_ba_gain": direct["metrics"]["hard_auc"]["mean_pp"] > 0,
            "observable_query_probability_effect": query["p_asd_measurably_changed"] > 0,
            "observable_query_hard_effect": query["flips"] > 0,
            "mechanism_intervention_changes_cam": variant_max > CFG["comparison"]["minimum_mechanism_delta"],
        }
        decisions[group] = {"performance": "GO" if all(performance_go.values()) else "NO-GO",
                            "mechanism": "GO" if all(mechanism_go.values()) else "NO-GO",
                            "performance_checks": performance_go, "mechanism_checks": mechanism_go}
    return decisions


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(map(str, cells)) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def summary_table(summary, scope):
    summaries = [(hist, summary["G1"][scope][hist]) for hist in HISTORICAL]
    summaries += [(f"E82.4-{g} {h.GROUPS[g]['method']}", summary[g][scope]["full"]) for g in METHODS]
    return table(("方法", *METRIC_NAMES), [
        (name, *(f"{metrics[key]['mean']:.4f} ± {metrics[key]['sd']:.4f}" for key in h.METRICS))
        for name, metrics in summaries
    ])


def paired_row(group, comparison, values):
    numbers = values["metrics"]
    def wtl(key):
        return "/".join(str(numbers[key][k]) for k in ("wins", "ties", "losses"))
    return (group, comparison, *(f"{numbers[k]['mean_pp']:+.4f}" for k in h.METRICS), wtl("acc"), wtl("hard_auc"),
            f"{values['asd_corrected']}/{values['asd_harmed']}", f"{values['td_corrected']}/{values['td_harmed']}",
            f"{values['corrected']}/{values['harmed']}/{values['net']:+d}")


def paired_table(rows):
    return table(("组", "来源 → 目标", *METRIC_NAMES, "ACC 胜/平/负", "BA 胜/平/负",
                  "ASD 纠错/误伤", "TD 纠错/误伤", "纠错/误伤/净"), rows)


def fold_mean(rows, key, section="mechanism"):
    return float(np.mean([r[section][key] for r in rows]))


def group_mechanism(group, rows):
    def mean(key):
        return fold_mean(rows, key)
    if group == "G1":
        keep = mean("query_common_kept_mean_abs_delta_cam_change")
        return (f"G1 的患者旋转偏离单位矩阵平均 Frobenius 范数 {mean('rotation_deviation'):.6f}，患者间旋转散布 "
                f"{mean('rotation_patient_variation'):.6f}。Query 间 cosine {mean('query_cosine'):.6f}，谱方向选择 "
                f"cosine {mean('prompt_pairwise_cosine'):.6f}。保留未收缩 Query 的固定权重干预使单坐标残差平均变化 {keep:.7f}。")
    if group == "G2":
        return (f"G2 的局部融合 β 均值 {mean('local_beta_mean'):.4f}，患者内折标准差均值 {mean('local_beta_sd'):.4f}。"
                f"局部候选回投后保留 {100 * mean('local_projection_energy'):.2f}% 能量，"
                f"融合前乘权后的局部与全局谱范数分别为 {mean('local_contribution_norm'):.6f}、"
                f"{mean('global_contribution_norm'):.6f}。替换 Query 后 16 槽位权重平均 L1 变化 "
                f"{mean('query_replaced_mean_l1_slot_affinity'):.5f}。槽位是 CAM 的潜在坐标。")
    oof = fold_mean(rows, "oof_calibrated_auc", "reference")
    raw_auc = fold_mean(rows, "oof_auc", "reference")
    brier = fold_mean(rows, "oof_brier", "reference")
    condition = max(np.max(r["reference"]["gaussian_condition"]) for r in rows)
    stats = (f"OOF 原始/校准 AUC {raw_auc:.4f}/{oof:.4f}、校准 Brier {brier:.4f}，收缩协方差最大条件数 "
             f"{condition:.1f}。测试患者均值的可靠度 {mean('reliability_mean'):.4f}、后验饱和比例 "
             f"{mean('posterior_saturation'):.4f}，Prompt 权重与绝对类别谱方向的 cosine {mean('prompt_class_evidence_cosine'):.4f}。")
    if group == "G3":
        return (f"G3 的分方向控制范围均值 {mean('control_range'):.6f}，类别顺序的方向范数 "
                f"{mean('class_direction_asd_norm'):.4f}/{mean('class_direction_td_norm'):.4f}。{stats}")
    gram_error = max(r["mechanism"]["operator_orthogonality_error"] for r in rows)
    stage1 = mean("stage_half_test_margin")
    stage2 = mean("stage_full_minus_half_test_margin")
    correct = sum(r["mechanism"]["protected_correct_train"] for r in rows)
    drops1, drops2 = (sum(r["mechanism"][key] for r in rows)
                     for key in ("protected_train_half_declines", "protected_train_second_half_declines"))
    return (f"G4 各折 V 列秩为 4，Gram 最大误差 {gram_error:.2e}，四系数患者间散布 "
            f"{mean('operator_coefficients_patient_variation'):.6f}，算子非对角能量占比 {mean('operator_offdiagonal_energy'):.4f}。"
            f"测试集零→半、半→完整残差的真实类 Margin 平均变化 {stage1:+.7f}、{stage2:+.7f}。"
            f"合法训练行的正 Margin 患者折内出现 {correct} 次，两段各有 {drops1}、{drops2} 次下降超过保护容许。{stats}")


def recommendation(decisions, paired):
    kept = [g for g, v in decisions.items() if v["performance"] == v["mechanism"] == "GO"]
    if kept:
        leaders = sorted(kept, key=lambda g: paired[g][SCOPES[0]]["DARG->full"]["metrics"]["hard_auc"]["mean_pp"], reverse=True)
        label = "、".join(f"{g} {h.GROUPS[g]['method']}" for g in leaders)
        return f"{label} 达到本实验的候选规则。后续独立研究可优先验证其固定权重净纠错能否在独立选模、锁定测试、匹配医学文本对照和多 seed 下复现。"
    useful = [g for g in METHODS if decisions[g]["mechanism"] == "GO"]
    if useful:
        label = "、".join(f"{g} {h.GROUPS[g]['method']}" for g in useful)
        return f"四组均未同时达到性能和机制 GO。{label} 的固定权重信号可作为后续诊断线索。独立研究可先查明即时净收益与相对 DARG 总性能之间的落差，再用独立选模、锁定测试和匹配文本对照验证。"
    return ("四组均未同时达到性能和机制 GO，本轮无推荐的第二创新模块候选。"
            "后续独立研究若继续，优先在锁定测试上验证合法训练内识别错误病例的能力和固定权重残差净纠错，并用匹配文本对照与多 seed 评估稳定性。")


def write_tables(audits, comparisons):
    fold_metrics, mechanisms = [], []
    for group, rows in audits.items():
        for row in rows:
            names = (*HISTORICAL, "full") if group == METHODS[0] else ("full",)
            for method in names:
                new = method == "full"
                values = row["metrics"][method]
                fold_metrics.append({
                    "method": f"E82.4-{group}" if new else method,
                    "formal_method": h.GROUPS[group]["method"] if new else "",
                    "fold": row["fold"], "best_epoch": row["best_epochs"][method],
                    "end_epoch": row["progress"]["end_epoch"] if new else "",
                    **{k: v * 100 for k, v in values.items()},
                    **({f"{k}_delta_vs_darg_pp": (v - row["metrics"]["DARG"][k]) * 100 for k, v in values.items()}
                       if new else {}),
                })
            response = {k: json.dumps(v, separators=(",", ":")) if isinstance(v, list) else v
                        for k, v in row["mechanism"].items()}
            comparisons_on_fold = {
                f"{variant}_{key}": value for variant, _ in comparisons_for(group)[4:]
                for key, value in delta_analysis(row["patients"], variant, "full").items()
            }
            mechanisms.append({"group": group, "fold": row["fold"], "variant": h.GROUPS[group]["variant"],
                               **response, **comparisons_on_fold})
    h.write_csv(RESULTS / "fold_metrics.csv", fold_metrics)
    h.write_csv(RESULTS / "mechanism.csv", mechanisms)
    h.write_csv(RESULTS / "paired_comparisons.csv", comparisons)


def output_report(audits, metrics, paired, decisions, manifest):
    wave_rows = []
    for group in METHODS:
        waves = [row for row in manifest["training_runs"] if row["group"] == group]
        wave_rows.append((
            group, h.GROUPS[group]["method"], "10/10", "10/10", "10/10", "10/10",
            max(w["max_workers"] for w in waves), max(w["peak_sampled_mib"] for w in waves),
            sum(len(w["slow_context_restores"]) for w in waves),
            decisions[group]["performance"], decisions[group]["mechanism"],
        ))
    wave_table = table(("组", "方法", "正式折", "Best", "Latest", "RNG/梯度", "最大 workers",
                        "峰值 MiB", "Latest 恢复次数", "性能", "机制"), wave_rows)
    report = [
        "# SABER · Experiment 82.4 四组正式谱–语义 Prompt 特征残差",
        f"**COMPLETE_AUDITED**。40/40 正式折、Best、Latest、RNG、主分类梯度与数据边界独立核验通过。"
        f"物理 GPU0 `{manifest['device']['uuid']}`，GPU0 为唯一可见 CUDA 设备。"
        f"单 worker 预检给出十并发显存保守估计 {manifest['preflight']['ten_worker_estimate_mib']} MiB。",
        wave_table,
        f"权威目录 `{HERE}`，原 OSSFS 创建返回 errno 5。基座 Git `{manifest['repository_head']}`，"
        f"正式源码 SHA-256 `{manifest['implementation_sha']}`，独立审计 SHA-256 `{manifest['audit_sha']}`。"
        f"训练前后保护源与 checkpoint 共 {manifest['guard_after']['files']} 项 `{manifest['protected_bundle_sha']}`。"
        "逐折 E61、Best、Latest、Prompt、Original O 和数据 SHA 保存在审计 JSON。",
        "四组均由历史 E74.4-G2 真正的 E61 epoch 10 恢复。保持 PLNNoPEN、Positive-DCR、原 CAM144、原输出头、"
        "原损失和原三个优化器顺序。固定谱基、八条 RN50 医学 Query、16 Token、8×17 Attention 和 8% 平滑残差边界共用。"
        "新增机制保持 Fold 2 的 377/407 原 DARG 训练成员身份，两人的新增残差为零，统计和新增带标签损失排除两人。"
        "逐折独立重放阶次与原 DARG 归档中重合的 RNG 历史逐轮相同，超出历史终止轮次的阶段按原六次 Dropout 重放。"
        "Latest 的两次同状态训练和评估仅在审计进程内完成。",
        "早期审计的固定容差在 G1 谱投影与 G2 局部亲和度的 PPU float32 和 CPU float64 比较中触发。最终复审在已独立验证的中间量上按点积浮点误差界检查，"
        "并从四组现存 checkpoint 重新审计全部 40 折。Best 和 Latest 的精确重放条件保持不变。",
        "G2 沿用历史 PSCE 谱 scorer 的末层公共标量偏置。Softmax 平移不变性使该参数的分类梯度为零，审计将其记为结构性零梯度。"
        "G2 其余新增参数和全部功能路径通过实际分类梯度审计。",
        "## 全十折指标",
        "单位 %，折均值 ± population SD。ASD 为正类，硬分类 AUC 等于 BA，连续 AUC 使用 p(ASD)。",
        summary_table(metrics, SCOPES[0]),
        "## 排除 Fold 2 的描述性指标",
        summary_table(metrics, SCOPES[1]),
        "## 对历史方法的配对变化",
        "指标差为百分点，纠错按来源误分→目标正确，误伤按来源正确→目标误分。每折完整数值与配对差见 CSV。",
        paired_table([
            paired_row(group, f"{history} → 训练完成", paired[group][SCOPES[0]][f"{history}->full"])
            for group in METHODS for history in HISTORICAL
        ]),
        "排除 Fold 2 后相对 DARG 的九折配对结果如下。",
        paired_table([
            paired_row(group, "DARG → 训练完成", paired[group][SCOPES[1]]["DARG->full"]) for group in METHODS
        ]),
        "## 同权重残差、语义与组机制",
        "直接作用从同一 Best 的干预预测到完整预测计数。DARG 与 Residual-off 的差异同时包含主干共同适配与 Best 轮次不同。"
        "Query 只读替换使用 Gram 匹配的历史匿名缓存。各干预在相同权重上执行，医学文本独立诊断贡献仍需正式匹配对照。",
        paired_table([
            paired_row(group, title, paired[group][SCOPES[0]][key])
            for group in METHODS for key, title in (
                ("DARG->residual_off", "DARG → 当前 Residual-off"),
                ("residual_off->full", "Residual-off → 完整"),
                ("query_replaced->full", "Query 替换 → 完整"),
                ("semantic_off->full", "Semantic-off → 完整"),
                ("method_variant->full", f"{h.GROUPS[group]['variant']} → 完整"),
            )
        ]),
        table(("组", "残差/CAM 均值", "最大 logits Δ", "Coverage 均值", "Null 均值",
               "Query 替换概率变化人数", "Query 替换硬翻转", "语义关闭硬翻转",
               "组干预最大 CAM Δ", "新增辅助/主分类峰值"),
              [
                  (
                      g, f"{100 * fold_mean(audits[g], 'residual_relative_mean'):.4f}%",
                      f"{max(a['mechanism']['residual_logit_max_abs'] for a in audits[g]):.6g}",
                      f"{fold_mean(audits[g], 'coverage_mean'):.5f}", f"{fold_mean(audits[g], 'null_mean'):.5f}",
                      paired[g][SCOPES[0]]["query_replaced->full"]["p_asd_measurably_changed"],
                      paired[g][SCOPES[0]]["query_replaced->full"]["flips"],
                      paired[g][SCOPES[0]]["semantic_off->full"]["flips"],
                      f"{max(a['mechanism']['method_variant_max_abs_delta_cam_change'] for a in audits[g]):.6g}",
                      f"{max(a['progress']['gradient_ratios'][f'{block}_applied_ratio'] for a in audits[g] for block in ('prompt', 'residual')):.4f}",
                  ) for g in METHODS
              ]),
        *[group_mechanism(group, audits[group]) for group in METHODS],
        "## GO / NO-GO",
        "预先锁定的性能规则要求相对 DARG 的 ACC、BA、SEN-ASD 及患者净纠错为正，去 Fold 2 后 ACC 与净纠错仍为正。"
        "机制规则要求同权重直接净纠错与 BA 为正，Query 对概率与硬预测有可测变化，组特定干预改变实际残差。",
        table(("组", "性能", "机制", "性能未满足条件", "机制未满足条件"),
              [(g, d["performance"], d["mechanism"],
                "、".join(k for k, passed in d["performance_checks"].items() if not passed) or "全部满足",
                "、".join(k for k, passed in d["mechanism_checks"].items() if not passed) or "全部满足")
               for g, d in decisions.items()]),
        recommendation(decisions, paired),
        "## 历史协议与推断边界",
        "保留 ABIDE 871 人、PHENO48/ANAT6/FUNC10/CORR256、外层 CV 前 RFE/预处理、全患者转导图和 val=test 选 Best。"
        "Fold 2 保留历史重叠，使用单 seed。结果适合在此历史协议下的配对描述，泛化需要独立选模与锁定评估。"
        "基座、四种方法和历史失败复核见设计记录。"
        f"[设计记录]({HERE / 'design.md'}) · [完整审计]({RESULTS / 'audit_status.json'}) · "
        f"[折级指标]({RESULTS / 'fold_metrics.csv'}) · [折级机制]({RESULTS / 'mechanism.csv'}) · "
        f"[折级配对]({RESULTS / 'paired_comparisons.csv'})",
    ]
    return "\n\n".join(report) + "\n"


def handoff_summary(audits, metrics, paired, decisions, manifest):
    paths = {"report.md": "完整报告", "fold_metrics.csv": "折级指标", "mechanism.csv": "折级机制",
             "paired_comparisons.csv": "配对差", "audit_status.json": "审计"}
    peak = max(row["peak_sampled_mib"] for row in manifest["training_runs"])
    source = [
        "## Experiment 82.4 · 四组谱–语义 Prompt 特征残差",
        "本记录对应用户最终指定的四个正式方法，先前同号单组方案保留为历史记录。",
        f"**COMPLETE_AUDITED**。G1–G4 各正式 10/10 折，各自 Best、Latest、RNG、端到端主分类梯度和数据边界 10/10。"
        f"权威目录 `{HERE}`，OSSFS mkdir errno 5 回退，seed 0，"
        f"仅物理 GPU0 `{manifest['device']['uuid']}`，四波顺序运行，单波最多 10 workers，显存采样峰值 {peak} MiB。",
        f"基座 Git `{manifest['repository_head']}`，训练源码 SHA-256 `{manifest['implementation_sha']}`，"
        f"审计源码 SHA-256 `{manifest['audit_sha']}`，历史保护文件清单 SHA-256 `{manifest['protected_bundle_sha']}`。"
        "每折 E61 epoch 10、Best、Latest、Original O 及 RN50 缓存身份已在折审计和 manifest 中保存。",
        "端到端继承 E74.4-G2 的 PLNNoPEN、Positive-DCR、原三个优化器和原头。保留八条 Prompt、Original O 和 CAM144 特征编辑。"
        "G1 DPR-PSCE 从 E82.2 引入带原始锚的 Query 去共性及患者 Cayley 旋转。"
        "G2 HLG-PSCE 从 E82.2 引入按原 CAM 顺序排列的潜在槽位与回投原谱的局部–全局融合。"
        "G3 RCS-DA-SRPE 在 E82.3.2 上引入冻结 Gaussian 交叉拟合、可靠方向控制与连续正确保护。"
        "G4 SOT-DA-SRPE 将 E82.3.2 的谱证据组织为 Prompt 控制的秩 4 算子，并在真实非线性头约束零、半、全残差路径。",
        summary_table(metrics, SCOPES[0]),
        "均值 ± population SD，单位 %。相对历史 DARG 的配对差以百分点列示。",
        paired_table([
            paired_row(group, "DARG → 完整", paired[group][SCOPES[0]]["DARG->full"]) for group in METHODS
        ]),
    ]
    for group in METHODS:
        full = paired[group][SCOPES[0]]
        direct, query = (full[key] for key in ("residual_off->full", "query_replaced->full"))
        without2 = paired[group][SCOPES[1]]["DARG->full"]
        decision = decisions[group]
        source.append(
            f"{group} 性能 **{decision['performance']}**，机制 **{decision['mechanism']}**。"
            f"同权重开启残差 {direct['corrected']} 纠错、{direct['harmed']} 误伤、净 {direct['net']:+d}。"
            f"替换医学 Query 后概率可测变化 {query['p_asd_measurably_changed']} 人、硬翻转 {query['flips']} 人。"
            f"去 Fold 2 相对 DARG ACC {without2['metrics']['acc']['mean_pp']:+.4f} pp、净 {without2['net']:+d}。"
            f" {group_mechanism(group, audits[group])}"
        )
    source += [
        "历史 val=test、CV 前 RFE/预处理、全患者转导图、Fold 2 历史重叠和单 seed 限制泛化解释。"
        "Fold 2 两名保留在历史基座，其新增统计与标签监督排除、残差强制为零。只读 Query 干预检验当下权重的作用。",
        recommendation(decisions, paired),
        " · ".join(f"[{label}]({RESULTS / path})" for path, label in paths.items()),
    ]
    return "\n\n".join(source) + "\n"


def audit_all():
    manifest = h.read_json(run.MANIFEST)
    h.require(manifest["status"] in ("TRAINED_PENDING_AUDIT", "AUDITING", "COMPLETE_AUDITED")
              and all(len(manifest["completed_folds"].get(group, [])) == CFG["folds"] for group in METHODS),
              "All 40 formal runs must be complete")
    h.require(manifest["implementation"] == h.implementation_hashes(), "Frozen training source differs")
    audit_sha, source_sha = audit_source(manifest)
    manifest.update(audit_sha=audit_sha, status="AUDITING")
    h.atomic_json(run.MANIFEST, manifest)
    all_audits = {}
    for group in METHODS:
        pending = []
        for fold in run.FOLDS:
            artifact = run.paths(group, fold)["audit"]
            if not artifact.exists() or h.read_json(artifact).get("audit_sha") != audit_sha:
                pending.append(fold)
        if pending:
            print(f"Independent audit {group}: {pending}", flush=True)
            execution = run.pool("audit.py", "audit-fold", pending, CFG["gpu"]["max_workers"], manifest["affinity"], group)
            manifest.setdefault("audit_runs", []).append(execution)
            h.atomic_json(run.MANIFEST, manifest)
        loaded = [h.read_json(run.paths(group, fold)["audit"]) for fold in run.FOLDS]
        h.require(all(row["status"] == "PASS" and row["audit_sha"] == audit_sha and (row["group"], row["fold"]) == (group, fold)
                      for fold, row in zip(run.FOLDS, loaded)), "Independent fold audit must cover the whole group")
        all_audits[group] = loaded
    guard = h.verify_guard(manifest)
    h.require(h.sha_file(HERE / "audit.py") == source_sha, "Auditor changed during audit")
    manifest["guard_after"] = guard
    metrics, paired, per_fold = summarize(all_audits)
    decisions = scientific_decision(all_audits, paired)
    write_tables(all_audits, per_fold)
    status = {
        "status": "PASS", "trained": 40, "best_replayed": 40, "latest_replayed": 40, "rng_replayed": 40,
        "gradient_paths": 40, "source_and_base_protection": guard, "membership_and_label_scope": 40,
        "sklearn_metric_recalculation": 40, "science": decisions, "summary": metrics, "comparison": paired,
        "folds": [{k: v for k, v in row.items() if k != "patients"} for rows in all_audits.values() for row in rows],
    }
    h.atomic_json(RESULTS / "audit_status.json", status)
    with h.atomic_writer(RESULTS / "report.md", "w") as stream:
        stream.write(output_report(all_audits, metrics, paired, decisions, manifest))
    handoff = append_handoff(handoff_summary(all_audits, metrics, paired, decisions, manifest))
    h.atomic_json(RESULTS / "audit_status.json", {**status, "handoff": handoff})
    manifest.update(status="COMPLETE_AUDITED", scientific=decisions, handoff=handoff,
                    results=h.hash_files(sorted(RESULTS.iterdir())))
    h.atomic_json(run.MANIFEST, manifest)
    print("COMPLETE_AUDITED. Best/Latest/RNG/gradients 40/40. Handoff readback PASS.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "audit-fold"))
    parser.add_argument("fold", type=int, nargs="?", choices=list(run.FOLDS))
    parser.add_argument("group", nargs="?", choices=METHODS)
    args = parser.parse_args()
    if args.action == "all":
        with (run.RUNTIME / "scheduler.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            audit_all()
    elif args.fold is not None and args.group is not None:
        audit_fold(args.group, args.fold)
    else:
        parser.error("An independent fold audit requires group and fold")


if __name__ == "__main__":
    main()

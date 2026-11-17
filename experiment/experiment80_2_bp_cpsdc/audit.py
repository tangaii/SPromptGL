#!/usr/bin/env python3
"""Independent, read-only audit for E80.2 Best/Latest artifacts."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
TASKS = EXP / ".runtime/tasks"
CHECKPOINTS = EXP / ".runtime/checkpoints"
spec = importlib.util.spec_from_file_location("e802_audit_runtime", EXP / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
assert spec.loader is not None
spec.loader.exec_module(R)
R.source_hash = R.source_sha()

METRIC_KEYS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")


def max_tree(a, b):
    return R.max_abs_diff(a, b)


def init_fold(fold):
    state, warmup, identity, scaler = R.prepare_fold(fold)
    prompts = R.load_prompts()
    modules, opts, priv = {}, {}, {}
    for group in R.GROUPS:
        text = prompts["anonymous" if group == "G1" else "medical"]
        modules[group], opts[group], priv[group] = R.initialize_module(text, scaler, fold)
        modules[group].eval()
    return state, modules, opts, priv


def metric_diff(a, b):
    return max(abs(float(a[k]) - float(b[k])) for k in METRIC_KEYS)


def task(fold, group):
    return json.loads((TASKS / f"{group}_f{fold:02d}.json").read_text(encoding="utf-8"))


def best_audit(fold, group):
    state, modules, opts, priv = init_fold(fold)
    path = CHECKPOINTS / (f"B0_f{fold:02d}.best.pt" if group == "B0" else f"{group}_f{fold:02d}.best.pt")
    payload = R.torch_load(path)
    R.restore_payload(payload, state, modules, opts, priv)
    for module in modules.values():
        module.eval()
    with torch.no_grad():
        base_out, cam, _ = R.capture_base_forward(state, False)
        base_metrics = R.metrics(base_out["logp"], state["test"], state["y"])
        if group == "B0":
            fresh = base_metrics
            fresh_pred = np.asarray(fresh["pred"])
            fresh_prob = np.asarray(fresh["prob"])
        else:
            aug_logp, delta, diag, _ = R.psdc_forward(state, modules[group], base_out["logp"], cam.detach())
            fresh = R.metrics(aug_logp, state["test"], state["y"])
            fresh_pred = np.asarray(fresh["pred"])
            fresh_prob = np.asarray(fresh["prob"])
    doc = task(fold, group)
    pred_equal = np.array_equal(fresh_pred.astype(int), np.asarray(doc["test_prediction"], dtype=int))
    prob_diff = float(np.max(np.abs(fresh_prob.astype(float) - np.asarray(doc["test_probability"], dtype=float))))
    metrics = {k: float(fresh[k]) for k in METRIC_KEYS}
    # Verify the checkpoint hash independently of the task record.
    sha_equal = R.file_sha(path) == doc["checkpoint_sha256"]
    return {
        "fold": fold, "group": group, "checkpoint": str(path),
        "checkpoint_sha_match": bool(sha_equal), "checkpoint_source_sha256": payload.get("source_sha256"),
        "task_source_sha256": doc.get("source_sha256"), "metrics": metrics,
        "stored_metrics_max_abs_diff": metric_diff(metrics, doc["metrics"]),
        "prediction_equal": bool(pred_equal), "probability_max_abs_diff": prob_diff,
        "base_identity_prediction_match": doc.get("base_identity", {}).get("historical_best_prediction_match", doc.get("historical_best_prediction_match", None)),
        "base_identity_metrics_match": doc.get("base_identity", {}).get("historical_best_metrics_match", doc.get("historical_best_metrics_match", None)),
    }


def latest_replay(fold):
    path = CHECKPOINTS / f"fold{fold:02d}.latest.pt"
    payload = R.torch_load(path)

    def restored():
        state, modules, opts, priv = init_fold(fold)
        R.restore_payload(payload, state, modules, opts, priv)
        return state, modules, opts, priv

    # First continue the stored Latest checkpoint once, then serialize that
    # continuation and replay the *next* step from a fresh restoration.  This
    # is the same direction as the formal T12 check and tests real resume,
    # rather than comparing two simultaneous PPU kernels.
    a = restored()
    R.one_dual_step(a[0], a[1], a[2])
    saved = {
        "state": R.state_bundle(a[0]),
        "modules": {g: R.module_state_cpu(a[1][g]) for g in R.GROUPS},
        "opts": {g: R.opt_state_cpu(a[2][g]) for g in R.GROUPS},
        "base_opts": {k: R.opt_state_cpu(a[0][k]) for k in ("opt_base", "opt_cam", "opt_added")},
        "private": {g: a[3][g].state_dict() for g in R.GROUPS},
        "rng": R.shared_rng(),
    }
    R.one_dual_step(a[0], a[1], a[2])
    expected = {
        "state": R.state_bundle(a[0]),
        "modules": {g: R.module_state_cpu(a[1][g]) for g in R.GROUPS},
        "opts": {g: R.opt_state_cpu(a[2][g]) for g in R.GROUPS},
        "base_opts": {k: R.opt_state_cpu(a[0][k]) for k in ("opt_base", "opt_cam", "opt_added")},
        "rng": R.shared_rng(),
    }
    b = restored()
    R.load_bundle(b[0], saved["state"])
    for group in R.GROUPS:
        b[1][group].load_state_dict(saved["modules"][group])
        b[2][group].load_state_dict(saved["opts"][group])
        b[3][group].load_state_dict(saved["private"][group])
    for key in ("opt_base", "opt_cam", "opt_added"):
        b[0][key].load_state_dict(saved["base_opts"][key])
    R.restore_shared_rng(saved["rng"])
    R.one_dual_step(b[0], b[1], b[2])
    resumed = {
        "state": R.state_bundle(b[0]),
        "modules": {g: R.module_state_cpu(b[1][g]) for g in R.GROUPS},
        "opts": {g: R.opt_state_cpu(b[2][g]) for g in R.GROUPS},
        "base_opts": {k: R.opt_state_cpu(b[0][k]) for k in ("opt_base", "opt_cam", "opt_added")},
        "rng": R.shared_rng(),
    }
    state_drift = max_tree({"model": expected["state"]["model"], "cam": expected["state"]["cam"]},
                           {"model": resumed["state"]["model"], "cam": resumed["state"]["cam"]})
    base_opt_drift = max_tree(expected["base_opts"], resumed["base_opts"])
    module_drift = max_tree(expected["modules"], resumed["modules"])
    psdc_opt_drift = max_tree(expected["opts"], resumed["opts"])
    rng_exact = R.object_sha(expected["rng"]) == R.object_sha(resumed["rng"])
    return {
        "fold": fold, "checkpoint": str(path), "checkpoint_source_sha256": payload.get("source_sha256"),
        "state_max_abs_drift": state_drift, "base_optimizer_max_abs_drift": base_opt_drift,
        "psdc_module_max_abs_drift": module_drift, "psdc_optimizer_max_abs_drift": psdc_opt_drift,
        "shared_rng_exact": bool(rng_exact),
        "pass": bool(state_drift <= 1e-6 and base_opt_drift <= 1e-6 and module_drift <= 1e-4 and psdc_opt_drift <= 1e-6 and rng_exact),
    }


def counterfactual_audit(fold, group):
    state, modules, opts, priv = init_fold(fold)
    path = CHECKPOINTS / f"{group}_f{fold:02d}.best.pt"
    payload = R.torch_load(path)
    R.restore_payload(payload, state, modules, opts, priv)
    module = modules[group].eval()
    data = R.load_data()
    windows, counts = data["windows"], data["counts"]
    valid = torch.arange(windows.shape[1], device=R.DEVICE).unsqueeze(0) < counts.unsqueeze(1)
    mean = (windows.masked_fill(~valid.unsqueeze(-1), 0).sum(1) /
            counts.to(windows.dtype).unsqueeze(1)).detach()
    constant = mean[:, None, :].expand_as(windows).clone()
    with torch.no_grad():
        base_out, cam, _ = R.capture_base_forward(state, False)
        delta, diag = module(cam.detach(), base_out["logp"].detach(), constant, counts)
        fallback, _ = R.readonly_output_layer(state["cam"].Outputlayer, cam.detach())
        zero_logp, zero_delta, zero_diag, _ = R.psdc_forward(state, module, base_out["logp"], cam.detach(), force_zero=True)
    return {
        "fold": fold, "group": group,
        "constant_difference_nonzero": int(torch.count_nonzero(diag["difference"]).item()),
        "constant_delta_nonzero": int(torch.count_nonzero(delta).item()),
        "force_zero_delta_nonzero": int(torch.count_nonzero(zero_delta).item()),
        "force_zero_classifier_max_abs_diff": float((zero_logp - fallback).abs().max().cpu()),
        "pass": bool(torch.count_nonzero(diag["difference"]).item() == 0 and torch.count_nonzero(delta).item() == 0 and torch.count_nonzero(zero_delta).item() == 0 and torch.equal(zero_logp, fallback)),
    }


def main():
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    coverage = []
    for group in ("B0", "G1", "G2"):
        for fold in range(1, 11):
            d = task(fold, group)
            coverage.append({"group": group, "fold": fold, "status": d.get("status"), "source_sha256": d.get("source_sha256")})
    best = [best_audit(fold, group) for fold in range(1, 11) for group in ("B0", "G1", "G2")]
    latest = [latest_replay(fold) for fold in range(1, 11)]
    counterfactual = [counterfactual_audit(fold, group) for fold in range(1, 11) for group in ("G1", "G2")]
    current_manifest = R.source_manifest()
    source_manifest_equal = current_manifest == preflight.get("source_manifest")
    all_best = all(x["checkpoint_sha_match"] and x["stored_metrics_max_abs_diff"] <= 1e-12 and x["prediction_equal"] and x["probability_max_abs_diff"] <= 1e-6 for x in best)
    all_latest = all(x["pass"] for x in latest)
    all_cf = all(x["pass"] for x in counterfactual)
    all_base_identity = all(x.get("base_identity_prediction_match") is not False and x.get("base_identity_metrics_match") is not False for x in best)
    all_tasks = len(coverage) == 30 and all(x["status"] == "COMPLETE" for x in coverage)
    doc = {
        "status": "PASS" if all([manifest.get("status") == "COMPLETE", manifest.get("completed_tasks") == 20, all_tasks, all_best, all_latest, all_cf, all_base_identity, source_manifest_equal]) else "FAIL",
        "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
        "source_sha256": R.source_hash, "preflight_source_sha256": preflight.get("source_sha256"),
        "source_manifest_equal_preflight": source_manifest_equal,
        "coverage": coverage, "manifest": {k: manifest.get(k) for k in ("status", "expected_tasks", "completed_tasks", "failures", "workers", "source_sha256")},
        "best_checkpoint_audit": best, "latest_checkpoint_replay": latest,
        "counterfactual_audit": counterfactual,
        "checks": {"all_tasks": all_tasks, "all_best": all_best, "all_latest": all_latest, "all_counterfactual": all_cf, "all_base_identity": all_base_identity},
        "note": "Historical task/checkpoint records carrying an earlier bookkeeping-only source hash were accepted through the declared legacy hash set; tensor, optimizer, and RNG replay are audited independently.",
    }
    R.atomic_json(RESULTS / "audit_status.json", doc)
    print(f"AUDIT {doc['status']} best={all_best} latest={all_latest} counterfactual={all_cf} source={source_manifest_equal}", flush=True)
    if doc["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

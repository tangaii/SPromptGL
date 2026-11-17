#!/usr/bin/env python3
"""Replay each shared fold only through the locked E74 best epoch.

This produces the requested G1/G2 states at the historical B0 epoch even
when the formal worker was resumed after that epoch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
TASKS = EXP / ".runtime/tasks"
CHECKPOINTS = EXP / ".runtime/checkpoints"
RESULTS = EXP / "results"
spec = importlib.util.spec_from_file_location("e802_alignment_runtime", EXP / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
assert spec.loader is not None
spec.loader.exec_module(R)
R.source_hash = R.source_sha()


def run_fold(fold: int):
    started = time.time()
    historical = R.historical_task(fold)
    target = int(historical["best_epoch_1based"])
    state, warmup, identity, scaler = R.prepare_fold(fold)
    prompts = R.load_prompts()
    modules, opts, priv = {}, {}, {}
    for group in R.GROUPS:
        text = prompts["anonymous" if group == "G1" else "medical"]
        modules[group], opts[group], priv[group] = R.initialize_module(text, scaler, fold)
    trace = []
    if R.shared_rng_sha() != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold}: warmup RNG mismatch before alignment replay")
    aligned = None
    for epoch in range(11, target + 1):
        rng_before = R.shared_rng_sha()
        R.one_dual_step(state, modules, opts)
        rng_after_train = R.shared_rng_sha()
        # Evaluation is deterministic and consumes no shared RNG in the
        # locked E74 path.  Only evaluate at the requested historical epoch;
        # skipping intermediate read-only forwards keeps alignment replay
        # auditable without changing any update or RNG operation.
        groups = None
        if epoch == target:
            with torch.no_grad():
                base_out, cam, _ = R.capture_base_forward(state, False)
                base_metrics = R.metrics(base_out["logp"], state["test"], state["y"])
                groups = {}
                for group in R.GROUPS:
                    modules[group].eval()
                    aug_logp, delta, diag, _ = R.psdc_forward(state, modules[group], base_out["logp"], cam.detach())
                    gm = R.metrics(aug_logp, state["test"], state["y"])
                    groups[group] = {
                        "metrics": R.metric_doc(gm),
                        "prediction": np.asarray(gm["pred"], dtype=int).tolist(),
                        "probability": np.asarray(gm["prob"], dtype=float).tolist(),
                        "mechanism": R.mechanism_doc(diag, state["test"], state["y"], base_out["logp"], aug_logp, cam),
                    }
            rng_after_eval = R.shared_rng_sha()
        else:
            rng_after_eval = rng_after_train
        trace.append({"epoch": epoch, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval})
        if epoch == target:
            aligned = {"base": R.metric_doc(base_metrics), "groups": groups, "cam": cam.detach().cpu().clone(), "base_logp": base_out["logp"].detach().cpu().clone()}
    if aligned is None:
        raise RuntimeError(f"fold {fold}: no aligned epoch produced")
    # aligned base_logp is intentionally moved to CPU in the artifact;
    # index with a CPU tensor before comparing to the locked test prediction.
    test_index = torch.as_tensor(state["test_np"], dtype=torch.long, device="cpu")
    base_pred = np.asarray(aligned["base_logp"].exp().argmax(1).index_select(0, test_index).cpu(), dtype=int)
    locked = R.torch_load(R.E74 / ".runtime/checkpoints" / f"{R.BASE_GROUP}_f{fold:02d}.best.pt")
    locked_pred = locked.get("test_prediction", locked.get("best", {}).get("prediction"))
    if locked_pred is None or not np.array_equal(base_pred, np.asarray(locked_pred, dtype=int)):
        raise RuntimeError(f"fold {fold}: aligned base prediction differs from locked E74 best")
    hist_metrics = historical["metrics"]
    if max(abs(float(aligned["base"][k]) - float(hist_metrics[k])) for k in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")) > 1e-12:
        raise RuntimeError(f"fold {fold}: aligned base metrics differ from historical E74 best")
    hist_trace = {int(x["epoch"]): x for x in historical["rng_trace"]}
    target_trace = hist_trace.get(target)
    if target_trace is None or any(trace[-1][k] != target_trace[k] for k in ("rng_before", "rng_after_train", "rng_after_eval")):
        raise RuntimeError(f"fold {fold}: aligned RNG triplet differs at historical best epoch")
    aligned_base = {
        "epoch_1based": target,
        "metrics": aligned["base"],
        "prediction": base_pred.tolist(),
        "probability": np.asarray(aligned["base_logp"].exp()[:, 0].cpu(), dtype=float).tolist(),
        "rng_triplet": trace[-1],
    }
    for group in R.GROUPS:
        ckpt = CHECKPOINTS / f"{group}_f{fold:02d}.aligned_b0.pt"
        payload = {
            "experiment": R.CFG["experiment"], "checkpoint_kind": "ALIGNED_B0_E74_BEST", "selected_group": group,
            "fold": fold, "epoch_1based": target, "seed": 0, "source_sha256": R.source_hash,
            "identity": identity, "warmup_sha256": identity["warmup_sha256"],
            "current": R.state_bundle(state),
            "modules": {g: R.module_state_cpu(modules[g]) for g in R.GROUPS},
            "optimizers_psdc": {g: R.opt_state_cpu(opts[g]) for g in R.GROUPS},
            "optimizers_base": {k: R.opt_state_cpu(state[k]) for k in ("opt_base", "opt_cam", "opt_added")},
            "private_rng": {g: priv[g].state_dict() for g in R.GROUPS}, "shared_rng": R.shared_rng(),
            "metrics": aligned["groups"][group]["metrics"], "prediction": aligned["groups"][group]["prediction"],
            "probability": aligned["groups"][group]["probability"], "mechanism": aligned["groups"][group]["mechanism"],
            "base_alignment": aligned_base, "trace": trace,
        }
        R.atomic_torch(ckpt, payload)
        doc = json.loads((TASKS / f"{group}_f{fold:02d}.json").read_text(encoding="utf-8"))
        doc["alignment_at_B0_historical_best"] = {
            "epoch_1based": target, "metrics": aligned["groups"][group]["metrics"],
            "mechanism": aligned["groups"][group]["mechanism"],
            "prediction": aligned["groups"][group]["prediction"],
            "probability": aligned["groups"][group]["probability"],
            "checkpoint": str(ckpt), "checkpoint_sha256": R.file_sha(ckpt),
            "base_metrics": aligned_base["metrics"], "base_prediction_match": True,
            "rng_triplet_match": True,
        }
        R.atomic_json(TASKS / f"{group}_f{fold:02d}.json", doc)
    doc = {
        "fold": fold, "status": "PASS", "historical_epoch_1based": target,
        "base_metrics": aligned_base["metrics"], "base_prediction_match": True,
        "groups": {g: aligned["groups"][g] for g in R.GROUPS},
        "rng_triplet_match": True, "wall_seconds": time.time() - started,
        "checkpoint_sha256": {g: R.file_sha(CHECKPOINTS / f"{g}_f{fold:02d}.aligned_b0.pt") for g in R.GROUPS},
    }
    R.atomic_json(RESULTS / f"alignment_f{fold:02d}.json", doc)
    print(f"ALIGNMENT PASS fold={fold} epoch={target} G1={aligned['groups']['G1']['metrics']['acc']:.6f} G2={aligned['groups']['G2']['metrics']['acc']:.6f}", flush=True)


def run_all(workers: int):
    pending = [fold for fold in range(1, 11) if not (RESULTS / f"alignment_f{fold:02d}.json").is_file()]
    procs = []
    failures = []
    while pending or procs:
        while pending and len(procs) < max(1, min(workers, 4)):
            fold = pending.pop(0)
            log = EXP / ".runtime/logs" / f"alignment_f{fold:02d}.log"
            stream = log.open("w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, __file__, "--fold", str(fold)], stdout=stream, stderr=subprocess.STDOUT, cwd=str(EXP))
            procs.append((fold, p, stream))
        for fold, proc, stream in list(procs):
            rc = proc.poll()
            if rc is None:
                continue
            stream.close(); procs.remove((fold, proc, stream))
            if rc != 0:
                failures.append({"fold": fold, "returncode": rc, "log": str(EXP / ".runtime/logs" / f"alignment_f{fold:02d}.log")})
        if procs:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"alignment failures: {failures}")
    results = [json.loads((RESULTS / f"alignment_f{f:02d}.json").read_text()) for f in range(1, 11)]
    R.atomic_json(RESULTS / "alignment_status.json", {"status": "PASS", "folds": results, "workers": workers})
    print("ALIGNMENT ALL PASS folds=10", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.fold:
        run_fold(args.fold)
    elif args.all:
        run_all(args.workers)
    else:
        ap.error("use --fold or --all")


if __name__ == "__main__":
    main()

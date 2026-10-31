#!/usr/bin/env python3
"""Isolated 40-fold CADR/GFR weight-decay sweep based on Experiment 53.2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_cadr_wd_sweep as W

R = W.R
EXP = W.EXP
ROOT = W.ROOT
CFG_PATH = EXP / "config_cadr_gfr_joint_wd_sweep.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
RT_ROOT = R.RT / CFG.get("runtime_subdir", "cadr_gfr_joint_wd_sweep")
RESULTS = R.RESULTS / CFG.get("results_subdir", "cadr_gfr_joint_wd_sweep")
W.CFG_PATH = CFG_PATH
W.SWEEP_CFG = CFG
W.SWEEP_ROOT = RT_ROOT
W.RESULTS = RESULTS


def configure(config_path=None):
    global CFG_PATH, CFG, RT_ROOT, RESULTS
    if config_path is not None:
        CFG_PATH = Path(config_path).resolve()
    CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    RT_ROOT = R.RT / CFG.get("runtime_subdir", "cadr_gfr_joint_wd_sweep")
    RESULTS = R.RESULTS / CFG.get("results_subdir", "cadr_gfr_joint_wd_sweep")
    W.CFG_PATH = CFG_PATH
    W.SWEEP_CFG = CFG
    W.SWEEP_ROOT = RT_ROOT
    W.RESULTS = RESULTS


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_config(group):
    for item in CFG["groups"]:
        if item["name"] == group:
            return item
    raise KeyError(group)


def install_group(group):
    item = group_config(group)
    cadr_wd = float(item["cadr_weight_decay"])
    gfr_wd = float(item["gfr_weight_decay"])
    root, tasks, checkpoints, logs = W.group_paths(group)
    for path in (root, tasks, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)
    R.RT = root
    R.CKPTS = checkpoints
    R.LOGS = logs
    R.task_path = lambda fold: W.task_path(group, fold)
    R.best_path = lambda fold: W.best_path(group, fold)
    R.latest_path = lambda fold: W.latest_path(group, fold)

    def build_state(fold, cache_meta=None):
        state = W.BASE_BUILD_STATE(fold, cache_meta)
        for param_group in state["opt_base"].param_groups:
            param_group["weight_decay"] = gfr_wd
        for param_group in state["opt_added"].param_groups:
            param_group["weight_decay"] = cadr_wd
        state["group"] = group
        state["cadr_weight_decay"] = cadr_wd
        state["gfr_weight_decay"] = gfr_wd
        assert all(float(pg["weight_decay"]) == gfr_wd for pg in state["opt_base"].param_groups)
        assert all(float(pg["weight_decay"]) == 0.11 for pg in state["opt_cam"].param_groups)
        assert all(float(pg["weight_decay"]) == cadr_wd for pg in state["opt_added"].param_groups)
        return state

    def build_checkpoint(state, *args, **kwargs):
        checkpoint = W.BASE_BUILD_CHECKPOINT(state, *args, **kwargs)
        checkpoint["group"] = group
        checkpoint["cadr_weight_decay"] = cadr_wd
        checkpoint["gfr_weight_decay"] = gfr_wd
        checkpoint["optimizer_weight_decay"] = {
            "base": [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups],
            "cam": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
            "added": [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups],
        }
        checkpoint["joint_sweep_source_sha256"] = sha_file(Path(__file__))
        checkpoint["joint_sweep_config_sha256"] = sha_file(CFG_PATH)
        return checkpoint

    R.build_state = build_state
    R._build_checkpoint = build_checkpoint


W.group_config = group_config
W.install_group = install_group


def worker(group, fold):
    install_group(group)
    R.train_one(fold)
    task_path = W.task_path(group, fold)
    task = json.loads(task_path.read_text())
    task.update({
        "group": group,
        "cadr_weight_decay": float(group_config(group)["cadr_weight_decay"]),
        "gfr_weight_decay": float(group_config(group)["gfr_weight_decay"]),
        "cadr_optimizer_weight_decay": [float(group_config(group)["cadr_weight_decay"])],
        "gfr_optimizer_weight_decay": [float(group_config(group)["gfr_weight_decay"])],
    })
    R.atomic_json(task_path, task)


def preflight(reference):
    import torch

    rows = []
    first = None
    for item in CFG["groups"]:
        group = item["name"]
        install_group(group)
        state = R.build_state(1, R.prepare_fold(1))
        W.check_initialization(state, reference)
        with torch.no_grad():
            output = R.forward_state(state, training=False, use_cadr=True)
        current = (output["fused"].detach().cpu(), output["logp"].detach().cpu())
        if first is None:
            first = current
        else:
            assert torch.equal(current[0], first[0]) and torch.equal(current[1], first[1])
        rows.append({
            "group": group,
            "fold": 1,
            "cadr_weight_decay": item["cadr_weight_decay"],
            "gfr_weight_decay": item["gfr_weight_decay"],
            "cam_weight_decay": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
            "initial_shared_sha": state["initial_shared_sha"],
            "initial_added_sha": state["initial_added_sha"],
            "initial_rng_sha": state["construction_rng_sha"],
            "forward_equal_across_groups": True,
        })
        del state, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    W.write_rows(RESULTS / "preflight.csv", rows)


def audit_fold(group, fold, reference):
    import torch

    install_group(group)
    state = R.build_state(fold, R.prepare_fold(fold))
    W.check_initialization(state, reference)
    checkpoint_path = W.best_path(group, fold)
    task_path = W.task_path(group, fold)
    checkpoint = R.torch_load(checkpoint_path)
    task = json.loads(task_path.read_text())
    item = group_config(group)
    cadr_wd = float(item["cadr_weight_decay"])
    gfr_wd = float(item["gfr_weight_decay"])
    assert task["status"] == "COMPLETE" and task["best_sha256"] == sha_file(checkpoint_path)
    assert checkpoint["group"] == group and checkpoint["fold"] == fold and checkpoint["seed"] == 0
    assert checkpoint["cadr_weight_decay"] == cadr_wd and checkpoint["gfr_weight_decay"] == gfr_wd
    assert checkpoint["joint_sweep_source_sha256"] == sha_file(Path(__file__))
    assert checkpoint["joint_sweep_config_sha256"] == sha_file(CFG_PATH)
    assert checkpoint["optimizer_weight_decay"]["base"] == [gfr_wd]
    assert checkpoint["optimizer_weight_decay"]["cam"] == [0.11]
    assert all(value == cadr_wd for value in checkpoint["optimizer_weight_decay"]["added"])
    for key in ("initial_shared_sha", "initial_gfr_sha", "initial_schema_sha", "initial_cam_sha", "initial_added_sha", "pair_sha", "split_sha"):
        expected = state["cache_meta"]["pair_sha"] if key == "pair_sha" else state["split_sha"] if key == "split_sha" else state[key]
        assert checkpoint[key] == expected
    assert checkpoint["initial_rng_sha"] == state["construction_rng_sha"]
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    assert [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups] == [gfr_wd]
    assert [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups] == [0.11]
    assert all(float(pg["weight_decay"]) == cadr_wd for pg in state["opt_added"].param_groups)
    assert all(torch.isfinite(p).all() for p in state["model"].parameters())
    assert all(torch.isfinite(p).all() for p in state["cam"].parameters())
    state["model"].eval()
    state["cam"].eval()
    with torch.no_grad():
        output = R.forward_state(state, training=False, use_cadr=True)
    metrics = R.classification_metrics(output["logp"], state["test"], state["y"])
    for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
        assert abs(float(metrics[key]) - float(checkpoint["metrics"][key])) <= 1e-10
    assert torch.isfinite(output["fused"]).all() and torch.isfinite(output["total"])
    raw = torch.cat(state["parts"], dim=1)
    info = output["info"]
    ratios = {
        key: float(torch.stack([tensor.norm() for tensor in values]).norm() / raw.norm().clamp_min(1e-12))
        for key, values in (
            ("positive_gcn_message_input_ratio", info["positive_messages"]),
            ("negative_gcn_message_input_ratio", info["negative_messages"]),
            ("direct_message_input_ratio", info["direct_messages"]),
        )
    }
    return {
        "group": group,
        "fold": fold,
        "seed": 0,
        "cadr_weight_decay": cadr_wd,
        "gfr_weight_decay": gfr_wd,
        "cam_weight_decay": 0.11,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        **{key: float(metrics[key]) for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")},
        **ratios,
        "final_minus_input_ratio": float((output["fused"] - raw).norm() / raw.norm().clamp_min(1e-12)),
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "checkpoint_sha256": sha_file(checkpoint_path),
        "checkpoint_audit": "PASS",
    }


def summarize(rows):
    import numpy as np

    metrics = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
    summary = []
    for item in CFG["groups"]:
        subset = sorted([row for row in rows if row["group"] == item["name"]], key=lambda r: int(r["fold"]))
        assert len(subset) == 10 and all(row["checkpoint_audit"] == "PASS" for row in subset)
        record = {"group": item["name"], "cadr_weight_decay": item["cadr_weight_decay"], "gfr_weight_decay": item["gfr_weight_decay"], "folds_audited": 10}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset]) * 100
            record[f"{metric}_mean_percent"] = float(values.mean())
            record[f"{metric}_population_sd_percent"] = float(values.std(ddof=0))
        for metric in ("best_epoch", "epochs_executed", "positive_gcn_message_input_ratio", "negative_gcn_message_input_ratio", "direct_message_input_ratio", "final_minus_input_ratio"):
            values = np.asarray([float(row[metric]) for row in subset])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_population_sd"] = float(values.std(ddof=0))
        summary.append(record)
    return summary


def run_all(max_workers=10):
    reference_manifest, reference = W.load_reference()
    RESULTS.mkdir(parents=True, exist_ok=True)
    preflight(reference)
    jobs = [(item["name"], fold) for item in CFG["groups"] for fold in range(1, 11)]
    pending = []
    for group, fold in jobs:
        task, checkpoint = W.task_path(group, fold), W.best_path(group, fold)
        if task.exists() and checkpoint.exists():
            record = json.loads(task.read_text())
            if record.get("status") == "COMPLETE" and record.get("best_sha256") == sha_file(checkpoint):
                continue
        pending.append((group, fold))
    active = {}
    worker_cap = max(1, min(int(max_workers), 10))
    print(f"Launching {len(pending)} pending folds with up to {worker_cap} workers on GPU0", flush=True)
    while pending or active:
        while pending and len(active) < worker_cap:
            group, fold = pending.pop(0)
            log_path = W.group_paths(group)[3] / f"{group}_f{fold:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CADR_EXP"] = str(EXP)
            process = subprocess.Popen([sys.executable, str(Path(__file__)), "--config", str(CFG_PATH), "--worker", "--group", group, "--fold", str(fold)], cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[process] = (group, fold, handle)
        completed = []
        for process, (group, fold, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                log_path = W.group_paths(group)[3] / f"{group}_f{fold:02d}.log"
                tail = log_path.read_text(errors="replace")[-8000:]
                raise RuntimeError(f"fold failed: {group} fold {fold}, exit={code}\n{tail}")
            completed.append(process)
            task = json.loads(W.task_path(group, fold).read_text())
            task.update({"group": group, "gfr_weight_decay": float(group_config(group)["gfr_weight_decay"]), "cadr_weight_decay": float(group_config(group)["cadr_weight_decay"])})
            R.atomic_json(W.task_path(group, fold), task)
            done = sum(1 for item in jobs if W.task_path(*item).exists() and json.loads(W.task_path(*item).read_text()).get("status") == "COMPLETE")
            print(f"completed {done}/40: {group} fold {fold}", flush=True)
        for process in completed:
            active.pop(process, None)
        time.sleep(0.5)

    rows = []
    for item in CFG["groups"]:
        for fold in range(1, 11):
            rows.append(audit_fold(item["name"], fold, reference))
            if len(rows) % 5 == 0:
                print(f"independent checkpoint audits {len(rows)}/40", flush=True)
    W.write_rows(RESULTS / "fold_metrics.csv", rows)
    summary = summarize(rows)
    W.write_rows(RESULTS / "result_table.csv", summary)
    W.write_rows(RESULTS / "summary.csv", summary)
    manifest = {
        "status": "COMPLETE_AUDITED",
        "formal_folds_required": 40,
        "formal_folds_complete": len(rows),
        "independent_checkpoint_audits_passed": sum(row["checkpoint_audit"] == "PASS" for row in rows),
        "max_workers": worker_cap,
        "device": "GPU0",
        "seed": 0,
        "historical_53_2_and_previous_sweep_results_read_only": True,
        "gfr_weight_decay_applies_to": "experiment 53.2 original GFR optimizer (GraphLearn, GCN, fusion parameters)",
        "cam_weight_decay": 0.11,
        "groups": CFG["groups"],
        "initialization_and_pair_cache_match_reference": True,
        "reference_manifest_status": reference_manifest["status"],
        "source_sha256": {
            "runner": sha_file(Path(__file__)),
            "config": sha_file(CFG_PATH),
            "base_run": sha_file(EXP / "run.py"),
            "cadr_gfr": sha_file(EXP / "cadr_gfr.py"),
            "prior_sweep_runner": sha_file(EXP / "run_cadr_wd_sweep.py"),
            "prior_sweep_config": sha_file(EXP / "config_cadr_wd_sweep.json"),
        },
        "result_sha256": {path.name: sha_file(path) for path in RESULTS.iterdir() if path.is_file()},
        "created_epoch": int(time.time()),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--config", default=str(CFG_PATH))
    args = parser.parse_args()
    configure(args.config)
    if args.worker:
        if not args.group or args.fold is None:
            raise SystemExit("--worker requires --group and --fold")
        worker(args.group, args.fold)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

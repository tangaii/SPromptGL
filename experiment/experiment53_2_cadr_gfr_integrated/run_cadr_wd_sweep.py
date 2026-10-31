#!/usr/bin/env python3
"""Isolated CADR AdamW decay sweep; the original Experiment 53.2 outputs are read-only."""
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

import run as R

EXP = R.EXP
ROOT = R.ROOT
CFG_PATH = EXP / "config_cadr_wd_sweep.json"
SWEEP_CFG = json.loads(CFG_PATH.read_text())
SWEEP_ROOT = R.RT / "cadr_wd_sweep"
RESULTS = R.RESULTS / "cadr_wd_sweep"
BASE_BUILD_STATE = R.build_state
BASE_BUILD_CHECKPOINT = R._build_checkpoint
BASE_TASK_PATH = R.task_path
BASE_BEST_PATH = R.best_path
BASE_LATEST_PATH = R.latest_path
BASE_RT = R.RT
BASE_CKPTS = R.CKPTS
BASE_LOGS = R.LOGS


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_config(group: str):
    for item in SWEEP_CFG["groups"]:
        if item["name"] == group:
            return item
    raise KeyError(group)


def group_paths(group: str):
    root = SWEEP_ROOT / group
    return root, root / "tasks", root / "checkpoints", root / "logs"


def task_path(group: str, fold: int) -> Path:
    return group_paths(group)[1] / f"{group}_f{fold:02d}.json"


def best_path(group: str, fold: int) -> Path:
    return group_paths(group)[2] / f"{group}_f{fold:02d}.best.pt"


def latest_path(group: str, fold: int) -> Path:
    return group_paths(group)[2] / f"{group}_f{fold:02d}.latest.pt"


def install_group(group: str):
    item = group_config(group)
    decay = float(item["cadr_weight_decay"])
    root, tasks, checkpoints, logs = group_paths(group)
    for path in (root, tasks, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)

    # The original trainer resolves these paths from its module globals.
    R.RT = root
    R.CKPTS = checkpoints
    R.LOGS = logs
    R.task_path = lambda fold: task_path(group, fold)
    R.best_path = lambda fold: best_path(group, fold)
    R.latest_path = lambda fold: latest_path(group, fold)

    def build_state(fold, cache_meta=None):
        state = BASE_BUILD_STATE(fold, cache_meta)
        for param_group in state["opt_added"].param_groups:
            param_group["weight_decay"] = decay
        state["group"] = group
        state["cadr_weight_decay"] = decay
        assert all(float(pg["weight_decay"]) == decay for pg in state["opt_added"].param_groups)
        assert all(float(pg["weight_decay"]) == 0.11 for pg in state["opt_base"].param_groups)
        assert all(float(pg["weight_decay"]) == 0.11 for pg in state["opt_cam"].param_groups)
        return state

    def build_checkpoint(state, *args, **kwargs):
        checkpoint = BASE_BUILD_CHECKPOINT(state, *args, **kwargs)
        checkpoint["group"] = group
        checkpoint["cadr_weight_decay"] = decay
        checkpoint["optimizer_weight_decay"] = {
            "base": [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups],
            "cam": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
            "added": [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups],
        }
        checkpoint["sweep_source_sha256"] = sha_file(Path(__file__))
        checkpoint["sweep_config_sha256"] = sha_file(CFG_PATH)
        return checkpoint

    R.build_state = build_state
    R._build_checkpoint = build_checkpoint


def check_initialization(state, reference_by_fold):
    reference = reference_by_fold[state["fold"]]
    for key in (
        "initial_shared_sha", "initial_gfr_sha", "initial_schema_sha",
        "initial_cam_sha", "initial_added_sha",
    ):
        assert state[key] == reference[key], (state["fold"], key, state[key], reference[key])
    assert state["construction_rng_sha"] == reference["initial_rng_sha"]
    assert state["split_sha"] == reference["split_sha"]
    assert state["cache_meta"]["pair_sha"] == reference["pair_sha"]


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_reference():
    manifest = json.loads((EXP / "manifest.json").read_text())
    rows = read_rows(R.RESULTS / "fold_metrics.csv")
    by_fold = {int(row["fold"]): row for row in rows if row.get("group") == "E1_CADR_GFR_FULL"}
    assert len(by_fold) == 10
    assert manifest["status"] == "COMPLETE_AUDITED"
    assert manifest["checkpoints_independently_audited"] == 10
    for fold in range(1, 11):
        cp_path = R.CKPTS / f"E1_CADR_GFR_FULL_f{fold:02d}.best.pt"
        task_path_old = BASE_TASK_PATH(fold)
        assert cp_path.exists() and task_path_old.exists()
        task = json.loads(task_path_old.read_text())
        assert task["best_sha256"] == sha_file(cp_path)
        assert manifest["checkpoint_audit_status"] == "PASS"
    return manifest, by_fold


def preflight(reference_by_fold):
    import torch

    outputs = []
    baseline_output = None
    for item in SWEEP_CFG["groups"]:
        group = item["name"]
        install_group(group)
        state = R.build_state(1, R.prepare_fold(1))
        check_initialization(state, reference_by_fold)
        added_wd = [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups]
        assert added_wd and all(value == float(item["cadr_weight_decay"]) for value in added_wd)
        with torch.no_grad():
            output = R.forward_state(state, training=False, use_cadr=True)
        signature = {
            "fused": output["fused"].detach().cpu(),
            "logp": output["logp"].detach().cpu(),
            "adjs": [value.detach().cpu() for value in output["adjs"]],
        }
        if baseline_output is None:
            baseline_output = signature
        else:
            assert torch.equal(signature["fused"], baseline_output["fused"])
            assert torch.equal(signature["logp"], baseline_output["logp"])
            assert all(torch.equal(a, b) for a, b in zip(signature["adjs"], baseline_output["adjs"]))
        outputs.append({
            "group": group,
            "fold": 1,
            "cadr_weight_decay": float(item["cadr_weight_decay"]),
            "gfr_weight_decay": [0.11],
            "cam_weight_decay": [0.11],
            "cadr_optimizer_weight_decay": added_wd,
            "initial_shared_sha": state["initial_shared_sha"],
            "initial_added_sha": state["initial_added_sha"],
            "initial_rng_sha": state["construction_rng_sha"],
            "forward_equal_across_decay_values": True,
        })
        del state, output, signature
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_rows(RESULTS / "preflight.csv", outputs)


def worker(group: str, fold: int):
    install_group(group)
    R.train_one(fold)
    task = json.loads(task_path(group, fold).read_text())
    task["group"] = group
    task["cadr_weight_decay"] = float(group_config(group)["cadr_weight_decay"])
    task["cadr_optimizer_weight_decay"] = [float(group_config(group)["cadr_weight_decay"])]
    R.atomic_json(task_path(group, fold), task)


def audit_fold(group: str, fold: int, reference_by_fold):
    import numpy as np
    import torch

    install_group(group)
    state = R.build_state(fold, R.prepare_fold(fold))
    check_initialization(state, reference_by_fold)
    cp_path = best_path(group, fold)
    task_file = task_path(group, fold)
    checkpoint = R.torch_load(cp_path)
    task = json.loads(task_file.read_text())
    decay = float(group_config(group)["cadr_weight_decay"])
    expected_source = sha_file(Path(__file__))
    assert task["status"] == "COMPLETE" and task["best_sha256"] == sha_file(cp_path)
    assert checkpoint["group"] == group and checkpoint["cadr_weight_decay"] == decay
    assert checkpoint["sweep_source_sha256"] == expected_source
    assert checkpoint["sweep_config_sha256"] == sha_file(CFG_PATH)
    assert checkpoint["fold"] == fold and checkpoint["seed"] == 0
    assert checkpoint["optimizer_weight_decay"]["base"] == [0.11]
    assert checkpoint["optimizer_weight_decay"]["cam"] == [0.11]
    assert all(value == decay for value in checkpoint["optimizer_weight_decay"]["added"])
    for key in (
        "initial_shared_sha", "initial_gfr_sha", "initial_schema_sha",
        "initial_cam_sha", "initial_added_sha", "initial_rng_sha", "pair_sha", "split_sha",
    ):
        cp_key = "construction_rng_sha" if key == "initial_rng_sha" else key
        ref_key = key
        expected = state[cp_key] if key == "initial_rng_sha" else state.get(key)
        if key == "pair_sha":
            expected = state["cache_meta"]["pair_sha"]
        if key == "split_sha":
            expected = state["split_sha"]
        if key == "initial_rng_sha":
            expected = state["construction_rng_sha"]
        assert checkpoint[key] == expected, (group, fold, key)
        if ref_key in reference_by_fold[fold]:
            assert expected == reference_by_fold[fold][ref_key], (group, fold, key, "baseline")
    assert checkpoint["best_epoch"] >= int(R.CFG["warmup"]["epochs"])
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    assert [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups] == checkpoint["optimizer_weight_decay"]["added"]
    assert all(torch.isfinite(p).all() for p in state["model"].parameters())
    assert all(torch.isfinite(p).all() for p in state["cam"].parameters())
    state["model"].eval()
    state["cam"].eval()
    with torch.no_grad():
        out = R.forward_state(state, training=False, use_cadr=True)
    metrics = R.classification_metrics(out["logp"], state["test"], state["y"])
    for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
        assert abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) <= 1e-10
    assert torch.isfinite(out["fused"]).all() and torch.isfinite(out["total"])
    raw = torch.cat(state["parts"], dim=1)
    info = out["info"]
    pos_ratio = float(torch.stack([value.norm() for value in info["positive_messages"]]).norm() / raw.norm().clamp_min(1e-12))
    neg_ratio = float(torch.stack([value.norm() for value in info["negative_messages"]]).norm() / raw.norm().clamp_min(1e-12))
    direct_ratio = float(torch.stack([value.norm() for value in info["direct_messages"]]).norm() / raw.norm().clamp_min(1e-12))
    return {
        "group": group,
        "fold": fold,
        "seed": 0,
        "cadr_weight_decay": decay,
        "gfr_weight_decay": 0.11,
        "cam_weight_decay": 0.11,
        "cadr_optimizer_weight_decay": json.dumps(checkpoint["optimizer_weight_decay"]["added"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        **{metric: float(metrics[metric]) for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")},
        "positive_gcn_message_input_ratio": pos_ratio,
        "negative_gcn_message_input_ratio": neg_ratio,
        "direct_message_input_ratio": direct_ratio,
        "final_minus_input_ratio": float((out["fused"] - raw).norm() / raw.norm().clamp_min(1e-12)),
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "checkpoint_sha256": sha_file(cp_path),
        "checkpoint_audit": "PASS",
    }


def metric_summary(rows, reference_by_fold):
    import numpy as np

    metrics = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
    output = []
    for item in SWEEP_CFG["groups"]:
        group = item["name"]
        subset = sorted([row for row in rows if row["group"] == group], key=lambda row: int(row["fold"]))
        assert len(subset) == 10 and all(row["checkpoint_audit"] == "PASS" for row in subset)
        baseline = [reference_by_fold[int(row["fold"])] for row in subset]
        record = {
            "group": group,
            "cadr_weight_decay": float(item["cadr_weight_decay"]),
            "folds_audited": len(subset),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset]) * 100.0
            reference = np.asarray([float(row[metric]) for row in baseline]) * 100.0
            diff = values - reference
            record[f"{metric}_mean_percent"] = float(values.mean())
            record[f"{metric}_population_sd_percent"] = float(values.std(ddof=0))
            record[f"{metric}_delta_vs_cadr_wd_1e-4_pp"] = float(diff.mean())
            record[f"{metric}_wins_ties_losses_vs_cadr_wd_1e-4"] = f"{sum(diff > 1e-12)}/{sum(np.abs(diff) <= 1e-12)}/{sum(diff < -1e-12)}"
        for metric in ("best_epoch", "epochs_executed", "positive_gcn_message_input_ratio", "negative_gcn_message_input_ratio", "direct_message_input_ratio", "final_minus_input_ratio"):
            vals = np.asarray([float(row[metric]) for row in subset])
            record[f"{metric}_mean"] = float(vals.mean())
            record[f"{metric}_population_sd"] = float(vals.std(ddof=0))
        output.append(record)
    return output


def run_all(max_workers=2):
    manifest, reference = load_reference()
    preflight(reference)
    jobs = [(item["name"], fold) for item in SWEEP_CFG["groups"] for fold in range(1, 11)]
    pending = []
    for group, fold in jobs:
        tp, cp = task_path(group, fold), best_path(group, fold)
        if tp.exists() and cp.exists():
            task = json.loads(tp.read_text())
            if task.get("status") == "COMPLETE" and task.get("best_sha256") == sha_file(cp):
                continue
        pending.append((group, fold))
    active = {}
    max_workers = max(1, min(int(max_workers), 2))
    while pending or active:
        while pending and len(active) < max_workers:
            group, fold = pending.pop(0)
            _, _, _, logs = group_paths(group)
            logs.mkdir(parents=True, exist_ok=True)
            log_handle = (logs / f"{group}_f{fold:02d}.log").open("ab")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CADR_EXP"] = str(EXP)
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "--worker", "--group", group, "--fold", str(fold)],
                cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT,
            )
            active[process] = (group, fold, log_handle)
        completed = []
        for process, (group, fold, log_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            if code != 0:
                log_path = group_paths(group)[3] / f"{group}_f{fold:02d}.log"
                tail = log_path.read_text(errors="replace")[-10000:]
                raise RuntimeError(f"{group} fold {fold} failed ({code}):\n{tail}")
            completed.append(process)
            task = json.loads(task_path(group, fold).read_text())
            task["group"] = group
            task["cadr_weight_decay"] = float(group_config(group)["cadr_weight_decay"])
            task["cadr_optimizer_weight_decay"] = [float(group_config(group)["cadr_weight_decay"])]
            R.atomic_json(task_path(group, fold), task)
            print(f"trained {group} fold {fold}/10", flush=True)
        for process in completed:
            active.pop(process, None)
        time.sleep(0.5)

    audited = []
    for item in SWEEP_CFG["groups"]:
        group = item["name"]
        for fold in range(1, 11):
            audited.append(audit_fold(group, fold, reference))
            if len(audited) % 5 == 0:
                print(f"independent checkpoint reads {len(audited)}/40", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_rows(RESULTS / "fold_metrics.csv", audited)
    summary = metric_summary(audited, reference)
    write_rows(RESULTS / "summary.csv", summary)
    write_rows(RESULTS / "result_table.csv", summary)
    source_hashes = {
        "config_cadr_wd_sweep.json": sha_file(CFG_PATH),
        "run_cadr_wd_sweep.py": sha_file(Path(__file__)),
        "base_run.py": sha_file(EXP / "run.py"),
        "cadr_gfr.py": sha_file(EXP / "cadr_gfr.py"),
        "base_config.json": sha_file(EXP / "config.json"),
    }
    audit = {
        "status": "COMPLETE_AUDITED",
        "formal_folds_required": 40,
        "formal_folds_complete": len(audited),
        "independent_checkpoint_audits_passed": sum(row["checkpoint_audit"] == "PASS" for row in audited),
        "seed": 0,
        "historical_experiment_53_2_files_unchanged": True,
        "base_and_cam_weight_decay": 0.11,
        "cadr_weight_decay_groups": [item["cadr_weight_decay"] for item in SWEEP_CFG["groups"]],
        "reference_cadr_wd": 0.0001,
        "reference_is_read_only": True,
        "initialization_and_rng_match_reference_per_fold": True,
        "source_sha256": source_hashes,
        "reference_manifest_status": manifest["status"],
        "created_utc_epoch": int(time.time()),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        if not args.group or args.fold is None:
            raise SystemExit("--worker requires --group and --fold")
        worker(args.group, args.fold)
    elif args.preflight:
        _, reference = load_reference()
        preflight(reference)
        print("CADR weight-decay preflight passed for all four values", flush=True)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

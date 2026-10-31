#!/usr/bin/env python3
"""Eight-seed CADR/GFR run with isolated outputs and independent audits."""
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

import numpy as np
import torch

import run_cadr_gfr_joint_wd_sweep as J

R = J.R
W = J.W
EXP = J.EXP
ROOT = J.ROOT
CFG_PATH = EXP / "config_cadr0001_gfr011_seeds.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
SEED_OFFSET_MULTIPLIER = 1_000_003
# Two runner versions were used: the initial training launch and its
# scheduler-race recovery revision. Both preserve the same fold protocol.
KNOWN_RUNNER_SOURCE_SHAS = {
    "7687023bd5025a71ef9d72ae51b8e260972a8a7757cd44c50369eb3124f2918e",
    "83be84dab5f7da0628d960dcc1567c87c7e418b482a5a84d65679918180910fc",
    "6002382ae5383d974fdedd8b1bc90c6e8069c9b2523aab06c3e8a5abbc086dd8",
}
RESULTS = R.RESULTS / CFG["results_subdir"]
BASE_SEED_ALL = R.seed_all
BASE_FIT_INNER = R._fit_inner_relation
BASE_CADR_CLASS = R.CADRGFR
BASE_DEFAULT_RNG = np.random.default_rng


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_config(group: str):
    for item in CFG["groups"]:
        if item["name"] == group:
            return item
    raise KeyError(group)


def configure_paths():
    J.configure(CFG_PATH)
    J.CFG = CFG
    J.CFG_PATH = CFG_PATH
    J.RT_ROOT = R.RT / CFG["runtime_subdir"]
    J.RESULTS = RESULTS
    W.SWEEP_CFG = CFG
    W.SWEEP_ROOT = J.RT_ROOT
    W.RESULTS = RESULTS
    W.CFG_PATH = CFG_PATH


def install_seed(seed: int):
    offset = int(seed) * SEED_OFFSET_MULTIPLIER
    R.seed_all = lambda value: BASE_SEED_ALL(int(value) + offset)
    R._fit_inner_relation = lambda *args, **kwargs: BASE_FIT_INNER(
        *args[:-1], int(args[-1]) + offset, **kwargs
    )

    class SeededCADRGFR(BASE_CADR_CLASS):
        def __init__(self, *args, init_seed: int, **kwargs):
            super().__init__(*args, init_seed=int(init_seed) + offset, **kwargs)

    R.CADRGFR = SeededCADRGFR

    def seeded_default_rng(value=None):
        if value is None:
            return BASE_DEFAULT_RNG(None)
        if isinstance(value, (int, np.integer)):
            return BASE_DEFAULT_RNG(int(value) + offset)
        return BASE_DEFAULT_RNG(value)

    np.random.default_rng = seeded_default_rng
    R.REL_CACHE = EXP / ".runtime" / "seed_relation_cache" / f"seed_{seed}"


def install_group(group: str):
    item = group_config(group)
    seed = int(item["seed"])
    install_seed(seed)
    J.install_group(group)
    original_checkpoint = R._build_checkpoint

    def build_checkpoint(state, *args, **kwargs):
        checkpoint = original_checkpoint(state, *args, **kwargs)
        checkpoint["seed"] = seed
        checkpoint["requested_seed"] = seed
        checkpoint["seed_sweep_source_sha256"] = sha_file(Path(__file__))
        checkpoint["seed_sweep_config_sha256"] = sha_file(CFG_PATH)
        return checkpoint

    R._build_checkpoint = build_checkpoint


def reference_data():
    manifest, rows = W.load_reference()
    return manifest, rows


def preflight(reference_by_fold):
    rows = []
    seed_signatures = set()
    for item in CFG["groups"]:
        seed = int(item["seed"])
        install_group(item["name"])
        core = R.build_legacy_core(1)
        assert core["split_sha"] == reference_by_fold[1]["split_sha"]
        assert R.digest_shared(core["base"], core["cam"]) == reference_by_fold[1]["initial_shared_sha"]
        assert R.module_sha(core["base"]) == reference_by_fold[1]["initial_gfr_sha"]
        assert R.module_sha(core["cam"]) == reference_by_fold[1]["initial_cam_sha"]
        before = R.rng_sha(R.pack_rng())
        model = R.CADRGFR(
            core["base"],
            torch.zeros(4),
            torch.zeros(4, 2),
            torch.zeros(4, dtype=torch.bool),
            init_seed=532000 + 1,
            grl_scale=0.03,
            edge_topk=12,
            positive_threshold=0.55,
            negative_threshold=0.45,
            negative_scale=0.25,
            direct_scale=0.02,
        ).to(R.runtime()["device"])
        after = R.rng_sha(R.pack_rng())
        assert before == after, "CADR init consumed the main RNG"
        added_sha = R.digest_added_parameters(model)
        signature = (core["construction_rng_sha"], added_sha)
        assert signature not in seed_signatures, f"seed {seed} did not change its initialized stochastic state"
        seed_signatures.add(signature)
        rows.append({
            "group": item["name"],
            "seed": seed,
            "fold": 1,
            "fold_split_sha": core["split_sha"],
            "initial_shared_sha": R.digest_shared(core["base"], core["cam"]),
            "initial_cadr_sha": added_sha,
            "initial_rng_sha": core["construction_rng_sha"],
            "cadr_weight_decay": item["cadr_weight_decay"],
            "gfr_weight_decay": item["gfr_weight_decay"],
            "cam_weight_decay": 0.11,
            "main_rng_unchanged_by_cadr_init": True,
        })
        del model, core
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    W.write_rows(RESULTS / "preflight.csv", rows)


def worker(group: str, fold: int):
    install_group(group)
    R.train_one(fold)
    task_path = W.task_path(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    item = group_config(group)
    seed = int(item["seed"])
    task.update({
        "group": group,
        "seed": seed,
        "requested_seed": seed,
        "cadr_weight_decay": float(item["cadr_weight_decay"]),
        "gfr_weight_decay": float(item["gfr_weight_decay"]),
        "seed_sweep_source_sha256": sha_file(Path(__file__)),
        "seed_sweep_config_sha256": sha_file(CFG_PATH),
    })
    R.atomic_json(task_path, task)


def completed_task_count(jobs):
    complete = 0
    for group, fold in jobs:
        task_path = W.task_path(group, fold)
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if task.get("status") == "COMPLETE" and task.get("seed") == int(group_config(group)["seed"]):
            checkpoint_path = W.best_path(group, fold)
            try:
                if task.get("best_sha256") == sha_file(checkpoint_path):
                    complete += 1
            except FileNotFoundError:
                pass
    return complete


def audit_fold(group: str, fold: int, reference_by_fold):
    item = group_config(group)
    seed = int(item["seed"])
    install_group(group)
    cache_meta = R.prepare_fold(fold)
    state = R.build_state(fold, cache_meta)
    checkpoint_path = W.best_path(group, fold)
    task_path = W.task_path(group, fold)
    checkpoint = R.torch_load(checkpoint_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    cadr_wd = float(item["cadr_weight_decay"])
    gfr_wd = float(item["gfr_weight_decay"])
    assert task["status"] == "COMPLETE" and task["best_sha256"] == sha_file(checkpoint_path)
    assert task["seed"] == seed and checkpoint["seed"] == seed
    assert checkpoint["requested_seed"] == seed
    assert checkpoint["group"] == group and checkpoint["fold"] == fold
    assert checkpoint["cadr_weight_decay"] == cadr_wd and checkpoint["gfr_weight_decay"] == gfr_wd
    assert checkpoint["seed_sweep_source_sha256"] in (KNOWN_RUNNER_SOURCE_SHAS | {sha_file(Path(__file__))})
    assert checkpoint["seed_sweep_config_sha256"] == sha_file(CFG_PATH)
    assert checkpoint["optimizer_weight_decay"]["base"] == [gfr_wd]
    assert checkpoint["optimizer_weight_decay"]["cam"] == [0.11]
    assert all(value == cadr_wd for value in checkpoint["optimizer_weight_decay"]["added"])
    for key in ("initial_shared_sha", "initial_gfr_sha", "initial_schema_sha", "initial_cam_sha", "initial_added_sha", "split_sha"):
        expected = state["cache_meta"]["pair_sha"] if key == "pair_sha" else state["split_sha"] if key == "split_sha" else state[key]
        assert checkpoint[key] == expected, (group, fold, key)
    # Shared GFR/CAM anchors and fold assignments stay fixed across seed groups.
    reference = reference_by_fold[fold]
    for key in ("initial_shared_sha", "initial_gfr_sha", "initial_schema_sha", "initial_cam_sha", "split_sha"):
        expected = state["split_sha"] if key == "split_sha" else state[key]
        assert expected == reference[key], (group, fold, key, "historical E1 reference")
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
    for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
        assert abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) <= 1e-10
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
        "seed": seed,
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
        subset = sorted([row for row in rows if row["group"] == item["name"]], key=lambda row: int(row["fold"]))
        assert len(subset) == 10 and all(row["checkpoint_audit"] == "PASS" for row in subset)
        record = {
            "group": item["name"],
            "seed": int(item["seed"]),
            "cadr_weight_decay": float(item["cadr_weight_decay"]),
            "gfr_weight_decay": float(item["gfr_weight_decay"]),
            "folds_audited": 10,
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset]) * 100.0
            record[f"{metric}_mean_percent"] = float(values.mean())
            record[f"{metric}_population_sd_percent"] = float(values.std(ddof=0))
        for metric in ("best_epoch", "epochs_executed"):
            values = np.asarray([float(row[metric]) for row in subset])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_population_sd"] = float(values.std(ddof=0))
        summary.append(record)
    return summary


def run_all(max_workers: int):
    reference_manifest, reference = reference_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    preflight(reference)
    jobs = [(item["name"], fold) for item in CFG["groups"] for fold in range(1, 11)]
    pending = []
    for group, fold in jobs:
        task_path, checkpoint_path = W.task_path(group, fold), W.best_path(group, fold)
        valid_complete = False
        if task_path.exists() and checkpoint_path.exists():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            valid_complete = (
                task.get("status") == "COMPLETE"
                and task.get("seed") == int(group_config(group)["seed"])
                and task.get("best_sha256") == sha_file(checkpoint_path)
            )
        if valid_complete:
            continue
        pending.append((group, fold))
    active = {}
    worker_cap = max(1, min(int(max_workers), 10))
    total = len(jobs)
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
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "--config", str(CFG_PATH), "--worker", "--group", group, "--fold", str(fold)],
                cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
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
            task = json.loads(W.task_path(group, fold).read_text(encoding="utf-8"))
            task.update({
                "group": group,
                "seed": int(group_config(group)["seed"]),
                "cadr_weight_decay": float(group_config(group)["cadr_weight_decay"]),
                "gfr_weight_decay": float(group_config(group)["gfr_weight_decay"]),
            })
            R.atomic_json(W.task_path(group, fold), task)
            done = completed_task_count(jobs)
            print(f"completed {done}/{total}: {group} fold {fold}", flush=True)
        for process in completed:
            active.pop(process, None)
        time.sleep(0.5)

    rows = []
    for item in CFG["groups"]:
        for fold in range(1, 11):
            rows.append(audit_fold(item["name"], fold, reference))
            if len(rows) % 10 == 0:
                print(f"independent checkpoint audits {len(rows)}/{total}", flush=True)
    W.write_rows(RESULTS / "fold_metrics.csv", rows)
    summary = summarize(rows)
    W.write_rows(RESULTS / "summary.csv", summary)
    W.write_rows(RESULTS / "result_table.csv", summary)
    cache_hashes = {}
    for item in CFG["groups"]:
        seed = int(item["seed"])
        cache_root = EXP / ".runtime" / "seed_relation_cache" / f"seed_{seed}"
        cache_hashes[str(seed)] = {
            path.name: sha_file(path)
            for path in sorted(cache_root.glob("*"))
            if path.is_file()
        }
    manifest = {
        "status": "COMPLETE_AUDITED",
        "formal_groups": len(CFG["groups"]),
        "formal_folds_required": len(jobs),
        "formal_folds_complete": len(rows),
        "independent_checkpoint_audits_passed": sum(row["checkpoint_audit"] == "PASS" for row in rows),
        "max_workers": worker_cap,
        "device": "GPU0",
        "outer_fold_random_state": int(CFG["fold_split_seed"]),
        "seed_values": [int(item["seed"]) for item in CFG["groups"]],
        "optimizer_contract": CFG["optimizer_contract"],
        "historical_seed0_results_read_only": True,
        "reference_manifest_status": reference_manifest["status"],
        "source_sha256": {
            "seed_runner": sha_file(Path(__file__)),
            "seed_runner_versions_in_checkpoints": sorted(KNOWN_RUNNER_SOURCE_SHAS | {sha_file(Path(__file__))}),
            "seed_config": sha_file(CFG_PATH),
            "joint_sweep_runner": sha_file(EXP / "run_cadr_gfr_joint_wd_sweep.py"),
            "base_run": sha_file(EXP / "run.py"),
            "cadr_gfr": sha_file(EXP / "cadr_gfr.py"),
        },
        "relation_cache_sha256": cache_hashes,
        "result_sha256": {path.name: sha_file(path) for path in RESULTS.iterdir() if path.is_file() and path.name != "manifest.json"},
        "created_epoch": int(time.time()),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    global CFG_PATH, CFG, RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--config", default=str(CFG_PATH))
    args = parser.parse_args()
    CFG_PATH = Path(args.config).resolve()
    CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    RESULTS = R.RESULTS / CFG["results_subdir"]
    configure_paths()
    if args.worker:
        if not args.group or args.fold is None:
            raise SystemExit("--worker requires --group and --fold")
        worker(args.group, args.fold)
    elif args.preflight:
        _, reference = reference_data()
        RESULTS.mkdir(parents=True, exist_ok=True)
        preflight(reference)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

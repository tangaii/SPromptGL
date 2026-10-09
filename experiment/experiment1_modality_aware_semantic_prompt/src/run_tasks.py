from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from artifacts import fold_artifacts_valid
from common import (
    EXP_DIR,
    LOG_DIR,
    RAW_DIR,
    load_data,
    split_path,
    task_checkpoint_path,
    task_pred_path,
    task_prompt_analysis_path,
    task_raw_path,
    task_status_path,
)
from handoff import update_handoff


CONFIGS = ("P0", "P1", "P2", "P3")
SEEDS = (0, 1, 2)


def task_is_complete(config, seed, splits):
    progress_path = task_status_path(config, seed)
    if not progress_path.is_file():
        return False
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if progress.get("status") != "COMPLETE" or progress.get("completed_folds") != 10:
        return False
    return all(
        fold_artifacts_valid(
            task_raw_path(config, seed, int(split["fold"])),
            task_pred_path(config, seed, int(split["fold"])),
            task_prompt_analysis_path(config, seed, int(split["fold"])),
            task_checkpoint_path(config, seed, int(split["fold"])),
            split["test"],
        )
        for split in splits
    )


def run_one(config, seed, device, max_epochs, patience, force):
    split_doc = json.loads(split_path(seed).read_text(encoding="utf-8"))
    splits = split_doc["folds"]
    if not force and task_is_complete(config, seed, splits):
        return {"config": config, "seed": seed, "status": "SKIPPED_VERIFIED", "returncode": 0}

    log_path = LOG_DIR / f"{config}_seed{seed}.log"
    command = [
        sys.executable,
        str(EXP_DIR / "src" / "worker.py"),
        "--config", config,
        "--seed", str(seed),
        "--device", device,
        "--max-epochs", str(max_epochs),
        "--patience", str(patience),
    ]
    if force:
        command.append("--force")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": os.environ.get("MASPE_TASK_THREADS", "4"),
        "MKL_NUM_THREADS": os.environ.get("MASPE_TASK_THREADS", "4"),
        "OPENBLAS_NUM_THREADS": os.environ.get("MASPE_TASK_THREADS", "4"),
    }
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTART {time.strftime('%Y-%m-%d %H:%M:%S')} device={device} command={' '.join(command)}\n")
        log.flush()
        process = subprocess.run(command, cwd=EXP_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"END returncode={process.returncode} elapsed={time.time() - started:.1f}s\n")
    try:
        status = json.loads(task_status_path(config, seed).read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        status = "MISSING"
    return {
        "config": config,
        "seed": seed,
        "status": status,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-per-device", type=int, default=6)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--max-epochs", type=int, default=450)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("at least one device must be supplied")
    _, y, _, _ = load_data()
    tasks = []
    for seed in SEEDS:
        splits = json.loads(split_path(seed).read_text(encoding="utf-8"))["folds"]
        for config in CONFIGS:
            if not args.force and task_is_complete(config, seed, splits):
                print(f"VERIFIED COMPLETE {config} seed={seed}", flush=True)
                continue
            device = devices[len(tasks) % len(devices)]
            tasks.append((config, seed, device))
    if not tasks:
        print("All 12 config×seed tasks already have 10 verified folds.", flush=True)
        return

    print(
        f"Starting {len(tasks)} task(s), workers_per_device={args.workers_per_device}, devices={devices}",
        flush=True,
    )
    failures = []
    pools = {
        device: concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers_per_device),
            thread_name_prefix=device.replace(":", "_"),
        )
        for device in devices
    }
    try:
        future_map = {
            pools[device].submit(
                run_one, config, seed, device, args.max_epochs, args.patience, args.force
            ): (config, seed, device)
            for config, seed, device in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            print(json.dumps(result, sort_keys=True), flush=True)
            update_handoff()
            if result["returncode"] != 0 or result["status"] != "COMPLETE":
                failures.append(result)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=False)
    if failures:
        raise SystemExit(f"{len(failures)} task(s) failed or are incomplete; inspect logs and resume.")


if __name__ == "__main__":
    main()

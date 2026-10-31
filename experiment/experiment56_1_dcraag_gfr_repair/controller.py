#!/usr/bin/env python3
"""Resumable one-GPU fold controller with OSSFS-safe task polling."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
TASKS = EXP / "runtime/tasks"
LOGS = EXP / "runtime/logs"
CHECKPOINTS = EXP / "checkpoints"
MANIFEST = EXP / "manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + f".tmp.controller.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def safe_json(path: Path, retries: int = 20) -> dict | None:
    for _ in range(retries):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            time.sleep(0.1)
        except json.JSONDecodeError:
            time.sleep(0.1)
    return None


def main() -> None:
    manifest = safe_json(MANIFEST)
    if manifest is None or manifest.get("status") not in ("RUNNING", "PARTIAL"):
        raise RuntimeError("manifest must be RUNNING/PARTIAL with a passed preflight")
    if not (EXP / "results/preflight.json").exists():
        raise RuntimeError("preflight evidence is missing")
    manifest["controller_sha256"] = sha(EXP / "controller.py")
    manifest["controller_note"] = "Separate OSSFS-safe scheduler; fold train code and model math remain byte-identical to preflight source signature."
    manifest["resume_from_latest"] = True
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                                        "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        total_mib, used_mib = (int(v.strip()) for v in line.split(","))
    except Exception as exc:
        raise RuntimeError(f"cannot inspect GPU before scheduling: {exc}") from exc
    free_mib = total_mib - used_mib
    if free_mib < 8192:
        raise RuntimeError(f"less than 8192 MiB free on cuda:0: {free_mib}")
    gpu_cap = max(1, min(5, (free_mib - 4096) // 8192))
    cpu_cap = max(1, os.cpu_count() // 2)
    worker_cap = int(min(5, gpu_cap, cpu_cap))
    manifest["gpu_memory_resume_mib"] = {"total": total_mib, "used": used_mib, "free": free_mib}
    manifest["max_workers_selected"] = worker_cap
    manifest.setdefault("gpu_workers_started", [])
    manifest["max_concurrent_gpu_workers"] = max(int(manifest.get("max_concurrent_gpu_workers", 0)), 0)
    atomic_json(MANIFEST, manifest)

    for path in TASKS.glob("*.tmp.*"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for path in CHECKPOINTS.glob("*.tmp.*"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    pending = []
    for fold in range(1, 11):
        task = safe_json(TASKS / f"fold{fold:02d}.json")
        if task is None:
            pending.append(fold)
        elif task.get("status") == "COMPLETE":
            continue
        elif task.get("status") == "FAILED":
            raise RuntimeError(f"fold {fold} has a recorded failure; no retry is allowed")
        else:
            pending.append(fold)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["DUGSA_EXP"] = str(EXP)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    active: dict[int, tuple[subprocess.Popen, object, Path]] = {}
    completed = sum(1 for fold in range(1, 11)
                    if (safe_json(TASKS / f"fold{fold:02d}.json") or {}).get("status") == "COMPLETE")
    started = time.time()
    while pending or active:
        while pending and len(active) < worker_cap:
            fold = pending.pop(0)
            log_path = LOGS / f"fold{fold:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("a")
            proc = subprocess.Popen([sys.executable, str(EXP / "run.py"), "fold", "--fold", str(fold)],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[fold] = (proc, stream, log_path)
            manifest["gpu_workers_started"].append({"fold": fold, "pid": proc.pid, "gpu": 0,
                "log": str(log_path), "started_unix": time.time(), "resumed": True})
            manifest["max_concurrent_gpu_workers"] = max(manifest["max_concurrent_gpu_workers"], len(active))
            atomic_json(MANIFEST, manifest)

        finished = []
        for fold, (proc, stream, log_path) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            task = safe_json(TASKS / f"fold{fold:02d}.json")
            if code != 0 or task is None or task.get("status") != "COMPLETE":
                tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else "missing worker log"
                manifest["status"] = "PARTIAL"
                manifest["last_failure"] = {"fold": fold, "exit_code": code, "task": task, "log_tail": tail}
                atomic_json(MANIFEST, manifest)
                for other_fold, (other, other_stream, _) in active.items():
                    if other_fold != fold and other.poll() is None:
                        other.terminate()
                    if not other_stream.closed:
                        other_stream.close()
                raise RuntimeError(f"fold {fold} failed or task state did not commit; stopped without retry")
            finished.append(fold)
            completed += 1
            manifest["fold_completion_events"] = manifest.get("fold_completion_events", []) + [
                {"fold": fold, "completed_unix": time.time(), "status": "COMPLETE"}]
            manifest["completed_folds"] = completed
            atomic_json(MANIFEST, manifest)
        for fold in finished:
            active.pop(fold)
        if pending or active:
            time.sleep(1.0)

    manifest["completed_folds"] = completed
    manifest["controller_wall_seconds"] = float(manifest.get("controller_wall_seconds", 0.0)) + (time.time() - started)
    manifest["status"] = "RUNNING" if completed == 10 else "PARTIAL"
    atomic_json(MANIFEST, manifest)
    print(f"TRAINING_DONE {completed}/10 workers={manifest['max_concurrent_gpu_workers']}", flush=True)


if __name__ == "__main__":
    main()

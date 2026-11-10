#!/usr/bin/env python3
"""Resume incomplete E77 folds from their individually saved checkpoints."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import run


ROOT = Path(__file__).resolve().parent
WORKERS = min(8, int(run.CFG["execution"]["worker_limit"]))


def run_fold(fold: int, source_sha: str) -> tuple[int, int]:
    log_path = ROOT / "runtime" / "logs" / f"fold{fold:02d}.stdout.log"
    command = [sys.executable, str(ROOT / "run.py"), "--worker", "--fold", str(fold),
               "--source-sha", source_sha]
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with log_path.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    return fold, result.returncode


def main() -> None:
    preflight_path = ROOT / "results" / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("formal training requires the preserved passing parity gate")
    expected = preflight["source_sha256"]
    if run.source_sha() != expected:
        raise RuntimeError("E77 source SHA does not match the migrated preflight")

    pending = []
    for fold in range(1, 11):
        complete = True
        for group in run.GROUPS:
            task_path = run.task_path(group, fold)
            if not task_path.is_file():
                complete = False
                continue
            task = json.loads(task_path.read_text(encoding="utf-8"))
            checkpoint = run.best_path(group, fold)
            if (task.get("status") != "COMPLETE" or task.get("source_sha256") != expected
                    or not checkpoint.is_file()
                    or task.get("checkpoint_sha256") != run.sha_file(checkpoint)):
                raise RuntimeError(f"task/checkpoint audit failed before resume: {group} fold {fold}")
        if not complete:
            pending.append(fold)

    print(json.dumps({"pending_folds": pending, "workers": min(WORKERS, len(pending)),
                      "source_sha256": expected}, ensure_ascii=True), flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(pending)))) as pool:
        futures = {pool.submit(run_fold, fold, expected): fold for fold in pending}
        for future in concurrent.futures.as_completed(futures):
            fold, code = future.result()
            print(f"fold={fold} exit_code={code}", flush=True)
            if code != 0:
                failures.append(fold)
    if failures:
        raise RuntimeError(f"formal worker failures: {failures}; latest checkpoints remain resumable")

    for group in run.GROUPS:
        for fold in range(1, 11):
            task = json.loads(run.task_path(group, fold).read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != run.sha_file(run.best_path(group, fold)):
                raise RuntimeError(f"incomplete final task/checkpoint: {group} fold {fold}")

    result = subprocess.run([sys.executable, str(ROOT / "audit.py"), "finalize"], cwd=ROOT,
                            check=False)
    if result.returncode:
        raise RuntimeError(f"independent final audit failed with exit {result.returncode}")


if __name__ == "__main__":
    main()

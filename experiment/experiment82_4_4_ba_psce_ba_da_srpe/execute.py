"""Freeze preflight, run sequential GPU0 waves and publish the independent audit."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from common import history as h
from run import MANIFEST, RESULTS, ROOT, RUNTIME, paths

CFG = h.CFG


def cpu_pool(device):
    pci = Path("/sys/bus/pci/devices") / device["bus"][-12:].lower()
    node = (pci / "numa_node").read_text().strip()
    ranges = (Path("/sys/devices/system/node") / f"node{node}" / "cpulist").read_text().strip().split(",")
    affinity = os.sched_getaffinity(0)
    local = []
    for item in ranges:
        span = list(map(int, item.split("-")))
        local.extend(i for i in range(span[0], span[-1] + 1) if i in affinity)
    h.require(len(local) >= CFG["gpu"]["workers"], "GPU0 lacks ten distinct available CPU affinities")
    return local[:CFG["gpu"]["workers"]]


def pool(phase, cpus, folds, group=None, *, width=10):
    todo = deque(folds)
    active, retry, failures = {}, {}, {}
    max_active = 0
    gpu_peak = 0
    started = time.monotonic()
    logs = RUNTIME / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while todo or active:
            while todo and len(active) < width:
                fold = todo.popleft()
                logfile = logs / f"{phase}{'_' + group if group else ''}_f{fold:02d}.log"
                arguments = [group, str(fold)] if phase == "run" else [str(fold)]
                cmd = ["taskset", "-c", str(cpus[fold - 1]), sys.executable, "-u",
                       str(ROOT / f"{phase}.py"), *arguments]
                with logfile.open("a", encoding="utf-8") as stream:
                    active[fold] = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                    env={**os.environ, "PYTHONHASHSEED": "0"})
                max_active = max(max_active, len(active))
                if width > 1:
                    time.sleep(0.25)
            gpu_peak = max(gpu_peak, h.archive.gpu_sample()["used_mib"])
            for fold, proc in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[fold]
                if code == 0:
                    continue
                transient = code in (-9, -15, 137, 143, 75)
                if phase == "run" and transient and retry.get(fold, 0) == 0 and paths(group, fold)["latest"].is_file():
                    retry[fold] = 1
                    todo.append(fold)
                else:
                    failures[fold] = code
            if active:
                time.sleep(1)
    except BaseException:
        for proc in active.values():
            proc.terminate()
        for proc in active.values():
            proc.wait(timeout=10)
        raise
    h.require(not failures, f"{phase} {group or ''} worker failure {failures}. See {logs}")
    return {"phase": phase, "group": group, "folds": list(folds), "max_workers": max_active, "gpu_peak_mib": gpu_peak,
            "wall_seconds": time.monotonic() - started, "retries": retry}


def verify_protected(manifest):
    archived = manifest["protected_sha256"]
    readback = h.archive.hash_files([Path(p) for p in archived])
    differences = [p for p, ident in archived.items() if readback[p] != ident]
    h.require(not differences, f"Protected historical input changed: {differences}")
    return {"status": "PASS", "files": len(archived), "sha256": h.sha(readback)}


def verify_context_recoveries():
    ledger = RUNTIME / "context_recoveries.json"
    if not ledger.exists():
        return {}
    context = h.read_json(ledger)
    for group, entries in context.items():
        for fold_text, item in entries.items():
            fold = int(fold_text)
            files = paths(group, fold)
            raw = files["journal"].read_bytes()
            offset = item["journal_size"]
            before, after = raw[:offset], raw[offset:]
            h.require(hashlib.sha256(before).hexdigest() == item["journal_sha"] and after,
                      f"Committed recovery prefix was lost at {group}/{fold}")
            last = json.loads(before.splitlines()[-1])
            first = json.loads(after.splitlines()[0])
            h.require(last["epoch"] == item["epoch"] and first["epoch"] == item["epoch"] + 1
                      and last["checkpoint_state_sha"] == item["state_sha"]
                      and last["ema_after"] == first["ema_before"] == item["ema"]
                      and last["rng"]["rng_after_eval"] == first["rng"]["rng_before"] == item["rng"],
                      f"Continuation boundary differs at {group}/{fold}")
            log_path = RUNTIME / "logs" / f"run_{group}_f{fold:02d}.log"
            h.require(f"Resumed {group} fold {fold} from epoch {item['epoch']}" in log_path.read_text(),
                      f"Worker resumption record is missing at {group}/{fold}")
            item["audit"] = "exact_committed_journal_state_rng_ema_boundary"
    return context


def new_manifest(device, cpus):
    protected = h.protected_files()
    implementation = h.implementation()
    manifest = {"experiment": CFG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
                "preferred_directory": CFG["preferred_directory"], "fallback": CFG["directory_fallback"],
                "config": CFG, "implementation": implementation, "implementation_sha": h.sha(implementation),
                "protected_sha256": protected, "protected_bundle_sha": h.sha(protected),
                "runtime": h.archive.packages(), "device": device, "cpu": cpus, "history": h.hist_config(),
                "sources": CFG["sources"], "training_runs": [], "completed_folds": {g: [] for g in h.GROUPS}}
    amendment = RUNTIME / "preflight_amendments.json"
    if amendment.is_file():
        manifest["preflight_amendments"] = h.read_json(amendment)
    h.write_json(MANIFEST, manifest)
    return manifest


def results(task, group):
    return [f for f in h.FOLDS if paths(group, f)[task].is_file()]


def execute(stage):
    gpu = h.archive.verify_device()
    cpus = cpu_pool(gpu)
    hashes = h.implementation()
    existing = h.read_json(MANIFEST) if MANIFEST.is_file() else None
    started = any(ROOT.joinpath("checkpoints").glob("*.latest.pt"))
    if existing and existing["implementation"] == hashes:
        manifest = existing
    else:
        h.require(not started, "Frozen sources changed after a formal checkpoint")
        manifest = new_manifest(gpu, cpus)
    if manifest["status"] == "COMPLETE_AUDITED":
        print("COMPLETE_AUDITED and already published", flush=True)
        return

    if manifest.get("preflight", {}).get("status") != "PASS":
        print("Starting two-group single-worker preflight", flush=True)
        one = pool("preflight", cpus, [1], width=1)
        used = max(one["gpu_peak_mib"] - gpu["used_mib"],
                   *(h.read_json(paths(g, 1)["preflight"])["gpu_peak_mib"] for g in h.GROUPS))
        estimate = gpu["used_mib"] + 1.5 * CFG["gpu"]["workers"] * used
        h.require(estimate < 0.9 * gpu["total_mib"], "GPU0 ten-worker memory estimate exceeds budget")
        print(f"GPU0 10-worker estimate {estimate:.0f}/{gpu['total_mib']} MiB. Preflighting folds 2–10.", flush=True)
        rest = pool("preflight", cpus, list(h.FOLDS)[1:])
        all_paths = [paths(g, f)["preflight"] for g in h.GROUPS for f in h.FOLDS]
        rows = [h.read_json(p) for p in all_paths]
        h.require(len(rows) == 20 and all(r["status"] == "PASS" and r["implementation"] == hashes and r["rng_matches"]
                                        for r in rows), "Fold prerequisites did not all pass")
        manifest["preflight"] = {"status": "PASS", "verified": 20, "gpu_estimate_mib": estimate, "run": [one, rest],
                                 "sha256": h.archive.hash_files(all_paths)}
        h.write_json(RESULTS / "preflight.json", manifest["preflight"])
        manifest["guard_before"] = verify_protected(manifest)
        manifest["status"] = "PREFLIGHT_PASS"
        h.write_json(MANIFEST, manifest)
        print("PREFLIGHT PASS 20/20. Source, optimizer and historical RNG identities frozen.", flush=True)
    if stage == "preflight":
        return
    for group in h.GROUPS:
        done = results("task", group)
        missing = [f for f in h.FOLDS if f not in done]
        if missing:
            manifest["status"] = "TRAINING_" + group
            h.write_json(MANIFEST, manifest)
            print(f"Formal wave {group}, folds {missing}", flush=True)
            manifest["training_runs"].append(pool("run", cpus, missing, group))
        certified = [h.read_json(paths(group, fold)["task"]) for fold in h.FOLDS]
        for row in certified:
            h.require(row["status"] == "TRAINED" and row["implementation"] == hashes,
                      f"Unexpected formal task status {group} fold {row['fold']}")
            h.require(all(h.sha_file(paths(group, row["fold"])[key]) == row[key + "_sha"]
                          for key in ("best", "latest", "journal")), "Completed checkpoint or journal differs")
        manifest["completed_folds"][group] = certified
        h.write_json(MANIFEST, manifest)
        print(f"Formal wave {group} complete and integrity checked 10/10", flush=True)
    manifest["status"] = "AUDITING"
    h.write_json(MANIFEST, manifest)
    missing = [f for f in h.FOLDS if any(f not in results("audit", g) for g in h.GROUPS)]
    if missing:
        print(f"Independent audit for folds {missing}", flush=True)
        manifest["audit_run"] = pool("audit", cpus, missing)
    manifest["guard_after"] = verify_protected(manifest)
    manifest["context_recoveries"] = verify_context_recoveries()
    h.require(hashes == h.implementation(), "Frozen experiment code changed during execution")
    from publish import finish
    finish(manifest)


def record_failure(error):
    manifest = h.read_json(MANIFEST) if MANIFEST.is_file() else {}
    stage = manifest.get("status", "ENVIRONMENT")
    status = {"status": "INTERRUPTED_" + stage.removeprefix("INTERRUPTED_"), "blocked_stage": stage,
              "error": str(error), "preflight": {g: results("preflight", g) for g in h.GROUPS},
              "trained": {g: results("task", g) for g in h.GROUPS}, "audited": {g: results("audit", g) for g in h.GROUPS}}
    manifest.update(status=status["status"], last_failure=status)
    h.write_json(MANIFEST, manifest)
    h.write_json(RUNTIME / "execution_failure.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "full"), default="full")
    stage = parser.parse_args().stage
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / "scheduler.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(stage)
        except BaseException as error:
            record_failure(error)
            raise


if __name__ == "__main__":
    main()

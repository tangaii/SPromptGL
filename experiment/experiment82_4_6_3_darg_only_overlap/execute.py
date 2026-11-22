"""Verify two folds sequentially, train on physical GPU0 and replay independently."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from darg_only import (
    CFG, CONFIG, FOLDS, FULL_PATH, GROUP, MANIFEST, RESULTS, ROOT, RUNTIME, engine, ext, h,
    implementation, paths, source_manifests,
)


def verify_guard(manifest):
    expected = manifest["protected_sha256"]
    actual = h.archive.hash_files([Path(name) for name in expected])
    h.require(actual == expected, f"Historical byte changes: {[key for key in expected if actual[key] != expected[key]]}")
    return {"status": "PASS", "files": len(actual), "bundle_sha": h.sha(actual)}


def freeze_candidate():
    _, _, full = source_manifests()
    guard = dict(full["protected_sha256"])
    for batch in (full["source_verification"] if "source_verification" in full else [], CONFIG["source_verification"]):
        guard.update({item["path"]: item["server_sha256"] for item in batch})
    for batch in (full["implementation"], full["result_files"]):
        guard.update({str(FULL_PATH / key): sha for key, sha in batch.items()})
    guard[str(FULL_PATH / "manifest.json")] = CONFIG["full_path_manifest_sha256"]
    adapter = FULL_PATH / "runtime" / "finalize_audit.py"
    guard[str(adapter)] = full["reporting_recovery"]["adapter_sha256"]
    for fold in FOLDS:
        files, task = ext.paths(GROUP, fold), full["completed_folds"][str(fold)]
        guard.update({str(files[key]): task[key + "_sha"] for key in ("best", "latest", "journal")})
        guard[str(files["preflight"])] = full["preflight"]["folds"][str(fold)]["sha"]
        guard[str(files["task"])] = h.sha_file(files["task"])
    device = h.archive.verify_device()
    h.require((device["index"], device["uuid"]) == (CONFIG["physical_gpu_index"], CONFIG["physical_gpu_uuid"]),
              "Physical GPU0 does not match canonical history")
    h.require((CFG["seed"], CFG["start_epoch_completed"] + 1, CFG["max_epochs"], CFG["early_stop_wait_gt"]) ==
              (CONFIG["seed"], CONFIG["first_epoch"], CONFIG["max_epoch"], CONFIG["early_stop_wait_greater_than"]),
              "Canonical epoch or seed policy differs")
    source = implementation()
    manifest = {"experiment": CONFIG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
                "config": CONFIG, "parent_training_config": CFG, "implementation": source,
                "implementation_sha": h.sha(source), "protected_sha256": guard, "device": device,
                "packages": h.archive.packages(), "historical_configuration": h.hist_config(),
                "expected_tasks": list(FOLDS)}
    manifest["guard_before"] = verify_guard(manifest)
    return manifest


def preflight(manifest):
    from audit_darg_only import preflight_fold

    checked = {}
    for fold in FOLDS:
        file = paths(GROUP, fold)["preflight"]
        if file.exists():
            result = h.read_json(file)
        else:
            result = preflight_fold(fold, manifest["implementation"])
            h.write_json(file, result)
        h.require(result["status"] == "PASS" and result["implementation"] == manifest["implementation"]
                  and result["device"]["uuid"] == manifest["device"]["uuid"], "Preflight used a different freeze or device")
        checked[str(fold)] = {"sha": h.sha_file(file), "peak_reserved_mib": result["peak_reserved_mib"]}
        print(f"PREFLIGHT Fold {fold} PASS: 789 native, 784 SCMR legal, five exact masks", flush=True)
    device = h.archive.verify_device()
    free = device["total_mib"] - device["used_mib"]
    need = sum(x["peak_reserved_mib"] for x in checked.values())
    h.require(free > max(x["peak_reserved_mib"] for x in checked.values()) + 1024, "Insufficient GPU0 memory")
    h.require(implementation() == manifest["implementation"], "Preflight source changed")
    manifest.update(status="PREFLIGHT_PASSED", preflight={"status": "PASS", "folds": checked, "sequential": True},
                    worker_plan={"concurrent": CONFIG["max_workers"] if free >= need + 2048 else 1,
                                 "free_mib": free, "sum_preflight_peaks_mib": need},
                    frozen_before_training=True)
    manifest.pop("last_error", None)
    h.write_json(MANIFEST, manifest)
    print(f"FROZEN {manifest['implementation_sha']} | GPU0 | {manifest['worker_plan']['concurrent']} worker(s)", flush=True)


def train(manifest):
    def complete(fold):
        files = paths(GROUP, fold)
        if not files["task"].exists():
            return False
        task = h.read_json(files["task"])
        h.require(task["status"] == "TRAINED" and task["implementation"] == manifest["implementation"] and
                  all(h.sha_file(files[k]) == task[k + "_sha"] for k in ("best", "latest", "journal")),
                  "Completed fold artifact cannot be verified")
        return True

    cpus = dict(zip(FOLDS, CONFIG["worker_cpu_candidates"], strict=True))
    h.require(len(set(cpus.values())) == len(FOLDS) and set(cpus.values()).issubset(os.sched_getaffinity(0)),
              "Each worker requires its registered CPU")
    outstanding = [fold for fold in FOLDS if not complete(fold)]
    count = manifest["worker_plan"]["concurrent"]
    manifest["status"] = "TRAINING"
    h.write_json(MANIFEST, manifest)
    for index in range(0, len(outstanding), count):
        workers = {}
        try:
            for fold in outstanding[index:index + count]:
                workers[fold] = subprocess.Popen(["taskset", "-c", str(cpus[fold]), sys.executable, "-u",
                                                 str(ROOT / "execute.py"), "--worker", str(fold)], cwd=ROOT)
            codes = {fold: worker.wait() for fold, worker in workers.items()}
            h.require(all(code == 0 for code in codes.values()), f"One or more fold workers stopped: {codes}")
        finally:
            for worker in workers.values():
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait()
    h.require(all(complete(fold) for fold in FOLDS), "Formal checkpoint commit missing")
    manifest.update(status="TRAINED", completed_folds={str(fold): h.read_json(paths(GROUP, fold)["task"])
                                                       for fold in FOLDS})
    h.write_json(MANIFEST, manifest)


def audit_only():
    from audit_darg_only import audit
    from report_darg_only import publish

    manifest = h.read_json(MANIFEST)
    h.require(manifest["status"] in ("TRAINED", "AUDIT_FAILED"), "Independent replay requires both completed folds")
    try:
        results = audit(manifest)
        manifest["guard_after"] = verify_guard(manifest)
        publish(manifest, *results)
    except Exception as error:
        manifest.update(status="AUDIT_FAILED", last_error=f"{type(error).__name__}: {error}")
        h.write_json(MANIFEST, manifest)
        h.write_json(RESULTS / "audit_status.json", {"status": "AUDIT_FAILED", "error": manifest["last_error"]})
        raise


def orchestrate():
    for folder in (RUNTIME, RESULTS, ROOT / "checkpoints"):
        folder.mkdir(exist_ok=True)
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if MANIFEST.exists():
            manifest = h.read_json(MANIFEST)
            h.require(implementation() == manifest["implementation"] and CONFIG == manifest["config"],
                      "Existing artifacts require their recorded source; inspect before changing it")
            if manifest["status"] == "COMPLETE_AUDITED":
                verify_guard(manifest)
                h.require(all(h.sha_file(ROOT / k) == value for k, value in manifest["result_files"].items()),
                          "An authoritative result changed")
                print("COMPLETE_AUDITED | Verified existing results", flush=True)
                return
        else:
            h.require(not any(paths(GROUP, fold)[k].exists() for fold in FOLDS
                              for k in ("best", "latest", "journal", "task")), "Unidentified formal artifacts")
            manifest = freeze_candidate()
            h.write_json(MANIFEST, manifest)
        if manifest["status"] not in ("TRAINED", "AUDIT_FAILED"):
            try:
                preflight(manifest)
            except Exception as error:
                manifest.update(status="PREFLIGHT_FAILED", last_error=f"{type(error).__name__}: {error}")
                h.write_json(MANIFEST, manifest)
                raise
            try:
                train(manifest)
            except (Exception, KeyboardInterrupt) as error:
                manifest.update(status="INTERRUPTED", last_error=f"{type(error).__name__}: {error}")
                h.write_json(MANIFEST, manifest)
                raise
        subprocess.run([sys.executable, "-u", str(ROOT / "execute.py"), "--audit-only"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", type=int, choices=FOLDS)
    mode.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        h.require(len(os.sched_getaffinity(0)) == 1, "Worker CPU affinity differs")
        h.archive.verify_device()
        engine(args.worker).train_fold(GROUP, args.worker)
    elif args.audit_only:
        audit_only()
    else:
        orchestrate()


if __name__ == "__main__":
    main()

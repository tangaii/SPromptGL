"""Preflight, train and independently audit the two affected folds."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from extension import (CFG, CONFIG, FOLDS, GROUP, MANIFEST, PARENT, PRIOR, RESULTS, ROOT, RUNTIME,
                       engine, h, implementation, parent, paths, prior, source_manifests)


def freeze():
    original, previous = source_manifests()
    guard = {Path(path): value for path, value in previous["protected_sha256"].items()}
    guard.update({Path(x["path"]): x["server_sha256"] for x in CONFIG["source_verification"]})
    guard.update({PRIOR / name: sha for name, sha in previous["implementation"].items()})
    guard.update({PRIOR / name: sha for name, sha in previous["result_files"].items()})
    guard[PRIOR / "manifest.json"] = CONFIG["prior_manifest_sha256"]
    prev_files = prior.paths(GROUP, prior.FOLD)
    guard.update({prev_files[k]: previous["training"][k + "_sha"] for k in ("best", "latest", "journal")})
    guard[prev_files["preflight"]] = previous["preflight"]["sha"]
    guard[prev_files["task"]] = h.sha_file(prev_files["task"])
    for fold in FOLDS:
        archived = parent.paths(GROUP, fold)
        task = next(row for row in original["completed_folds"][GROUP] if row["fold"] == fold)
        guard.update({archived[k]: task[k + "_sha"] for k in ("best", "latest", "journal")})
    for name in ("main.py", "layers.py", "network.py", "model.py", "utils.py"):
        path = h.REPO / name
        guard.setdefault(path, h.sha_file(path))
    current = h.archive.hash_files(sorted(guard))
    h.require(all(current[str(path)] == expected for path, expected in guard.items()),
              "Historical SHA differs from published identity before preflight")
    source = implementation()
    device = h.archive.verify_device()
    h.require((device["index"], device["uuid"]) == (CONFIG["physical_gpu_index"], CONFIG["physical_gpu_uuid"]),
              "Physical GPU identity differs")
    return {"experiment": CONFIG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
            "config": CONFIG, "parent_training_config": CFG, "implementation": source,
            "implementation_sha": h.sha(source), "protected_sha256": current, "protected_bundle_sha": h.sha(current),
            "device": device, "packages": h.archive.packages(), "historical_configuration": h.hist_config(),
            "expected_tasks": list(FOLDS)}


def verify_guard(manifest):
    then = manifest["protected_sha256"]
    now = h.archive.hash_files([Path(path) for path in then])
    h.require(then == now, f"Protected file changed: {[path for path in then if then[path] != now[path]]}")
    return {"status": "PASS", "files": len(now), "bundle_sha": h.sha(now)}


def preflight(manifest):
    from audit_extension import preflight_fold

    certificates = {}
    for fold in FOLDS:
        file = paths(GROUP, fold)["preflight"]
        if file.is_file():
            result = h.read_json(file)
        else:
            result = preflight_fold(fold, manifest["implementation"])
            h.write_json(file, result)
        h.require(result["status"] == "PASS" and result["implementation"] == manifest["implementation"] and
                  result["device"]["uuid"] == manifest["device"]["uuid"], "Preflight identity differs")
        certificates[str(fold)] = {"sha": h.sha_file(file), "peak_reserved_mib": result["peak_reserved_mib"]}
    current = h.archive.verify_device()
    available = current["total_mib"] - current["used_mib"]
    need = sum(row["peak_reserved_mib"] for row in certificates.values())
    # Keep a context/driver margin beyond the measured all-in PyTorch reserved memory.
    workers = CONFIG["max_workers"] if available >= need + 2048 else 1
    h.require(available > max(row["peak_reserved_mib"] for row in certificates.values()) + 1024,
              "Insufficient VRAM to reserve one checked worker")
    manifest.update(preflight={"status": "PASS", "folds": certificates, "sequential_preflight": True},
                    worker_plan={"concurrent": workers, "available_mib": available,
                                 "sum_preflight_peaks_mib": need, "gpu": current},
                    guard_before=verify_guard(manifest), status="PREFLIGHT_PASSED")
    manifest.pop("last_error", None)
    h.write_json(MANIFEST, manifest)
    print(f"PREFLIGHT PASSED | Fold 2 and 3 | GPU 0 | {workers} concurrent worker(s) | "
          f"freeze {manifest['implementation_sha']}", flush=True)


def train(manifest):
    cpus = CONFIG["worker_cpu_candidates"]
    h.require(set(cpus).issubset(os.sched_getaffinity(0)) and len(cpus) == len(set(cpus)) == len(FOLDS),
              "The audited workers need distinct permitted CPU cores")
    counts = manifest["worker_plan"]["concurrent"]
    manifest["status"] = "TRAINING"
    h.write_json(MANIFEST, manifest)
    for index in range(0, len(FOLDS), counts):
        tasks = []
        try:
            for fold, cpu in zip(FOLDS[index:index + counts], cpus[index:index + counts]):
                proc = subprocess.Popen(["taskset", "-c", str(cpu), sys.executable, "-u",
                                         str(ROOT / "execute.py"), "--worker", str(fold)], cwd=ROOT)
                tasks.append((fold, proc))
            codes = {fold: proc.wait() for fold, proc in tasks}
            h.require(all(code == 0 for code in codes.values()), f"Formal worker stopped: {codes}")
        finally:
            pending = [proc for _, proc in tasks if proc.poll() is None]
            for proc in pending:
                proc.terminate()
            for proc in pending:
                proc.wait()
    manifest.update(status="TRAINED", completed_folds={str(fold): h.read_json(paths(GROUP, fold)["task"])
                                                       for fold in FOLDS})
    h.write_json(MANIFEST, manifest)


def audit_only():
    from audit_extension import audit
    from report_extension import publish

    manifest = h.read_json(MANIFEST)
    h.require(manifest["status"] in ("TRAINED", "AUDIT_FAILED"), "Audit requires both formal training tasks")
    try:
        darg, historic, prior_run, current, results = audit(manifest)
        manifest["guard_after"] = verify_guard(manifest)
        publish(manifest, darg, historic, prior_run, current, results)
    except Exception as error:
        manifest.update(status="AUDIT_FAILED", last_error=f"{type(error).__name__}: {error}")
        h.write_json(MANIFEST, manifest)
        h.write_json(RESULTS / "audit_status.json", {"status": "AUDIT_FAILED", "error": manifest["last_error"]})
        raise


def orchestrate():
    RUNTIME.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if MANIFEST.is_file():
            manifest = h.read_json(MANIFEST)
            h.require(manifest["implementation"] == implementation() and manifest["config"] == CONFIG,
                      "Existing artifacts require their frozen source on resume")
            if manifest["status"] == "COMPLETE_AUDITED":
                verify_guard(manifest)
                h.require(all(h.sha_file(ROOT / name) == digest for name, digest in manifest["result_files"].items()),
                          "An audited output differs from manifest")
                print("COMPLETE_AUDITED, existing result verified", flush=True)
                return
        else:
            h.require(not any(paths(GROUP, fold)[k].exists() for fold in FOLDS for k in ("best", "latest", "journal", "task")),
                      "Unidentified existing formal artifacts")
            manifest = freeze()
            h.write_json(MANIFEST, manifest)

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
        h.require(len(os.sched_getaffinity(0)) == 1, "Worker must be bound to one CPU")
        h.archive.verify_device()
        engine(args.worker).train_fold(GROUP, args.worker)
    elif args.audit_only:
        audit_only()
    else:
        orchestrate()


if __name__ == "__main__":
    main()

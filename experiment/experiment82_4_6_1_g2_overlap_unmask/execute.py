"""Run or resume the single formal Fold 2 task and audit it in a fresh process."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from overlap import CFG, CONFIG, FOLD, GROUP, MANIFEST, PARENT, RESULTS, ROOT, RUNTIME
from overlap import engine, h, implementation, original_manifest, parent, paths


def freeze():
    historic = original_manifest()
    status = h.read_json(PARENT / "results" / "audit_status.json")
    guard = {Path(p): digest for p, digest in historic["protected_sha256"].items()}
    guard.update({PARENT / name: digest for name, digest in historic["implementation"].items()})
    guard.update({PARENT / "results" / name: digest for name, digest in historic["result_sha256"].items()})
    guard[PARENT / "manifest.json"] = h.sha_file(PARENT / "manifest.json")
    e69 = h.runner().load_history().load_e69()
    for fold in h.FOLDS:
        old = parent.paths(GROUP, fold)
        record = next(row for row in historic["completed_folds"][GROUP] if row["fold"] == fold)
        guard.update({old["best"]: record["best_sha"], old["preflight"]: historic["preflight"]["sha256"][str(old["preflight"])],
                      old["audit"]: status["results_sha256"][str(old["audit"])], e69.warmup_path(fold): None})
        if fold == FOLD:
            guard.update({old["latest"]: record["latest_sha"], old["journal"]: record["journal_sha"]})
    extra = h.REPO / "experiment" / "experiment69_3_g2_two_rows"
    for name in ("run.py", "config.json", "manifest.json", "results/report.md"):
        path = extra / name
        if path.is_file():
            guard.setdefault(path, None)

    current = h.archive.hash_files(sorted(guard))
    h.require(all(expected is None or expected == current[str(path)] for path, expected in guard.items()),
              "Historical input differs from its published SHA")
    run_source = implementation()
    return {"experiment": CONFIG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
            "config": CONFIG, "parent_training_config": CFG, "implementation": run_source,
            "implementation_sha": h.sha(run_source), "protected_sha256": current,
            "protected_bundle_sha": h.sha(current), "device": h.archive.verify_device(),
            "affinity": sorted(os.sched_getaffinity(0)), "packages": h.archive.packages(),
            "historical_configuration": h.hist_config()}


def verify_guard(manifest):
    protected = manifest["protected_sha256"]
    now = h.archive.hash_files([Path(path) for path in protected])
    h.require(now == protected, f"Protected files changed: {[p for p in protected if protected[p] != now[p]]}")
    return {"status": "PASS", "files": len(now), "bundle_sha": h.sha(now)}


def audit_only():
    from independent import audit
    from report import publish

    manifest = h.read_json(MANIFEST)
    h.require(manifest["status"] in ("TRAINED", "AUDIT_FAILED"), "Audit requires a complete formal training task")
    status_file = RESULTS / "audit_status.json"
    try:
        historic, new, audit_result = audit(manifest)
        manifest["guard_after"] = verify_guard(manifest)
        publish(manifest, historic, new, audit_result)
    except Exception as error:
        manifest.update(status="AUDIT_FAILED", last_error=f"{type(error).__name__}: {error}")
        h.write_json(MANIFEST, manifest)
        h.write_json(status_file, {"status": "AUDIT_FAILED", "fold": FOLD, "error": manifest["last_error"]})
        raise


def orchestrate(preflight_only=False):
    from independent import preflight

    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        h.require(len(os.sched_getaffinity(0)) == CONFIG["workers"] == 1, "Pin the worker to one CPU")
        files = paths()
        if MANIFEST.is_file():
            manifest = h.read_json(MANIFEST)
            h.require(manifest["implementation"] == implementation() and manifest["config"] == CONFIG,
                      "Existing formal experiment must be resumed with its frozen implementation")
            if manifest["status"] == "COMPLETE_AUDITED":
                h.require(all(h.sha_file(ROOT / name) == digest for name, digest in manifest["result_files"].items()),
                          "An audited result changed")
                verify_guard(manifest)
                print("COMPLETE_AUDITED, existing immutable result verified", flush=True)
                return
        else:
            h.require(not any(files[k].exists() for k in ("best", "latest", "journal", "task")),
                      "Unidentified formal artifacts are present")
            manifest = freeze()
            h.write_json(MANIFEST, manifest)
        if not files["preflight"].exists():
            try:
                outcome = preflight(engine(), manifest["implementation"])
                h.require(outcome["status"] == "PASS", "Preflight did not pass")
            except Exception as error:
                manifest.update(status="PREFLIGHT_FAILED", last_error=f"{type(error).__name__}: {error}")
                h.write_json(MANIFEST, manifest)
                raise
            h.write_json(files["preflight"], outcome)
        else:
            outcome = h.read_json(files["preflight"])
        h.require(outcome["status"] == "PASS" and outcome["implementation"] == manifest["implementation"] and
                  outcome["device"]["uuid"] == manifest["device"]["uuid"] == CONFIG["physical_gpu_uuid"],
                  "Frozen preflight or GPU identity disagrees")
        manifest.update(preflight={"status": "PASS", "sha": h.sha_file(files["preflight"])},
                        guard_before=verify_guard(manifest), status="PREFLIGHT_PASSED")
        manifest.pop("last_error", None)
        h.write_json(MANIFEST, manifest)
        print(f"PREFLIGHT PASSED | GPU {manifest['device']['index']} | freeze {manifest['implementation_sha']}", flush=True)
        if preflight_only:
            return
        try:
            manifest["status"] = "TRAINING"
            h.write_json(MANIFEST, manifest)
            engine().train_fold(GROUP, FOLD)
        except (Exception, KeyboardInterrupt) as error:
            manifest.update(status="INTERRUPTED", last_error=f"{type(error).__name__}: {error}")
            h.write_json(MANIFEST, manifest)
            raise
        manifest.update(status="TRAINED", training=h.read_json(files["task"]))
        h.write_json(MANIFEST, manifest)
        subprocess.run([sys.executable, "-u", str(ROOT / "execute.py"), "--audit-only"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--preflight-only", action="store_true")
    stage.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit_only()
    else:
        orchestrate(args.preflight_only)


if __name__ == "__main__":
    main()

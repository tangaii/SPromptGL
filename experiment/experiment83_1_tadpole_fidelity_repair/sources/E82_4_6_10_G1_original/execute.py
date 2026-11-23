"""Gate two formal tasks behind historical replay and frozen preflight."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback

from core import (
    CONFIG, FOLD, GROUPS, MANIFEST, MODEL_VARIANT, ROOT, RUNTIME, SOURCES,
    engine, h, implementation, parent, paths,
)
from sources import E9, replay_archives, utc_now, verify_guard, verify_sources

from audit_tasks import checkpoint_audit, preflight

COMPLETE = {"COMPLETE_AUDITED", "AUDITED_WITH_RELATION_COVERAGE_LIMITATION"}
HISTORY_REPLAY = RUNTIME / "archived_best_replay.json"


def check_frozen(manifest):
    h.require(manifest["config"] == CONFIG and manifest["implementation"] == implementation(),
              "Experiment source or configuration changed after freeze")
    h.require(h.sha(manifest["implementation"]) == manifest["implementation_sha"],
              "Frozen bundle digest differs")


def initialize():
    if MANIFEST.exists():
        manifest = h.read_json(MANIFEST)
        check_frozen(manifest)
        return manifest
    h.require(str(ROOT) == CONFIG["authoritative_directory"] and set(GROUPS) == {"G1", "G2"}
              and CONFIG["gpu"]["formal_tasks"] == len(GROUPS) == 2,
              "Unexpected experiment directory or official task count")
    training = CONFIG["training"]
    history = h.hist_config()
    original = history["training"]
    h.require(all(training[k] == original[k] for k in ("max_epochs", "early_stop_wait_gt", "checkpoint_selection",
                                                       "pair_count", "balanced_pairs", "no_per_epoch_reseed", "loss"))
              and training["optimizer_configs"] == original["optimizers"]
              and parent.CFG["residual"]["fraction"] == training["cam_residual_cap"]
              and parent.CFG["full_classification"]["shared_cap"] == training["shared_differential_gradient_cap"]
              and parent.START == CONFIG["start_epoch_completed"] == 10
              and parent.CFG["seed"] == history["data"]["seed"] == CONFIG["seed"] == 0
              and history["data"]["validation_equals_test"], "Historical training configuration differs")
    device = h.archive.verify_device()
    h.require(device["index"] == CONFIG["gpu"]["physical_index"] and device["uuid"] == CONFIG["gpu"]["uuid"],
              "Physical GPU0 mapping differs")
    cores = {group_info["cpu"] for group_info in GROUPS.values()}
    h.require(cores <= os.sched_getaffinity(0) and len(cores) == len(GROUPS), "Worker affinity plan is unavailable")
    manifests, guards, checkpoints = verify_sources()
    guard = verify_guard(guards)
    hashes = implementation()
    manifest = {
        "experiment": CONFIG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
        "config": CONFIG, "implementation": hashes, "implementation_sha": h.sha(hashes),
        "frozen_at_utc": utc_now(), "device": device, "environment": h.archive.packages(),
        "historical_training": history, "scmr_config_sha256": h.sha_file(SOURCES["base"] / "config.json"),
        "scmr_optimizer": parent.CFG["new_optimizer"], "scmr_init_seed": parent.CFG["seed_stream"][MODEL_VARIANT] * 100 + FOLD,
        "sources": {name: {"path": str(SOURCES[name] / "manifest.json"),
                           "sha256": CONFIG["source_manifests"][name][1], "status": doc["status"]}
                    for name, doc in manifests.items()},
        "protected_sha256": guards, "guard_before": guard,
        "historical_single_checkpoints": checkpoints, "actual_new_folds": [FOLD],
        "reuse_reference": E9,
        "audit_adaptation": {
            "source": str(SOURCES["e469"] / "audit_tasks.py"),
            "source_sha256": guards[str(SOURCES["e469"] / "audit_tasks.py")],
            "dual_patient_frozen_sha256": hashes["audit_tasks.py"],
            "change": "Independent two-row membership, mask, gradient, intervention and Latest readback checks",
            "training_adapter_frozen_sha256": hashes["core.py"],
        },
        "reused_folds": [fold for fold in range(1, 11) if fold != FOLD],
        "groups": {group: {"status": "REGISTERED", **task} for group, task in GROUPS.items()},
    }
    h.write_json(MANIFEST, manifest)
    return manifest


def run_preflight(manifest):
    if manifest.get("preflight", {}).get("status") == "PASS":
        require_preflight(manifest)
        return
    h.require(not any(paths(group)["latest"].exists() for group in GROUPS),
              "Formal checkpoint exists without a completed preflight")
    if not manifest.get("historical_replay"):
        manifests = {key: h.read_json(path / "manifest.json") for key, path in SOURCES.items()}
        evaluations = replay_archives(manifests, manifest["historical_single_checkpoints"])
        h.require(evaluations["status"] == "PASS" and evaluations["reused_rows"] == 784,
                  "Historical replay must certify the nine reused folds")
        h.write_json(HISTORY_REPLAY, evaluations)
        manifest["historical_replay"] = {
            "sha256": h.sha_file(HISTORY_REPLAY), "path": str(HISTORY_REPLAY), "reused_rows": 784,
            "completed_at_utc": evaluations["completed_at_utc"], "direct_reference": evaluations["direct_reference"],
            "fold8_original": evaluations["folds"][str(FOLD)]["metrics"],
            "direct_fold8": evaluations["single_patient_reference"]["G2"]["metrics"],
        }
        h.write_json(MANIFEST, manifest)
    hashes = manifest["implementation"]
    gpu = h.archive.gpu_sample()
    certificates = {}
    for group in GROUPS:
        certificate = preflight(group, hashes)
        path = paths(group)["preflight"]
        h.write_json(path, certificate)
        certificates[group] = certificate
        reg = certificate["membership"]
        occurrences = {i: p["endpoint_occurrences"] for i, p in reg["target_pairs"].items()}
        print(f"{group} preflight PASS: patients {reg['patient_ids_1based']}, "
              f"endpoints {reg['covered_rows']}/{reg['native_rows']}, target occurrences {occurrences}, "
              f"peak reserve {certificate['peak_reserved_mib']:.0f} MiB", flush=True)

    sample = next(iter(certificates.values()))
    shared = sample["shared_initial"]
    h.require(all(cert["shared_initial"] == shared and cert["first_rng"] == sample["first_rng"]
                  and cert["dropout"] == sample["dropout"] for cert in certificates.values()),
              "The two starts, RNG streams or historical Dropout inventories differ")
    h.require(len({cert["membership"]["extended_train_sha"] for cert in certificates.values()}) == len(GROUPS)
              and len({cert["membership"]["pair_sha"] for cert in certificates.values()}) == len(GROUPS),
              "Independent task membership or relation pairs are duplicated")
    # Estimate per worker using the larger of the device allocation delta and allocator reservation.
    per_worker = max(max(cert["process_gpu_mib"], cert["peak_reserved_mib"]) for cert in certificates.values())
    gpu_free = gpu["total_mib"] - gpu["used_mib"]
    workers = min(len(GROUPS), CONFIG["gpu"]["worker_limit"], int(gpu_free // (2 * per_worker)))
    h.require(workers >= 1, "GPU0 lacks the memory estimated for one worker")
    manifest["preflight"] = {
        "status": "PASS", "completed_at_utc": utc_now(), "shared_initial": shared,
        "dropout": sample["dropout"], "historical_replay_completed_before_training": True,
        "groups": {group: {"path": str(paths(group)["preflight"]), "sha256": h.sha_file(paths(group)["preflight"]),
                           "completed_at_utc": cert["completed_at_utc"], "status": "PASS"}
                   for group, cert in certificates.items()},
        "gpu": {"idle_sample": gpu, "conservative_per_worker_mib": 2 * per_worker,
                "formal_concurrency": workers, "estimated_formal_mib": 2 * per_worker * workers},
    }
    manifest["status"] = "PREFLIGHT_PASSED"
    for group in GROUPS:
        manifest["groups"][group]["status"] = "PREFLIGHT_PASSED"
    h.write_json(MANIFEST, manifest)
    h.sync_dir(ROOT)
    print(f"ALL PREFLIGHTS PASSED. GPU0 capacity for {workers} concurrent formal workers.", flush=True)


def require_preflight(manifest):
    check_frozen(manifest)
    preflight = manifest["preflight"]
    history = manifest["historical_replay"]
    replay = h.read_json(HISTORY_REPLAY)
    h.require(preflight["status"] == "PASS" and set(preflight["groups"]) == set(GROUPS)
              and h.sha_file(HISTORY_REPLAY) == history["sha256"]
              and replay["status"] == "PASS" and replay["reused_rows"] == 784
              and history["completed_at_utc"] == replay["completed_at_utc"] < preflight["completed_at_utc"]
              and all(h.sha_file(paths(group)["preflight"]) == preflight["groups"][group]["sha256"]
                      and preflight["groups"][group]["status"] == "PASS" for group in GROUPS),
              "Formal start lacks the historical replay or two frozen preflight certificates")


def worker(group):
    manifest = h.read_json(MANIFEST)
    require_preflight(manifest)
    files = paths(group)
    cpu = GROUPS[group]["cpu"]
    h.require(os.sched_getaffinity(0) == {cpu}, f"{group} affinity must be CPU {cpu}")
    h.archive.verify_device()
    with files["lock"].open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        h.require(manifest["first_formal_launch_utc"] > manifest["preflight"]["completed_at_utc"],
                  "Formal worker launch ordering is invalid")
        if files["journal"].exists() and not files["latest"].exists():
            h.require(files["journal"].stat().st_size == 0, "A nonempty interrupted run requires verified Latest")
        print(f"{utc_now()} {group} Fold {FOLD} formal worker entering historical training", flush=True)
        engine(group).train_fold(MODEL_VARIANT, FOLD)


def verify_completed(group):
    files = paths(group)
    if not files["task"].exists():
        return None
    done = h.read_json(files["task"])
    h.require(done["status"] == "TRAINED" and done["implementation"] == implementation()
              and done["group"] == done["task_group"] == group
              and all(h.sha_file(files[k]) == done[k + "_sha"] for k in ("best", "latest", "journal")),
              f"{group} completed training artifact differs")
    return done


def start_worker(group, attempts):
    cpu = GROUPS[group]["cpu"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": CONFIG["gpu"]["uuid"]}
    command = ["taskset", "-c", str(cpu), sys.executable, "-u", str(ROOT / "execute.py"), "--worker", group]
    with paths(group)["log"].open("a", encoding="utf-8") as out:
        print(f"{utc_now()} controller launch {group} attempt {attempts}", file=out, flush=True)
        return subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)


def train(manifest):
    require_preflight(manifest)
    manifest["guard_prelaunch"] = verify_guard(manifest["protected_sha256"])
    plan = manifest["preflight"]["gpu"]["formal_concurrency"]
    queue = [group for group in GROUPS if verify_completed(group) is None]
    active = {}
    attempts = manifest.setdefault("training_attempts", {group: [] for group in GROUPS})
    manifest.setdefault("gpu_training_peak_mib", 0)
    if queue:
        manifest.setdefault("first_formal_launch_utc", utc_now())
        manifest["status"] = "TRAINING"
        h.write_json(MANIFEST, manifest)
        h.sync_dir(ROOT)
    with (RUNTIME / "gpu_training.csv").open("a", encoding="utf-8") as telemetry:
        if telemetry.tell() == 0:
            telemetry.write("utc,used_mib,utilization_pct,formal_workers\n")
        while queue or active:
            while queue and len(active) < plan:
                group = queue.pop(0)
                proc = start_worker(group, len(attempts[group]) + 1)
                receipt = {"started_at_utc": utc_now(), "pid": proc.pid, "from_latest": paths(group)["latest"].exists()}
                attempts[group].append(receipt)
                active[group] = proc
                manifest["groups"][group]["status"] = "TRAINING"
                print(f"{group} worker PID {proc.pid} on GPU0", flush=True)
            sample = h.archive.gpu_sample()
            manifest["gpu_training_peak_mib"] = max(manifest["gpu_training_peak_mib"], sample["used_mib"])
            manifest["peak_formal_workers"] = max(manifest.get("peak_formal_workers", 0), len(active))
            telemetry.write(f"{utc_now()},{sample['used_mib']},{sample['utilization']},{len(active)}\n")
            telemetry.flush()
            for group, proc in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                active.pop(group)
                attempts[group][-1].update(exit_code=code, ended_at_utc=utc_now())
                done = verify_completed(group) if code == 0 else None
                if done is not None:
                    manifest["groups"][group].update(status="TRAINED", training=done)
                    print(f"{group} trained: Best {done['best_correct']}/87 at {done['best_epoch']}, Latest {done['epoch']}", flush=True)
                elif len(attempts[group]) < 3 and paths(group)["latest"].is_file():
                    print(f"{group} worker exited {code}; retrying from its Latest", flush=True)
                    queue.append(group)
                else:
                    manifest["groups"][group]["status"] = "TRAINING_FAILED"
                    print(f"{group} worker exited {code}; see {paths(group)['log']}", flush=True)
            h.write_json(MANIFEST, manifest)
            if active:
                time.sleep(2)
    for group in GROUPS:
        done = verify_completed(group)
        if done is not None:
            manifest["groups"][group].update(status="TRAINED", training=done)
    manifest["status"] = "TRAINED" if all(record["status"] == "TRAINED" for record in manifest["groups"].values()) else "TRAINING_INCOMPLETE"
    h.write_json(MANIFEST, manifest)


def audit_all(manifest):
    manifest["guard_preaudit"] = verify_guard(manifest["protected_sha256"])
    for group in GROUPS:
        if verify_completed(group) is None:
            continue
        files = paths(group)
        try:
            recorded = manifest["groups"][group].get("audit")
            if recorded and recorded["status"] in COMPLETE and h.sha_file(files["audit"]) == recorded["sha256"]:
                manifest["groups"][group]["status"] = recorded["status"]
                continue
            with open(files["log"], "ab") as log:
                proc = subprocess.run(["taskset", "-c", str(GROUPS[group]["cpu"]), sys.executable, "-u",
                                       str(ROOT / "execute.py"), "--audit", group], cwd=ROOT, stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
            h.require(proc.returncode == 0, f"{group} independent audit failed; see {files['log']}")
            certificate = h.read_json(files["audit"])
            h.require(certificate["status"] in COMPLETE, f"{group} audit did not issue a completed certificate")
            manifest["groups"][group].update(status=certificate["status"],
                audit={"status": certificate["status"], "path": str(files["audit"]), "sha256": h.sha_file(files["audit"])})
            print(f"{group} audit {certificate['status']}", flush=True)
        except Exception as exc:
            manifest["groups"][group].update(status="AUDIT_FAILED", audit_error=str(exc))
            print(f"{group} audit failed: {exc}", flush=True)
        h.write_json(MANIFEST, manifest)
    manifest["guard_after"] = verify_guard(manifest["protected_sha256"])
    manifest["status"] = "AUDITED" if all(g["status"] in COMPLETE for g in manifest["groups"].values()) else "AUDIT_INCOMPLETE"
    h.write_json(MANIFEST, manifest)


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--train-only", action="store_true")
    actions.add_argument("--audit-only", action="store_true")
    actions.add_argument("--worker", choices=GROUPS)
    actions.add_argument("--audit", choices=GROUPS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.audit:
        manifest = h.read_json(MANIFEST)
        require_preflight(manifest)
        certificate = checkpoint_audit(args.audit, manifest)
        h.write_json(paths(args.audit)["audit"], certificate)
        print(f"{utc_now()} {args.audit} {certificate['status']}", flush=True)
        return

    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = initialize()
        try:
            if manifest.get("historical_replay") is None or manifest.get("preflight", {}).get("status") != "PASS":
                h.require(not args.audit_only, "Audit requires the formal gate to have passed")
                run_preflight(manifest)
            else:
                require_preflight(manifest)
            if args.preflight_only:
                return
            if not args.audit_only:
                train(manifest)
            if not args.train_only:
                audit_all(manifest)
        except Exception:
            manifest["last_error"] = traceback.format_exc()
            h.write_json(MANIFEST, manifest)
            raise


if __name__ == "__main__":
    main()

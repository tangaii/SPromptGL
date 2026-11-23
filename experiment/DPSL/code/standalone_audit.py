"""Strict inference, restoration, training, provenance, and negative audits."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dependency_guard import ExternalProjectAccess
from dpsl_runtime import (FOLDS, ROOT, checkpoint_source, history_records, load_checkpoint, official_manifest,
                          read_json, sha256_file, source_records, validate_checkpoint, validate_evaluation,
                          validate_membership, verify_sha, write_json)
from real_model_runtime import check_order, check_residual, evaluate, recovery, require
from training_runtime import warmup_check

RUNTIME = ROOT / "runtime"
RESULTS = ROOT / "results" / "recheck"


@dataclass(frozen=True)
class NegativeTest:
    name: str
    exception: type[Exception]
    check: object


def verify_inventory():
    inventory = RUNTIME / "operational_sha256.json"
    if not inventory.exists():
        return {"status": "PARTIALLY_STANDALONE", "reason": "Runtime SHA inventory has not been frozen."}
    files = read_json(inventory)["files"]
    require(isinstance(files, dict) and all(name in files for name in (
        "cli.py", "code/real_model_runtime.py", "code/training_runtime.py", "config.json",
        "checkpoints/fold08.best.pt", "inputs/warmup/fold08.pt")), "Operational inventory is incomplete")
    for name, digest in files.items():
        path = ROOT / name
        require(path.is_relative_to(ROOT) and path.resolve() == path, f"Inventory file is not local: {name}")
        verify_sha(path, digest)
    return {"status": "PASS", "count": len(files), "inventory_sha256": sha256_file(inventory)}


def inventory_and_provenance():
    records = source_records()
    copied = [*records["manifest_records"], *records["warmup_records"], *records["fixed_records"], *history_records()]
    for item in copied:
        verify_sha(ROOT / item["path"], item["sha256"])
    frozen = read_json(ROOT / "manifest.json")
    official = official_manifest()
    for row in frozen["formal_checkpoint_mapping"]:
        fold = row["fold"]
        for kind in ("best", "latest"):
            path, digest, epoch = checkpoint_source(fold, kind)
            require(row[f"{kind}_source_path"] == path and row[f"{kind}_source_sha256"] == digest,
                    f"Fold {fold} manifest source differs")
    for basename in ("patient_level.csv", "fold_metrics.csv"):
        verify_sha(ROOT / "code/provenance/source_results" / basename, official["result_files"][f"results/{basename}"])
    return {"status": "PASS", "groups": len(copied),
            "official_manifest_sha256": sha256_file(ROOT / "code/provenance/manifests/E82_4_6_10-G1.json")}


def compare_original_forward(forward):
    path = RUNTIME / "source_forward_sha256.json"
    if not path.is_file():
        return {"status": "PARTIALLY_STANDALONE", "reason": "Original executable forward tensor receipt is missing."}
    source = read_json(path)
    require(len(source["folds"]) == 10, "Original forward receipt is incomplete")
    for local, old in zip(forward["folds"], source["folds"], strict=True):
        require(local["fold"] == old["fold"] and local["best_epoch"] == old["best_epoch"]
                and local["full_tensor_sha"] == old["full_tensor_sha"] and local["result_sha"] == old["result_sha"]
                and local["scope"]["membership_sha"] == old["membership_sha"]
                and local["configuration_sha"] == old["configuration_sha"]
                and local["best_ema_sha"] == old["ema_sha"]
                and checkpoint_source(local["fold"], "best")[1] == old["original_best_sha"],
                f"Fold {local['fold']} differs from the original executable forward receipt")
    return {"status": "PASS", "folds": 10, "all_graph_tensor_shape": [871, 144],
            "maximum_absolute_error": 0.0, "receipt_sha256": sha256_file(path)}


def negative_checks(guard):
    fold = 8
    saved = load_checkpoint(fold, "latest")
    epoch = saved["epoch"]
    rng = {k: v for k, v in saved["rng"].items() if k != "cuda"}
    bad_eval = {k: v for k, v in saved["evaluation"].items() if k != "logits"}
    bad_eval["unrelated"] = True
    bad_scope = {**saved["membership"], "legal": saved["membership"]["train"]}
    bad_optimizer = {**saved["state"]["optimizer"], "state": {}}
    bad_state = {**saved["state"], "optimizer": bad_optimizer}
    moments = dict(saved["state"]["optimizer"]["state"])
    key = next(iter(moments))
    moments[key] = {**moments[key], "exp_avg": moments[key]["exp_avg"].reshape(-1)[:0]}
    corrupt_state = {**saved["state"], "optimizer": {**saved["state"]["optimizer"], "state": moments}}
    test = saved["membership"]["test"]
    old_order = [{"patient_index_0based": int(i)} for i in test]
    wrong_order = test.copy()
    wrong_order[:2] = wrong_order[:2][::-1]
    delta = np.zeros((2, 144), dtype=np.float32)
    delta[0, 0] = 0.25
    cam = np.zeros_like(delta)
    outside = Path(checkpoint_source(fold, "latest")[0])
    with TemporaryDirectory(prefix="negative-audit-", dir=RUNTIME) as temp:
        corrupt_file = Path(temp) / "latest.pt"
        corrupt_file.write_bytes(b"checkpoint audit tamper")
        tests = [
            NegativeTest("missing_checkpoint_field", ValueError,
                         lambda: validate_checkpoint({k: v for k, v in saved.items() if k != "state_sha"}, fold, "latest", epoch)),
            NegativeTest("evaluation_missing_required_key_with_extra_key", ValueError,
                         lambda: validate_evaluation(bad_eval, len(test))),
            NegativeTest("empty_optimizer_state", ValueError,
                         lambda: validate_checkpoint({**saved, "state": bad_state}, fold, "latest", epoch)),
            NegativeTest("invalid_optimizer_moment_shape", ValueError,
                         lambda: validate_checkpoint({**saved, "state": corrupt_state}, fold, "latest", epoch)),
            NegativeTest("wrong_checkpoint_sha", ValueError, lambda: verify_sha(corrupt_file, checkpoint_source(fold, "latest")[1])),
            NegativeTest("test_order_changed", ValueError, lambda: check_order(wrong_order, old_order)),
            NegativeTest("overlap_in_added_supervision", ValueError, lambda: validate_membership(bad_scope, fold)),
            NegativeTest("overlap_residual_nonzero", ValueError, lambda: check_residual(delta, cam, cam, [0])),
            NegativeTest("external_experiment_access", ExternalProjectAccess, outside.read_bytes),
            NegativeTest("missing_device_rng", ValueError,
                         lambda: validate_checkpoint({**saved, "rng": rng}, fold, "latest", epoch)),
        ]
        before = set(guard.denied)
        checks = {}
        for test_case in tests:
            try:
                test_case.check()
            except test_case.exception as error:
                checks[test_case.name] = {"status": "PASS", "damaged_input_status": "FAIL", "reason": str(error)}
            else:
                checks[test_case.name] = {"status": "FAIL", "reason": "Mutation reached the success path"}
        expected = str(outside)
        if expected in guard.denied - before:
            guard.denied.remove(expected)
    return {"status": "PASS" if all(x["status"] == "PASS" for x in checks.values()) else "FAIL", "cases": checks}


def migration_receipts():
    paths = [RUNTIME / name for name in ("historical_source_integrity.json", "external_trace.json")]
    if not all(path.is_file() for path in paths):
        return {"status": "PARTIALLY_STANDALONE", "reason": "Historical byte comparison or OS file trace is pending."}
    source, trace = [read_json(path) for path in paths]
    require(source["status"] == trace["status"] == "PASS", "Historical source or OS dependency audit failed")
    require(trace["operational_inventory_sha256"] == sha256_file(RUNTIME / "operational_sha256.json"),
            "OS trace and current executable package have different identities")
    require(not trace["external_project_accesses"] and not source["changed"], "External paths or source modifications")
    return {"status": "PASS", "historical_sources_unchanged": source["checked_count"],
            "outside_working_directory": trace["working_directory"], "os_file_syscalls": trace["file_syscalls"],
            "os_trace_sha256": trace["trace_sha256"]}


def audit(guard, *, record):
    """Each failed mandatory check takes precedence over an incomplete receipt."""
    stages = {}
    def run_stage(name, function):
        try:
            value = function()
        except Exception as error:
            value = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
            if isinstance(error, FileNotFoundError):
                value["classification"] = "BLOCKED_MISSING_SOURCE"
        stages[name] = value
        return value

    integrity = run_stage("operational_files", verify_inventory)
    if integrity["status"] != "FAIL":
        provenance = run_stage("historical_manifests_and_artifacts", inventory_and_provenance)
        if provenance["status"] == "PASS":
            forward = run_stage("best_real_forward", lambda: evaluate(out=RESULTS if record else None))
            if forward["status"] == "PASS":
                run_stage("original_model_tensor_comparison", lambda: compare_original_forward(forward))
            run_stage("latest_next_train_eval", lambda: recovery(out=RESULTS if record else None))
            warmup = run_stage("warmup_and_training_resume", warmup_check)
            if record:
                write_json(RESULTS / "warmup_step_check.json", warmup)
            run_stage("negative_cases", lambda: negative_checks(guard))
            run_stage("historical_source_and_os_receipts", migration_receipts)
    run_stage("runtime_dependency_closure", guard.report)
    states = {v["status"] for v in stages.values()}
    result = {"status": "FAIL" if "FAIL" in states else "PARTIALLY_STANDALONE" if "PARTIALLY_STANDALONE" in states
              else "COMPLETE_AUDITED_STANDALONE",
              "stages": stages, "full_ten_fold_retraining_executed": False}
    if record:
        write_json(RESULTS / "audit_status.json", result)
    return result
